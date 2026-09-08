"""
Wrapper module to adapt mini-swe-agent to work with slime's RL training framework.

Mirrors the class structure of mini-swe-agent's DefaultAgent (see ideas/swe_agent_default.py)
but adapted for slime's async inference engine.
"""

import asyncio
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from slime.utils.http_utils import post

logger = logging.getLogger(__name__)

# Import mini-swe-agent environment components
try:
    from minisweagent import Environment
    from minisweagent.environments import get_environment
    from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

    try:
        from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
        HAS_DOCKER = True
    except ImportError:
        HAS_DOCKER = False
        DockerEnvironment = None
        DockerEnvironmentConfig = None
except ImportError as e:
    raise ImportError(
        "mini-swe-agent not found. Please install it first:\n"
        "  pip install mini-swe-agent\n"
        f"Original error: {e}"
    )

try:
    from .azure_modal_docker import AzureDockerEnvironment, AzureDockerEnvironmentConfig
    HAS_AZURE_DOCKER = True
except ImportError:
    HAS_AZURE_DOCKER = False
    AzureDockerEnvironment = None
    AzureDockerEnvironmentConfig = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0


# ---------------------------------------------------------------------------
# Exceptions (same hierarchy as DefaultAgent)
# ---------------------------------------------------------------------------

class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


# ---------------------------------------------------------------------------
# SWEAgent – mirrors DefaultAgent but works with slime's inference
# ---------------------------------------------------------------------------

class SWEAgent:
    """Agent for SWE-bench tasks, following the DefaultAgent interface."""

    def __init__(self, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.env = env
        self.extra_template_vars: dict[str, Any] = {}
        self.n_calls = 0
        self.cost = 0.0

    # -- template helpers ---------------------------------------------------

    def get_template_vars(self) -> dict[str, Any]:
        return {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = (
            self.config.model_dump()
            | self.env.get_template_vars()
            | self.get_template_vars()
        )
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    # -- message management -------------------------------------------------

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            **kwargs,
        })

    # -- setup --------------------------------------------------------------

    def setup(self, task: str, **kwargs):
        """Initialize messages with system and instance prompts."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))

    # -- action parsing & execution (mirrors DefaultAgent) ------------------

    def check_limits(self):
        if 0 < self.config.step_limit <= self.n_calls:
            raise LimitsExceeded()

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the response. Returns dict with 'action' key."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(
            self.render_template(self.config.format_error_template, actions=actions)
        )

    async def execute_action(self, action: dict) -> dict:
        """Execute action in the environment. May raise ExecutionTimeoutError or Submitted."""
        start_time = time.time()
        try:
            if HAS_AZURE_DOCKER and isinstance(self.env, AzureDockerEnvironment):
                output = await self.env.execute(action["action"])
            else:
                # Use asyncio.to_thread to avoid blocking the event loop
                # (env.execute is a synchronous blocking call)
                output = await asyncio.to_thread(self.env.execute, action["action"])
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            elapsed = time.time() - start_time
            logger.info("Action execution timed out after %.2fs", elapsed)
            output_text = (
                e.output.decode("utf-8", errors="replace")
                if getattr(e, "output", None)
                else ""
            )
            raise ExecutionTimeoutError(
                self.render_template(
                    self.config.timeout_template, action=action, output=output_text
                )
            )
        elapsed = time.time() - start_time
        logger.info("Action execution completed in %.2fs", elapsed)
        self.has_finished(output)
        return output | {"action": action["action"]}

    async def get_observation(self, response: dict) -> tuple[dict, str]:
        """Parse action, execute, render observation. Returns (output, observation_text).

        Unlike DefaultAgent.get_observation, this does NOT add a user message
        automatically – the caller (generate) handles that so it can track tokens.
        """
        output = await self.execute_action(self.parse_action(response))
        observation = self.render_template(
            self.config.action_observation_template, output=output
        )
        return output, observation

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in [
            "MINI_SWE_AGENT_FINAL_OUTPUT",
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ]:
            raise Submitted("".join(lines[1:]))

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise agent state for metadata."""
        return {
            "messages": self.messages,
            "n_steps": self.n_calls,
        }


# ---------------------------------------------------------------------------
# Config / environment factory helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    """Load mini-swe-agent configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_docker_image_name(instance: dict) -> str:
    """Get the Docker image name for a SWE-bench instance.
    
    Follows the SWE-bench Docker image naming convention:
    docker.io/swebench/sweb.eval.x86_64.<instance_id>:latest
    
    Args:
        instance: Dict containing instance information with 'instance_id' key,
                  and optionally 'image_url' for custom images.
                  
    Returns:
        Docker image name string.
    """
    image_name = instance.get("image_url", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance.get("instance_id", "unknown")
        id_docker_compatible = iid.replace("__", "_1776_")
        #image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        image_name = f"mirror.gcr.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    else:
        if image_name.startswith("aweaiteam/") or image_name.startswith("swerebench/"):
            image_name = "mirror.gcr.io/" + image_name
        elif image_name.startswith("docker.io/"):
            image_name = "mirror.gcr.io/" + image_name[len("docker.io/"):]
        else:
            image_name = image_name
    return image_name


async def create_environment(
    config: dict,
    instance_id: str | None = None,
    instance: dict | None = None,
    startup_command: str | None = None,
) -> Environment:
    """Create a mini-swe-agent environment (local, docker, singularity, swerex_modal, or azure_docker).
    
    This function properly handles environment creation for SWE-bench tasks,
    including Docker image setup and startup command execution.
    
    Args:
        config: Environment configuration dict with 'environment_class' key.
        instance_id: Optional instance ID (currently unused, kept for API compatibility).
        instance: Optional SWE-bench instance dict for Docker image resolution.
        startup_command: Optional command to run after environment starts.
        
    Returns:
        Configured Environment instance.
    """
    overall_start = time.time()
    env_config = config.copy()
    env_type = env_config.get("environment_class", "local")
    
    # Handle Docker image setup for SWE-bench instances
    if env_type in ["docker", "swerex_modal", "singularity", "azure_docker"]:
        # Get Docker image name from instance if available
        if instance is not None and "image" not in env_config:
            image_name = get_docker_image_name(instance)
            if env_type == "singularity":
                env_config["image"] = "docker://" + image_name
            else:
                env_config["image"] = image_name
    
    # Set cwd from instance workdir if available, default to /testbed
    if instance is not None and "cwd" not in env_config:
        env_config["cwd"] = instance.get("workdir", "/testbed")

    env = None
    try:
        # Azure Docker is not part of minisweagent, so handle it directly
        if env_type == "azure_docker":
            if not HAS_AZURE_DOCKER:
                raise ImportError(
                    "Azure Docker environment requested but not available. "
                    "Ensure azure_modal_docker.py and the azure-modal client are installed."
                )
            clean_config = {k: v for k, v in env_config.items() if k != "environment_class"}
            env = AzureDockerEnvironment(config_class=AzureDockerEnvironmentConfig, **clean_config)
            logger.info("Created AzureDockerEnvironment, now env.start_container() for %s...", instance_id)
            await env.start_container()
            logger.info("AzureDockerEnvironment container started for %s.", instance_id)
        else:
            # Use minisweagent's get_environment for proper environment creation
            # This is the recommended approach as shown in swebench.py
            try:
                env = get_environment(env_config)
            except Exception:
                # Fallback to direct instantiation if get_environment fails
                # Remove environment_class as it's not a valid config parameter
                clean_config = {k: v for k, v in env_config.items() if k != "environment_class"}

                if env_type == "local":
                    env = LocalEnvironment(config_class=LocalEnvironmentConfig, **clean_config)
                elif env_type == "docker":
                    if not HAS_DOCKER:
                        raise ImportError("Docker environment requested but not available")
                    env = DockerEnvironment(config_class=DockerEnvironmentConfig, **clean_config)
                else:
                    raise ValueError(f"Unsupported environment type: {env_type}")
        
        env_elapsed = time.time() - overall_start
        logger.info("Environment (%s) created in %.2fs", env_type, env_elapsed)

        pre_commands = instance.get("pre_commands", "").replace("\\n", "\n").strip()
        startup_command = startup_command + " && " + pre_commands if startup_command else pre_commands

        # Execute startup command if provided
        if startup_command:
            startup_start = time.time()
            # Render template variables if instance provided
            if instance is not None:
                startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
            if env_type == "azure_docker":
                out = await env.execute(startup_command)
            else:
                out = await asyncio.to_thread(env.execute, startup_command)
            if out.get("returncode", 0) != 0:
                raise RuntimeError(f"Error executing startup command: {out}")
            startup_elapsed = time.time() - startup_start
            logger.info("Startup command executed in %.2fs", startup_elapsed)

        total_elapsed = time.time() - overall_start
        logger.info("Total environment setup completed in %.2fs", total_elapsed)
        env._created_at = time.time()
        return env
    except BaseException:
        # Clean up partially-created environment on any failure, including
        # CancelledError (raised by asyncio.wait_for on timeout).  Without
        # this, the underlying aiohttp ClientSession inside
        # AzureDockerEnvironment leaks and produces "Unclosed client session /
        # Unclosed connector" warnings at interpreter shutdown.
        if env is not None:
            logger.warning("create_environment failed; cleaning up partially-created env for %s", instance_id)
            try:
                await stop_environment(env)
            except Exception as cleanup_exc:
                logger.warning("Cleanup of partially-created env failed: %s", cleanup_exc)
        raise


async def stop_environment(env: Environment) -> None:
    """Safely stop and cleanup an environment.

    Tries multiple cleanup methods to ensure the environment is properly stopped.

    Args:
        env: Environment instance to stop.
    """
    if hasattr(env, "_created_at"):
        lifetime = time.time() - env._created_at
        logger.info("Environment lived for %.2fs before stopping", lifetime)
    try:
        if HAS_AZURE_DOCKER and isinstance(env, AzureDockerEnvironment):
            await asyncio.wait_for(env.cleanup(), timeout=300)
        elif hasattr(env, "stop"):
            env.stop()
        elif hasattr(env, "cleanup"):
            env.cleanup()
        elif hasattr(env, "close"):
            env.close()
    except BaseException as e:
        logger.warning("Environment cleanup failed: %s: %s", type(e).__name__, e)
