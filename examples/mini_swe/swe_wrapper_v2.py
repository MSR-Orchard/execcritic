"""
Wrapper module for mini-swe-agent v2 (tool-call based) with slime's RL training framework.

v2 differences from v1:
- Uses OpenAI-style tool/function calling instead of regex-based action parsing.
- Assistant responses contain tool_calls with structured arguments.
- Observations are returned as 'tool' role messages with tool_call_id.
- Follows the DefaultAgent (default.py) pattern from mini-swe-agent.
"""

import asyncio
import json
import logging
import re
import subprocess
import time
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment

# Import shared environment helpers from v1
from .swe_wrapper import (
    create_environment,
    get_docker_image_name,
    load_config,
    stop_environment,
)

try:
    from .azure_modal_docker import AzureDockerEnvironment, AzureDockerEnvironmentConfig
    HAS_AZURE_DOCKER = True
except ImportError:
    HAS_AZURE_DOCKER = False
    AzureDockerEnvironment = None
    AzureDockerEnvironmentConfig = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definition (matches mini-swe-agent v2 BASH_TOOL)
# ---------------------------------------------------------------------------

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

TOOLS = [BASH_TOOL]


# ---------------------------------------------------------------------------
# Exceptions (follows InterruptAgentFlow pattern from default.py)
# ---------------------------------------------------------------------------

class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages.
    Mirrors minisweagent.exceptions.InterruptAgentFlow.
    """
    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format."""


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class TimeExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its wall-clock time limit."""


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""
    def __init__(self, submission: str):
        self.submission = submission
        super().__init__({
            "role": "exit",
            "content": submission,
            "extra": {"exit_status": "Submitted", "submission": submission},
        })


class EnvironmentUnavailable(Exception):
    """Raised when the remote environment is no longer reachable (e.g., container 404)."""


class NonTerminatingException(Exception):
    """Kept for backward compatibility."""


def _is_environment_error(output: str) -> str:
    """Classify command output as an environment error type.

    Returns:
        "env_unavailable" for irrecoverable environment errors (e.g., container
            deleted or expired).
        "exec_timeout" for remote execution timeouts reported by the sandbox.
        "" (empty string) if the output does not indicate an environment error.
    """
    if "container exec failed" in output and ("404" in output or "Not Found" in output):
        return "env_unavailable"
    if "error running the cmd: Execution timed out after" in output:
        return "exec_timeout"
    return ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Agent configuration. Check the config files in configs/ for example settings."""

    system_template: str
    """Template for the system message."""
    instance_template: str
    """Template for the first user message specifying the task."""
    observation_template: str
    """Template for formatting tool execution output."""
    format_error_template: str
    """Template for format error messages."""
    timeout_template: str
    """Template for timeout error messages."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding this cost."""
    time_limit: float = 2400.0
    """Maximum wall-clock time in seconds (0 = no limit). Default 2400s (40 min)."""
    use_sglang_parser: bool = False
    """If True, parse_response delegates to sglang's FunctionCallParser instead of
    the built-in regex/JSON parser."""
    sglang_tool_call_parser: str = "qwen25"
    """Name of the sglang tool-call parser to use when use_sglang_parser is True
    (e.g. 'qwen25', 'qwen3_coder')."""


# ---------------------------------------------------------------------------
# SWEAgentV2 – tool-call based agent following DefaultAgent pattern
# ---------------------------------------------------------------------------

class SWEAgentV2:
    """Agent for SWE-bench tasks using tool-call based interaction (v2).

    Follows the DefaultAgent interface from mini-swe-agent's default.py,
    adapted for slime's async inference engine.
    """

    def __init__(self, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.env = env
        self.extra_template_vars: dict[str, Any] = {}
        self.n_calls = 0
        self.cost = 0.0
        self._created_at = time.time()
        self._exec_timeout_count = 0
        self._sglang_parser = None

    # -- template helpers ---------------------------------------------------

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {
            **self.config.model_dump(),
            **self.env.get_template_vars(),
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            **self.extra_template_vars,
            **kwargs,
        }

    def render_template(self, template: str, **kwargs) -> str:
        return Template(template, undefined=StrictUndefined).render(
            **self.get_template_vars(**kwargs)
        )

    # -- message management (matches DefaultAgent.add_messages) -------------

    def add_messages(self, *messages: dict) -> list[dict]:
        """Add one or more messages to the conversation. Returns the added messages."""
        for msg in messages:
            msg.setdefault("timestamp", time.time())
        self.messages.extend(messages)
        return list(messages)

    # -- setup (matches DefaultAgent.run setup) -----------------------------

    def setup(self, task: str, **kwargs):
        """Initialize messages with system and instance prompts."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            {"role": "system", "content": self.render_template(self.config.system_template)},
            {"role": "user", "content": self.render_template(self.config.instance_template)},
        )

    # -- limit checking (matches DefaultAgent.query limit check) ------------

    def check_limits(self):
        """Check step, cost, and time limits. Raises LimitsExceeded or TimeExceeded."""
        if (0 < self.config.step_limit <= self.n_calls) or (0 < self.config.cost_limit <= self.cost):
            raise LimitsExceeded({
                "role": "exit",
                "content": "LimitsExceeded",
                "extra": {"exit_status": "LimitsExceeded", "submission": ""},
            })
        if 0 < self.config.time_limit <= (time.time() - self._created_at):
            elapsed = time.time() - self._created_at
            logger.warning(
                "TimeExceeded: agent has been running for %.1fs (limit %.1fs)",
                elapsed, self.config.time_limit,
            )
            raise TimeExceeded({
                "role": "exit",
                "content": "TimeExceeded",
                "extra": {"exit_status": "TimeExceeded", "submission": ""},
            })

    # -- response parsing (replaces model.query + _parse_actions) -----------

    def parse_response(self, response_text: str) -> dict:
        """Parse model response text into a message dict with extra.actions.

        Mirrors what model.query() returns in DefaultAgent: a message dict with
        role="assistant", content, tool_calls, and extra.actions.

        Raises FormatError (with message dicts) if no valid tool calls found.
        """
        if self.config.use_sglang_parser:
            return self.parse_response_sglang(response_text)
        # Parse tool calls from text (Qwen3/ChatML format or JSON format)
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tool_call_pattern, response_text, re.DOTALL)

        if not matches:
            json_pattern = r'\{"name"\s*:\s*"bash"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)

        if not matches:
            raise FormatError({
                "role": "user",
                "content": self.render_template(
                    self.config.format_error_template,
                    error="No tool calls found in the response. Every response MUST include at least one bash tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            })

        actions = []
        formatted_tool_calls = []

        for i, match in enumerate(matches):
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError as e:
                raise FormatError({
                    "role": "user",
                    "content": self.render_template(
                        self.config.format_error_template,
                        error=f"Error parsing tool call JSON: {e}",
                        actions=[],
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                })

            name = parsed.get("name", "")
            if name != "bash":
                raise FormatError({
                    "role": "user",
                    "content": self.render_template(
                        self.config.format_error_template,
                        error=f"Unknown tool '{name}'. Only 'bash' tool is supported.",
                        actions=[],
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                })

            arguments = parsed.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass

            if not isinstance(arguments, dict) or "command" not in arguments:
                raise FormatError({
                    "role": "user",
                    "content": self.render_template(
                        self.config.format_error_template,
                        error="Missing 'command' argument in bash tool call.",
                        actions=[],
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                })

            tool_call_id = f"call_{self.n_calls}_{i}"
            actions.append({
                "command": arguments["command"],
                "tool_call_id": tool_call_id,
            })
            formatted_tool_calls.append({
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": arguments,
                },
                "id": tool_call_id,
            })

        return {
            "tool_calls": formatted_tool_calls,
            "extra": {"actions": actions},
        }

    # -- sglang-based response parsing --------------------------------------

    def _get_sglang_parser(self):
        """Lazily build and cache an sglang FunctionCallParser for our tools."""
        if self._sglang_parser is None:
            from sglang.srt.entrypoints.openai.protocol import Function, Tool
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            sg_tools = [
                Tool(type="function", function=Function(**t["function"])) for t in TOOLS
            ]
            self._sglang_parser = FunctionCallParser(
                tools=sg_tools, tool_call_parser=self.config.sglang_tool_call_parser
            )
        return self._sglang_parser

    def parse_response_sglang(self, response_text: str) -> dict:
        """Parse model response text using sglang's FunctionCallParser.

        Drop-in alternative to parse_response() that returns the same dict shape
        ({"tool_calls": [...], "extra": {"actions": [...]}}) but delegates the
        format-specific parsing to sglang. The active parser is selected by
        AgentConfig.sglang_tool_call_parser (e.g. 'qwen25', 'qwen3_coder').

        Raises FormatError (with message dicts) if no valid bash tool calls found.
        """
        parser = self._get_sglang_parser()

        try:
            _normal_text, calls = parser.parse_non_stream(response_text)
        except Exception as e:
            raise FormatError({
                "role": "user",
                "content": self.render_template(
                    self.config.format_error_template,
                    error=f"Error parsing tool calls: {e}",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            })

        if not calls:
            raise FormatError({
                "role": "user",
                "content": self.render_template(
                    self.config.format_error_template,
                    error="No tool calls found in the response. Every response MUST include at least one bash tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            })

        actions = []
        formatted_tool_calls = []

        for i, call in enumerate(calls):
            name = call.name or ""
            if name != "bash":
                raise FormatError({
                    "role": "user",
                    "content": self.render_template(
                        self.config.format_error_template,
                        error=f"Unknown tool '{name}'. Only 'bash' tool is supported.",
                        actions=[],
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                })

            try:
                arguments = json.loads(call.parameters or "{}")
            except json.JSONDecodeError as e:
                raise FormatError({
                    "role": "user",
                    "content": self.render_template(
                        self.config.format_error_template,
                        error=f"Error parsing tool call arguments JSON: {e}",
                        actions=[],
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                })

            if not isinstance(arguments, dict) or "command" not in arguments:
                raise FormatError({
                    "role": "user",
                    "content": self.render_template(
                        self.config.format_error_template,
                        error="Missing 'command' argument in bash tool call.",
                        actions=[],
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                })

            tool_call_id = f"call_{self.n_calls}_{i}"
            actions.append({
                "command": arguments["command"],
                "tool_call_id": tool_call_id,
            })
            formatted_tool_calls.append({
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": arguments,
                },
                "id": tool_call_id,
            })

        return {
            "tool_calls": formatted_tool_calls,
            "extra": {"actions": actions},
        }

    # -- action execution ---------------------------------------------------

    def _check_submission(self, output: dict[str, str]):
        """Check if output indicates task submission. Raises Submitted if so."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() in [
                "MINI_SWE_AGENT_FINAL_OUTPUT",
                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            ]
            and output.get("returncode") == 0
            #and len("".join(lines[1:])) != 0
        ):
            raise Submitted("".join(lines[1:]))

    async def execute_actions(
        self, actions: list[dict], timeout: float | None = None
    ) -> tuple[list[dict], dict | None]:
        """Execute a list of actions sequentially.

        Args:
            actions: List of action dicts, each with 'command' and 'tool_call_id'.
            timeout: Optional per-action timeout in seconds.

        Returns:
            Tuple of (outputs, submitted_info):
            - outputs: List of output dicts from executed actions.
            - submitted_info: None if no submission, or a dict with 'submission'
              and 'tool_call_id' if the task was submitted.
        """
        outputs = []
        for action in actions:
            command = action["command"]
            tool_call_id = action["tool_call_id"]
            start_time = time.time()
            try:
                coro = (
                    self.env.execute(command)
                    if HAS_AZURE_DOCKER and isinstance(self.env, AzureDockerEnvironment)
                    else asyncio.to_thread(self.env.execute, command)
                )
                if timeout is not None:
                    action_output = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    action_output = await coro
                elapsed = time.time() - start_time
                logger.info("Action execution completed in %.2fs", elapsed)
                # Detect irrecoverable environment errors (e.g., container deleted/expired)
                output_text = action_output.get("output", "")
                if action_output.get("returncode") != 0:
                    error_type = _is_environment_error(output_text)
                    if error_type == "env_unavailable":
                        raise EnvironmentUnavailable(output_text)
                    if error_type == "exec_timeout":
                        self._exec_timeout_count += 1
                        if self._exec_timeout_count > 3:
                            raise EnvironmentUnavailable(output_text)
                self._check_submission(action_output)
                if "exception_info" not in action_output:
                    action_output["exception_info"] = ""
                outputs.append(action_output)
            except (TimeoutError, asyncio.TimeoutError, subprocess.TimeoutExpired) as e:
                elapsed = time.time() - start_time
                logger.warning(
                    "Action execution timed out after %.2fs for tool_call %s",
                    elapsed, tool_call_id,
                )
                output_text = (
                    e.output.decode("utf-8", errors="replace")
                    if getattr(e, "output", None)
                    else ""
                )
                content = self.render_template(
                    self.config.timeout_template, output=output_text
                )
                outputs.append({
                    "output": content,
                    "returncode": -1,
                    "exception_info": "",
                })
            except Submitted as e:
                return outputs, {"submission": e.submission, "tool_call_id": tool_call_id}
            except EnvironmentUnavailable:
                raise  # Propagate to retry loop — do not add to messages
            except Exception as e:
                logger.warning(
                    "Unexpected error during execute_action: %s", e, exc_info=True,
                )
                outputs.append({
                    "output": f"Error: unexpected failure during action execution: {e}",
                    "returncode": -1,
                    "exception_info": str(e),
                })
        return outputs, None

    # -- observation formatting (matches DefaultAgent.execute_actions) ------

    def format_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]:
        """Format execution outputs as tool response messages.

        Mirrors model.format_observation_messages() from default.py.
        Creates one tool response message per action/output pair.
        """
        actions = message.get("extra", {}).get("actions", [])
        not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
        padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))

        results = []
        for action, output in zip(actions, padded_outputs):
            content = self.render_template(
                self.config.observation_template, output=output
            )
            msg = {
                "role": "tool",
                "content": content,
                "tool_call_id": action.get("tool_call_id", ""),
            }
            results.append(msg)
        return results

    # -- serialisation (matches DefaultAgent.serialize) ---------------------

    def serialize(self) -> dict:
        """Serialize agent state to a JSON-compatible dictionary."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        return {
            "info": {
                "model_stats": {
                    "api_calls": self.n_calls,
                    "instance_cost": self.cost,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
        }

    def to_dict(self) -> dict:
        """Alias for serialize() for backward compatibility."""
        return self.serialize()


# Config / environment factory helpers are imported from swe_wrapper (v1):
# load_config, get_docker_image_name, create_environment, stop_environment
