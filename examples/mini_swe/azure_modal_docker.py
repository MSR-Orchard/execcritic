import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any
import random
from rich.markup import escape as rich_escape


# Default endpoint; override via AZURE_CAAS_ENDPOINT env var
DEFAULT_CAAS_ENDPOINT = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Sandbox manifest tracking – persists sandbox IDs + endpoint to disk so that
# a separate cleanup script can delete leaked sandboxes after Ctrl+C.
# Override the directory via AZURE_SANDBOX_MANIFEST_DIR env var.
# ---------------------------------------------------------------------------
_SANDBOX_MANIFEST_DIR = os.environ.get(
    "AZURE_SANDBOX_MANIFEST_DIR",
    "/tmp/.azure_sandboxes",
)


def _save_sandbox_id(sandbox_id: str, endpoint: str) -> None:
    """Record a sandbox ID and its endpoint as a JSON marker file."""
    os.makedirs(_SANDBOX_MANIFEST_DIR, exist_ok=True)
    marker = os.path.join(_SANDBOX_MANIFEST_DIR, sandbox_id)
    with open(marker, "w") as f:
        json.dump({"sandbox_id": sandbox_id, "endpoint": endpoint}, f)


def _remove_sandbox_id(sandbox_id: str) -> None:
    """Remove the marker file for a sandbox ID."""
    marker = os.path.join(_SANDBOX_MANIFEST_DIR, sandbox_id)
    try:
        os.remove(marker)
    except OSError:
        pass


def _list_sandbox_records() -> list[dict[str, str]]:
    """Return all sandbox records (sandbox_id + endpoint) from the manifest directory."""
    if not os.path.isdir(_SANDBOX_MANIFEST_DIR):
        return []
    records = []
    for name in os.listdir(_SANDBOX_MANIFEST_DIR):
        marker = os.path.join(_SANDBOX_MANIFEST_DIR, name)
        try:
            with open(marker) as f:
                records.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            # Fallback: treat filename as sandbox_id with unknown endpoint
            records.append({"sandbox_id": name, "endpoint": ""})
    return records

@dataclass
class AzureDockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    exe_timeout: int = 900
    """Timeout for executing commands in the container."""
    cpu: str = "2"
    """Number of CPUs to allocate to the container."""
    memory: str = "8Gi"
    """Memory to allocate to the container."""
    container_ready_timeout: int = 1200
    """Max duration (seconds) to wait for container to become ready."""
    cleanup_timeout: int = 300
    """Max duration (seconds) to wait for container deletion/close operations."""
    caas_endpoint: str = ""
    """Azure CaaS endpoint URL. If empty, uses AZURE_CAAS_ENDPOINT env var or default."""
    heartbeat_interval: int = 90
    """Interval (seconds) for sending heartbeat signals to keep the container alive."""
    

class AzureDockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = AzureDockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """Execute bash commands in an Azure CaaS sandbox container.

        See `AzureDockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container = None
        self._sandbox_id = None  # stored for fallback cleanup if container handle is lost
        self.config = config_class(**kwargs)
        self.logger.info(f"Using configuration: {self.config}")

        # Resolve CaaS endpoint
        self._endpoint = (
            self.config.caas_endpoint
            or os.environ.get("AZURE_CAAS_ENDPOINT")
            or DEFAULT_CAAS_ENDPOINT
        )

        # Lazy import: the azure-modal client path can be set via AZURE_MODAL_PATH env var
        azure_modal_path = os.environ.get("AZURE_MODAL_PATH")
        if azure_modal_path:
            import sys
            if azure_modal_path not in sys.path:
                sys.path.insert(0, azure_modal_path)
        from client import AsyncSandboxClient

        self.client = AsyncSandboxClient(self._endpoint)

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    async def start_container(self, max_retries: int = 5):
        """Start the Azure sandbox container with retry on transient failures."""
        self.logger.info(f"Starting container with image {self.config.image}")

        self.logger.info(f"[Env] start self.client.create_sandbox() for {self.config.image}")
        self.container = await self.client.create_sandbox(
                            image=self.config.image,
                            cpu=self.config.cpu,
                            memory=self.config.memory,
                            timeout=self.config.container_ready_timeout,
                            block_network=False,
                        )
        self._sandbox_id = getattr(self.container, "sandbox_id", None)
        if self._sandbox_id:
            _save_sandbox_id(self._sandbox_id, self._endpoint)
        self.logger.info(f"[Env] complete self.client.create_sandbox() for {self.config.image}")

        if self.config.heartbeat_interval > 0:
            jittered_interval = self.config.heartbeat_interval + random.randint(-30, 30)
            jittered_interval = max(30, jittered_interval)  # ensure positive interval
            self.logger.info(f"Starting heartbeat task with interval {jittered_interval} seconds (base: {self.config.heartbeat_interval})")
            self.container.start_heartbeat(interval=jittered_interval)


        # Set default working directory in bashrc
        bashrc_command = f"echo 'cd {self.config.cwd}' >> /root/.bashrc"
        try:
            result = await asyncio.wait_for(
                self.container.exec(
                    bashrc_command, timeout=self.config.exe_timeout, cwd=self.config.cwd,
                    login_shell=True, env=None,
                ),
                timeout=self.config.exe_timeout + 60,
            )
        except asyncio.TimeoutError:
            self.logger.error("Timed out executing bashrc setup command in new container")
            raise
        except Exception as e:
            self.logger.error(f"Error executing bashrc setup command in new container: {e}")
            raise
        if result.stderr:
            self.logger.info(f"Executed bashrc command with output: {result.stdout}, error: {result.stderr}")

    async def execute(self, command: str) -> dict[str, Any]:
        """Execute a command in the container and return the result as a dict."""
        assert self.container is not None, "Container not started"

        #await asyncio.sleep(random.random())
        env = dict(self.config.env)

        try:
            result = await asyncio.wait_for(
                self.container.exec(
                    command, timeout=self.config.exe_timeout, cwd=self.config.cwd,
                    login_shell=True, env=env,
                ),
                timeout=self.config.exe_timeout + 30,
            )
        except asyncio.TimeoutError:
            self.logger.error(
                "Container exec timed out (container may have been killed by Azure). "
                f"command={command!r}"
            )
            return {"output": "error: container exec timed out — container may have been terminated", "returncode": 1}
        except Exception as e:
            self.logger.error(f"Container exec failed (container may have been killed): {rich_escape(str(e))}")
            return {"output": f"error: container exec failed — {e}", "returncode": 1}

        # Print command sent and command output for easier debugging in Azure logs
        # Escape Rich markup characters to prevent MarkupError
        # safe_command = rich_escape(str(command))[:100] + "..."
        # self.logger.info("---------------------------------------------")
        # self.logger.info(f"Executed command: {safe_command}")
        # for k in sorted(result.__dict__):
        #     value = rich_escape(str(result.__dict__[k]))[:100] + "..."  # limit length to prevent log flooding
        #     self.logger.info(f"{k}: {value}")
        # self.logger.info("---------------------------------------------")

        if result.error is not None:
            safe_error = rich_escape(str(result.error))
            self.logger.error(f"Error running command: {safe_error}")
            return {"output": f"error running the cmd: {result.error}", "returncode": 1}
        # Merge stderr into output (same as minisweagent's DockerEnvironment)
        # Test frameworks like unittest write results to stderr
        output = result.stdout or ""
        if result.stderr:
            output = output + result.stderr
        return {"output": output, "returncode": result.exit_code}

    async def cleanup(self):
        """Stop and remove the container."""
        self.logger.info("Shutting the container down")
        sandbox_id = getattr(self.container, "sandbox_id", None) or self._sandbox_id

        deleted = False
        try:
            if self.container is not None:
                await asyncio.wait_for(self.container.delete(), timeout=self.config.cleanup_timeout)
                deleted = True
        except asyncio.TimeoutError:
            self.logger.warning("Container delete timed out (container may already be gone)")
        except Exception as e:
            self.logger.warning(f"Container delete failed (container may already be gone): {e}")

        # Best-effort fallback deletion by sandbox id if the instance delete path failed.
        if sandbox_id and hasattr(self.client, "delete_sandbox"):
            try:
                await asyncio.wait_for(self.client.delete_sandbox(sandbox_id), timeout=self.config.cleanup_timeout)
                deleted = True
            except Exception:
                pass

        # Only remove the manifest marker after the sandbox has been successfully deleted.
        # If deletion failed, keep the marker so that cleanup_azure_sandboxes.py can retry.
        if deleted and sandbox_id:
            _remove_sandbox_id(sandbox_id)
        elif sandbox_id:
            self.logger.warning(f"Sandbox {sandbox_id} marker kept on disk — deletion was not confirmed")

        # Close the underlying aiohttp session to avoid "Unclosed client session" warnings.
        try:
            if hasattr(self.client, "close"):
                await asyncio.wait_for(self.client.close(cleanup=False), timeout=self.config.cleanup_timeout)
            elif hasattr(self.client, "_session") and self.client._session and not self.client._session.closed:
                await asyncio.wait_for(self.client._session.close(), timeout=self.config.cleanup_timeout)
            elif hasattr(self.client, "session") and self.client.session and not self.client.session.closed:
                await asyncio.wait_for(self.client.session.close(), timeout=self.config.cleanup_timeout)
        except Exception as e:
            self.logger.warning(f"Client session close failed: {e}")

        self.container = None
        self.logger.info("Shutting the container down succeeded")

    def __del__(self):
        if self.container is None:
            return
        else:
            self.container.delete_sync()
