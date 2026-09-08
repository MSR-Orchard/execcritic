import logging
import os
import time
from typing import Any

from pydantic import BaseModel

from minisweagent.environments.submission import raise_if_submitted
from minisweagent.utils.serialize import recursive_merge


class AzureModalEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the sandbox."""
    forward_env: list[str] = []
    """Environment variables to forward from the host to the sandbox."""
    timeout: int = 30
    """Timeout for executing commands in the sandbox."""
    base_url: str = os.getenv("SANDBOX_BASE_URL", "http://localhost:8000")
    """Base URL of the Azure Modal sandbox orchestrator service."""
    api_key: str = os.getenv("SANDBOX_API_KEY", "")
    """API key for authenticating with the orchestrator."""
    block_network: bool = True
    """Whether to block network egress in the sandbox."""
    cpu: str | None = None
    """CPU request/limit (e.g., '4', '2000m')."""
    memory: str | None = None
    """Memory request/limit (e.g., '16Gi')."""
    sandbox_timeout: int = 3600
    """Timeout in seconds for sandbox to become ready."""
    sandbox_ready_poll_interval: float = 0
    """Poll readiness at this interval; 0 uses the server-side long wait."""
    login_shell: bool = True
    """Whether to use login shell (bash -lc) for command execution."""
    heartbeat_interval: int = 60
    """Interval in seconds between heartbeats to keep sandbox alive. 0 to disable."""
    request_timeout: int = 1200
    """Default client request timeout in seconds."""
    max_output_chars: int = 200_000
    """Maximum returned tool-output characters; 0 disables truncation."""
    cleanup_timeout: int = 120
    """Timeout in seconds for each sandbox deletion attempt."""
    cleanup_retries: int = 3
    """Number of deletion attempts before leaving the durable marker for recovery."""
    cleanup_retry_wait: float = 1.0
    """Delay in seconds between deletion attempts."""


class AzureModalEnvironment:
    def __init__(
        self,
        *,
        config_class: type = AzureModalEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """Execute commands in a remote sandbox via Azure Modal sandbox client SDK.
        See `AzureModalEnvironmentConfig` for keyword arguments.
        """
        from client.sandbox_client import SandboxClient

        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self._sandbox = None
        self._cleaned_up = False
        self._client = SandboxClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key or None,
            timeout=self.config.request_timeout,
        )
        try:
            self._sandbox = self._client.create_sandbox(
                image=self.config.image,
                block_network=self.config.block_network,
                cpu=self.config.cpu,
                memory=self.config.memory,
                timeout=self.config.sandbox_timeout,
                poll_interval=self.config.sandbox_ready_poll_interval or 1.0,
                use_server_wait=self.config.sandbox_ready_poll_interval <= 0,
            )
            self.logger.info(f"Created sandbox {self._sandbox.sandbox_id}")
            if self.config.heartbeat_interval > 0:
                self._sandbox.start_heartbeat(interval=self.config.heartbeat_interval)

            bashrc_command = f"echo 'cd {self.config.cwd}' >> /root/.bashrc"
            self._sandbox.exec(
                command=bashrc_command,
                timeout=self.config.timeout,
                cwd=self.config.cwd,
                login_shell=True,
                env=None,
            )
        except BaseException:
            try:
                self.cleanup()
            except Exception as cleanup_exc:
                self.logger.warning("Failed to clean up partially-created sandbox: %s", cleanup_exc)
            raise

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        environment_config = self.config.model_dump(mode="json")
        if environment_config.get("api_key"):
            environment_config["api_key"] = "***REDACTED***"
        return {
            "info": {
                "config": {
                    "environment": environment_config,
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def execute(self, action: dict, cwd: str = "/", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the sandbox and return the result as a dict."""
        command = action.get("command", "")
        cwd = self.config.cwd if cwd == '/' else cwd  # use config.cwd if cwd is not specified or is root
        exec_timeout = timeout or self.config.timeout

        # Merge environment variables
        env: dict[str, str] = {}
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                env[key] = value
        env.update(self.config.env)

        # self.logger.debug(f"[exec] running cmd={command!r} ...")
        try:
            t0 = time.perf_counter()
            result = self._sandbox.exec(
                command=command,
                timeout=exec_timeout,
                cwd=cwd,
                env=env or None,
                login_shell=self.config.login_shell,
            )
            elapsed = time.perf_counter() - t0
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            combined = stdout + stderr if stderr else stdout
            if self.config.max_output_chars > 0 and len(combined) > self.config.max_output_chars:
                marker = (
                    "\n... [tool output truncated from "
                    f"{len(combined)} to {self.config.max_output_chars} characters] ...\n"
                )
                kept_chars = self.config.max_output_chars - len(marker)
                if kept_chars > 0:
                    head_chars = (kept_chars + 1) // 2
                    tail_chars = kept_chars - head_chars
                    head = combined[:head_chars]
                    tail = combined[-tail_chars:] if tail_chars else ""
                    combined = head + marker + tail
                else:
                    combined = combined[: self.config.max_output_chars]
            output = {
                "output": combined,
                "returncode": result.exit_code if result.exit_code is not None else (-1 if result.failed else 0),
                "exception_info": result.error or "",
            }
            # self.logger.debug(f"[exec] cmd={command!r} elapsed={elapsed:.2f}s rc={output['returncode']}")
            # self.logger.debug(f"[exec] output={combined[:500]}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
            # self.logger.debug(f"[exec] cmd={command!r} elapsed={elapsed:.2f}s EXCEPTION={e}")

        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        raise_if_submitted(output)

    def cleanup(self):
        """Delete the sandbox and close the client."""
        if self._cleaned_up:
            return

        if self._sandbox is None:
            self._client.close()
            self._cleaned_up = True
            return

        sandbox_id = self._sandbox.sandbox_id
        last_error = None
        for attempt in range(1, max(1, self.config.cleanup_retries) + 1):
            try:
                self._sandbox.delete(timeout=self.config.cleanup_timeout)
                self.logger.info(f"Deleted sandbox {sandbox_id}")
                self._client.close()
                self._cleaned_up = True
                return
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Failed to delete sandbox %s (attempt %s/%s): %s",
                    sandbox_id,
                    attempt,
                    max(1, self.config.cleanup_retries),
                    exc,
                )
                if attempt < max(1, self.config.cleanup_retries):
                    time.sleep(max(0.0, self.config.cleanup_retry_wait))

        raise RuntimeError(
            f"Failed to delete sandbox {sandbox_id} after {max(1, self.config.cleanup_retries)} attempts"
        ) from last_error

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
