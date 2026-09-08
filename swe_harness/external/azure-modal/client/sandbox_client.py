"""Small sync/async client for Sandbox Orchestrator 0.1.x.

The implementation intentionally covers the API surface used by this source kit:
create/wait, command execution, heartbeat, file transfer, patch application, and
deletion.  It is implemented against the service OpenAPI contract and has no
dependency on the separately distributed ``aks_modal`` client package.
"""

from __future__ import annotations

import atexit
import asyncio
import base64
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiohttp
import requests


_pending_cleanup: set[tuple[str, str, str | None]] = set()
_cleanup_registered = False


def _url_id(value: str) -> str:
    return quote(value, safe="")


def _manifest_dir() -> Path:
    return Path(os.environ.get("AZURE_SANDBOX_MANIFEST_DIR", "/tmp/.azure_sandboxes"))


def _manifest_path(sandbox_id: str) -> Path:
    return _manifest_dir() / sandbox_id.replace("/", "_")


def _save_marker(base_url: str, sandbox_id: str) -> None:
    try:
        directory = _manifest_dir()
        directory.mkdir(parents=True, exist_ok=True)
        _manifest_path(sandbox_id).write_text(
            json.dumps({"sandbox_id": sandbox_id, "endpoint": base_url}) + "\n",
            encoding="utf-8",
        )
    except OSError:
        # A durable marker improves recovery, but inability to write one must not
        # hide a successfully created remote sandbox from the caller.
        pass


def _forget(base_url: str, sandbox_id: str, api_key: str | None) -> None:
    _pending_cleanup.discard((base_url, sandbox_id, api_key))
    try:
        _manifest_path(sandbox_id).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _register_cleanup() -> None:
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True

    def cleanup() -> None:
        for base_url, sandbox_id, api_key in list(_pending_cleanup):
            headers = {"X-API-Key": api_key} if api_key else {}
            try:
                response = requests.delete(
                    f"{base_url}/sandboxes/{_url_id(sandbox_id)}",
                    headers=headers,
                    timeout=5,
                )
                if response.status_code not in (200, 404):
                    continue
            except Exception:
                continue
            _forget(base_url, sandbox_id, api_key)

    atexit.register(cleanup)


class JobResult:
    """Attribute-oriented view of a command-job response."""

    def __init__(self, data: dict[str, Any]):
        self.job_id = str(data.get("job_id", ""))
        self.sandbox_id = str(data.get("sandbox_id", ""))
        self.command = data.get("command", "")
        self.status = str(data.get("status", "queued"))
        self.stdout = str(data.get("stdout") or "")
        self.stderr = str(data.get("stderr") or "")
        self.exit_code = data.get("exit_code")
        self.error = data.get("error")
        self.created_at = data.get("created_at", time.time())
        self.started_at = data.get("started_at")
        self.completed_at = data.get("completed_at")

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    @property
    def is_complete(self) -> bool:
        return self.status in {"succeeded", "failed"}

    def __repr__(self) -> str:
        return f"JobResult(status={self.status!r}, exit_code={self.exit_code!r})"


class SandboxInstance:
    def __init__(self, client: "SandboxClient", sandbox_id: str, data: dict[str, Any]):
        self._client = client
        self.sandbox_id = sandbox_id
        self.namespace = data.get("namespace")
        self.image = data.get("image")
        self.block_network = data.get("block_network")
        self._deleted = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def _ensure_active(self) -> None:
        if self._deleted:
            raise RuntimeError("Sandbox has been deleted")

    def start_heartbeat(self, interval: int = 60) -> None:
        if interval <= 0:
            raise ValueError("heartbeat interval must be positive")
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()

        def run() -> None:
            while not self._heartbeat_stop.wait(interval):
                if self._deleted:
                    return
                try:
                    self._client._request(
                        "POST", f"/sandboxes/{_url_id(self.sandbox_id)}/heartbeat"
                    )
                except Exception:
                    pass

        self._heartbeat_thread = threading.Thread(
            target=run, daemon=True, name=f"heartbeat-{self.sandbox_id[:8]}"
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

    def exec(
        self,
        command: str | list[str],
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        wait: bool = True,
        poll_interval: float = 0.1,
        login_shell: bool = False,
    ) -> JobResult:
        self._ensure_active()
        exec_timeout = timeout or 300
        response = self._client._request(
            "POST",
            f"/sandboxes/{_url_id(self.sandbox_id)}/exec",
            json={
                "command": command,
                "timeout_seconds": timeout,
                "cwd": cwd,
                "env": env,
                "login_shell": login_shell,
                "wait": wait,
            },
            timeout=exec_timeout + 60,
        )
        result = JobResult(response)
        if not wait or result.is_complete:
            return result
        if not result.job_id:
            raise RuntimeError("exec response did not include job_id")
        deadline = time.monotonic() + exec_timeout + 60
        while time.monotonic() < deadline:
            result = self.get_job(result.job_id)
            if result.is_complete:
                return result
            time.sleep(max(0.01, poll_interval))
        raise TimeoutError(f"Job {result.job_id} did not complete within {exec_timeout}s")

    def get_job(self, job_id: str) -> JobResult:
        return JobResult(self._client._request("GET", f"/jobs/{_url_id(job_id)}"))

    def apply_patch(self, patch: str, timeout: int = 30) -> dict[str, Any]:
        self._ensure_active()
        return self._client._request(
            "POST",
            f"/sandboxes/{_url_id(self.sandbox_id)}/apply_patch",
            json={"patch": patch, "timeout_seconds": timeout},
        )

    def upload_content(self, content: bytes, remote_path: str) -> dict[str, Any]:
        self._ensure_active()
        return self._client._request(
            "POST",
            f"/sandboxes/{_url_id(self.sandbox_id)}/files",
            json={"path": remote_path, "content": base64.b64encode(content).decode("ascii")},
        )

    def upload_file(self, local_path: str, remote_path: str) -> dict[str, Any]:
        return self.upload_content(Path(local_path).read_bytes(), remote_path)

    def download_content(self, remote_path: str) -> bytes:
        self._ensure_active()
        response = self._client._request(
            "GET",
            f"/sandboxes/{_url_id(self.sandbox_id)}/files",
            params={"path": remote_path},
        )
        return base64.b64decode(response["content"], validate=True)

    def download_file(self, remote_path: str, local_path: str) -> int:
        content = self.download_content(remote_path)
        Path(local_path).write_bytes(content)
        return len(content)

    def list_files(self, remote_path: str = "/") -> list[dict[str, Any]]:
        self._ensure_active()
        response = self._client._request(
            "GET",
            f"/sandboxes/{_url_id(self.sandbox_id)}/files/list",
            params={"path": remote_path},
        )
        return list(response["files"])

    def delete(self, timeout: int | None = None) -> None:
        if self._deleted:
            return
        self.stop_heartbeat()
        self._client.delete_sandbox(self.sandbox_id, timeout=timeout)
        self._deleted = True

    def __enter__(self) -> "SandboxInstance":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.delete()


class SandboxClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 1200,
        auto_cleanup: bool = True,
        api_key: str | None = None,
        prefix: str | None = None,
    ):
        self.base_url = (
            base_url or os.environ.get("SANDBOX_BASE_URL") or "http://localhost:8000"
        ).rstrip("/")
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("SANDBOX_API_KEY")
        self.prefix = prefix or os.environ.get("SANDBOX_PREFIX")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["X-API-Key"] = self.api_key
        self._created_sandboxes: set[str] = set()
        self._auto_cleanup = auto_cleanup
        if auto_cleanup:
            _register_cleanup()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("timeout", self.timeout)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.request(method, f"{self.base_url}{path}", **kwargs)
                if response.status_code == 503 and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    def _make_id(self, sandbox_id: str | None) -> str | None:
        if not self.prefix:
            return sandbox_id
        if sandbox_id and sandbox_id.startswith(f"{self.prefix}-"):
            return sandbox_id
        return f"{self.prefix}-{sandbox_id or uuid.uuid4().hex[:8]}"

    @staticmethod
    def _is_ready(data: dict[str, Any]) -> bool:
        return bool(data.get("ready")) or data.get("status") == "ready"

    @staticmethod
    def _raise_failed(sandbox_id: str, data: dict[str, Any]) -> None:
        if data.get("status") == "failed":
            message = data.get("status_message") or "sandbox failed"
            raise RuntimeError(f"Sandbox {sandbox_id} failed: {message}")

    def create_sandbox(
        self,
        image: str,
        block_network: bool = True,
        sandbox_id: str | None = None,
        cpu: str | None = None,
        memory: str | None = None,
        timeout: int | None = None,
        wait_ready: bool = True,
        poll_interval: float = 1.0,
        use_server_wait: bool = True,
    ) -> SandboxInstance:
        requested_id = self._make_id(sandbox_id)
        payload: dict[str, Any] = {"image": image, "block_network": block_network}
        for key, value in {
            "sandbox_id": requested_id,
            "cpu": cpu,
            "memory": memory,
            "timeout": timeout,
        }.items():
            if value is not None:
                payload[key] = value

        created_id: str | None = None
        try:
            created = self._request("POST", "/sandboxes", json=payload)
            created_id = str(created["sandbox_id"])
            self._created_sandboxes.add(created_id)
            if self._auto_cleanup:
                _pending_cleanup.add((self.base_url, created_id, self.api_key))
            _save_marker(self.base_url, created_id)
            if not wait_ready:
                return SandboxInstance(self, created_id, created)

            ready_timeout = int(created.get("timeout") or timeout or 3600)
            if use_server_wait:
                try:
                    waited = self._request(
                        "GET",
                        f"/sandboxes/{_url_id(created_id)}/wait",
                        params={"timeout": ready_timeout},
                        timeout=ready_timeout + 30,
                    )
                    self._raise_failed(created_id, waited)
                    if self._is_ready(waited):
                        return SandboxInstance(self, created_id, {**created, **waited})
                    raise RuntimeError(
                        f"Sandbox {created_id} wait returned status={waited.get('status')!r}"
                    )
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status == 408:
                        raise TimeoutError(
                            f"Sandbox {created_id} did not become ready within {ready_timeout}s"
                        ) from exc
                    if status not in (404, 405):
                        raise

            deadline = time.monotonic() + ready_timeout
            while time.monotonic() < deadline:
                state = self._request("GET", f"/sandboxes/{_url_id(created_id)}")
                self._raise_failed(created_id, state)
                if self._is_ready(state):
                    return SandboxInstance(self, created_id, {**created, **state})
                time.sleep(max(0.05, poll_interval))
            raise TimeoutError(
                f"Sandbox {created_id} did not become ready within {ready_timeout}s"
            )
        except BaseException:
            if created_id:
                try:
                    self.delete_sandbox(created_id, timeout=min(self.timeout, 120))
                except Exception:
                    pass
            raise

    def get_sandbox(self, sandbox_id: str) -> SandboxInstance:
        data = self._request("GET", f"/sandboxes/{_url_id(sandbox_id)}")
        return SandboxInstance(self, str(data.get("sandbox_id", sandbox_id)), data)

    def delete_sandbox(self, sandbox_id: str, timeout: int | None = None) -> None:
        try:
            self._request(
                "DELETE",
                f"/sandboxes/{_url_id(sandbox_id)}",
                timeout=timeout or min(self.timeout, 120),
            )
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
        self._created_sandboxes.discard(sandbox_id)
        _forget(self.base_url, sandbox_id, self.api_key)

    def cleanup_all(self) -> None:
        for sandbox_id in list(self._created_sandboxes):
            self.delete_sandbox(sandbox_id)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "SandboxClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class AsyncSandboxInstance:
    def __init__(self, client: "AsyncSandboxClient", sandbox_id: str, data: dict[str, Any]):
        self._client = client
        self.sandbox_id = sandbox_id
        self.namespace = data.get("namespace")
        self.image = data.get("image")
        self.block_network = data.get("block_network")
        self._deleted = False
        self._heartbeat_task: asyncio.Task[None] | None = None

    def _ensure_active(self) -> None:
        if self._deleted:
            raise RuntimeError("Sandbox has been deleted")

    def start_heartbeat(self, interval: int = 60) -> None:
        if interval <= 0:
            raise ValueError("heartbeat interval must be positive")
        if self._heartbeat_task and not self._heartbeat_task.done():
            return

        async def run() -> None:
            try:
                while not self._deleted:
                    await asyncio.sleep(interval)
                    if not self._deleted:
                        try:
                            await self._client._request(
                                "POST", f"/sandboxes/{_url_id(self.sandbox_id)}/heartbeat"
                            )
                        except Exception:
                            pass
            except asyncio.CancelledError:
                pass

        self._heartbeat_task = asyncio.create_task(run())

    def stop_heartbeat(self) -> None:
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        self._heartbeat_task = None

    async def exec(
        self,
        command: str | list[str],
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        wait: bool = True,
        poll_interval: float = 0.1,
        login_shell: bool = False,
    ) -> JobResult:
        self._ensure_active()
        exec_timeout = timeout or 300
        request_timeout = aiohttp.ClientTimeout(total=exec_timeout + 60)
        response = await self._client._request(
            "POST",
            f"/sandboxes/{_url_id(self.sandbox_id)}/exec",
            json={
                "command": command,
                "timeout_seconds": timeout,
                "cwd": cwd,
                "env": env,
                "login_shell": login_shell,
                "wait": wait,
            },
            timeout=request_timeout,
        )
        result = JobResult(response)
        if not wait or result.is_complete:
            return result
        if not result.job_id:
            raise RuntimeError("exec response did not include job_id")
        deadline = asyncio.get_running_loop().time() + exec_timeout + 60
        while asyncio.get_running_loop().time() < deadline:
            result = await self.get_job(result.job_id)
            if result.is_complete:
                return result
            await asyncio.sleep(max(0.01, poll_interval))
        raise TimeoutError(f"Job {result.job_id} did not complete within {exec_timeout}s")

    async def get_job(self, job_id: str) -> JobResult:
        return JobResult(await self._client._request("GET", f"/jobs/{_url_id(job_id)}"))

    async def apply_patch(self, patch: str, timeout: int = 30) -> dict[str, Any]:
        self._ensure_active()
        return await self._client._request(
            "POST",
            f"/sandboxes/{_url_id(self.sandbox_id)}/apply_patch",
            json={"patch": patch, "timeout_seconds": timeout},
        )

    async def upload_content(self, content: bytes, remote_path: str) -> dict[str, Any]:
        self._ensure_active()
        return await self._client._request(
            "POST",
            f"/sandboxes/{_url_id(self.sandbox_id)}/files",
            json={"path": remote_path, "content": base64.b64encode(content).decode("ascii")},
        )

    async def upload_file(self, local_path: str, remote_path: str) -> dict[str, Any]:
        return await self.upload_content(Path(local_path).read_bytes(), remote_path)

    async def download_content(self, remote_path: str) -> bytes:
        self._ensure_active()
        response = await self._client._request(
            "GET",
            f"/sandboxes/{_url_id(self.sandbox_id)}/files",
            params={"path": remote_path},
        )
        return base64.b64decode(response["content"], validate=True)

    async def download_file(self, remote_path: str, local_path: str) -> int:
        content = await self.download_content(remote_path)
        Path(local_path).write_bytes(content)
        return len(content)

    async def list_files(self, remote_path: str = "/") -> list[dict[str, Any]]:
        self._ensure_active()
        response = await self._client._request(
            "GET",
            f"/sandboxes/{_url_id(self.sandbox_id)}/files/list",
            params={"path": remote_path},
        )
        return list(response["files"])

    async def delete(self, timeout: int | None = None) -> None:
        if self._deleted:
            return
        self.stop_heartbeat()
        await self._client.delete_sandbox(self.sandbox_id, timeout=timeout)
        self._deleted = True

    def delete_sync(self) -> None:
        if self._deleted:
            return
        self.stop_heartbeat()
        headers = {"X-API-Key": self._client.api_key} if self._client.api_key else {}
        try:
            response = requests.delete(
                f"{self._client.base_url}/sandboxes/{_url_id(self.sandbox_id)}",
                headers=headers,
                timeout=10,
            )
            if response.status_code not in (200, 404):
                response.raise_for_status()
        except Exception:
            return
        self._client._created_sandboxes.discard(self.sandbox_id)
        _forget(self._client.base_url, self.sandbox_id, self._client.api_key)
        self._deleted = True

    async def __aenter__(self) -> "AsyncSandboxInstance":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.delete()


class AsyncSandboxClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 1200,
        auto_cleanup: bool = True,
        api_key: str | None = None,
        prefix: str | None = None,
    ):
        self.base_url = (
            base_url or os.environ.get("SANDBOX_BASE_URL") or "http://localhost:8000"
        ).rstrip("/")
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("SANDBOX_API_KEY")
        self.prefix = prefix or os.environ.get("SANDBOX_PREFIX")
        self._session: aiohttp.ClientSession | None = None
        self._created_sandboxes: set[str] = set()
        self._auto_cleanup = auto_cleanup
        if auto_cleanup:
            _register_cleanup()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"X-API-Key": self.api_key} if self.api_key else {}
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(limit=200, limit_per_host=200),
            )
        return self._session

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        session = await self._get_session()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with session.request(method, f"{self.base_url}{path}", **kwargs) as response:
                    if response.status == 503 and attempt < 2:
                        await asyncio.sleep(2**attempt)
                        continue
                    response.raise_for_status()
                    return await response.json()
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    def _make_id(self, sandbox_id: str | None) -> str | None:
        if not self.prefix:
            return sandbox_id
        if sandbox_id and sandbox_id.startswith(f"{self.prefix}-"):
            return sandbox_id
        return f"{self.prefix}-{sandbox_id or uuid.uuid4().hex[:8]}"

    async def create_sandbox(
        self,
        image: str,
        block_network: bool = True,
        sandbox_id: str | None = None,
        cpu: str | None = None,
        memory: str | None = None,
        timeout: int | None = None,
        wait_ready: bool = True,
        poll_interval: float = 2.0,
        use_server_wait: bool = True,
    ) -> AsyncSandboxInstance:
        requested_id = self._make_id(sandbox_id)
        payload: dict[str, Any] = {"image": image, "block_network": block_network}
        for key, value in {
            "sandbox_id": requested_id,
            "cpu": cpu,
            "memory": memory,
            "timeout": timeout,
        }.items():
            if value is not None:
                payload[key] = value

        created_id: str | None = None
        try:
            created = await self._request("POST", "/sandboxes", json=payload)
            created_id = str(created["sandbox_id"])
            self._created_sandboxes.add(created_id)
            if self._auto_cleanup:
                _pending_cleanup.add((self.base_url, created_id, self.api_key))
            _save_marker(self.base_url, created_id)
            if not wait_ready:
                return AsyncSandboxInstance(self, created_id, created)

            ready_timeout = int(created.get("timeout") or timeout or 3600)
            if use_server_wait:
                try:
                    waited = await self._request(
                        "GET",
                        f"/sandboxes/{_url_id(created_id)}/wait",
                        params={"timeout": ready_timeout},
                        timeout=aiohttp.ClientTimeout(total=ready_timeout + 30),
                    )
                    SandboxClient._raise_failed(created_id, waited)
                    if SandboxClient._is_ready(waited):
                        return AsyncSandboxInstance(self, created_id, {**created, **waited})
                    raise RuntimeError(
                        f"Sandbox {created_id} wait returned status={waited.get('status')!r}"
                    )
                except aiohttp.ClientResponseError as exc:
                    if exc.status == 408:
                        raise TimeoutError(
                            f"Sandbox {created_id} did not become ready within {ready_timeout}s"
                        ) from exc
                    if exc.status not in (404, 405):
                        raise
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"Sandbox {created_id} did not become ready within {ready_timeout}s"
                    ) from exc

            deadline = asyncio.get_running_loop().time() + ready_timeout
            while asyncio.get_running_loop().time() < deadline:
                state = await self._request("GET", f"/sandboxes/{_url_id(created_id)}")
                SandboxClient._raise_failed(created_id, state)
                if SandboxClient._is_ready(state):
                    return AsyncSandboxInstance(self, created_id, {**created, **state})
                await asyncio.sleep(max(0.05, poll_interval))
            raise TimeoutError(
                f"Sandbox {created_id} did not become ready within {ready_timeout}s"
            )
        except BaseException:
            if created_id:
                try:
                    await self.delete_sandbox(created_id, timeout=min(self.timeout, 120))
                except Exception:
                    pass
            raise

    async def get_sandbox(self, sandbox_id: str) -> AsyncSandboxInstance:
        data = await self._request("GET", f"/sandboxes/{_url_id(sandbox_id)}")
        return AsyncSandboxInstance(self, str(data.get("sandbox_id", sandbox_id)), data)

    async def delete_sandbox(self, sandbox_id: str, timeout: int | None = None) -> None:
        try:
            await self._request(
                "DELETE",
                f"/sandboxes/{_url_id(sandbox_id)}",
                timeout=aiohttp.ClientTimeout(total=timeout or min(self.timeout, 120)),
            )
        except aiohttp.ClientResponseError as exc:
            if exc.status != 404:
                raise
        self._created_sandboxes.discard(sandbox_id)
        _forget(self.base_url, sandbox_id, self.api_key)

    async def cleanup_all(self) -> None:
        for sandbox_id in list(self._created_sandboxes):
            await self.delete_sandbox(sandbox_id)

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def close(self, cleanup: bool = True) -> None:
        if cleanup and self._auto_cleanup:
            await self.cleanup_all()
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "AsyncSandboxClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close(cleanup=True)


Sandbox = SandboxClient
AsyncSandbox = AsyncSandboxClient
