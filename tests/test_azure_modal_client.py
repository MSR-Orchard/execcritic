import asyncio
import sys
from pathlib import Path


CLIENT_ROOT = Path(__file__).parents[1] / "swe_harness" / "external" / "azure-modal"
sys.path.insert(0, str(CLIENT_ROOT))

from client import AsyncSandboxClient, JobResult, SandboxClient  # noqa: E402


def _created(status="pending"):
    return {
        "sandbox_id": "unit-box",
        "namespace": "unit",
        "image": "python:3.11",
        "block_network": True,
        "cpu": "2",
        "memory": "8Gi",
        "timeout": 30,
        "status": status,
    }


def _job(status="succeeded"):
    return {
        "job_id": "job-1",
        "sandbox_id": "unit-box",
        "command": "echo ok",
        "status": status,
        "stdout": "ok\n",
        "stderr": "",
        "exit_code": 0,
        "created_at": 1.0,
    }


class FakeSyncClient(SandboxClient):
    def __init__(self):
        super().__init__("https://sandbox.invalid", auto_cleanup=False, api_key="test-key")
        self.calls = []

    def _request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if method == "POST" and path == "/sandboxes":
            return _created()
        if path.endswith("/wait"):
            return {**_created("ready"), "ready": True}
        if path.endswith("/exec"):
            return _job()
        if method == "DELETE":
            return {"deleted": True}
        raise AssertionError((method, path, kwargs))


class FakeAsyncClient(AsyncSandboxClient):
    def __init__(self):
        super().__init__("https://sandbox.invalid", auto_cleanup=False, api_key="test-key")
        self.calls = []

    async def _request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if method == "POST" and path == "/sandboxes":
            return _created()
        if path.endswith("/wait"):
            return {**_created("ready"), "ready": True}
        if path.endswith("/exec"):
            return _job()
        if method == "DELETE":
            return {"deleted": True}
        raise AssertionError((method, path, kwargs))


def test_job_result_properties():
    result = JobResult(_job())
    assert result.succeeded
    assert not result.failed
    assert result.is_complete
    assert result.stdout == "ok\n"


def test_sync_create_exec_delete_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_SANDBOX_MANIFEST_DIR", str(tmp_path))
    client = FakeSyncClient()
    sandbox = client.create_sandbox("python:3.11", cpu="2", memory="8Gi")
    assert sandbox.sandbox_id == "unit-box"
    assert (tmp_path / "unit-box").is_file()

    result = sandbox.exec("echo ok", timeout=5, cwd="/workspace", login_shell=True)
    assert result.succeeded
    exec_payload = next(call[2]["json"] for call in client.calls if call[1].endswith("/exec"))
    assert exec_payload["wait"] is True
    assert exec_payload["login_shell"] is True

    sandbox.delete()
    assert not (tmp_path / "unit-box").exists()


def test_async_create_exec_delete_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_SANDBOX_MANIFEST_DIR", str(tmp_path))

    async def exercise():
        client = FakeAsyncClient()
        sandbox = await client.create_sandbox("python:3.11", cpu="2", memory="8Gi")
        result = await sandbox.exec("echo ok", timeout=5)
        assert result.succeeded
        await sandbox.delete()
        await client.close(cleanup=False)

    asyncio.run(exercise())
    assert not (tmp_path / "unit-box").exists()
