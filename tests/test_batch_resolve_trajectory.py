import importlib.util
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from minisweagent.run.benchmarks import gentest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "swe_harness" / "gentest_v2" / "batch_resolve.py"
NUM_GPUS = 0


def load_batch_resolve():
    spec = importlib.util.spec_from_file_location("batch_resolve", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEnvironment:
    def __init__(self):
        self.cleaned = False
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleaned = True
        self.cleanup_calls += 1


class FakeRebenchV2Environment(FakeEnvironment):
    def __init__(self, output="GENTEST_REBENCH_BASE_COMMIT=abc123\n", returncode=0):
        super().__init__()
        self.output = output
        self.returncode = returncode
        self.execute_calls = []

    def execute(self, action, cwd="/", timeout=None):
        self.execute_calls.append((action, cwd, timeout))
        return {"returncode": self.returncode, "output": self.output, "exception_info": ""}


class LocalShellEnvironment:
    def __init__(self, repository: Path):
        self.repository = repository

    def execute(self, action, timeout=None):
        command = str(action["command"]).replace(
            "cd /testbed",
            f"cd {shlex.quote(str(self.repository))}",
            1,
        )
        completed = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "output": completed.stdout,
            "exception_info": completed.stderr,
        }


class FakeAgent:
    def __init__(self, config, environments, fail_run=False):
        self.output_path = Path(config["output_path"])
        self.verify_runner = config["verify_runner"]
        self.environments = environments
        self.fail_run = fail_run
        self.n_calls = 2
        self.messages = [
            {"role": "assistant", "extra": {"actions": [{"tool": "bash", "command": "git diff"}]}},
            {"role": "tool", "content": "diff output"},
        ]

    def run(self, task=""):
        if self.fail_run:
            raise RuntimeError("model failed after tool output")
        self.verify_runner(
            "diff --git a/tests/test_bug.py b/tests/test_bug.py\n",
            "pytest tests/test_bug.py",
        )
        self.save(self.output_path)
        return {
            "exit_status": "Submitted",
            "submission": "diff --git a/tests/test_bug.py b/tests/test_bug.py\n",
            "resolve_gate": {"base_clean_fail": True, "test_command": "pytest tests/test_bug.py"},
        }

    def save(self, path, *extra_dicts):
        assert not any(environment.cleaned for environment in self.environments)
        payload = {
            "info": {"model_stats": {"api_calls": self.n_calls}},
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        for extra in extra_dicts:
            payload["info"].update(extra.get("info", {}))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return payload


def configure_fakes(monkeypatch, batch_resolve, environments, *, fail_run=False, environment_instances=None):
    monkeypatch.setattr(batch_resolve, "build_model", lambda *_args: object())

    def get_environment(_config, instance):
        if environment_instances is not None:
            environment_instances.append(instance)
        return environments.pop(0)

    monkeypatch.setattr(batch_resolve, "get_resolve_environment", get_environment)
    monkeypatch.setattr(batch_resolve, "get_agent_environment", lambda environment, _instance: environment)
    monkeypatch.setattr(batch_resolve, "GitHistoryGuardEnv", lambda environment: environment)
    monkeypatch.setattr(batch_resolve, "sanitize_git_history", lambda _environment: None)

    all_environments = list(environments)

    def get_agent(_model, environment, config, default_type=None):
        assert default_type == "testpatch_resolve"
        all_environments.insert(0, environment)
        return FakeAgent(config, all_environments, fail_run=fail_run)

    monkeypatch.setattr(batch_resolve, "get_agent", get_agent)


def make_args(output, *, shared_verify=False):
    return SimpleNamespace(
        output=str(output),
        step_limit=10,
        instance_timeout=600,
        install_timeout=30,
        no_gold_check=False,
        shared_verify=shared_verify,
    )


def test_resolve_environment_routes_scaleswe_through_native_adapter(monkeypatch):
    batch_resolve = load_batch_resolve()
    calls = []
    monkeypatch.setattr(
        batch_resolve,
        "get_gentest_environment",
        lambda config, instance: calls.append(("scaleswe", config, instance)) or "scale-env",
    )
    monkeypatch.setattr(
        batch_resolve,
        "get_sb_environment",
        lambda config, instance: calls.append(("swebench", config, instance)) or "swe-env",
    )

    scale_instance = {"dataset": batch_resolve.SCALESWE_DATASET, "instance_id": "scale-1"}
    assert batch_resolve.get_resolve_environment({}, scale_instance) == "scale-env"
    assert batch_resolve.get_resolve_environment({}, {"instance_id": "swe-1"}) == "swe-env"
    assert [call[0] for call in calls] == ["scaleswe", "swebench"]


def test_resolve_environment_links_rebench_v2_repo_to_testbed(monkeypatch):
    batch_resolve = load_batch_resolve()
    environment = FakeRebenchV2Environment()
    monkeypatch.setattr(batch_resolve, "get_sb_environment", lambda _config, _instance: environment)
    instance = {
        "dataset": batch_resolve.REBENCH_V2_DATASET,
        "instance_id": "wtforms__wtforms-614",
        "repo": "wtforms/wtforms",
        "base_commit": "abc123",
    }

    assert batch_resolve.get_resolve_environment({}, instance) is environment
    assert environment.execute_calls == [
        (
            {
                "command": (
                    "test -d /wtforms && "
                    "if [ -e /testbed ] && [ ! -L /testbed ]; then "
                    "echo 'ReBench V2 adapter refuses to replace existing /testbed' >&2; exit 73; fi; "
                    "ln -sfn /wtforms /testbed && commit=$(git -C /wtforms rev-parse HEAD) && "
                    "printf 'GENTEST_REBENCH_BASE_COMMIT=%s\\n' \"$commit\""
                )
            },
            "/wtforms",
            120,
        )
    ]
    assert environment.cleaned is False


def test_resolve_environment_cleans_rebench_v2_on_commit_mismatch(monkeypatch):
    batch_resolve = load_batch_resolve()
    environment = FakeRebenchV2Environment(output="GENTEST_REBENCH_BASE_COMMIT=wrong-commit\n")
    monkeypatch.setattr(batch_resolve, "get_sb_environment", lambda _config, _instance: environment)
    instance = {
        "dataset": batch_resolve.REBENCH_V2_DATASET,
        "repo": "wtforms/wtforms",
        "base_commit": "expected-commit",
    }

    with pytest.raises(RuntimeError, match="base commit mismatch"):
        batch_resolve.get_resolve_environment({}, instance)
    assert environment.cleaned is True


def test_agent_environment_skips_missing_testbed_conda_env_for_rebench_v2(monkeypatch):
    batch_resolve = load_batch_resolve()
    environment = FakeEnvironment()
    activations = []
    monkeypatch.setattr(
        gentest,
        "ActivatingEnv",
        lambda inner, prefix: activations.append((inner, prefix)) or "activated-environment",
    )
    monkeypatch.setattr(gentest, "activation_for_instance", lambda _instance: "activate-testbed")

    rebench_instance = {"dataset": batch_resolve.REBENCH_V2_DATASET}
    assert batch_resolve.get_agent_environment(environment, rebench_instance) is environment
    assert activations == []

    assert batch_resolve.get_agent_environment(environment, {"instance_id": "django__django-1"}) == (
        "activated-environment"
    )
    assert activations == [(environment, "activate-testbed")]


def test_resolve_environment_retries_transient_404(monkeypatch):
    batch_resolve = load_batch_resolve()
    attempts = []
    sleeps = []

    def get_environment(_config, _instance):
        attempts.append(1)
        if len(attempts) < 3:
            response = requests.Response()
            response.status_code = 404
            raise requests.exceptions.HTTPError(response=response)
        return "environment"

    monkeypatch.setattr(batch_resolve, "get_resolve_environment", get_environment)
    monkeypatch.setattr(batch_resolve.time, "sleep", sleeps.append)

    assert batch_resolve._get_resolve_environment_with_retry(
        {}, {"instance_id": "repo__issue-1"}, phase="generation"
    ) == "environment"
    assert len(attempts) == 3
    assert sleeps == [1, 2]


def test_shared_verify_resets_then_restores_agent_patch_and_removes_context(monkeypatch):
    environment = FakeEnvironment()
    events = []

    def write_file(_environment, path, text, timeout=None):
        events.append(("write", path, text, timeout))

    def shell(_environment, command, timeout=None):
        if command.startswith("test -s"):
            phase = "check_restore"
        elif "git reset --hard" in command:
            phase = "reset"
        elif "git apply" in command:
            phase = "restore_patch"
        elif command.startswith("rm -f"):
            phase = "cleanup_context"
        else:
            raise AssertionError(command)
        events.append((phase, command, timeout))
        return {"returncode": 0, "output": ""}

    def verify(_environment, _instance, patch, command, timeout, skip_install=False):
        events.append(("official_verify", patch, command, timeout, skip_install))
        return {"verdict": "fail", "clean_fail": True}

    monkeypatch.setattr(gentest, "write_file", write_file)
    monkeypatch.setattr(gentest, "shell", shell)
    monkeypatch.setattr(gentest, "run_test_patch_official", verify)

    result = gentest.run_shared_test_patch_official(
        environment,
        {"base_commit": "abc123"},
        "agent-owned patch",
        "pytest tests/test_bug.py",
        30,
        skip_install=False,
        restore_agent_workspace=True,
    )

    assert result["clean_fail"] is True
    assert [event[0] for event in events] == [
        "write",
        "check_restore",
        "reset",
        "official_verify",
        "reset",
        "restore_patch",
        "cleanup_context",
    ]
    cleanup_command = events[-1][1]
    assert "/tmp/gentest_shared_restore.diff" in cleanup_command
    assert "/tmp/test_patch.diff" in cleanup_command
    assert "/tmp/eval.sh" in cleanup_command


def test_shared_verify_restores_and_cleans_after_verifier_exception(monkeypatch):
    events = []

    monkeypatch.setattr(gentest, "write_file", lambda *_args, **_kwargs: events.append("write"))

    def shell(_environment, command, timeout=None):
        if command.startswith("test -s"):
            events.append("check_restore")
        elif "git reset --hard" in command:
            events.append("reset")
        elif "git apply" in command:
            events.append("restore_patch")
        elif command.startswith("rm -f"):
            events.append("cleanup_context")
        return {"returncode": 0, "output": ""}

    def fail_verify(*_args, **_kwargs):
        events.append("official_verify")
        raise RuntimeError("official verifier failed")

    monkeypatch.setattr(gentest, "shell", shell)
    monkeypatch.setattr(gentest, "run_test_patch_official", fail_verify)

    with pytest.raises(RuntimeError, match="official verifier failed"):
        gentest.run_shared_test_patch_official(
            FakeEnvironment(),
            {"base_commit": "abc123"},
            "agent-owned patch",
            "pytest tests/test_bug.py",
            30,
            skip_install=False,
            restore_agent_workspace=True,
        )

    assert events[-3:] == ["reset", "restore_patch", "cleanup_context"]


def test_resume_mode_rejects_mixed_or_legacy_shared_results(tmp_path):
    batch_resolve = load_batch_resolve()
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps({"instance_id": "old-row"}) + "\n")

    with pytest.raises(ValueError, match="use a new output directory"):
        batch_resolve._validate_resume_verify_mode(results, shared_verify=True)

    batch_resolve._validate_resume_verify_mode(results, shared_verify=False)
    results.write_text(json.dumps({"instance_id": "shared-row", "shared_verify": True}) + "\n")
    with pytest.raises(ValueError, match="use a new output directory"):
        batch_resolve._validate_resume_verify_mode(results, shared_verify=False)


def test_sanitize_git_history_prunes_hidden_fix_and_proves_postconditions(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            capture_output=True,
            text=True,
            check=check,
        )

    git("init", "-b", "main")
    git("config", "user.email", "release-test@example.invalid")
    git("config", "user.name", "Release Test")
    tracked = repository / "tracked.txt"
    tracked.write_text("base\n")
    git("add", "tracked.txt")
    git("commit", "-m", "base")
    base_commit = git("rev-parse", "HEAD").stdout.strip()
    tracked.write_text("hidden fix\n")
    git("commit", "-am", "hidden fix")
    hidden_fix = git("rev-parse", "HEAD").stdout.strip()
    git("reset", "--hard", base_commit)
    git("tag", "hidden-fix", hidden_fix)
    git("remote", "add", "origin", "https://example.invalid/repository.git")

    result = gentest.sanitize_git_history(LocalShellEnvironment(repository))

    assert "GENTEST_GIT_HISTORY_SANITIZED" in result["output"]
    assert git("remote").stdout.strip() == ""
    assert git("for-each-ref", "--format=%(refname)").stdout.strip() == "refs/heads/main"
    assert git("reflog", "show", "--all").stdout.strip() == ""
    assert git("cat-file", "-e", f"{hidden_fix}^{{commit}}", check=False).returncode != 0


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"returncode": 17, "output": "failed", "exception_info": "git gc failed"}, "returncode=17"),
        ({"returncode": 0, "output": "", "exception_info": ""}, "postcondition marker is missing"),
    ],
)
def test_sanitize_git_history_rejects_unproven_success(result, message):
    class ResultEnvironment:
        def execute(self, _action, timeout=None):
            assert timeout == 180
            return result

    with pytest.raises(RuntimeError, match=message):
        gentest.sanitize_git_history(ResultEnvironment())


def test_run_one_persists_messages_and_final_b2g_metadata(tmp_path, monkeypatch):
    batch_resolve = load_batch_resolve()
    generator_environment = FakeEnvironment()
    verifier_environment = FakeEnvironment()
    environments = [generator_environment, verifier_environment]
    configure_fakes(monkeypatch, batch_resolve, environments)
    verify_calls = []

    def verify(_environment, _instance, patch, command, _timeout, skip_install=False):
        verify_calls.append((patch, command, skip_install))
        return {"verdict": "pass", "passed": 1, "failed": 0, "errors": 0}

    monkeypatch.setattr(batch_resolve, "run_test_patch_official", verify)
    instance = {"instance_id": "repo__issue-1", "repo": "repo/name", "problem_statement": "bug", "patch": "gold"}

    record = batch_resolve.run_one(
        instance,
        make_args(tmp_path),
        {},
        {},
        {},
        "http://endpoint/v1",
        "repo__issue-1#rep2",
    )

    expected_path = tmp_path / "trajectories" / "repo__issue-1" / "repo__issue-1#rep2.traj.json"
    payload = json.loads(expected_path.read_text())
    assert record["trajectory_path"] == str(expected_path)
    assert payload["messages"][0]["extra"]["actions"][0]["tool"] == "bash"
    assert payload["messages"][1]["content"] == "diff output"
    assert payload["info"]["resolve_result"]["gold_pass"] is True
    assert payload["info"]["resolve_result"]["base_fail_gold_pass"] is True
    assert payload["info"]["resolve_result"]["isolated_verify"] is True
    assert len(verify_calls) == 2
    assert [call[2] for call in verify_calls] == [False, True]
    assert generator_environment.cleaned is True
    assert verifier_environment.cleaned is True


def test_run_one_shared_verify_uses_one_sandbox_and_hides_gold_from_messages(tmp_path, monkeypatch):
    batch_resolve = load_batch_resolve()
    environment = FakeEnvironment()
    environments = [environment]
    environment_instances = []
    configure_fakes(
        monkeypatch,
        batch_resolve,
        environments,
        environment_instances=environment_instances,
    )
    shared_verify_calls = []

    def shared_verify(_environment, _instance, patch, command, _timeout, **kwargs):
        shared_verify_calls.append((patch, command, kwargs))
        return {"verdict": "pass", "passed": 1, "failed": 0, "errors": 0}

    monkeypatch.setattr(batch_resolve, "run_shared_test_patch_official", shared_verify)
    instance = {"instance_id": "repo__issue-3", "repo": "repo/name", "problem_statement": "bug", "patch": "gold"}

    record = batch_resolve.run_one(
        instance,
        make_args(tmp_path, shared_verify=True),
        {},
        {},
        {},
        "http://endpoint/v1",
        "repo__issue-3",
    )

    payload = json.loads(Path(record["trajectory_path"]).read_text())
    assert environments == []
    assert len(environment_instances) == 1
    assert not {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"} & environment_instances[0].keys()
    assert record["shared_verify"] is True
    assert record["isolated_verify"] is False
    assert environment.cleanup_calls == 1
    assert len(shared_verify_calls) == 2
    assert [call[2]["restore_agent_workspace"] for call in shared_verify_calls] == [True, False]
    assert "gold" not in json.dumps(payload["messages"]).lower()
    assert payload["info"]["resolve_result"]["gold_pass"] is True


def test_run_one_saves_exception_trajectory_before_cleanup(tmp_path, monkeypatch):
    batch_resolve = load_batch_resolve()
    generator_environment = FakeEnvironment()
    verifier_environment = FakeEnvironment()
    environments = [generator_environment, verifier_environment]
    configure_fakes(monkeypatch, batch_resolve, environments, fail_run=True)
    monkeypatch.setattr(batch_resolve, "run_test_patch_official", lambda *_args, **_kwargs: {})
    instance = {"instance_id": "repo__issue-2", "repo": "repo/name", "problem_statement": "bug"}

    record = batch_resolve.run_one(instance, make_args(tmp_path), {}, {}, {}, "", "repo__issue-2")

    payload = json.loads(Path(record["trajectory_path"]).read_text())
    assert record["error"].startswith("RuntimeError: model failed")
    assert payload["messages"][1]["content"] == "diff output"
    assert payload["info"]["resolve_result"]["error"] == record["error"]
    assert generator_environment.cleaned is True
    assert verifier_environment.cleaned is True


def test_run_one_sanitize_failure_never_builds_model_and_cleans_environments(tmp_path, monkeypatch):
    batch_resolve = load_batch_resolve()
    generator_environment = FakeEnvironment()
    verifier_environment = FakeEnvironment()
    environments = iter((generator_environment, verifier_environment))
    for name in batch_resolve._STRICT_HISTORY_SETTINGS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(
        batch_resolve,
        "_get_resolve_environment_with_retry",
        lambda *_args, **_kwargs: next(environments),
    )
    monkeypatch.setattr(
        batch_resolve,
        "sanitize_git_history",
        lambda _environment: (_ for _ in ()).throw(RuntimeError("git prune failed")),
    )
    monkeypatch.setattr(
        batch_resolve,
        "build_model",
        lambda *_args, **_kwargs: pytest.fail("model must not be built after sanitization failure"),
    )
    monkeypatch.setattr(
        batch_resolve,
        "get_agent",
        lambda *_args, **_kwargs: pytest.fail("agent must not run after sanitization failure"),
    )

    record = batch_resolve.run_one(
        {"instance_id": "repo__issue-4", "problem_statement": "bug"},
        make_args(tmp_path),
        {},
        {},
        {},
        "http://endpoint/v1",
    )

    assert record["exit_status"] == "sandbox_infra_error"
    assert record["infra_failure"] is True
    assert record["error_category"] == "git_history_sanitization_failed"
    assert record["git_history_sanitized"] is False
    assert record["error"].startswith("RuntimeError: strict behavior-contract git-history sanitization failed")
    assert generator_environment.cleaned is True
    assert verifier_environment.cleaned is True
    assert not Path(record["trajectory_path"]).exists()


@pytest.mark.parametrize("setting", ["GENTEST_SANITIZE_GIT_HISTORY", "GENTEST_GIT_HISTORY_GUARD"])
def test_batch_behavior_contract_rejects_disabled_history_isolation(setting, monkeypatch):
    batch_resolve = load_batch_resolve()
    for name in batch_resolve._STRICT_HISTORY_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(setting, "0")

    with pytest.raises(ValueError, match=rf"strict behavior-contract batch generation requires {setting}=1"):
        batch_resolve._require_strict_history_isolation()


@pytest.mark.parametrize(
    ("instance_id", "run_key", "field"),
    [
        ("../../outside", None, "instance_id"),
        ("repo__issue-5", "../outside", "run_key"),
        ("/absolute", None, "instance_id"),
    ],
)
def test_run_one_rejects_unsafe_trajectory_components(instance_id, run_key, field, tmp_path):
    batch_resolve = load_batch_resolve()

    with pytest.raises(ValueError, match=rf"unsafe {field} for trajectory path"):
        batch_resolve.run_one(
            {"instance_id": instance_id},
            make_args(tmp_path),
            {},
            {},
            {},
            "",
            run_key,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
