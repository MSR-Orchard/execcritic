import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.mini_swe import resolve_native_rollout, swe_reward
from minisweagent.agents.extra.bash_testpatch_resolve import BashTestPatchResolveAgent as BashResolveAgent
from minisweagent.agents.extra.testpatch_resolve import TestPatchResolveAgent as StructuredResolveAgent
from minisweagent.run.benchmarks import gentest
from slime.utils.types import Sample


REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_GPUS = 0


@pytest.fixture(autouse=True)
def clear_overlay_cache(monkeypatch):
    monkeypatch.setattr(resolve_native_rollout, "_RESOLVE_OVERLAY_CACHE", None)


def test_v11_overlay_selects_bash_native_protocol(monkeypatch):
    overlay = REPO_ROOT / "swe_harness/gentest_v2/versions/v11_resolve_testpatch.yaml"
    monkeypatch.setenv("RESOLVE_OVERLAY", str(overlay))
    base = {
        "agent": {"step_limit": 7, "system_template": "base"},
        "model": {
            "extra_tools": ["grounding_plan"],
            "format_error_template": "base format error",
            "observation_template": "keep transport setting",
        },
    }

    agent_class, agent_cfg, model_cfg, agent_class_name, protocol, overlay_path = (
        resolve_native_rollout._resolve_runtime_config(base)
    )

    assert agent_class is BashResolveAgent
    assert agent_class_name == "bash_testpatch_resolve"
    assert protocol == "v11_native_bash"
    assert overlay_path == overlay.resolve()
    assert "agent_class" not in agent_cfg
    assert "/tmp/test_command" in agent_cfg["instance_template"]
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in agent_cfg["instance_template"]
    assert model_cfg["extra_tools"] == []
    assert "standard bash tool call" in model_cfg["format_error_template"]
    assert model_cfg["observation_template"] == "keep transport setting"


def test_behavior_contract_overlay_selects_behavior_contract_protocol(monkeypatch):
    overlay = REPO_ROOT / "swe_harness/gentest_v2/versions/behavior_contract.yaml"
    monkeypatch.setenv("RESOLVE_OVERLAY", str(overlay))

    agent_class, agent_cfg, model_cfg, agent_class_name, protocol, overlay_path = (
        resolve_native_rollout._resolve_runtime_config({})
    )

    assert agent_class is BashResolveAgent
    assert agent_class_name == "bash_testpatch_resolve"
    assert protocol == "behavior_contract"
    assert overlay_path == overlay.resolve()
    assert agent_cfg["test_contract_path"] == "/tmp/test_contract.json"
    assert agent_cfg["require_test_contract"] is True
    assert agent_cfg["require_exact_test_selector"] is True
    assert model_cfg["extra_tools"] == []


def test_default_overlay_is_behavior_contract(monkeypatch):
    monkeypatch.delenv("RESOLVE_OVERLAY", raising=False)
    monkeypatch.chdir(REPO_ROOT)

    _, agent_cfg, model_cfg, agent_class_name, protocol, overlay_path = (
        resolve_native_rollout._resolve_runtime_config({})
    )

    assert overlay_path == (REPO_ROOT / "swe_harness/gentest_v2/versions/behavior_contract.yaml").resolve()
    assert agent_class_name == "bash_testpatch_resolve"
    assert protocol == "behavior_contract"
    assert agent_cfg["require_test_contract"] is True
    assert agent_cfg["require_exact_test_selector"] is True
    assert model_cfg["extra_tools"] == []


def test_release_training_launchers_are_portable_and_keep_secrets_off_argv():
    gentest = (REPO_ROOT / "examples/mini_swe/scripts/train_resolve_scaleswe.sh").read_text()
    repair = (REPO_ROOT / "examples/mini_swe/scripts/train_swe_qwen3.5_35B_A3B_repair.sh").read_text()

    for launcher in (gentest, repair):
        assert "/data/users/" not in launcher
        assert "/root/" not in launcher
        assert "runtime-env-json" not in launcher
        assert "--wandb-key" not in launcher
        assert "pkill" not in launcher
        assert "--runtime-env=" in launcher
        assert ".ray-runtime-env." in launcher
        assert 'chmod 600 "$RUNTIME_ENV_FILE"' in launcher
        for required in ("HF_CHECKPOINT", "REF_LOAD", "PROMPT_DATA", "RUN_DIR", "SANDBOX_BASE_URL", "MEGATRON_ROOT"):
            assert required in launcher

    assert "behavior_contract.yaml" in gentest
    assert 'RESOLVE_SHARED_VERIFY="${RESOLVE_SHARED_VERIFY:-0}"' in gentest
    assert 'SWE_TIMEOUT_REWARD_EXECUTE="${SWE_TIMEOUT_REWARD_EXECUTE:-90}"' in gentest
    assert 'SWE_TIMEOUT_REWARD_TOTAL="${SWE_TIMEOUT_REWARD_TOTAL:-900}"' in gentest
    assert "--label-key patch" not in repair


def test_v10_overlay_remains_compatible(monkeypatch):
    overlay = REPO_ROOT / "examples/mini_swe/configs/resolve_testpatch_qwen3.5.yaml"
    monkeypatch.setenv("RESOLVE_OVERLAY", str(overlay))

    agent_class, agent_cfg, model_cfg, agent_class_name, protocol, _ = resolve_native_rollout._resolve_runtime_config(
        {}
    )

    assert agent_class is StructuredResolveAgent
    assert agent_class_name == "testpatch_resolve"
    assert protocol == "v10_structured_tools"
    assert agent_cfg["step_limit"] == 40
    assert model_cfg["extra_tools"] == resolve_native_rollout.RESOLVE_EXTRA_TOOLS
    assert "grounding_plan" in agent_cfg["instance_template"]


def test_resolve_label_preserves_sandbox_infra_error():
    assert resolve_native_rollout._resolve_label("sandbox_infra_error", None) == "sandbox_infra_error"


def test_resolve_generate_retries_and_excludes_exhausted_infra_sample(monkeypatch):
    attempts = 0

    async def failed_generate_once(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise resolve_native_rollout.EnvironmentUnavailable(
            "shared verify transaction failed: post-verify reset failed:; " "verifier context cleanup failed:"
        )

    def fill_placeholder(_args, sample, _instance):
        sample.tokens = [1, 2]
        sample.response_length = 1
        sample.loss_mask = [1]

    monkeypatch.setenv("GENTEST_MAX_ENV_RETRIES", "2")
    monkeypatch.setenv("GENTEST_ENV_RETRY_WAIT", "0")
    monkeypatch.setattr(resolve_native_rollout, "configure_blocking_executor", lambda _workers: None)
    monkeypatch.setattr(resolve_native_rollout, "_generate_once", failed_generate_once)
    monkeypatch.setattr(
        resolve_native_rollout,
        "_instance_from_sample",
        lambda _sample: {"instance_id": "repo__issue-1"},
    )
    monkeypatch.setattr(resolve_native_rollout, "_fill_placeholder_trajectory", fill_placeholder)

    sample = asyncio.run(
        resolve_native_rollout.generate(
            SimpleNamespace(reward_key="raw_reward"),
            Sample(index=0, metadata={}),
            {},
        )
    )

    assert attempts == 2
    assert sample.status == Sample.Status.ABORTED
    assert sample.remove_sample is True
    assert sample.loss_mask == [0]
    assert sample.reward == {"raw_reward": 0.0}
    assert sample.metadata["sandbox_infra_error"] is True
    assert sample.metadata["gentest_record"]["retryable_infrastructure_error"] is True


class FakeEnvironment:
    def __init__(self):
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


def test_shared_verify_creates_and_cleans_one_environment(monkeypatch):
    environment = FakeEnvironment()
    create_calls = []

    async def create_environment(config, instance):
        create_calls.append((config, instance))
        return environment

    async def unexpected_isolated_create(*_args, **_kwargs):
        raise AssertionError("isolated verifier should not be created")

    async def run_blocking(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(resolve_native_rollout, "_create_environment", create_environment)
    monkeypatch.setattr(resolve_native_rollout, "_create_isolated_environments", unexpected_isolated_create)
    monkeypatch.setattr(resolve_native_rollout, "_run_blocking", run_blocking)
    timings = {}

    workspace, verifier = asyncio.run(
        resolve_native_rollout._create_resolve_environments(
            {"environment": {"cpu": "2"}},
            {"environment": {"cpu": "4"}},
            {"instance_id": "repo__issue-1"},
            timings,
            shared_verify=True,
        )
    )
    asyncio.run(
        resolve_native_rollout._cleanup_resolve_environments(
            workspace,
            verifier,
            timings,
            instance_id="repo__issue-1",
        )
    )

    assert workspace is environment
    assert verifier is environment
    assert len(create_calls) == 1
    assert timings["verify_environment_create_sec"] == 0.0
    assert environment.cleanup_calls == 1
    assert "verifier_environment_cleanup_sec" not in timings


def test_isolated_verify_keeps_two_environment_lifecycles(monkeypatch):
    workspace = FakeEnvironment()
    verifier = FakeEnvironment()
    isolated_calls = []

    async def unexpected_shared_create(*_args, **_kwargs):
        raise AssertionError("shared environment creation should not run")

    async def create_isolated(workspace_config, verifier_config, instance, timings):
        isolated_calls.append((workspace_config, verifier_config, instance, timings))
        return workspace, verifier

    async def run_blocking(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(resolve_native_rollout, "_create_environment", unexpected_shared_create)
    monkeypatch.setattr(resolve_native_rollout, "_create_isolated_environments", create_isolated)
    monkeypatch.setattr(resolve_native_rollout, "_run_blocking", run_blocking)
    timings = {}

    created_workspace, created_verifier = asyncio.run(
        resolve_native_rollout._create_resolve_environments(
            {"environment": {"cpu": "2"}},
            {"environment": {"cpu": "4"}},
            {"instance_id": "repo__issue-2"},
            timings,
            shared_verify=False,
        )
    )
    asyncio.run(
        resolve_native_rollout._cleanup_resolve_environments(
            created_workspace,
            created_verifier,
            timings,
            instance_id="repo__issue-2",
        )
    )

    assert len(isolated_calls) == 1
    assert workspace.cleanup_calls == 1
    assert verifier.cleanup_calls == 1
    assert "workspace_environment_cleanup_sec" in timings
    assert "verifier_environment_cleanup_sec" in timings


def test_cleanup_finishes_both_environments_when_cancelled(monkeypatch):
    workspace = FakeEnvironment()
    verifier = FakeEnvironment()
    first_cleanup_started = asyncio.Event()
    release_first_cleanup = asyncio.Event()
    cleanup_calls = []

    async def run_blocking(function, *_args, **_kwargs):
        cleanup_calls.append(function)
        if len(cleanup_calls) == 1:
            first_cleanup_started.set()
            await release_first_cleanup.wait()
        function()

    async def exercise():
        task = asyncio.create_task(
            resolve_native_rollout._cleanup_resolve_environments(
                workspace,
                verifier,
                {},
                instance_id="repo__issue-cancelled",
            )
        )
        await first_cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_first_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    monkeypatch.setattr(resolve_native_rollout, "_run_blocking", run_blocking)

    asyncio.run(exercise())

    assert workspace.cleanup_calls == 1
    assert verifier.cleanup_calls == 1


def test_shared_verify_reward_uses_transaction_for_gold_and_candidates(monkeypatch):
    calls = []

    def shared_verify(environment, instance, patch, command, timeout, **kwargs):
        calls.append((environment, instance, patch, command, timeout, kwargs))
        return {"verdict": "pass" if patch.startswith("gold") else "fail"}

    def unexpected_isolated_verify(*_args, **_kwargs):
        raise AssertionError("shared reward should not call the isolated verifier directly")

    monkeypatch.setattr(gentest, "run_shared_test_patch_official", shared_verify)
    monkeypatch.setattr(gentest, "run_test_patch_official", unexpected_isolated_verify)
    environment = object()
    sample = Sample(
        metadata={
            "shared_verify": True,
            "patch_classification_verifier_warmed": True,
            "patch_classification_include_gold": True,
        }
    )

    swe_reward._compute_patch_classification_reward_testpatch(
        SimpleNamespace(swe_patch_classification_reward_metric="accuracy"),
        sample,
        existing_env=environment,
        details={},
        instance={"instance_id": "repo__issue-3", "patch": "gold source patch"},
        instance_id="repo__issue-3",
        test_patch="generated test patch",
        test_command="pytest tests/test_bug.py",
        candidates=[
            {
                "instance_id": "repo__issue-3",
                "patch_sha256": "candidate-sha",
                "patch_text": "candidate source patch",
                "oracle_resolved": False,
            }
        ],
        rollout_validation={"base_clean_fail": True},
        reward_start_time=time.time(),
    )

    assert len(calls) == 2
    assert all(call[0] is environment for call in calls)
    assert all(call[5]["restore_agent_workspace"] is False for call in calls)
    assert all(call[5]["skip_install"] is True for call in calls)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
