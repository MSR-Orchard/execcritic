from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slime.utils.types import Sample

from . import repair_reward


PATCH = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1 +1 @@
-old
+new
"""


def _sample(
    *,
    f2p_passed=False,
    had_submission=True,
    forbidden=None,
    final_output=PATCH,
    repair_rounds_started=1,
):
    return Sample(
        metadata={
            "instance_id": "repo__case-1",
            "repo": "owner/repo",
            "base_commit": "base",
            "FAIL_TO_PASS": ["tests/test_core.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_core.py::test_existing"],
            "final_output": final_output,
            "f2p_repair": {
                "had_submission": had_submission,
                "f2p_passed": f2p_passed,
                "repair_rounds_started": repair_rounds_started,
                "final_forbidden_test_paths": forbidden or [],
            },
        }
    )


@pytest.mark.parametrize(
    ("resolved", "f2p_passed", "expected", "reason"),
    [
        (True, False, 1.0, "official_resolved"),
        (False, True, 0.2, "repair_f2p_passed"),
        (False, False, 0.1, "official_unresolved"),
    ],
)
def test_repair_reward_outcomes(monkeypatch, resolved, f2p_passed, expected, reason):
    run_tests = AsyncMock(
        return_value={
            "resolved": resolved,
            "resolution": "RESOLVED_FULL" if resolved else "RESOLVED_NO",
            "f2p_passed": 1 if resolved else 0,
            "f2p_total": 1,
            "p2p_passed": 1 if resolved else 0,
            "p2p_total": 1,
            "error": "",
        }
    )
    monkeypatch.setattr(repair_reward, "run_tests_in_docker", run_tests)
    sample = _sample(f2p_passed=f2p_passed)

    reward = asyncio.run(repair_reward.reward_func(SimpleNamespace(), sample))

    assert reward == {
        "raw_reward": expected,
        "reward": expected,
        "submit_reward": 1.0,
        "repair_reward": float(resolved or f2p_passed),
        "pass_reward": float(resolved),
        "direct_reward": 0.0,
    }
    assert sample.metadata["repair_reward"]["reason"] == reason
    assert sample.metadata["repair_reward"]["submit_reward"] == 1.0
    assert sample.metadata["repair_reward"]["repair_reward"] == float(resolved or f2p_passed)
    assert sample.metadata["repair_reward"]["pass_reward"] == float(resolved)
    assert sample.metadata["repair_reward"]["direct_reward"] == 0.0
    run_tests.assert_awaited_once()


def test_repair_reward_reuses_rollout_official_result_without_third_environment(monkeypatch):
    run_tests = AsyncMock()
    monkeypatch.setattr(repair_reward, "run_tests_in_docker", run_tests)
    sample = _sample(f2p_passed=True)
    sample.metadata["f2p_repair"]["official_verify"] = {
        "resolved": True,
        "resolution": "RESOLVED_FULL",
        "f2p_passed": 1,
        "f2p_total": 1,
        "p2p_passed": 1,
        "p2p_total": 1,
        "reward_eval_seconds": 12.5,
        "error": "",
    }

    reward = asyncio.run(repair_reward.reward_func(SimpleNamespace(), sample))

    assert reward["raw_reward"] == 1.0
    assert sample.metadata["repair_reward"]["official_eval_seconds"] == 12.5
    run_tests.assert_not_awaited()


def test_round0_official_resolved_gets_direct_reward(monkeypatch):
    run_tests = AsyncMock()
    monkeypatch.setattr(repair_reward, "run_tests_in_docker", run_tests)
    monkeypatch.setenv("SWE_REPAIR_DIRECT_REWARD", "1.5")
    sample = _sample(f2p_passed=True, repair_rounds_started=0)
    sample.metadata["f2p_repair"]["official_verify"] = {
        "resolved": True,
        "resolution": "RESOLVED_FULL",
        "f2p_passed": 1,
        "f2p_total": 1,
        "p2p_passed": 1,
        "p2p_total": 1,
        "error": "",
    }

    reward = asyncio.run(repair_reward.reward_func(SimpleNamespace(), sample))

    assert reward == {
        "raw_reward": 1.5,
        "reward": 1.5,
        "submit_reward": 1.0,
        "repair_reward": 1.0,
        "pass_reward": 1.0,
        "direct_reward": 1.0,
    }
    assert sample.metadata["repair_reward"]["reason"] == "official_resolved_direct"
    run_tests.assert_not_awaited()


@pytest.mark.parametrize(
    ("sample", "reason"),
    [
        (_sample(had_submission=False), "no_valid_submission"),
        (_sample(forbidden=["tests/test_core.py"]), "forbidden_test_modification"),
        (_sample(final_output=""), "invalid_or_empty_patch"),
    ],
)
def test_invalid_repair_outputs_are_zero_without_official_eval(monkeypatch, sample, reason):
    run_tests = AsyncMock()
    monkeypatch.setattr(repair_reward, "run_tests_in_docker", run_tests)

    reward = asyncio.run(repair_reward.reward_func(SimpleNamespace(), sample))

    assert reward == {
        "raw_reward": 0.0,
        "reward": 0.0,
        "submit_reward": 0.0,
        "repair_reward": 0.0,
        "pass_reward": 0.0,
        "direct_reward": 0.0,
    }
    assert sample.metadata["repair_reward"]["reason"] == reason
    assert sample.metadata["repair_reward"]["submit_reward"] == 0.0
    assert sample.metadata["repair_reward"]["repair_reward"] == 0.0
    assert sample.metadata["repair_reward"]["pass_reward"] == 0.0
    run_tests.assert_not_awaited()


@pytest.mark.parametrize(("f2p_passed", "expected"), [(False, 0.1), (True, 0.2)])
def test_official_infrastructure_error_keeps_repair_tier_reward(monkeypatch, f2p_passed, expected):
    monkeypatch.setattr(
        repair_reward,
        "run_tests_in_docker",
        AsyncMock(return_value={"resolved": False, "error": "sandbox unavailable"}),
    )
    sample = _sample(f2p_passed=f2p_passed)

    reward = asyncio.run(repair_reward.reward_func(SimpleNamespace(), sample))

    assert reward == {
        "raw_reward": expected,
        "reward": expected,
        "submit_reward": 1.0,
        "repair_reward": float(f2p_passed),
        "pass_reward": 0.0,
        "direct_reward": 0.0,
    }
    assert sample.metadata["repair_reward"]["reason"] == "official_eval_error"
    assert sample.metadata["repair_reward"]["submit_reward"] == 1.0
    assert sample.metadata["repair_reward"]["repair_reward"] == float(f2p_passed)
    assert sample.metadata["repair_reward"]["pass_reward"] == 0.0


@pytest.mark.parametrize(("f2p_passed", "expected"), [(False, 0.1), (True, 0.2)])
def test_official_timeout_keeps_repair_tier_reward(monkeypatch, f2p_passed, expected):
    async def delayed_official_eval(**kwargs):
        del kwargs
        await asyncio.sleep(1)

    run_tests = AsyncMock(side_effect=delayed_official_eval)
    monkeypatch.setattr(repair_reward, "run_tests_in_docker", run_tests)
    monkeypatch.setattr(repair_reward, "SWE_TIMEOUT_REWARD_TOTAL", 0.001)
    sample = _sample(f2p_passed=f2p_passed)

    reward = asyncio.run(repair_reward.reward_func(SimpleNamespace(), sample))

    assert reward == {
        "raw_reward": expected,
        "reward": expected,
        "submit_reward": 1.0,
        "repair_reward": float(f2p_passed),
        "pass_reward": 0.0,
        "direct_reward": 0.0,
    }
    assert sample.metadata["repair_reward"]["reason"] == "official_eval_timeout"
    assert sample.metadata["repair_reward"]["official_error"] == (
        "official evaluation timed out after 0.001s"
    )
    assert sample.metadata["repair_reward"]["official_eval_seconds"] > 0
    run_tests.assert_awaited_once()


def test_center_rewards_by_actual_group_index():
    args = SimpleNamespace(n_samples_per_prompt=4, reward_key="raw_reward")
    samples = [
        Sample(reward={"raw_reward": 1.0, "reward": 1.0, "submit_reward": 1.0, "repair_reward": 1.0, "pass_reward": 1.0}, group_index=10),
        Sample(reward={"raw_reward": 0.1, "reward": 0.1, "submit_reward": 1.0, "repair_reward": 0.0, "pass_reward": 0.0}, group_index=20),
        Sample(reward={"raw_reward": 0.2, "reward": 0.2, "submit_reward": 1.0, "repair_reward": 1.0, "pass_reward": 0.0}, group_index=10),
        Sample(reward={"raw_reward": 1.0, "reward": 1.0, "submit_reward": 1.0, "repair_reward": 1.0, "pass_reward": 1.0}, group_index=20),
    ]

    raw, centered = repair_reward.center_rewards_by_group(args, samples)

    assert raw == [1.0, 0.1, 0.2, 1.0]
    assert centered == pytest.approx([0.4, -0.45, -0.4, 0.45])
    assert sum(centered[position] for position in (0, 2)) == pytest.approx(0.0)
    assert sum(centered[position] for position in (1, 3)) == pytest.approx(0.0)
    assert [sample.reward["reward"] for sample in samples] == pytest.approx(centered)
    assert [sample.reward["submit_reward"] for sample in samples] == [1.0] * 4
    assert [sample.reward["repair_reward"] for sample in samples] == [1.0, 0.0, 1.0, 1.0]
    assert [sample.reward["pass_reward"] for sample in samples] == [1.0, 0.0, 0.0, 1.0]


def test_center_rewards_assigns_zero_to_aborted_rollout():
    args = SimpleNamespace(n_samples_per_prompt=4, reward_key="raw_reward")
    sample = Sample(index=7, group_index=2, reward=None, status=Sample.Status.ABORTED)

    raw, centered = repair_reward.center_rewards_by_group(args, [sample])

    assert raw == [0.0]
    assert centered == [0.0]
    assert sample.reward == {
        "raw_reward": 0.0,
        "reward": 0.0,
        "submit_reward": 0.0,
        "repair_reward": 0.0,
        "pass_reward": 0.0,
        "direct_reward": 0.0,
    }
    assert sample.metadata["repair_reward"]["reason"] == "rollout_aborted"


def test_center_rewards_excludes_aborted_rollout_from_group_baseline():
    args = SimpleNamespace(n_samples_per_prompt=4, reward_key="raw_reward")
    samples = [
        Sample(reward={"raw_reward": 1.0, "reward": 1.0}, group_index=2),
        Sample(reward={"raw_reward": 0.2, "reward": 0.2}, group_index=2),
        Sample(reward={"raw_reward": 0.0, "reward": 0.0}, group_index=2),
        Sample(reward=None, group_index=2, status=Sample.Status.ABORTED),
    ]

    raw, centered = repair_reward.center_rewards_by_group(args, samples)

    assert raw == [1.0, 0.2, 0.0, 0.0]
    assert centered == pytest.approx([0.6, -0.2, -0.4, 0.0])
    assert sum(centered[:3]) == pytest.approx(0.0)
    assert samples[-1].reward["reward"] == 0.0
    assert samples[-1].metadata["repair_reward"]["reason"] == "rollout_aborted"


def test_center_rewards_rejects_missing_reward_for_completed_rollout():
    args = SimpleNamespace(n_samples_per_prompt=4, reward_key="raw_reward")
    sample = Sample(index=8, group_index=2, reward=None, status=Sample.Status.COMPLETED)

    with pytest.raises(ValueError, match="Missing reward for non-aborted repair sample"):
        repair_reward.center_rewards_by_group(args, [sample])


def test_center_rewards_requires_raw_reward_key():
    with pytest.raises(ValueError, match="--reward-key raw_reward"):
        repair_reward.center_rewards_by_group(
            SimpleNamespace(n_samples_per_prompt=4, reward_key=None),
            [Sample(reward={"raw_reward": 1.0})],
        )
