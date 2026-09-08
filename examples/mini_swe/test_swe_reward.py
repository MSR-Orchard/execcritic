"""
Unit tests for SWE-bench reward computation.

Tests cover:
- Patch extraction and validation
- Reward computation functions
- Docker test execution interface
"""

import asyncio
import importlib
import json
import sys
import threading
import time
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slime.utils.types import Sample

from examples.mini_swe.swe_reward import (
    HAS_SWEBENCH_HARNESS,
    START_TEST_OUTPUT,
    END_TEST_OUTPUT,
    # Utility functions
    extract_patch_from_output,
    extract_generated_test_code_from_sample,
    is_valid_patch,
    load_patch_classification_candidates,
    patch_classification_metrics,
    _base_gold_reward_from_validation,
    # Docker test execution
    run_tests_in_docker,
    run_official_swerebench_in_environment,
    _run_tests_with_swebench_harness,
    _map_official_swerebench_result,
    _is_unlabeled_swerebench_instance,
    build_prebuilt_test_command,
    grade_prebuilt_pytest_output,
    # Reward functions
    combine_rewards,
    # Process-shaping helpers (mode: process_shaped)
    _localization_score,
    _read_before_edit_bonus,
    _extra_files_penalty,
    _detect_flip,
    _error_rate,
    _no_progress_penalty,
    _integrity_gate,
    _is_test_path,
    combine_process_rewards,
)


def test_official_swerebench_eval_script_accepts_focused_command_override():
    from examples.mini_swe.swe_reward import _ensure_swe_harness_on_path

    _ensure_swe_harness_on_path()
    from minisweagent.run.benchmarks.swerebench_verify_azure_modal import build_eval_script

    script = build_eval_script(
        {
            "instance_id": "owner__repo-1",
            "base_commit": "abc123",
            "test_patch": (
                "diff --git a/tests/test_fix.py b/tests/test_fix.py\n"
                "--- a/tests/test_fix.py\n"
                "+++ b/tests/test_fix.py\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            ),
            "install_config": {
                "packages": "requirements.txt",
                "python": "3.10",
                "eval_commands": ["export PROJECT_MODE=test"],
                "install": "pip install -e .",
                "test_cmd": "pytest -q tests/test_fix.py tests/test_old.py",
            },
        },
        test_command_override="pytest -q tests/test_fix.py::test_selected",
    )

    assert "export PROJECT_MODE=test" in script
    assert "pip install -e ." in script
    assert "pytest -q tests/test_fix.py::test_selected" in script
    assert "pytest -q tests/test_fix.py tests/test_old.py" not in script


def test_prepared_swerebench_eval_script_omits_setup_commands():
    from examples.mini_swe.swe_reward import _ensure_swe_harness_on_path

    _ensure_swe_harness_on_path()
    from minisweagent.run.benchmarks.swerebench_verify_azure_modal import build_eval_script

    instance = {
        "instance_id": "owner__repo-1",
        "base_commit": "abc123",
        "test_patch": (
            "diff --git a/tests/test_fix.py b/tests/test_fix.py\n"
            "--- a/tests/test_fix.py\n"
            "+++ b/tests/test_fix.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
        "install_config": {
            "packages": "requirements.txt",
            "python": "3.10",
            "eval_commands": ["export PROJECT_MODE=test"],
            "install": "pip install -e .",
            "test_cmd": "pytest -q tests/test_fix.py",
        },
    }

    kept = build_eval_script(instance)
    stripped = build_eval_script(instance, environment_prepared=True)

    # install / eval_commands dropped, but the test command and reset survive.
    assert "pip install -e ." in kept
    assert "pip install -e ." not in stripped
    assert "export PROJECT_MODE=test" not in stripped
    assert "pytest -q tests/test_fix.py" in stripped
    assert "git checkout abc123 tests/test_fix.py" in stripped


def test_prepared_non_python_eval_still_builds_submitted_patch():
    from examples.mini_swe.swe_reward import _ensure_swe_harness_on_path

    _ensure_swe_harness_on_path()
    from minisweagent.run.benchmarks.swerebench_verify_azure_modal import build_eval_script

    script = build_eval_script(
        {
            "instance_id": "owner__repo-1",
            "base_commit": "abc123",
            "test_patch": "",
            "install_config": {
                "build": ["go build ./..."],
                "test_cmd": ["go test ./..."],
            },
        },
        environment_prepared=True,
    )

    assert "go build ./..." in script
    assert "go test ./..." in script


def test_build_eval_script_does_not_read_skip_install_environment(monkeypatch):
    from examples.mini_swe.swe_reward import _ensure_swe_harness_on_path

    _ensure_swe_harness_on_path()
    from minisweagent.run.benchmarks.swerebench_verify_azure_modal import build_eval_script

    instance = {
        "instance_id": "owner__repo-1",
        "base_commit": "abc123",
        "test_patch": (
            "diff --git a/tests/test_fix.py b/tests/test_fix.py\n"
            "--- a/tests/test_fix.py\n"
            "+++ b/tests/test_fix.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
        "install_config": {
            "packages": "requirements.txt",
            "python": "3.10",
            "install": "pip install -e .",
            "test_cmd": "pytest -q tests/test_fix.py",
        },
    }

    monkeypatch.setenv("SWE_SKIP_EVAL_INSTALL", "1")
    monkeypatch.setenv("GENTEST_SKIP_EVAL_INSTALL", "1")

    assert "pip install -e ." in build_eval_script(instance)


# ==============================================================================
# Test Utility Functions
# ==============================================================================

class TestExtractPatchFromOutput:
    """Tests for patch extraction."""

    def test_extract_diff_format(self):
        """Test extracting patch from diff code block."""
        text = """
Here's the fix:

```diff
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
-    return "world"
+    return "universe"
```
"""
        patch = extract_patch_from_output(text)
        assert "--- a/file.py" in patch
        assert "+++ b/file.py" in patch

    def test_extract_patch_format(self):
        """Test extracting patch from patch code block."""
        text = """
```patch
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
```
"""
        patch = extract_patch_from_output(text)
        assert "--- a/file.py" in patch

    def test_extract_plain_code_block(self):
        """Test extracting patch from plain code block."""
        text = """
```
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
```
"""
        patch = extract_patch_from_output(text)
        assert "diff --git" in patch

    def test_extract_raw_diff(self):
        """Test extracting raw diff without code block."""
        text = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
"""
        patch = extract_patch_from_output(text)
        assert "diff --git" in patch

    def test_no_patch_found(self):
        """Test when no patch is found."""
        text = "This is just regular text with no patch."
        patch = extract_patch_from_output(text)
        assert patch == ""

    def test_preserves_required_patch_trailing_newline(self):
        text = """```diff
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
```"""

        patch = extract_patch_from_output(text)

        assert patch.endswith("\n")
        assert not patch.endswith("\n\n")


class TestIsValidPatch:
    """Tests for patch validation."""

    def test_valid_git_diff(self):
        """Test valid git diff patch."""
        patch = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
"""
        assert is_valid_patch(patch) is True

    def test_valid_unified_diff(self):
        """Test valid unified diff patch."""
        patch = """--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
"""
        assert is_valid_patch(patch) is True

    def test_empty_patch(self):
        """Test empty patch is invalid."""
        assert is_valid_patch("") is False

    def test_no_markers(self):
        """Test patch without markers is invalid."""
        patch = "This is not a patch"
        assert is_valid_patch(patch) is False


# ==============================================================================
# Test Reward Functions
# ==============================================================================

class TestCombineRewards:
    """Tests for reward combination."""

    def test_full_resolution_reward(self):
        """Test combining rewards for full resolution."""
        rewards = {
            "resolved": 1.0,
            "fail_to_pass_rate": 1.0,
            "pass_to_pass_rate": 1.0,
            "patch_valid": 1.0,
            "completed": 1.0,
            "step_efficiency": 0.5,
            "token_efficiency": 0.5,
        }
        result = combine_rewards(rewards)
        # (1.0*1.0 + 0.5*1.0 + 0.3*1.0 + 0.1*1.0 + 0.1*1.0 + 0.05*0.5 + 0.05*0.5) * 10
        expected = (1.0 + 0.5 + 0.3 + 0.1 + 0.1 + 0.025 + 0.025) * 10.0
        assert result == pytest.approx(expected)

    def test_zero_rewards(self):
        """Test combining zero rewards."""
        rewards = {
            "resolved": 0.0,
            "fail_to_pass_rate": 0.0,
            "pass_to_pass_rate": 0.0,
            "patch_valid": 0.0,
        }
        result = combine_rewards(rewards)
        assert result == 0.0

    def test_custom_weights(self):
        """Test combining rewards with custom weights."""
        rewards = {
            "resolved": 1.0,
            "fail_to_pass_rate": 0.5,
        }
        custom_weights = {
            "resolved": 2.0,
            "fail_to_pass_rate": 1.0,
        }
        result = combine_rewards(rewards, weights=custom_weights)
        # (2.0*1.0 + 1.0*0.5) * 10 = 25.0
        assert result == pytest.approx(25.0)


class TestPatchClassificationRewardHelpers:
    """Tests for generated-test patch-classification reward helpers."""

    PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1 +1 @@
-old
+new
"""

    def test_extract_generated_test_code_from_sample_response(self):
        sample = Sample(response="```python\nimport pytest\n\ndef test_bug():\n    assert True\n```")

        code = extract_generated_test_code_from_sample(sample)

        assert code.startswith("import pytest")
        assert "def test_bug" in code

    def test_extract_generated_test_code_does_not_use_patch_label(self):
        sample = Sample(label=self.PATCH)

        assert extract_generated_test_code_from_sample(sample) == ""

    def test_load_inline_patch_classification_candidates(self, tmp_path):
        patch_file = tmp_path / "candidate.patch"
        patch_file.write_text(self.PATCH, encoding="utf-8")
        metadata = {
            "instance_id": "repo__issue-1",
            "patch_classification_patches": [
                {"patch": self.PATCH, "label": 1, "sample_idx": 0},
                {"patch_file": str(patch_file), "verify_status": "failed", "sample_idx": 1},
                {"instance_id": "other__issue-2", "patch": self.PATCH, "label": 1},
            ],
        }

        candidates = load_patch_classification_candidates(metadata)

        assert len(candidates) == 2
        assert [c["oracle_resolved"] for c in candidates] == [True, False]
        assert {c["sample_idx"] for c in candidates} == {0, 1}

    def test_load_patch_classification_candidates_from_samples_jsonl(self, tmp_path):
        samples = tmp_path / "patch_samples.jsonl"
        rows = [
            {"instance_id": "repo__issue-1", "patch_sha256": "abc", "patch": self.PATCH, "label": 1},
            {"instance_id": "repo__issue-1", "patch_sha256": "abc", "patch": self.PATCH, "label": 0},
            {"instance_id": "repo__issue-1", "patch_sha256": "other", "patch": self.PATCH, "label": 1},
            {"instance_id": "other__issue-2", "patch_sha256": "abc", "patch": self.PATCH, "label": 1},
        ]
        samples.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        metadata = {
            "instance_id": "repo__issue-1",
            "patch_sha256": "abc",
            "patch_classification_samples_jsonl": str(samples),
        }

        candidates = load_patch_classification_candidates(metadata)

        assert len(candidates) == 2
        assert [c["oracle_resolved"] for c in candidates] == [True, False]

    def test_patch_classification_metrics(self):
        records = [
            {"generated_pred_resolved": True, "oracle_resolved": True},
            {"generated_pred_resolved": True, "oracle_resolved": False},
            {"generated_pred_resolved": False, "oracle_resolved": False},
            {"generated_pred_resolved": False, "oracle_resolved": True},
        ]

        metrics = patch_classification_metrics(records)

        assert metrics["tp"] == 1
        assert metrics["fp"] == 1
        assert metrics["tn"] == 1
        assert metrics["fn"] == 1
        assert metrics["accuracy"] == pytest.approx(0.5)
        assert metrics["balanced_accuracy"] == pytest.approx(0.5)

    def test_base_gold_reward_from_validation(self):
        rewards = _base_gold_reward_from_validation(
            {"base_clean_fail": True, "gold_pass": True, "label": "gold_validated"}
        )

        assert rewards == {"base_reward": 1.0, "gold_reward": 1.0, "patch_reward": 1.0}

    def test_base_gold_reward_requires_strict_gold_validated(self):
        rewards = _base_gold_reward_from_validation(
            {"base_clean_fail": False, "gold_pass": True, "label": "base_not_clean_fail"}
        )

        assert rewards == {"base_reward": 0.0, "gold_reward": 1.0, "patch_reward": 0.0}

    def test_gold_pass_post_process_reward_is_one_even_if_base_not_clean(self):
        post_process = importlib.import_module("examples.mini_swe.post_process_rewards")
        sample = Sample(
            metadata={
                "gentest_record": {
                    "validation": {"label": "base_not_clean_fail"},
                }
            }
        )
        sample.reward = {"base_reward": 0.0, "gold_reward": 1.0, "patch_reward": 0.0}

        raw_rewards, rewards = post_process.patch_classification_rule_based(SimpleNamespace(), [sample])

        assert raw_rewards == [1.0]
        assert rewards == [0.0]
        assert sample.metadata["patch_classification_post_reward"] == 1.0

    def test_base_not_clean_fail_without_gold_pass_post_process_reward_is_zero(self):
        post_process = importlib.import_module("examples.mini_swe.post_process_rewards")
        sample = Sample(
            metadata={
                "gentest_record": {
                    "validation": {"label": "base_not_clean_fail"},
                }
            }
        )
        sample.reward = {"base_reward": 0.0, "gold_reward": 0.0, "patch_reward": 0.0}

        raw_rewards, rewards = post_process.patch_classification_rule_based(SimpleNamespace(), [sample])

        assert raw_rewards == [0.0]
        assert rewards == [0.0]
        assert sample.metadata["patch_classification_post_reward"] == 0.0

    def test_patch_classification_post_process_only_subtracts_group_mean(self):
        post_process = importlib.import_module("examples.mini_swe.post_process_rewards")
        samples = []
        for base_reward, gold_reward in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]:
            sample = Sample(metadata={"gentest_record": {"validation": {"label": ""}}})
            sample.reward = {
                "base_reward": base_reward,
                "gold_reward": gold_reward,
                "patch_reward": 0.0,
            }
            samples.append(sample)

        raw_rewards, rewards = post_process.patch_classification_rule_based(
            SimpleNamespace(n_samples_per_prompt=2), samples
        )

        assert raw_rewards == [0.0, 0.1, 1.0, 0.0]
        assert rewards == pytest.approx([-0.05, 0.05, 0.5, -0.5])

    def test_patch_classification_normalizes_precomputed_raw_reward(self):
        post_process = importlib.import_module("examples.mini_swe.post_process_rewards")
        base_fail_sample = Sample(metadata={})
        base_fail_sample.reward = {
            "base_reward": 1.0,
            "gold_reward": 0.0,
            "patch_cls_reward": 0.0,
            "raw_reward": 0.05,
        }
        base_pass_sample = Sample(metadata={})
        base_pass_sample.reward = {
            "base_reward": 0.0,
            "gold_reward": 0.0,
            "patch_cls_reward": 0.0,
            "raw_reward": 0.0,
        }

        raw_rewards, rewards = post_process.patch_classification_raw_reward_normalization(
            SimpleNamespace(
                reward_key="raw_reward",
                n_samples_per_prompt=2,
                rewards_normalization=True,
                grpo_std_normalization=False,
            ),
            [base_fail_sample, base_pass_sample],
        )

        assert raw_rewards == [0.05, 0.0]
        assert rewards == pytest.approx([0.025, -0.025])
        assert base_fail_sample.reward["raw_reward"] == 0.05
        assert base_pass_sample.reward["raw_reward"] == 0.0
        assert base_fail_sample.reward["normalized_reward"] == pytest.approx(0.025)
        assert base_pass_sample.reward["normalized_reward"] == pytest.approx(-0.025)

    def test_patch_classification_turn_penalty_includes_half_reward_tier(self, monkeypatch):
        post_process = importlib.import_module("examples.mini_swe.post_process_rewards")

        def _make(agent_calls):
            sample = Sample(metadata={"gentest_record": {"agent_calls": agent_calls}})
            sample.reward = {"raw_reward": 0.5}
            return sample

        fast, slow = _make(3), _make(12)
        monkeypatch.setenv("SWE_TURN_PENALTY_ENABLE", "1")
        monkeypatch.setenv("SWE_TURN_PENALTY_TURN_MARGIN", "8")
        monkeypatch.setenv("SWE_TURN_PENALTY_FACTOR", "0.5")
        monkeypatch.delenv("SWE_TURN_PENALTY_TIERS", raising=False)

        raw_rewards, rewards = post_process.patch_classification_raw_reward_normalization(
            SimpleNamespace(
                reward_key="raw_reward",
                n_samples_per_prompt=2,
                rewards_normalization=False,
                grpo_std_normalization=False,
            ),
            [fast, slow],
        )

        assert raw_rewards == [0.5, 0.25]
        assert rewards == [0.5, 0.25]
        assert fast.metadata["turn_penalty_applied"] is False
        assert slow.metadata["turn_penalty_applied"] is True
        assert slow.reward["raw_reward"] == 0.25

    def test_patch_classification_excludes_aborted_from_group_stats(self):
        post_process = importlib.import_module("examples.mini_swe.post_process_rewards")

        def _make(raw, status=Sample.Status.COMPLETED):
            s = Sample(metadata={})
            s.reward = {"raw_reward": raw}
            s.status = status
            return s

        live = [_make(1.0), _make(1.0), _make(1.0)]
        aborted = _make(-1.0, status=Sample.Status.ABORTED)
        samples = live + [aborted]

        raw_rewards, rewards = post_process.patch_classification_raw_reward_normalization(
            SimpleNamespace(
                reward_key="raw_reward",
                n_samples_per_prompt=4,
                rewards_normalization=True,
                grpo_std_normalization=False,
            ),
            samples,
        )

        # Group mean is over live samples only (1.0), so live rewards center to 0
        # and the aborted sample gets a normalized reward of exactly 0 — it neither
        # shifts the others nor contributes gradient (its loss mask is already zero).
        assert raw_rewards == [1.0, 1.0, 1.0, -1.0]
        assert rewards == pytest.approx([0.0, 0.0, 0.0, 0.0])
        assert aborted.reward["normalized_reward"] == 0.0

    def test_patch_classification_excludes_removed_sample_from_group_stats(self):
        post_process = importlib.import_module("examples.mini_swe.post_process_rewards")

        def _make(raw, *, remove_sample=False):
            sample = Sample(metadata={}, remove_sample=remove_sample)
            sample.reward = {"raw_reward": raw}
            return sample

        low = _make(0.0)
        high = _make(1.0)
        infra = _make(100.0, remove_sample=True)

        raw_rewards, rewards = post_process.patch_classification_raw_reward_normalization(
            SimpleNamespace(
                reward_key="raw_reward",
                n_samples_per_prompt=3,
                rewards_normalization=True,
                grpo_std_normalization=False,
            ),
            [low, high, infra],
        )

        assert raw_rewards == [0.0, 1.0, 100.0]
        assert rewards == pytest.approx([-0.5, 0.5, 0.0])
        assert infra.reward["normalized_reward"] == 0.0

    def test_patch_classification_skips_candidate_evaluation_after_gold_fail(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        swe_reward_module._ensure_swe_harness_on_path()
        from minisweagent.run.benchmarks import gentest, patch_gentest, swerebench

        environment = MagicMock()
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_INSTALL_TIMEOUT", "0")
        monkeypatch.setattr(swerebench, "_resolve_per_instance_api_base", lambda _config, _instance_id: None)
        monkeypatch.setattr(swerebench, "get_sb_environment", lambda _config, _instance: environment)
        monkeypatch.setattr(gentest, "reset_base", lambda _environment: None)
        monkeypatch.setattr(gentest, "generated_test_file_for_instance", lambda _instance, filename: filename)
        monkeypatch.setattr(
            gentest,
            "validate_generated_test",
            lambda *_args, **_kwargs: {
                "base_clean_fail": True,
                "gold_applied": True,
                "gold_pass": False,
                "label": "gold_failed",
            },
        )

        def fail_candidate_evaluation(*_args, **_kwargs):
            raise AssertionError("candidate patch evaluation should be skipped after gold failure")

        monkeypatch.setattr(patch_gentest, "validate_candidate_patch", fail_candidate_evaluation)
        sample = Sample(
            response="```python\ndef test_regression():\n    assert False\n```",
            metadata={
                "instance_id": "repo__issue-1",
                "exit_status": "Submitted",
                "patch_classification_config": {"environment": {}},
                "patch_classification_require_gold_validated": True,
                "patch_classification_patches": [
                    {"patch": self.PATCH, "label": 1, "sample_idx": 0},
                ],
            },
        )

        reward = swe_reward_module._compute_patch_classification_reward_sync(SimpleNamespace(), sample)

        assert reward == {
            "base_reward": 1.0,
            "gold_reward": 0.0,
            "patch_cls_reward": 0.0,
            "raw_reward": 0.05,
        }
        details = sample.metadata["patch_classification_reward"]
        assert details["patch_evaluation_skipped"] == "gold_failed"
        assert details["gold_validation_gate"] == "failed"
        assert details["records"] == []
        environment.cleanup.assert_called_once()

    def test_patch_classification_rewards_perfect_base_infra_validation(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        swe_reward_module._ensure_swe_harness_on_path()
        from minisweagent.run.benchmarks import gentest, patch_gentest, swerebench

        environment = MagicMock()
        negative_patch = self.PATCH.replace("+new", "+still-broken")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_INSTALL_TIMEOUT", "0")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_BALANCED_ACC_THRESHOLD", "1.0")
        monkeypatch.setattr(swerebench, "_resolve_per_instance_api_base", lambda _config, _instance_id: None)
        monkeypatch.setattr(swerebench, "get_sb_environment", lambda _config, _instance: environment)
        monkeypatch.setattr(gentest, "reset_base", lambda _environment: None)
        monkeypatch.setattr(gentest, "generated_test_file_for_instance", lambda _instance, filename: filename)
        monkeypatch.setattr(
            gentest,
            "validate_generated_test",
            lambda *_args, **_kwargs: {
                "base_clean_fail": False,
                "gold_applied": True,
                "gold_pass": True,
                "label": "gold_validated",
                "base_validation_label": "base_infra_failure",
                "infra_failure": True,
            },
        )
        monkeypatch.setattr(
            patch_gentest,
            "validate_candidate_patch",
            lambda *_args, patch_text, **_kwargs: {"candidate_pass": patch_text == self.PATCH},
        )
        sample = Sample(
            response="```python\ndef test_regression():\n    assert False\n```",
            metadata={
                "instance_id": "repo__issue-1",
                "exit_status": "Submitted",
                "patch_classification_config": {"environment": {}},
                "patch_classification_max_candidates": 2,
                "patch_classification_patches": [
                    {"patch": self.PATCH, "label": 1, "sample_idx": 0},
                    {"patch": negative_patch, "label": 0, "sample_idx": 1},
                ],
            },
        )

        reward = swe_reward_module._compute_patch_classification_reward_sync(SimpleNamespace(), sample)

        assert reward == {
            "base_reward": 0.0,
            "gold_reward": 1.0,
            "patch_cls_reward": 1.0,
            "raw_reward": 1.0,
        }
        details = sample.metadata["patch_classification_reward"]
        assert details["base_infra_failure"] is True
        assert details["balanced_accuracy"] == 1.0
        assert details["timings"]["sandbox_create_sec"] >= 0.0
        assert details["timings"]["sandbox_prepare_sec"] >= 0.0
        assert details["timings"]["candidate_evaluation_sec"] >= 0.0
        assert len(details["timings"]["candidate_sec_by_patch"]) == 2
        assert details["timings"]["sandbox_cleanup_sec"] >= 0.0
        environment.cleanup.assert_called_once()

    def test_patch_classification_reuses_rollout_environment_without_cleanup(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        swe_reward_module._ensure_swe_harness_on_path()
        from minisweagent.run.benchmarks import gentest, patch_gentest, swerebench

        environment = MagicMock()
        create_environment = MagicMock(side_effect=AssertionError("must reuse rollout environment"))
        negative_patch = self.PATCH.replace("+new", "+still-broken")
        candidate_environments = []
        candidate_skip_install = []
        candidate_execution_contracts = []

        def validate_candidate(candidate_env, *_args, patch_text, **_kwargs):
            candidate_environments.append(candidate_env)
            candidate_skip_install.append(_kwargs.get("official_skip_install"))
            candidate_execution_contracts.append(
                (_kwargs.get("filename"), _kwargs.get("test_command"))
            )
            return {"candidate_pass": patch_text == self.PATCH}

        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_INSTALL_TIMEOUT", "0")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_BALANCED_ACC_THRESHOLD", "1.0")
        monkeypatch.setattr(swerebench, "_resolve_per_instance_api_base", lambda _config, _instance_id: None)
        monkeypatch.setattr(swerebench, "get_sb_environment", create_environment)
        monkeypatch.setattr(
            gentest,
            "generated_test_file_for_instance",
            lambda *_args: pytest.fail("v6 execution contract must bypass the legacy path resolver"),
        )
        monkeypatch.setattr(patch_gentest, "validate_candidate_patch", validate_candidate)
        selected_path = "tests/generated/test_issue.py"
        selected_command = "python -m pytest tests/generated/test_issue.py"
        sample = Sample(
            response="```python\ndef test_regression():\n    assert False\n```",
            metadata={
                "instance_id": "repo__issue-1",
                "exit_status": "Submitted",
                "test_filename": selected_path,
                "generated_test_command": selected_command,
                "agent_inferred_execution_contract": True,
                "patch_classification_config": {"environment": {}},
                "patch_classification_max_candidates": 2,
                "patch_classification_patches": [
                    {"patch": self.PATCH, "label": 1, "sample_idx": 0},
                    {"patch": negative_patch, "label": 0, "sample_idx": 1},
                ],
                "gentest_record": {
                    "agent_inferred_execution_contract": True,
                    "generated_test_command": selected_command,
                    "official_verify_format": True,
                    "official_verifier": {"setup_once": True},
                    "validation": {
                        "base_clean_fail": True,
                        "base_validation_label": "base_clean_fail",
                        "gold_applied": True,
                        "gold_pass": True,
                        "label": "gold_validated",
                    }
                },
            },
        )

        reward = swe_reward_module.compute_patch_classification_reward_in_env(
            SimpleNamespace(), sample, environment
        )

        assert reward == {
            "base_reward": 1.0,
            "gold_reward": 1.0,
            "patch_cls_reward": 1.0,
            "raw_reward": 1.0,
        }
        assert candidate_environments == [environment, environment]
        assert candidate_skip_install == [True, True]
        assert candidate_execution_contracts == [
            (selected_path, selected_command),
            (selected_path, selected_command),
        ]
        create_environment.assert_not_called()
        environment.cleanup.assert_not_called()
        details = sample.metadata["patch_classification_reward"]
        assert details["sandbox_reused_from_rollout"] is True
        assert details["validation_reused_from_rollout"] is True
        assert details["timings"]["sandbox_create_sec"] == 0.0
        assert details["timings"]["sandbox_prepare_sec"] == 0.0
        assert "sandbox_cleanup_sec" not in details["timings"]

    def test_in_environment_reward_failure_does_not_retry_completed_rollout(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        environment = MagicMock()
        sample = Sample(metadata={"instance_id": "repo__issue-1"})
        monkeypatch.setattr(
            swe_reward_module,
            "_compute_patch_classification_reward_sync",
            MagicMock(side_effect=RuntimeError("candidate evaluator failed")),
        )

        reward = swe_reward_module.compute_patch_classification_reward_in_env(
            SimpleNamespace(), sample, environment
        )

        assert reward == {
            "base_reward": 0.0,
            "gold_reward": 0.0,
            "patch_cls_reward": 0.0,
            "raw_reward": 0.0,
        }
        assert sample.metadata["patch_classification_reward"]["sandbox_reused_from_rollout"] is True
        assert "candidate evaluator failed" in sample.metadata["patch_classification_reward"]["error"]
        environment.cleanup.assert_not_called()

    def test_patch_classification_rejects_non_submitted_trajectory_before_sandbox(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_REQUIRE_SUBMITTED", "true")
        sample = Sample(
            response="```python\ndef test_regression():\n    assert False\n```",
            metadata={
                "instance_id": "repo__issue-1",
                "exit_status": "LimitsExceeded",
                "format_error_count": 25,
            },
        )

        reward = swe_reward_module._compute_patch_classification_reward_sync(SimpleNamespace(), sample)

        assert reward == {
            "base_reward": 0.0,
            "gold_reward": 0.0,
            "patch_cls_reward": 0.0,
            "raw_reward": -1.0,
        }
        details = sample.metadata["patch_classification_reward"]
        assert details["exit_status"] == "LimitsExceeded"
        assert details["format_error_count"] == 25
        assert details["error"] == "trajectory did not submit: exit_status=LimitsExceeded"

    def test_reward_func_patch_classification_returns_reward_dict(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        expected = {
            "base_reward": 1.0,
            "gold_reward": 1.0,
            "patch_cls_reward": 0.5,
            "raw_reward": 0.1,
        }
        monkeypatch.setattr(
            swe_reward_module,
            "_compute_patch_classification_reward_sync",
            lambda _args, _sample: expected,
        )

        reward = asyncio.run(
            swe_reward_module.reward_func(
                SimpleNamespace(swe_reward_mode="patch_classification"),
                Sample(metadata={"instance_id": "repo__issue-1"}),
            )
        )

        assert reward == expected

    def test_reward_func_patch_classification_waits_for_cleanup_owned_worker(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        expected = {
            "base_reward": 1.0,
            "gold_reward": 1.0,
            "patch_cls_reward": 1.0,
            "raw_reward": 1.0,
        }

        def slow_reward(_args, _sample):
            time.sleep(0.02)
            return expected

        monkeypatch.setattr(swe_reward_module, "_compute_patch_classification_reward_sync", slow_reward)
        monkeypatch.setattr(swe_reward_module, "SWE_TIMEOUT_REWARD_TOTAL", 0.001)
        sample = Sample(metadata={"instance_id": "repo__issue-1"})
        start = time.monotonic()

        reward = asyncio.run(
            swe_reward_module.reward_func(
                SimpleNamespace(swe_reward_mode="patch_classification"),
                sample,
            )
        )

        assert reward == expected
        assert time.monotonic() - start >= 0.02

    def test_cleanup_owned_worker_finishes_before_cancellation_propagates(self):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        finished = threading.Event()
        executor = swe_reward_module._reward_executor(
            SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=1)
        )

        def worker():
            time.sleep(0.02)
            finished.set()

        async def cancel_worker():
            task = asyncio.create_task(swe_reward_module._run_cleanup_owned_sync(executor, worker))
            await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_worker())
        assert finished.is_set()

    def test_reward_worker_count_defaults_to_full_rollout_pipeline(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        monkeypatch.delenv("SWE_REWARD_WORKERS", raising=False)
        args = SimpleNamespace(rollout_batch_size=16, n_samples_per_prompt=8)

        assert swe_reward_module._reward_worker_count(args) == 128
        assert swe_reward_module._reward_executor(args)._max_workers == 128

        monkeypatch.setenv("SWE_REWARD_WORKERS", "512")
        assert swe_reward_module._reward_worker_count(args) == 512
        assert swe_reward_module._reward_executor(args)._max_workers == 512

    def test_patch_classification_raw_reward_state_mapping(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_BALANCED_ACC_THRESHOLD", "1.0")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_GOLD_SCORE", "0.2")

        assert swe_reward_module._patch_classification_raw_reward(1.0, 1.0, 1.0) == 1.0
        assert swe_reward_module._patch_classification_raw_reward(1.0, 1.0, 0.75) == 0.2
        assert swe_reward_module._patch_classification_raw_reward(1.0, 0.0, 0.0) == 0.05
        assert swe_reward_module._patch_classification_raw_reward(0.0, 1.0, 1.0) == 0.0

    def test_patch_classification_partial_balanced_accuracy_reward_boundaries(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_BALANCED_ACC_THRESHOLD", "1.0")
        monkeypatch.setenv(
            "SWE_PATCH_CLASSIFICATION_POST_REWARD_PARTIAL_BALANCED_ACC_THRESHOLD",
            "0.8",
        )
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_SUCCESS_SCORE", "1.0")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_PARTIAL_SCORE", "0.5")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_GOLD_SCORE", "0.2")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_BASE_SCORE", "0.05")

        reward = swe_reward_module._patch_classification_raw_reward
        assert reward(1.0, 1.0, 0.799999) == 0.2
        assert reward(1.0, 1.0, 0.8) == 0.5
        assert reward(1.0, 1.0, 1.0) == 1.0
        assert reward(1.0, 0.0, 0.9) == 0.05

    def test_patch_classification_rewards_perfect_base_infra_classifier(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_BALANCED_ACC_THRESHOLD", "1.0")

        assert swe_reward_module._patch_classification_raw_reward(
            0.0,
            1.0,
            1.0,
            base_infra_failure=True,
        ) == 1.0
        assert swe_reward_module._patch_classification_raw_reward(
            0.0,
            1.0,
            0.75,
            base_infra_failure=True,
        ) == 0.0
        monkeypatch.setenv("SWE_PATCH_CLASSIFICATION_POST_REWARD_BALANCED_ACC_THRESHOLD", "0.75")
        assert swe_reward_module._patch_classification_raw_reward(
            0.0,
            1.0,
            0.75,
            base_infra_failure=True,
        ) == 0.0

    def test_reward_func_patch_classification_accepts_batch(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")

        def fake_reward(_args, sample):
            patch_cls_reward = float(sample.metadata["score"])
            return {
                "base_reward": 1.0,
                "gold_reward": 1.0,
                "patch_cls_reward": patch_cls_reward,
                "raw_reward": 1.0 if patch_cls_reward >= 1.0 else 0.1,
            }

        monkeypatch.setattr(swe_reward_module, "_compute_patch_classification_reward_sync", fake_reward)

        rewards = asyncio.run(
            swe_reward_module.reward_func(
                SimpleNamespace(swe_reward_mode="patch_classification"),
                [
                    Sample(metadata={"instance_id": "repo__issue-1", "score": 0.25}),
                    Sample(metadata={"instance_id": "repo__issue-2", "score": 0.75}),
                ],
            )
        )

        assert rewards == [
            {
                "base_reward": 1.0,
                "gold_reward": 1.0,
                "patch_cls_reward": 0.25,
                "raw_reward": 0.1,
            },
            {
                "base_reward": 1.0,
                "gold_reward": 1.0,
                "patch_cls_reward": 0.75,
                "raw_reward": 0.1,
            },
        ]

    def test_reward_func_base_gold_returns_reward_dict(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        expected = {"base_reward": 1.0, "gold_reward": 1.0, "patch_reward": 1.0}
        monkeypatch.setattr(
            swe_reward_module,
            "_compute_base_gold_reward_sync",
            lambda _args, _sample: expected,
        )

        reward = asyncio.run(
            swe_reward_module.reward_func(
                SimpleNamespace(swe_reward_mode="base_gold"),
                Sample(metadata={"instance_id": "repo__issue-1"}),
            )
        )

        assert reward == expected

    def test_base_gold_reuses_gentest_record_validation_without_sandbox(self, monkeypatch, tmp_path):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        trajectory_path = tmp_path / "sample.traj.json"
        validation = {"base_clean_fail": True, "gold_pass": True, "label": "gold_validated"}

        def fail_if_sandbox_path_runs():
            raise AssertionError("cached base_gold reward should not initialize the sandbox path")

        monkeypatch.setattr(swe_reward_module, "_ensure_swe_harness_on_path", fail_if_sandbox_path_runs)

        reward = swe_reward_module._compute_base_gold_reward_sync(
            SimpleNamespace(),
            Sample(
                metadata={
                    "instance_id": "repo__issue-1",
                    "trajectory_path": str(trajectory_path),
                    "test_code": "def test_regression():\n    assert True\n",
                    "test_filename": "test_model_gen.py",
                    "gentest_record": {"validation": validation},
                }
            ),
        )

        assert reward == {"base_reward": 1.0, "gold_reward": 1.0, "patch_reward": 1.0}
        details = json.loads((tmp_path / "sample.traj_rewards.json").read_text(encoding="utf-8"))
        assert details["validation_source"] == "gentest_record"
        assert details["validation_label"] == "gold_validated"
        assert details["test_filename"] == "test_model_gen.py"


# ==============================================================================
# Test SWE-bench Harness Integration
# ==============================================================================

class TestSwebenchHarnessIntegration:
    """Tests for SWE-bench harness integration."""

    def test_has_swebench_harness_flag_exists(self):
        """Test that HAS_SWEBENCH_HARNESS flag is defined."""
        assert isinstance(HAS_SWEBENCH_HARNESS, bool)

    def test_swebench_harness_imports(self):
        """Test that swebench harness imports are handled properly."""
        # If HAS_SWEBENCH_HARNESS is True, the imports should be available
        # If False, the fallback constants should be defined
        from .swe_reward import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION

        assert KEY_INSTANCE_ID is not None
        assert KEY_MODEL is not None
        assert KEY_PREDICTION is not None


class TestRunTestsInDocker:
    """Tests for run_tests_in_docker function."""

    def test_invalid_patch_returns_error(self):
        """Test that invalid patch returns error result."""
        result = asyncio.run(run_tests_in_docker(
            instance_id="test-instance",
            patch="not a valid patch",
            repo="test/repo",
            base_commit="abc123",
        ))

        assert result["passed"] is False
        assert result["resolved"] is False
        assert result["error"] == "Invalid patch format"

    def test_empty_patch_returns_error(self):
        """Test that empty patch returns error result."""
        result = asyncio.run(run_tests_in_docker(
            instance_id="test-instance",
            patch="",
            repo="test/repo",
            base_commit="abc123",
        ))

        assert result["passed"] is False
        assert result["error"] == "Invalid patch format"

    def test_result_structure(self):
        """Test that result has expected structure even on error."""
        result = asyncio.run(run_tests_in_docker(
            instance_id="test-instance",
            patch="invalid",
            repo="test/repo",
            base_commit="abc123",
        ))

        # Check all expected keys exist
        expected_keys = [
            "passed", "resolved", "resolution",
            "fail_to_pass_rate", "pass_to_pass_rate",
            "tests_run", "tests_passed",
            "f2p_total", "f2p_passed",
            "p2p_total", "p2p_passed",
            "error", "output"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_requires_swebench_harness(self):
        """Test that function requires swebench harness."""
        if HAS_SWEBENCH_HARNESS:
            pytest.skip("swebench harness is installed")

        result = asyncio.run(run_tests_in_docker(
            instance_id="test-instance",
            patch="diff --git a/f.py b/f.py\n@@ -1 +1 @@\n-a\n+b",
            repo="test/repo",
            base_commit="abc123",
        ))

        assert result["passed"] is False
        assert "SWE-bench harness is required" in result["error"]


class TestRunTestsWithSwebenchHarness:
    """Tests for _run_tests_with_swebench_harness function."""

    @pytest.mark.skipif(not HAS_SWEBENCH_HARNESS, reason="swebench harness not installed")
    def test_creates_test_spec_from_instance(self):
        """Test that TestSpec is created from instance dict."""
        # This test requires swebench to be installed
        # We mock the environment creation to avoid Docker dependency
        with patch('examples.mini_swe.swe_reward.create_environment') as mock_create_env, \
             patch('examples.mini_swe.swe_reward.make_test_spec') as mock_make_spec, \
             patch('examples.mini_swe.swe_reward.stop_environment'):

            # Setup mocks
            mock_spec = MagicMock()
            mock_spec.eval_script = "echo 'test'"
            mock_make_spec.return_value = mock_spec

            mock_env = MagicMock()
            mock_env.execute.return_value = {"returncode": 0, "output": ""}
            mock_create_env.return_value = mock_env

            instance = {
                "instance_id": "test__instance",
                "repo": "test/repo",
                "base_commit": "abc123",
            }

            result = {
                "passed": False, "resolved": False,
                "resolution": "RESOLVED_NO",
                "fail_to_pass_rate": 0.0, "pass_to_pass_rate": 0.0,
                "tests_run": 0, "tests_passed": 0,
                "f2p_total": 0, "f2p_passed": 0,
                "p2p_total": 0, "p2p_passed": 0,
                "error": "", "output": "",
            }

            # Test will fail at get_eval_report, but we verify make_test_spec is called
            try:
                asyncio.run(_run_tests_with_swebench_harness(
                    instance=instance,
                    patch="diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
                    config={"environment": {}},
                    result=result,
                    fail_to_pass=[],
                    pass_to_pass=[],
                ))
            except Exception:
                pass  # Expected to fail without full setup

            mock_make_spec.assert_called_once_with(instance)

    def test_handles_test_spec_creation_failure(self):
        """Test handling when TestSpec creation fails."""
        if not HAS_SWEBENCH_HARNESS:
            pytest.skip("swebench harness not installed")

        with patch('examples.mini_swe.swe_reward.make_test_spec') as mock_make_spec:
            mock_make_spec.side_effect = ValueError("Invalid instance")

            instance = {"instance_id": "bad-instance"}
            result = {
                "passed": False, "resolved": False,
                "resolution": "RESOLVED_NO",
                "fail_to_pass_rate": 0.0, "pass_to_pass_rate": 0.0,
                "tests_run": 0, "tests_passed": 0,
                "f2p_total": 0, "f2p_passed": 0,
                "p2p_total": 0, "p2p_passed": 0,
                "error": "", "output": "",
            }

            result = asyncio.run(_run_tests_with_swebench_harness(
                instance=instance,
                patch="diff",
                config={"environment": {}},
                result=result,
                fail_to_pass=[],
                pass_to_pass=[],
            ))

            assert "Failed to create TestSpec" in result["error"]

    def test_recognizes_unlabeled_prebuilt_swerebench_instance(self):
        instance = {
            "instance_id": "autrainer__autrainer-105",
            "repo": "autrainer/autrainer",
            "image_url": "swerebench/sweb.eval.x86_64.autrainer_1776_autrainer-105",
            "install_config": {
                "test_cmd": "pytest --no-header -rA --tb=line --color=no",
                "log_parser": "parse_log_pytest",
            },
        }

        assert _is_unlabeled_swerebench_instance(instance) is True

    @pytest.mark.skipif(not HAS_SWEBENCH_HARNESS, reason="swebench harness not installed")
    def test_does_not_override_known_standard_swebench_repo(self):
        instance = {
            "instance_id": "django__django-11099",
            "repo": "django/django",
            "image_url": "custom/prebuilt-image",
            "install_config": {"test_cmd": "./tests/runtests.py"},
        }

        assert _is_unlabeled_swerebench_instance(instance) is False

    @pytest.mark.skipif(not HAS_SWEBENCH_HARNESS, reason="swebench harness not installed")
    def test_routes_unlabeled_prebuilt_instance_to_swerebench_builder(self):
        instance = {
            "instance_id": "autrainer__autrainer-105",
            "repo": "autrainer/autrainer",
            "image_url": "swerebench/sweb.eval.x86_64.autrainer_1776_autrainer-105",
            "install_config": {
                "test_cmd": "pytest --no-header -rA --tb=line --color=no",
                "log_parser": "parse_log_pytest",
            },
        }
        result = {
            "passed": False, "resolved": False,
            "resolution": "RESOLVED_NO",
            "fail_to_pass_rate": 0.0, "pass_to_pass_rate": 0.0,
            "tests_run": 0, "tests_passed": 0,
            "f2p_total": 0, "f2p_passed": 0,
            "p2p_total": 0, "p2p_passed": 0,
            "error": "", "output": "",
        }

        with patch(
            "examples.mini_swe.swe_reward.make_test_spec_swerebench_v2",
            side_effect=ValueError("selected swerebench builder"),
        ) as swerebench_builder, patch(
            "examples.mini_swe.swe_reward.make_test_spec"
        ) as standard_builder:
            result = asyncio.run(
                _run_tests_with_swebench_harness(
                    instance=instance,
                    patch="diff",
                    config={"environment": {}},
                    result=result,
                    fail_to_pass=[],
                    pass_to_pass=[],
                )
            )

        assert "selected swerebench builder" in result["error"]
        swerebench_builder.assert_called_once_with(instance)
        standard_builder.assert_not_called()

    def test_grades_prebuilt_pytest_output_without_standard_repo_map(self):
        instance = {"install_config": {"log_parser": "parse_log_pytest"}}
        test_spec = SimpleNamespace(
            FAIL_TO_PASS=["tests/test_fix.py::test_fix"],
            PASS_TO_PASS=["tests/test_existing.py::test_existing"],
        )
        result = {
            "passed": False, "resolved": False,
            "resolution": "RESOLVED_NO",
            "fail_to_pass_rate": 0.0, "pass_to_pass_rate": 0.0,
            "tests_run": 0, "tests_passed": 0,
            "f2p_total": 0, "f2p_passed": 0,
            "p2p_total": 0, "p2p_passed": 0,
            "error": "", "output": "",
        }
        output = f"{START_TEST_OUTPUT}\npytest output\n{END_TEST_OUTPUT}"

        with patch(
            "examples.mini_swe.swe_reward.parse_log_pytest",
            return_value={
                "tests/test_fix.py::test_fix": "PASSED",
                "tests/test_existing.py::test_existing": "PASSED",
            },
        ):
            graded = grade_prebuilt_pytest_output(instance, test_spec, output, result)

        assert graded["resolved"] is True
        assert graded["f2p_passed"] == graded["f2p_total"] == 1
        assert graded["p2p_passed"] == graded["p2p_total"] == 1

    def test_subfailed_is_failure_even_when_parent_node_is_reported_passed(self):
        node = "tests/test_more.py::DistinctPermutationsTests::test_unsortable"
        instance = {"install_config": {"log_parser": "parse_log_pytest_v2"}}
        test_spec = SimpleNamespace(FAIL_TO_PASS=[node], PASS_TO_PASS=[])
        result = {
            "passed": False, "resolved": False,
            "resolution": "RESOLVED_NO",
            "fail_to_pass_rate": 0.0, "pass_to_pass_rate": 0.0,
            "tests_run": 0, "tests_passed": 0,
            "f2p_total": 0, "f2p_passed": 0,
            "p2p_total": 0, "p2p_passed": 0,
            "error": "", "output": "",
        }
        output = (
            f"{START_TEST_OUTPUT}\n"
            f"PASSED {node}\n"
            f"SUBFAILED(iterable=[1, 2]) {node}\n"
            f"{END_TEST_OUTPUT}"
        )

        with patch(
            "examples.mini_swe.swe_reward.parse_log_pytest_v2",
            return_value={node: "PASSED"},
        ):
            graded = grade_prebuilt_pytest_output(
                instance,
                test_spec,
                output,
                result,
                returncode=0,
            )

        assert graded["resolved"] is False
        assert graded["f2p_passed"] == 0
        assert graded["f2p_total"] == 1

    def test_nonzero_pytest_returncode_cannot_resolve_prebuilt_instance(self):
        node = "tests/test_fix.py::test_fix"
        instance = {"install_config": {"log_parser": "parse_log_pytest"}}
        test_spec = SimpleNamespace(FAIL_TO_PASS=[node], PASS_TO_PASS=[])
        result = {
            "passed": False, "resolved": False,
            "resolution": "RESOLVED_NO",
            "fail_to_pass_rate": 0.0, "pass_to_pass_rate": 0.0,
            "tests_run": 0, "tests_passed": 0,
            "f2p_total": 0, "f2p_passed": 0,
            "p2p_total": 0, "p2p_passed": 0,
            "error": "", "output": "",
        }
        output = f"{START_TEST_OUTPUT}\nPASSED {node}\n{END_TEST_OUTPUT}"

        with patch(
            "examples.mini_swe.swe_reward.parse_log_pytest",
            return_value={node: "PASSED"},
        ):
            graded = grade_prebuilt_pytest_output(
                instance,
                test_spec,
                output,
                result,
                returncode=1,
            )

        assert graded["f2p_passed"] == 1
        assert graded["resolved"] is False
        assert graded["test_returncode"] == 1

    def test_scale_focused_command_is_full_verify_command_with_selected_subset(self):
        selected = ["tests/test_more.py::TestCase::test_fix"]
        command = build_prebuilt_test_command(
            {"dataset": "AweAI-Team/Scale-SWE"},
            selected,
            [],
            focused=True,
        )

        assert command.startswith("pytest --no-header -rA --tb=line --color=no")
        assert command.endswith(selected[0])


class TestRunTestsDispatch:
    """Tests for run_tests_in_docker dispatch logic."""

    def test_dispatches_to_harness_when_available(self):
        """Test that function dispatches to harness implementation when available."""
        if not HAS_SWEBENCH_HARNESS:
            pytest.skip("swebench harness not installed")

        async def mock_harness_coro(*_args, **_kwargs):
            return {"resolved": True, "passed": True}

        with patch(
            'examples.mini_swe.swe_reward._run_tests_with_swebench_harness',
            side_effect=mock_harness_coro,
        ) as mock_harness:
            asyncio.run(run_tests_in_docker(
                instance_id="test-instance",
                patch="diff --git a/f.py b/f.py\n@@ -1 +1 @@\n-a\n+b",
                repo="test/repo",
                base_commit="abc123",
                config={"environment": {}},
            ))

            mock_harness.assert_called_once()

    def test_dispatches_swerebench_v2_to_official_verifier(self, monkeypatch):
        monkeypatch.setenv("SWE_MAX_ENV_RETRIES", "1")
        monkeypatch.setenv("SWE_ENV_RETRY_WAIT", "0")
        instance = {
            "instance_id": "autrainer__autrainer-105",
            "dataset": "nebius/SWE-rebench-V2",
            "repo": "autrainer/autrainer",
            "base_commit": "abc123",
            "image_url": "swerebench/sweb.eval.x86_64.autrainer_1776_autrainer-105",
            "install_config": {"test_cmd": "pytest -rA", "log_parser": "parse_log_pytest"},
        }

        async def mock_official(**kwargs):
            return {**kwargs["result"], "resolved": True, "passed": True}

        with patch(
            "examples.mini_swe.swe_reward._run_official_swerebench_verifier",
            side_effect=mock_official,
        ) as official, patch(
            "examples.mini_swe.swe_reward._run_tests_with_swebench_harness"
        ) as legacy:
            result = asyncio.run(
                run_tests_in_docker(
                    instance_id=instance["instance_id"],
                    patch="diff --git a/f.py b/f.py\n@@ -1 +1 @@\n-a\n+b\n",
                    repo=instance["repo"],
                    base_commit=instance["base_commit"],
                    fail_to_pass=["tests/test_fix.py::test_fix"],
                    pass_to_pass=[],
                    instance=instance,
                    config={"environment": {}},
                )
            )

        assert result["resolved"] is True
        official.assert_called_once()
        legacy.assert_not_called()

    def test_official_swerebench_eval_uses_command_timeout(self, monkeypatch):
        swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
        swe_reward_module._ensure_swe_harness_on_path()
        verifier = importlib.import_module(
            "minisweagent.run.benchmarks.swerebench_verify_azure_modal"
        )
        environment_module = importlib.import_module(
            "minisweagent.environments.extra.azure_modal"
        )

        class EvalStarted(Exception):
            pass

        class FakeEnvironment:
            cleaned = False

            def __init__(self, **kwargs):
                self.config = kwargs

            def execute(self, action, cwd="/", *, timeout=None):
                del cwd
                if action["command"] == "bash /tmp/eval.sh":
                    raise EvalStarted(timeout)
                return {"returncode": 0, "output": "", "exception_info": ""}

            def cleanup(self):
                type(self).cleaned = True

        monkeypatch.setattr(environment_module, "AzureModalEnvironment", FakeEnvironment)
        instance = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "image_name": "owner/repo:latest",
            "base_commit": "abc123",
            "install_config": {
                "packages": "requirements.txt",
                "test_cmd": "pytest -q",
                "log_parser": "parse_log_pytest",
            },
            "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
            "PASS_TO_PASS": [],
            "test_patch": "",
        }

        with pytest.raises(EvalStarted) as exc_info:
            verifier.evaluate_instance_azure_modal(
                instance,
                "diff --git a/f.py b/f.py\n@@ -1 +1 @@\n-a\n+b\n",
                {"sandbox_timeout": 480, "timeout": 360},
            )

        assert exc_info.value.args == (360,)
        assert FakeEnvironment.cleaned is True

    def test_official_swerebench_reuses_existing_environment(self):
        class ExistingEnvironment:
            def __init__(self):
                self.commands = []

            def execute(self, command):
                self.commands.append(command)
                if command == "bash /tmp/eval.sh 2>&1":
                    return {
                        "returncode": 0,
                        "output": (
                            f"{START_TEST_OUTPUT}\n"
                            "PASSED tests/test_fix.py::test_fix\n"
                            "PASSED tests/test_old.py::test_old\n"
                            f"{END_TEST_OUTPUT}\n"
                        ),
                    }
                return {"returncode": 0, "output": ""}

        env = ExistingEnvironment()
        instance = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "abc123",
            "install_config": {
                "packages": "requirements.txt",
                "test_cmd": "pytest -q",
                "log_parser": "parse_log_pytest",
            },
            "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_old.py::test_old"],
            "test_patch": "",
        }

        result = asyncio.run(
            run_official_swerebench_in_environment(
                env,
                instance,
                fail_to_pass=instance["FAIL_TO_PASS"],
                pass_to_pass=instance["PASS_TO_PASS"],
            )
        )

        assert result["resolved"] is True
        assert result["reused_rollout_verify_environment"] is True
        assert "bash /tmp/eval.sh 2>&1" in env.commands

    def test_standard_swebench_keeps_legacy_verifier(self, monkeypatch):
        monkeypatch.setenv("SWE_MAX_ENV_RETRIES", "1")
        monkeypatch.setenv("SWE_ENV_RETRY_WAIT", "0")
        instance = {
            "instance_id": "django__django-11099",
            "dataset": "princeton-nlp/SWE-bench_Verified",
            "repo": "django/django",
            "base_commit": "abc123",
        }

        async def mock_legacy(**kwargs):
            return {**kwargs["result"], "resolved": True, "passed": True}

        with patch(
            "examples.mini_swe.swe_reward._run_tests_with_swebench_harness",
            side_effect=mock_legacy,
        ) as legacy, patch(
            "examples.mini_swe.swe_reward._run_official_swerebench_verifier"
        ) as official, patch(
            "examples.mini_swe.swe_reward.HAS_SWEBENCH_HARNESS", True
        ):
            result = asyncio.run(
                run_tests_in_docker(
                    instance_id=instance["instance_id"],
                    patch="diff --git a/f.py b/f.py\n@@ -1 +1 @@\n-a\n+b\n",
                    repo=instance["repo"],
                    base_commit=instance["base_commit"],
                    instance=instance,
                    config={"environment": {}},
                )
            )

        assert result["resolved"] is True
        legacy.assert_called_once()
        official.assert_not_called()

    def test_maps_official_result_to_reward_schema(self):
        base_result = {
            "passed": False,
            "resolved": False,
            "resolution": "RESOLVED_NO",
            "fail_to_pass_rate": 0.0,
            "pass_to_pass_rate": 0.0,
            "tests_run": 0,
            "tests_passed": 0,
            "f2p_total": 0,
            "f2p_passed": 0,
            "p2p_total": 0,
            "p2p_passed": 0,
            "error": "",
            "output": "",
        }
        official = {
            "resolved": True,
            "exit_code": 0,
            "patch_applied": True,
            "fail_to_pass_passed": ["tests/test_fix.py::test_fix"],
            "fail_to_pass_failed": [],
            "pass_to_pass_passed": 2,
            "pass_to_pass_failed": [],
            "parsed_tests_count": 3,
            "error": "",
        }

        mapped = _map_official_swerebench_result(
            official,
            base_result,
            ["tests/test_fix.py::test_fix"],
            ["tests/test_old.py::test_a", "tests/test_old.py::test_b"],
        )

        assert mapped["passed"] is True
        assert mapped["fail_to_pass_rate"] == 1.0
        assert mapped["pass_to_pass_rate"] == 1.0
        assert mapped["tests_passed"] == mapped["tests_run"] == 3
        assert mapped["official_swerebench_verifier"] is True

    def test_patch_apply_failure_is_an_explicit_official_eval_error(self):
        mapped = _map_official_swerebench_result(
            {
                "resolved": True,
                "patch_applied": False,
                "fail_to_pass_passed": ["test_fix"],
                "pass_to_pass_passed": 0,
            },
            {"error": "", "output": ""},
            ["test_fix"],
            [],
        )

        assert mapped["resolved"] is False
        assert "failed to apply model patch" in mapped["error"]

    def test_returns_error_when_harness_unavailable(self):
        """Test that function returns error when harness is unavailable."""
        if HAS_SWEBENCH_HARNESS:
            pytest.skip("swebench harness is installed - cannot test fallback")

        result = asyncio.run(run_tests_in_docker(
            instance_id="test-instance",
            patch="diff --git a/f.py b/f.py\n@@ -1 +1 @@\n-a\n+b",
            repo="test/repo",
            base_commit="abc123",
        ))

        assert result["passed"] is False
        assert "SWE-bench harness is required" in result["error"]


# ==============================================================================
# Test Process-Shaping Helpers (mode: process_shaped)
# ==============================================================================

from types import SimpleNamespace


def _step(idx, command="", returncode=0, observation=""):
    """Build a lightweight Step-like object for trajectory helper tests."""
    return SimpleNamespace(idx=idx, command=command, returncode=returncode, observation=observation)


def _traj(steps, gold_changed_files=None, gold_test_files=None):
    return SimpleNamespace(
        steps=steps,
        gold_changed_files=gold_changed_files or [],
        gold_test_files=gold_test_files or [],
    )


class TestIsTestPath:
    def test_test_dir(self):
        assert _is_test_path("tests/test_foo.py")
        assert _is_test_path("pkg/test_bar.py")
        assert _is_test_path("pkg/bar_test.py")
        assert _is_test_path("conftest.py")

    def test_non_test(self):
        assert not _is_test_path("src/pkg/module.py")
        assert not _is_test_path("pkg/contest.py")


class TestLocalizationScore:
    def test_full_overlap(self):
        assert _localization_score(["a/b.py"], ["a/b.py"]) == pytest.approx(1.0)

    def test_partial_overlap(self):
        score = _localization_score(["a/b.py"], ["a/b.py", "a/c.py"])
        assert score == pytest.approx(0.5)

    def test_no_gold_is_inert(self):
        assert _localization_score(["a/b.py"], []) == 0.0

    def test_gold_test_files_excluded(self):
        # Only gold *source* files count toward the denominator.
        score = _localization_score(["a/b.py"], ["a/b.py", "tests/test_b.py"])
        assert score == pytest.approx(1.0)


class TestReadBeforeEditBonus:
    def test_read_then_edit(self):
        steps = [
            _step(0, command="cat src/mod.py"),
            _step(1, command="sed -i 's/a/b/' src/mod.py"),
        ]
        assert _read_before_edit_bonus(_traj(steps, gold_changed_files=["src/mod.py"]), ["src/mod.py"]) == 1.0

    def test_edit_without_prior_read(self):
        steps = [
            _step(0, command="sed -i 's/a/b/' src/mod.py"),
            _step(1, command="cat src/mod.py"),
        ]
        assert _read_before_edit_bonus(_traj(steps), ["src/mod.py"]) == 0.0

    def test_no_gold_is_inert(self):
        steps = [_step(0, command="cat src/mod.py")]
        assert _read_before_edit_bonus(_traj(steps), []) == 0.0


class TestExtraFilesPenalty:
    def test_no_extra(self):
        assert _extra_files_penalty(["a/b.py"], ["a/b.py"]) == 0.0

    def test_half_extra(self):
        pen = _extra_files_penalty(["a/b.py", "a/x.py"], ["a/b.py"])
        assert pen == pytest.approx(0.5)

    def test_no_gold_is_inert(self):
        assert _extra_files_penalty(["a/b.py"], []) == 0.0

    def test_test_files_ignored(self):
        # Submitted test files are not counted as source over-reach here.
        pen = _extra_files_penalty(["a/b.py", "tests/test_b.py"], ["a/b.py"])
        assert pen == pytest.approx(0.0)


class TestDetectFlip:
    def test_anchored_flip(self):
        steps = [
            _step(0, observation="tests/test_x.py::test_foo FAILED"),
            _step(1, observation="tests/test_x.py::test_foo PASSED"),
        ]
        assert _detect_flip(_traj(steps), ["tests/test_x.py::test_foo"]) == 1.0

    def test_no_flip_only_pass(self):
        steps = [_step(0, observation="tests/test_x.py::test_foo PASSED")]
        assert _detect_flip(_traj(steps), ["tests/test_x.py::test_foo"]) == 0.0

    def test_ran_test_without_flip_not_rewarded(self):
        # A test that fails and stays failing is not a flip.
        steps = [
            _step(0, observation="tests/test_x.py::test_foo FAILED"),
            _step(1, observation="tests/test_x.py::test_foo FAILED"),
        ]
        assert _detect_flip(_traj(steps), ["tests/test_x.py::test_foo"]) == 0.0

    def test_unanchored_flip_partial_credit(self):
        steps = [
            _step(0, observation="tests/test_y.py::test_bar FAILED"),
            _step(1, observation="tests/test_y.py::test_bar PASSED"),
        ]
        # Gold names exist but the flipped test isn't one of them -> partial.
        assert _detect_flip(_traj(steps), ["tests/test_x.py::test_foo"]) == 0.5


class TestErrorRate:
    def test_all_ok(self):
        steps = [_step(0, returncode=0), _step(1, returncode=0)]
        assert _error_rate(_traj(steps)) == 0.0

    def test_half_errors(self):
        steps = [_step(0, returncode=0), _step(1, returncode=1)]
        assert _error_rate(_traj(steps)) == pytest.approx(0.5)

    def test_none_returncodes_ignored(self):
        steps = [_step(0, returncode=None), _step(1, returncode=1)]
        assert _error_rate(_traj(steps)) == pytest.approx(1.0)


class TestNoProgressPenalty:
    def test_no_repeats(self):
        steps = [_step(0, command="ls"), _step(1, command="cat a")]
        assert _no_progress_penalty(_traj(steps)) == 0.0

    def test_repeated_command(self):
        steps = [_step(0, command="ls"), _step(1, command="ls"), _step(2, command="ls")]
        # 2 of 3 commands are repeats.
        assert _no_progress_penalty(_traj(steps)) == pytest.approx(2 / 3)


class TestIntegrityGate:
    def test_clean(self):
        patch = "diff --git a/src/mod.py b/src/mod.py\n@@ -1 +1 @@\n-a\n+b"
        steps = [_step(0, command="pytest")]
        assert _integrity_gate(patch, _traj(steps)) == 1.0

    def test_edits_test_file(self):
        patch = "diff --git a/tests/test_mod.py b/tests/test_mod.py\n@@ -1 +1 @@\n-a\n+b"
        assert _integrity_gate(patch, _traj([])) == 0.0

    def test_edits_gold_test_file(self):
        patch = "diff --git a/foo/bar.py b/foo/bar.py\n@@ -1 +1 @@\n-a\n+b"
        # bar.py isn't a test path by heuristic but is a known gold test file.
        assert _integrity_gate(patch, _traj([]), gold_test_files=["foo/bar.py"]) == 0.0

    def test_hardcoded_skip(self):
        patch = (
            "diff --git a/src/mod.py b/src/mod.py\n@@ -1 +1 @@\n"
            "+    pytest.skip('flaky')\n"
        )
        assert _integrity_gate(patch, _traj([])) == 0.0

    def test_destructive_command(self):
        patch = "diff --git a/src/mod.py b/src/mod.py\n@@ -1 +1 @@\n-a\n+b"
        steps = [_step(0, command="rm -rf /testbed")]
        assert _integrity_gate(patch, _traj(steps)) == 0.0


class TestCombineProcessRewards:
    def test_gate_zeroes_reward(self):
        rewards = {"resolved": 1.0, "f2p_rate": 1.0, "integrity_gate": 0.0}
        assert combine_process_rewards(SimpleNamespace(), rewards) == 0.0

    def test_outcome_dominates(self):
        rewards = {"resolved": 1.0, "integrity_gate": 1.0}
        # w0=1.0 default, ×10.
        assert combine_process_rewards(SimpleNamespace(), rewards) == pytest.approx(10.0)

    def test_penalties_subtracted(self):
        rewards = {"resolved": 1.0, "extra_files": 1.0, "integrity_gate": 1.0}
        # (1.0*1.0 - 0.05*1.0) * 10 = 9.5
        assert combine_process_rewards(SimpleNamespace(), rewards) == pytest.approx(9.5)

    def test_zero_variance_break(self):
        # Two all-fail rollouts (resolved=0) differ in localization -> different
        # reward, so a GRPO group of all-fail samples is no longer zero-variance.
        a = {"resolved": 0.0, "localization": 1.0, "integrity_gate": 1.0}
        b = {"resolved": 0.0, "localization": 0.0, "integrity_gate": 1.0}
        ra = combine_process_rewards(SimpleNamespace(), a)
        rb = combine_process_rewards(SimpleNamespace(), b)
        assert ra != rb


def test_reward_config_runtime_endpoint_overrides_raw_config(monkeypatch):
    swe_reward_module = importlib.import_module("examples.mini_swe.swe_reward")
    monkeypatch.setenv("SANDBOX_BASE_URL", "https://sandbox.example")
    monkeypatch.setenv("SANDBOX_API_KEY", "runtime-key")
    metadata = {
        "patch_classification_config": {
            "environment": {
                "base_url": "http://stale-sandbox.example",
                "api_key": "stale-key",
            }
        }
    }

    config = swe_reward_module._build_patch_classification_config(
        SimpleNamespace(), metadata, MagicMock(), MagicMock()
    )

    assert config["environment"]["base_url"] == "https://sandbox.example"
    assert config["environment"]["api_key"] == "runtime-key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
