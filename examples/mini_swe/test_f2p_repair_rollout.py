from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from slime.utils.types import Sample

from . import f2p_repair_rollout as rollout
from .gentest_filters import mask_aborted


SOURCE_PATCH_0 = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1 +1 @@
-old
+round0
"""

SOURCE_PATCH_1 = SOURCE_PATCH_0.replace("round0", "repaired")
TEST_PATCH = """diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1 +1 @@
-old test
+hidden test
"""


def test_instance_id_rejects_trajectory_path_traversal():
    with pytest.raises(ValueError, match="unsafe instance_id"):
        rollout._instance_from_sample(
            Sample(index=1, metadata={"instance_id": "../../outside", "problem_statement": "fix it"})
        )


def test_group_siblings_choose_the_same_f2p_node(monkeypatch):
    monkeypatch.setenv("SWE_F2P_REPAIR_N_TESTS", "1")
    args = SimpleNamespace(rollout_seed=17)
    instance = {
        "instance_id": "repo__case-1",
        "FAIL_TO_PASS": ["tests/test_a.py::test_a", "tests/test_b.py::test_b"],
    }
    first = Sample(index=10, group_index=3)
    sibling = Sample(index=11, group_index=3)

    assert rollout.select_f2p_nodes(args, first, instance) == rollout.select_f2p_nodes(args, sibling, instance)


def test_f2p_selection_drops_malformed_dataset_markers(monkeypatch):
    monkeypatch.setenv("SWE_F2P_REPAIR_N_TESTS", "3")
    args = SimpleNamespace(rollout_seed=17)
    instance = {
        "instance_id": "repo__case-1",
        "FAIL_TO_PASS": ["[", "tests/test_a.py::test_a[ok]", "tests/test_b.py::test_b[broken"],
    }

    assert rollout.select_f2p_nodes(args, Sample(group_index=1), instance) == [
        "tests/test_a.py::test_a[ok]"
    ]


def test_forbidden_test_changes_include_hidden_and_generic_test_paths():
    model_patch = TEST_PATCH + "\n" + SOURCE_PATCH_0 + "\n" + TEST_PATCH.replace(
        "tests/test_core.py", "checks/test_other.py"
    )

    assert rollout.forbidden_test_paths(model_patch, TEST_PATCH) == [
        "tests/test_core.py",
        "checks/test_other.py",
    ]
    assert rollout._looks_like_test_path("conftest.py")


def test_normalizes_eos_and_think_prefix_without_retokenizing():
    tokenizer = _FakeTokenizer()

    assert rollout._normalize_response_text("answer<eos>", tokenizer) == "answer"
    assert rollout._normalize_response_text("reason\n</think>answer<eos>", tokenizer) == (
        "<think>\nreason\n</think>answer"
    )


def test_persistent_official_session_reuses_shared_dataset_gate(monkeypatch):
    node = "tests/test_core.py::test_hidden"
    captured = []

    def evaluate_patch(instance, patch, *, fail_to_pass, pass_to_pass, timeout):
        captured.append(
            {
                "instance": instance,
                "patch": patch,
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
                "timeout": timeout,
            }
        )
        return {
            "resolved": True,
            "patch_applied": True,
            "exit_code": 0,
            "fail_to_pass_passed": list(fail_to_pass),
            "fail_to_pass_failed": [],
            "pass_to_pass_passed": len(pass_to_pass),
            "pass_to_pass_failed": [],
            "parsed_tests_count": len(fail_to_pass) + len(pass_to_pass),
            "error": "",
            "output": f"PASSED {node}",
            "official_verifier": "swerebench",
        }

    def normalize_result(result, *, fail_to_pass, pass_to_pass):
        return {
            **result,
            "f2p_total": len(fail_to_pass),
            "f2p_passed": len(result["fail_to_pass_passed"]),
            "p2p_total": len(pass_to_pass),
            "p2p_passed": result["pass_to_pass_passed"],
            "error": result.get("error", ""),
        }

    class FakePersistentGate:
        def __init__(self, instance, *, timeout):
            captured.append({"created_instance": instance, "created_timeout": timeout})
            self.closed = False

        def evaluate(self, patch, *, fail_to_pass, pass_to_pass, timeout):
            return evaluate_patch(
                {},
                patch,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                timeout=timeout,
            )

        def close(self):
            self.closed = True
            captured.append({"closed": True})

    fake_gate = SimpleNamespace(
        PersistentOfficialGate=FakePersistentGate,
        normalize_result=normalize_result,
    )
    monkeypatch.setattr(rollout, "_load_official_f2p_gate", lambda: fake_gate)
    session = rollout.OfficialVerifySession(
        {
            "instance_id": "owner__repo-1",
            "dataset": "nebius/SWE-rebench-V2",
            "FAIL_TO_PASS": [node],
            "PASS_TO_PASS": ["tests/test_core.py::test_existing"],
            "test_patch": TEST_PATCH,
        },
        test_timeout=240,
        feedback_chars=6000,
    )
    asyncio.run(session.initialize(600))

    result = asyncio.run(session.check_f2p(SOURCE_PATCH_0, [node]))

    assert result["passed"] is True
    assert result["persistent_verifier"] is True
    assert result["official_gate"]["official_verifier"] == "swerebench"
    assert captured[1]["patch"] == SOURCE_PATCH_0
    assert captured[1]["fail_to_pass"] == [node]
    assert captured[1]["pass_to_pass"] == []
    assert captured[1]["timeout"] == 240

    final = asyncio.run(session.run_final_official(SOURCE_PATCH_1))

    assert captured[2]["patch"] == SOURCE_PATCH_1
    assert captured[2]["fail_to_pass"] == [node]
    assert captured[2]["pass_to_pass"] == ["tests/test_core.py::test_existing"]
    assert final["f2p_total"] == 1
    assert final["p2p_total"] == 1
    assert final["persistent_verifier"] is True
    asyncio.run(session.close())
    assert captured[-1] == {"closed": True}


def test_persistent_official_patch_apply_failure_is_gate_feedback(monkeypatch):
    class FakePersistentGate:
        def __init__(self, *_args, **_kwargs):
            pass

        def evaluate(self, *_args, **_kwargs):
            return {
                "resolved": False,
                "patch_applied": False,
                "exit_code": 1,
                "fail_to_pass_passed": [],
                "fail_to_pass_failed": ["tests/test_core.py::test_hidden"],
                "error": "",
                "output": "error: patch failed",
                "official_verifier": "swebench",
            }

        def close(self):
            pass

    fake_gate = SimpleNamespace(PersistentOfficialGate=FakePersistentGate)
    monkeypatch.setattr(rollout, "_load_official_f2p_gate", lambda: fake_gate)
    session = rollout.OfficialVerifySession(
        {"instance_id": "repo__case-1", "test_patch": TEST_PATCH},
        test_timeout=240,
        feedback_chars=6000,
    )
    asyncio.run(session.initialize(600))

    result = asyncio.run(
        session.check_f2p(SOURCE_PATCH_0, ["tests/test_core.py::test_hidden"])
    )

    assert result["passed"] is False
    assert result["submitted_patch_apply_error"] == "SUBMITTED_PATCH_APPLY_FAILED"
    assert result["forbidden_test_paths"] == []
    assert "SUBMITTED_PATCH_APPLY_FAILED" in result["feedback"]


class _FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None):
        del tokenize, tools
        text = "".join(f"<{message['role']}>{message.get('content', '')}" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": self.encode(text)}

    def encode(self, text):
        return [1] * len(text)


class _FakeAgent:
    def __init__(self, env, **config):
        self.env = env
        self.config = SimpleNamespace(**config)
        self.messages = []
        self.n_calls = 0

    def setup(self, task, workdir):
        self.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"{task} in {workdir}"},
        ]

    def add_messages(self, *messages):
        self.messages.extend(messages)

    def check_limits(self):
        return None

    def parse_response(self, response):
        return {"extra": {"actions": [{"command": response, "tool_call_id": "call"}]}}

    async def execute_actions(self, actions, timeout=None):
        del timeout
        if "submit" not in actions[0]["command"]:
            return [{"output": "work in progress", "returncode": 0}], None
        submission = SOURCE_PATCH_1 if "repair" in actions[0]["command"] else SOURCE_PATCH_0
        return [], {"submission": submission, "tool_call_id": "call"}

    def format_observation_messages(self, message, outputs):
        del message
        return [{"role": "tool", "content": outputs[-1]["output"], "tool_call_id": "call"}]

    def serialize(self):
        return {"messages": self.messages}


def test_masked_suffix_is_all_or_nothing_at_context_limit():
    sample = Sample(tokens=[1], response_length=0, loss_mask=[], rollout_log_probs=[])
    agent = SimpleNamespace(
        messages=[
            {"role": "assistant", "content": "answer"},
            {"role": "tool", "content": "a long observation"},
        ]
    )

    complete = asyncio.run(
        rollout._append_messages(
            SimpleNamespace(),
            sample,
            _FakeTokenizer(),
            agent,
            1,
            max_all_tokens=2,
        )
    )

    assert complete is False
    assert sample.tokens == [1]
    assert sample.response_length == 0


def test_generate_assigns_zero_reward_after_final_environment_failure(monkeypatch):
    monkeypatch.setenv("SWE_MAX_ENV_RETRIES", "1")
    monkeypatch.setenv("SWE_REPAIR_ZERO_REWARD", "0.0")
    monkeypatch.setattr(
        rollout,
        "_generate_once",
        AsyncMock(side_effect=ModuleNotFoundError("No module named 'client'")),
    )
    monkeypatch.setattr(
        rollout,
        "GenerateState",
        lambda args: SimpleNamespace(tokenizer=_FakeTokenizer()),
    )
    sample = Sample(index=11, prompt="fix it", metadata={"test_patch": TEST_PATCH})

    result = asyncio.run(rollout.generate(SimpleNamespace(), sample, {}))

    assert result.status == Sample.Status.ABORTED
    assert result.reward is None
    assert result.tokens
    assert result.loss_mask == []
    assert "ModuleNotFoundError" in result.metadata["f2p_repair_error"]


def test_generate_skips_missing_oracle_before_environment_retry(monkeypatch):
    generate_once = AsyncMock()
    monkeypatch.setattr(rollout, "_generate_once", generate_once)
    monkeypatch.setattr(
        rollout,
        "GenerateState",
        lambda args: SimpleNamespace(tokenizer=_FakeTokenizer()),
    )
    sample = Sample(
        index=12,
        prompt="fix it",
        metadata={
            "instance_id": "projectmesa_mesa_pr1667",
            "FAIL_TO_PASS": ["test_fail_to_pass.py::test_fix"],
            "test_patch": "",
        },
    )

    result = asyncio.run(rollout.generate(SimpleNamespace(), sample, {}))

    assert result.status == Sample.Status.ABORTED
    assert result.reward is None
    assert result.tokens
    assert result.loss_mask == []
    assert "MissingF2POracle" in result.metadata["f2p_repair_error"]
    generate_once.assert_not_awaited()


def test_mask_aborted_keeps_group_and_zeros_only_aborted_loss():
    normal = Sample(response_length=2, loss_mask=[1, 1], status=Sample.Status.COMPLETED)
    aborted = Sample(response_length=3, loss_mask=[1, 0, 1], status=Sample.Status.ABORTED)

    decision = mask_aborted(SimpleNamespace(), [normal, aborted])

    assert decision.keep is True
    assert normal.remove_sample is False
    assert normal.loss_mask == [1, 1]
    assert aborted.remove_sample is True
    assert aborted.loss_mask == [0, 0, 0]


@pytest.mark.parametrize("submission_mode", ["explicit", "none", "forced", "forced_default"])
def test_generate_keeps_round0_and_repair_in_one_trainable_sample(monkeypatch, tmp_path, submission_mode):
    explicit_submission = submission_mode == "explicit"
    forced_submission = submission_mode in {"forced", "forced_default"}
    completed = submission_mode != "none"
    monkeypatch.setenv("SWE_CONFIG_PATH", "unused.yaml")
    monkeypatch.setenv("SWE_MAX_ENV_RETRIES", "1")
    monkeypatch.setenv("SWE_MAX_CREATE_ENV_RETRIES", "1")
    monkeypatch.setenv("SWE_ENV_CREATE_JITTER_MAX", "0")
    monkeypatch.setenv("SWE_F2P_REPAIR_ROUND0_STEPS", "200" if explicit_submission else "1")
    monkeypatch.setenv("SWE_F2P_REPAIR_STEPS_PER_ROUND", "30" if explicit_submission else "1")
    monkeypatch.setenv("SWE_F2P_REPAIR_MAX_ROUNDS", "5")
    if submission_mode == "forced_default":
        monkeypatch.delenv("SWE_F2P_REPAIR_FORCE_SUBMIT_ON_TURN_LIMIT", raising=False)
    else:
        monkeypatch.setenv("SWE_F2P_REPAIR_FORCE_SUBMIT_ON_TURN_LIMIT", "1" if forced_submission else "0")

    state = SimpleNamespace(tokenizer=_FakeTokenizer(), rollout_id=9)
    monkeypatch.setattr(rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(
        rollout,
        "load_config",
        lambda path: {
            "agent": {
                "system_template": "system",
                "instance_template": "{{task}}",
                "observation_template": "{{output.output}}",
                "format_error_template": "{{error}}",
                "timeout_template": "timeout",
                "step_limit": 200,
                "cost_limit": 3.0,
                "time_limit": 1500,
                "max_all_tokens": 65536,
            },
            "environment": {},
        },
    )
    monkeypatch.setattr(rollout, "SWEAgentV2", _FakeAgent)
    create_environment = AsyncMock(side_effect=[SimpleNamespace(name="workspace")])
    stop_environment = AsyncMock()
    monkeypatch.setattr(rollout, "create_environment", create_environment)
    monkeypatch.setattr(rollout, "stop_environment", stop_environment)
    monkeypatch.setattr(rollout, "_current_diff", AsyncMock(return_value=SOURCE_PATCH_1))
    monkeypatch.setattr(rollout, "_trajectory_root", lambda: tmp_path)

    verb = "submit" if explicit_submission else "work"
    outputs = iter(
        [
            {
                "text": f"round0 {verb}",
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [[-0.1, 10]],
                    "top_p_token_ids": [10, 110],
                    "top_p_token_offsets": [0, 2],
                },
            },
            {
                "text": f"repair {verb}",
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [[-0.2, 11]],
                    "top_p_token_ids": [11, 111],
                    "top_p_token_offsets": [0, 2],
                },
            },
        ]
    )

    async def fake_post(url, payload):
        del url, payload
        return next(outputs)

    check_results = []
    if explicit_submission:
        check_results.extend(
            [
                {"passed": False, "feedback": "expected 3 got 4", "patch": SOURCE_PATCH_0},
                {"passed": True, "feedback": "passed", "patch": SOURCE_PATCH_1},
            ]
        )
    elif forced_submission:
        check_results.append({"passed": True, "feedback": "passed", "patch": SOURCE_PATCH_1})
    checks = iter(check_results)
    checked_submissions = []

    class FakeVerifySession:
        def __init__(self, instance, **kwargs):
            del instance, kwargs

        async def initialize(self, setup_timeout):
            del setup_timeout

        async def check_f2p(self, submitted_patch, nodes):
            del nodes
            checked_submissions.append(submitted_patch)
            return next(checks)

        async def run_final_official(self, patch):
            assert patch == SOURCE_PATCH_1
            return {
                "resolved": True,
                "resolution": "RESOLVED_FULL",
                "f2p_passed": 1,
                "f2p_total": 1,
                "p2p_passed": 0,
                "p2p_total": 0,
                "error": "",
            }

        async def close(self):
            return None

    monkeypatch.setattr(rollout, "post", fake_post)
    monkeypatch.setattr(rollout, "OfficialVerifySession", FakeVerifySession)

    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_seed=42,
        rollout_top_p=0.95,
    )
    sample = Sample(
        index=4,
        group_index=1,
        prompt="fix it",
        metadata={
            "instance_id": "repo__case-1",
            "problem_statement": "fix it",
            "repo": "owner/repo",
            "FAIL_TO_PASS": ["tests/test_core.py::test_hidden"],
            "PASS_TO_PASS": [],
            "test_patch": TEST_PATCH,
        },
    )

    result = asyncio.run(
        rollout.generate(
            args,
            sample,
            {"max_new_tokens": 4096, "temperature": 0.95, "top_p": 0.95},
        )
    )

    assert result.status == (Sample.Status.COMPLETED if completed else Sample.Status.TRUNCATED)
    assert result.metadata["final_output"] == (SOURCE_PATCH_1 if completed else "")
    assert result.metadata["f2p_repair"]["f2p_passed"] is completed
    assert result.metadata["f2p_repair"]["had_submission"] is completed
    assert result.metadata["f2p_repair"]["force_submit_on_turn_limit"] is forced_submission
    assert result.metadata["f2p_repair"]["round0_turns"] == 1
    assert result.metadata["f2p_repair"]["repair_turns"] == 1
    assert result.metadata["f2p_repair"]["repair_rounds_started"] == 1
    assert result.metadata["f2p_repair"]["repair_rounds_used"] == 1
    expected_submissions = [SOURCE_PATCH_0, SOURCE_PATCH_1] if explicit_submission else []
    if forced_submission:
        expected_submissions = [SOURCE_PATCH_1]
    assert checked_submissions == expected_submissions
    assert result.metadata["f2p_repair"]["repair_rounds"][0]["forced_submission"] is forced_submission
    if completed:
        assert result.metadata["f2p_repair"]["f2p_checks"][-1]["forced_submission"] is forced_submission
    assert create_environment.await_count == 1
    assert stop_environment.await_count == 1
    assert result.metadata["f2p_repair"]["persistent_verifier"] is True
    assert result.metadata["f2p_repair"]["verify_environment_reused_for_official"] is completed
    if completed:
        assert result.metadata["f2p_repair"]["official_verify"]["resolved"] is True
    assert sum(result.loss_mask) == 2
    assert result.response_length == len(result.loss_mask) == len(result.rollout_log_probs)
    assert len(result.rollout_top_p_token_offsets) == result.response_length + 1
    assert int(result.rollout_top_p_token_offsets[-1]) == len(result.rollout_top_p_token_ids)
    torch.testing.assert_close(
        result.rollout_top_p_token_ids,
        torch.tensor([10, 110, 11, 111], dtype=torch.int32),
    )
    expected_feedback = "expected 3 got 4" if explicit_submission else "verifier has not tested"
    assert any(
        message.get("role") == "tool" and expected_feedback in message.get("content", "")
        for message in result.metadata["trajectory"]["messages"]
    )
    assert sum(
        message.get("role") == "user" for message in result.metadata["trajectory"]["messages"]
    ) == 1


def test_append_rollout_metrics_reports_repair_population():
    samples = [
        Sample(
            metadata={
                "f2p_repair": {
                    "repair_rounds_started": 2,
                    "repair_turns": 7,
                    "f2p_passed": True,
                }
            }
        ),
        Sample(
            metadata={
                "f2p_repair": {
                    "repair_rounds_started": 4,
                    "repair_turns": 11,
                    "f2p_passed": False,
                }
            }
        ),
        Sample(
            metadata={
                "f2p_repair": {
                    "repair_rounds_started": 0,
                    "repair_turns": 0,
                    "f2p_passed": True,
                }
            }
        ),
    ]
    extra_metrics = {}

    skip_default_logger = rollout.append_rollout_metrics(
        3, SimpleNamespace(), samples, extra_metrics, 55.0
    )

    assert skip_default_logger is False
    assert extra_metrics == {
        "f2p_repair/entered_samples": 2,
        "f2p_repair/repair_rounds_mean": 3.0,
        "f2p_repair/success_rate": 0.5,
        "f2p_repair/repair_turns_total": 9.0,
    }
