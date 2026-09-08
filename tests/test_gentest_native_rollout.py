import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.mini_swe import gentest_native_rollout
from examples.mini_swe.scripts import validate_official_f2p_gold
from slime.utils.types import Sample

NUM_GPUS = 0


def test_instance_id_rejects_trajectory_path_traversal():
    with pytest.raises(ValueError, match="unsafe instance_id"):
        gentest_native_rollout._instance_from_sample(
            Sample(index=1, metadata={"instance_id": "../../outside", "problem_statement": "bug"})
        )


def test_patch_classification_reward_uses_live_rollout_environment(monkeypatch):
    environment = object()
    expected_reward = {
        "base_reward": 1.0,
        "gold_reward": 1.0,
        "patch_cls_reward": 1.0,
        "raw_reward": 1.0,
    }
    calls = []

    def fake_reward(args, sample, env):
        calls.append((args, sample, env))
        return expected_reward

    monkeypatch.setattr(gentest_native_rollout, "compute_patch_classification_reward_in_env", fake_reward)
    args = SimpleNamespace(
        custom_rm_path=gentest_native_rollout.PATCH_CLASSIFICATION_REWARD_PATH,
        swe_reward_mode="patch_classification",
    )
    sample = Sample(metadata={"format_error_count": 0})
    sample.status = Sample.Status.COMPLETED

    reused = asyncio.run(
        gentest_native_rollout._compute_reward_before_environment_cleanup(
            args,
            sample,
            environment,
            evaluation=False,
        )
    )

    assert reused is True
    assert calls == [(args, sample, environment)]
    assert sample.reward == expected_reward


def test_patch_classification_reward_skips_format_error_sample(monkeypatch):
    monkeypatch.setattr(
        gentest_native_rollout,
        "compute_patch_classification_reward_in_env",
        lambda *_args, **_kwargs: pytest.fail("format-error sample must not run patch classification"),
    )
    args = SimpleNamespace(
        custom_rm_path=gentest_native_rollout.PATCH_CLASSIFICATION_REWARD_PATH,
        swe_reward_mode="patch_classification",
    )
    sample = Sample(metadata={"format_error_count": 1})
    sample.status = Sample.Status.COMPLETED

    reused = asyncio.run(
        gentest_native_rollout._compute_reward_before_environment_cleanup(
            args,
            sample,
            object(),
            evaluation=False,
        )
    )

    assert reused is False
    assert sample.reward is None


def test_patch_classification_reward_timeout_drains_worker_and_fails_closed(monkeypatch):
    worker_started = asyncio.Event()
    release_worker = asyncio.Event()

    async def delayed_run_blocking(function, *args, **kwargs):
        worker_started.set()
        await release_worker.wait()
        return function(*args, **kwargs)

    monkeypatch.setattr(gentest_native_rollout, "_run_blocking", delayed_run_blocking)
    monkeypatch.setattr(gentest_native_rollout, "SWE_TIMEOUT_REWARD_TOTAL", 0.01)
    monkeypatch.setattr(
        gentest_native_rollout,
        "compute_patch_classification_reward_in_env",
        lambda *_args, **_kwargs: {"raw_reward": 1.0},
    )
    args = SimpleNamespace(
        custom_rm_path=gentest_native_rollout.PATCH_CLASSIFICATION_REWARD_PATH,
        swe_reward_mode="patch_classification",
        reward_key="raw_reward",
    )
    sample = Sample(
        index=7,
        response_length=3,
        loss_mask=[1, 1, 1],
        metadata={"format_error_count": 0, "gentest_record": {"instance_id": "repo__issue-7"}},
    )
    sample.status = Sample.Status.COMPLETED

    async def exercise():
        task = asyncio.create_task(
            gentest_native_rollout._compute_reward_before_environment_cleanup(
                args,
                sample,
                object(),
                evaluation=False,
            )
        )
        await worker_started.wait()
        await asyncio.sleep(0.02)
        assert not task.done(), "timeout must wait for the environment-using worker to drain"
        release_worker.set()
        assert await task is True

    asyncio.run(exercise())

    assert sample.status == Sample.Status.ABORTED
    assert sample.remove_sample is True
    assert sample.loss_mask == [0, 0, 0]
    assert sample.reward == {"raw_reward": 0.0}
    assert sample.metadata["gentest_record"]["reward_timeout"] is True
    assert sample.metadata["gentest_record"]["infra_failure"] is True


def test_append_rollout_metrics_restores_gentest_wandb_series():
    samples = [
        Sample(
            metadata={
                "gentest_record": {
                    "agent_exit_status": "Submitted",
                    "agent_calls": 4,
                    "format_error_count": 0,
                    "no_tool_error_count": 0,
                    "validation": {"label": "gold_validated"},
                    "timings": {"environment_create_sec": 10.0, "reward_sec": 20.0, "total_sec": 40.0},
                }
            }
        ),
        Sample(
            metadata={
                "gentest_record": {
                    "agent_exit_status": "LimitsExceeded",
                    "agent_calls": 6,
                    "format_error_count": 1,
                    "no_tool_error_count": 1,
                    "validation": {"label": "gold_failed"},
                    "timings": {"environment_create_sec": 14.0, "reward_sec": 24.0, "total_sec": 50.0},
                }
            }
        ),
    ]
    extra_metrics = {}

    skip_default_logger = gentest_native_rollout.append_rollout_metrics(
        3, SimpleNamespace(), samples, extra_metrics, 55.0
    )

    assert skip_default_logger is False
    assert extra_metrics["gentest_native/num_samples"] == 2
    assert extra_metrics["gentest_native/status/Submitted"] == 1
    assert extra_metrics["gentest_native/status/LimitsExceeded"] == 1
    assert extra_metrics["gentest_native/label/gold_validated"] == 1
    assert extra_metrics["gentest_native/agent_calls_mean"] == 5.0
    assert extra_metrics["gentest_native/format_error_sample_rate"] == 0.5
    assert extra_metrics["gentest_native/timing/environment_create_sec_mean"] == 12.0
    assert extra_metrics["gentest_native/timing/reward_sec_max"] == 24.0
    assert extra_metrics["gentest_native/timing/total_sec_max"] == 50.0


def test_rollout_metrics_count_sandbox_infra_exit_as_environment_error():
    sample = Sample(
        metadata={
            "gentest_record": {
                "agent_exit_status": "sandbox_infra_error",
                "sandbox_infra_error": True,
                "validation": {"label": "sandbox_infra_error", "infra_failure": True},
            }
        },
        remove_sample=True,
    )

    metrics = gentest_native_rollout._gentest_rollout_metrics([sample], 1.0)

    assert metrics["gentest_native/env_error_count"] == 1
    assert metrics["gentest_native/status/sandbox_infra_error"] == 1


def test_cancelled_environment_creation_waits_for_cleanup(monkeypatch):
    class FakeEnvironment:
        cleaned = False

        def cleanup(self):
            self.cleaned = True

    environment = FakeEnvironment()

    async def fake_run_blocking(function, *_args, **_kwargs):
        if function is gentest_native_rollout.get_gentest_environment:
            await asyncio.sleep(0.02)
            return environment
        function()

    async def cancel_create():
        task = asyncio.create_task(
            gentest_native_rollout._create_environment({}, {"instance_id": "repo__issue-1"})
        )
        await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    monkeypatch.setattr(gentest_native_rollout, "_run_blocking", fake_run_blocking)
    asyncio.run(cancel_create())
    assert environment.cleaned is True


def test_isolated_environments_are_created_concurrently(monkeypatch):
    started = 0
    both_started = asyncio.Event()
    release = asyncio.Event()
    seen_configs = []
    timings = {}

    async def fake_create(config, _instance):
        nonlocal started
        seen_configs.append(config)
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()
        return config["role"]

    async def create_both():
        task = asyncio.create_task(
            gentest_native_rollout._create_isolated_environments(
                {"role": "workspace"},
                {"role": "verifier"},
                {"instance_id": "repo__issue-1"},
                timings,
            )
        )
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()
        return await task

    monkeypatch.setattr(gentest_native_rollout, "_create_environment", fake_create)

    assert asyncio.run(create_both()) == ("workspace", "verifier")
    assert seen_configs == [{"role": "workspace"}, {"role": "verifier"}]
    assert timings["workspace_environment_create_sec"] >= 0
    assert timings["verify_environment_create_sec"] >= 0
    assert timings["environment_create_sec"] >= 0


def test_parallel_environment_creation_cleans_successful_sibling_on_failure(monkeypatch):
    class FakeEnvironment:
        cleaned = False

        def cleanup(self):
            self.cleaned = True

    environment = FakeEnvironment()
    calls = 0

    async def fake_create(_config, _instance):
        nonlocal calls
        calls += 1
        if calls == 1:
            return environment
        raise RuntimeError("verifier creation failed")

    async def fake_run_blocking(function, *_args, **_kwargs):
        function()

    monkeypatch.setattr(gentest_native_rollout, "_create_environment", fake_create)
    monkeypatch.setattr(gentest_native_rollout, "_run_blocking", fake_run_blocking)

    with pytest.raises(RuntimeError, match="verifier creation failed"):
        asyncio.run(
            gentest_native_rollout._create_isolated_environments(
                {"role": "workspace"},
                {"role": "verifier"},
                {"instance_id": "repo__issue-1"},
                {},
            )
        )
    assert environment.cleaned is True


def test_official_verifier_config_is_independent_and_allows_network(monkeypatch):
    monkeypatch.setattr(
        gentest_native_rollout,
        "build_config",
        lambda *_args, **_kwargs: {
            "environment": {
                "environment_class": "azure_modal",
                "block_network": True,
                "cpu": "2",
            }
        },
    )
    monkeypatch.setattr(
        gentest_native_rollout,
        "_resolve_per_instance_api_base",
        lambda _config, _instance_id: None,
    )
    monkeypatch.delenv("GENTEST_VERIFY_ENV_CPU", raising=False)

    config = gentest_native_rollout._load_official_verifier_config("repo__issue-1")

    assert config["environment"]["environment_class"] == "azure_modal"
    assert config["environment"]["block_network"] is False
    assert config["environment"]["cpu"] == "2"


def test_native_v6_rollout_uses_agent_execution_contract_end_to_end(monkeypatch, tmp_path):
    selected_path = "tests/generated/test_issue.py"
    selected_command = "python -m pytest tests/generated/test_issue.py"
    test_code = "def test_regression():\n    assert False\n"
    captured = {"official": [], "validation": []}

    class FakeState:
        tokenizer = object()
        rollout_id = 7

        def __init__(self, _args):
            pass

    class FakeEnvironment:
        def cleanup(self):
            pass

    class FakeModel:
        timings = {}
        _last_structured_write_record_idx = None

        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    class FakeAgent:
        n_calls = 1
        cost = 0.0
        gate_info = {"self_test_attempts": []}
        messages = []

        def __init__(self, _model, _env, *, self_test_runner, **kwargs):
            assert kwargs["allowed_generated_paths"] == []
            assert "self_test_command" not in kwargs
            self.self_test_runner = self_test_runner

        def run(self, _task, **kwargs):
            assert kwargs["test_style_guidance"] == ""
            self.self_test_runner(test_code, selected_path, selected_command)
            return {
                "exit_status": "Submitted",
                "submission": test_code,
                "gentest_gate": {
                    "test_code": test_code,
                    "test_file": selected_path,
                    "test_command": selected_command,
                    "command_evidence": "pyproject.toml and adjacent tests",
                    "oracle_quality": {"contract": {}},
                    "latest_self_test": {},
                },
            }

        def save(self, *_args, **_kwargs):
            pass

    async def run_blocking(function, *args, **kwargs):
        return function(*args, **kwargs)

    async def create_environments(*_args, **_kwargs):
        return FakeEnvironment(), FakeEnvironment()

    async def skip_reward(*_args, **_kwargs):
        return False

    def official(_env, _instance, filename, code, _timeout, **kwargs):
        captured["official"].append((filename, code, kwargs.get("test_command")))
        return {"clean_fail": True, "test_hash": "unused"}

    def validate(_env, _instance, code, **kwargs):
        captured["validation"].append((code, kwargs))
        return {"label": "gold_validated", "base_clean_fail": True, "gold_pass": True}

    for legacy_name in (
        "generated_test_file_for_instance",
        "starter_code_for_sample",
        "test_command_for_instance",
        "test_style_guidance_for_instance",
        "write_generated_test_file",
    ):
        monkeypatch.setattr(
            gentest_native_rollout,
            legacy_name,
            lambda *_args, _name=legacy_name, **_kwargs: pytest.fail(f"legacy resolver used: {_name}"),
        )
    monkeypatch.setenv("GENTEST_TRAJECTORY_DIR", str(tmp_path))
    monkeypatch.setenv("GENTEST_PROMPT_MODE", "zero_shot")
    monkeypatch.setenv("GENTEST_EXAMPLE_SHOT", "none")
    monkeypatch.setenv("GENTEST_STARTER", "empty")
    monkeypatch.setenv("GENTEST_APPLY_TEST_PATCH_AT_START", "0")
    monkeypatch.setenv("GENTEST_VERIFY_ASSUME_PREPARED", "1")
    monkeypatch.setattr(gentest_native_rollout, "GenerateState", FakeState)
    monkeypatch.setattr(
        gentest_native_rollout,
        "_load_gentest_config",
        lambda _instance_id: {
            "agent": {"require_execution_contract": True},
            "model": {},
            "environment": {},
        },
    )
    monkeypatch.setattr(
        gentest_native_rollout,
        "_load_official_verifier_config",
        lambda _instance_id: {"environment": {}},
    )
    monkeypatch.setattr(gentest_native_rollout, "_create_isolated_environments", create_environments)
    monkeypatch.setattr(gentest_native_rollout, "_run_blocking", run_blocking)
    monkeypatch.setattr(gentest_native_rollout, "_SGLangGentestModel", FakeModel)
    monkeypatch.setattr(gentest_native_rollout, "GeneratedTestSubmitAgent", FakeAgent)
    monkeypatch.setattr(gentest_native_rollout, "reset_base", lambda _env: None)
    monkeypatch.setattr(gentest_native_rollout, "run_generated_test_official", official)
    monkeypatch.setattr(gentest_native_rollout, "validate_generated_test", validate)
    monkeypatch.setattr(
        gentest_native_rollout,
        "_build_rollout_tensors",
        lambda *_args: ([1], "prompt", [2], [1], [-0.1], None, "response"),
    )
    monkeypatch.setattr(gentest_native_rollout, "_compute_reward_before_environment_cleanup", skip_reward)
    monkeypatch.setattr(gentest_native_rollout, "evaluate_oracle_contract", lambda *_args: {})

    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=1024,
        use_rollout_routing_replay=False,
        num_layers=1,
        moe_router_topk=1,
        num_experts=1,
    )
    sample = Sample(
        index=0,
        prompt="bug",
        metadata={"instance_id": "repo__issue-1", "problem_statement": "bug"},
    )

    result = asyncio.run(
        gentest_native_rollout._generate_once(args, sample, {}, resource_level=1)
    )

    assert result.status == Sample.Status.COMPLETED
    assert result.metadata["test_filename"] == selected_path
    assert result.metadata["generated_test_command"] == selected_command
    assert captured["official"] == [(selected_path, test_code, selected_command)]
    assert captured["validation"][0][1]["filename"] == selected_path
    assert captured["validation"][0][1]["test_command"] == selected_command


def test_official_f2p_smoke_prepares_once_and_uses_canonical_evaluator(monkeypatch):
    events = []

    class FakeEnvironment:
        def cleanup(self):
            events.append("cleanup")

    environment = FakeEnvironment()
    monkeypatch.setattr(
        validate_official_f2p_gold,
        "_load_official_env_config",
        lambda _args: (
            {"environment_class": "azure_modal", "block_network": False},
            ["official.yaml"],
        ),
    )
    monkeypatch.setattr(
        validate_official_f2p_gold,
        "create_environment",
        lambda _instance, _config: events.append("create") or environment,
    )
    monkeypatch.setattr(
        validate_official_f2p_gold,
        "prepare_environment_for_evaluation",
        lambda env, _instance, timeout: events.append(("prepare", env, timeout))
        or {"returncode": 0},
    )

    def evaluate(_instance, patch, env, _config, *, environment_prepared):
        events.append(("evaluate", patch, env, environment_prepared))
        if patch:
            return {
                "resolved": True,
                "patch_applied": True,
                "parsed_tests_count": 20,
                "fail_to_pass_passed": ["test_regression"],
                "fail_to_pass_failed": [],
                "error": "",
            }
        return {
            "resolved": False,
            "patch_applied": True,
            "parsed_tests_count": 20,
            "fail_to_pass_passed": [],
            "fail_to_pass_failed": ["test_regression"],
            "error": "",
        }

    monkeypatch.setattr(
        validate_official_f2p_gold,
        "evaluate_instance_in_environment",
        evaluate,
    )
    args = SimpleNamespace(config=None, install_timeout=600, test_timeout=240)
    row = {
        "metadata": {
            "instance_id": "repo__issue-1",
            "FAIL_TO_PASS": ["test_regression"],
            "PASS_TO_PASS": ["test_existing"],
            "patch": "gold patch",
            "install_config": {"test_cmd": "pytest", "log_parser": "pytest"},
        }
    }

    result = validate_official_f2p_gold._run_one((0, row, args))

    assert result["validation_label"] == "gold_validated"
    assert result["base_clean_fail"] is True
    assert result["gold_pass"] is True
    assert result["official_verifier"]["block_network"] is False
    assert [event[0] for event in events if isinstance(event, tuple)] == [
        "prepare",
        "evaluate",
        "evaluate",
    ]
    assert all(event[3] is True for event in events if event[0] == "evaluate")
    assert events[-1] == "cleanup"


def test_sglang_gentest_model_requests_and_records_routed_experts(monkeypatch):
    class _Tokenizer:
        eos_token = None

        def __call__(self, _text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [11, 12]}

    model = gentest_native_rollout._SGLangGentestModel(
        {"extra_tools": []},
        tokenizer=_Tokenizer(),
        url="http://unused/generate",
        sampling_params={"max_new_tokens": 4},
        llm_timeout=1,
        max_llm_attempts=1,
        max_all_tokens=32,
        use_rollout_routing_replay=True,
        num_layers=2,
        moe_router_topk=2,
        num_experts=8,
    )
    captured_payload = {}

    def fake_post(payload):
        captured_payload.update(payload)
        return {
            "text": "answer",
            "meta_info": {
                "output_token_logprobs": [[-0.25, 13]],
                "routed_experts": list(range(8)),
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(gentest_native_rollout, "_apply_chat_template", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(model, "_post_generate", fake_post)
    monkeypatch.setattr(model, "parse_text_actions", lambda *_args: ([], []))

    try:
        message = model.query([{"role": "user", "content": "question"}])
    finally:
        model.close()

    assert captured_payload["input_ids"] == [11, 12]
    assert "text" not in captured_payload
    assert captured_payload["sampling_params"] == {"max_new_tokens": 4}
    assert captured_payload["return_routed_experts"] is True
    assert model.records[0]["routed_experts"].shape == (2, 2, 2)
    assert model.records[0]["response_length_after"] == 1
    assert model.timings["generation_calls"] == 1
    assert model.timings["generation_sec"] >= 0.0
    assert model.timings["chat_template_sec"] >= 0.0
    assert model.timings["tokenization_sec"] >= 0.0
    assert message["content"] == "answer"
    rollout_tensors = gentest_native_rollout._build_rollout_tensors(model)
    assert rollout_tensors[0] == [11, 12]
    assert rollout_tensors[2] == [13]
    assert rollout_tensors[5].shape == (2, 2, 2)


def test_sglang_gentest_model_abort_check_stops_before_request(monkeypatch):
    model = gentest_native_rollout._SGLangGentestModel(
        {"extra_tools": []},
        tokenizer=object(),
        url="http://unused/generate",
        sampling_params={"max_new_tokens": 4},
        llm_timeout=1,
        max_llm_attempts=1,
        max_all_tokens=32,
        use_rollout_routing_replay=False,
        num_layers=0,
        moe_router_topk=0,
        num_experts=0,
        abort_check=lambda: True,
    )
    monkeypatch.setattr(
        model,
        "_post_generate",
        lambda _payload: pytest.fail("aborted resolve sample must not issue another model request"),
    )

    try:
        with pytest.raises(RuntimeError, match="Rollout generation aborted"):
            model.query([])
    finally:
        model.close()


def test_sglang_gentest_model_abort_check_stops_retry(monkeypatch):
    aborted = False
    post_calls = 0
    sleep_calls = 0
    model = gentest_native_rollout._SGLangGentestModel(
        {"extra_tools": []},
        tokenizer=object(),
        url="http://unused/generate",
        sampling_params={"max_new_tokens": 4},
        llm_timeout=1,
        max_llm_attempts=2,
        max_all_tokens=32,
        use_rollout_routing_replay=False,
        num_layers=0,
        moe_router_topk=0,
        num_experts=0,
        abort_check=lambda: aborted,
    )

    def fail_post(*_args, **_kwargs):
        nonlocal aborted, post_calls
        post_calls += 1
        aborted = True
        raise ConnectionError("request failed during cutoff")

    def fake_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1

    monkeypatch.setattr(model._client, "post", fail_post)
    monkeypatch.setattr(gentest_native_rollout.time, "sleep", fake_sleep)

    try:
        with pytest.raises(RuntimeError, match="Rollout generation aborted"):
            model._post_generate({"input_ids": [1]})
    finally:
        model.close()

    assert post_calls == 1
    assert sleep_calls == 0


def test_sglang_gentest_model_abort_check_stops_after_response(monkeypatch):
    aborted = False

    class Tokenizer:
        def __call__(self, _text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [1]}

    model = gentest_native_rollout._SGLangGentestModel(
        {"extra_tools": []},
        tokenizer=Tokenizer(),
        url="http://unused/generate",
        sampling_params={"max_new_tokens": 4},
        llm_timeout=1,
        max_llm_attempts=1,
        max_all_tokens=32,
        use_rollout_routing_replay=False,
        num_layers=0,
        moe_router_topk=0,
        num_experts=0,
        abort_check=lambda: aborted,
    )

    def return_after_abort(_payload):
        nonlocal aborted
        aborted = True
        return {"text": "must not be processed", "meta_info": {}}

    monkeypatch.setattr(gentest_native_rollout, "_apply_chat_template", lambda *_args, **_kwargs: "prompt")
    monkeypatch.setattr(model, "_post_generate", return_after_abort)
    monkeypatch.setattr(
        model,
        "parse_text_actions",
        lambda *_args: pytest.fail("response processing must stop after cutoff"),
    )

    try:
        with pytest.raises(RuntimeError, match="Rollout generation aborted"):
            model.query([{"role": "user", "content": "question"}])
    finally:
        model.close()


def test_sglang_gentest_model_parses_multiple_tool_calls_in_one_turn():
    response = """Inspect independent paths.
<tool_call>
<function=bash>
<parameter=command>pwd</parameter>
</function>
</tool_call>
<tool_call>
<function=bash>
<parameter=command>git status --short</parameter>
</function>
</tool_call>"""
    model = gentest_native_rollout._SGLangGentestModel(
        {"extra_tools": []},
        tokenizer=object(),
        url="http://unused/generate",
        sampling_params={"max_new_tokens": 128},
        llm_timeout=1,
        max_llm_attempts=1,
        max_all_tokens=256,
        use_rollout_routing_replay=False,
        num_layers=0,
        moe_router_topk=0,
        num_experts=0,
    )

    try:
        actions, tool_calls = model.parse_text_actions(response, "call_7")
    finally:
        model.close()

    assert [(action["tool"], action["command"]) for action in actions] == [
        ("bash", "pwd"),
        ("bash", "git status --short"),
    ]
    assert [action["tool_call_id"] for action in actions] == ["call_7_0", "call_7_1"]
    assert [tool_call["id"] for tool_call in tool_calls] == ["call_7_0", "call_7_1"]
    assert [tool_call["function"]["name"] for tool_call in tool_calls] == ["bash", "bash"]


def test_multi_turn_gentest_preserves_generated_ids_and_only_tokenizes_new_suffix(monkeypatch):
    raw_tool_call = '\n<tool_call>{"name":"write_generated_test","arguments":{"test_code":"x"}}</tool_call>'
    suffix = "<assistant_end><tool_observation><assistant_start>"

    class _Tokenizer:
        eos_token = None

        def __init__(self):
            self.tokenized_texts = []

        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            self.tokenized_texts.append(text)
            if text == "prompt":
                return {"input_ids": [11, 12]}
            if text == suffix:
                return {"input_ids": [90, 91]}
            if text == raw_tool_call:
                pytest.fail("generated model text must never be re-tokenized")
            pytest.fail(f"unexpected tokenizer input: {text!r}")

    tokenizer = _Tokenizer()
    model = gentest_native_rollout._SGLangGentestModel(
        {"extra_tools": []},
        tokenizer=tokenizer,
        url="http://unused/generate",
        sampling_params={"max_new_tokens": 4},
        llm_timeout=1,
        max_llm_attempts=1,
        max_all_tokens=32,
        use_rollout_routing_replay=True,
        num_layers=1,
        moe_router_topk=1,
        num_experts=8,
    )
    payloads = []
    outputs = iter(
        [
            {
                "text": raw_tool_call,
                "meta_info": {
                    "output_token_logprobs": [[-0.1, 20], [-0.2, 21], [-0.3, 22]],
                    "routed_experts": list(range(4)),
                    "finish_reason": {"type": "stop"},
                },
            },
            {
                "text": "done",
                "meta_info": {
                    "output_token_logprobs": [[-0.4, 30]],
                    "routed_experts": list(range(7)),
                    "finish_reason": {"type": "stop"},
                },
            },
        ]
    )
    parsed_actions = iter(
        [
            (
                [{"tool": "write_generated_test", "test_code": "x"}],
                [
                    {
                        "id": "call_1_0",
                        "type": "function",
                        "function": {"name": "write_generated_test", "arguments": '{"test_code":"x"}'},
                    }
                ],
            ),
            ([], []),
        ]
    )

    def fake_post(payload):
        payloads.append(payload)
        return next(outputs)

    def fake_apply_chat_template(_tokenizer, messages, **_kwargs):
        if len(messages) == 1:
            return "prompt"
        boundary = messages[1]["content"]
        assert boundary.startswith("__SLIME_GENTEST_ASSISTANT_BOUNDARY_0_")
        # Simulate a real chat template that trims/normalizes the historical
        # response, so its full rendering is not an append-only text prefix.
        return "rerendered prompt with normalized assistant content" + boundary + suffix

    monkeypatch.setattr(gentest_native_rollout, "_apply_chat_template", fake_apply_chat_template)
    monkeypatch.setattr(model, "_post_generate", fake_post)
    monkeypatch.setattr(model, "parse_text_actions", lambda *_args: next(parsed_actions))

    try:
        first_message = model.query([{"role": "user", "content": "question"}])
        second_message = model.query(
            [
                {"role": "user", "content": "question"},
                first_message,
                {"role": "tool", "tool_call_id": "call_1_0", "content": "observation"},
            ]
        )
    finally:
        model.close()

    assert first_message["content"] == ""
    assert first_message["extra"]["raw_response"] == raw_tool_call
    assert first_message["tool_calls"][0]["function"]["name"] == "write_generated_test"
    assert second_message["content"] == "done"
    assert tokenizer.tokenized_texts == ["prompt", suffix]
    assert payloads[0]["input_ids"] == [11, 12]
    assert payloads[1]["input_ids"] == [11, 12, 20, 21, 22, 90, 91]
    assert all("text" not in payload for payload in payloads)

    rollout_tensors = gentest_native_rollout._build_rollout_tensors(model)
    assert rollout_tensors[0] == [11, 12]
    assert rollout_tensors[2] == [20, 21, 22, 90, 91, 30]
    assert rollout_tensors[3] == [1, 1, 1, 0, 0, 1]
    assert rollout_tensors[4] == [-0.1, -0.2, -0.3, 0.0, 0.0, -0.4]
    assert rollout_tensors[6] == raw_tool_call + suffix + "done"
    torch.testing.assert_close(rollout_tensors[5], torch.arange(7, dtype=torch.int32).reshape(7, 1, 1))

    truncated_tensors = gentest_native_rollout._build_rollout_tensors(
        model, model._last_structured_write_record_idx
    )
    assert truncated_tensors[2] == [20, 21, 22]
    assert truncated_tensors[3] == [1, 1, 1]
    assert truncated_tensors[4] == [-0.1, -0.2, -0.3]
    assert truncated_tensors[6] == raw_tool_call
    torch.testing.assert_close(truncated_tensors[5], torch.arange(4, dtype=torch.int32).reshape(4, 1, 1))


def test_multi_turn_gentest_does_not_duplicate_sampled_assistant_terminator(monkeypatch):
    raw_tool_call = '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call><|im_end|>'

    class _Tokenizer:
        eos_token = "<|im_end|>"
        eos_token_id = 99

        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            if text == "prompt":
                return {"input_ids": [11]}
            if text == "\n<tool_observation>ok</tool_observation>":
                return {"input_ids": [90]}
            pytest.fail(f"unexpected tokenizer input: {text!r}")

    model = gentest_native_rollout._SGLangGentestModel(
        {"extra_tools": []},
        tokenizer=_Tokenizer(),
        url="http://unused/generate",
        sampling_params={"max_new_tokens": 4},
        llm_timeout=1,
        max_llm_attempts=1,
        max_all_tokens=32,
        use_rollout_routing_replay=False,
        num_layers=0,
        moe_router_topk=0,
        num_experts=0,
    )
    outputs = iter(
        [
            {
                "text": raw_tool_call,
                "meta_info": {
                    "output_token_logprobs": [[-0.1, 20], [-0.2, 99]],
                    "finish_reason": {"type": "stop"},
                },
            },
            {
                "text": "done",
                "meta_info": {
                    "output_token_logprobs": [[-0.3, 30]],
                    "finish_reason": {"type": "stop"},
                },
            },
        ]
    )
    parsed_actions = iter(
        [
            (
                [{"tool": "bash", "command": "pwd"}],
                [
                    {
                        "id": "call_1_0",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                    }
                ],
            ),
            ([], []),
        ]
    )

    def fake_apply_chat_template(_tokenizer, messages, **_kwargs):
        if len(messages) == 1:
            return "prompt"
        boundary = messages[1]["content"]
        return boundary + "<|im_end|>\n<tool_observation>ok</tool_observation>"

    monkeypatch.setattr(gentest_native_rollout, "_apply_chat_template", fake_apply_chat_template)
    monkeypatch.setattr(model, "_post_generate", lambda _payload: next(outputs))
    monkeypatch.setattr(model, "parse_text_actions", lambda *_args: next(parsed_actions))

    try:
        first_message = model.query([{"role": "user", "content": "question"}])
        model.query(
            [
                {"role": "user", "content": "question"},
                first_message,
                {"role": "tool", "tool_call_id": "call_1_0", "content": "ok"},
            ]
        )
    finally:
        model.close()

    rollout_tensors = gentest_native_rollout._build_rollout_tensors(model)
    assert rollout_tensors[2] == [20, 99, 90, 30]
    assert rollout_tensors[3] == [1, 1, 0, 1]
    assert rollout_tensors[6] == raw_tool_call + "\n<tool_observation>ok</tool_observation>done"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
