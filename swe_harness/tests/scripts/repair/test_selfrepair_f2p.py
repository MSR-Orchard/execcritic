import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts" / "repair" / "selfrepair_f2p.py"
SPEC = importlib.util.spec_from_file_location("selfrepair_f2p_test_module", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_trapi_config_drops_unsupported_temperature():
    config = MODULE.build_config(
        "unused",
        "EMPTY",
        "http://sandbox",
        "sandbox-key",
        0.6,
        model_name="gpt-5.6-sol_2026-07-09",
        model_class="trapi_response",
        top_p=0.95,
        reasoning_effort="medium",
    )

    assert "temperature" not in config["model"]["model_kwargs"]
    assert "top_p" not in config["model"]["model_kwargs"]
    assert config["model"]["model_kwargs"]["max_output_tokens"] == 8192
    assert config["model"]["model_kwargs"]["reasoning"] == {"effort": "medium", "summary": "auto"}


def test_litellm_config_sets_top_p():
    config = MODULE.build_config(
        "http://localhost:8000/v1",
        "EMPTY",
        "http://sandbox",
        "sandbox-key",
        0.95,
        model_name="openai/qwen",
        model_class="litellm",
        top_p=0.95,
    )

    assert config["model"]["model_kwargs"]["temperature"] == 0.95
    assert config["model"]["model_kwargs"]["top_p"] == 0.95


def test_chat_seed_preserves_paired_tools_and_drops_dangling_calls(tmp_path):
    trajectory = tmp_path / "case.traj.json"
    trajectory.write_text(json.dumps({"messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "issue"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "paired", "type": "function", "function": {"name": "bash", "arguments": {"command": "pwd"}}},
        ]},
        {"role": "tool", "tool_call_id": "paired", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "dangling", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
        ]},
    ]}))

    seed = MODULE.load_seed_messages(trajectory, seed_format="chat")

    assert [message["role"] for message in seed] == ["system", "user", "assistant", "tool"]
    assert seed[2]["tool_calls"][0]["id"] == "paired"
    assert seed[2]["tool_calls"][0]["function"]["arguments"] == '{"command": "pwd"}'
    assert seed[3]["tool_call_id"] == "paired"


def test_responses_seed_remains_responses_items(tmp_path):
    trajectory = tmp_path / "case.traj.json"
    trajectory.write_text(json.dumps({"messages": [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ]}))

    seed = MODULE.load_seed_messages(trajectory)

    assert [item["type"] for item in seed] == ["function_call", "function_call_output"]


def test_live_trapi_seed_flattens_response_and_drops_dangling_submit():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "issue"},
        {"object": "response", "output": [
            {"type": "reasoning", "id": "reasoning-1"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "inspect"}]},
            {"type": "function_call", "call_id": "paired", "name": "bash", "arguments": '{"command":"pwd"}'},
        ]},
        {"type": "function_call_output", "call_id": "paired", "output": "ok"},
        {"object": "response", "output": [
            {"type": "function_call", "call_id": "dangling-submit", "name": "bash", "arguments": "{}"},
        ]},
        {"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}},
    ]

    seed = MODULE.live_seed_messages(messages)

    assert seed[:2] == messages[:2]
    assert [item.get("type") for item in seed[2:]] == ["message", "function_call", "function_call_output"]
    assert {item.get("call_id") for item in seed} == {None, "paired"}
    assert all(item.get("type") != "reasoning" for item in seed)


def test_workspace_replay_extracts_all_supported_action_schemas_once():
    messages = [
        {
            "role": "assistant",
            "extra": {"actions": [{"command": "echo from-extra"}]},
            "tool_calls": [
                {
                    "function": {
                        "arguments": json.dumps({"command": "echo duplicate"})
                    }
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "arguments": json.dumps({"command": "echo from-chat"})
                    }
                }
            ],
        },
        {
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "arguments": json.dumps({"command": "echo from-response"}),
                }
            ],
        },
        {
            "type": "function_call",
            "arguments": {"command": "echo from-item"},
        },
        {
            "role": "assistant",
            "extra": {
                "actions": [
                    {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}
                ]
            },
        },
    ]

    assert MODULE.rca.extract_replay_commands(messages) == [
        "echo from-extra",
        "echo from-chat",
        "echo from-response",
        "echo from-item",
    ]


def test_strict_workspace_replay_rejects_missing_submitted_patch(monkeypatch):
    monkeypatch.setattr(
        MODULE.rca,
        "replay_trajectory",
        lambda *_args, **_kwargs: {
            "total": 2,
            "ran": 2,
            "failed": 0,
            "truncated": False,
            "seconds": 1.0,
        },
    )
    monkeypatch.setattr(
        MODULE.rca,
        "_submitted_patch_is_present",
        lambda *_args: (False, "reverse check failed"),
    )
    monkeypatch.setattr(
        MODULE.rca,
        "apply_prior_patch",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("strict replay must not fall back to patch-only restoration")
        ),
    )

    applied, error, stats = MODULE.rca.restore_state(
        object(),
        [{"role": "assistant"}],
        "diff --git a/pkg.py b/pkg.py\n",
        SimpleNamespace(
            restore_mode="replay",
            replay_cmd_timeout=30,
            replay_budget=60,
        ),
        strict_replay=True,
    )

    assert applied is False
    assert "did not reproduce the submitted patch" in error
    assert stats["prior_patch_present"] is False


def test_provided_patch_mode_can_restore_workspace_by_strict_replay(monkeypatch, tmp_path):
    patch = (
        "diff --git a/pkg.py b/pkg.py\n"
        "--- a/pkg.py\n"
        "+++ b/pkg.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    source_messages = [
        {
            "role": "assistant",
            "extra": {"actions": [{"command": "python -m pip install helper"}]},
        }
    ]
    captured = {}

    class FakeEnvironment:
        def cleanup(self):
            captured["cleaned"] = True

    def fake_restore(_env, messages, prior_patch, args, *, strict_replay):
        captured.update(
            messages=messages,
            prior_patch=prior_patch,
            restore_mode=args.restore_mode,
            strict_replay=strict_replay,
        )
        return True, "", {
            "total": 1,
            "ran": 1,
            "failed": 0,
            "truncated": False,
            "prior_patch_present": True,
        }

    monkeypatch.setattr(MODULE, "restore_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE.rca, "get_model", lambda config: object())
    monkeypatch.setattr(
        MODULE.rca.swerebench_runner,
        "_resolve_per_instance_api_base",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        MODULE.rca.swerebench_runner,
        "get_sb_environment",
        lambda *_args: FakeEnvironment(),
    )
    monkeypatch.setattr(MODULE.rca, "restore_state", fake_restore)
    monkeypatch.setattr(
        MODULE.rca,
        "shell",
        lambda _env, command, **_kwargs: {
            "returncode": 0,
            "output": patch if command.endswith("git diff") else "",
        },
    )
    monkeypatch.setattr(
        MODULE,
        "f2p_check",
        lambda *_args, **_kwargs: (
            True,
            "passed",
            {"selected_statuses": {"tests/test_pkg.py::test_fix": "PASSED"}},
        ),
    )
    monkeypatch.setattr(
        MODULE.official_f2p_gate,
        "PersistentOfficialGate",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    args = SimpleNamespace(
        seed="test",
        trajectory_dir=tmp_path / "trajs",
        install_timeout=30,
        test_timeout=30,
        max_rounds=2,
        steps=10,
        n_f2p=1,
        fresh_solve_round0=False,
        seed_trajectory=False,
        restore_mode="replay",
        replay_cmd_timeout=30,
        replay_budget=60,
        full_eval_script_gate=True,
    )
    instance = {
        "instance_id": "example__case-1",
        "problem_statement": "Fix the package bug",
        "FAIL_TO_PASS": ["tests/test_pkg.py::test_fix"],
        "test_patch": "diff --git a/tests/test_pkg.py b/tests/test_pkg.py\n",
    }
    config = {
        "agent": {"system_template": "system", "instance_template": "{{task}}"},
        "model": {"model_class": "litellm"},
    }

    result = MODULE.process_one(
        instance["instance_id"],
        instance,
        [patch],
        [],
        config,
        args,
        source_messages,
    )

    assert result["workspace_restore_mode"] == "replay"
    assert result["workspace_replay"]["prior_patch_present"] is True
    assert result["wrong_patch_applied"] is True
    assert result["initial_pass"] is True
    assert captured["messages"] == source_messages
    assert captured["prior_patch"] == patch
    assert captured["strict_replay"] is True
    assert captured["cleaned"] is True


def test_fresh_round0_trajectory_is_used_as_repair_seed(monkeypatch, tmp_path):
    patch = "diff --git a/pkg.py b/pkg.py\n--- a/pkg.py\n+++ b/pkg.py\n@@ -1 +1 @@\n-old\n+new\n"
    captured = {}

    class FakeEnvironment:
        def cleanup(self):
            captured["cleaned"] = True

    class FakeSolveAgent:
        def __init__(self, _model, _env, **kwargs):
            self.output_path = kwargs["output_path"]
            self.n_calls = 0
            self.messages = []

        def run(self, task):
            captured["round0_task"] = task
            self.n_calls = 3
            self.messages = [
                {"role": "system", "content": "system"},
                {"role": "user", "content": task},
                {"object": "response", "output": [
                    {"type": "function_call", "call_id": "paired", "name": "bash", "arguments": "{}"},
                ]},
                {"type": "function_call_output", "call_id": "paired", "output": "edited"},
                {"object": "response", "output": [
                    {"type": "function_call", "call_id": "submit", "name": "bash", "arguments": "{}"},
                ]},
                {"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}},
            ]
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(json.dumps({"messages": self.messages}))
            return {"exit_status": "Submitted", "submission": patch}

    class FakeRepairAgent:
        def __init__(self, _model, _env, *, seed_messages, on_submit, **_kwargs):
            captured["repair_seed"] = seed_messages
            self._on_submit = on_submit
            self.n_calls = 0
            self.rounds = []
            self.passed = False
            self.stop_reason = ""

        def run_seeded(self, prompt, **_kwargs):
            captured["repair_prompt"] = prompt
            passed, _feedback, round_patch, gate_metadata = self._on_submit(patch)
            self.n_calls = 2
            self.rounds = [{
                "round": 1,
                "turns": 2,
                "passed": passed,
                "patch": round_patch,
                "gate_metadata": gate_metadata,
            }]
            self.passed = passed
            self.stop_reason = "passed" if passed else "max_rounds"

    monkeypatch.setattr(MODULE, "DefaultAgent", FakeSolveAgent)
    monkeypatch.setattr(MODULE, "SeedRepairAgent", FakeRepairAgent)
    monkeypatch.setattr(MODULE, "restore_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE.rca, "get_model", lambda config: object())
    monkeypatch.setattr(MODULE.rca.swerebench_runner, "_resolve_per_instance_api_base", lambda *_args: None)
    monkeypatch.setattr(MODULE.rca.swerebench_runner, "get_sb_environment", lambda *_args: FakeEnvironment())
    monkeypatch.setattr(
        MODULE.rca,
        "apply_prior_patch",
        lambda *_args: (_ for _ in ()).throw(AssertionError("old input patch must not be applied")),
    )
    monkeypatch.setattr(
        MODULE.rca,
        "shell",
        lambda _env, command, **_kwargs: {"returncode": 0, "output": patch if command.endswith("git diff") else ""},
    )
    checks = iter([
        (False, "first failure", {"selected_statuses": {"tests/test_pkg.py::test_fix": "FAILED"}}),
        (True, "passed", {"selected_statuses": {"tests/test_pkg.py::test_fix": "PASSED"}}),
    ])
    def fake_f2p_check(*_args, **kwargs):
        captured.setdefault("checked_submissions", []).append(kwargs.get("submitted_patch"))
        return next(checks)

    monkeypatch.setattr(MODULE, "f2p_check", fake_f2p_check)
    monkeypatch.setattr(
        MODULE.official_f2p_gate,
        "PersistentOfficialGate",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    args = SimpleNamespace(
        seed="test",
        trajectory_dir=tmp_path / "trajs",
        install_timeout=30,
        test_timeout=30,
        max_rounds=2,
        steps=10,
        n_f2p=1,
        fresh_solve_round0=True,
        round0_steps=20,
        round0_cost_limit=5.0,
        seed_trajectory=False,
    )
    instance = {
        "instance_id": "example__case-1",
        "problem_statement": "Fix the package bug",
        "FAIL_TO_PASS": ["tests/test_pkg.py::test_fix"],
        "test_patch": "diff --git a/tests/test_pkg.py b/tests/test_pkg.py\n",
    }
    config = {
        "agent": {"system_template": "system", "instance_template": "{{task}}"},
        "model": {"model_class": "trapi_response"},
    }

    result = MODULE.process_one(instance["instance_id"], instance, ["old seed patch"], [], config, args)

    assert result["round0_mode"] == "fresh_e2e_solve"
    assert result["seed_origin"] == "fresh_round0_e2e_trajectory"
    assert result["round0_turns"] == 3
    assert result["repair_turns"] == 2
    assert result["total_turns"] == 5
    assert result["initial_pass"] is False
    assert result["rescued"] is True
    assert result["final_patch"] == patch
    assert captured["round0_task"] == instance["problem_statement"]
    assert [item.get("call_id") for item in captured["repair_seed"] if item.get("call_id")] == ["paired", "paired"]
    assert "submit" not in json.dumps(captured["repair_seed"])
    assert captured["checked_submissions"] == [patch, patch]
    assert captured["cleaned"] is True


def test_django_invocation_targets_exact_test_label():
    instance = {
        "repo": "django/django",
        "test_patch": "diff --git a/tests/forms_tests/tests/test_validators.py b/tests/forms_tests/tests/test_validators.py\n",
    }

    command = MODULE._build_test_invocation(
        instance,
        ["test_value_placeholder_with_decimal_field (forms_tests.tests.test_validators.ValidatorCustomMessageTests)"],
    )

    assert command == (
        "./tests/runtests.py --verbosity 2 "
        "forms_tests.tests.test_validators.ValidatorCustomMessageTests.test_value_placeholder_with_decimal_field"
    )


def test_pytest_invocation_targets_node_and_feature_detects_no_header():
    node = "testing/test_conftest.py::test_setinitial_conftest_subdirs[tests]"

    command = MODULE._build_test_invocation({"repo": "pytest-dev/pytest", "test_patch": ""}, [node])

    assert "pytest --help" in command
    assert "PYTEST_NO_HEADER" in command
    assert "--no-header" not in command.split("python -m pytest $PYTEST_NO_HEADER", 1)[1]
    assert repr(node) in command or node in command


def test_focused_feedback_keeps_selected_assertion_ahead_of_teardown():
    node = "test_example (example.tests.ExampleTests)"
    output = "\n".join(
        [
            f"FAIL: {node}",
            "Traceback (most recent call last):",
            "AssertionError: expected 3 but got 4",
            *[f"Destroying test database for alias 'db_{index}'" for index in range(500)],
        ]
    )

    feedback = MODULE._focused_test_feedback(output, [node], {node: "FAILED"}, 1)

    assert len(feedback) <= 6000
    assert f"- {node}: FAILED" in feedback
    assert "AssertionError: expected 3 but got 4" in feedback


def test_f2p_check_preserves_test_return_code_and_writes_raw_log(monkeypatch, tmp_path):
    commands = []
    node = "tests/test_example.py::test_failure"

    def fake_shell(_env, command, timeout=120):
        commands.append(command)
        if "Start Test Output" in command:
            return {
                "returncode": 0,
                "output": (
                    ">>>>> Start Test Output\n"
                    f"FAILED {node} - AssertionError: expected 1\n"
                    ">>>>> End Test Output\n"
                    "__RC__=1\n"
                ),
            }
        return {"returncode": 0, "output": ""}

    monkeypatch.setattr(MODULE.rca, "shell", fake_shell)
    monkeypatch.setattr(MODULE.swebench_verify, "_parse_test_output", lambda _output, _instance: {node: "FAILED"})
    log_path = tmp_path / "round.test.log"
    instance = {
        "repo": "example/example",
        "test_patch": "diff --git a/tests/test_example.py b/tests/test_example.py\n",
    }

    passed, feedback, metadata = MODULE.f2p_check(
        object(),
        instance,
        [node],
        30,
        log_path,
        full_eval_script_gate=False,
    )

    assert passed is False
    assert metadata["return_code"] == 1
    assert metadata["selected_statuses"] == {node: "FAILED"}
    assert "test_rc=$?" in next(command for command in commands if "Start Test Output" in command)
    assert "AssertionError: expected 1" in feedback
    assert "return_code: 1" in log_path.read_text()


def test_f2p_check_tests_submitted_patch_and_leaves_it_in_worktree(monkeypatch):
    node = "tests/test_example.py::test_fix"
    submitted_patch = (
        "diff --git a/pkg.py b/pkg.py\n"
        "--- a/pkg.py\n"
        "+++ b/pkg.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+fixed\n"
    )
    state = {"source": False, "hidden": False}
    commands = []

    def fake_shell(_env, command, timeout=120):
        del timeout
        commands.append(command)
        if "git reset --hard" in command:
            state.update(source=False, hidden=False)
        elif "git apply --check" in command and "/tmp/submitted.patch" in command:
            pass
        elif "git apply --whitespace=nowarn /tmp/submitted.patch" in command:
            state["source"] = True
        elif "git -c core.fileMode=false diff HEAD --binary" in command:
            return {"returncode": 0, "output": submitted_patch if state["source"] else ""}
        elif "git apply --whitespace=nowarn /tmp/test_patch.diff" in command:
            state["hidden"] = True
        elif "git apply -R" in command and "/tmp/test_patch.diff" in command:
            state["hidden"] = False
        elif "Start Test Output" in command:
            assert state == {"source": True, "hidden": True}
            return {
                "returncode": 0,
                "output": (
                    f">>>>> Start Test Output\nPASSED {node}\n"
                    ">>>>> End Test Output\n__RC__=0\n"
                ),
            }
        return {"returncode": 0, "output": ""}

    monkeypatch.setattr(MODULE.rca, "shell", fake_shell)
    monkeypatch.setattr(
        MODULE.swebench_verify,
        "_parse_test_output",
        lambda _output, _instance: {node: "PASSED"},
    )

    passed, _feedback, metadata = MODULE.f2p_check(
        object(),
        {
            "base_commit": "abc123",
            "repo": "example/example",
            "test_patch": "diff --git a/tests/test_example.py b/tests/test_example.py\n",
        },
        [node],
        30,
        submitted_patch=submitted_patch,
        full_eval_script_gate=False,
    )

    assert passed is True
    assert state == {"source": True, "hidden": False}
    assert metadata["submitted_patch_sha256"]
    assert metadata["gate_patch_sha256"]
    reset_index = next(i for i, command in enumerate(commands) if "git reset --hard abc123" in command)
    apply_index = next(
        i
        for i, command in enumerate(commands)
        if "git apply --whitespace=nowarn /tmp/submitted.patch" in command
    )
    test_index = next(i for i, command in enumerate(commands) if "Start Test Output" in command)
    assert reset_index < apply_index < test_index


def test_full_eval_script_gate_uses_persistent_official_verifier_without_text_blocking(
    monkeypatch, tmp_path
):
    node = "Migration directories without an __init__.py file are loaded."
    patch = (
        "diff --git a/django/core.py b/django/core.py\n"
        "--- a/django/core.py\n"
        "+++ b/django/core.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    captured = {}

    def fake_evaluate(
        submitted_patch,
        *,
        fail_to_pass,
        pass_to_pass,
        timeout,
    ):
        captured.update(
            submitted_patch=submitted_patch,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            timeout=timeout,
        )
        return {
            "resolved": True,
            "exit_code": 0,
            "patch_applied": True,
            "fail_to_pass_passed": [node],
            "fail_to_pass_failed": [],
            "parsed_tests_count": 1,
            "error": "",
            "output": (
                "ImportError is expected text inside this passing regression test\n"
                f"{node} ... ok\n"
            ),
            "official_verifier": "swebench",
            "test_command": "./tests/runtests.py --verbosity 2 tests.case.Test.test_fix",
            "test_files": ["tests/test_case.py"],
            "test_runner": "django runtests.py",
        }

    gate = SimpleNamespace(evaluate=fake_evaluate)
    log_path = tmp_path / "focused.log"
    instance = {
        "instance_id": "django__django-1",
        "repo": "django/django",
        "test_patch": "diff --git a/tests/test_case.py b/tests/test_case.py\n",
        "FAIL_TO_PASS": [node, "another test"],
        "PASS_TO_PASS": ["existing behavior"],
    }

    passed, feedback, metadata = MODULE.f2p_check(
        object(),
        instance,
        [node],
        240,
        log_path,
        submitted_patch=patch,
        official_gate_session=gate,
    )

    assert passed is True
    assert captured["fail_to_pass"] == [node]
    assert captured["pass_to_pass"] == []
    assert captured["submitted_patch"] == patch
    assert captured["timeout"] == 240
    assert metadata["command"].startswith("./tests/runtests.py")
    assert metadata["official_verifier"] == "swebench"
    assert metadata["selected_statuses"] == {node: "PASSED"}
    assert "ImportError is expected text" in log_path.read_text()
    assert node in feedback
    assert "test_execution_contract:" in feedback
    assert "- language: python" in feedback
    assert "- runner: django runtests.py" in feedback
    assert '- test_files: ["tests/test_case.py"]' in feedback
    assert "- test_source_visibility: hidden_verifier_only" in feedback


def test_full_eval_script_gate_never_passes_when_submission_did_not_apply(monkeypatch):
    gate = SimpleNamespace(
        evaluate=lambda *_args, **_kwargs: {
            "resolved": True,
            "patch_applied": False,
            "exit_code": 0,
            "fail_to_pass_passed": ["tests/test_pkg.py::test_fix"],
            "fail_to_pass_failed": [],
            "parsed_tests_count": 1,
            "error": "",
            "output": "PASSED tests/test_pkg.py::test_fix",
            "official_verifier": "swebench",
        }
    )

    passed, feedback, metadata = MODULE.f2p_check(
        object(),
        {
            "instance_id": "owner__repo-1",
            "test_patch": "diff --git a/tests/test_pkg.py b/tests/test_pkg.py\n",
        },
        ["tests/test_pkg.py::test_fix"],
        240,
        submitted_patch="diff --git a/pkg.py b/pkg.py\n",
        official_gate_session=gate,
    )

    assert passed is False
    assert "SUBMITTED_PATCH_APPLY_FAILED" in feedback
    assert metadata["gate_patch_sha256"] == ""


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], True), (["--no-full-eval-script-gate"], False)],
)
def test_cli_full_eval_script_gate_default_and_opt_out(monkeypatch, extra_args, expected):
    original_parse_args = MODULE.argparse.ArgumentParser.parse_args

    class ParsedArguments(Exception):
        pass

    def parse_and_stop(parser):
        args = original_parse_args(parser)
        assert args.full_eval_script_gate is expected
        assert args.force_submit_on_turn_limit is True
        assert args.max_gate_checks == 0
        assert args.continue_on_duplicate_patch is False
        raise ParsedArguments

    monkeypatch.setattr(MODULE.argparse.ArgumentParser, "parse_args", parse_and_stop)
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        [
            "selfrepair_f2p.py",
            "--ids",
            "ids.json",
            "--source-run",
            "source",
            "--out",
            "output.jsonl",
            *extra_args,
        ],
    )

    with pytest.raises(ParsedArguments):
        MODULE.main()


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], True), (["--no-force-submit-on-turn-limit"], False)],
)
def test_cli_force_submit_on_turn_limit_default_and_opt_out(monkeypatch, extra_args, expected):
    original_parse_args = MODULE.argparse.ArgumentParser.parse_args

    class ParsedArguments(Exception):
        pass

    def parse_and_stop(parser):
        args = original_parse_args(parser)
        assert args.force_submit_on_turn_limit is expected
        raise ParsedArguments

    monkeypatch.setattr(MODULE.argparse.ArgumentParser, "parse_args", parse_and_stop)
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        [
            "selfrepair_f2p.py",
            "--ids",
            "ids.json",
            "--source-run",
            "source",
            "--out",
            "output.jsonl",
            *extra_args,
        ],
    )

    with pytest.raises(ParsedArguments):
        MODULE.main()


def test_f2p_check_rejects_unapplicable_submission_before_test(monkeypatch):
    commands = []

    def fake_shell(_env, command, timeout=120):
        del timeout
        commands.append(command)
        if "git apply --check" in command and "/tmp/submitted.patch" in command:
            return {"returncode": 1, "output": "error: patch does not apply"}
        return {"returncode": 0, "output": ""}

    monkeypatch.setattr(MODULE.rca, "shell", fake_shell)

    passed, feedback, metadata = MODULE.f2p_check(
        object(),
        {"base_commit": "abc123", "test_patch": "test patch"},
        ["tests/test_example.py::test_fix"],
        30,
        submitted_patch="broken patch",
        full_eval_script_gate=False,
    )

    assert passed is False
    assert metadata["error"] == "SUBMITTED_PATCH_DOES_NOT_APPLY"
    assert "patch does not apply" in feedback
    assert not any("Start Test Output" in command for command in commands)


def test_seed_repair_agent_passes_submitted_payload_to_gate(monkeypatch):
    submitted_patch = "diff --git a/pkg.py b/pkg.py\n"
    captured = {}

    def raise_submission(_self):
        raise MODULE.Submitted(
            {
                "role": "exit",
                "content": submitted_patch,
                "extra": {"exit_status": "Submitted", "submission": submitted_patch},
            }
        )

    monkeypatch.setattr(MODULE.DefaultAgent, "step", raise_submission)
    agent = object.__new__(MODULE.SeedRepairAgent)
    agent._on_submit = lambda patch: (
        captured.setdefault("patch", patch) is not None,
        "passed",
        patch,
    )
    agent._max_rounds = 1
    agent._steps_per_round = 40
    agent._force_submit_on_turn_limit = True
    agent._round_start_calls = 0
    agent.rounds = []
    agent.passed = False
    agent.final_test_result = None
    agent.stop_reason = ""
    agent.n_calls = 1

    with pytest.raises(MODULE.Submitted):
        agent.step()

    assert captured["patch"] == submitted_patch
    assert agent.rounds[0]["forced_submission"] is False


def test_seed_repair_agent_force_submits_worktree_diff_at_round_limit(monkeypatch):
    forced_patch = "diff --git a/pkg.py b/pkg.py\n--- a/pkg.py\n+++ b/pkg.py\n"
    captured = {}

    monkeypatch.setattr(
        MODULE.DefaultAgent,
        "step",
        lambda _self: (_ for _ in ()).throw(AssertionError("turn limit must submit before another LLM call")),
    )

    def fake_shell(_env, command, **_kwargs):
        captured["command"] = command
        return {"returncode": 0, "output": forced_patch}

    monkeypatch.setattr(MODULE.rca, "shell", fake_shell)
    agent = object.__new__(MODULE.SeedRepairAgent)
    agent.env = object()
    agent._on_submit = lambda patch: (
        captured.setdefault("patch", patch) is not None,
        "passed",
        patch,
    )
    agent._max_rounds = 1
    agent._steps_per_round = 2
    agent._force_submit_on_turn_limit = True
    agent._round_start_calls = 0
    agent.rounds = []
    agent.passed = False
    agent.final_test_result = None
    agent.stop_reason = ""
    agent.n_calls = 2

    with pytest.raises(MODULE.Submitted):
        agent.step()

    assert captured["patch"] == forced_patch
    assert captured["command"] == "cd /testbed && git -c core.fileMode=false diff HEAD --binary"
    assert agent.rounds == [
        {
            "round": 1,
            "turns": 2,
            "passed": True,
            "patch": forced_patch,
            "forced_submission": True,
        }
    ]


def _bare_repair_agent(*, max_gate_checks=10, continue_on_duplicate_patch=True):
    agent = object.__new__(MODULE.SeedRepairAgent)
    agent._max_rounds = 5
    agent._max_gate_checks = max_gate_checks
    agent._gate_limit_stop_reason = "max_gate_checks"
    agent._continue_on_duplicate_patch = continue_on_duplicate_patch
    agent._round_start_calls = 0
    agent.rounds = []
    agent.messages = []
    agent.passed = False
    agent.final_test_result = None
    agent.stop_reason = ""
    agent.duplicate_submissions = 0
    agent.n_calls = 0
    agent.model = SimpleNamespace(
        format_message=lambda *, role, content: {"role": role, "content": content}
    )
    agent.add_messages = lambda *messages: agent.messages.extend(messages) or list(messages)
    return agent


def test_duplicate_patch_is_non_terminal_and_does_not_rerun_gate():
    checked = []
    agent = _bare_repair_agent()
    agent._on_submit = lambda patch: (
        checked.append(patch) is None and False,
        "still failing",
        patch,
        {"gate_status": "hard_fail"},
    )
    patch = "diff --git a/pkg.py b/pkg.py\n--- a/pkg.py\n+++ b/pkg.py\n"

    agent.n_calls = 1
    agent._handle_submission(patch, forced_submission=False)
    agent.n_calls = 2
    agent._handle_submission(patch, forced_submission=False)

    assert checked == [patch]
    assert len(agent.rounds) == 1
    assert agent.duplicate_submissions == 1
    assert agent.stop_reason == ""
    assert MODULE.DUPLICATE_PATCH_FOLLOWUP in agent.messages[-1]["content"]


def test_ten_gate_checks_can_continue_past_legacy_five_with_same_turn_counter():
    checked = []
    agent = _bare_repair_agent(max_gate_checks=10)
    agent._on_submit = lambda patch: (
        checked.append(patch) is None and False,
        "still failing",
        patch,
        {"gate_status": "hard_fail"},
    )

    for gate_index in range(6):
        agent.n_calls = (gate_index + 1) * 10
        agent._handle_submission(f"patch-{gate_index}", forced_submission=False)

    assert len(checked) == 6
    assert len(agent.rounds) == 6
    assert agent.stop_reason == ""
    assert agent.n_calls == 60
    assert agent.n_calls < 5 * 30
