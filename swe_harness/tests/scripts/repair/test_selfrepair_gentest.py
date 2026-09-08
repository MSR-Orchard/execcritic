import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


NUM_GPUS = 0


SCRIPT = Path(__file__).parents[3] / "scripts" / "repair" / "cache" / "selfrepair_gentest.py"
SPEC = importlib.util.spec_from_file_location("selfrepair_gentest_test_module", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_trapi_config_sets_medium_reasoning():
    config = MODULE.build_config(
        "unused",
        "EMPTY",
        "http://sandbox",
        "sandbox-key",
        0.6,
        model_name="gpt-5.3-codex_2026-02-24",
        model_class="trapi_response",
        reasoning_effort="medium",
    )

    assert config["model"]["model_kwargs"]["reasoning"] == {"effort": "medium", "summary": "auto"}


def test_string_test_entry_reconstructs_astropy_path_and_runner():
    instance = {
        "repo": "astropy/astropy",
        "version": "4.3",
        "test_patch": "diff --git a/astropy/units/tests/test_quantity.py b/astropy/units/tests/test_quantity.py\n"
        "--- a/astropy/units/tests/test_quantity.py\n"
        "+++ b/astropy/units/tests/test_quantity.py\n",
        "FAIL_TO_PASS": ["astropy/units/tests/test_quantity.py::test_example"],
    }

    code, filename, command = MODULE._test_entry_parts("def test_example(): pass", 0, instance)

    assert code == "def test_example(): pass"
    assert filename == "astropy/units/tests/test_model_gen.py"
    assert "pytest" in command
    assert "astropy/units/tests/test_model_gen.py" in command


def test_string_test_entry_reconstructs_django_module_runner():
    instance = {
        "repo": "django/django",
        "version": "2.2",
        "test_patch": "diff --git a/tests/admin_docs/tests.py b/tests/admin_docs/tests.py\n"
        "--- a/tests/admin_docs/tests.py\n"
        "+++ b/tests/admin_docs/tests.py\n",
        "FAIL_TO_PASS": ["test_example (admin_docs.tests.AdminDocsTests.test_example)"],
    }

    _, filename, command = MODULE._test_entry_parts("class Test: pass", 0, instance)

    assert filename == "tests/admin_docs/test_model_gen.py"
    assert "tests/runtests.py" in command
    assert "admin_docs.test_model_gen" in command


def test_source_diff_excludes_alignment_test_paths(monkeypatch):
    commands = []

    def fake_shell(_env, command, timeout=120):
        commands.append(command)
        return {"output": "diff --git a/pkg/source.py b/pkg/source.py\n@@ -1 +1 @@\n-old\n+new\n"}

    monkeypatch.setattr(MODULE.rca, "shell", fake_shell)

    diff = MODULE._source_diff(object(), ["tests/example_test.py"])

    assert diff.startswith("diff --git")
    assert ":(exclude)tests/example_test.py" in commands[0]


def test_feedback_exposes_hidden_python_runner_contract():
    feedback = MODULE._feedback_from_test_result({
        "passed": False,
        "gate_status": "hard_fail",
        "gate_reason": "assertion_failure",
        "n_tests": 1,
        "failed_test_names": ["tests/test_model_gen.py::test_fix"],
        "tests": [{
            "index": 0,
            "filename": "tests/test_model_gen.py",
            "runner_command": "cd /testbed && python -m pytest tests/test_model_gen.py",
            "rc": 1,
            "gate_status": "hard_fail",
            "failure_kind": "assertion_failure",
        }],
        "output_tail": "assert 1 == 2",
    })

    assert "test_execution_contract:" in feedback
    assert "language: python" in feedback
    assert "working_directory: /testbed" in feedback
    assert '"test_filename": "tests/test_model_gen.py"' in feedback
    assert '"runner_command": "cd /testbed && python -m pytest tests/test_model_gen.py"' in feedback
    assert "test_source_visibility: hidden_verifier_only" in feedback


def _test_patch_entry():
    return {
        "test_patch": (
            "diff --git a/tests/example.py b/tests/example.py\n"
            "index 1111111..2222222 100644\n"
            "--- a/tests/example.py\n"
            "+++ b/tests/example.py\n"
            "@@ -1,2 +1,5 @@\n"
            " existing = True\n"
            "+\n"
            "+def test_regression():\n"
            "+    assert True\n"
            " tail_context = True\n"
        ),
        "test_command": "cd /testbed && python -m pytest tests/example.py::test_regression",
    }


def test_testpatch_check_applies_runs_and_restores_hidden_patch(monkeypatch):
    commands = []

    def fake_shell(_env, command, timeout=120):
        commands.append(command)
        if "python -m pytest" in command:
            return {"output": "1 passed in 0.01s\n__RC__=0\n", "returncode": 0}
        return {"output": "", "returncode": 0}

    monkeypatch.setattr(MODULE.rca, "shell", fake_shell)

    result = MODULE.generated_test_check(
        object(),
        {"repo": "django/django", "version": "2.2"},
        [_test_patch_entry()],
        240,
    )

    assert result["passed"] is True
    assert result["gate_status"] == "pass"
    assert result["tests"][0]["test_patch"] is True
    assert result["tests"][0]["runner_command"].endswith(
        "python -m pytest tests/example.py::test_regression"
    )
    assert any("git apply --check" in command for command in commands)
    assert any("git apply --reverse --check" in command for command in commands)
    assert sum("git status --porcelain" in command for command in commands) >= 2


def test_testpatch_check_supports_behavior_contract_existing_node_updates(monkeypatch):
    commands = []
    events = []
    entry = {
        "test_patch": (
            "diff --git a/tests/utils_tests/test_html.py b/tests/utils_tests/test_html.py\n"
            "index 1111111..2222222 100644\n"
            "--- a/tests/utils_tests/test_html.py\n"
            "+++ b/tests/utils_tests/test_html.py\n"
            "@@ -27,7 +27,7 @@ class TestUtilsHtml(SimpleTestCase):\n"
            "-            (\"'\", '&#39;'),\n"
            "+            (\"'\", '&#x27;'),\n"
        ),
        "test_command": (
            "python tests/runtests.py utils_tests.test_html.TestUtilsHtml.test_escape"
        ),
    }

    def fake_shell(_env, command, timeout=120):
        commands.append(command)
        if "git apply --reverse --check" in command:
            events.append("restore")
        elif "git apply --check" in command:
            events.append("apply")
        elif entry["test_command"] in command:
            events.append("declared")
            return {"output": "1 passed in 0.01s\n__RC__=0\n", "returncode": 0}
        elif "runtests.py" in command:
            pytest.fail(f"unexpected test command: {command}")
        return {"output": "", "returncode": 0}

    monkeypatch.setattr(MODULE.rca, "shell", fake_shell)
    monkeypatch.setattr(
        MODULE.gentest_runner,
        "test_command_for_instance",
        lambda *_args, **_kwargs: pytest.fail("test-patch gates must use the declared test_command"),
    )

    result = MODULE.generated_test_check(
        object(),
        {
            "repo": "django/django",
            "version": "2.2",
            "test_patch": (
                "diff --git a/tests/other_app/test_hidden.py b/tests/other_app/test_hidden.py\n"
                "--- a/tests/other_app/test_hidden.py\n"
                "+++ b/tests/other_app/test_hidden.py\n"
            ),
        },
        [entry],
        240,
    )

    assert result["passed"] is True
    assert result["failure_kind"] == "passed"
    assert events == ["apply", "declared", "restore"]
    assert result["tests"][0]["runner_command"] == entry["test_command"]
    assert result["tests"][0]["test_paths"] == ["tests/utils_tests/test_html.py"]
    assert sum("runtests.py" in command for command in commands) == 1
    assert sum(
        "git apply --check" in command and "--reverse" not in command
        for command in commands
    ) == 1
    assert sum("git apply --reverse --check" in command for command in commands) == 1
    assert not any(" cp " in f" {command} " for command in commands)
    assert not any("other_app.test_hidden" in command for command in commands)
    assert "public_regression_checks" not in result


def test_test_patch_parser_preserves_final_context_marker():
    patch, paths, command, error = MODULE._test_patch_entry_parts(_test_patch_entry())

    assert error == ""
    assert paths == ["tests/example.py"]
    assert command.endswith("tests/example.py::test_regression")
    assert patch.endswith(" tail_context = True\n")


def test_testpatch_check_rejects_unsafe_paths_without_touching_sandbox(monkeypatch):
    called = []
    monkeypatch.setattr(MODULE.rca, "shell", lambda *_args, **_kwargs: called.append(True))
    entry = _test_patch_entry()
    entry["test_patch"] = entry["test_patch"].replace("tests/example.py", "../gold.patch")

    result = MODULE.generated_test_check(object(), {}, [entry], 240)

    assert result["passed"] is False
    assert result["failure_kind"] == "test_patch_invalid"
    assert "unsafe test_patch path" in result["output_tail"]
    assert called == []


def test_persistent_generated_test_gate_uses_separate_reset_workspace(monkeypatch):
    class FakeEnv:
        cleaned = False

        def cleanup(self):
            self.cleaned = True

    gate_env = FakeEnv()
    commands = []
    applied = []
    checked = []

    monkeypatch.setattr(
        MODULE.rca.swerebench_runner,
        "get_sb_environment",
        lambda _cfg, _instance: gate_env,
    )

    def fake_shell(env, command, timeout=120):
        assert env is gate_env
        commands.append(command)
        if "git rev-parse" in command:
            return {"output": "abc123\n", "returncode": 0}
        return {"output": "", "returncode": 0}

    monkeypatch.setattr(MODULE.rca, "shell", fake_shell)
    monkeypatch.setattr(MODULE.gentest_runner, "restore_env", lambda *_a, **_k: None)
    monkeypatch.setattr(
        MODULE.gentest_runner,
        "apply_test_patch_for_alignment",
        lambda *_a, **_k: {"reason": "", "changed_paths": []},
    )
    monkeypatch.setattr(
        MODULE.rca,
        "apply_prior_patch",
        lambda env, patch: applied.append((env, patch)) or (True, ""),
    )
    monkeypatch.setattr(
        MODULE,
        "generated_test_check",
        lambda env, _instance, tests, _timeout: checked.append((env, tests))
        or MODULE._test_result(passed=True, failure_kind="passed"),
    )

    gate = MODULE.PersistentGeneratedTestGate(
        {"environment": {}},
        {"instance_id": "org__repo-1", "base_commit": "abc123"},
        [_test_patch_entry()],
        SimpleNamespace(
            test_timeout=240,
            install_timeout=600,
            apply_test_patch_at_start=False,
        ),
    )
    result = gate.evaluate("diff --git a/pkg.py b/pkg.py\n")
    gate.close()

    assert result["passed"] is True
    assert result["isolated_to_verifier"] is True
    assert result["persistent_verifier"] is True
    assert applied == [(gate_env, "diff --git a/pkg.py b/pkg.py\n")]
    assert checked[0][0] is gate_env
    assert sum("git reset --hard abc123" in command for command in commands) == 4
    assert gate_env.cleaned is True


def test_persistent_generated_test_gate_retries_transient_reset_failure(monkeypatch):
    class FakeEnv:
        def cleanup(self):
            pass

    reset_calls = 0

    monkeypatch.setattr(
        MODULE.rca.swerebench_runner,
        "get_sb_environment",
        lambda _cfg, _instance: FakeEnv(),
    )

    def fake_shell(_env, command, timeout=120):
        nonlocal reset_calls
        if "git rev-parse" in command:
            return {"output": "abc123\n", "returncode": 0}
        if "git reset --hard" in command:
            reset_calls += 1
            if reset_calls == 1:
                return {"output": "temporary transport failure", "returncode": -1}
        return {"output": "", "returncode": 0}

    monkeypatch.setattr(MODULE.rca, "shell", fake_shell)
    monkeypatch.setattr(MODULE.gentest_runner, "restore_env", lambda *_a, **_k: None)

    gate = MODULE.PersistentGeneratedTestGate(
        {"environment": {}},
        {"instance_id": "org__repo-1", "base_commit": "abc123"},
        [],
        SimpleNamespace(
            test_timeout=240,
            install_timeout=600,
            apply_test_patch_at_start=False,
        ),
    )
    gate.close()

    assert reset_calls == 3


def test_persistent_generated_test_gate_recreates_after_reset_retries_exhausted(monkeypatch):
    class FakeEnv:
        def __init__(self, name):
            self.name = name
            self.cleaned = False

        def cleanup(self):
            self.cleaned = True

    environments = [FakeEnv("first"), FakeEnv("second")]
    created = []
    first_reset_calls = 0
    checked = []

    def fake_environment(_cfg, _instance):
        env = environments[len(created)]
        created.append(env)
        return env

    monkeypatch.setattr(
        MODULE.rca.swerebench_runner,
        "get_sb_environment",
        fake_environment,
    )

    def fake_shell(env, command, timeout=120):
        nonlocal first_reset_calls
        if "git rev-parse" in command:
            return {"output": "abc123\n", "returncode": 0}
        if env is environments[0] and "git reset --hard" in command:
            first_reset_calls += 1
            if first_reset_calls > 2:
                return {"output": "stuck verifier", "returncode": -1}
        return {"output": "", "returncode": 0}

    monkeypatch.setattr(MODULE.rca, "shell", fake_shell)
    monkeypatch.setattr(MODULE.gentest_runner, "restore_env", lambda *_a, **_k: None)
    monkeypatch.setattr(
        MODULE.gentest_runner,
        "apply_test_patch_for_alignment",
        lambda *_a, **_k: {"reason": "", "changed_paths": []},
    )
    monkeypatch.setattr(
        MODULE,
        "generated_test_check",
        lambda env, *_a, **_k: checked.append(env)
        or MODULE._test_result(passed=True, failure_kind="passed"),
    )

    gate = MODULE.PersistentGeneratedTestGate(
        {"environment": {}},
        {"instance_id": "org__repo-1", "base_commit": "abc123"},
        [],
        SimpleNamespace(
            test_timeout=240,
            install_timeout=600,
            apply_test_patch_at_start=False,
        ),
    )
    result = gate.evaluate("")
    gate.close()

    assert result["passed"] is True
    assert result["verifier_recreates"] == 1
    assert checked == [environments[1]]
    assert environments[0].cleaned is True
    assert environments[1].cleaned is True


def test_empty_patch_repair_prompt_does_not_claim_the_generated_test_failed():
    prompt = (
        MODULE.EMPTY_PATCH_REPAIR_TEMPLATE
        .replace("{{task}}", "implement requested behavior")
        .replace("{{feedback}}", "unused")
    )

    assert "without producing any SOURCE patch" in prompt
    assert "generated regression test currently passes" in prompt
    assert "implement requested behavior" in prompt
    assert MODULE.CANDIDATE_CHECK_MARKER in prompt
    assert MODULE.FINAL_SUBMIT_MARKER in prompt


def _bare_repair_agent(initial_patch="patch-0", *, max_gate_checks=10):
    state = {"patch": initial_patch}
    agent = object.__new__(MODULE.RepairAgent)
    agent._max_rounds = 5
    agent._max_gate_checks = max_gate_checks
    agent._gate_limit_stop_reason = "max_gate_checks"
    agent._continue_on_duplicate_patch = True
    agent._round_start_calls = 0
    agent._last_gated_patch = initial_patch
    agent._get_current_patch = lambda: state["patch"]
    agent._initial_patch = initial_patch
    agent.rounds = []
    agent.messages = []
    agent.passed = False
    agent.final_test_result = MODULE._test_result(failure_kind="assertion_failure")
    agent.model_final_submit = False
    agent.model_final_submit_gate_reused = False
    agent.model_final_decision = False
    agent.model_keep_original = False
    agent.fallback_to_round0 = False
    agent.selected_patch = None
    agent.stop_reason = ""
    agent._decision_only = False
    agent._decision_deadline = None
    agent._post_pass_deadline = None
    agent.duplicate_submissions = 0
    agent.n_calls = 0
    agent.model = SimpleNamespace(
        format_message=lambda *, role, content: {"role": role, "content": content}
    )
    agent.add_messages = lambda *messages: agent.messages.extend(messages) or list(messages)
    return agent, state


def test_duplicate_of_initial_patch_is_non_terminal_and_does_not_run_gate():
    agent, _state = _bare_repair_agent()
    checked = []
    agent._on_submit = lambda: checked.append(True)

    agent.n_calls = 3
    agent._handle_submission("", forced_submission=False)

    assert checked == []
    assert agent.rounds == []
    assert agent.duplicate_submissions == 1
    assert agent.stop_reason == ""
    assert MODULE.DUPLICATE_PATCH_FOLLOWUP in agent.messages[-1]["content"]


def test_ten_gate_budget_can_continue_past_legacy_five_with_fixed_turn_counter():
    agent, state = _bare_repair_agent(max_gate_checks=10)
    checked = []

    def on_submit():
        checked.append(state["patch"])
        return MODULE._test_result(failure_kind="assertion_failure"), state["patch"]

    agent._on_submit = on_submit
    for gate_index in range(1, 7):
        state["patch"] = f"patch-{gate_index}"
        agent.n_calls = gate_index * 10
        agent._handle_submission(state["patch"], forced_submission=False)

    assert len(checked) == 6
    assert len(agent.rounds) == 6
    assert agent.stop_reason == ""
    assert agent.n_calls == 60
    assert agent.n_calls < 5 * 30


def test_candidate_check_marker_is_structured_without_text_guessing():
    output = {
        "returncode": 0,
        "output": (
            "COMPLETE_TASK_AND_CHECK_CANDIDATE_PATCH\n"
            "diff --git a/pkg.py b/pkg.py\n"
        ),
    }

    assert MODULE._submission_after_marker(output, MODULE.CANDIDATE_CHECK_MARKER) == (
        "diff --git a/pkg.py b/pkg.py\n"
    )
    assert MODULE._submission_after_marker(output, MODULE.FINAL_SUBMIT_MARKER) is None


def test_keep_original_marker_is_exact_and_structured():
    output = {
        "returncode": 0,
        "output": "COMPLETE_TASK_AND_KEEP_ORIGINAL_PATCH\n",
    }

    assert MODULE._submission_after_marker(output, MODULE.KEEP_ORIGINAL_MARKER) == ""
    assert MODULE._submission_after_marker(
        {"returncode": 0, "output": "prefix COMPLETE_TASK_AND_KEEP_ORIGINAL_PATCH\n"},
        MODULE.KEEP_ORIGINAL_MARKER,
    ) is None


def test_execute_actions_emits_explicit_candidate_check_decision():
    agent = object.__new__(MODULE.RepairAgent)
    agent.env = SimpleNamespace(execute=lambda _action: {
        "returncode": 0,
        "output": (
            "COMPLETE_TASK_AND_CHECK_CANDIDATE_PATCH\n"
            "diff --git a/pkg.py b/pkg.py\n"
        ),
    })
    agent.model = SimpleNamespace(
        format_observation_messages=lambda _message, _outputs, _vars: []
    )
    agent.get_template_vars = lambda: {}
    agent.messages = []
    agent.add_messages = lambda *messages: agent.messages.extend(messages) or list(messages)

    with pytest.raises(MODULE.Submitted) as submitted:
        agent.execute_actions({"extra": {"actions": [{"command": "check"}]}})

    request = submitted.value.messages[0]["extra"]
    assert request["exit_status"] == "CandidateCheckRequested"
    assert request["repair_decision"] == "candidate_check"
    assert request["submission"].startswith("diff --git")


def test_gate_failure_returns_feedback_and_keeps_repair_running():
    agent, state = _bare_repair_agent()
    state["patch"] = "patch-1"
    agent._on_submit = lambda: (
        MODULE._test_result(
            failure_kind="assertion_failure",
            output_tail="assert expected == actual",
        ),
        state["patch"],
    )

    agent._handle_submission(
        state["patch"],
        forced_submission=False,
        decision="candidate_check",
    )

    assert agent.passed is False
    assert agent.model_final_submit is False
    assert agent.stop_reason == ""
    assert agent.rounds[-1]["decision"] == "candidate_check"
    assert "not an absolute veto" in agent.messages[-1]["content"]
    assert MODULE.FINAL_SUBMIT_MARKER in agent.messages[-1]["content"]


def test_gate_pass_returns_feedback_without_finishing_or_implicit_submit():
    agent, state = _bare_repair_agent()
    state["patch"] = "patch-1"
    agent._on_submit = lambda: (
        MODULE._test_result(passed=True, failure_kind="passed", output_tail="1 passed"),
        state["patch"],
    )

    agent._handle_submission(
        state["patch"],
        forced_submission=False,
        decision="candidate_check",
    )

    assert agent.passed is True
    assert agent.model_final_submit is False
    assert agent.model_final_decision is False
    assert agent.stop_reason == ""
    assert agent._post_pass_deadline == MODULE.POST_PASS_REVIEW_TURNS
    assert "not an automatic final" in agent.messages[-1]["content"]


def test_keep_original_is_explicit_final_decision_and_selects_initial_patch():
    agent, state = _bare_repair_agent()
    state["patch"] = "bad-repair"

    with pytest.raises(MODULE.Submitted) as submitted:
        agent._handle_submission(
            "",
            forced_submission=False,
            decision="keep_original",
        )

    assert agent.model_final_decision is True
    assert agent.model_keep_original is True
    assert agent.model_final_submit is False
    assert agent.selected_patch == "patch-0"
    assert agent.stop_reason == "model_keep_original"
    assert submitted.value.messages[0]["extra"]["submission"] == "patch-0"


def test_missing_final_decision_falls_back_to_initial_patch():
    agent, state = _bare_repair_agent()
    state["patch"] = "bad-repair"

    with pytest.raises(MODULE.Submitted) as submitted:
        agent._fallback_to_initial("no_explicit_final_decision")

    assert agent.model_final_decision is False
    assert agent.fallback_to_round0 is True
    assert agent.selected_patch == "patch-0"
    assert submitted.value.messages[0]["extra"]["fallback_to_round0"] is True


def test_post_pass_review_window_requests_one_final_decision_turn():
    agent, _state = _bare_repair_agent()
    agent._post_pass_deadline = 10
    agent.n_calls = 10

    agent.step()

    assert agent._decision_only is True
    assert agent._decision_deadline == 11
    assert "post-pass review window" in agent.messages[-1]["content"]


def test_model_can_final_submit_unchanged_patch_after_gate_failure():
    agent, state = _bare_repair_agent()
    checked = []
    agent._on_submit = lambda: checked.append(True)

    with pytest.raises(MODULE.Submitted) as submitted:
        agent._handle_submission(
            state["patch"],
            forced_submission=False,
            decision="final_submit",
        )

    assert checked == []
    assert agent.model_final_submit is True
    assert agent.model_final_submit_gate_reused is True
    assert agent.passed is False
    assert agent.stop_reason == "model_final_submit"
    assert agent.final_test_result["gate_status"] == "hard_fail"
    assert submitted.value.messages[0]["extra"]["model_final_submit"] is True
    assert submitted.value.messages[0]["extra"]["final_gate_status"] == "hard_fail"


def test_model_final_submit_new_patch_records_but_is_not_vetoed_by_failing_gate():
    agent, state = _bare_repair_agent()
    state["patch"] = "patch-1"
    checked = []

    def on_submit():
        checked.append(state["patch"])
        return MODULE._test_result(failure_kind="assertion_failure"), state["patch"]

    agent._on_submit = on_submit
    with pytest.raises(MODULE.Submitted):
        agent._handle_submission(
            state["patch"],
            forced_submission=False,
            decision="final_submit",
        )

    assert checked == ["patch-1"]
    assert agent.model_final_submit is True
    assert agent.model_final_submit_gate_reused is False
    assert agent.passed is False
    assert agent.rounds[-1]["decision"] == "final_submit"
    assert agent.rounds[-1]["gate_status"] == "hard_fail"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
