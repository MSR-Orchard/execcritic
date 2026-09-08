import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from jinja2 import StrictUndefined, Template

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from minisweagent.exceptions import FormatError, Submitted
from minisweagent.agents.extra.bash_testpatch_resolve import BashTestPatchResolveAgent
from minisweagent.agents.extra.testpatch_resolve import TestPatchResolveAgent as ResolveAgent
from minisweagent.models.utils.actions_toolcall import parse_toolcall_actions, tools_for_names
from minisweagent.models.utils.actions_toolcall_response import (
    parse_toolcall_actions_response,
    tools_for_names_response_api,
)
NUM_GPUS = 0

PRODUCT_PATCH = """\
diff --git a/django/middleware/common.py b/django/middleware/common.py
--- a/django/middleware/common.py
+++ b/django/middleware/common.py
@@ -1 +1 @@
-old
+new
"""

TEST_PATCH = """\
diff --git a/tests/test_redirect.py b/tests/test_redirect.py
--- a/tests/test_redirect.py
+++ b/tests/test_redirect.py
@@ -1 +1,2 @@
 old
+def test_prefix(): assert False
"""

MULTI_TEST_PATCH = """\
diff --git a/tests/test_redirect.py b/tests/test_redirect.py
--- a/tests/test_redirect.py
+++ b/tests/test_redirect.py
@@ -1 +1,3 @@
 old
+def test_prefix(): assert False
+def test_prefix_query(): assert False
"""

FIXTURE_ONLY_PATCH = """\
diff --git a/tests/redirect_cases.txt b/tests/redirect_cases.txt
--- a/tests/redirect_cases.txt
+++ b/tests/redirect_cases.txt
@@ -1 +1,2 @@
 /old
+/broken-prefix
"""


def test_patch_safety_check_allows_product_source_paths():
    ok, reason, changed = ResolveAgent._patch_safety_check(PRODUCT_PATCH)

    assert ok
    assert reason == ""
    assert changed == ["django/middleware/common.py"]


def test_patch_safety_check_still_rejects_path_traversal():
    patch = """\
diff --git a/../outside.py b/../outside.py
--- a/../outside.py
+++ b/../outside.py
@@ -1 +1 @@
-old
+new
"""

    ok, reason, changed = ResolveAgent._patch_safety_check(patch)

    assert not ok
    assert reason == "unsafe path: ../outside.py"
    assert changed == []


def test_chat_write_test_patch_requires_and_parses_test_command():
    tool = next(
        tool for tool in tools_for_names(["write_test_patch"])
        if tool["function"]["name"] == "write_test_patch"
    )
    assert tool["function"]["parameters"]["required"] == ["test_command"]

    tool_call = MagicMock()
    tool_call.function.name = "write_test_patch"
    tool_call.function.arguments = json.dumps(
        {
            "test_command": "python tests/runtests.py admin_views.tests.RedirectTests.test_prefix",
        }
    )
    tool_call.id = "call_write"

    action = parse_toolcall_actions(
        [tool_call],
        format_error_template="{{ error }}",
        extra_tools=["write_test_patch"],
    )[0]

    assert action["test_command"] == (
        "python tests/runtests.py admin_views.tests.RedirectTests.test_prefix"
    )


def test_responses_write_test_patch_requires_and_parses_test_command():
    tool = next(
        tool for tool in tools_for_names_response_api(["write_test_patch"])
        if tool["name"] == "write_test_patch"
    )
    assert tool["parameters"]["required"] == ["test_command"]

    action = parse_toolcall_actions_response(
        [
            {
                "type": "function_call",
                "name": "write_test_patch",
                "call_id": "call_write",
                "arguments": json.dumps(
                    {
                    "test_command": "python tests/runtests.py admin_views.tests.RedirectTests.test_prefix",
                    }
                ),
            }
        ],
        format_error_template="{{ error }}",
        extra_tools=["write_test_patch"],
    )[0]

    assert action["test_command"] == (
        "python tests/runtests.py admin_views.tests.RedirectTests.test_prefix"
    )


@pytest.mark.parametrize("parser_kind", ["chat", "responses"])
def test_write_test_patch_rejects_missing_test_command(parser_kind):
    arguments = json.dumps({"test_patch": PRODUCT_PATCH})
    with pytest.raises(FormatError) as exc_info:
        if parser_kind == "chat":
            tool_call = MagicMock()
            tool_call.function.name = "write_test_patch"
            tool_call.function.arguments = arguments
            tool_call.id = "call_write"
            parse_toolcall_actions(
                [tool_call],
                format_error_template="{{ error }}",
                extra_tools=["write_test_patch"],
            )
        else:
            parse_toolcall_actions_response(
                [
                    {
                        "type": "function_call",
                        "name": "write_test_patch",
                        "call_id": "call_write",
                        "arguments": arguments,
                    }
                ],
                format_error_template="{{ error }}",
                extra_tools=["write_test_patch"],
            )
    assert "test_command" in json.dumps(exc_info.value.messages)


def test_verify_patch_executes_declared_command_and_hash_binds_it():
    calls = []
    agent = object.__new__(ResolveAgent)
    agent._verify_runner = lambda patch, command: calls.append((patch, command)) or {
        "clean_fail": True,
        "verdict": "fail",
        "failed": 1,
        "failure_category": "assertion_failure_clean",
    }

    django_command = "python tests/runtests.py admin_views.tests.RedirectTests.test_prefix"
    pytest_command = "python -m pytest tests/admin_views/tests.py -rA"
    django_report = agent._verify_patch(PRODUCT_PATCH, django_command)
    pytest_report = agent._verify_patch(PRODUCT_PATCH, pytest_command)

    assert calls == [(PRODUCT_PATCH, django_command), (PRODUCT_PATCH, pytest_command)]
    assert django_report["test_command"] == django_command
    assert django_report["patch_hash"] != pytest_report["patch_hash"]


class _NativeSubmissionEnv:
    test_command = "python -m pytest tests/test_redirect.py::test_prefix"
    test_contract = ""

    def execute(self, action, **kwargs):
        command = action["command"]
        if "/tmp/test_contract.json" in command:
            return {"output": self.test_contract, "returncode": 0 if self.test_contract else 1}
        if "/tmp/test_command" in command:
            return {"output": self.test_command, "returncode": 0}
        raise AssertionError(command)


def _native_submission(patch=TEST_PATCH):
    return Submitted(
        {
            "role": "exit",
            "content": patch,
            "extra": {"exit_status": "Submitted", "submission": patch},
        }
    )


def _native_agent(report):
    agent = object.__new__(BashTestPatchResolveAgent)
    agent.config = MagicMock(
        test_command_path="/tmp/test_command",
        test_contract_path="/tmp/test_contract.json",
        require_test_contract=False,
        require_exact_test_selector=False,
        stop_at_first_reject=False,
    )
    agent.env = _NativeSubmissionEnv()
    agent.model = MagicMock()
    agent.model.format_message.side_effect = lambda **kwargs: kwargs
    agent._verify_runner = lambda patch, command: report
    agent._last_patch = ""
    agent._last_canonical_patch = ""
    agent._last_patch_hash = ""
    agent._last_report = None
    agent._last_test_contract = None
    return agent


def _behavior_contract_contract(**updates):
    contract = {
        "entrypoint": "public redirect API",
        "trigger": "a redirect prefix that previously loses its slash",
        "expected_output": "the observable redirect preserves the issue-required prefix",
        "issue_evidence": ["the issue states that the public redirect loses its prefix"],
        "alternative_hypotheses": ["the prefix may be normalized rather than preserved; adjacent tests reject this"],
        "non_assertions": ["internal middleware flags and unstated full response text"],
        "test_nodes": ["test_prefix"],
    }
    contract.update(updates)
    return json.dumps(contract)


def _native_behavior_contract_agent(report, *, command=None, contract=None):
    agent = _native_agent(report)
    agent.config.require_test_contract = True
    agent.config.require_exact_test_selector = True
    agent.env.test_command = command or "python -m pytest tests/test_redirect.py::test_prefix"
    agent.env.test_contract = _behavior_contract_contract() if contract is None else contract
    return agent


def test_bash_native_submission_accepts_clean_base_failure():
    agent = _native_agent({
        "clean_fail": True,
        "verdict": "fail",
        "failed": 1,
        "errors": 0,
        "failure_category": "assertion_failure_clean",
    })

    accepted, message = agent._review_native_submission(_native_submission())

    assert accepted
    assert message["extra"]["exit_status"] == "Submitted"
    assert message["extra"]["submission"] == TEST_PATCH
    assert message["extra"]["resolve_gate"]["submission_mode"] == "native_bash"
    assert message["extra"]["resolve_gate"]["test_command"].endswith("::test_prefix")


def test_bash_native_submission_returns_verifier_feedback_without_exiting():
    agent = _native_agent({
        "clean_fail": False,
        "verdict": "pass",
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "failure_category": "base_passed",
    })

    accepted, output = agent._review_native_submission(_native_submission())

    assert not accepted
    assert output["returncode"] == 2
    assert "base_clean_fail: False" in output["output"]
    assert output["extra"]["resolve_native_submission_rejected"]


def test_behavior_contract_native_submission_accepts_valid_contract_and_exact_selector():
    agent = _native_behavior_contract_agent({
        "clean_fail": True,
        "verdict": "fail",
        "failed": 1,
        "errors": 0,
        "failure_category": "assertion_failure_clean",
    })

    accepted, message = agent._review_native_submission(_native_submission())

    assert accepted
    assert message["extra"]["resolve_gate"]["behavior_contract"]["test_nodes"] == ["test_prefix"]


def test_behavior_contract_native_submission_rejects_missing_contract_before_verifier():
    agent = _native_behavior_contract_agent({}, contract="")
    agent._verify_runner = MagicMock()

    accepted, output = agent._review_native_submission(_native_submission())

    assert not accepted
    assert "missing non-empty contract file" in output["output"]
    agent._verify_runner.assert_not_called()


def test_behavior_contract_native_submission_rejects_broad_selector_before_verifier():
    agent = _native_behavior_contract_agent({}, command="python -m pytest tests/test_redirect.py")
    agent._verify_runner = MagicMock()

    accepted, output = agent._review_native_submission(_native_submission())

    assert not accepted
    assert "must explicitly select every behavior-contract test node" in output["output"]
    agent._verify_runner.assert_not_called()


def test_behavior_contract_native_submission_rejects_contract_node_not_in_patch():
    agent = _native_behavior_contract_agent({}, contract=_behavior_contract_contract(test_nodes=["test_other"]))
    agent._verify_runner = MagicMock()

    accepted, output = agent._review_native_submission(_native_submission())

    assert not accepted
    assert "must include every test_*" in output["output"]
    agent._verify_runner.assert_not_called()


def test_behavior_contract_native_submission_accepts_multiple_declared_test_nodes():
    agent = _native_behavior_contract_agent(
        {
            "clean_fail": True,
            "verdict": "fail",
            "failed": 2,
            "errors": 0,
            "failure_category": "assertion_failure_clean",
        },
        command=(
            "python -m pytest tests/test_redirect.py::test_prefix "
            "tests/test_redirect.py::test_prefix_query"
        ),
        contract=_behavior_contract_contract(test_nodes=["test_prefix", "test_prefix_query"]),
    )

    accepted, message = agent._review_native_submission(_native_submission(MULTI_TEST_PATCH))

    assert accepted
    assert message["extra"]["resolve_gate"]["behavior_contract"]["test_nodes"] == [
        "test_prefix",
        "test_prefix_query",
    ]


def test_behavior_contract_native_submission_rejects_undeclared_added_test_node():
    agent = _native_behavior_contract_agent(
        {},
        command="python -m pytest tests/test_redirect.py::test_prefix",
        contract=_behavior_contract_contract(test_nodes=["test_prefix"]),
    )
    agent._verify_runner = MagicMock()

    accepted, output = agent._review_native_submission(_native_submission(MULTI_TEST_PATCH))

    assert not accepted
    assert "test_prefix_query" in output["output"]
    assert "must include every test_*" in output["output"]
    agent._verify_runner.assert_not_called()


def test_behavior_contract_native_submission_rejects_command_missing_declared_node():
    agent = _native_behavior_contract_agent(
        {},
        command="python -m pytest tests/test_redirect.py::test_prefix",
        contract=_behavior_contract_contract(test_nodes=["test_prefix", "test_prefix_query"]),
    )
    agent._verify_runner = MagicMock()

    accepted, output = agent._review_native_submission(_native_submission(MULTI_TEST_PATCH))

    assert not accepted
    assert "must explicitly select every" in output["output"]
    assert "test_prefix_query" in output["output"]
    agent._verify_runner.assert_not_called()


def test_behavior_contract_native_submission_accepts_existing_node_for_fixture_only_patch():
    agent = _native_behavior_contract_agent(
        {
            "clean_fail": True,
            "verdict": "fail",
            "failed": 1,
            "errors": 0,
            "failure_category": "assertion_failure_clean",
        },
        command="python -m pytest tests/test_redirect.py::test_redirect_cases",
        contract=_behavior_contract_contract(test_nodes=["test_redirect_cases"]),
    )

    accepted, message = agent._review_native_submission(_native_submission(FIXTURE_ONLY_PATCH))

    assert accepted
    assert message["extra"]["resolve_gate"]["behavior_contract"]["test_nodes"] == [
        "test_redirect_cases"
    ]


def test_bash_native_stops_immediately_on_agent_connection_error():
    agent = object.__new__(BashTestPatchResolveAgent)
    agent.config = MagicMock(sandbox_infra_timeout_limit=2)
    agent._consecutive_sandbox_timeouts = 0

    reason = agent._sandbox_infra_error({
        "output": "",
        "returncode": -1,
        "exception_info": (
            "Agent connection error: Cannot connect to host 192.0.2.142:8080 "
            "ssl:default [Connect call failed]"
        ),
    })

    assert "Agent connection error" in reason


def test_bash_native_stops_immediately_on_azure_sandbox_exec_404():
    agent = object.__new__(BashTestPatchResolveAgent)
    agent.config = MagicMock(sandbox_infra_timeout_limit=2)
    agent._consecutive_sandbox_timeouts = 0

    reason = agent._sandbox_infra_error({
        "output": (
            "<exception>An error occurred while executing the command:\n"
            "404 Client Error: Not Found for url:\n"
            "https://sandbox.example/sandboxes/resolve-repo--issue-0-123/exec</exception>\n"
            "<returncode>-1</returncode>"
        ),
        "returncode": -1,
        "exception_info": "",
    })

    assert "404 Client Error" in reason
    assert "/sandboxes/resolve-repo--issue-0-123/exec" in reason
    ordinary_404 = {
        "output": "application request returned 404 Not Found",
        "returncode": 1,
        "exception_info": "",
    }
    assert agent._sandbox_infra_error(ordinary_404) == ""


def test_bash_native_shared_verifier_cleanup_error_is_infra_not_rejection():
    agent = _native_agent({})
    agent._consecutive_sandbox_timeouts = 0
    error = (
        "shared verify transaction failed: post-verify reset failed: ; "
        "verifier context cleanup failed:"
    )

    def failed_verify(_patch, _command):
        raise RuntimeError(error)

    agent._verify_runner = failed_verify

    accepted, output = agent._review_native_submission(_native_submission())

    assert not accepted
    assert output["returncode"] == -1
    assert output["extra"]["sandbox_infra_error"] is True
    assert "resolve_native_submission_rejected" not in output["extra"]
    assert error in output["exception_info"]
    assert agent._sandbox_infra_error(output)


def test_bash_native_other_verifier_error_remains_submission_feedback():
    agent = _native_agent({})

    def failed_verify(_patch, _command):
        raise RuntimeError("invalid deterministic verifier configuration")

    agent._verify_runner = failed_verify

    accepted, output = agent._review_native_submission(_native_submission())

    assert not accepted
    assert output["returncode"] == 2
    assert output["extra"]["resolve_native_submission_rejected"] is True
    assert "Revise the workspace and submit again" in output["output"]


def test_bash_native_connection_error_exits_without_another_model_turn():
    agent = object.__new__(BashTestPatchResolveAgent)
    agent.config = MagicMock(sandbox_infra_timeout_limit=2)
    agent._consecutive_sandbox_timeouts = 0
    agent.env = MagicMock()
    agent.env.execute.return_value = {
        "output": "",
        "returncode": -1,
        "exception_info": "Agent connection error: Cannot connect to host 192.0.2.142:8080",
    }
    agent.model = MagicMock()
    agent.model.format_message.side_effect = lambda **kwargs: kwargs
    agent.model.format_observation_messages.return_value = []
    agent.get_template_vars = MagicMock(return_value={})
    agent.add_messages = MagicMock(side_effect=lambda *messages: list(messages))
    message = {"extra": {"actions": [{"command": "echo ok"}]}}

    result = agent.execute_actions(message)

    assert result[0]["extra"]["exit_status"] == "sandbox_infra_error"
    assert result[0]["extra"]["sandbox_infra_error"] is True
    agent.env.execute.assert_called_once()


def test_bash_native_requires_two_consecutive_command_timeouts():
    agent = object.__new__(BashTestPatchResolveAgent)
    agent.config = MagicMock(sandbox_infra_timeout_limit=2)
    agent._consecutive_sandbox_timeouts = 0
    timeout = {
        "output": "Command exceeded timeout of 60s and was forcibly terminated",
        "returncode": -1,
        "exception_info": "Execution timed out after 60s",
    }

    assert agent._sandbox_infra_error(timeout) == ""
    assert "2 consecutive" in agent._sandbox_infra_error(timeout)

    agent._consecutive_sandbox_timeouts = 1
    assert agent._sandbox_infra_error({"output": "ok", "returncode": 0}) == ""
    assert agent._consecutive_sandbox_timeouts == 0


def test_v11_exposes_only_standard_bash_tool():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "gentest_v2/versions/v11_resolve_testpatch.yaml").read_text()
    )

    assert config["agent"]["agent_class"] == "bash_testpatch_resolve"
    assert config["model"]["extra_tools"] == []


def test_v11_prompt_prioritizes_early_native_submission():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "gentest_v2/versions/v11_resolve_testpatch.yaml").read_text()
    )
    prompt = Template(
        config["agent"]["instance_template"], undefined=StrictUndefined
    ).render(task="Regression-test issue", step_limit=config["agent"]["step_limit"])
    normalized_prompt = " ".join(prompt.split())

    assert "You have at most 60 model turns" in normalized_prompt
    assert "no later than turn" not in normalized_prompt
    assert (
        "your next bash call must prepare /tmp/test_command and /tmp/test.patch"
        in normalized_prompt
    )
    assert "the following bash call must submit them" in normalized_prompt
    assert "Run the exact focused repository-native command once" in normalized_prompt
    assert "Do not rerun the unchanged test" in normalized_prompt
    assert "submit again promptly; do not restart broad exploration" in normalized_prompt
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in normalized_prompt
    assert "Do not run `git log`, `git show`, `git branch`" in normalized_prompt


def test_behavior_contract_requires_gold_free_behavior_contract_and_exact_node_selectors():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "gentest_v2/versions/behavior_contract.yaml").read_text()
    )
    prompt = Template(
        config["agent"]["instance_template"], undefined=StrictUndefined
    ).render(task="Regression-test issue", step_limit=config["agent"]["step_limit"])
    normalized_prompt = " ".join(prompt.split())

    assert config["agent"]["agent_class"] == "bash_testpatch_resolve"
    assert config["agent"]["require_test_contract"] is True
    assert config["agent"]["require_exact_test_selector"] is True
    assert config["model"]["extra_tools"] == []
    assert "You do not have that fix, its patch, or oracle tests" in normalized_prompt
    assert "/tmp/test_contract.json" in normalized_prompt
    assert "alternative_hypotheses" in normalized_prompt
    assert "non_assertions" in normalized_prompt
    assert "one or more new `test_*` functions or methods" in normalized_prompt
    assert "fixture, parameter, helper, or data files" in normalized_prompt
    assert "Existing nodes are allowed for fixture/parameter/data edits" in normalized_prompt
    assert "must explicitly select every test node" in normalized_prompt
    assert "A clean Base failure is necessary but not sufficient" in normalized_prompt
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in normalized_prompt


def test_smoke_rejects_submitted_result_that_fails_fresh_replay():
    from swe_harness.gentest_v2.smoke_resolve_unrestricted import smoke_passed

    summary = {
        "child_exit_code": 0,
        "exit_status": "Submitted",
        "response_errors": [],
        "tool_calls": {"grounding_plan": 1, "write_test_patch": 1, "submit_test_patch": 1},
        "fresh_replay": {"clean_fail": False, "infra_failure": True},
    }

    assert not smoke_passed(summary, require_submit=False)


def test_smoke_accepts_submitted_result_that_passes_fresh_replay():
    from swe_harness.gentest_v2.smoke_resolve_unrestricted import smoke_passed

    summary = {
        "child_exit_code": 0,
        "exit_status": "Submitted",
        "response_errors": [],
        "tool_calls": {"grounding_plan": 1, "write_test_patch": 1, "submit_test_patch": 1},
        "fresh_replay": {"clean_fail": True, "infra_failure": False},
    }

    assert smoke_passed(summary, require_submit=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
