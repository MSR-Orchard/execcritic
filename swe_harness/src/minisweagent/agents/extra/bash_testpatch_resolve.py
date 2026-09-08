"""Bash-native resolve agent for regression-test patches.

The model receives only the standard bash tool. It inspects and edits the workspace directly,
records its focused test command in a fixed file, and submits a normal git patch through
mini-swe-agent's ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` marker. The agent intercepts that
native submission, runs the official Base verifier, and either accepts it or returns paired tool
feedback so the model can revise the workspace and submit again.
"""

from __future__ import annotations

import json
import re
import shlex

from minisweagent.agents.extra.testpatch_resolve import TestPatchResolveAgent, TestPatchResolveConfig
from minisweagent.exceptions import Submitted


class BashTestPatchResolveConfig(TestPatchResolveConfig):
    test_command_path: str = "/tmp/test_command"
    test_contract_path: str = "/tmp/test_contract.json"
    require_test_contract: bool = False
    require_exact_test_selector: bool = False
    sandbox_infra_timeout_limit: int = 2
    # When True, the FIRST time a native submission is gate-rejected (base test does not
    # cleanly fail) the agent stops with exit_status="SubmitRejected" instead of returning
    # feedback for another attempt. Used to harvest first-submit-failure states for
    # cross-model (DeepSeek) recovery distillation. Default False keeps the retry behavior.
    stop_at_first_reject: bool = False


_SANDBOX_AGENT_CONNECTION_ERROR = re.compile(
    r"agent connection error|cannot connect to host\s+10\.\d+\.\d+\.\d+:8080|"
    r"container exec failed|sandbox (?:pod )?(?:disappeared|expired|unavailable)",
    re.IGNORECASE,
)
_AZURE_SANDBOX_EXEC_NOT_FOUND = re.compile(
    r"404 Client Error:\s*Not Found for url:\s*"
    r"https?://[^\s<>]+/sandboxes/[^\s<>/]+/exec\b",
    re.IGNORECASE,
)
_SHARED_VERIFIER_LIFECYCLE_ERROR = re.compile(
    r"\bshared verify transaction failed:",
    re.IGNORECASE,
)
_SANDBOX_COMMAND_TIMEOUT = re.compile(
    r"execution timed out after \d+s|command timed out after \d+s|"
    r"command exceeded timeout of \d+s",
    re.IGNORECASE,
)


class BashTestPatchResolveAgent(TestPatchResolveAgent):
    def __init__(self, model, env, **kwargs):
        super().__init__(model, env, config_class=BashTestPatchResolveConfig, **kwargs)
        self._consecutive_sandbox_timeouts = 0
        self._last_test_contract: dict | None = None

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute standard bash actions and intercept native patch submissions."""
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for action in actions:
            try:
                output = self.env.execute(action)
            except Submitted as submitted:
                accepted, result = self._review_native_submission(submitted)
                if accepted:
                    # stop_at_first_reject returns an exit message (SubmitRejected) instead of a
                    # real submission; end the trajectory here carrying the stuck state.
                    if isinstance(result, dict) and result.get("role") == "exit":
                        self.add_messages(result)
                        return [result]
                    raise Submitted(result)
                # A submission was attempted but rejected before the clean-fail gate (empty patch,
                # missing/invalid test_command, patch safety, verifier error). Under
                # stop_at_first_reject, do NOT let the model keep retrying — end here so no turns
                # are burned and the first-submit-failure state is harvested cleanly.
                if getattr(self.config, "stop_at_first_reject", False) and isinstance(result, dict) \
                        and (result.get("extra") or {}).get("resolve_native_submission_rejected"):
                    exit_message = self.model.format_message(
                        role="exit",
                        content="SubmitRejected",
                        extra={
                            "exit_status": "SubmitRejected",
                            "submission": "",
                            "resolve_gate": {
                                "base_clean_fail": False,
                                "test_command": self._read_workspace_file(self.config.test_command_path),
                                "submission_mode": "native_bash",
                                "reject_stage": "pre_gate",
                            },
                            "rejected_patch": getattr(self, "_last_patch", "") or "",
                            "rejection_feedback": str(result.get("output") or ""),
                        },
                    )
                    self.add_messages(exit_message)
                    return [exit_message]
                output = result
            outputs.append(output)
            if reason := self._sandbox_infra_error(output):
                exit_message = self.model.format_message(
                    role="exit",
                    content="sandbox_infra_error",
                    extra={
                        "exit_status": "sandbox_infra_error",
                        "submission": "",
                        "sandbox_infra_error": True,
                        "sandbox_infra_reason": reason,
                    },
                )
                self.add_messages(
                    *self.model.format_observation_messages(
                        message, self._pad(outputs, actions), self.get_template_vars()
                    ),
                    exit_message,
                )
                return [exit_message]
        return self.add_messages(
            *self.model.format_observation_messages(message, self._pad(outputs, actions), self.get_template_vars())
        )

    def _sandbox_infra_error(self, output: dict) -> str:
        """Return a fatal infra reason without confusing one slow command for a dead sandbox."""
        text = "\n".join(str(output.get(key) or "") for key in ("exception_info", "output"))
        if (
            _SANDBOX_AGENT_CONNECTION_ERROR.search(text)
            or _AZURE_SANDBOX_EXEC_NOT_FOUND.search(text)
            or _SHARED_VERIFIER_LIFECYCLE_ERROR.search(text)
        ):
            self._consecutive_sandbox_timeouts = 0
            return text.strip()[:500]

        timed_out = output.get("returncode") == -1 and bool(_SANDBOX_COMMAND_TIMEOUT.search(text))
        if timed_out:
            self._consecutive_sandbox_timeouts += 1
            limit = max(1, int(self.config.sandbox_infra_timeout_limit))
            if self._consecutive_sandbox_timeouts >= limit:
                return (
                    f"{self._consecutive_sandbox_timeouts} consecutive sandbox command timeouts: "
                    f"{text.strip()[:400]}"
                )
        else:
            self._consecutive_sandbox_timeouts = 0
        return ""

    def _review_native_submission(self, submitted: Submitted) -> tuple[bool, dict]:
        patch = self._submitted_patch(submitted)
        if not patch.strip():
            return False, self._rejection("Native submission contained an empty patch.")

        ok, reason, _ = self._patch_safety_check(patch)
        if not ok:
            return False, self._rejection(f"Patch rejected: {reason}.")

        test_command = self._read_workspace_file(self.config.test_command_path)
        if not test_command:
            return False, self._rejection(
                f"Missing focused test command. Write exactly one command to {self.config.test_command_path}."
            )
        if "\n" in test_command:
            return False, self._rejection(
                f"{self.config.test_command_path} must contain exactly one non-empty command line."
            )

        contract = None
        if getattr(self.config, "require_test_contract", False):
            contract_text = self._read_workspace_file(self.config.test_contract_path)
            contract, reason = self._validate_test_contract(contract_text, patch, test_command)
            if reason:
                return False, self._rejection(
                    f"Invalid behavior contract in {self.config.test_contract_path}: {reason}"
                )
        elif getattr(self.config, "require_exact_test_selector", False):
            reason = self._exact_selector_rejection(patch, test_command, None)
            if reason:
                return False, self._rejection(reason)

        report = self._verify_patch(patch, test_command)
        if report.get("error"):
            error = str(report["error"])
            if _SHARED_VERIFIER_LIFECYCLE_ERROR.search(error):
                return False, {
                    "output": "",
                    "returncode": -1,
                    "exception_info": f"Verifier error: {error}",
                    "extra": {
                        "resolve_verifier_error": error,
                        "sandbox_infra_error": True,
                    },
                }
            return False, self._rejection(f"Verifier error: {error}")

        self._last_patch = patch
        self._last_canonical_patch = report["canonical_patch"]
        self._last_patch_hash = report["patch_hash"]
        self._last_report = report
        self._last_test_contract = contract
        if not report.get("clean_fail"):
            feedback = self._format_feedback(report)
            if getattr(self.config, "stop_at_first_reject", False):
                # Harvest mode: end the trajectory at the first gate rejection instead of
                # letting the policy retry. The exit carries the exact stuck state so a
                # different model can resume from here (cross-model recovery distillation).
                exit_message = self.model.format_message(
                    role="exit",
                    content="SubmitRejected",
                    extra={
                        "exit_status": "SubmitRejected",
                        "submission": "",
                        "resolve_gate": {
                            "patch_hash": self._last_patch_hash,
                            "base_clean_fail": False,
                            "test_command": report.get("test_command", ""),
                            "official_verifier_kind": report.get("official_verifier_kind"),
                            "submission_mode": "native_bash",
                        },
                        "rejected_patch": self._last_canonical_patch,
                        "rejection_feedback": feedback,
                    },
                )
                return True, exit_message
            return False, {
                "output": feedback,
                "returncode": 2,
                "exception_info": "",
                "extra": {"resolve_native_submission_rejected": True},
            }

        exit_message = self.model.format_message(
            role="exit",
            content=self._last_canonical_patch,
            extra={
                "exit_status": "Submitted",
                "submission": self._last_canonical_patch,
                "resolve_gate": {
                    "patch_hash": self._last_patch_hash,
                    "base_clean_fail": True,
                    "test_command": report["test_command"],
                    "official_verifier_kind": report.get("official_verifier_kind"),
                    "submission_mode": "native_bash",
                    "behavior_contract": self._last_test_contract,
                },
            },
        )
        return True, exit_message

    @staticmethod
    def _added_test_nodes(patch: str) -> list[str]:
        nodes = []
        pattern = re.compile(r"^\+\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(")
        for line in patch.splitlines():
            if line.startswith("+++"):
                continue
            match = pattern.match(line)
            if match and match.group(1) not in nodes:
                nodes.append(match.group(1))
        return nodes

    def _exact_selector_rejection(
        self, patch: str, test_command: str, contract_nodes: list[str] | None
    ) -> str:
        added_nodes = self._added_test_nodes(patch)
        selected_nodes = contract_nodes if contract_nodes is not None else added_nodes
        if not selected_nodes:
            return (
                "behavior-contract requires at least one exact test node. Add a test_* function/method, or declare "
                "an existing test node exercised by changed fixture, parameter, helper, or data files."
            )
        if len(selected_nodes) != len(set(selected_nodes)):
            return "behavior contract test_nodes must not contain duplicates."
        undeclared_added = [node for node in added_nodes if node not in selected_nodes]
        if undeclared_added:
            return (
                "behavior contract test_nodes must include every test_* function or method added by "
                f"the patch (undeclared={undeclared_added!r}, contract={selected_nodes!r})."
            )
        missing_selectors = [
            node
            for node in selected_nodes
            if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(node)}(?![A-Za-z0-9_])", test_command)
        ]
        if missing_selectors:
            return (
                "test_command must explicitly select every behavior-contract test node "
                f"(missing={missing_selectors!r}); file-, class-, directory-, and suite-level "
                "selectors are rejected."
            )
        return ""

    def _validate_test_contract(
        self, contract_text: str, patch: str, test_command: str
    ) -> tuple[dict | None, str]:
        if not contract_text:
            return None, "missing non-empty contract file"
        try:
            contract = json.loads(contract_text)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc.msg}"
        if not isinstance(contract, dict):
            return None, "top-level value must be an object"

        for key in ("entrypoint", "trigger", "expected_output"):
            if not isinstance(contract.get(key), str) or not contract[key].strip():
                return None, f"{key} must be a non-empty string"
        for key in ("issue_evidence", "alternative_hypotheses", "non_assertions", "test_nodes"):
            values = contract.get(key)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                return None, f"{key} must be a non-empty list of non-empty strings"

        contract_nodes = [value.strip() for value in contract["test_nodes"]]
        if getattr(self.config, "require_exact_test_selector", False):
            if reason := self._exact_selector_rejection(patch, test_command, contract_nodes):
                return None, reason
        contract = {
            **contract,
            "entrypoint": contract["entrypoint"].strip(),
            "trigger": contract["trigger"].strip(),
            "expected_output": contract["expected_output"].strip(),
            "issue_evidence": [value.strip() for value in contract["issue_evidence"]],
            "alternative_hypotheses": [value.strip() for value in contract["alternative_hypotheses"]],
            "non_assertions": [value.strip() for value in contract["non_assertions"]],
            "test_nodes": contract_nodes,
        }
        return contract, ""

    @staticmethod
    def _submitted_patch(submitted: Submitted) -> str:
        for message in submitted.messages:
            patch = (message.get("extra") or {}).get("submission")
            if isinstance(patch, str):
                return patch
        return ""

    def _read_workspace_file(self, path: str) -> str:
        quoted = shlex.quote(path)
        output = self.env.execute({"command": f"test -s {quoted} && cat -- {quoted}"}, timeout=30)
        if output.get("returncode") != 0:
            return ""
        return str(output.get("output") or "").strip()

    @staticmethod
    def _rejection(reason: str) -> dict:
        return {
            "output": f"NATIVE TEST-PATCH SUBMISSION BLOCKED: {reason}\nRevise the workspace and submit again.",
            "returncode": 2,
            "exception_info": "",
            "extra": {"resolve_native_submission_rejected": True},
        }
