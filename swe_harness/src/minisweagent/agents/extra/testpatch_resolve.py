"""TestPatchResolveAgent — resolve-style, candidate-free regression-test generation.

Design: RESOLVE_STYLE_TESTPATCH_DESIGN.md (v0.2, fork). This is a NEW minimal agent, not a
subclass of GeneratedTestSubmitAgent — it deliberately drops the v6 new-file / declared-command
baggage.

Protocol per turn:
  1. First response MUST be exactly one grounding_plan.
  2. Then free bash exploration in the GENERATOR sandbox (self.env) — running tests is allowed.
  3. write_test_patch(diff, command, hint?) -> harness validates the diff, then hands it to the injected
     OFFICIAL verify runner (dataset eval_script in a dedicated warm verifier sandbox: reset ->
     git apply test_patch -> run command). Returns a base-clean-fail feedback block + patch_hash.
  4. The MODEL alone decides: explore more, write again, or submit_test_patch(patch_hash).
  5. No auto-submit, no force-submit. Step-limit with no submit => LimitsExceeded (void).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from minisweagent.agents.default import DefaultAgent

WRITE_TOOL = "write_test_patch"
SUBMIT_TOOL = "submit_test_patch"
PLAN_TOOL = "grounding_plan"


class TestPatchResolveConfig(BaseModel):
    system_template: str = "You are a maintainer writing a regression test for a bug."
    instance_template: str
    step_limit: int = 40
    cost_limit: float = 5.0
    time_limit: float = 0.0
    """Stop agent after exceeding this wall-clock time (seconds) since run() started. 0 disables."""
    output_path: Path | None = None
    require_grounding_plan_first: bool = True
    verifier_timeout: int = 900


def _fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()[:16]


class TestPatchResolveAgent(DefaultAgent):
    def __init__(
        self,
        model,
        env,
        *,
        verify_runner: Callable[[str, str], dict] | None = None,
        config_class: type[TestPatchResolveConfig] = TestPatchResolveConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        # verify_runner(test_patch, test_command) -> classified dict. Injected by the runner; it
        # runs the dataset's OFFICIAL eval script in a dedicated, kept-warm verifier sandbox
        # (reset -> git apply test_patch -> run command). All env handling is upstream-correct.
        self._verify_runner = verify_runner
        self._grounding_plan: dict | None = None
        self._last_patch: str = ""
        self._last_canonical_patch: str = ""
        self._last_patch_hash: str = ""
        self._last_report: dict | None = None

    # --- control flow -------------------------------------------------------

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        if self.config.require_grounding_plan_first and self._grounding_plan is None:
            if not (len(actions) == 1 and actions[0].get("tool") == PLAN_TOOL):
                return self.add_messages(
                    self.model.format_message(
                        role="user",
                        content=(
                            "PROTOCOL: your first response must be exactly one grounding_plan tool call "
                            "before any bash, write_test_patch, or submit."
                        ),
                    )
                )
        outputs = []
        for action in actions:
            tool = action.get("tool")
            if tool == PLAN_TOOL:
                outputs.append(self._record_plan(action))
            elif tool == WRITE_TOOL:
                outputs.append(self._handle_write(action))
            elif tool == SUBMIT_TOOL:
                exit_msg = self._handle_submit(action)
                if exit_msg is not None:
                    self.add_messages(*self.model.format_observation_messages(message, self._pad(outputs, actions), self.get_template_vars()))
                    self.add_messages(exit_msg)
                    return [exit_msg]
                outputs.append(self._reject_submit())
            else:  # bash — free exploration in generator sandbox
                outputs.append(self.env.execute(action))
        return self.add_messages(
            *self.model.format_observation_messages(message, self._pad(outputs, actions), self.get_template_vars())
        )

    @staticmethod
    def _pad(outputs: list[dict], actions: list[dict]) -> list[dict]:
        not_run = {"output": "", "returncode": -1, "exception_info": "not executed", "extra": {}}
        for o in outputs:
            o.setdefault("extra", {})
            o.setdefault("exception_info", None)
            o.setdefault("returncode", 0)
        return outputs + [not_run] * (len(actions) - len(outputs))

    # --- grounding plan -----------------------------------------------------

    def _record_plan(self, action: dict) -> dict:
        self._grounding_plan = {k: v for k, v in action.items() if k not in ("tool", "tool_call_id")}
        return {
            "output": "Recorded grounding_plan. Now explore the repo with bash (running tests is allowed), "
            "then submit a test_patch.",
            "returncode": 0,
            "extra": {"resolve_grounding_plan": True},
        }

    # --- write_test_patch ---------------------------------------------------

    def _handle_write(self, action: dict) -> dict:
        test_command = action.get("test_command")
        if not isinstance(test_command, str) or not test_command.strip():
            return {
                "output": "write_test_patch rejected: missing non-empty test_command",
                "returncode": 2,
                "extra": {"resolve_write_rejected": "missing non-empty test_command"},
            }
        test_command = test_command.strip()
        # Extract the patch from the workspace via git diff (mini-swe-agent v2 pattern): the model
        # has edited real files with bash; git generates a correct unified diff. The model never
        # hand-writes hunk headers, so there is no UnidiffParseError.
        patch = self._extract_workspace_diff()
        if not patch.strip():
            return {
                "output": "write_test_patch rejected: `git diff` in /testbed is empty — no file edits "
                "detected. Edit the test file(s) with bash first (sed/python/cat), then call "
                "write_test_patch.",
                "returncode": 2,
                "extra": {"resolve_write_rejected": "empty workspace diff"},
            }
        ok, reason, _ = self._patch_safety_check(patch)
        if not ok:
            return {"output": f"write_test_patch rejected: {reason}", "returncode": 2,
                    "extra": {"resolve_write_rejected": reason}}
        report = self._verify_patch(patch, test_command)
        if report.get("error"):
            return {"output": f"verifier error: {report['error']}", "returncode": 2,
                    "extra": {"resolve_verifier_error": report["error"]}}
        self._last_patch = patch
        self._last_canonical_patch = report["canonical_patch"]
        self._last_patch_hash = report["patch_hash"]
        self._last_report = report
        return {
            "output": self._format_feedback(report),
            "returncode": 0,
            "extra": {"resolve_write": True, "patch_hash": report["patch_hash"],
                      "base_clean_fail": report.get("clean_fail")},
        }

    def _extract_workspace_diff(self) -> str:
        # git diff over tracked edits + include newly-added untracked files (git add -N stages
        # intent-to-add so they appear in the diff). Runs in the generator workspace (self.env).
        out = self.env.execute(
            {"command": "cd /testbed && git add -AN >/dev/null 2>&1; git diff --no-color"},
            timeout=60,
        )
        return str(out.get("output") or "")

    @staticmethod
    def _patch_safety_check(patch: str) -> tuple[bool, str, list[str]]:
        if not patch.strip():
            return False, "empty patch", []
        changed = _changed_paths_from_patch(patch)
        if not changed:
            return False, "no file paths parsed from diff", []
        for p in changed:
            if p.startswith("/") or ".." in p.split("/"):
                return False, f"unsafe path: {p}", []
        return True, "", changed

    def _verify_patch(self, patch: str, test_command: str) -> dict:
        # Verify via the injected OFFICIAL eval runner (dataset's own eval_script): it resets the
        # repo, activates conda, installs (or skips after warmup), git-applies the test_patch, and
        # runs the model-provided test command unchanged. All env handling is upstream-correct.
        # gold-free: we only
        # read the classified base result, never gold FAIL_TO_PASS.
        if self._verify_runner is None:
            return {"error": "no verify_runner injected (official eval unavailable)"}
        try:
            classified = self._verify_runner(patch, test_command)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        clean_fail = bool(classified.get("clean_fail"))
        return {
            "clean_fail": clean_fail,
            "verdict": classified.get("verdict"),
            "passed": int(classified.get("passed") or 0),
            "failed": int(classified.get("failed") or 0),
            "errors": int(classified.get("errors") or 0),
            "infra_failure": bool(classified.get("infra_failure")),
            "failure_category": classified.get("failure_category"),
            "official_verifier_kind": classified.get("official_verifier_kind"),
            "tail": (classified.get("official_output_tail") or classified.get("tail") or "")[-1500:],
            "test_command": test_command,
            "canonical_patch": patch,
            "patch_hash": _fingerprint(f"{patch}\0{test_command}"),
        }

    def _format_feedback(self, report: dict) -> str:
        clean = report.get("clean_fail")
        lines = [
            "test_patch_gate:",
            f"  base_clean_fail: {bool(clean)}",
            f"  verdict: {report.get('verdict')}  passed={report.get('passed')} "
            f"failed={report.get('failed')} errors={report.get('errors')}",
            f"  failure_category: {report.get('failure_category')}",
            f"  official_verifier_kind: {report.get('official_verifier_kind')}",
            f"  test_command: {report.get('test_command')}",
            f"  patch_hash: {report.get('patch_hash')}",
            "",
        ]
        if clean:
            lines.append("  The test cleanly FAILS on the buggy base for an assertion/behavior reason — "
                         "this is the target signal. If it matches the issue (asserts the CORRECTED "
                         "behavior, not the bug symptom), call submit_test_patch with the patch_hash above.")
        elif report.get("infra_failure"):
            lines.append("  The test did NOT actually run (import/collection/env error). Fix imports, "
                         "test path, or the exercised API, then write_test_patch again.")
        else:
            lines.append("  The test did not cleanly fail on base (it passed or was inconclusive), so it "
                         "does not yet reproduce the bug. Strengthen the assertion for the corrected "
                         "behavior, then write_test_patch again.")
        lines += ["", "base_test_output:", "<tail>", report.get("tail") or "", "</tail>"]
        return "\n".join(lines)

    # --- submit -------------------------------------------------------------

    def _handle_submit(self, action: dict):
        want = action.get("patch_hash") or ""
        if not self._last_patch_hash or want != self._last_patch_hash:
            return None  # caller emits reject
        return self.model.format_message(
            role="exit",
            content=self._last_canonical_patch,
            extra={
                "exit_status": "Submitted",
                "submission": self._last_canonical_patch,
                "resolve_gate": {
                    "patch_hash": self._last_patch_hash,
                    "base_clean_fail": (self._last_report or {}).get("clean_fail"),
                    "test_command": (self._last_report or {}).get("test_command", ""),
                    "official_verifier_kind": (self._last_report or {}).get("official_verifier_kind"),
                    "grounding_plan": self._grounding_plan,
                },
            },
        )

    def _reject_submit(self) -> dict:
        return {
            "output": "submit_test_patch rejected: patch_hash does not match the latest write_test_patch. "
            "Write a test_patch first, then submit its reported patch_hash.",
            "returncode": 2,
            "extra": {"resolve_submit_rejected": True},
        }


# --- module-level patch helpers ---------------------------------------------

_DIFF_HDR = re.compile(r"^\+\+\+ b/(?P<path>\S+)", re.M)
_DIFF_HDR_A = re.compile(r"^--- a/(?P<path>\S+)", re.M)


def _changed_paths_from_patch(patch: str) -> list[str]:
    paths = set()
    for m in _DIFF_HDR.finditer(patch):
        p = m.group("path")
        if p != "/dev/null":
            paths.add(p)
    for m in _DIFF_HDR_A.finditer(patch):
        p = m.group("path")
        if p != "/dev/null":
            paths.add(p)
    return sorted(paths)
