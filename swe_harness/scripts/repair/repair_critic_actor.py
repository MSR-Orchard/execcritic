#!/usr/bin/env python3
"""Repair loop for SWE-bench trajectories: apply -> collect feedback -> summarize
-> critic diagnosis -> repair actor -> verify.

Pipeline (per selected instance, in ONE sandbox):

  Stage A  apply previous patch to the ORIGINAL repo working tree.
           A failed apply is itself recorded as feedback.
  Stage B  collect STRUCTURED environment feedback deterministically: re-run the
           previous attempt's own test/repro commands on the patched repo and
           capture command, exit code, failing test names, traceback snippets,
           changed files. (Non-oracle: uses the prior attempt's own checks, not
           the gold FAIL_TO_PASS tests.)
  Stage C  trajectory summary  (1 model call, non-agentic) — faithful summary of
           the previous attempt: actions, files, assumptions, what passed/failed.
  Stage D  critic diagnosis    (1 model call, non-agentic) — reads task + prev
           patch + changed files + summary + structured feedback + raw snippets,
           produces a root-cause diagnosis and a keep/modify/revert recommendation.
           Separated from the actor to reduce self-justification.
  Stage E  repair actor        (agentic, patch already applied) — verify-first;
           keep the patch if it already works, otherwise apply the minimal fix.

Both raw evidence (real traceback / failing-test lines) and compressed summaries
are fed forward, so the critical failure line is never summarized away.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import importlib.util
import json
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "src"))
sys.path.insert(0, str(HARNESS_ROOT / "external" / "azure-modal"))

import litellm  # noqa: E402
from datasets import load_dataset  # noqa: E402

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.exceptions import InterruptAgentFlow  # noqa: E402
from minisweagent.config import get_config_from_spec  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from minisweagent.run.benchmarks import swebench as swebench_runner  # noqa: E402
from minisweagent.run.benchmarks import swerebench as swerebench_runner  # noqa: E402
from minisweagent.utils.log import add_file_handler, logger  # noqa: E402
from minisweagent.utils.serialize import UNSET, recursive_merge  # noqa: E402

RUNNERS = {
    "swebench": swebench_runner,
    "swerebench": swerebench_runner,
}
DATASET_MAPPING = {**swebench_runner.DATASET_MAPPING, **swerebench_runner.DATASET_MAPPING}


def get_benchmark_runner(benchmark: str):
    try:
        return RUNNERS[benchmark]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark: {benchmark}") from exc


def load_repair_module():
    """Reuse select_instances / run_verify / run_tally / summaries from the
    single-call repair script without copy-pasting them."""
    path = HARNESS_ROOT / "scripts" / "repair" / "repair_self_test_fail_unknown.py"
    if not path.exists():
        path = HARNESS_ROOT / "scripts" / "repair" / "cache" / "repair_self_test_fail_unknown.py"
    spec = importlib.util.spec_from_file_location("repair_self_test_fail_unknown", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load_repair_module()

SYSTEM_TEMPLATE = "You are a careful software engineer that interacts with a computer shell to solve programming tasks."


# ---- Stage C: trajectory summary (single completion) ----
SUMMARY_SYS = "You faithfully summarize a previous coding attempt so a repair stage can build on it. Be concise but keep concrete details (file/function names, what was verified)."


def summary_user_prompt(task: str, trajectory_digest: str, prior_patch: str) -> str:
    return f"""Summarize the PREVIOUS attempt at this task for a repair stage.

<original_task>
{task}
</original_task>

<previous_trajectory>
{trajectory_digest}
</previous_trajectory>

<previous_patch>
{prior_patch}
</previous_patch>

Produce a concise summary covering: what actions were taken, which files/functions were modified, what assumptions the previous attempt seemed to make, what verification was attempted, what passed, what failed, and what risks/uncertainties remain. Plain text, <= 25 lines."""


# ---- Stage D: critic diagnosis (single completion) ----
CRITIC_SYS = "You are a critical reviewer (NOT the original author). You read evidence and diagnose the root cause of a failure. Do not assume the previous patch is correct; question its assumptions."


def critic_user_prompt(task: str, prior_patch: str, changed_files: str, summary: str, feedback: str) -> str:
    return f"""Diagnose why the previous attempt is failing or uncertain, using the evidence below.

<original_task>
{task}
</original_task>

<previous_patch>
{prior_patch}
</previous_patch>

<changed_files>
{changed_files}
</changed_files>

<trajectory_summary>
{summary}
</trajectory_summary>

<environment_feedback>
{feedback}
</environment_feedback>

Decision rules (follow STRICTLY):
- NOT-APPLIED OVERRIDE: if the feedback begins with "PRIOR PATCH DID NOT APPLY", there is NO patch in the tree — KEEP is IMPOSSIBLE. You MUST output RECREATE and give a from-scratch fix plan.
- KEEP-GATE: output MODIFY or REVERT ONLY if at least one check shows a REAL failure — a non-empty failing_tests entry that is a genuine test node (NOT an import/loader/collection token), OR a reproduced traceback/assertion you can quote, OR a check tagged BROKEN-BY-PATCH. If every check has exit_code 0, or the only "failures" are import errors / "No module named" / missing file / "collected 0 items" / "no tests ran" / an empty AssertionError / debug-print experiments / a check tagged NON-SIGNAL or EXIT-CHANGED, you MUST output KEEP. Never invent a failure that is not present verbatim in the feedback; unverifiable feedback is evidence FOR keeping, not license to refactor.
- SOURCE-GROUNDED EXCEPTION: you MAY output MODIFY without a failing test node ONLY if you can cite a CONCRETE, code-grounded contract/logic violation at a specific line of the previous patch (quote it); in that case the FIX PLAN MUST instruct the actor to first synthesize a reproduction that fails before fixing. Do NOT use this for vague/speculative worries.
- A check tagged FIXED-BY-PATCH is positive proof the patch works for that scenario — prefer KEEP unless another check is BROKEN-BY-PATCH.
- TESTS ARE THE ORACLE: repository test files (e.g. under tests/) are immutable ground truth. NEVER recommend editing a test to match the patch. A failing repository test ALWAYS means the patch is wrong — fix the SOURCE to emit exactly the expected output/SQL/shape, never the test. Do NOT excuse a failing test as "stale/pre-existing/environmental" UNLESS the check is explicitly tagged base_verdict=PRE-EXISTING (fails without the patch too).
- Ground every structural claim (a method/override/hook exists, a helper recurses, etc.) in evidence present in the feedback; never assert code structure from memory.

Output (plain text, <= 20 lines):
1. VERDICT: exactly one of KEEP | MODIFY | REVERT on its own first line. KEEP = submit the prior patch unchanged; MODIFY = edit the applied patch; REVERT only when an APPLIED patch is actively wrong (if prior_patch did not apply, use MODIFY/RE-CREATE, never REVERT).
2. ROOT CAUSE: the specific cause; when MODIFY/REVERT, quote the actual failing test name / assertion from the feedback.
3. FIX PLAN (omit if KEEP): the SINGLE minimal primary edit that makes the failing test pass — name file/function and the exact change; no unrelated edits."""


# ---- Stage E: repair actor (agentic, patch pre-applied) ----
REPAIR_TEMPLATE = """\
You are making a SECOND attempt at the following issue. Working directory: /testbed.

<pr_description>
{{task}}
</pr_description>

<diagnosis_from_critic>
{{diagnosis}}
</diagnosis_from_critic>

<previous_attempt_summary>
{{summary}}
</previous_attempt_summary>

<environment_feedback>
{{feedback}}
</environment_feedback>

# Critic VERDICT: {{verdict}}   (pre_applied={{patch_applied}})
Act on the verdict — do NOT just resubmit the existing tree:
- MODIFY: the patch is applied but flawed. Apply the critic's fix to SOURCE, then RE-RUN the named failing test. Resubmitting the unchanged patch is a FAILURE.
- REVERT: the working tree has been RESET to the clean base (the bad patch is gone). Re-implement the fix correctly from the diagnosis, then run the test.
- RECREATE (or pre_applied=False): there is NO patch in the tree. Implement the fix from scratch per the diagnosis and the issue, then run the test.
- KEEP: the applied patch is endorsed. Verify it passes the relevant test, then submit the prior diff UNCHANGED.

1. Inspect the tree: `cd /testbed && git diff --stat` and `git diff`.
2. VERIFY-FIRST: run the named failing test / reproduction the diagnosis points to (before your edit).
3. Make the SINGLE minimal change the critic's FIX PLAN specifies (for MODIFY/REVERT/RECREATE), then re-run that exact test until it passes. Do not pile on unrelated changes.
4. Do NOT modify test files or configuration; only non-test source files.
- Every command runs in a fresh subshell; prefix with `cd /testbed && ...`.

# Submission
The final patch MUST be the FULL fix = the already-applied previous patch PLUS any changes you made.
As SEPARATE commands:
Step 1: `cd /testbed && git diff --name-only`   (shows every changed file incl. the pre-applied patch)
Step 2: `cd /testbed && git diff -- <non-test source files> > patch.txt`   (include ALL such files; do NOT commit)
Step 3: verify patch.txt shows `--- a/` / `+++ b/` headers and only intended changes.
Step 4: submit with EXACTLY (must exit 0):
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```
Creating/viewing the patch and submitting MUST be separate commands. You CANNOT continue after submitting.

HARD RULES:
- If the critic VERDICT is KEEP and the applied patch passes its re-run, submit the prior diff UNCHANGED — do not rewrite endorsed logic or swap libraries/services.
- Your submission MUST be a valid unified diff starting with `diff --git`. NEVER submit shell error text (e.g. `cat: patch.txt: No such file`), a placeholder, or an empty result. Before Step 4, confirm `patch.txt` is non-empty and starts with `diff --git`; if not, regenerate it with `git diff`.
- Before submitting any file you edited, run `python -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" <file>` and fix it if it fails to parse.
"""


def _msg_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return ""


def model_complete(model, system: str, user: str) -> str:
    """Plain text completion (no tools). model.query() forces the BASH tool and
    parses for an action, so it cannot be used for free-form summary/critic calls."""
    cfg = model.config
    drop = {"tools", "tool_choice", "parallel_tool_calls"}
    kwargs = {k: v for k, v in cfg.model_kwargs.items() if k not in drop}
    kwargs.setdefault("drop_params", True)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last = ""
    for attempt in range(4):
        try:
            resp = litellm.completion(model=cfg.model_name, messages=messages, **kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            time.sleep(2 * (attempt + 1))
    logger.warning("model_complete failed after retries: %s", last)
    return ""


def shell(env, command: str, timeout: int = 120) -> dict:
    try:
        return env.execute({"command": command}, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return {"output": str(exc), "returncode": -1}


def parse_changed_files(patch: str) -> list[str]:
    files = re.findall(r"(?m)^\+\+\+ b/(.+)$", patch)
    if not files:
        files = re.findall(r"(?m)^diff --git a/\S+ b/(\S+)$", patch)
    return list(dict.fromkeys(f.strip() for f in files))


def apply_prior_patch(env, patch: str) -> tuple[bool, str]:
    if not patch.strip():
        return False, "previous patch was empty"
    b64 = base64.b64encode(patch.encode("utf-8")).decode("ascii")
    shell(env, f"printf %s '{b64}' | base64 -d > /tmp/prev.patch")
    for cmd in (
        "cd /testbed && git apply --whitespace=nowarn /tmp/prev.patch",
        "cd /testbed && git apply --3way --whitespace=nowarn /tmp/prev.patch",
        "cd /testbed && patch -p1 --fuzz=3 < /tmp/prev.patch",
    ):
        res = shell(env, cmd)
        if res.get("returncode") == 0:
            return True, ""
        last_err = (res.get("output") or "")[-1500:]
    return False, last_err


_SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def _command_from_arguments(arguments) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) or {}
        except Exception:  # noqa: BLE001
            return ""
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("command") or "").strip()


def extract_replay_commands(messages: list[dict]) -> list[str]:
    """Pull the ordered list of shell commands the ORIGINAL actor ran, from the
    normalized ``extra.actions`` when available, then from chat tool calls or
    Responses API function-call items. Falls back to a content-embedded
    ``{"command": ...}`` for the text protocol. The final submit sentinel is
    dropped because it only returns the already-materialized patch."""
    cmds: list[str] = []
    for m in messages:
        actions = (m.get("extra") or {}).get("actions") or []
        action_commands = [
            str(action.get("command") or "").strip()
            for action in actions
            if isinstance(action, dict) and action.get("command")
        ]
        if action_commands:
            cmds.extend(
                command for command in action_commands if _SUBMIT_SENTINEL not in command
            )
            continue

        response_items = []
        if m.get("object") == "response":
            response_items = m.get("output") or []
        elif m.get("type") == "function_call":
            response_items = [m]
        response_commands = [
            _command_from_arguments(item.get("arguments"))
            for item in response_items
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        response_commands = [command for command in response_commands if command]
        if response_commands:
            cmds.extend(
                command for command in response_commands if _SUBMIT_SENTINEL not in command
            )
            continue

        if m.get("role") != "assistant":
            continue
        tool_commands = [
            _command_from_arguments((tool_call.get("function") or {}).get("arguments"))
            for tool_call in m.get("tool_calls") or []
            if isinstance(tool_call, dict)
        ]
        tool_commands = [command for command in tool_commands if command]
        if tool_commands:
            for cmd in tool_commands:
                if _SUBMIT_SENTINEL not in cmd:
                    cmds.append(cmd)
            continue

        if not m.get("tool_calls"):  # text-protocol fallback
            text = _msg_text(m)
            mm = re.search(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if not mm:
                continue
            cmd = mm.group(1).encode().decode("unicode_escape").strip()
            if cmd and _SUBMIT_SENTINEL not in cmd:
                cmds.append(cmd)
    return cmds


def replay_trajectory(env, messages: list[dict], *, cmd_timeout: int, budget: int) -> dict:
    """Reconstruct the ORIGINAL sandbox state by re-running every command the prior
    actor issued, in order, on the fresh (clean-base) sandbox. This recovers state
    that `git apply` of the final diff cannot: pip installs, generated/untracked
    files, build artifacts, working-dir mutations. Per-command failures are kept
    (an originally-failing command faithfully fails again); a wall-clock `budget`
    stops replay early so a long original run can't blow the instance timeout."""
    cmds = extract_replay_commands(messages)
    stats = {"total": len(cmds), "ran": 0, "failed": 0, "truncated": False, "seconds": 0.0}
    start = time.time()
    for cmd in cmds:
        if time.time() - start > budget:
            stats["truncated"] = True
            break
        res = shell(env, cmd, timeout=cmd_timeout)
        stats["ran"] += 1
        if (res.get("returncode") or 0) != 0:
            stats["failed"] += 1
    stats["seconds"] = round(time.time() - start, 1)
    return stats


def _submitted_patch_is_present(env, patch: str) -> tuple[bool, str]:
    if not patch.strip():
        return False, "previous patch was empty"
    b64 = base64.b64encode(patch.encode("utf-8")).decode("ascii")
    shell(env, f"printf %s '{b64}' | base64 -d > /tmp/replayed_submission.patch")
    result = shell(
        env,
        "cd /testbed && git apply -R --check --whitespace=nowarn "
        "/tmp/replayed_submission.patch 2>&1",
    )
    shell(env, "rm -f /tmp/replayed_submission.patch")
    return result.get("returncode") == 0, str(result.get("output") or "")[-1500:]


def restore_state(
    env,
    messages: list[dict],
    prior_patch: str,
    args,
    *,
    strict_replay: bool = False,
) -> tuple[bool, str, dict]:
    """Bring the working tree to the prior attempt's end-state. In `replay` mode,
    re-run the original commands to rebuild full env state, then top up with the
    patch if replay did not reproduce a diff; otherwise just apply the patch.

    ``strict_replay`` is for experiments that require a faithful continuation of
    the original workspace. It rejects a truncated replay or a resulting tree
    that does not contain the original submitted patch instead of silently
    falling back to patch-only restoration.

    Returns (patch_present, apply_error, replay_stats)."""
    if args.restore_mode != "replay":
        applied, apply_err = apply_prior_patch(env, prior_patch)
        return applied, apply_err, {}
    stats = replay_trajectory(env, messages, cmd_timeout=args.replay_cmd_timeout, budget=args.replay_budget)
    patch_present, patch_error = _submitted_patch_is_present(env, prior_patch)
    stats["prior_patch_present"] = patch_present
    if strict_replay:
        if stats["truncated"]:
            return False, "trajectory replay exceeded its wall-clock budget", stats
        if not patch_present:
            return False, f"trajectory replay did not reproduce the submitted patch: {patch_error}", stats
        return True, "", stats

    cur_diff = shell(
        env,
        "cd /testbed && git -c core.fileMode=false diff HEAD --binary",
    ).get("output", "") or ""
    if _looks_like_diff(cur_diff):
        # replay reproduced the actor's edits; the full env state is also in place
        return True, "", stats
    # replay produced no diff (edits not reproduced) — keep the rebuilt env state but
    # guarantee the patch is present so repair operates on at least the patched tree
    applied, apply_err = apply_prior_patch(env, prior_patch)
    stats["topped_up_with_patch"] = True
    return applied, apply_err, stats


FAIL_NAME_RE = re.compile(r"(?m)^(?:FAILED|ERROR)\s+(\S+)")
FAIL_NAME_RE2 = re.compile(r"(?m)^(?:FAIL|ERROR):\s+(\S+)")
_JUNK_FAIL_TOKENS = {"tests", "test", "errors", "failures"}
_NOISE_MARKERS = (
    "no module named", "no such file", "modulenotfounderror", "importerror",
    "doesn't declare an explicit app_label", "improperlyconfigured",
    "collected 0 items", "ran 0 tests", "no tests ran", "no tests collected",
    "command not found", "cannot import name", "unrecognized arguments",
    "_failedtest", "failed to import test module", "has no attribute",
)
# external-service flakiness (e.g. httpbin 5xx) is not a code signal
_EXT_5XX_RE = re.compile(r"\b(50[0-9])\s+(server error|service unavailable|bad gateway|internal)", re.I)
# an in-script assertion/repro failure even when the process exits 0
_SCRIPT_FAIL_RE = re.compile(r"AssertionError|Traceback \(most recent call last\)", re.I)


def extract_failing(output: str) -> list[str]:
    names = FAIL_NAME_RE.findall(output) + FAIL_NAME_RE2.findall(output)
    out: list[str] = []
    for n in names:
        if n.strip().lower() in _JUNK_FAIL_TOKENS or n.startswith("(") or not re.search(r"[A-Za-z]", n):
            continue
        out.append(n)
    return list(dict.fromkeys(out))[:25]


def extract_traceback(output: str) -> str:
    idx = output.rfind("Traceback (most recent call last):")
    return output[idx : idx + 2000] if idx >= 0 else ""


def _is_noise(cmd: str, output: str) -> bool:
    o = (output or "").lower()
    if any(m in o for m in _NOISE_MARKERS):
        return True
    if _EXT_5XX_RE.search(output or ""):
        return True
    return bool(re.search(r"git\s+(checkout|stash|reset|restore)\b", cmd or ""))


def prior_test_commands(events: list[dict], max_checks: int) -> list[str]:
    cmds: list[str] = []
    for ev in events:
        cmd = ev.get("command", "")
        m = re.search(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', cmd)
        if m:
            cmd = m.group(1).encode().decode("unicode_escape")
        cmd = cmd.strip()
        if cmd and cmd not in cmds:
            cmds.append(cmd)
    return cmds[-max_checks:]


def run_check(env, cmd: str, timeout: int = 240) -> dict:
    """Run one check capturing the REAL exit code via PIPESTATUS (so a trailing
    `2>&1 | tail` cannot mask a crash as exit 0) and tag infra/harness noise."""
    run_cmd = cmd if cmd.strip().startswith("cd ") else f"cd /testbed && {cmd}"
    run_cmd = run_cmd + ' ; echo "__MSWE_RC__=${PIPESTATUS[0]:-$?}"'
    res = shell(env, run_cmd, timeout=timeout)
    out = res.get("output", "") or ""
    m = re.search(r"__MSWE_RC__=(-?\d+)\s*$", out.strip())
    rc = int(m.group(1)) if m else res.get("returncode")
    out = re.sub(r"\n?__MSWE_RC__=-?\d+\s*$", "", out)
    fails = extract_failing(out)
    noise = _is_noise(cmd, out)
    tb = extract_traceback(out)
    script_fail = bool(_SCRIPT_FAIL_RE.search(out))  # in-script assertion even on exit 0
    has_assertion = bool(fails) or bool(tb) or script_fail
    return {
        "command": cmd, "exit_code": rc, "failing_tests": fails,
        "traceback": tb, "output_tail": out[-2200:], "signal": not noise,
        "has_assertion": has_assertion,
        "passed": (rc == 0 and not fails and not noise and not script_fail),
    }


def build_feedback(env, applied: bool, apply_err: str, base_checks: list[dict], patched_checks: list[dict]) -> dict:
    """Combine clean-tree control runs with patched runs so the critic can tell a
    real defect (BROKEN/PRE-EXISTING) from a fix (FIXED) from noise (NON-SIGNAL)."""
    changed = shell(env, "cd /testbed && git diff --name-only").get("output", "").strip()
    base_by_cmd = {c["command"]: c for c in base_checks}
    checks = []
    for pc in patched_checks:
        bc = base_by_cmd.get(pc["command"])
        if not pc["signal"]:
            verdict = "NON-SIGNAL (infra/harness noise; ignore)"
        elif bc is None or not bc["signal"]:
            verdict = "patched-signal-only (no clean-tree control)"
        elif not bc["passed"] and pc["passed"]:
            verdict = "FIXED-BY-PATCH (fails without patch, passes with it)"
        elif bc["passed"] and not pc["passed"]:
            if pc.get("has_assertion"):
                verdict = "BROKEN-BY-PATCH (passes without patch, real assertion/traceback with it)"
            else:
                verdict = "EXIT-CHANGED (exit code differs but NO assertion/traceback; likely benign — do NOT revert on this alone)"
        elif not bc["passed"] and not pc["passed"]:
            verdict = "PRE-EXISTING (fails with AND without the patch)"
        else:
            verdict = "passes with and without patch (may not exercise the issue)"
        pc = dict(pc)
        pc["base_verdict"] = verdict
        pc["base_exit_code"] = bc["exit_code"] if bc is not None else "?"
        checks.append(pc)
    return {
        "patch_applied": applied,
        "patch_apply_error": apply_err if not applied else "",
        "changed_files": [c for c in changed.splitlines() if c.strip()],
        "checks": checks,
    }


def render_feedback(struct: dict) -> str:
    lines = [f"patch_applied: {struct['patch_applied']}"]
    if struct.get("patch_apply_error"):
        lines.append(f"patch_apply_error:\n{struct['patch_apply_error']}")
    lines.append("changed_files: " + (", ".join(struct["changed_files"]) or "<none>"))
    if not struct["checks"]:
        lines.append("checks: <no prior test/repro command found to re-run>")
    for i, c in enumerate(struct["checks"]):
        lines.append(f"--- check[{i}] ---")
        lines.append(f"command: {c['command']}")
        lines.append(f"exit_code(with_patch): {c['exit_code']}  exit_code(clean_base): {c.get('base_exit_code','?')}  -> {c.get('base_verdict','?')}")
        if c["failing_tests"]:
            lines.append("failing_tests: " + ", ".join(c["failing_tests"]))
        if c["traceback"]:
            lines.append("traceback (raw):\n" + c["traceback"])
        lines.append("output_tail (raw):\n" + c["output_tail"])
    return "\n".join(lines)


def _looks_like_diff(s: str) -> bool:
    s = (s or "").lstrip()
    return s.startswith("diff --git") or s.startswith("--- ") or s.startswith("Index:")


def parse_verdict(diagnosis: str) -> str:
    """Extract the critic's single decisive verdict: KEEP | MODIFY | REVERT | RECREATE."""
    d = diagnosis or ""
    m = re.search(r"\bVERDICT\b\s*[:\-]?\s*\**\s*(KEEP|MODIFY|REVERT|RE-?CREATE|RECREATE)", d, re.I)
    tok = m.group(1).upper().replace("-", "") if m else ""
    if not tok:
        for cand in ("RECREATE", "REVERT", "MODIFY", "KEEP"):
            if re.search(rf"\b{cand}\b", d, re.I):
                tok = cand
                break
    if tok.startswith("RECREATE"):
        return "RECREATE"
    return tok or "MODIFY"


def condense_trajectory(messages: list[dict], limit: int = 12000) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role")
        if role not in ("assistant", "tool"):
            continue
        text = _msg_text(m)
        if not text.strip():
            continue
        tag = "ACTION" if role == "assistant" else "OBSERVATION"
        parts.append(f"[{tag}] {text.strip()[:1200]}")
    digest = "\n".join(parts)
    return digest[-limit:]


# ---- Stage E (continue mode): resume the ORIGINAL tool-calling conversation ----
CONTINUE_ACTION = {
    "MODIFY": "Apply the SINGLE minimal fix the critic named, then re-run the exact failing test to confirm it now passes.",
    "REVERT": "Your previous changes were reverted to the base code. Re-implement a correct minimal fix from scratch, then verify it with the relevant test.",
    "RECREATE": "There is no usable patch in the tree. Implement the fix from scratch, then verify it with the relevant test.",
    "KEEP": "Verify the applied patch passes the relevant test, then submit the existing diff unchanged.",
}


def continue_inject(verdict: str, diagnosis: str, feedback: str) -> str:
    return (
        "[INDEPENDENT REVIEW — your task is NOT finished; keep working in THIS same session]\n"
        "Your attempt above was independently reviewed, and the repository (with your changes still applied in /testbed) was re-checked with deterministic test runs on BOTH the patched tree and the clean base tree. "
        "Do NOT assume your previous patch is correct, and do NOT simply resubmit it without acting on this review.\n\n"
        f"Critic VERDICT: {verdict}\n"
        f"<critic_diagnosis>\n{diagnosis}\n</critic_diagnosis>\n\n"
        f"<environment_recheck>\n{feedback}\n</environment_recheck>\n\n"
        f"Now: {CONTINUE_ACTION.get(verdict, CONTINUE_ACTION['MODIFY'])}\n"
        "Do not modify test files. When done, submit with TWO separate commands: first `git diff -- <source files> > patch.txt`, then `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`."
    )


def prep_history(messages: list[dict], cap: int) -> list[dict]:
    """Make the original conversation safe to resume: drop the trailing exit and any
    dangling assistant tool-call with no observation, then cap length keeping the
    system+task head and the most recent turns (without orphaning a tool message)."""
    msgs = [m for m in messages if m.get("role") != "exit"]
    while msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("tool_calls"):
        msgs.pop()  # incomplete final step (no observation) would break the API
    if cap and len(msgs) > cap:
        head = msgs[:2]
        tail = msgs[-(cap - len(head)):]
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)  # don't start the tail on an orphan observation
        msgs = head + tail
    return msgs


def run_continue_actor(model, env, original_messages: list[dict], injected: str, *, step_limit: int, cost_limit: float, history_cap: int):
    """Resume the original tool-calling conversation with an appended review turn."""
    agent = DefaultAgent(model, env, system_template=SYSTEM_TEMPLATE, instance_template="{{task}}", step_limit=step_limit, cost_limit=cost_limit)
    hist = prep_history(original_messages, history_cap)
    if not hist:
        raise RuntimeError("no resumable history")
    agent.messages = list(hist) + [model.format_message(role="user", content=injected)]
    while True:
        try:
            agent.step()
        except InterruptAgentFlow as e:
            agent.add_messages(*e.messages)
        except Exception as e:  # noqa: BLE001
            agent.handle_uncaught_exception(e)  # records an exit message; do not re-raise
        if agent.messages and agent.messages[-1].get("role") == "exit":
            break
    return agent.messages[-1].get("extra", {}), agent


def _pipeline(iid: str, meta: dict, instance: dict, config: dict, output_dir: Path, args, result_queue) -> None:
    """Full per-instance pipeline; runs in a spawn subprocess so the parent can
    enforce a hard wall-clock timeout. Writes the traj here; returns submission +
    model_name via result_queue so the PARENT writes preds.json (cross-process-safe)."""
    benchmark_runner = get_benchmark_runner(args.benchmark)
    cfg = copy.deepcopy(config)
    benchmark_runner._resolve_per_instance_api_base(cfg, iid)
    prior_patch = R.load_patch_from_traj(meta["traj_data"]) or ""
    prior_traj = meta["traj_data"]
    problem = instance["problem_statement"]
    record = {"instance_id": iid, "category": meta["category"], "prior_patch_nonempty": bool(prior_patch.strip())}
    env = None
    stages: dict = {}
    submission = ""
    model_name = "unknown"
    try:
        model = get_model(config=cfg.get("model", {}))
        model_name = model.config.model_name
        env = benchmark_runner.get_sb_environment(cfg, instance)

        # Stage A0: clean tree; run the prior attempt's checks as a base control
        shell(env, "cd /testbed && git checkout -- . && git clean -fdq")
        cmds = prior_test_commands(meta["events"], args.max_checks)
        base_checks = [run_check(env, c) for c in cmds]

        # Stage A: restore the prior attempt's end-state on the working tree.
        # patch mode: git-apply the final diff only. replay mode: re-run the original
        # trajectory's commands to rebuild full env state (pip installs, generated
        # files, ...) that a bare diff loses, then top up with the patch if needed.
        applied, apply_err, replay_stats = restore_state(env, prior_traj.get("messages", []), prior_patch, args)
        record["prior_patch_applied"] = applied
        if replay_stats:
            record["replay"] = replay_stats

        ext = (getattr(args, "external_decisions", None) or {}).get(iid)
        if ext:
            # EXTERNAL-GENERATOR mode: a strong external generator (e.g. Claude) already
            # produced the env-feedback / summary / decision. Skip the weak model's
            # Stage B/C/D entirely and hand its outputs straight to the actor. This
            # isolates the actor's EXECUTION ability from critic/feedback quality.
            feedback_text = ext.get("feedback") or "<no external feedback>"
            summary = ext.get("summary") or "<no external summary>"
            diagnosis = ext.get("diagnosis") or "<no external diagnosis>"
            verdict = parse_verdict(ext.get("verdict") or diagnosis)
            if not applied and verdict == "KEEP":
                verdict = "RECREATE"
            stages["feedback"] = {"external": True}
            stages["summary"] = summary
            stages["diagnosis"] = diagnosis
            stages["verdict"] = verdict
            record["external_decision"] = True
            record["verdict"] = verdict
            record["diagnosis_chars"] = len(diagnosis)
        else:
            # Stage B: re-run the same checks on the patched tree and compare vs control
            patched_checks = [run_check(env, c) for c in cmds]
            feedback_struct = build_feedback(env, applied, apply_err, base_checks, patched_checks)
            feedback_text = render_feedback(feedback_struct)
            if not applied:
                feedback_text = (
                    "PRIOR PATCH DID NOT APPLY — there is NO patch in the working tree; "
                    "KEEP is impossible and the fix must be RE-CREATED from scratch.\n\n"
                ) + feedback_text
            stages["feedback"] = feedback_struct

            # Stage C: faithful trajectory summary (single call)
            digest = condense_trajectory(prior_traj.get("messages", []))
            summary = model_complete(model, SUMMARY_SYS, summary_user_prompt(problem, digest, R.short_text(prior_patch, 6000) or "<empty>")) or "<no summary>"
            stages["summary"] = summary

            # Stage D: critic diagnosis (single call, separate role)
            changed_files = "\n".join(parse_changed_files(prior_patch)) or "<none parsed>"
            diagnosis = model_complete(
                model, CRITIC_SYS,
                critic_user_prompt(problem, R.short_text(prior_patch, 6000) or "<empty>", changed_files, summary, feedback_text),
            ) or "<no diagnosis>"
            stages["diagnosis"] = diagnosis
            record["diagnosis_chars"] = len(diagnosis)
            record["summary_chars"] = len(summary)

            # parse the decisive verdict; a non-applied patch can never be KEPT
            verdict = parse_verdict(diagnosis)
            if not applied:
                verdict = "RECREATE"
            stages["verdict"] = verdict
            record["verdict"] = verdict

        # Stage E: act on the verdict
        repair_agent = None
        submission, exit_status = "", "Unknown"
        if verdict == "KEEP" and applied and _looks_like_diff(prior_patch):
            # deterministic KEEP: re-emit the endorsed prior diff verbatim. The feedback
            # already verified it, and this avoids the actor regenerating a worse submission.
            submission = prior_patch
            exit_status = "KeepVerbatim"
            record["keep_verbatim"] = True
        else:
            if verdict == "REVERT" and applied:
                # materialize the revert: drop the bad patch so the actor re-implements from base
                shell(env, "cd /testbed && git checkout -- . && git clean -fdq")
                record["reverted_to_base"] = True
            ran = False
            if args.actor_mode == "continue":
                # resume the ORIGINAL conversation with an appended independent-review turn
                try:
                    rinfo, repair_agent = run_continue_actor(
                        model, env, prior_traj.get("messages", []),
                        continue_inject(verdict, R.short_text(diagnosis, 8000), R.short_text(feedback_text, 8000)),
                        step_limit=args.repair_steps, cost_limit=args.repair_cost, history_cap=args.history_cap,
                    )
                    submission = rinfo.get("submission", "") or ""
                    exit_status = rinfo.get("exit_status", "Unknown")
                    record["actor_mode"] = "continue"
                    ran = True
                except Exception as exc:  # noqa: BLE001
                    logger.error("continue actor failed for %s: %s — falling back to fresh", iid, exc)
                    record["continue_fallback"] = str(exc)[:150]
            if not ran:
                repair_agent = DefaultAgent(
                    model, env,
                    system_template=SYSTEM_TEMPLATE,
                    instance_template=REPAIR_TEMPLATE,
                    step_limit=args.repair_steps,
                    cost_limit=args.repair_cost,
                )
                try:
                    rinfo = repair_agent.run(
                        task=problem,
                        diagnosis=R.short_text(diagnosis, 8000),
                        summary=R.short_text(summary, 4000),
                        feedback=R.short_text(feedback_text, 8000),
                        patch_applied=str(applied),
                        verdict=verdict,
                    )
                    submission = rinfo.get("submission", "") or ""
                    exit_status = rinfo.get("exit_status", "Unknown")
                except Exception as exc:  # noqa: BLE001
                    logger.error("repair actor failed for %s: %s", iid, exc)
                    exit_status = type(exc).__name__

            # submission validation gate: never record an empty/error/non-diff stub.
            # Only KEEP may fall back to the prior patch (REVERT/RECREATE empty => genuinely no fix).
            if not _looks_like_diff(submission):
                gd = shell(env, "cd /testbed && git diff").get("output", "") or ""
                if _looks_like_diff(gd):
                    submission = gd
                    record["submission_fallback"] = "git_diff"
                elif verdict == "KEEP" and _looks_like_diff(prior_patch):
                    submission = prior_patch
                    record["submission_fallback"] = "prior_patch"

            # flag verdicts the actor failed to materialize (resubmitted the unchanged patch)
            if verdict in ("MODIFY", "REVERT", "RECREATE") and submission.strip() and submission.strip() == (prior_patch or "").strip():
                record["verdict_not_materialized"] = True

        traj_path = output_dir / iid / f"{iid}.traj.json"
        # persist the replay restore stats (commands ran/failed, truncation, fallback)
        # so the trajectory can be audited later — empty {} in patch mode is omitted.
        info = {"exit_status": exit_status, "submission": submission, **stages, "prior_patch_applied": applied}
        if replay_stats:
            info["replay"] = replay_stats
        traj_info = {"info": info, "instance_id": iid}
        if repair_agent is not None:
            repair_agent.save(traj_path, traj_info)
        else:
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            traj_path.write_text(json.dumps(traj_info, indent=2))
        record["exit_status"] = exit_status
        record["submission_nonempty"] = bool(submission.strip())
    except Exception as exc:  # noqa: BLE001
        logger.error("instance %s crashed: %s", iid, exc, exc_info=True)
        record["error"] = str(exc)
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception:  # noqa: BLE001
                pass
    result = dict(record)
    result["_submission"] = submission
    result["_model_name"] = model_name
    try:
        result_queue.put(result)
    except Exception:  # noqa: BLE001
        pass


def process_one(iid: str, meta: dict, instance: dict, config: dict, output_dir: Path, args) -> dict:
    """Run _pipeline in a spawn subprocess with a hard per-instance wall-clock
    timeout; the parent thread kills it on timeout and writes preds.json."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_pipeline, args=(iid, meta, instance, config, output_dir, args, q), daemon=True)
    proc.start()
    proc.join(timeout=args.instance_timeout)
    base = {"instance_id": iid, "category": meta["category"]}
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
            proc.join(5)
        return {**base, "exit_status": "InstanceTimeout", "submission_nonempty": False}
    try:
        res = q.get_nowait()
    except Exception:  # noqa: BLE001
        res = None
    if not res:
        return {**base, "exit_status": "NoResult", "submission_nonempty": False}
    submission = res.pop("_submission", "")
    model_name = res.pop("_model_name", "unknown")
    get_benchmark_runner(args.benchmark).update_preds_file(output_dir / "preds.json", iid, model_name, submission)
    return res


def build_config(args) -> dict:
    specs = [
        args.benchmark,
        "swebench_azure_modal",
        f"model.model_kwargs.temperature={args.temperature}",
        f"model.model_kwargs.api_key={args.api_key}",
        f"model.model_kwargs.api_base={args.api_base}",
        f"environment.base_url={args.sandbox_base_url}",
        f"environment.api_key={args.sandbox_api_key}",
    ]
    configs = [get_config_from_spec(spec) for spec in specs]
    configs.append({"environment": {"environment_class": UNSET}, "model": {"model_name": UNSET, "model_class": UNSET}})
    return recursive_merge(*configs)


def run_verify(output_dir: Path, *, benchmark: str, subset: str, split: str, workers: int, extra_args: list[str]) -> None:
    if benchmark == "swerebench":
        cmd = [
            sys.executable,
            "-m",
            "minisweagent.run.utilities.mini_extra",
            "swerebench-verify-azure-modal",
            "-c",
            "swebench_azure_modal",
            "--subset",
            subset,
            "--split",
            split,
            "--workers",
            str(workers),
            "-o",
            str(output_dir),
            *extra_args,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "minisweagent.run.utilities.mini_extra",
            "swebench-verify-azure-modal-cli",
            "-c",
            "swebench",
            "--subset",
            subset,
            "--split",
            split,
            "--workers",
            str(workers),
            "--output",
            str(output_dir),
            *extra_args,
        ]
    import subprocess

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--benchmark", choices=sorted(RUNNERS), default=os.getenv("REPAIR_BENCHMARK", "swebench"))
    parser.add_argument("--categories", default="fail_self_test,test_unknown")
    parser.add_argument("--subset", default="verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=int(os.getenv("REPAIR_WORKERS", "16")))
    parser.add_argument("--verify-workers", type=int, default=int(os.getenv("REPAIR_VERIFY_WORKERS", "64")))
    parser.add_argument("--temperature", default=os.getenv("TEMPERATURE", "1"))
    parser.add_argument("--api-base", default=os.getenv("MODEL_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--sandbox-base-url", default=os.getenv("SANDBOX_BASE_URL", ""))
    parser.add_argument("--sandbox-api-key", default=os.getenv("SANDBOX_API_KEY", ""))
    parser.add_argument("--denom", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-checks", type=int, default=int(os.getenv("REPAIR_MAX_CHECKS", "3")))
    parser.add_argument("--instance-timeout", type=int, default=int(os.getenv("REPAIR_INSTANCE_TIMEOUT", "900")), help="hard per-instance wall-clock timeout (s); the instance subprocess is killed past this (default 900 = 15 min)")
    parser.add_argument("--deadline", type=int, default=int(os.getenv("REPAIR_DEADLINE", "5400")), help="overall wall-clock budget (s) for the generation phase; stragglers past this are abandoned so verify can run")
    parser.add_argument("--restore-mode", choices=["patch", "replay"], default=os.getenv("REPAIR_RESTORE_MODE", "patch"), help="patch = git-apply the prior final diff only (default); replay = re-run the original trajectory's shell commands to rebuild full sandbox state (pip installs, generated/untracked files) before repair, then top up with the patch if replay produced no diff")
    parser.add_argument("--replay-cmd-timeout", type=int, default=int(os.getenv("REPAIR_REPLAY_CMD_TIMEOUT", "120")), help="per-command timeout (s) when replaying the original trajectory in --restore-mode replay")
    parser.add_argument("--replay-budget", type=int, default=int(os.getenv("REPAIR_REPLAY_BUDGET", "600")), help="overall wall-clock budget (s) for replaying one trajectory; replay stops early past this so a long original run can't blow the per-instance timeout")
    parser.add_argument("--actor-mode", choices=["fresh", "continue"], default=os.getenv("REPAIR_ACTOR_MODE", "fresh"), help="fresh = new dialog seeded with diagnosis text (default); continue = resume the original tool-calling conversation with an appended review turn")
    parser.add_argument("--history-cap", type=int, default=int(os.getenv("REPAIR_HISTORY_CAP", "80")), help="max messages of original history to resume in continue mode (keeps system+task head + most recent turns)")
    parser.add_argument("--repair-steps", type=int, default=int(os.getenv("REPAIR_REPAIR_STEPS", "150")))
    parser.add_argument("--repair-cost", type=float, default=float(os.getenv("REPAIR_REPAIR_COST", "3.0")))
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--ids-file", type=Path, default=None, help="JSON list of instance_ids to restrict the repair to (e.g. a pilot subset)")
    parser.add_argument("--external-decisions", type=Path, default=None, help="JSONL/JSON of external generator decisions {instance_id, verdict, diagnosis, feedback, summary}; when present for an instance, skip the model's Stage B/C/D and feed these to the actor")
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()
    if not args.dry_run and not args.sandbox_base_url:
        parser.error("--sandbox-base-url or SANDBOX_BASE_URL must be set explicitly")

    source_run = args.source_run.resolve()
    if not source_run.exists():
        raise SystemExit(f"source run does not exist: {source_run}")
    output_dir = Path(args.output or source_run.parent / f"{source_run.name}_critic_actor_{time.strftime('%Y%m%d_%H%M%S')}")
    categories = {c.strip() for c in args.categories.split(",") if c.strip()}

    include_ids = None
    if args.ids_file:
        include_ids = set(json.loads(Path(args.ids_file).read_text()))

    # Load the external generator decisions (argparse stored a Path under
    # args.external_decisions); replace it in-place with an {iid: decision} dict so
    # _pipeline subprocesses receive a plain dict via spawn pickling.
    ext_map: dict = {}
    if args.external_decisions:
        for line in Path(args.external_decisions).read_text().splitlines():
            line = line.strip()
            if line:
                d = json.loads(line)
                ext_map[d["instance_id"]] = d
        print(f"external decisions loaded: {len(ext_map)}")
    args.external_decisions = ext_map

    selected, metadata, _ = R.select_instances(source_run, categories, require_nonempty_patch=False, include_ids=include_ids, limit=args.limit)
    counts: dict[str, int] = {}
    for row in selected:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(f"source_run={source_run}")
    print(f"output_dir={output_dir}")
    print(f"selected={len(selected)} categories={counts} workers={args.workers}")
    if args.dry_run:
        for row in selected[:20]:
            print(row)
        return
    if not selected:
        print("no selected instances; stopping")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_dir / "repair_critic_actor.log")
    (output_dir / "selected_repair_instances.json").write_text(json.dumps(selected, indent=2))

    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    instances_by_id = {row["instance_id"]: dict(row) for row in load_dataset(dataset_path, split=args.split)}
    config = build_config(args)

    todo = [row for row in selected if row["instance_id"] in instances_by_id]
    missing = [row["instance_id"] for row in selected if row["instance_id"] not in instances_by_id]
    if missing:
        logger.warning("%d selected ids missing from dataset: %s", len(missing), missing[:10])

    print(f"running repair loop for {len(todo)} instances (deadline={args.deadline}s)")
    done = 0
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        ex.submit(process_one, row["instance_id"], metadata[row["instance_id"]], instances_by_id[row["instance_id"]], config, output_dir, args): row["instance_id"]
        for row in todo
    }
    try:
        for fut in concurrent.futures.as_completed(futures, timeout=args.deadline):
            iid = futures[fut]
            done += 1
            try:
                rec = fut.result()
                print(f"[{done}/{len(todo)}] {iid} applied={rec.get('prior_patch_applied')} diag={rec.get('diagnosis_chars')} submit={rec.get('submission_nonempty')} exit={rec.get('exit_status')}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{done}/{len(todo)}] {iid} CRASHED: {exc}")
    except concurrent.futures.TimeoutError:
        stuck = [futures[f] for f in futures if not f.done()]
        print(f"deadline hit; {len(stuck)} instance(s) did not finish and are abandoned: {stuck[:10]}")
    # do not wait on hung worker threads (one stuck model call must not block verify)
    ex.shutdown(wait=False, cancel_futures=True)

    if not args.no_verify:
        print(f"running verification workers={args.verify_workers}")
        run_verify(output_dir, benchmark=args.benchmark, subset=args.subset, split=args.split, workers=args.verify_workers, extra_args=passthrough)
        R.run_tally(output_dir, args.denom)
    print(f"done: {output_dir}")
    # non-daemon worker threads from an abandoned instance can otherwise wedge interpreter exit
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
