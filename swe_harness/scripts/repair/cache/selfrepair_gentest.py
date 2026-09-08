#!/usr/bin/env python3
"""Self-repair driven by MODEL-GENERATED tests (leak-free, self-bootstrapped).

Like selfrepair_f2p.py, but the in-loop oracle is NOT a gold FAIL_TO_PASS test —
it is a set of self-generated tests. Resolve-style behavior-contract results are converted with
scripts/repair/build_testpatch_testmap.py; legacy full-file gentest results use
scripts/repair/build_gentest_testmap.py. No gold test or source patch is needed at
repair time.

Population: WRONG (unresolved) instances that (a) submitted a patch in --source-run and
(b) have >=1 merged generated test.
Per instance, the canonical path uses TWO persistent sandboxes:
  1. the policy workspace contains only the evolving source patch;
  2. the verifier resets to the immutable dataset base before each gate, applies the
     complete candidate source patch, writes/applies the hidden generated test, runs it,
     then resets again. The policy model never receives verifier filesystem access.
  3. if all pass -> rescued (initial). else up to --max-rounds repair rounds: the model
     gets the failing test OUTPUT (not the test source) and edits SOURCE in the sandbox
     for up to --steps steps; re-check after each round.
  rescued = ALL generated tests pass after repair.
  gate_status is tri-state: pass, hard_fail, or inconclusive.
Saves final_patch (full repaired diff) for later ORACLE verification.

Reuses repair_critic_actor (rca) for sandbox env + model + DefaultAgent, exactly like
selfrepair_f2p.py.
"""
import argparse, base64, copy, hashlib, json, os, random, re, shlex, sys
import concurrent.futures
import importlib.util
from pathlib import Path, PurePosixPath

HARNESS = Path(__file__).resolve()
while HARNESS.parent != HARNESS and not (HARNESS / "scripts" / "repair" / "repair_critic_actor.py").exists():
    HARNESS = HARNESS.parent
sys.path.insert(0, str(HARNESS / "src"))
sys.path.insert(0, str(HARNESS / "external" / "azure-modal"))

spec = importlib.util.spec_from_file_location("rca", HARNESS / "scripts" / "repair" / "repair_critic_actor.py")
rca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rca)

from datasets import load_dataset
from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks import gentest as gentest_runner
from minisweagent.utils.serialize import UNSET, recursive_merge
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.submission import SUBMISSION_MARKER
from minisweagent.exceptions import InterruptAgentFlow, Submitted

f2p_spec = importlib.util.spec_from_file_location(
    "selfrepair_f2p_helpers", HARNESS / "scripts" / "repair" / "selfrepair_f2p.py"
)
f2p_helpers = importlib.util.module_from_spec(f2p_spec)
f2p_spec.loader.exec_module(f2p_helpers)

SYSTEM = "You are a careful software engineer that interacts with a computer shell to solve programming tasks."

ORACLE_GLOB = "_oracle_test_"
TEST_OUTPUT_TAIL_CHARS = 8000
REPAIR_FEEDBACK_CHARS = 6000
FINAL_TAIL_CHARS = 400
POST_PASS_REVIEW_TURNS = 10
VERIFIER_RESET_ATTEMPTS = 3
VERIFIER_RECREATE_ATTEMPTS = 2
CANDIDATE_CHECK_MARKER = "COMPLETE_TASK_AND_CHECK_CANDIDATE_PATCH"
FINAL_SUBMIT_MARKER = SUBMISSION_MARKER
KEEP_ORIGINAL_MARKER = "COMPLETE_TASK_AND_KEEP_ORIGINAL_PATCH"
STOP_REPAIR_FAILURE_KINDS = {
    "no_generated_tests",
    "no_nonempty_generated_tests",
    "oracle_preclean_failed",
    "oracle_path_dirty",
    "mixed_generated_test_formats",
    "test_patch_apply_failure",
    "test_patch_invalid",
    "infra_or_collection",
}

REPAIR_TEMPLATE = """\
You are fixing a bug. A previous patch attempt is ALREADY APPLIED in /testbed but a hidden
regression test still FAILS. You CANNOT see the test file; you only get its failure output below.
Your job: edit the SOURCE code in /testbed so the described behavior is correct. Do NOT edit or
create test files.

<original_issue>
{{task}}
</original_issue>

<failing_test_output>
{{feedback}}
</failing_test_output>

The hidden generated test is useful evidence, but it may be incomplete, over-specific, or wrong.
Judge it together with the original issue, repository behavior, and checks you can run in /testbed.

Steps:
1. `cd /testbed && git diff` to see the already-applied previous attempt.
2. Read the failure output and infer the intended behavior from it plus the original issue.
3. Make the minimal SOURCE edit(s) needed. Only non-test files.
4. Re-derive your own quick check if helpful, then explicitly choose one action below.

To CHECK the current candidate against the hidden generated test, use EXACTLY two separate
commands (must exit 0):
  cd /testbed && git diff -- <non-test source files> > /tmp/fix.patch
  echo COMPLETE_TASK_AND_CHECK_CANDIDATE_PATCH && cat /tmp/fix.patch

The check always returns environment feedback and never ends repair by itself. A pass is useful
evidence, while a failure does not by itself prove the source patch is wrong.

To FINAL-SUBMIT the current patch because it is your best solution, even if the hidden generated
test is still failing or suspect, use EXACTLY two separate commands (must exit 0):
  cd /testbed && git diff -- <non-test source files> > /tmp/fix.patch
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/fix.patch

FINAL-SUBMIT ends repair. The harness records the latest gate result separately; your decision
to submit does not claim that the generated test passed.

To discard the repair and keep the original Round-0 patch, run:
  echo COMPLETE_TASK_AND_KEEP_ORIGINAL_PATCH
"""

EMPTY_PATCH_REPAIR_TEMPLATE = """\
You are fixing a bug. A previous attempt completed without producing any SOURCE patch, so
/testbed is still at the clean base checkout. A hidden generated regression test currently passes
on that checkout, but it may be incomplete and an empty patch does not implement the requested
change. Use the original issue and repository behavior to produce the best minimal source fix.
Do NOT edit or create test files.

<original_issue>
{{task}}
</original_issue>

The hidden generated test is only supporting evidence. Inspect the relevant implementation and
public checks in /testbed, make the minimal SOURCE edit, and then explicitly choose one action.

To CHECK the current candidate against the hidden generated test, use EXACTLY two separate
commands (must exit 0):
  cd /testbed && git diff -- <non-test source files> > /tmp/fix.patch
  echo COMPLETE_TASK_AND_CHECK_CANDIDATE_PATCH && cat /tmp/fix.patch

To FINAL-SUBMIT the current patch because it is your best solution, use EXACTLY two separate
commands (must exit 0):
  cd /testbed && git diff -- <non-test source files> > /tmp/fix.patch
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/fix.patch

To keep the original empty Round-0 patch, run:
  echo COMPLETE_TASK_AND_KEEP_ORIGINAL_PATCH
"""

PATCH_APPLY_FAILURE_TEMPLATE = """\
You are fixing a bug. A previous patch attempt could NOT be applied by the harness, so /testbed
is currently at the clean base checkout. You CANNOT see the hidden regression tests yet.

Your job in this round: create a valid minimal SOURCE patch in /testbed that addresses the
original issue. Use the failed patch/apply diagnostics only as hints. Do NOT edit or create test
files.

<original_issue>
{{task}}
</original_issue>

<failed_patch_apply_diagnostics>
{{feedback}}
</failed_patch_apply_diagnostics>

Steps:
1. `cd /testbed && git status && git diff` to confirm the clean starting point.
2. Read the issue and the patch-apply diagnostics. Infer the intended source change.
3. Make the minimal SOURCE edit(s). Only non-test files.
4. Submit a syntactically valid unified diff.

Submit with EXACTLY two separate commands (must exit 0):
  cd /testbed && git diff -- <non-test source files> > /tmp/fix.patch
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/fix.patch
"""

REPAIR_FOLLOWUP_TEMPLATE = """\
Your latest source patch was tested against the hidden generated regression test, but the gate
still does not pass. Treat the output as environment feedback, not an absolute veto. Compare it
with the original issue and repository behavior. Then either make another minimal source-only
edit and CHECK the candidate again, or FINAL-SUBMIT the current patch if the generated test is
wrong or over-specific and the source patch is already the best solution.

<updated_failing_test_output>
{feedback}
</updated_failing_test_output>

Do not edit or create tests.

CHECK candidate:
  cd /testbed && git diff -- <non-test source files> > /tmp/fix.patch
  echo COMPLETE_TASK_AND_CHECK_CANDIDATE_PATCH && cat /tmp/fix.patch

FINAL-SUBMIT current patch regardless of the gate result:
  cd /testbed && git diff -- <non-test source files> > /tmp/fix.patch
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/fix.patch

KEEP the original Round-0 patch instead of the current repair:
  echo COMPLETE_TASK_AND_KEEP_ORIGINAL_PATCH
"""

CANDIDATE_PASS_FOLLOWUP_TEMPLATE = """\
The hidden generated-test CHECK passed. This is useful evidence, but it is not an automatic final
submission. Reconcile it with the original issue and any public checks you have run. Then either
FINAL-SUBMIT the current patch, continue with a necessary source edit, or keep the original
Round-0 patch.

FINAL-SUBMIT current patch:
  cd /testbed && git diff -- <non-test source files> > /tmp/fix.patch
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/fix.patch

KEEP the original Round-0 patch:
  echo COMPLETE_TASK_AND_KEEP_ORIGINAL_PATCH
"""

FINAL_DECISION_TEMPLATE = """\
The repair {reason}. Make one final structured decision now; do not continue exploring.

FINAL-SUBMIT the current patch:
  cd /testbed && git diff -- <non-test source files> > /tmp/fix.patch
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/fix.patch

KEEP the original Round-0 patch:
  echo COMPLETE_TASK_AND_KEEP_ORIGINAL_PATCH
"""

DUPLICATE_PATCH_FOLLOWUP = """\
The latest candidate CHECK did not make a new source-code change, so the hidden generated test was
not rerun and this did not consume a gate check. Inspect the current diff and feedback. Either make
a concrete non-test source edit before checking again, or explicitly FINAL-SUBMIT the unchanged
patch if it is already the best solution and the generated test is wrong or over-specific.
"""

ACTIVATE = (
    "source /opt/miniconda3/bin/activate 2>/dev/null; "
    "source /opt/conda/bin/activate 2>/dev/null; "
    "conda activate testbed 2>/dev/null; "
)

PASS = re.compile(r"\b(\d+) passed\b")
FAIL = re.compile(r"\b(\d+) failed\b")
ERRS = re.compile(r"\b(\d+) errors?\b")
NOT_PASSED = re.compile(r"\b(\d+) (?:skipped|xfailed|xpassed)\b")
SKIPPED = re.compile(r"\b(\d+) skipped\b")
XFAILED = re.compile(r"\b(\d+) xfailed\b")
XPASSED = re.compile(r"\b(\d+) xpassed\b")
UNITTEST_RAN = re.compile(r"\bRan (\d+) tests?\b")
UNITTEST_FAILED = re.compile(r"\bFAILED \(([^)]*)\)")
UNITTEST_OK = re.compile(r"^OK\b", re.M)
PYTEST_FAILED_NODE = re.compile(r"^(?:FAILED|ERROR) (?P<node>\S+)", re.M)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BLOCK = re.compile(
    r"No module named|ModuleNotFoundError|ImportError|cannot import name|collected 0 items|"
    r"no tests ran|ERROR collecting|SyntaxError|INTERNALERROR|file or directory not found",
    re.I,
)
INCONCLUSIVE_INFRA = re.compile(
    r"attrib\(\) got an unexpected keyword argument 'convert'|"
    r"unrecognized arguments: --no-header|"
    r"find: .*/testbed.*No such file or directory|"
    r"could not find testbed|"
    r"ConnectionError|"
    r"heartbeat failed",
    re.I,
)
TIMEOUT = re.compile(r"command exceeded timeout|forcibly terminated|timed out|timeout", re.I)
MASKED_GENERATED_COMMAND_FLAGS = ("--no-header",)


def build_config(
    api_base,
    api_key,
    sandbox_url,
    sandbox_key,
    temperature,
    model_name=None,
    model_class=None,
    disable_thinking=False,
    top_p=None,
    reasoning_effort=None,
):
    specs = [
        "swerebench", "swebench_azure_modal",
        f"model.model_kwargs.temperature={temperature}",
        f"model.model_kwargs.api_key={api_key}",
        f"model.model_kwargs.api_base={api_base}",
        f"environment.base_url={sandbox_url}",
    ]
    if sandbox_key:
        specs.append(f"environment.api_key={sandbox_key}")
    if top_p is not None:
        specs.append(f"model.model_kwargs.top_p={top_p}")
    if model_name:
        specs.append(f"model.model_name={model_name}")
    cfgs = [get_config_from_spec(s) for s in specs]
    cfgs.append({"environment": {"environment_class": UNSET}, "model": {"model_name": UNSET, "model_class": UNSET}})
    cfg = recursive_merge(*cfgs)
    if model_class:
        cfg.setdefault("model", {})["model_class"] = model_class
    if disable_thinking:
        kwargs = cfg.setdefault("model", {}).setdefault("model_kwargs", {})
        kwargs.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if model_class == "trapi_response":
        kwargs = cfg.setdefault("model", {}).setdefault("model_kwargs", {})
        kwargs.pop("max_tokens", None)
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        kwargs.pop("api_base", None)
        kwargs.pop("api_key", None)
        kwargs.setdefault("max_output_tokens", 8192)
        kwargs.setdefault("tool_choice", "required")
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
    return cfg


def restore_env(env, instance, timeout):
    ic = instance.get("install_config", {}) or {}
    cmds = []
    ev = ic.get("eval_commands") or []
    if isinstance(ev, str):
        ev = [ev]
    cmds += list(ev)
    inst = ic.get("install")
    if inst:
        cmds += inst if isinstance(inst, list) else [inst]
    if not cmds:
        return
    script = " && ".join(c for c in cmds if str(c).strip())
    rca.shell(env, f"cd /testbed && ( {ACTIVATE} {script} ) 2>&1 | tail -5", timeout=timeout)


def _write(env, path, content):
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    rca.shell(env, f"printf %s '{b64}' | base64 -d > {shlex.quote(path)}")


def _pytest_rc_and_output(output):
    m = re.search(r"__RC__=(-?\d+)\s*$", (output or "").strip())
    rc = int(m.group(1)) if m else 1
    out_clean = re.sub(r"\n?__RC__=-?\d+\s*$", "", output or "")
    return rc, out_clean


def _pytest_all_passed(output, rc, expected_files):
    output = ANSI_ESCAPE.sub("", output or "")
    passed = sum(int(m.group(1)) for m in PASS.finditer(output))
    failed = sum(int(m.group(1)) for m in FAIL.finditer(output))
    errors = sum(int(m.group(1)) for m in ERRS.finditer(output))
    not_passed = sum(int(m.group(1)) for m in NOT_PASSED.finditer(output))
    ran = UNITTEST_RAN.search(output or "")
    unittest_failed = UNITTEST_FAILED.search(output or "")
    unittest_ok = UNITTEST_OK.search(output or "")
    if ran and unittest_failed:
        return False
    if ran and unittest_ok and int(ran.group(1)) >= expected_files and not BLOCK.search(output):
        return rc == 0
    return (
        rc == 0
        and passed >= expected_files
        and failed == 0
        and errors == 0
        and not_passed == 0
        and not BLOCK.search(output)
    )


def _pytest_counts(output):
    output = ANSI_ESCAPE.sub("", output or "")
    ran = UNITTEST_RAN.search(output or "")
    unittest_failed = UNITTEST_FAILED.search(output or "")
    return {
        "passed": sum(int(m.group(1)) for m in PASS.finditer(output)),
        "failed": sum(int(m.group(1)) for m in FAIL.finditer(output)),
        "errors": sum(int(m.group(1)) for m in ERRS.finditer(output)),
        "skipped": sum(int(m.group(1)) for m in SKIPPED.finditer(output)),
        "xfailed": sum(int(m.group(1)) for m in XFAILED.finditer(output)),
        "xpassed": sum(int(m.group(1)) for m in XPASSED.finditer(output)),
        "unittest_ran": int(ran.group(1)) if ran else None,
        "unittest_failed": unittest_failed.group(1) if unittest_failed else None,
        "unittest_ok": bool(UNITTEST_OK.search(output or "")),
    }


def _test_failure_kind(output, rc, passed):
    clean = ANSI_ESCAPE.sub("", output or "")
    if passed:
        return "passed"
    if TIMEOUT.search(clean):
        return "timeout"
    if BLOCK.search(clean) or INCONCLUSIVE_INFRA.search(clean):
        return "infra_or_collection"
    counts = _pytest_counts(clean)
    if counts["failed"] or counts["unittest_failed"] or "AssertionError" in clean:
        return "assertion_failure"
    if counts["errors"] or "Traceback" in clean:
        return "test_error"
    if rc != 0:
        return "nonzero_exit"
    if counts["passed"] == 0 and not counts["unittest_ok"]:
        return "no_pass_signal"
    return "unknown_failure"


def _gate_status_from_failure_kind(failure_kind):
    if failure_kind == "passed":
        return "pass"
    if failure_kind in {"assertion_failure", "test_error"}:
        return "hard_fail"
    return "inconclusive"


def _summarize_test_output(output, rc, expected_files):
    clean = ANSI_ESCAPE.sub("", output or "")
    passed = _pytest_all_passed(clean, rc, expected_files)
    counts = _pytest_counts(clean)
    failed_nodes = [m.group("node") for m in PYTEST_FAILED_NODE.finditer(clean)]
    failure_kind = _test_failure_kind(clean, rc, passed)
    return {
        "passed": bool(passed),
        "gate_status": _gate_status_from_failure_kind(failure_kind),
        "rc": rc,
        "expected_files": expected_files,
        "counts": counts,
        "failed_test_names": failed_nodes,
        "failure_kind": failure_kind,
        "blocked_by_harness": bool(BLOCK.search(clean)),
        "inconclusive_infra": bool(INCONCLUSIVE_INFRA.search(clean)),
        "timeout_detected": bool(TIMEOUT.search(clean)),
        "output_tail": clean[-TEST_OUTPUT_TAIL_CHARS:],
    }


def _aggregate_failure_kind(results):
    if results and all(r.get("passed") for r in results):
        return "passed"
    for kind in ("timeout", "infra_or_collection", "test_error", "assertion_failure",
                 "nonzero_exit", "no_pass_signal", "unknown_failure"):
        if any(r.get("failure_kind") == kind for r in results):
            return kind
    return "no_tests_run"


def _test_result(passed=False, failure_kind="not_run", output_tail="", tests=None, **extra):
    tests = tests or []
    result = {
        "passed": bool(passed),
        "gate_status": _gate_status_from_failure_kind(failure_kind),
        "gate_reason": failure_kind,
        "failure_kind": failure_kind,
        "n_tests": len(tests),
        "tests": tests,
        "output_tail": output_tail[-TEST_OUTPUT_TAIL_CHARS:],
    }
    result.update(extra)
    return result


def _feedback_from_test_result(result):
    tests = result.get("tests") or []
    lines = [
        "test_execution_contract:",
        "  language: python",
        "  working_directory: /testbed",
        "  test_source_visibility: hidden_verifier_only",
        "  per_test:",
    ]
    for test in tests:
        lines.append("    " + json.dumps({
            "index": test.get("index"),
            "test_filename": test.get("filename"),
            "runner_command": (test.get("runner_command") or test.get("command") or "")[:1000],
        }, ensure_ascii=False, sort_keys=True))
    lines += [
        "generated_test_gate:",
        f"  gate_status: {result.get('gate_status')}",
        f"  gate_reason: {result.get('gate_reason')}",
        f"  passed: {bool(result.get('passed'))}",
        f"  n_tests: {result.get('n_tests')}",
        f"  failed_test_names: {json.dumps(result.get('failed_test_names') or [], ensure_ascii=False)}",
        "  per_test:",
    ]
    for test in tests:
        fields = {
            "index": test.get("index"),
            "filename": test.get("filename"),
            "rc": test.get("rc"),
            "gate_status": test.get("gate_status"),
            "failure_kind": test.get("failure_kind"),
            "counts": test.get("counts"),
            "failed_test_names": test.get("failed_test_names") or [],
            "timeout_detected": bool(test.get("timeout_detected")),
            "blocked_by_harness": bool(test.get("blocked_by_harness")),
            "inconclusive_infra": bool(test.get("inconclusive_infra")),
            "command_sanitized": bool(test.get("command_sanitized")),
            "removed_command_flags": test.get("removed_command_flags") or [],
        }
        if test.get("custom_command"):
            fields["command"] = (test.get("command") or "")[:500]
        lines.append("    " + json.dumps(fields, ensure_ascii=False, sort_keys=True))
    header = "\n".join(lines)
    raw = result.get("output_tail") or ""
    raw_budget = max(0, REPAIR_FEEDBACK_CHARS - len(header) - 64)
    return (
        f"{header}\n\n"
        "<raw_test_output_tail>\n"
        f"{raw[-raw_budget:]}\n"
        "</raw_test_output_tail>"
    )


def _mask_generated_command(command):
    sanitized = command or ""
    removed = []
    for flag in MASKED_GENERATED_COMMAND_FLAGS:
        next_cmd = re.sub(rf"(?<!\S){re.escape(flag)}(?!\S)", "", sanitized)
        if next_cmd != sanitized:
            removed.append(flag)
            sanitized = next_cmd
    sanitized = re.sub(r"[ \t]{2,}", " ", sanitized).strip()
    return sanitized, removed


def _with_testbed_activation(command, activation, *, add_rc_marker=True):
    run_cmd = f"( {activation} {command} ) 2>&1"
    if add_rc_marker and "__RC__=" not in command:
        run_cmd = f"{run_cmd}; rc=$?; echo \"__RC__=$rc\""
    return run_cmd


def _test_entry_parts(entry, idx, instance):
    fallback_filename = gentest_runner.generated_test_file_for_instance(
        instance, gentest_runner.TEST_FILE if idx == 0 else f"test_model_gen_{idx}.py"
    )
    if isinstance(entry, dict):
        code = (entry.get("test_code") or entry.get("code") or "").strip()
        filename = (
            entry.get("test_file_path")
            or entry.get("test_filename")
            or entry.get("filename")
            or fallback_filename
        ).strip()
        command = (entry.get("command") or "").strip()
    else:
        code = str(entry or "").strip()
        filename = fallback_filename
        command = ""
    filename = filename.lstrip("/")
    if not command:
        command = f"cd /testbed && {gentest_runner.test_command_for_instance(instance, filename)}"
    return code, filename, command


def _test_patch_entry_parts(entry):
    if not isinstance(entry, dict):
        return "", [], "", "test patch entry must be an object"
    patch = str(entry.get("test_patch") or "")
    command = str(entry.get("test_command") or entry.get("command") or "").strip()
    if not patch.strip():
        return "", [], command, "empty test_patch"
    patch = patch.rstrip("\n") + "\n"
    if not command:
        return patch, [], "", "test patch entry requires test_command"
    if not patch.startswith("diff --git "):
        return patch, [], command, "test_patch is not a git unified diff"
    if "+++ /dev/null" in patch:
        return patch, [], command, "test_patch may not delete files"

    paths = list(dict.fromkeys(gentest_runner._changed_paths_from_patch(patch)))
    if not paths:
        return patch, [], command, "test_patch has no changed paths"
    for path in paths:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] == ".git":
            return patch, [], command, f"unsafe test_patch path: {path!r}"
    return patch, paths, command, ""


def _oracle_paths_clean(env, paths):
    qpaths = " ".join(shlex.quote(path) for path in paths)
    result = rca.shell(
        env,
        "cd /testbed && "
        f"if test -z \"$(git status --porcelain --untracked-files=all -- {qpaths})\"; "
        "then exit 0; else git status --porcelain --untracked-files=all -- "
        f"{qpaths}; exit 3; fi",
    )
    return result.get("returncode", -1) == 0, result.get("output", "") or ""


def _restore_test_patch_paths(env, patch_file, paths):
    qpatch = shlex.quote(patch_file)
    qpaths = " ".join(shlex.quote(path) for path in paths)
    reverse = rca.shell(
        env,
        "cd /testbed && "
        f"git apply --reverse --check {qpatch} && git apply --reverse {qpatch}",
    )
    clean, clean_output = _oracle_paths_clean(env, paths)
    if reverse.get("returncode", -1) == 0 and clean:
        rca.shell(env, f"rm -f -- {qpatch}")
        return True, ""

    # Pre-clean proves these paths contained no source edits, so this bounded
    # fallback cannot discard a candidate repair. It only restores the hidden
    # oracle paths if reverse-apply itself failed.
    fallback = rca.shell(
        env,
        "cd /testbed && "
        f"(git checkout -- {qpaths} 2>/dev/null || true) && git clean -f -- {qpaths}",
    )
    clean, final_output = _oracle_paths_clean(env, paths)
    rca.shell(env, f"rm -f -- {qpatch}")
    output = "\n".join(
        str(part or "")
        for part in (reverse.get("output"), clean_output, fallback.get("output"), final_output)
        if part
    )
    return clean, output


def _cleanup_oracle_paths(env, paths):
    qpaths = " ".join(shlex.quote(f"/testbed/{p.lstrip('/')}") for p in paths if p)
    cleanup_outputs = []
    if qpaths:
        cleanup = rca.shell(env, f"rm -f -- {qpaths}")
        cleanup_outputs.append(cleanup.get("output", "") or "")
    pattern = f"/testbed/{ORACLE_GLOB}*.py"
    cleanup = rca.shell(env, f"rm -f -- {pattern}")
    cleanup_outputs.append(cleanup.get("output", "") or "")
    status_parts = []
    for path in paths:
        if not path:
            continue
        status = rca.shell(env, f"test -e {shlex.quote('/testbed/' + path.lstrip('/'))} && echo PRESENT || true")
        status_parts.append(status.get("output", "") or "")
    status = rca.shell(env, f"find /testbed -maxdepth 1 -name {shlex.quote(ORACLE_GLOB + '*.py')} -print")
    status_parts.append(status.get("output", "") or "")
    clean = not "".join(status_parts).strip()
    output = "\n".join(x for x in [*cleanup_outputs, *status_parts] if x)
    return clean, output


def _source_diff(env, excluded_paths=()):
    pathspecs = [".", *(f":(exclude){path}" for path in excluded_paths if path)]
    command = "cd /testbed && git diff -- " + " ".join(shlex.quote(path) for path in pathspecs)
    diff = rca.shell(env, command).get("output", "") or ""
    return diff if rca._looks_like_diff(diff) else ""


def gentest_check(env, instance, tests, timeout):
    """Write generated full-file tests as hidden oracle files, run them, then delete them."""
    paths = []
    if not tests:
        return _test_result(failure_kind="no_generated_tests", output_tail="NO_GENERATED_TESTS")

    entries = []
    for k, entry in enumerate(tests):
        code, filename, command = _test_entry_parts(entry, k, instance)
        if not code:
            continue
        entries.append((code, filename, command, entry))
        paths.append(filename)
    if not entries:
        return _test_result(failure_kind="no_nonempty_generated_tests", output_tail="NO_NONEMPTY_GENERATED_TESTS")

    clean, cleanup_output = _cleanup_oracle_paths(env, paths)
    if not clean:
        return _test_result(
            failure_kind="oracle_preclean_failed",
            output_tail=f"ORACLE_TEST_PRECLEAN_FAILED\n{cleanup_output}",
            n_tests=len(entries),
            test_files=paths,
        )

    try:
        outputs = []
        test_results = []
        all_passed = True
        for idx, (code, filename, command, original_entry) in enumerate(entries):
            full_path = f"/testbed/{filename}"
            parent = shlex.quote(str(Path(full_path).parent))
            rca.shell(env, f"mkdir -p {parent}")
            _write(env, full_path, code)
            original_command = command
            run_cmd, removed_command_flags = _mask_generated_command(command)
            run_cmd = _with_testbed_activation(run_cmd, gentest_runner.activation_for_instance(instance))
            run = rca.shell(env, run_cmd, timeout=timeout)
            rc, out_clean = _pytest_rc_and_output(run.get("output", "") or "")
            summary = _summarize_test_output(out_clean, rc, 1)
            summary.update({
                "index": idx,
                "filename": filename,
                "test_paths": [filename],
                "custom_command": isinstance(original_entry, dict) and bool(original_entry.get("command")),
                "command": run_cmd,
                "runner_command": _mask_generated_command(command)[0],
                "original_command": original_command if removed_command_flags else "",
                "command_sanitized": bool(removed_command_flags),
                "command_activated": True,
                "removed_command_flags": removed_command_flags,
                "shell_returncode": run.get("returncode"),
            })
            test_results.append(summary)
            outputs.append(out_clean)
            all_passed = all_passed and bool(summary["passed"])
        output_tail = "\n\n".join(outputs)[-TEST_OUTPUT_TAIL_CHARS:]
        failure_kind = _aggregate_failure_kind(test_results)
        result = _test_result(
            passed=all_passed,
            failure_kind=failure_kind,
            output_tail=output_tail,
            tests=test_results,
            test_files=[r["filename"] for r in test_results],
            failed_test_names=sorted({n for r in test_results for n in r.get("failed_test_names", [])}),
        )
    finally:
        clean, cleanup_output = _cleanup_oracle_paths(env, paths)
        if not clean:
            raise RuntimeError(f"ORACLE_TEST_CLEANUP_FAILED\n{cleanup_output[-1200:]}")

    return result


def testpatch_check(env, instance, tests, timeout):
    """Apply resolve-style test patches as hidden repair gates, then restore them."""
    if not tests:
        return _test_result(failure_kind="no_generated_tests", output_tail="NO_GENERATED_TESTS")

    entries = []
    for index, entry in enumerate(tests):
        patch, paths, command, error = _test_patch_entry_parts(entry)
        if error:
            return _test_result(
                failure_kind="test_patch_invalid",
                output_tail=error,
                invalid_entry_index=index,
            )
        entries.append((patch, paths, command, entry))

    outputs = []
    test_results = []
    all_passed = True
    for index, (patch, paths, command, original_entry) in enumerate(entries):
        clean, dirty_output = _oracle_paths_clean(env, paths)
        if not clean:
            return _test_result(
                failure_kind="oracle_path_dirty",
                output_tail=dirty_output or "generated test patch overlaps the candidate source patch",
                n_tests=len(entries),
                test_files=paths,
            )

        patch_file = f"/tmp/{ORACLE_GLOB}patch_{index}.diff"
        _write(env, patch_file, patch)
        qpatch = shlex.quote(patch_file)
        applied = rca.shell(
            env,
            "cd /testbed && "
            f"git apply --check {qpatch} && git apply --whitespace=nowarn {qpatch}",
        )
        if applied.get("returncode", -1) != 0:
            rca.shell(env, f"rm -f -- {qpatch}")
            return _test_result(
                failure_kind="test_patch_apply_failure",
                output_tail=applied.get("output", "") or "generated test patch did not apply",
                n_tests=len(entries),
                test_files=paths,
            )

        try:
            original_command = command
            run_command, removed_command_flags = _mask_generated_command(command)
            run_command = _with_testbed_activation(
                run_command,
                gentest_runner.activation_for_instance(instance),
            )
            run = rca.shell(env, run_command, timeout=timeout)
            rc, output = _pytest_rc_and_output(run.get("output", "") or "")
            summary = _summarize_test_output(output, rc, 1)
            summary.update({
                "index": index,
                "filename": paths[0],
                "test_paths": paths,
                "test_patch": True,
                "custom_command": True,
                "command": run_command,
                "runner_command": _mask_generated_command(command)[0],
                "original_command": original_command if removed_command_flags else "",
                "command_sanitized": bool(removed_command_flags),
                "command_activated": True,
                "removed_command_flags": removed_command_flags,
                "shell_returncode": run.get("returncode"),
            })
            test_results.append(summary)
            outputs.append(output)
            all_passed = all_passed and bool(summary["passed"])
        finally:
            restored, restore_output = _restore_test_patch_paths(env, patch_file, paths)
            if not restored:
                raise RuntimeError(f"ORACLE_TEST_PATCH_CLEANUP_FAILED\n{restore_output[-1200:]}")

    output_tail = "\n\n".join(outputs)[-TEST_OUTPUT_TAIL_CHARS:]
    failure_kind = _aggregate_failure_kind(test_results)
    return _test_result(
        passed=all_passed,
        failure_kind=failure_kind,
        output_tail=output_tail,
        tests=test_results,
        test_files=sorted({path for result in test_results for path in result["test_paths"]}),
        failed_test_names=sorted({
            name for result in test_results for name in result.get("failed_test_names", [])
        }),
    )


def generated_test_check(env, instance, tests, timeout):
    """Route legacy full-file tests and resolve-style test-patch entries safely."""
    patch_entries = [
        isinstance(entry, dict) and bool(str(entry.get("test_patch") or "").strip())
        for entry in (tests or [])
    ]
    if any(patch_entries):
        if not all(patch_entries):
            return _test_result(
                failure_kind="mixed_generated_test_formats",
                output_tail="cannot mix full-file generated tests and canonical test patches",
            )
        return testpatch_check(env, instance, tests, timeout)
    return gentest_check(env, instance, tests, timeout)


class PersistentGeneratedTestGate:
    """Evaluate complete candidate patches in a separate reset-before-grade sandbox."""

    def __init__(self, cfg, instance, tests, args):
        self.cfg = copy.deepcopy(cfg)
        self.instance = copy.deepcopy(instance)
        self.tests = copy.deepcopy(tests)
        self.timeout = int(args.test_timeout)
        self.install_timeout = int(args.install_timeout)
        self.apply_test_patch_at_start = bool(args.apply_test_patch_at_start)
        self.env = None
        self.closed = False
        self.recreate_count = 0
        self._open_environment()

    def _open_environment(self):
        last_error = None
        for _attempt in range(VERIFIER_RECREATE_ATTEMPTS):
            candidate_env = None
            try:
                candidate_env = rca.swerebench_runner.get_sb_environment(
                    copy.deepcopy(self.cfg), self.instance
                )
                self.env = candidate_env
                requested_base = str(
                    self.instance.get("base_commit")
                    or self.instance.get("parent_commit")
                    or "HEAD"
                )
                resolved = rca.shell(
                    self.env,
                    "cd /testbed && git rev-parse "
                    f"{shlex.quote(requested_base)}^{{commit}}",
                )
                self.baseline_commit = str(resolved.get("output") or "").strip()
                if resolved.get("returncode", -1) != 0 or not self.baseline_commit:
                    raise RuntimeError(
                        "generated-test verifier base commit is unavailable: "
                        + str(resolved.get("output") or "")[-1000:]
                    )
                self._reset()
                gentest_runner.restore_env(
                    self.env,
                    self.instance,
                    self.install_timeout,
                    activation=gentest_runner.activation_for_instance(self.instance),
                )
                self._reset()
                return
            except BaseException as exc:
                last_error = exc
                if candidate_env is not None:
                    candidate_env.cleanup()
                self.env = None
        raise RuntimeError(
            "generated-test verifier could not initialize after "
            f"{VERIFIER_RECREATE_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def _recreate_environment(self):
        if self.env is not None:
            self.env.cleanup()
        self.env = None
        self.recreate_count += 1
        self._open_environment()

    def _reset_or_recreate(self):
        try:
            self._reset()
        except RuntimeError:
            self._recreate_environment()

    def _reset(self):
        failures = []
        for _attempt in range(VERIFIER_RESET_ATTEMPTS):
            reset = rca.shell(
                self.env,
                "cd /testbed && git reset --hard "
                f"{shlex.quote(self.baseline_commit)} && git clean -fdq",
                timeout=120,
            )
            if reset.get("returncode", -1) == 0:
                return
            failures.append(str(reset.get("output") or "")[-1000:])
        raise RuntimeError(
            "generated-test verifier reset failed after "
            f"{VERIFIER_RESET_ATTEMPTS} attempts: " + (failures[-1] if failures else "")
        )

    def evaluate(self, source_patch):
        if self.closed:
            raise RuntimeError("generated-test verifier is already closed")
        source_patch = str(source_patch or "")
        source_sha = hashlib.sha256(source_patch.encode()).hexdigest() if source_patch else ""
        self._reset_or_recreate()
        try:
            if source_patch.strip():
                applied, apply_error = rca.apply_prior_patch(self.env, source_patch)
                if not applied:
                    return _test_result(
                        failure_kind="source_patch_apply_failure",
                        output_tail=apply_error or "candidate source patch did not apply",
                        isolated_to_verifier=True,
                        source_patch_sha256=source_sha,
                        source_patch_applied=False,
                        persistent_verifier=True,
                    )

            alignment = gentest_runner.apply_test_patch_for_alignment(
                self.env,
                self.instance,
                enabled=self.apply_test_patch_at_start,
                phase="repair_verifier",
            )
            if alignment.get("reason"):
                return _test_result(
                    failure_kind=alignment["reason"],
                    output_tail=alignment.get("error") or alignment["reason"],
                    isolated_to_verifier=True,
                    source_patch_sha256=source_sha,
                    source_patch_applied=True,
                    persistent_verifier=True,
                    test_patch_alignment=alignment,
                )

            result = generated_test_check(
                self.env,
                self.instance,
                self.tests,
                self.timeout,
            )
            result.update({
                "isolated_to_verifier": True,
                "source_patch_sha256": source_sha,
                "source_patch_applied": True,
                "persistent_verifier": True,
                "verifier_recreates": self.recreate_count,
                "test_patch_alignment": alignment,
            })
            return result
        finally:
            self._reset_or_recreate()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.env is not None:
            self.env.cleanup()


def _submission_request_from_interrupt(exc):
    for message in reversed(getattr(exc, "messages", ())):
        extra = message.get("extra") or {}
        if "submission" in extra:
            return (
                str(extra.get("submission") or ""),
                str(extra.get("repair_decision") or "final_submit"),
            )
    return "", "final_submit"


def _submission_after_marker(output, marker):
    """Return text after an exact successful marker line, or None if absent."""
    if output.get("returncode") != 0:
        return None
    lines = str(output.get("output") or "").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == marker:
            return "".join(lines[index + 1:])
    return None


class RepairAgent(DefaultAgent):
    """One persistent repair conversation with a fixed turn budget and explicit gate budget."""

    def __init__(
        self,
        model,
        env,
        *,
        on_submit,
        get_current_patch,
        initial_patch,
        max_rounds,
        steps_per_round,
        max_gate_checks=None,
        continue_on_duplicate_patch=False,
        force_submit_on_turn_limit=True,
        initial_test_result=None,
        seed_messages=None,
        **kwargs,
    ):
        super().__init__(model, env, **kwargs)
        self._on_submit = on_submit
        self._get_current_patch = get_current_patch
        self._initial_patch = str(initial_patch or "")
        self._last_gated_patch = str(initial_patch or "").rstrip("\n")
        self._max_rounds = max_rounds
        self._gate_limit_stop_reason = (
            "max_gate_checks" if max_gate_checks not in (None, 0) else "max_rounds"
        )
        self._max_gate_checks = int(max_gate_checks or max_rounds)
        if self._max_gate_checks <= 0:
            raise ValueError(f"max_gate_checks must be positive, got {self._max_gate_checks}")
        if steps_per_round <= 0:
            raise ValueError(f"steps_per_round must be positive, got {steps_per_round}")
        self._steps_per_round = steps_per_round
        self._continue_on_duplicate_patch = bool(continue_on_duplicate_patch)
        self._force_submit_on_turn_limit = bool(force_submit_on_turn_limit)
        self._round_start_calls = 0
        self.duplicate_submissions = 0
        self.rounds = []
        self.passed = False
        self.final_test_result = initial_test_result
        self.model_final_submit = False
        self.model_final_submit_gate_reused = False
        self.model_final_decision = False
        self.model_keep_original = False
        self.fallback_to_round0 = False
        self.selected_patch = None
        self.stop_reason = ""
        self._decision_only = False
        self._decision_deadline = None
        self._post_pass_deadline = None
        self._seed = list(seed_messages or [])

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for action in actions:
            try:
                output = self.env.execute(action)
            except Submitted:
                outputs.append({
                    "output": "Final submission received.",
                    "returncode": 0,
                    "exception_info": "",
                })
                self.add_messages(
                    *self.model.format_observation_messages(message, outputs, self.get_template_vars())
                )
                raise
            candidate_patch = _submission_after_marker(output, CANDIDATE_CHECK_MARKER)
            if candidate_patch is not None:
                outputs.append({
                    "output": "Candidate patch received; running the hidden generated-test gate.",
                    "returncode": 0,
                    "exception_info": "",
                })
                self.add_messages(
                    *self.model.format_observation_messages(message, outputs, self.get_template_vars())
                )
                raise Submitted({
                    "role": "exit",
                    "content": candidate_patch,
                    "extra": {
                        "exit_status": "CandidateCheckRequested",
                        "submission": candidate_patch,
                        "repair_decision": "candidate_check",
                    },
                })
            if _submission_after_marker(output, KEEP_ORIGINAL_MARKER) is not None:
                outputs.append({
                    "output": "Final decision received; keeping the original Round-0 patch.",
                    "returncode": 0,
                    "exception_info": "",
                })
                self.add_messages(
                    *self.model.format_observation_messages(message, outputs, self.get_template_vars())
                )
                raise Submitted({
                    "role": "exit",
                    "content": "",
                    "extra": {
                        "exit_status": "KeepOriginalRequested",
                        "submission": "",
                        "repair_decision": "keep_original",
                    },
                })
            outputs.append(output)
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )

    def _handle_submission(self, submitted_patch, *, forced_submission, decision="candidate_check"):
        if decision not in {"candidate_check", "final_submit", "keep_original"}:
            raise ValueError(f"unknown repair decision: {decision!r}")
        if decision == "keep_original":
            self.model_final_decision = True
            self.model_keep_original = True
            self.selected_patch = self._initial_patch
            self.passed = bool((self.final_test_result or {}).get("passed"))
            self.stop_reason = "model_keep_original"
            raise Submitted({
                "role": "exit",
                "content": "",
                "extra": {
                    "exit_status": "KeepOriginal",
                    "submission": self._initial_patch,
                    "model_final_submit": False,
                    "model_keep_original": True,
                },
            })
        current_patch = str(self._get_current_patch() or "")
        normalized_patch = current_patch.rstrip("\n")
        if (
            decision == "candidate_check"
            and self._continue_on_duplicate_patch
            and normalized_patch == self._last_gated_patch
        ):
            self.duplicate_submissions += 1
            self._round_start_calls = self.n_calls
            self.add_messages(self.model.format_message(role="user", content=DUPLICATE_PATCH_FOLLOWUP))
            return {}

        reuse_gate = (
            decision == "final_submit"
            and normalized_patch == self._last_gated_patch
            and self.final_test_result is not None
        )
        if reuse_gate:
            test_result = self.final_test_result
            round_diff = current_patch or submitted_patch or ""
        else:
            test_result, round_diff = self._on_submit()
        passed = bool(test_result.get("passed"))
        previous_gated_patch = self._last_gated_patch
        if not reuse_gate:
            self._last_gated_patch = str(round_diff or "").rstrip("\n")
            self.rounds.append({
                "round": len(self.rounds) + 1,
                "turns": self.n_calls - self._round_start_calls,
                "decision": decision,
                "passed": passed,
                "gate_status": test_result.get("gate_status"),
                "gate_reason": test_result.get("gate_reason"),
                "test_result": test_result,
                "patch": round_diff,
                "forced_submission": forced_submission,
            })
        self.final_test_result = test_result

        if decision == "final_submit":
            self.passed = passed
            self.model_final_submit = True
            self.model_final_decision = True
            self.model_final_submit_gate_reused = reuse_gate
            self.selected_patch = round_diff or submitted_patch or ""
            self.stop_reason = "model_final_submit"
            raise Submitted({
                "role": "exit",
                "content": test_result.get("output_tail", ""),
                "extra": {
                    "exit_status": "Submitted",
                    "submission": round_diff or submitted_patch or "",
                    "model_final_submit": True,
                    "final_gate_status": test_result.get("gate_status"),
                },
            })

        round_no = len(self.rounds)
        failure_kind = test_result.get("failure_kind")
        no_progress = (
            not passed
            and str(round_diff or "").rstrip("\n") == str(previous_gated_patch or "").rstrip("\n")
        )
        stop_kind = failure_kind in STOP_REPAIR_FAILURE_KINDS
        self.passed = passed
        self._round_start_calls = self.n_calls
        self._post_pass_deadline = self.n_calls + POST_PASS_REVIEW_TURNS if passed else None
        if round_no >= self._max_gate_checks:
            self._request_final_decision(self._gate_limit_stop_reason)
        elif stop_kind:
            self._request_final_decision(f"cannot run another reliable gate ({failure_kind})")
        elif no_progress and not self._continue_on_duplicate_patch:
            self._request_final_decision("made no source-code progress")
        elif passed:
            self.add_messages(self.model.format_message(
                role="user",
                content=CANDIDATE_PASS_FOLLOWUP_TEMPLATE,
            ))
        else:
            feedback = _feedback_from_test_result(test_result)[:REPAIR_FEEDBACK_CHARS]
            self.add_messages(self.model.format_message(
                role="user",
                content=REPAIR_FOLLOWUP_TEMPLATE.format(feedback=feedback),
            ))
        return {}

    def _request_final_decision(self, reason):
        if self._decision_only:
            return
        self._decision_only = True
        self._decision_deadline = self.n_calls + 1
        self.add_messages(self.model.format_message(
            role="user",
            content=FINAL_DECISION_TEMPLATE.format(reason=reason),
        ))

    def _fallback_to_initial(self, reason):
        self.fallback_to_round0 = True
        self.selected_patch = self._initial_patch
        self.stop_reason = reason
        raise Submitted({
            "role": "exit",
            "content": "",
            "extra": {
                "exit_status": "KeepOriginalFallback",
                "submission": self._initial_patch,
                "model_final_submit": False,
                "model_keep_original": False,
                "fallback_to_round0": True,
            },
        })

    def step(self) -> dict:
        if self._decision_only and self._decision_deadline is not None:
            if self.n_calls >= self._decision_deadline:
                self._fallback_to_initial("no_explicit_final_decision")
        if (
            not self._decision_only
            and self._post_pass_deadline is not None
            and self.n_calls >= self._post_pass_deadline
        ):
            self._request_final_decision("has completed its post-pass review window")
            return {}
        if not self._decision_only and 0 < self.config.step_limit <= self.n_calls + 1:
            self._request_final_decision("has reached its total turn budget")
            return {}
        if (
            not self._decision_only
            and
            self._force_submit_on_turn_limit
            and self.n_calls - self._round_start_calls >= self._steps_per_round
        ):
            return self._handle_submission(
                self._get_current_patch(),
                forced_submission=True,
                decision="candidate_check",
            )
        try:
            return super().step()
        except Submitted as exc:
            submitted_patch, decision = _submission_request_from_interrupt(exc)
            return self._handle_submission(
                submitted_patch,
                forced_submission=False,
                decision=decision,
            )

    def run_seeded(self, repair_prompt: str, *, system_prompt=None) -> dict:
        """Continue a freshly generated e2e solve trajectory with gentest feedback."""
        self.messages = list(self._seed)
        if not self.messages and system_prompt is not None:
            self.add_messages(self.model.format_message(role="system", content=system_prompt))
        self.add_messages(self.model.format_message(role="user", content=repair_prompt))
        while True:
            try:
                self.step()
            except InterruptAgentFlow as exc:
                self.add_messages(*getattr(exc, "messages", []))
                if self.messages and self.messages[-1].get("role") == "exit":
                    break
            except Exception as exc:  # noqa: BLE001
                self.stop_reason = self.stop_reason or f"agent_error: {type(exc).__name__}: {str(exc)[:120]}"
                break
            finally:
                if self.config.output_path:
                    self.save(self.config.output_path)
        return {"passed": self.passed, "stop_reason": self.stop_reason}


def run_repair_rounds(
    model_cfg,
    env,
    task,
    feedback,
    steps,
    max_rounds,
    on_submit,
    *,
    get_current_patch,
    initial_patch,
    max_gate_checks=None,
    continue_on_duplicate_patch=False,
    force_submit_on_turn_limit=True,
    initial_test_result=None,
    seed_messages=None,
    output_path=None,
    repair_template=REPAIR_TEMPLATE,
):
    agent = RepairAgent(
        rca.get_model(config=model_cfg),
        env,
        on_submit=on_submit,
        get_current_patch=get_current_patch,
        initial_patch=initial_patch,
        max_rounds=max_rounds,
        steps_per_round=steps,
        max_gate_checks=max_gate_checks,
        continue_on_duplicate_patch=continue_on_duplicate_patch,
        force_submit_on_turn_limit=force_submit_on_turn_limit,
        initial_test_result=initial_test_result,
        seed_messages=seed_messages,
        system_template=SYSTEM,
        instance_template=REPAIR_TEMPLATE,
        step_limit=steps * max_rounds + 1,
        cost_limit=5.0 * max_rounds,
        output_path=output_path,
    )
    try:
        if seed_messages:
            prompt = (
                repair_template
                .replace("{{task}}", task)
                .replace("{{feedback}}", feedback[:REPAIR_FEEDBACK_CHARS])
            )
            agent.run_seeded(prompt)
        else:
            prompt = (
                repair_template
                .replace("{{task}}", task)
                .replace("{{feedback}}", feedback[:REPAIR_FEEDBACK_CHARS])
            )
            agent.run_seeded(prompt, system_prompt=SYSTEM)
    except Exception as e:  # noqa: BLE001
        agent.stop_reason = agent.stop_reason or f"agent_error: {type(e).__name__}: {str(e)[:120]}"
    return agent


def _apply_failure_feedback(patch_candidates, apply_errors):
    parts = []
    for idx, err in enumerate(apply_errors or []):
        patch = patch_candidates[idx] if idx < len(patch_candidates or []) else ""
        parts.append(
            "PATCH CANDIDATE {idx}\n"
            "apply_error_tail:\n{err}\n\n"
            "patch_excerpt:\n{patch}".format(
                idx=idx,
                err=(err.get("error_tail") or "")[-1200:] if isinstance(err, dict) else str(err)[-1200:],
                patch=(patch or "")[:8000],
            )
        )
    return "\n\n---\n\n".join(parts) or "NO_PATCH_CANDIDATES"


def run_apply_failure_round(model_cfg, env, task, patch_candidates, apply_errors, steps):
    agent = DefaultAgent(
        rca.get_model(config=model_cfg),
        env,
        system_template=SYSTEM,
        instance_template=PATCH_APPLY_FAILURE_TEMPLATE,
        step_limit=steps,
        cost_limit=5.0,
    )
    feedback = _apply_failure_feedback(patch_candidates, apply_errors)
    try:
        agent.run(task=task, feedback=feedback[:12000])
    except Exception as e:  # noqa: BLE001
        return getattr(agent, "n_calls", 0), f"agent_error: {type(e).__name__}: {str(e)[:120]}"
    return getattr(agent, "n_calls", 0), "ok"


def process_one(iid, instance, patch_candidates, tests, cfg, args, source_messages=None):
    rec = {"instance_id": iid, "n_tests": len(tests)}
    env = None
    generated_test_gate = None
    try:
        if isinstance(patch_candidates, str):
            patch_candidates = [patch_candidates]
        patch_candidates = [p for p in (patch_candidates or []) if (p or "").strip()]
        rec["n_patch_candidates"] = len(patch_candidates)

        fresh_solve_round0 = bool(getattr(args, "fresh_solve_round0", False))
        rec["round0_mode"] = "fresh_e2e_solve" if fresh_solve_round0 else "provided_patch"
        trace_dir = getattr(args, "trajectory_dir", args.out.parent / f"{args.out.stem}.round_trajs") / iid
        trace_dir.mkdir(parents=True, exist_ok=True)
        cfgi = copy.deepcopy(cfg)
        rca.swerebench_runner._resolve_per_instance_api_base(cfgi, iid)
        env = rca.swerebench_runner.get_sb_environment(cfgi, instance)

        rca.shell(env, "cd /testbed && git checkout -- . && git clean -fdq")
        activation = gentest_runner.activation_for_instance(instance)
        gentest_runner.restore_env(env, instance, args.install_timeout, activation=activation)
        model_cfg = cfgi.get("model", {})
        isolated_generated_test_gate = bool(
            getattr(args, "isolated_generated_test_gate", False)
        )
        rec["isolated_generated_test_gate"] = isolated_generated_test_gate
        repair_seed_messages = []
        round0_turns = 0
        if fresh_solve_round0:
            solve_agent_config = copy.deepcopy(cfgi.get("agent", {}))
            solve_agent_config["step_limit"] = int(getattr(args, "round0_steps", args.steps))
            solve_agent_config["cost_limit"] = float(getattr(args, "round0_cost_limit", 5.0))
            solve_agent_config["output_path"] = trace_dir / "round_00_solve.traj.json"
            solve_agent = DefaultAgent(rca.get_model(config=model_cfg), env, **solve_agent_config)
            solve_result = solve_agent.run(instance["problem_statement"])
            round0_turns = solve_agent.n_calls
            seed_format = "responses" if model_cfg.get("model_class") == "trapi_response" else "chat"
            repair_seed_messages = f2p_helpers.live_seed_messages(solve_agent.messages, seed_format=seed_format)
            rec["round0_exit_status"] = solve_result.get("exit_status", "")
            rec["round0_submission_chars"] = len((solve_result.get("submission") or "").strip())
            rec["round0_turns"] = round0_turns
            rec["round0_trajectory"] = str(solve_agent_config["output_path"])
            rec["seed_msg_count"] = len(repair_seed_messages)
            rec["seed_origin"] = "fresh_round0_e2e_trajectory"
            rec["wrong_patch_applied"] = None
            rec["wrong_patch_apply_errors"] = []
            rec["initial_patch"] = _source_diff(env)
            rec["round0_patch"] = rec["initial_patch"]
            rec["round0_patch_chars"] = len(rec["round0_patch"])
        else:
            applied = False
            apply_errors = []
            if not patch_candidates and bool(getattr(args, "allow_empty_source_patch", False)):
                applied = True
                rec["applied_patch_candidate"] = None
            for idx, patch in enumerate(patch_candidates):
                applied, apply_err = rca.apply_prior_patch(env, patch)
                apply_errors.append({
                    "candidate": idx,
                    "chars": len(patch),
                    "applied": bool(applied),
                    "error_tail": (apply_err or "")[-600:],
                })
                if applied:
                    rec["applied_patch_candidate"] = idx
                    break
                rca.shell(env, "cd /testbed && git checkout -- . && git clean -fdq")
            rec["wrong_patch_applied"] = applied
            rec["wrong_patch_apply_errors"] = apply_errors
            rec["round0_patch"] = patch_candidates[0] if patch_candidates else ""
            rec["round0_patch_chars"] = len(rec["round0_patch"])
            if bool(getattr(args, "seed_trajectory", False)):
                seed_format = "responses" if model_cfg.get("model_class") == "trapi_response" else "chat"
                if source_messages is not None:
                    repair_seed_messages = f2p_helpers.live_seed_messages(
                        source_messages,
                        seed_format=seed_format,
                    )
                    rec["seed_origin"] = "provided_round0_trajectory"
                else:
                    tp = args.source_run / iid / f"{iid}.traj.json"
                    if not tp.exists():
                        tp = args.source_run / f"{iid}.traj.json"
                    repair_seed_messages = f2p_helpers.load_seed_messages(tp, seed_format=seed_format)
                    rec["seed_origin"] = "provided_round0_trajectory_file"
                rec["seed_msg_count"] = len(repair_seed_messages)
        if not fresh_solve_round0 and not applied:
            recovery_attempts = []
            recovered_patch = ""
            for recovery_round in range(1, args.apply_failure_rounds + 1):
                ncalls, status = run_apply_failure_round(
                    model_cfg,
                    env,
                    instance["problem_statement"],
                    patch_candidates,
                    apply_errors,
                    args.apply_failure_steps,
                )
                recovered_diff = rca.shell(env, "cd /testbed && git diff").get("output", "") or ""
                recovered_patch = recovered_diff if rca._looks_like_diff(recovered_diff) else ""
                recovery_attempts.append({
                    "round": recovery_round,
                    "turns": ncalls,
                    "agent_status": status,
                    "patch": recovered_patch,
                })
                if recovered_patch:
                    break
            rec["apply_failure_recoveries"] = recovery_attempts
            rec["apply_failure_recovery"] = recovery_attempts[-1] if recovery_attempts else {
                "round": 0,
                "turns": 0,
                "agent_status": "not_run",
                "patch": "",
            }
            if not recovered_patch:
                rec["error"] = "APPLY_FAILURE_RECOVERY_NO_PATCH"
                rec["initial_pass"] = False
                rec["rescued"] = False
                rec["gate_status"] = "inconclusive"
                rec["gate_reason"] = "apply_failure_recovery_no_patch"
                rec["final_gate_status"] = "inconclusive"
                rec["final_gate_reason"] = "apply_failure_recovery_no_patch"
                rec["model_final_submit"] = False
                rec["initial_test_result"] = _test_result(
                    failure_kind="apply_failure_recovery_no_patch",
                    output_tail=(apply_errors[-1].get("error_tail", "") if apply_errors else "NO_PATCH_CANDIDATES"),
                )
                rec["final_test_result"] = rec["initial_test_result"]
                rec["rounds_used"] = 0
                rec["turns_per_round"] = []
                rec["patches_per_round"] = []
                rec["total_turns"] = sum(int(a.get("turns") or 0) for a in recovery_attempts)
                rec["final_tail"] = rec["final_test_result"]["output_tail"][-FINAL_TAIL_CHARS:]
                rec["initial_patch"] = ""
                rec["final_patch"] = ""
                return rec
            rec["error"] = "WRONG_PATCH_APPLY_RECOVERED"
            rec["initial_patch"] = recovered_patch
        elif not fresh_solve_round0:
            initial_diff = rca.shell(env, "cd /testbed && git diff").get("output", "") or ""
            rec["initial_patch"] = initial_diff if rca._looks_like_diff(initial_diff) else ""

        if isolated_generated_test_gate:
            test_patch_alignment = {
                "enabled": bool(args.apply_test_patch_at_start),
                "phase": "repair",
                "applied": False,
                "reason": "",
                "error": "",
                "isolated_to_verifier": True,
                "changed_paths": [],
            }
        else:
            test_patch_alignment = gentest_runner.apply_test_patch_for_alignment(
                env, instance, enabled=args.apply_test_patch_at_start, phase="repair"
            )
        rec["test_patch_alignment"] = test_patch_alignment
        if test_patch_alignment["reason"]:
            rec["initial_pass"] = False
            rec["rescued"] = False
            rec["gate_status"] = "inconclusive"
            rec["gate_reason"] = test_patch_alignment["reason"]
            rec["final_gate_status"] = "inconclusive"
            rec["final_gate_reason"] = test_patch_alignment["reason"]
            rec["model_final_submit"] = False
            rec["initial_test_result"] = _test_result(
                failure_kind=test_patch_alignment["reason"],
                output_tail=test_patch_alignment.get("error") or test_patch_alignment["reason"],
            )
            rec["final_test_result"] = rec["initial_test_result"]
            rec["rounds_used"] = 0
            rec["turns_per_round"] = []
            rec["patches_per_round"] = []
            rec["total_turns"] = 0
            rec["final_tail"] = rec["initial_test_result"]["output_tail"][-FINAL_TAIL_CHARS:]
            rec["final_patch"] = ""
            return rec

        if isolated_generated_test_gate and tests:
            generated_test_gate = PersistentGeneratedTestGate(cfgi, instance, tests, args)
            rec["generated_test_gate_workspace"] = "separate_persistent_sandbox"
            test_result = generated_test_gate.evaluate(_source_diff(env))
        else:
            rec["generated_test_gate_workspace"] = "policy_workspace"
            test_result = generated_test_check(env, instance, tests, args.test_timeout)
        passed = bool(test_result.get("passed"))
        tail = test_result.get("output_tail", "")
        feedback = _feedback_from_test_result(test_result)
        rec["initial_pass"] = passed
        rec["initial_gate_status"] = test_result.get("gate_status")
        rec["initial_gate_reason"] = test_result.get("gate_reason")
        rec["initial_test_result"] = test_result
        final_test_result = test_result
        model_final_submit = False
        model_final_submit_gate_reused = False
        model_final_decision = False
        model_keep_original = False
        fallback_to_round0 = False
        selected_patch = None
        rounds = 0
        turns_per_round = []
        patches_per_round = []
        stop_reason = ""
        stop_for_blocking_gate = test_result.get("failure_kind") in STOP_REPAIR_FAILURE_KINDS
        initial_patch_empty = not bool((rec.get("initial_patch") or "").strip())
        rec["initial_patch_empty"] = initial_patch_empty
        if stop_for_blocking_gate:
            stop_reason = f"stop_kind:{test_result.get('failure_kind')}"
        if (not passed or initial_patch_empty) and not stop_for_blocking_gate:
            excluded_paths = test_patch_alignment.get("changed_paths", [])

            def _current_patch():
                return _source_diff(env, excluded_paths)

            def _on_submit():
                source_diff = _current_patch()
                if generated_test_gate is not None:
                    result = generated_test_gate.evaluate(source_diff)
                else:
                    result = generated_test_check(env, instance, tests, args.test_timeout)
                return result, source_diff

            configured_max_gate_checks = int(getattr(args, "max_gate_checks", 0) or 0)
            max_gate_checks = configured_max_gate_checks or args.max_rounds
            continue_on_duplicate_patch = bool(getattr(args, "continue_on_duplicate_patch", False))
            force_submit_on_turn_limit = bool(getattr(args, "force_submit_on_turn_limit", True))
            rec["max_gate_checks"] = max_gate_checks
            rec["continue_on_duplicate_patch"] = continue_on_duplicate_patch
            rec["force_submit_on_turn_limit"] = force_submit_on_turn_limit

            agent = run_repair_rounds(
                model_cfg,
                env,
                instance["problem_statement"],
                feedback,
                args.steps,
                args.max_rounds,
                _on_submit,
                get_current_patch=_current_patch,
                initial_patch=rec.get("initial_patch", ""),
                max_gate_checks=configured_max_gate_checks or None,
                continue_on_duplicate_patch=continue_on_duplicate_patch,
                force_submit_on_turn_limit=force_submit_on_turn_limit,
                initial_test_result=test_result,
                seed_messages=repair_seed_messages,
                output_path=trace_dir / "seed_repair.traj.json",
                repair_template=(
                    EMPTY_PATCH_REPAIR_TEMPLATE
                    if initial_patch_empty and passed else REPAIR_TEMPLATE
                ),
            )
            rounds = len(agent.rounds)
            turns_per_round = [row["turns"] for row in agent.rounds]
            patches_per_round = [
                {key: row[key] for key in (
                    "round", "turns", "decision", "passed", "gate_status", "gate_reason",
                    "test_result", "patch", "forced_submission"
                )}
                for row in agent.rounds
            ]
            passed = bool(agent.passed)
            stop_reason = agent.stop_reason
            rec["duplicate_submissions"] = agent.duplicate_submissions
            model_final_submit = agent.model_final_submit
            model_final_submit_gate_reused = agent.model_final_submit_gate_reused
            model_final_decision = agent.model_final_decision
            model_keep_original = agent.model_keep_original
            fallback_to_round0 = agent.fallback_to_round0
            selected_patch = agent.selected_patch
            if not model_final_decision and selected_patch is None:
                selected_patch = rec.get("initial_patch", "")
                fallback_to_round0 = True
            if agent.final_test_result is not None:
                final_test_result = agent.final_test_result
                tail = final_test_result.get("output_tail", "")
        rec["rescued"] = bool(passed)
        rec["stop_reason"] = stop_reason
        rec["gate_status"] = final_test_result.get("gate_status")
        rec["gate_reason"] = final_test_result.get("gate_reason")
        rec["final_gate_status"] = final_test_result.get("gate_status")
        rec["final_gate_reason"] = final_test_result.get("gate_reason")
        rec["model_final_submit"] = bool(model_final_submit)
        rec["model_final_submit_gate_reused"] = bool(model_final_submit_gate_reused)
        rec["model_final_decision"] = bool(model_final_decision)
        rec["model_keep_original"] = bool(model_keep_original)
        rec["fallback_to_round0"] = bool(fallback_to_round0)
        rec["final_test_result"] = final_test_result
        rec["rounds_used"] = rounds
        rec["turns_per_round"] = turns_per_round
        rec["patches_per_round"] = patches_per_round
        recovery_turns = sum(int(a.get("turns") or 0) for a in rec.get("apply_failure_recoveries", []))
        if not recovery_turns:
            recovery_turns = int((rec.get("apply_failure_recovery") or {}).get("turns") or 0)
        rec["repair_turns"] = agent.n_calls if not rec["initial_pass"] and not stop_for_blocking_gate else 0
        rec["total_turns"] = round0_turns + rec["repair_turns"] + recovery_turns
        rec["final_tail"] = tail[-FINAL_TAIL_CHARS:]
        rec["final_patch"] = (
            selected_patch
            if selected_patch is not None
            else _source_diff(env, test_patch_alignment.get("changed_paths", []))
        )
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec.setdefault("gate_status", "inconclusive")
        rec.setdefault("gate_reason", "harness_exception")
        rec.setdefault("final_gate_status", rec["gate_status"])
        rec.setdefault("final_gate_reason", rec["gate_reason"])
        rec.setdefault("model_final_submit", False)
        rec.setdefault("final_test_result", _test_result(failure_kind="harness_exception", output_tail=rec["error"]))
        rec.setdefault("final_tail", rec["error"])
    finally:
        try:
            if generated_test_gate is not None:
                generated_test_gate.close()
        except Exception:
            pass
        try:
            if env is not None:
                env.cleanup()
        except Exception:
            pass
    return rec


def _msg_text(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return ""


def recover_patch_from_traj(d):
    best = ""
    for m in d.get("messages", []) or []:
        t = _msg_text(m)
        if "diff --git" not in t:
            continue
        cand = t[t.find("diff --git"):]
        cand = re.split(r"\n(?:COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|\$ |\(testbed\)|root@)", cand)[0]
        if ("@@" in cand or ("+++ " in cand and "--- " in cand)) and len(cand) > len(best):
            best = cand
    return best.strip()


def patch_candidates_from_traj(d):
    """Return likely unified diffs from a source trajectory, preferring recorded submission.

    Some trajectories have a non-empty info.submission that is not a patch (for example a
    source snippet or "Patch created"). Keep only diff-like candidates and fall back to
    the best diff found in the trajectory messages.
    """
    candidates = []
    for patch in [
        (d.get("info", {}) or {}).get("submission", "") or "",
        recover_patch_from_traj(d),
    ]:
        patch = (patch or "").strip()
        if not patch or not rca._looks_like_diff(patch):
            continue
        if patch not in candidates:
            candidates.append(patch)
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tests",
        required=True,
        type=Path,
        help=(
            "JSON {instance_id: [test,...]}; entries are canonical "
            "{test_patch,test_command} or legacy full-file {test_code,...}"
        ),
    )
    ap.add_argument("--source-run", required=True, type=Path, help="dir with <iid>/<iid>.traj.json wrong patches")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--subset", default="rebench")
    ap.add_argument("--split", default="filtered")
    ap.add_argument("--api-base", default="http://127.0.0.1:{port}/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--sandbox-url", default=os.getenv("SANDBOX_BASE_URL", ""))
    ap.add_argument("--sandbox-key", default=os.getenv("SANDBOX_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--test-timeout", type=int, default=240)
    ap.add_argument("--install-timeout", type=int, default=600)
    ap.add_argument("--apply-failure-rounds", type=int, default=3,
                    help="patch-recovery rounds to try when the source patch cannot be applied")
    ap.add_argument("--apply-failure-steps", type=int, default=100,
                    help="steps for the patch-recovery round when the source patch cannot be applied")
    ap.add_argument("--max-tests", type=int, default=3, help="cap tests used per instance")
    ap.add_argument(
        "--apply-test-patch-at-start",
        action="store_true",
        help="apply the instance test_patch before generated-test checks, matching gentest generation/validation",
    )
    ap.add_argument(
        "--isolated-generated-test-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run generated-test checks in a separate persistent reset-before-grade sandbox",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None,
                    help="override model_name (e.g. hosted_vllm/<served-path>); default keeps swerebench.yaml model")
    ap.add_argument("--model-class", default=None,
                    help="override model_class (e.g. trapi_response for Codex/TRAPI)")
    ap.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Responses API reasoning effort for trapi_response models.",
    )
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Set model.model_kwargs.chat_template_kwargs.enable_thinking=false for Qwen-style tool calling.")
    args = ap.parse_args()
    if not args.sandbox_url:
        ap.error("--sandbox-url or SANDBOX_BASE_URL must be set explicitly")

    tests_map = json.loads(args.tests.read_text())
    tests_map = {i: (v or [])[: args.max_tests] for i, v in tests_map.items() if v}
    ds = {r["instance_id"]: dict(r) for r in load_dataset(rca.DATASET_MAPPING.get(args.subset, args.subset), split=args.split)}

    todo = {}
    for iid in tests_map:
        if iid not in ds:
            continue
        tp = args.source_run / iid / f"{iid}.traj.json"
        if not tp.exists():
            tp = args.source_run / f"{iid}.traj.json"   # flat layout fallback
        if not tp.exists():
            continue
        d = json.loads(tp.read_text())
        patches = patch_candidates_from_traj(d)
        if patches:
            todo[iid] = patches
    if args.limit:
        todo = dict(list(todo.items())[: args.limit])

    existing = set()
    if args.out.exists():
        for l in args.out.read_text().splitlines():
            if l.strip():
                try:
                    existing.add(json.loads(l)["instance_id"])
                except Exception:
                    pass
    work = [(i, p) for i, p in todo.items() if i not in existing]
    print(f"tests_instances={len(tests_map)} with_wrong_patch={len(todo)} todo={len(work)} "
          f"existing={len(existing)} max_rounds={args.max_rounds} steps={args.steps} workers={args.workers}",
          flush=True)

    cfg = build_config(
        args.api_base,
        args.api_key,
        args.sandbox_url,
        args.sandbox_key,
        args.temperature,
        args.model,
        args.model_class,
        args.disable_thinking,
        reasoning_effort=args.reasoning_effort,
    )
    done = resc = init = 0
    gate_counts = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", buffering=1) as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, iid, ds[iid], patch, tests_map[iid], cfg, args): iid
                    for iid, patch in work}
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result()
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                done += 1
                resc += int(bool(r.get("rescued")))
                init += int(bool(r.get("initial_pass")))
                gate = r.get("gate_status") or "unknown"
                gate_counts[gate] = gate_counts.get(gate, 0) + 1
                if done == 1 or done % 5 == 0:
                    print(f"done={done}/{len(work)} rescued={resc} (initial_pass={init}) "
                          f"last={r.get('instance_id')} rescued={r.get('rescued')} "
                          f"gate={gate} gates={gate_counts} rounds={r.get('rounds_used')}",
                          flush=True)
    print(f"TOTAL done={done} rescued={resc} initial_pass={init} "
          f"rescue_rate={resc/max(1,done):.3f} gates={gate_counts}", flush=True)


if __name__ == "__main__":
    main()
