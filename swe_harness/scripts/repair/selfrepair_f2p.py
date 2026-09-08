#!/usr/bin/env python3
"""Test-driven SELF-REPAIR with ONE gold fail-to-pass test as the in-loop oracle.

Population: instances that submitted a patch but were WRONG (unresolved).
Per instance:
  1. restore the wrong patch onto /testbed (+ canonical env install)
  2. pick ONE random FAIL_TO_PASS node (seeded by instance_id)
  3. f2p-check: apply gold test_patch, run that node, capture pass/fail + output, reverse test_patch
  4. if it already passes -> rescued (initial)
     else: up to --max-rounds repair rounds. Each round the policy model gets the
     failing test OUTPUT (not the test source) as feedback and edits SOURCE in the
     sandbox for up to --steps steps; at that boundary the current diff is submitted
     by default, then the same f2p gate is re-checked.
  rescued = the chosen f2p passes after repair (per user's criterion).

Each round writes a complete agent trajectory and raw test log under
<out stem>.round_trajs/ by default.

Reuses repair_critic_actor (rca) for sandbox env + DefaultAgent.
"""
import argparse, base64, hashlib, json, os, random, re, shlex, sys, time, copy
import concurrent.futures
import importlib.util
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS / "src"))
sys.path.insert(0, str(HARNESS / "external" / "azure-modal"))

spec = importlib.util.spec_from_file_location("rca", HARNESS / "scripts" / "repair" / "repair_critic_actor.py")
rca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rca)

from datasets import load_dataset
from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks import official_f2p_gate
from minisweagent.run.benchmarks import swebench_verify_azure_modal as swebench_verify
from minisweagent.run.benchmarks import swebench_pro_verify_azure_modal as swebench_pro_verify
from minisweagent.utils.serialize import UNSET, recursive_merge
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow, Submitted

SYSTEM = "You are a careful software engineer that interacts with a computer shell to solve programming tasks."

# Verbatim repair prompt from the repair SFT corpus
# (sft_codex53_repair.no_think.repairloss.jsonl). When --seed-trajectory is set we reproduce the
# training format: original solve trajectory as seed history, then THIS user prompt, then continue.
SEED_REPAIR_TEMPLATE = (
    "Your previous patch attempt has been applied to /testbed, but a hidden regression test still "
    "FAILS. You cannot see the test source, only its failure output below. Continue editing the "
    "SOURCE code so the described behavior is correct and the failing test passes. Do not edit or "
    "create test files.\n\n<failing_test_output>\n{feedback}\n</failing_test_output>"
)

SEED_REPAIR_FOLLOWUP = (
    "The hidden regression test still FAILS after your last edit. Keep the existing source changes, "
    "use the updated failure output below, make the next minimal source-only edit, and submit again.\n\n"
    "<failing_test_output>\n{feedback}\n</failing_test_output>"
)

DUPLICATE_PATCH_FOLLOWUP = (
    "The repair gate was NOT rerun because this patch is identical to the previous failed candidate. "
    "This check did not consume the gate-query budget. Make a concrete SOURCE change before checking again."
)

REPAIR_FEEDBACK_CHARS = 6000

REPAIR_TEMPLATE = """\
You are fixing a bug. A previous patch attempt is ALREADY APPLIED in /testbed but a hidden
regression test still FAILS. You CANNOT see the test file; you only get its failure output below.
Your job: edit the SOURCE code in /testbed so the described behavior is correct and the failing
test would pass. Do NOT edit or create test files.

<original_issue>
{{task}}
</original_issue>

<failing_test_node>
{{node}}
</failing_test_node>

<failing_test_output>
{{feedback}}
</failing_test_output>

Steps:
1. `cd /testbed && git diff` to see the already-applied previous attempt.
2. Read the failing-test output carefully — it shows the exact assertion/exception and the
   actual-vs-expected values. Infer the required behavior from it + the issue. If MULTIPLE
   tests are shown, your fix must satisfy ALL of them.
3. Make the minimal SOURCE edit(s) to satisfy that behavior. Only non-test files.
4. Re-derive your own quick check if helpful, then submit.

Submit with EXACTLY two separate commands (must exit 0):
  cd /testbed && git diff -- <non-test source files> > /tmp/fix.patch
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/fix.patch
"""


def _msg_text(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return ""


def recover_patch_from_traj(d):
    """Recover a unified diff from the trajectory messages (used when info.submission
    is empty but the agent did emit a diff in the conversation)."""
    import re as _re
    best = ""
    for m in d.get("messages", []) or []:
        t = _msg_text(m)
        if "diff --git" not in t:
            continue
        cand = t[t.find("diff --git"):]
        cand = _re.split(r"\n(?:COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|\$ |\(testbed\)|root@)", cand)[0]
        if ("@@" in cand or ("+++ " in cand and "--- " in cand)) and len(cand) > len(best):
            best = cand
    return best.strip()


def patch_candidates_from_traj(d):
    """Return diff-like patch candidates from a trajectory.

    Some runs store a non-empty info.submission that is actually a source snippet
    or status text. Keep only diff-like candidates and fall back to the best diff
    emitted in the trajectory messages.
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


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return list(decoded) if isinstance(decoded, (list, tuple)) else [str(decoded)]
    return list(value)


def build_config(api_base, api_key, sandbox_url, sandbox_key, temperature, model_name=None, model_class=None,
                 disable_thinking=False, top_p=None, reasoning_effort=None):
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


ACTIVATE = (
    "source /opt/miniconda3/bin/activate 2>/dev/null; "
    "source /opt/conda/bin/activate 2>/dev/null; "
    "conda activate testbed 2>/dev/null; "
)

PYTEST_PASS = re.compile(r"\b(\d+) passed\b")
PYTEST_FAIL = re.compile(r"\b(\d+) failed\b")
PYTEST_ERROR = re.compile(r"\b(\d+) errors?\b")
PYTEST_NOT_PASSED = re.compile(r"\b(\d+) (?:skipped|xfailed|xpassed)\b")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PYTEST_BLOCKED = re.compile(
    r"No module named|ModuleNotFoundError|ImportError|cannot import name|"
    r"collected 0 items|no tests ran|ERROR collecting|file or directory not found|"
    r"SyntaxError|INTERNALERROR",
    re.I,
)


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


def write_file(env, path, content):
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    rca.shell(env, f"printf %s '{b64}' | base64 -d > {shlex.quote(path)}")


def _restore_submitted_patch(env, instance, submitted_patch):
    """Replace /testbed with the exact patch carried by a submit event."""
    base_commit = str(instance.get("base_commit") or "HEAD")
    reset = rca.shell(
        env,
        f"cd /testbed && git reset --hard {shlex.quote(base_commit)} && git clean -fdq",
    )
    if (reset.get("returncode") or 0) != 0:
        return False, f"SUBMITTED_PATCH_BASE_RESET_FAILED\n{(reset.get('output') or '')[-1000:]}", ""
    if not (submitted_patch or "").strip():
        return False, "SUBMITTED_PATCH_EMPTY", ""

    patch_path = "/tmp/submitted.patch"
    write_file(env, patch_path, submitted_patch)
    check = rca.shell(
        env,
        f"cd /testbed && git apply --check --whitespace=nowarn {patch_path} 2>&1",
    )
    if (check.get("returncode") or 0) != 0:
        rca.shell(env, f"rm -f {patch_path}")
        return False, f"SUBMITTED_PATCH_DOES_NOT_APPLY\n{(check.get('output') or '')[-1000:]}", ""
    applied = rca.shell(
        env,
        f"cd /testbed && git apply --whitespace=nowarn {patch_path} 2>&1",
    )
    rca.shell(env, f"rm -f {patch_path}")
    if (applied.get("returncode") or 0) != 0:
        rca.shell(
            env,
            f"cd /testbed && git reset --hard {shlex.quote(base_commit)} && git clean -fdq",
        )
        return False, f"SUBMITTED_PATCH_APPLY_FAILED\n{(applied.get('output') or '')[-1000:]}", ""
    gate_patch = rca.shell(
        env,
        "cd /testbed && git -c core.fileMode=false diff HEAD --binary",
    ).get("output", "") or ""
    return True, "", gate_patch


def _patch_paths(patch):
    paths = []
    for match in re.finditer(r"(?m)^(?:---|\+\+\+) [ab]/(.+)$", patch or ""):
        path = match.group(1).strip()
        if path and path != "/dev/null" and path not in paths:
            paths.append(path)
    return paths


def _cleanup_test_patch(env, test_patch):
    """Remove only the gold test patch so source edits from repair rounds persist."""
    rev = rca.shell(env, "cd /testbed && git apply -R --whitespace=nowarn /tmp/test_patch.diff 2>&1")
    if (rev.get("returncode") or 0) == 0:
        return True, rev.get("output", "") or ""

    paths = _patch_paths(test_patch)
    if not paths:
        return False, rev.get("output", "") or ""

    qpaths = " ".join(shlex.quote(p) for p in paths)
    cleanup = rca.shell(
        env,
        f"cd /testbed && "
        f"(git checkout -- {qpaths} 2>/dev/null || true) && "
        f"(git clean -fq -- {qpaths} 2>/dev/null || true)",
    )
    status = rca.shell(env, f"cd /testbed && git status --porcelain -- {qpaths}")
    clean = ((status.get("returncode") or 0) == 0) and not (status.get("output", "") or "").strip()
    output = "\n".join(x for x in (rev.get("output", ""), cleanup.get("output", ""), status.get("output", "")) if x)
    return clean, output


def _pytest_rc_and_output(output):
    m = re.search(r"__RC__=(-?\d+)\s*$", (output or "").strip())
    rc = int(m.group(1)) if m else 1
    out_clean = re.sub(r"\n?__RC__=-?\d+\s*$", "", output or "")
    return rc, out_clean


def _django_test_label(node):
    match = re.fullmatch(r"([^ ]+) \(([^)]+)\)", node or "")
    if not match:
        return ""
    method, test_case = match.groups()
    return f"{test_case}.{method}"


def _build_test_invocation(instance, nodes):
    repo = instance.get("repo", "")
    nodes = list(nodes) if isinstance(nodes, (list, tuple)) else [nodes]
    nodes = [node for node in nodes if node]

    if repo == "django/django":
        labels = [_django_test_label(node) for node in nodes]
        targets = labels if labels and all(labels) else swebench_verify._get_test_directives(instance)
        return "./tests/runtests.py --verbosity 2 " + " ".join(shlex.quote(target) for target in targets)

    if repo == "sympy/sympy":
        targets = swebench_verify._get_test_directives(instance)
        return "bin/test -C --verbose " + " ".join(shlex.quote(target) for target in targets)

    targets = nodes if nodes and all("::" in node for node in nodes) else swebench_verify._get_test_directives(instance)
    target_args = " ".join(shlex.quote(target) for target in targets)
    return (
        "PYTEST_NO_HEADER=''; "
        "python -m pytest --help 2>/dev/null | grep -q -- '--no-header' && PYTEST_NO_HEADER=--no-header; "
        "python -m pytest $PYTEST_NO_HEADER -rA --tb=short -p no:cacheprovider "
        f"{target_args}"
    )


def _node_search_terms(node):
    terms = [node]
    if "::" in node:
        terms.append(node.rsplit("::", 1)[-1].split("[")[0])
    django_match = re.fullmatch(r"([^ ]+) \(([^)]+)\)", node or "")
    if django_match:
        terms.extend(django_match.groups())
    return [term for term in dict.fromkeys(terms) if len(term) >= 4]


def _collection_error_hint(output, rc):
    """Detect pytest not-collected / collection-error states so a node reported MISSING
    because it was never run is distinguished from a node that ran and failed."""
    clean = ANSI_ESCAPE.sub("", output or "")
    signals = []
    if rc == 4:
        signals.append("pytest_rc=4 (usage/collection error)")
    if re.search(r"collected 0 items", clean):
        signals.append("collected 0 items")
    if re.search(r"ERROR: not found:", clean) or re.search(r"no tests? ran", clean):
        signals.append("selector matched no test (check node id / params)")
    if re.search(r"\berrors?\b.*during collection", clean, re.IGNORECASE):
        signals.append("error during collection (likely import error from patch)")
    return "; ".join(dict.fromkeys(signals))


def _focused_test_feedback(output, nodes, parsed, rc, max_chars=6000):
    clean = ANSI_ESCAPE.sub("", output or "")
    nodes = list(nodes) if isinstance(nodes, (list, tuple)) else [nodes]
    selected = {node: parsed.get(node, "MISSING") for node in nodes}
    header = [f"test_return_code: {rc}", "selected_f2p_statuses:"]
    header.extend(f"- {node}: {status}" for node, status in selected.items())
    coll = _collection_error_hint(clean, rc)
    if coll and any(status == "MISSING" for status in selected.values()):
        header.append(f"collection_note: {coll}")

    lines = clean.splitlines()
    selected_lines = set()
    for node in nodes:
        terms = _node_search_terms(node)
        for index, line in enumerate(lines):
            if any(term in line for term in terms):
                selected_lines.update(range(max(0, index - 12), min(len(lines), index + 28)))

    # Explicitly capture the pytest FAILURES/ERRORS traceback block: with --tb=short the
    # assertion/traceback lives in a "=== FAILURES ===" section that the node-name search
    # above misses, and the 1200-char summary tail would otherwise clip it away.
    failure_block = ""
    fail_start = None
    for index, line in enumerate(lines):
        if re.match(r"=+ (FAILURES|ERRORS) =+", line):
            fail_start = index
            break
    if fail_start is not None:
        fail_end = len(lines)
        for index in range(fail_start + 1, len(lines)):
            if re.match(r"=+ (short test summary|warnings summary|passed|failed|[0-9]) ", lines[index]) \
               or re.match(r"=+ \d+ (passed|failed|error)", lines[index]):
                fail_end = index
                break
        failure_block = "\n".join(lines[fail_start:fail_end])[-2600:]

    if selected_lines:
        focused = "\n".join(lines[index] for index in sorted(selected_lines))
    else:
        focused = clean[-4200:]
    summary_tail = clean[-1200:]
    feedback = "\n".join(header)
    if failure_block:
        feedback += "\n\nfailure_traceback:\n" + failure_block
    feedback += "\n\nfocused_test_output:\n" + focused
    if summary_tail and summary_tail not in focused:
        feedback += "\n\nfull_output_tail:\n" + summary_tail
    if len(feedback) > max_chars:
        # Preserve header + failure_traceback (the actionable part) and the summary tail;
        # trim the middle focused/full-output sections rather than the traceback.
        head = "\n".join(header)
        if failure_block:
            head += "\n\nfailure_traceback:\n" + failure_block
        budget = max_chars - len(head) - 1200
        focused_trimmed = focused[-budget:] if budget > 0 else ""
        feedback = head + "\n\nfocused_test_output:\n" + focused_trimmed + \
            "\n\nfull_output_tail:\n" + summary_tail[-1150:]
    return feedback[-max_chars:]


def _full_eval_script_f2p_check(
    env,
    instance,
    nodes,
    timeout,
    log_path,
    submitted_patch,
    official_gate_session=None,
):
    """Run the shared official evaluator in one reset-before-grade sandbox."""
    if submitted_patch is None:
        submitted_patch = rca.shell(
            env,
            "cd /testbed && git -c core.fileMode=false diff HEAD --binary",
        ).get("output", "") or ""
    submitted_patch = str(submitted_patch or "")
    if not submitted_patch.strip():
        return False, "SUBMITTED_PATCH_EMPTY", {
            "error": "SUBMITTED_PATCH_EMPTY",
            "command": "official_full_eval_script",
            "full_eval_script_gate": True,
        }

    owns_session = official_gate_session is None
    if owns_session:
        official_gate_session = official_f2p_gate.PersistentOfficialGate(
            instance,
            timeout=timeout,
        )
    try:
        result = official_gate_session.evaluate(
            submitted_patch,
            fail_to_pass=list(nodes),
            pass_to_pass=[],
            timeout=timeout,
        )
    finally:
        if owns_session:
            official_gate_session.close()
    output = str(result.get("output") or "")

    parsed = {
        **{node: "PASSED" for node in result.get("fail_to_pass_passed", [])},
        **{node: "FAILED" for node in result.get("fail_to_pass_failed", [])},
    }
    returncode = int(result.get("exit_code") if result.get("exit_code") is not None else -1)
    test_command = str(result.get("test_command") or "official_full_eval_script")
    test_files = [str(path) for path in result.get("test_files", [])]
    execution_contract = "\n".join([
        "test_execution_contract:",
        "- language: python",
        f"- runner: {result.get('test_runner') or result.get('official_verifier') or 'unknown'}",
        "- working_directory: /testbed",
        f"- selected_nodes: {json.dumps(list(nodes), ensure_ascii=False)}",
        f"- test_files: {json.dumps(test_files, ensure_ascii=False)}",
        f"- verifier_command: {test_command}",
        "- test_source_visibility: hidden_verifier_only",
    ])
    feedback_budget = max(2000, REPAIR_FEEDBACK_CHARS - len(execution_contract) - 2)
    feedback = execution_contract + "\n\n" + _focused_test_feedback(
        output,
        nodes,
        parsed,
        returncode,
        max_chars=feedback_budget,
    )
    if not result.get("patch_applied"):
        feedback = "SUBMITTED_PATCH_APPLY_FAILED\n" + feedback
    if result.get("error"):
        feedback = f"OFFICIAL_EVAL_ERROR: {result['error']}\n" + feedback
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"test_command: {test_command}\n" + output,
            errors="replace",
        )

    patch_sha = hashlib.sha256(submitted_patch.encode()).hexdigest()
    passed = bool(
        result.get("resolved")
        and result.get("patch_applied")
        and not result.get("error")
    )
    selected_statuses = {node: parsed.get(node, "MISSING") for node in nodes}
    if passed:
        gate_status, gate_reason = "pass", "passed"
    elif not result.get("patch_applied"):
        gate_status, gate_reason = "inconclusive", "invalid_patch"
    elif result.get("error"):
        gate_status, gate_reason = "inconclusive", "verifier_error"
    elif any(status == "MISSING" for status in selected_statuses.values()) or re.search(
        r"unrecognized arguments|collected 0 items|no tests ran|ERROR collecting|during collection",
        output,
        re.I,
    ):
        gate_status, gate_reason = "inconclusive", "collection_failure"
    else:
        gate_status, gate_reason = "hard_fail", "semantic_failure"
    return passed, feedback, {
        "command": test_command,
        "test_files": test_files,
        "test_runner": result.get("test_runner", ""),
        "full_eval_script_gate": True,
        "persistent_in_loop_sandbox": True,
        "official_verifier": result.get("official_verifier", ""),
        "return_code": returncode,
        "selected_statuses": selected_statuses,
        "gate_status": gate_status,
        "gate_reason": gate_reason,
        "parsed_test_count": int(result.get("parsed_tests_count") or 0),
        "patch_applied": bool(result.get("patch_applied")),
        "official_eval_error": str(result.get("error") or ""),
        "log_path": str(log_path) if log_path else "",
        "submitted_patch_sha256": patch_sha,
        "gate_patch_sha256": patch_sha if result.get("patch_applied") else "",
        "worktree_patch_sha256": patch_sha if result.get("patch_applied") else "",
    }


def _pytest_all_passed(output, rc, expected_items):
    output = ANSI_ESCAPE.sub("", output or "")
    passed = sum(int(m.group(1)) for m in PYTEST_PASS.finditer(output))
    failed = sum(int(m.group(1)) for m in PYTEST_FAIL.finditer(output))
    errors = sum(int(m.group(1)) for m in PYTEST_ERROR.finditer(output))
    not_passed = sum(int(m.group(1)) for m in PYTEST_NOT_PASSED.finditer(output))
    return (
        rc == 0
        and passed >= expected_items
        and failed == 0
        and errors == 0
        and not_passed == 0
        and not PYTEST_BLOCKED.search(output)
    )


def f2p_check(
    env,
    instance,
    nodes,
    timeout,
    log_path=None,
    submitted_patch=None,
    *,
    full_eval_script_gate=True,
    official_gate_session=None,
):
    """Run one or more F2P nodes; every selected node must pass.

    The default path sends the exact submission to the caller-owned persistent
    official verifier. The compatibility path below applies/reverses the gold
    test patch directly in ``env``.
    """
    if isinstance(nodes, str):
        nodes = [nodes]
    nodes = [str(n) for n in nodes if str(n).strip()]
    if not nodes:
        return False, "NO_F2P_NODES", {"error": "NO_F2P_NODES"}
    test_patch = instance.get("test_patch", "")
    if not (test_patch or "").strip():
        return False, "NO_TEST_PATCH", {"error": "NO_TEST_PATCH"}

    if full_eval_script_gate:
        return _full_eval_script_f2p_check(
            env,
            instance,
            nodes,
            timeout,
            log_path,
            submitted_patch,
            official_gate_session,
        )

    explicit_submission = submitted_patch is not None
    gate_patch = ""
    if explicit_submission:
        restored, restore_error, gate_patch = _restore_submitted_patch(
            env,
            instance,
            str(submitted_patch or ""),
        )
        if not restored:
            patch = str(submitted_patch or "")
            return False, restore_error, {
                "error": restore_error.splitlines()[0],
                "submitted_patch_apply_error": restore_error,
                "submitted_patch_sha256": hashlib.sha256(patch.encode()).hexdigest() if patch else "",
                "gate_patch_sha256": "",
                "worktree_patch_sha256": "",
            }

    write_file(env, "/tmp/test_patch.diff", test_patch or "")
    applied = False
    passed = False
    feedback = ""
    apply_output = ""

    for check_cmd, apply_cmd in (
        ("git apply --check --whitespace=nowarn /tmp/test_patch.diff",
         "git apply --whitespace=nowarn /tmp/test_patch.diff"),
        ("git apply --check --3way --whitespace=nowarn /tmp/test_patch.diff",
         "git apply --3way --whitespace=nowarn /tmp/test_patch.diff"),
    ):
        check = rca.shell(env, f"cd /testbed && {check_cmd} 2>&1")
        apply_output = check.get("output", "") or apply_output
        if (check.get("returncode") or 0) != 0:
            continue
        result = rca.shell(env, f"cd /testbed && {apply_cmd} 2>&1")
        apply_output = result.get("output", "") or apply_output
        if (result.get("returncode") or 0) == 0:
            applied = True
            break

    if not applied:
        clean, cleanup_output = _cleanup_test_patch(env, test_patch)
        detail = "\n".join(x for x in ("TEST_PATCH_APPLY_FAILED", apply_output, cleanup_output) if x)
        if not clean:
            raise RuntimeError(f"TEST_PATCH_CLEANUP_FAILED\n{detail[-1200:]}")
        return False, detail[-1500:], {"error": "TEST_PATCH_APPLY_FAILED"}

    try:
        test_cmd = _build_test_invocation(instance, nodes)
        run = rca.shell(
            env,
            f"cd /testbed && "
            f"( {ACTIVATE} echo '{swebench_verify.START_TEST_OUTPUT}'; "
            f"{test_cmd}; test_rc=$?; "
            f"echo '{swebench_verify.END_TEST_OUTPUT}'; exit $test_rc ) 2>&1; "
            f"rc=$?; echo \"__RC__=$rc\"",
            timeout=timeout,
        )
        rc, out_clean = _pytest_rc_and_output(run.get("output", "") or "")
        parsed = swebench_verify._parse_test_output(out_clean, instance)
        passed = all(parsed.get(node) in ("PASSED", "XFAIL") for node in nodes)
        feedback = _focused_test_feedback(out_clean, nodes, parsed, rc)
        if log_path:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                f"test_command: {test_cmd}\nreturn_code: {rc}\n\n{out_clean}",
                errors="replace",
            )
        metadata = {
            "command": test_cmd,
            "return_code": rc,
            "selected_statuses": {node: parsed.get(node, "MISSING") for node in nodes},
            "parsed_test_count": len(parsed),
            "raw_output_chars": len(out_clean),
            "collection_error": _collection_error_hint(out_clean, rc),
            "log_path": str(log_path) if log_path else "",
            "submitted_patch_sha256": (
                hashlib.sha256(str(submitted_patch).encode()).hexdigest()
                if explicit_submission and submitted_patch
                else ""
            ),
            "gate_patch_sha256": (
                hashlib.sha256(str(submitted_patch).encode()).hexdigest()
                if explicit_submission and submitted_patch
                else ""
            ),
            "worktree_patch_sha256": (
                hashlib.sha256(gate_patch.encode()).hexdigest() if gate_patch else ""
            ),
        }
    finally:
        clean, cleanup_output = _cleanup_test_patch(env, test_patch)
        if not clean:
            raise RuntimeError(f"TEST_PATCH_CLEANUP_FAILED\n{cleanup_output[-1200:]}")

    return passed, feedback, metadata


SEED_KEEP_ROLES = {"system", "user", "assistant", "tool"}


def _chat_to_responses_items(msg):
    """Translate one litellm/chat-format message into Responses API input item(s).

    Solve trajectories are stored in chat format (assistant.tool_calls + role="tool"
    results), but the Responses API only accepts assistant/system/developer/user text
    plus function_call / function_call_output items. Passing role="tool" verbatim is a
    400. Emit the equivalent Responses items so the seeded tool loop replays cleanly.
    """
    role = msg.get("role")
    if role == "tool":
        call_id = msg.get("tool_call_id") or msg.get("call_id")
        if not call_id:
            return []
        return [{
            "type": "function_call_output",
            "call_id": call_id,
            "output": _msg_text(msg),
        }]
    if role == "assistant" and msg.get("tool_calls"):
        items = []
        text = _msg_text(msg)
        if text.strip():
            items.append({"type": "message", "role": "assistant",
                          "content": [{"type": "output_text", "text": text}]})
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            call_id = call.get("id") or call.get("call_id")
            name = fn.get("name")
            if not call_id or not name:
                continue
            arguments = fn.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            items.append({"type": "function_call", "call_id": call_id,
                          "name": name, "arguments": arguments})
        return items
    # plain system/user/assistant text
    return [{"role": role, "content": _msg_text(msg)}]


def load_seed_messages(traj_path, *, seed_format="responses"):
    """Load the original solve trajectory as seed history for continuation.

    Keeps system/user/assistant/tool turns (drops trailing exit/limit rows) and translates
    the chat-format tool loop when ``seed_format=responses``. For a LiteLLM/chat model,
    ``seed_format=chat`` preserves chat messages while removing dangling tool calls/outputs.
    """
    if seed_format not in {"responses", "chat"}:
        raise ValueError(f"unsupported seed format: {seed_format}")
    data = json.loads(Path(traj_path).read_text())
    kept = []
    for m in data.get("messages", []) or []:
        role = m.get("role")
        if role not in SEED_KEEP_ROLES:
            continue
        kept.append(m)
    # drop any trailing non-assistant/tool tail so the seed ends on the model's last
    # action/result, matching "previous patch applied to /testbed"
    while kept and kept[-1].get("role") not in ("assistant", "tool"):
        kept.pop()
    if seed_format == "chat":
        call_ids = {
            call.get("id") or call.get("call_id")
            for message in kept
            if message.get("role") == "assistant"
            for call in (message.get("tool_calls") or [])
        }
        output_ids = {
            message.get("tool_call_id") or message.get("call_id")
            for message in kept
            if message.get("role") == "tool"
        }
        paired_ids = call_ids & output_ids
        seed = []
        for message in kept:
            item = copy.deepcopy(message)
            if item.get("role") == "assistant" and item.get("tool_calls"):
                item["tool_calls"] = [
                    call for call in item["tool_calls"]
                    if (call.get("id") or call.get("call_id")) in paired_ids
                ]
                for call in item["tool_calls"]:
                    function = call.get("function") or {}
                    arguments = function.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        function["arguments"] = json.dumps(arguments)
                if not item["tool_calls"]:
                    item.pop("tool_calls")
                    if not _msg_text(item).strip():
                        continue
            elif item.get("role") == "tool":
                call_id = item.get("tool_call_id") or item.get("call_id")
                if call_id not in paired_ids:
                    continue
            seed.append(item)
        return seed
    seed = []
    for m in kept:
        seed.extend(_chat_to_responses_items(m))
    # The Responses API rejects unpaired tool items: a function_call with no matching
    # function_call_output ("No tool output found for function call ..."), or an output
    # with no preceding call. Solve trajectories can end on a dangling assistant tool_call
    # whose result was never recorded. Keep only fully paired call/output items.
    have_output = {it["call_id"] for it in seed if it.get("type") == "function_call_output"}
    have_call = {it["call_id"] for it in seed if it.get("type") == "function_call"}
    def _keep(it):
        t = it.get("type")
        if t == "function_call":
            return it.get("call_id") in have_output
        if t == "function_call_output":
            return it.get("call_id") in have_call
        return True
    seed = [it for it in seed if _keep(it)]
    return seed


def live_seed_messages(messages, *, seed_format="responses"):
    """Turn a just-finished solve agent conversation into repair seed history.

    A TRAPI solve stores each assistant turn as a Response object whose ``output``
    contains message/function-call items.  The final submit call is intentionally
    dangling because ``Submitted`` stops environment execution.  Flatten Response
    objects and retain only paired tool calls so the repair request can replay the
    freshly generated round-0 trajectory through the stateless Responses API.
    """
    if seed_format not in {"responses", "chat"}:
        raise ValueError(f"unsupported seed format: {seed_format}")
    if seed_format == "chat":
        kept = [copy.deepcopy(message) for message in messages if message.get("role") in SEED_KEEP_ROLES]
        call_ids = {
            call.get("id") or call.get("call_id")
            for message in kept
            if message.get("role") == "assistant"
            for call in (message.get("tool_calls") or [])
        }
        output_ids = {
            message.get("tool_call_id") or message.get("call_id")
            for message in kept
            if message.get("role") == "tool"
        }
        paired_ids = call_ids & output_ids
        seed = []
        for message in kept:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                message["tool_calls"] = [
                    call for call in message["tool_calls"]
                    if (call.get("id") or call.get("call_id")) in paired_ids
                ]
                if not message["tool_calls"]:
                    message.pop("tool_calls")
                    if not _msg_text(message).strip():
                        continue
            elif message.get("role") == "tool":
                if (message.get("tool_call_id") or message.get("call_id")) not in paired_ids:
                    continue
            seed.append(message)
        return seed

    seed = []
    for message in messages:
        if message.get("role") == "exit":
            continue
        if message.get("object") == "response":
            seed.extend(copy.deepcopy(message.get("output") or []))
        elif message.get("role") == "tool":
            seed.extend(_chat_to_responses_items(message))
        elif message.get("role") in {"system", "user", "assistant"} or message.get("type") in {
            "message", "function_call", "function_call_output",
        }:
            seed.append(copy.deepcopy(message))

    have_output = {item.get("call_id") for item in seed if item.get("type") == "function_call_output"}
    have_call = {item.get("call_id") for item in seed if item.get("type") == "function_call"}
    return [
        item for item in seed
        if item.get("type") not in {"reasoning", "function_call", "function_call_output"}
        or item.get("type") == "function_call" and item.get("call_id") in have_output
        or item.get("type") == "function_call_output" and item.get("call_id") in have_call
    ]


def _submission_from_interrupt(exc):
    for message in reversed(getattr(exc, "messages", ())):
        extra = message.get("extra") or {}
        if "submission" in extra:
            return str(extra.get("submission") or "")
    return ""


class SeedRepairAgent(DefaultAgent):
    """Persistent repair agent seeded with the original solve trajectory, then continued with the
    training repair prompt. Keeps ONE conversation across repair rounds (re-checks oracle on submit)."""

    def __init__(
        self,
        model,
        env,
        *,
        seed_messages,
        on_submit,
        max_rounds,
        steps_per_round,
        max_gate_checks=None,
        continue_on_duplicate_patch=False,
        force_submit_on_turn_limit=True,
        **kwargs,
    ):
        super().__init__(model, env, **kwargs)
        self._seed = list(seed_messages)
        self._on_submit = on_submit
        self._max_rounds = max_rounds
        self._gate_limit_stop_reason = (
            "max_gate_checks" if max_gate_checks not in (None, 0) else "max_rounds"
        )
        self._max_gate_checks = int(max_gate_checks or max_rounds)
        if self._max_gate_checks <= 0:
            raise ValueError(f"max_gate_checks must be positive, got {self._max_gate_checks}")
        self._continue_on_duplicate_patch = bool(continue_on_duplicate_patch)
        self.duplicate_submissions = 0
        if steps_per_round <= 0:
            raise ValueError(f"steps_per_round must be positive, got {steps_per_round}")
        self._steps_per_round = steps_per_round
        self._force_submit_on_turn_limit = force_submit_on_turn_limit
        self._round_start_calls = 0
        self.rounds = []
        self.passed = False
        self.final_test_result = None
        self.stop_reason = ""

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            try:
                outputs.append(self.env.execute(action))
            except Submitted:
                outputs.append({
                    "output": "Submission received; running the hidden regression test.",
                    "returncode": 0,
                    "exception_info": "",
                })
                self.add_messages(
                    *self.model.format_observation_messages(message, outputs, self.get_template_vars())
                )
                raise
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )

    def _handle_submission(self, submitted_patch, *, forced_submission):
        submitted_patch = str(submitted_patch or "")
        continue_on_duplicate_patch = bool(
            getattr(self, "_continue_on_duplicate_patch", False)
        )
        max_gate_checks = int(getattr(self, "_max_gate_checks", self._max_rounds))
        gate_limit_stop_reason = getattr(
            self,
            "_gate_limit_stop_reason",
            "max_rounds" if max_gate_checks == self._max_rounds else "max_gate_checks",
        )
        if continue_on_duplicate_patch and not submitted_patch.strip():
            result = rca.shell(
                self.env,
                "cd /testbed && git -c core.fileMode=false diff HEAD --binary",
            )
            if result.get("returncode") == 0:
                submitted_patch = str(result.get("output") or "")
        if (
            continue_on_duplicate_patch
            and self.rounds
            and submitted_patch.rstrip("\n") == str(self.rounds[-1]["patch"] or "").rstrip("\n")
        ):
            self.duplicate_submissions = getattr(self, "duplicate_submissions", 0) + 1
            self._round_start_calls = self.n_calls
            self.add_messages(self.model.format_message(role="user", content=DUPLICATE_PATCH_FOLLOWUP))
            return {}

        outcome = self._on_submit(submitted_patch)
        if len(outcome) == 3:
            passed, feedback, round_diff = outcome
            gate_metadata = {}
        else:
            passed, feedback, round_diff, gate_metadata = outcome
        round_no = len(self.rounds) + 1
        prev_diff = self.rounds[-1]["patch"] if self.rounds else None
        round_record = {
            "round": round_no,
            "turns": self.n_calls - self._round_start_calls,
            "passed": bool(passed),
            "patch": round_diff,
            "forced_submission": forced_submission,
        }
        if gate_metadata:
            round_record["gate_metadata"] = gate_metadata
        self.rounds.append(round_record)
        self.final_test_result = passed
        no_progress = not passed and round_no > 1 and round_diff == prev_diff
        if passed:
            self.stop_reason = "passed"
        elif round_no >= max_gate_checks:
            self.stop_reason = gate_limit_stop_reason
        elif no_progress and not continue_on_duplicate_patch:
            self.stop_reason = "no_progress"
        if passed or round_no >= max_gate_checks or (
            no_progress and not continue_on_duplicate_patch
        ):
            self.passed = bool(passed)
            raise Submitted({
                "role": "exit",
                "content": "",
                "extra": {"exit_status": "Submitted", "submission": round_diff or ""},
            })
        self._round_start_calls = self.n_calls
        self.add_messages(self.model.format_message(
            role="user", content=SEED_REPAIR_FOLLOWUP.format(feedback=(feedback or "")[:6000]),
        ))
        return {}

    def step(self) -> dict:
        if (
            self._force_submit_on_turn_limit
            and self.n_calls - self._round_start_calls >= self._steps_per_round
        ):
            result = rca.shell(
                self.env,
                "cd /testbed && git -c core.fileMode=false diff HEAD --binary",
            )
            forced_patch = str(result.get("output") or "") if result.get("returncode") == 0 else ""
            return self._handle_submission(forced_patch, forced_submission=True)
        try:
            return super().step()
        except Submitted as exc:
            return self._handle_submission(
                _submission_from_interrupt(exc),
                forced_submission=False,
            )

    def run_seeded(self, repair_prompt: str, *, system_prompt: str | None = None) -> dict:
        """Like DefaultAgent.run() but keeps ONE conversation across repair rounds instead of
        resetting to system+instance each round. With a non-empty seed the original solve
        trajectory is replayed as history; with an empty seed a fresh system message is prepended
        so the first round starts from the repair prompt alone."""
        self.messages = list(self._seed)
        if not self.messages and system_prompt is not None:
            self.add_messages(self.model.format_message(role="system", content=system_prompt))
        self.add_messages(self.model.format_message(role="user", content=repair_prompt))
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*getattr(e, "messages", []))
                # FormatError carries a corrective user message and is recoverable; keep
                # the same seeded conversation alive so the model can retry. Submitted
                # and LimitsExceeded carry an exit message and end the persistent run.
                if self.messages and self.messages[-1].get("role") == "exit":
                    break
            except Exception as e:  # noqa: BLE001
                self.stop_reason = self.stop_reason or f"agent_error: {type(e).__name__}: {str(e)[:120]}"
                break
            finally:
                if self.config.output_path:
                    self.save(self.config.output_path)
        return {"passed": self.passed, "stop_reason": self.stop_reason}


def process_one(iid, instance, patch_candidates, fail_nodes, cfg, args, source_messages=None):
    rec = {"instance_id": iid}
    env = None
    official_gate_session = None
    rng = random.Random(f"{args.seed}:{iid}")
    trace_dir = args.trajectory_dir / iid
    trace_dir.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(patch_candidates, str):
            patch_candidates = [patch_candidates]
        patch_candidates = [p for p in (patch_candidates or []) if (p or "").strip()]
        rec["n_patch_candidates"] = len(patch_candidates)

        all_f2p = _as_list(instance.get("FAIL_TO_PASS"))
        if not all_f2p:
            return {**rec, "skip": "no_f2p"}
        # prefer f2p nodes that INITIALLY FAIL on the wrong patch (meaningful repair targets),
        # then top up with the rest. Pick up to args.n_f2p of them, seeded by instance.
        n_want = max(1, int(getattr(args, "n_f2p", 1)))
        failing = [n for n in _as_list(fail_nodes) if n in all_f2p]
        rng.shuffle(failing)
        rest = [n for n in all_f2p if n not in set(failing)]
        rng.shuffle(rest)
        ordered = failing + rest
        nodes = ordered[:n_want]
        node = nodes if len(nodes) > 1 else nodes[0]
        rec["f2p_nodes"] = nodes
        rec["n_f2p_used"] = len(nodes)
        rec["n_f2p"] = len(all_f2p)
        rec["nodes_from_failing"] = sum(1 for n in nodes if n in (fail_nodes or []))
        test_patch = instance.get("test_patch", "")

        fresh_solve_round0 = bool(getattr(args, "fresh_solve_round0", False))
        rec["round0_mode"] = "fresh_e2e_solve" if fresh_solve_round0 else "provided_patch"
        cfgi = copy.deepcopy(cfg)
        if instance.get("_swebench_pro"):
            cfgi.setdefault("environment", {})["cwd"] = "/app"
        rca.swerebench_runner._resolve_per_instance_api_base(cfgi, iid)
        if instance.get("_swebench_pro"):
            env = swebench_pro_verify.create_sandbox_with_retry(
                lambda: rca.swerebench_runner.get_sb_environment(cfgi, instance),
                iid,
            )
        else:
            env = rca.swerebench_runner.get_sb_environment(cfgi, instance)
        if bool(getattr(args, "full_eval_script_gate", True)):
            official_gate_session = official_f2p_gate.PersistentOfficialGate(
                instance,
                timeout=max(args.test_timeout, args.install_timeout),
            )
            rec["persistent_in_loop_sandbox"] = True
        if instance.get("_swebench_pro"):
            setup = rca.shell(
                env,
                "test -e /testbed || ln -s /app /testbed; "
                "test \"$(readlink -f /testbed)\" = /app",
            )
            if (setup.get("returncode") or 0) != 0:
                raise RuntimeError(f"failed to expose SWE-bench Pro /app as /testbed: {setup}")
        # Both modes begin from the canonical e2e sample environment. In fresh mode
        # Codex produces the seed trajectory and patch itself before the first F2P.
        rca.shell(env, "cd /testbed && git checkout -- . && git clean -fdq")
        restore_env(env, instance, args.install_timeout)
        fresh_seed_messages = []
        round0_submitted_patch = None
        round0_turns = 0
        if fresh_solve_round0:
            model_cfg = cfgi.get("model", {})
            solve_agent_config = copy.deepcopy(cfgi.get("agent", {}))
            solve_agent_config["step_limit"] = int(getattr(args, "round0_steps", args.steps))
            solve_agent_config["cost_limit"] = float(getattr(args, "round0_cost_limit", 5.0))
            solve_agent_config["output_path"] = trace_dir / "round_00_solve.traj.json"
            solve_agent = DefaultAgent(rca.get_model(config=model_cfg), env, **solve_agent_config)
            solve_result = solve_agent.run(instance["problem_statement"])
            round0_turns = solve_agent.n_calls
            seed_format = "responses" if model_cfg.get("model_class") == "trapi_response" else "chat"
            fresh_seed_messages = live_seed_messages(solve_agent.messages, seed_format=seed_format)
            rec["round0_exit_status"] = solve_result.get("exit_status", "")
            rec["round0_submission_chars"] = len((solve_result.get("submission") or "").strip())
            rec["round0_turns"] = round0_turns
            rec["round0_trajectory"] = str(solve_agent_config["output_path"])
            rec["seed_msg_count"] = len(fresh_seed_messages)
            rec["seed_origin"] = "fresh_round0_e2e_trajectory"
            rec["wrong_patch_applied"] = None
            rec["wrong_patch_apply_errors"] = []
            if solve_result.get("exit_status") == "Submitted":
                round0_submitted_patch = str(solve_result.get("submission") or "")
        else:
            applied = False
            apply_errors = []
            restore_mode = str(getattr(args, "restore_mode", "patch"))
            rec["workspace_restore_mode"] = restore_mode
            if restore_mode == "replay":
                if not source_messages:
                    apply_err = "source trajectory has no replayable message history"
                    replay_stats = {}
                else:
                    applied, apply_err, replay_stats = rca.restore_state(
                        env,
                        source_messages,
                        patch_candidates[0],
                        args,
                        strict_replay=True,
                    )
                rec["workspace_replay"] = replay_stats
                apply_errors.append({
                    "candidate": 0,
                    "chars": len(patch_candidates[0]),
                    "applied": bool(applied),
                    "error_tail": (apply_err or "")[-600:],
                })
                if applied:
                    rec["applied_patch_candidate"] = 0
            else:
                if not patch_candidates and bool(
                    getattr(args, "allow_empty_source_patch", False)
                ):
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
            if not applied:
                rec["error"] = (
                    "WORKSPACE_REPLAY_FAILED"
                    if restore_mode == "replay"
                    else "WRONG_PATCH_APPLY_FAILED"
                )
                rec["initial_pass"] = False
                rec["rescued"] = False
                rec["rounds_used"] = 0
                rec["turns_per_round"] = []
                rec["patches_per_round"] = []
                rec["total_turns"] = 0
                rec["final_tail"] = (
                    apply_errors[-1].get("error_tail", "") if apply_errors else "NO_PATCH_CANDIDATES"
                )
                rec["initial_patch"] = ""
                rec["final_patch"] = ""
                return rec
        initial_diff = rca.shell(env, "cd /testbed && git diff").get("output", "") or ""
        rec["initial_patch"] = initial_diff if rca._looks_like_diff(initial_diff) else ""
        if fresh_solve_round0:
            rec["round0_patch"] = rec["initial_patch"]
            rec["round0_patch_chars"] = len(rec["initial_patch"])

        passed, feedback, test_metadata = f2p_check(
            env,
            instance,
            node,
            args.test_timeout,
            trace_dir / "initial.test.log",
            submitted_patch=round0_submitted_patch,
            full_eval_script_gate=bool(getattr(args, "full_eval_script_gate", True)),
            official_gate_session=official_gate_session,
        )
        rec["initial_pass"] = passed
        rec["initial_test"] = test_metadata
        rec["initial_gate_status"] = test_metadata.get("gate_status")
        rec["initial_gate_reason"] = test_metadata.get("gate_reason")
        rounds = 0
        turns_per_round = []
        patches_per_round = []
        if not passed:
            model_cfg = cfgi.get("model", {})
            use_seed = fresh_solve_round0 or bool(getattr(args, "seed_trajectory", False))
            if fresh_solve_round0:
                seed_messages = fresh_seed_messages
            elif use_seed:
                seed_format = "responses" if model_cfg.get("model_class") == "trapi_response" else "chat"
                if source_messages is not None:
                    seed_messages = live_seed_messages(source_messages, seed_format=seed_format)
                else:
                    tp = args.source_run / iid / f"{iid}.traj.json"
                    if not tp.exists():
                        tp = args.source_run / f"{iid}.traj.json"
                    seed_messages = load_seed_messages(tp, seed_format=seed_format)
            else:
                seed_messages = []
            rec["seed_msg_count"] = len(seed_messages)
            force_submit_on_turn_limit = bool(getattr(args, "force_submit_on_turn_limit", True))
            rec["force_submit_on_turn_limit"] = force_submit_on_turn_limit
            configured_max_gate_checks = int(getattr(args, "max_gate_checks", 0) or 0)
            max_gate_checks = configured_max_gate_checks or args.max_rounds
            continue_on_duplicate_patch = bool(getattr(args, "continue_on_duplicate_patch", False))
            rec["max_gate_checks"] = max_gate_checks
            rec["continue_on_duplicate_patch"] = continue_on_duplicate_patch
            state = {"test_metadata": test_metadata, "feedback": feedback}

            def _on_submit(submitted_patch):
                p, fb, meta = f2p_check(
                    env,
                    instance,
                    node,
                    args.test_timeout,
                    submitted_patch=submitted_patch,
                    full_eval_script_gate=bool(getattr(args, "full_eval_script_gate", True)),
                    official_gate_session=official_gate_session,
                )
                state["test_metadata"] = meta
                state["feedback"] = fb
                return p, fb, submitted_patch, meta

            agent = SeedRepairAgent(
                rca.get_model(config=model_cfg),
                env,
                seed_messages=seed_messages,
                on_submit=_on_submit,
                max_rounds=args.max_rounds,
                steps_per_round=args.steps,
                max_gate_checks=configured_max_gate_checks or None,
                continue_on_duplicate_patch=continue_on_duplicate_patch,
                force_submit_on_turn_limit=force_submit_on_turn_limit,
                system_template=SYSTEM,
                instance_template=SEED_REPAIR_TEMPLATE,
                step_limit=args.steps * args.max_rounds,
                cost_limit=5.0 * args.max_rounds,
                output_path=trace_dir / "seed_repair.traj.json",
            )
            if use_seed:
                # solve trajectory already supplies the issue context; continue with the repair prompt
                agent.run_seeded(SEED_REPAIR_TEMPLATE.format(feedback=feedback[:6000]))
            else:
                # no seed history: first round must carry the issue + node + failing output itself
                node_str = "\n".join(node) if isinstance(node, (list, tuple)) else node
                first_prompt = (
                    REPAIR_TEMPLATE
                    .replace("{{task}}", instance["problem_statement"])
                    .replace("{{node}}", node_str)
                    .replace("{{feedback}}", feedback[:6000])
                )
                agent.run_seeded(first_prompt, system_prompt=SYSTEM)
            passed = bool(agent.passed)
            rounds = len(agent.rounds)
            turns_per_round = [r["turns"] for r in agent.rounds]
            patches_per_round = [
                {
                    "round": r["round"],
                    "turns": r["turns"],
                    "passed": r["passed"],
                    "patch": r["patch"],
                    "forced_submission": r.get("forced_submission", False),
                    "gate_metadata": r.get("gate_metadata", {}),
                }
                for r in agent.rounds
            ]
            test_metadata = state["test_metadata"]
            feedback = state["feedback"]
            rec["stop_reason"] = agent.stop_reason
            rec["duplicate_submissions"] = getattr(agent, "duplicate_submissions", 0)
        rec["gate_status"] = test_metadata.get("gate_status")
        rec["gate_reason"] = test_metadata.get("gate_reason")
        rec["rescued"] = bool(passed)
        rec["rounds_used"] = rounds
        rec["turns_per_round"] = turns_per_round
        rec["patches_per_round"] = patches_per_round
        rec["repair_turns"] = agent.n_calls if not passed or rounds else 0
        rec["total_turns"] = round0_turns + rec["repair_turns"]
        rec["final_feedback"] = feedback
        rec["final_tail"] = feedback[-400:]
        # capture the FULL repaired source diff (wrong patch + repair edits) for oracle verify
        diff = rca.shell(env, "cd /testbed && git diff").get("output", "") or ""
        if patches_per_round:
            final_patch = patches_per_round[-1]["patch"]
            rec["final_patch_from_round"] = patches_per_round[-1]["round"]
        elif round0_submitted_patch is not None:
            final_patch = round0_submitted_patch
            rec["final_patch_from_round"] = 0
        else:
            final_patch = diff if rca._looks_like_diff(diff) else ""
        rec["final_patch"] = final_patch
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        try:
            if official_gate_session is not None:
                official_gate_session.close()
        except Exception:
            pass
        try:
            if env is not None:
                env.cleanup()
        except Exception:
            pass
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, type=Path)
    ap.add_argument("--source-run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--subset", default="rebench")
    ap.add_argument("--split", default="filtered")
    ap.add_argument("--api-base", default="http://127.0.0.1:{port}/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--sandbox-url", default=os.getenv("SANDBOX_BASE_URL", ""))
    ap.add_argument("--sandbox-key", default="")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument(
        "--max-gate-checks",
        type=int,
        default=0,
        help=(
            "Maximum non-duplicate F2P gate evaluations during repair; 0 preserves the "
            "legacy --max-rounds limit without increasing the total model-turn budget."
        ),
    )
    ap.add_argument(
        "--continue-on-duplicate-patch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the conversation alive when a submission is empty or identical to the last "
            "failed candidate, without rerunning or charging the gate."
        ),
    )
    ap.add_argument("--n-f2p", type=int, default=1,
                    help="how many gold FAIL_TO_PASS nodes to use together as the in-loop oracle "
                         "(setting 1 -> 1, setting 2 -> 3). ALL chosen nodes must pass to rescue.")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument(
        "--force-submit-on-turn-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Submit the current worktree diff to the F2P gate when a repair round reaches --steps "
            "(default: enabled; use --no-force-submit-on-turn-limit for legacy LimitsExceeded behavior)."
        ),
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--test-timeout", type=int, default=240)
    ap.add_argument("--install-timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trajectory-dir", type=Path, default=None,
                    help="directory for per-instance round trajectories and raw test logs; "
                         "defaults beside --out")
    ap.add_argument("--model", default=None, help="override model_name")
    ap.add_argument("--model-class", default=None, help="override model_class")
    ap.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Responses API reasoning effort for trapi_response models.",
    )
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Set model.model_kwargs.chat_template_kwargs.enable_thinking=false for Qwen-style tool calling.")
    ap.add_argument("--seed-trajectory", action="store_true",
                    help="Reproduce the repair-SFT format: seed the persistent repair agent with the "
                         "original solve trajectory (from --source-run) before the first repair prompt. "
                         "Without this flag the agent starts from a fresh system+REPAIR_TEMPLATE message; "
                         "either way rounds share ONE conversation and the sandbox state carries over.")
    ap.add_argument(
        "--restore-mode",
        choices=["patch", "replay"],
        default="patch",
        help=(
            "patch = apply only the prior submitted diff (default); replay = rerun every "
            "workspace action from the source trajectory, require the replay to complete, and "
            "verify that the original submitted patch is present before repair starts"
        ),
    )
    ap.add_argument(
        "--replay-cmd-timeout",
        type=int,
        default=120,
        help="per-command timeout in seconds while restoring a source workspace",
    )
    ap.add_argument(
        "--replay-budget",
        type=int,
        default=600,
        help="per-instance wall-clock budget in seconds for source trajectory replay",
    )
    ap.add_argument(
        "--full-eval-script-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse one reset-before-grade sandbox for every focused official eval-script check "
            "(default: enabled; use --no-full-eval-script-gate for the legacy selector path)."
        ),
    )
    args = ap.parse_args()
    if not args.sandbox_url:
        ap.error("--sandbox-url or SANDBOX_BASE_URL must be set explicitly")
    if args.steps <= 0:
        ap.error("--steps must be positive")
    if args.max_rounds <= 0:
        ap.error("--max-rounds must be positive")
    if args.max_gate_checks < 0:
        ap.error("--max-gate-checks must be non-negative")
    if args.trajectory_dir is None:
        args.trajectory_dir = args.out.parent / f"{args.out.stem}.round_trajs"

    ids = json.loads(args.ids.read_text())
    if args.limit:
        ids = ids[: args.limit]
    ds = {r["instance_id"]: dict(r) for r in load_dataset(rca.DATASET_MAPPING.get(args.subset, args.subset), split=args.split)}

    # map instance -> list of f2p that FAILED on the wrong patch (from the source-run verify)
    fail_map = {}
    vpath = args.source_run / "verify_results_azure_modal.json"
    if vpath.exists():
        vr = json.loads(vpath.read_text())
        for iid, rec in vr.items():
            s = (rec.get("samples") or [{}])[0]
            fail_map[iid] = s.get("fail_to_pass_failed", []) or []
        print(f"loaded failing-f2p map for {len(fail_map)} instances")

    todo = {}
    for iid in ids:
        tp = args.source_run / iid / f"{iid}.traj.json"
        if not tp.exists():
            tp = args.source_run / f"{iid}.traj.json"
        if not tp.exists() or iid not in ds:
            continue
        d = json.loads(tp.read_text())
        patches = patch_candidates_from_traj(d)
        if patches:
            todo[iid] = (patches, d.get("messages", []) or [])

    existing = set()
    if args.out.exists():
        for l in args.out.read_text().splitlines():
            if l.strip():
                try:
                    existing.add(json.loads(l)["instance_id"])
                except Exception:
                    pass
    work = [(i, patches, messages) for i, (patches, messages) in todo.items() if i not in existing]
    print(f"ids={len(ids)} with_patch={len(todo)} todo={len(work)} existing={len(existing)} "
          f"max_rounds={args.max_rounds} steps={args.steps} "
          f"max_gate_checks={args.max_gate_checks or args.max_rounds} "
          f"continue_on_duplicate_patch={args.continue_on_duplicate_patch} "
          f"force_submit_on_turn_limit={args.force_submit_on_turn_limit} workers={args.workers}")

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
    with args.out.open("a", buffering=1) as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(
                    process_one,
                    iid,
                    ds[iid],
                    patches,
                    fail_map.get(iid, []),
                    cfg,
                    args,
                    messages,
                ): iid
                for iid, patches, messages in work
            }
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result()
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                done += 1
                resc += int(bool(r.get("rescued")))
                init += int(bool(r.get("initial_pass")))
                if done == 1 or done % 10 == 0:
                    print(f"done={done}/{len(work)} rescued={resc} (initial_pass={init}) last={r.get('instance_id')} "
                          f"rescued={r.get('rescued')} rounds={r.get('rounds_used')}")
    print(f"TOTAL done={done} rescued={resc} initial_pass={init} rescue_rate={resc/max(1,done):.3f}")


if __name__ == "__main__":
    main()
