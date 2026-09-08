#!/usr/bin/env python3
"""Inspect self-test command/output patterns in mini-swe-agent trajectories."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

STRONG_TEST_RE = re.compile(
    r"(^|[;&|\s])((python\s+[^\n;&|]*(pytest|unittest|runtests|manage\.py\s+test))|"
    r"(python\s+-m\s+(pytest|unittest))|"
    r"(pytest|py\.test|tox|nox|mvn\s+test|gradle\s+test|npm\s+test|yarn\s+test|pnpm\s+test|"
    r"go\s+test|cargo\s+test|bundle\s+exec\s+rspec|rspec|rake\s+test|"
    r"./runtests\.py|runtests\.py|manage\.py\s+test))\b",
    re.IGNORECASE,
)
WEAK_SCRIPT_RE = re.compile(
    r"(^|[;&|\s])((python\d*(\.\d+)?\s+(-u\s+)?([^\s;&|]*/)?[^/\s;&|]*(test|repro|check|verify|bug|issue|regression)[^/\s;&|]*\.py)|"
    r"(bash\s+([^\s;&|]*/)?[^/\s;&|]*(test|repro|check|verify|bug|issue|regression)[^/\s;&|]*\.sh)|"
    r"(sh\s+([^\s;&|]*/)?[^/\s;&|]*(test|repro|check|verify|bug|issue|regression)[^/\s;&|]*\.sh)|"
    r"(\./[^/\s;&|]*(test|repro|check|verify|bug|issue|regression)[^/\s;&|]*))",
    re.IGNORECASE,
)
INLINE_PYTHON_RE = re.compile(r"(^|[;&|\s])python\d*(\.\d+)?\s+(-c|-\s*<<)", re.IGNORECASE)
INLINE_SELF_TEST_SIGNAL_RE = re.compile(
    r"\b(assert|AssertionError|unittest|pytest|django\.setup\(|call_command\(['\"]test|print\(|raise SystemExit)\b",
    re.IGNORECASE,
)
SUBMIT_RE = re.compile(r"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|patch\.txt|git\s+diff", re.IGNORECASE)
PASS_RE = re.compile(
    r"(\bOK\b|\bPASSED\b|\bpassed\b|\bSUCCESS\b|\bSUCCESSFUL\b|"
    r"all\s+tests?\s+pass|no\s+failures?|"
    r"=+\s*\d+\s+passed|\d+\s+passed\s+in\s+|"
    r"Ran\s+\d+\s+tests?\s+in\s+[\d.]+s\s*\n\s*OK)",
    re.IGNORECASE,
)
FAIL_RE = re.compile(
    r"(\bFAILED\b|\bFAILURES?\b|\bERRORS?\b|\bfailed\b|\berror\b|"
    r"=+\s*.*\bfailed\b|\d+\s+failed|\d+\s+errors?|"
    r"Traceback \(most recent call last\)|AssertionError|ImportError|ModuleNotFoundError|"
    r"Expected .* but got|does not match|mismatch)",
    re.IGNORECASE,
)
RC_RE = re.compile(r"<returncode>(-?\d+)</returncode>")


def as_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def assistant_text(message: dict) -> str:
    parts = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            parts.append(args)
        elif args is not None:
            parts.append(json.dumps(args, ensure_ascii=False))
    return "\n".join(parts)


def command_kind(command: str) -> str | None:
    if STRONG_TEST_RE.search(command):
        return "strong"
    if WEAK_SCRIPT_RE.search(command):
        return "weak"
    if INLINE_PYTHON_RE.search(command) and INLINE_SELF_TEST_SIGNAL_RE.search(command):
        return "weak"
    return None


def status_from_output(output: str, kind: str) -> tuple[str, str]:
    rc_match = RC_RE.search(output)
    rc = int(rc_match.group(1)) if rc_match else None
    fail = bool(FAIL_RE.search(output))
    explicit_pass = bool(PASS_RE.search(output))
    if rc is not None and rc != 0:
        return "fail", "nonzero_rc"
    if fail:
        return "fail", "fail_keyword"
    if explicit_pass:
        return "pass", "pass_keyword"
    if rc == 0 and kind == "strong":
        return "pass", "strong_zero_rc"
    if rc == 0 and kind == "weak":
        return "unknown", "weak_zero_rc_no_pass_keyword"
    return "unknown", "no_signal"


def iter_test_events(data: dict):
    messages = data.get("messages", []) if isinstance(data, dict) else []
    pending: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            text = assistant_text(message)
            kind = command_kind(text)
            if kind:
                pending.append((kind, text))
        elif role == "tool" and pending:
            kind, command = pending.pop(0)
            output = as_text(message.get("content", ""))
            status, reason = status_from_output(output, kind)
            yield {
                "kind": kind,
                "command": command,
                "status": status,
                "reason": reason,
                "output": output,
            }


def classify(events: list[dict]) -> str:
    if not events:
        return "no_test"
    parseable = [event for event in events if event["status"] != "unknown"]
    if not parseable:
        return "test_unknown"
    return "pass_self_test" if parseable[-1]["status"] == "pass" else "fail_self_test"


def classify_last_event(events: list[dict]) -> str:
    if not events:
        return "no_test"
    status = events[-1]["status"]
    if status == "pass":
        return "pass_self_test"
    if status == "fail":
        return "fail_self_test"
    return "test_unknown"


def short(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    counts = Counter()
    last_event_counts = Counter()
    last_event_submitted_counts = Counter()
    by_kind = Counter()
    by_reason = Counter()
    command_patterns = Counter()
    examples = defaultdict(list)
    last_unknown_examples = []
    no_test_examples = []
    submitted_counts = Counter()

    for traj in sorted(args.run_dir.rglob("*.traj.json")):
        data = json.loads(traj.read_text())
        iid = data.get("instance_id") or traj.parent.name
        info = data.get("info", {}) if isinstance(data, dict) else {}
        submission_nonempty = bool((info.get("submission") or "").strip())
        events = list(iter_test_events(data))
        category = classify(events)
        last_event_category = classify_last_event(events)
        counts[category] += 1
        last_event_counts[last_event_category] += 1
        last_event_submitted_counts[(last_event_category, submission_nonempty)] += 1
        submitted_counts[(category, submission_nonempty)] += 1
        for event in events:
            by_kind[(event["kind"], event["status"])] += 1
            by_reason[(event["kind"], event["reason"])] += 1
            cmd = short(event["command"], 120)
            # Normalize instance-specific paths a little.
            cmd = re.sub(r"/tmp/[^\s]+", "/tmp/…", cmd)
            command_patterns[(event["kind"], cmd)] += 1
        if len(examples[category]) < args.examples:
            examples[category].append(iid)
        if events and events[-1]["status"] == "unknown" and len(last_unknown_examples) < args.examples:
            last_unknown_examples.append((iid, events[-1]["kind"], events[-1]["reason"], short(events[-1]["command"]), short(events[-1]["output"])))
        if not events and len(no_test_examples) < args.examples:
            no_test_examples.append(iid)

    total = sum(counts.values()) or 1
    print(f"run_dir={args.run_dir}")
    print(f"trajectories={total}")
    for key in ["pass_self_test", "fail_self_test", "test_unknown", "no_test", "parse_error"]:
        count = counts.get(key, 0)
        print(f"{key}\t{count}\t{count/total*100:.1f}%")
        if examples.get(key):
            print("  examples=" + ",".join(examples[key]))
    print("\nlast_event_trajectory_counts")
    for key in ["pass_self_test", "fail_self_test", "test_unknown", "no_test", "parse_error"]:
        count = last_event_counts.get(key, 0)
        print(f"{key}\t{count}\t{count/total*100:.1f}%")
    print("\nlast_event_submitted_by_category")
    for (category, nonempty), count in sorted(last_event_submitted_counts.items()):
        print(f"{category}\tnonempty_patch={nonempty}\t{count}")
    print("\nsubmitted_by_category")
    for (category, nonempty), count in sorted(submitted_counts.items()):
        print(f"{category}\tnonempty_patch={nonempty}\t{count}")
    print("\nevents_by_kind_status")
    for key, count in by_kind.most_common():
        print(f"{key}\t{count}")
    print("\nevents_by_kind_reason")
    for key, count in by_reason.most_common():
        print(f"{key}\t{count}")
    print("\ntop_command_patterns")
    for (kind, cmd), count in command_patterns.most_common(30):
        print(f"{count}\t{kind}\t{cmd}")
    if last_unknown_examples:
        print("\nlast_unknown_examples")
        for row in last_unknown_examples:
            print("\t".join(row))
    if no_test_examples:
        print("\nno_test_examples=" + ",".join(no_test_examples))


if __name__ == "__main__":
    main()
