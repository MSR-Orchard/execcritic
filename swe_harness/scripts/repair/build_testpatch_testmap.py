#!/usr/bin/env python3
"""Export resolve-style generated-test results for ``selfrepair_gentest.py``.

The output schema is the canonical test-patch form::

    {"instance_id": [{"test_patch": "...", "test_command": "..."}]}

Gold labels are reported in the sidecar summary for audit purposes, but never
copied into the map or used for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _canonical_entry(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if row.get("submitted") is not True:
        return None, "not_submitted"
    patch = str(row.get("test_patch") or "")
    command = str(row.get("test_command") or "").strip()
    if not patch.strip():
        return None, "empty_patch"
    if not command:
        return None, "empty_command"
    patch = patch.rstrip("\n") + "\n"
    if not patch.startswith("diff --git "):
        instance_id = row.get("instance_id")
        raise ValueError(f"submitted test_patch is not a git unified diff: {instance_id}")
    if "\n" in command:
        instance_id = row.get("instance_id")
        raise ValueError(f"test_command must be exactly one line: {instance_id}")
    return {
        "test_patch": patch,
        "test_command": command,
        "source_trajectory": row.get("trajectory_path"),
        "source_exit_status": row.get("exit_status"),
    }, "selected"


def build_test_map(
    rows: list[dict[str, Any]],
    *,
    max_tests_per_instance: int = 1,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if max_tests_per_instance < 1:
        raise ValueError("max_tests_per_instance must be positive")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    seen_run_keys: set[str] = set()
    duplicate_tests = 0

    for row_number, row in enumerate(rows, start=1):
        instance_id = str(row.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError(f"result row {row_number} has no instance_id")
        run_key = str(row.get("run_key") or instance_id)
        if run_key in seen_run_keys:
            raise ValueError(f"duplicate run_key in results: {run_key}")
        seen_run_keys.add(run_key)

        entry, disposition = _canonical_entry(row)
        if entry is None:
            skipped[disposition] += 1
            continue

        fingerprint = hashlib.sha256(
            (entry["test_patch"] + "\0" + entry["test_command"]).encode("utf-8", "replace")
        ).hexdigest()
        if any(candidate["_fingerprint"] == fingerprint for candidate in grouped[instance_id]):
            duplicate_tests += 1
            continue
        grouped[instance_id].append({**entry, "_fingerprint": fingerprint})

    test_map: dict[str, list[dict[str, Any]]] = {}
    truncated_tests = 0
    for instance_id, entries in grouped.items():
        truncated_tests += max(0, len(entries) - max_tests_per_instance)
        test_map[instance_id] = [
            {key: value for key, value in entry.items() if key != "_fingerprint"}
            for entry in entries[:max_tests_per_instance]
        ]

    summary = {
        "result_rows": len(rows),
        "unique_run_keys": len(seen_run_keys),
        "unique_instance_ids": len({str(row.get("instance_id") or "") for row in rows}),
        "submitted": sum(row.get("submitted") is True for row in rows),
        "base_clean_fail": sum(row.get("base_clean_fail") is True for row in rows),
        "gold_pass_audit_only": sum(row.get("gold_pass") is True for row in rows),
        "base_fail_gold_pass_audit_only": sum(
            row.get("base_fail_gold_pass") is True for row in rows
        ),
        "errors": sum(bool(row.get("error")) for row in rows),
        "test_map_ids": len(test_map),
        "test_entries": sum(len(entries) for entries in test_map.values()),
        "max_tests_per_instance": max_tests_per_instance,
        "duplicate_tests_dropped": duplicate_tests,
        "tests_truncated_by_cap": truncated_tests,
        "selection": "submitted rows with nonempty canonical test_patch and exact test_command",
        "gold_labels_used_for_selection": False,
        "source_labels_in_test_map": False,
        "skipped": dict(sorted(skipped.items())),
    }
    return test_map, summary


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path, help="batch_resolve.py results.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="self-repair test-map JSON")
    parser.add_argument("--summary", type=Path, help="audit sidecar; defaults next to --out")
    parser.add_argument("--max-tests-per-instance", type=int, default=1)
    args = parser.parse_args()

    rows = read_jsonl(args.results)
    test_map, summary = build_test_map(
        rows,
        max_tests_per_instance=args.max_tests_per_instance,
    )
    summary["source_results"] = str(args.results)
    write_json(args.out, test_map)
    write_json(args.summary or args.out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
