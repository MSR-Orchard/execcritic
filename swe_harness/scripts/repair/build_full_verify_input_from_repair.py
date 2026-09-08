#!/usr/bin/env python3
"""Build a full verification input by overlaying repair patches on source trajectories.

The repair JSONL usually only contains instances that had generated tests. For an
end-to-end score, keep every source trajectory and replace only the instances that
have a non-empty repaired final_patch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def _msg_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return ""


def recover_patch_from_traj(data: dict[str, Any]) -> str:
    best = ""
    for message in data.get("messages", []) or []:
        text = _msg_text(message)
        if "diff --git" not in text:
            continue
        candidate = text[text.find("diff --git") :]
        candidate = re.split(r"\n(?:COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|\$ |\(testbed\)|root@)", candidate)[0]
        if ("@@" in candidate or ("+++ " in candidate and "--- " in candidate)) and len(candidate) > len(best):
            best = candidate
    return best.strip()


def read_submission(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    patch = ((data.get("info") or {}).get("submission") or "").strip()
    return patch or recover_patch_from_traj(data)


def source_trajectories(source_run: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(source_run.glob("*.traj.json")):
        found[path.name[: -len(".traj.json")]] = path
    for path in sorted(source_run.glob("*/*.traj.json")):
        found[path.name[: -len(".traj.json")]] = path
    return found


def read_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        ids = []
        for line in text.splitlines():
            row = json.loads(line)
            if isinstance(row, str):
                ids.append(row)
            elif isinstance(row, dict):
                iid = row.get("instance_id") or row.get("id")
                if iid:
                    ids.append(str(iid))
        return ids
    data = json.loads(text)
    if isinstance(data, dict):
        values = data.get("instance_ids") or data.get("ids") or data.keys()
        return [str(x) for x in values]
    return [str(x) for x in data]


def repair_patches(repair_jsonl: Path | None) -> tuple[dict[str, str], int, int]:
    if repair_jsonl is None or not repair_jsonl.exists():
        return {}, 0, 0
    patches: dict[str, str] = {}
    rows = empty = 0
    for line in repair_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        row = json.loads(line)
        iid = row.get("instance_id")
        patch = (row.get("final_patch") or "").strip()
        if not iid or not patch:
            empty += 1
            continue
        patches[str(iid)] = patch
    return patches, rows, empty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--repair-jsonl", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ids", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    sources = source_trajectories(args.source_run)
    wanted = read_ids(args.ids)
    if wanted is None:
        wanted = sorted(sources)

    repaired, repair_rows, repair_empty = repair_patches(args.repair_jsonl)
    args.out.mkdir(parents=True, exist_ok=True)

    wrote = used_repair = used_source = missing_source = empty_source = 0
    missing_ids: list[str] = []
    empty_source_ids: list[str] = []
    for iid in wanted:
        source_path = sources.get(iid)
        if source_path is None:
            missing_source += 1
            missing_ids.append(iid)
            continue
        if iid in repaired:
            patch = repaired[iid]
            used_repair += 1
        else:
            patch = read_submission(source_path).strip()
            used_source += 1
            if not patch:
                empty_source += 1
                empty_source_ids.append(iid)
        out_path = args.out / f"{iid}.traj.json"
        out_path.write_text(json.dumps({"info": {"submission": patch}}, ensure_ascii=False) + "\n", encoding="utf-8")
        wrote += 1

    summary = {
        "source_run": str(args.source_run),
        "repair_jsonl": str(args.repair_jsonl) if args.repair_jsonl else None,
        "out": str(args.out),
        "source_traj_count": len(sources),
        "requested_ids": len(wanted),
        "repair_rows": repair_rows,
        "repair_patches": len(repaired),
        "repair_empty_patch_rows": repair_empty,
        "wrote_traj": wrote,
        "used_repair_patch": used_repair,
        "used_source_patch": used_source,
        "missing_source": missing_source,
        "empty_source_patch": empty_source,
        "missing_source_ids": missing_ids[:50],
        "empty_source_patch_ids": empty_source_ids[:50],
    }
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
