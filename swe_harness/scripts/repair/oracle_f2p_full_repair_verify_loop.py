#!/usr/bin/env python3
"""Run four independent oracle-F2P repairs per unresolved trajectory and verify each result."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from datasets import load_dataset


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def normalize_patch_for_apply(patch: str) -> str:
    """Preserve diff contents while ensuring git receives a final line break."""
    return patch.rstrip("\n") + "\n" if patch.strip() else ""


def output_text(content: str) -> str:
    match = re.search(r"<output>\n?(.*?)</output>", content, flags=re.S)
    return match.group(1) if match else content


def assistant_command(message: dict) -> str:
    commands = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, dict):
            command = arguments.get("command")
        else:
            command = str(arguments)
        if command:
            commands.append(str(command))
    return "\n".join(commands)


def validate_unified_diff(patch: str) -> tuple[bool, str, int]:
    lines = patch.splitlines()
    index = 0
    hunk_count = 0
    if not patch.startswith("diff --git "):
        return False, "missing diff --git at start", 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            index += 1
            continue
        if line.startswith(
            (
                "index ",
                "--- ",
                "+++ ",
                "new file mode ",
                "deleted file mode ",
                "old mode ",
                "new mode ",
                "similarity index ",
                "dissimilarity index ",
                "rename from ",
                "rename to ",
                "copy from ",
                "copy to ",
            )
        ):
            index += 1
            continue
        match = HUNK_RE.match(line)
        if not match:
            return False, f"unexpected line outside hunk at {index + 1}: {line[:80]!r}", hunk_count
        old_expected = int(match.group(2) or "1")
        new_expected = int(match.group(4) or "1")
        old_seen = 0
        new_seen = 0
        index += 1
        while index < len(lines):
            hunk_line = lines[index]
            if hunk_line.startswith("diff --git ") or HUNK_RE.match(hunk_line):
                break
            if hunk_line.startswith("\\ No newline"):
                pass
            elif hunk_line.startswith(" "):
                old_seen += 1
                new_seen += 1
            elif hunk_line.startswith("-"):
                old_seen += 1
            elif hunk_line.startswith("+"):
                new_seen += 1
            else:
                return False, f"invalid hunk line at {index + 1}: {hunk_line[:80]!r}", hunk_count
            index += 1
        if old_seen != old_expected or new_seen != new_expected:
            return (
                False,
                f"hunk count mismatch at line {index + 1}: old {old_seen}/{old_expected}, "
                f"new {new_seen}/{new_expected}",
                hunk_count,
            )
        hunk_count += 1
    if hunk_count == 0:
        return False, "no hunks found", 0
    return True, "", hunk_count


def extract_patch(row: dict) -> tuple[str, dict[str, Any]]:
    if "source_patch_override" in row:
        patch = normalize_patch_for_apply(str(row.get("source_patch_override") or ""))
        if not patch:
            return "", {
                "source": "source_patch_override",
                "patch_chars": 0,
                "valid": True,
                "invalid_reason": "",
                "hunk_count": 0,
            }
        valid, reason, hunk_count = validate_unified_diff(patch)
        return patch, {
            "source": "source_patch_override",
            "patch_chars": len(patch),
            "valid": valid,
            "invalid_reason": reason,
            "hunk_count": hunk_count,
        }

    submission = row.get("info", {}).get("submission", "")
    if isinstance(submission, str) and submission.strip():
        patch = normalize_patch_for_apply(submission)
        valid, reason, hunk_count = validate_unified_diff(patch)
        return patch, {
            "source": "info.submission",
            "patch_chars": len(patch),
            "valid": valid,
            "invalid_reason": reason,
            "hunk_count": hunk_count,
        }

    last_command = ""
    candidates = []
    for message_index, message in enumerate(row.get("messages") or []):
        role = message.get("role")
        if role == "assistant":
            command = assistant_command(message)
            if command:
                last_command = command
            continue
        if role != "tool":
            continue
        content = str(message.get("content") or "")
        if "diff --git" not in content or re.search(r"\bgit\s+show\b", last_command):
            continue
        patch = output_text(content)
        diff_start = patch.find("diff --git")
        if diff_start < 0:
            continue
        patch = patch[diff_start:]
        valid, reason, hunk_count = validate_unified_diff(patch)
        score = message_index
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in last_command:
            score += 2_000_000
        if "cat /testbed/patch.txt" in last_command:
            score += 1_000_000
        if "cat /tmp/patch.txt" in last_command or "cat patch.txt" in last_command:
            score += 900_000
        if re.search(r"\b(head|tail)\b", last_command):
            score -= 1_000_000
        if "sed -n" in last_command:
            score -= 500_000
        if patch.count("diff --git") > 1:
            score -= 10_000
        score += min(len(patch), 100_000)
        candidates.append(
            {
                "patch": patch,
                "message_index": message_index,
                "command": last_command,
                "patch_chars": len(patch),
                "valid": valid,
                "invalid_reason": reason,
                "hunk_count": hunk_count,
                "score": score if valid else -1,
            }
        )
    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    if not valid_candidates:
        best_invalid = max(candidates, key=lambda candidate: candidate["patch_chars"], default=None)
        return "", {
            "candidate_count": len(candidates),
            "invalid_reason": best_invalid.get("invalid_reason") if best_invalid else "no diff candidate",
        }
    best = max(valid_candidates, key=lambda candidate: candidate["score"])
    return best["patch"], {key: value for key, value in best.items() if key != "patch"}


def build_source_manifest(
    input_path: Path,
    manifest_path: Path,
    instances: dict[str, dict],
    *,
    max_source_rows: int = 0,
    rollouts_per_source: int = 4,
    allow_empty_source_patch: bool = False,
) -> dict:
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    counts = Counter()
    with input_path.open("rb") as source, temporary.open("w") as output:
        row_index = 0
        while True:
            offset = source.tell()
            raw_line = source.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue
            counts["input_rows"] += 1
            try:
                row = json.loads(raw_line)
            except Exception:
                counts["invalid_json"] += 1
                row_index += 1
                continue
            metadata = row.get("metadata") or {}
            instance_id = metadata.get("instance_id") or row.get("instance_id")
            if not instance_id or instance_id not in instances:
                counts["not_official_instance"] += 1
                row_index += 1
                continue
            if metadata.get("verify_status") not in (None, "unresolved", "unknown"):
                counts["not_unresolved"] += 1
                row_index += 1
                continue
            patch, patch_info = extract_patch(row)
            if not patch and not allow_empty_source_patch:
                counts["no_valid_patch"] += 1
                row_index += 1
                continue
            failed_f2p = metadata.get("fail_to_pass_failed") or []
            if not failed_f2p:
                counts["no_failed_f2p_hint"] += 1
            record = {
                "row_index": row_index,
                "byte_offset": offset,
                "byte_length": len(raw_line),
                "instance_id": str(instance_id),
                "source_sample_idx": int(metadata.get("sample_idx") or 0),
                "source": metadata.get("source"),
                "source_model": metadata.get("model"),
                "fail_to_pass_failed": failed_f2p,
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "patch_chars": len(patch),
                "patch_info": patch_info,
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            counts["eligible_source_rows"] += 1
            row_index += 1
            if max_source_rows and counts["eligible_source_rows"] >= max_source_rows:
                break
    temporary.replace(manifest_path)
    summary = {
        **counts,
        "eligible_source_rows": counts["eligible_source_rows"],
        "source_row_limit": max_source_rows,
        "selection": "first eligible source trajectory rows in input order",
        "rollouts_per_source": rollouts_per_source,
        "allow_empty_source_patch": allow_empty_source_patch,
        "total_attempt_units": counts["eligible_source_rows"] * rollouts_per_source,
        "input_path": str(input_path),
        "manifest_path": str(manifest_path),
    }
    atomic_json(manifest_path.with_suffix(".summary.json"), summary)
    return summary


def build_dataset_manifest(
    dataset_rows: list[dict],
    manifest_path: Path,
    *,
    max_source_rows: int = 0,
    rollouts_per_source: int = 1,
) -> dict:
    """Build one task-selection record per benchmark dataset instance."""
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    selected = dataset_rows[:max_source_rows] if max_source_rows else dataset_rows
    with temporary.open("w") as output:
        for row_index, instance in enumerate(selected):
            output.write(json.dumps({
                "row_index": row_index,
                "instance_id": str(instance["instance_id"]),
                "source_sample_idx": 0,
                "source": "benchmark_dataset",
                "source_model": None,
                "fail_to_pass_failed": [],
                "patch_sha256": "",
                "patch_chars": 0,
                "patch_info": {"selection_only": True},
            }, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    summary = {
        "input_rows": len(dataset_rows),
        "eligible_source_rows": len(selected),
        "source_row_limit": max_source_rows,
        "selection": "benchmark dataset instances in dataset order",
        "selection_mode": "dataset_instances",
        "rollouts_per_source": rollouts_per_source,
        "total_attempt_units": len(selected) * rollouts_per_source,
        "manifest_path": str(manifest_path),
    }
    atomic_json(manifest_path.with_suffix(".summary.json"), summary)
    return summary


def load_source_row(input_path: Path, source_record: dict) -> tuple[dict, str]:
    with input_path.open("rb") as handle:
        handle.seek(int(source_record["byte_offset"]))
        raw_line = handle.read(int(source_record["byte_length"]))
    row = json.loads(raw_line)
    patch, _ = extract_patch(row)
    digest = hashlib.sha256(patch.encode()).hexdigest()
    if digest != source_record["patch_sha256"]:
        raise RuntimeError(
            f"source patch mismatch for row {source_record['row_index']}: "
            f"expected {source_record['patch_sha256']}, got {digest}"
        )
    return row, patch


def task_key(source_record: dict, rollout_index: int) -> str:
    return f"row_{int(source_record['row_index']):06d}.rollout_{rollout_index}"


def task_dir(output_dir: Path, source_record: dict, rollout_index: int) -> Path:
    row_index = int(source_record["row_index"])
    return output_dir / "rows" / f"shard_{row_index // 1000:04d}" / f"row_{row_index:06d}" / f"rollout_{rollout_index}"


def compact_status(repair: dict, verify: dict | None, source_record: dict, rollout_index: int, elapsed: float) -> dict:
    return {
        "task_key": task_key(source_record, rollout_index),
        "row_index": source_record["row_index"],
        "instance_id": source_record["instance_id"],
        "source_sample_idx": source_record["source_sample_idx"],
        "rollout_index": rollout_index,
        "source_patch_sha256": source_record["patch_sha256"],
        "round0_mode": repair.get("round0_mode"),
        "round0_exit_status": repair.get("round0_exit_status"),
        "round0_turns": repair.get("round0_turns"),
        "round0_patch_chars": repair.get("round0_patch_chars"),
        "round0_trajectory": repair.get("round0_trajectory"),
        "seed_origin": repair.get("seed_origin"),
        "seed_msg_count": repair.get("seed_msg_count"),
        "repair_gate": repair.get("repair_gate", "oracle_f2p"),
        "generated_tests": repair.get("n_tests"),
        "isolated_generated_test_gate": repair.get("isolated_generated_test_gate"),
        "generated_test_gate_workspace": repair.get("generated_test_gate_workspace"),
        "initial_gate_status": repair.get("initial_gate_status"),
        "initial_gate_reason": repair.get("initial_gate_reason"),
        "final_gate_status": repair.get("final_gate_status", repair.get("gate_status")),
        "final_gate_reason": repair.get("final_gate_reason", repair.get("gate_reason")),
        "model_final_submit": repair.get("model_final_submit", False),
        "model_final_decision": repair.get("model_final_decision", False),
        "model_keep_original": repair.get("model_keep_original", False),
        "fallback_to_round0": repair.get("fallback_to_round0", False),
        "selected_f2p": repair.get("f2p_nodes"),
        "repair_error": repair.get("error"),
        "workspace_restore_mode": repair.get("workspace_restore_mode"),
        "workspace_replay": repair.get("workspace_replay"),
        "wrong_patch_applied": repair.get("wrong_patch_applied"),
        "initial_f2p_pass": repair.get("initial_pass"),
        "f2p_rescued": repair.get("rescued"),
        "rounds_used": repair.get("rounds_used"),
        "total_turns": repair.get("total_turns"),
        "final_patch_chars": len((repair.get("final_patch") or "").strip()),
        "official_verify_started": verify is not None,
        "official_resolved": verify.get("resolved") if verify else None,
        "official_patch_applied": verify.get("patch_applied") if verify else None,
        "official_verify_error": verify.get("error") if verify else None,
        "official_fail_to_pass_failed": verify.get("fail_to_pass_failed") if verify else None,
        "official_pass_to_pass_failed": verify.get("pass_to_pass_failed") if verify else None,
        "elapsed_seconds": round(elapsed, 3),
    }


def process_task(
    source_record: dict,
    rollout_index: int,
    args: argparse.Namespace,
    instances: dict[str, dict],
    repair_module,
    verify_module,
    repair_config: dict,
    verify_env_config: dict,
    tests_map: dict[str, list] | None = None,
) -> dict:
    directory = task_dir(args.output, source_record, rollout_index)
    status_path = directory / "final_status.json"
    if status_path.exists():
        return json.loads(status_path.read_text())
    started = time.time()
    directory.mkdir(parents=True, exist_ok=True)
    try:
        source_messages = []
        if args.fresh_solve_round0:
            patch = ""
            metadata = {}
        else:
            source_row, patch = load_source_row(args.input, source_record)
            metadata = source_row.get("metadata") or {}
            source_messages = source_row.get("messages") or []
        instance_id = source_record["instance_id"]
        instance = instances[instance_id]
        source_run = directory / "source_run"
        if args.seed_trajectory:
            trajectory_dir = source_run / instance_id
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            (trajectory_dir / f"{instance_id}.traj.json").write_text(json.dumps(source_row))
        repair_args = SimpleNamespace(
            seed=f"{args.seed}:{source_record['row_index']}:{rollout_index}",
            trajectory_dir=directory / "repair_round_trajs",
            out=directory / "repair.jsonl",
            install_timeout=args.install_timeout,
            test_timeout=args.test_timeout,
            max_rounds=args.max_rounds,
            max_gate_checks=args.max_gate_checks,
            continue_on_duplicate_patch=args.continue_on_duplicate_patch,
            steps=args.max_turns,
            force_submit_on_turn_limit=args.force_submit_on_turn_limit,
            n_f2p=1,
            seed_trajectory=args.seed_trajectory,
            fresh_solve_round0=args.fresh_solve_round0,
            round0_steps=args.round0_turns,
            round0_cost_limit=args.round0_cost_limit,
            source_run=source_run,
            apply_test_patch_at_start=args.apply_test_patch_at_start,
            isolated_generated_test_gate=args.isolated_generated_test_gate,
            apply_failure_rounds=2,
            apply_failure_steps=args.max_turns,
            full_eval_script_gate=args.full_eval_script_gate,
            restore_mode=args.restore_mode,
            replay_cmd_timeout=args.replay_cmd_timeout,
            replay_budget=args.replay_budget,
            allow_empty_source_patch=args.allow_empty_source_patch,
        )
        if tests_map is None:
            repair = repair_module.process_one(
                instance_id,
                instance,
                [patch],
                metadata.get("fail_to_pass_failed") or [],
                repair_config,
                repair_args,
                source_messages=source_messages,
            )
            repair["repair_gate"] = "oracle_f2p"
        else:
            selected_tests = (tests_map.get(instance_id) or [])[: args.max_tests]
            repair = repair_module.process_one(
                instance_id,
                instance,
                [patch],
                selected_tests,
                repair_config,
                repair_args,
                source_messages=source_messages,
            )
            repair["repair_gate"] = "generated_test"
        if not args.fresh_solve_round0:
            repair.setdefault("round0_exit_status", source_row.get("round0_exit_status"))
            repair.setdefault("round0_turns", source_row.get("round0_turns"))
            repair.setdefault("round0_patch_chars", len(patch))
            repair.setdefault("round0_trajectory", source_row.get("round0_trajectory"))
        repair.update(
            {
                "task_key": task_key(source_record, rollout_index),
                "row_index": source_record["row_index"],
                "source_sample_idx": source_record["source_sample_idx"],
                "rollout_index": rollout_index,
                "source_patch_sha256": source_record["patch_sha256"],
            }
        )
        atomic_json(directory / "repair.json", repair)
        final_patch = repair.get("final_patch") or ""
        # A unified diff must end on a line boundary. ``str.strip()`` makes an
        # otherwise valid git diff fail in a clean verify sandbox with
        # "corrupt patch at line N" when the last context line loses its LF.
        final_patch = normalize_patch_for_apply(final_patch)
        verify = None
        if final_patch:
            verify = verify_module.evaluate_instance_azure_modal(
                instance,
                final_patch,
                verify_env_config,
                logs_dir=directory / "official_verify_logs",
                sample_idx=int(source_record["row_index"]) * 1000 + rollout_index,
            )
            atomic_json(directory / "official_verify.json", verify)
        status = compact_status(repair, verify, source_record, rollout_index, time.time() - started)
    except Exception as exc:
        status = {
            "task_key": task_key(source_record, rollout_index),
            "row_index": source_record["row_index"],
            "instance_id": source_record["instance_id"],
            "source_sample_idx": source_record["source_sample_idx"],
            "rollout_index": rollout_index,
            "source_patch_sha256": source_record["patch_sha256"],
            "orchestrator_error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.time() - started, 3),
        }
    atomic_json(status_path, status)
    return status


def load_completed(path: Path) -> dict[str, dict]:
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("task_key"):
            completed[record["task_key"]] = record
    return completed


def source_records(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def pending_tasks(
    manifest_path: Path,
    completed: set[str],
    rollouts_per_source: int,
    rollout_index_start: int = 0,
) -> Iterator[tuple[dict, int]]:
    for source_record in source_records(manifest_path):
        for rollout_index in range(rollout_index_start, rollout_index_start + rollouts_per_source):
            if task_key(source_record, rollout_index) not in completed:
                yield source_record, rollout_index


def count_statuses(statuses: dict[str, dict]) -> dict:
    values = list(statuses.values())
    return {
        "completed_attempt_units": len(values),
        "round0_completed_units": sum(bool(record.get("round0_exit_status")) for record in values),
        "round0_patch_units": sum(bool(record.get("round0_patch_chars")) for record in values),
        "round0_f2p_pass_units": sum(
            record.get("round0_mode") == "fresh_e2e_solve" and record.get("initial_f2p_pass") is True
            for record in values
        ),
        "repair_started_units": sum(int(record.get("rounds_used") or 0) > 0 for record in values),
        "model_final_submit_units": sum(record.get("model_final_submit") is True for record in values),
        "repair_patch_units": sum(bool(record.get("final_patch_chars")) for record in values),
        "f2p_rescued_units": sum(record.get("f2p_rescued") is True for record in values),
        "official_verified_units": sum(record.get("official_verify_started") is True for record in values),
        "official_resolved_units": sum(record.get("official_resolved") is True for record in values),
        "repair_apply_failures": sum(record.get("repair_error") == "WRONG_PATCH_APPLY_FAILED" for record in values),
        "repair_errors": sum(bool(record.get("repair_error")) for record in values),
        "orchestrator_errors": sum(bool(record.get("orchestrator_error")) for record in values),
        "total_turns": sum(int(record.get("total_turns") or 0) for record in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Optional shared manifest path, allowing independent full-dataset rollout passes to reuse one scan.",
    )
    parser.add_argument("--harness-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--subset", default="rebench")
    parser.add_argument("--split", default="filtered")
    parser.add_argument("--endpoint", default=os.getenv("SANDBOX_BASE_URL", ""))
    parser.add_argument("--model", default="gpt-5.3-codex_2026-02-24")
    parser.add_argument("--model-class", default="trapi_response")
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Responses API reasoning effort for trapi_response models.",
    )
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Set chat_template_kwargs.enable_thinking=false for Qwen-style models.",
    )
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--rollouts-per-source", type=int, default=4)
    parser.add_argument(
        "--rollout-index-start",
        type=int,
        default=0,
        help="First rollout index generated by this invocation.",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=0,
        help="Stop manifest construction after this many eligible source trajectory rows; 0 means all.",
    )
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument(
        "--max-gate-checks",
        type=int,
        default=0,
        help=(
            "Maximum non-duplicate repair-gate evaluations; 0 preserves --max-rounds. "
            "This does not increase the max-rounds * max-turns model-turn budget."
        ),
    )
    parser.add_argument(
        "--continue-on-duplicate-patch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Do not terminate or rerun the gate for empty/duplicate submissions; ask the model "
            "for a concrete source change within the existing turn budget."
        ),
    )
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument(
        "--force-submit-on-turn-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Submit the current worktree diff to the repair gate at each --max-turns boundary "
            "(default: enabled; use --no-force-submit-on-turn-limit for legacy LimitsExceeded behavior)."
        ),
    )
    parser.add_argument(
        "--full-eval-script-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse one reset-before-grade sandbox for every focused official eval-script gate "
            "(default: enabled; use --no-full-eval-script-gate for the legacy selector path)."
        ),
    )
    parser.add_argument(
        "--fresh-solve-round0",
        action="store_true",
        help=(
            "Run the standard e2e issue-solving agent first. Its newly generated trajectory and patch "
            "become the seed for the one-F2P repair loop; the input trajectory is selection-only."
        ),
    )
    parser.add_argument(
        "--dataset-instances",
        action="store_true",
        help="Select one task per benchmark dataset instance instead of one task per input trajectory row.",
    )
    parser.add_argument(
        "--round0-turns",
        type=int,
        default=30,
        help="Maximum model turns for the fresh standard e2e solve before F2P repair.",
    )
    parser.add_argument("--round0-cost-limit", type=float, default=5.0)
    parser.add_argument(
        "--gentest-map",
        type=Path,
        default=None,
        help="Use generated tests from this {instance_id: [test,...]} map instead of oracle F2P.",
    )
    parser.add_argument("--max-tests", type=int, default=1)
    parser.add_argument("--apply-test-patch-at-start", action="store_true")
    parser.add_argument(
        "--isolated-generated-test-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run generated-test repair gates in a separate persistent reset-before-grade "
            "sandbox (default: enabled)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus-sampling cutoff; omitted preserves the model backend default.",
    )
    parser.add_argument("--test-timeout", type=int, default=240)
    parser.add_argument("--install-timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-trajectory",
        action="store_true",
        help="Prepend each source row's original solve trajectory before every repair rollout.",
    )
    parser.add_argument(
        "--restore-mode",
        choices=["patch", "replay"],
        default="patch",
        help="Restore only the submitted patch, or replay all source-trajectory workspace actions.",
    )
    parser.add_argument("--replay-cmd-timeout", type=int, default=120)
    parser.add_argument("--replay-budget", type=int, default=600)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument(
        "--allow-empty-source-patch",
        action="store_true",
        help=(
            "Keep source rows whose Round-0 worktree diff is empty. This is intended for matched "
            "repair replay; the repair starts from the clean base while retaining seed history."
        ),
    )
    args = parser.parse_args()
    if not args.endpoint:
        parser.error("--endpoint or SANDBOX_BASE_URL must be set explicitly")
    if args.seed_trajectory and args.fresh_solve_round0:
        parser.error("--seed-trajectory and --fresh-solve-round0 are mutually exclusive")
    if args.dataset_instances and not args.fresh_solve_round0:
        parser.error("--dataset-instances requires --fresh-solve-round0")
    if args.round0_turns <= 0:
        parser.error("--round0-turns must be positive")
    if args.max_turns <= 0:
        parser.error("--max-turns must be positive")
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    if args.max_gate_checks < 0:
        parser.error("--max-gate-checks must be non-negative")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    args.harness_root = args.harness_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.gentest_map is None:
        repair_module = load_module("selfrepair_f2p", args.harness_root / "scripts/repair/selfrepair_f2p.py")
        tests_map = None
    else:
        repair_module = load_module(
            "selfrepair_gentest", args.harness_root / "scripts/repair/cache/selfrepair_gentest.py"
        )
        tests_map = json.loads(args.gentest_map.read_text())
    if args.subset == "ScaleAI/SWE-bench_Pro":
        verify_family = "swebench_pro"
    elif args.subset in {"rebench", "rebench_v2"}:
        verify_family = "swerebench"
    else:
        verify_family = "swebench"
    verify_module = load_module(
        f"{verify_family}_verify_azure_modal",
        args.harness_root / f"src/minisweagent/run/benchmarks/{verify_family}_verify_azure_modal.py",
    )
    print(f"OFFICIAL_VERIFY_FAMILY {verify_family}", flush=True)
    dataset_path = repair_module.rca.DATASET_MAPPING.get(args.subset, args.subset)
    dataset_rows = [dict(row) for row in load_dataset(dataset_path, split=args.split)]
    if verify_family == "swebench_pro":
        dataset_rows = [verify_module.prepare_instance(row) for row in dataset_rows]
    instances = {row["instance_id"]: row for row in dataset_rows}

    manifest_path = args.source_manifest or (args.output / "source_manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rebuild_manifest or not manifest_path.exists():
        if args.dataset_instances:
            manifest_summary = build_dataset_manifest(
                dataset_rows,
                manifest_path,
                max_source_rows=args.max_source_rows,
                rollouts_per_source=args.rollouts_per_source,
            )
        else:
            manifest_summary = build_source_manifest(
                args.input,
                manifest_path,
                instances,
                max_source_rows=args.max_source_rows,
                rollouts_per_source=args.rollouts_per_source,
                allow_empty_source_patch=args.allow_empty_source_patch,
            )
    else:
        manifest_summary = json.loads(manifest_path.with_suffix(".summary.json").read_text())

    sandbox_key = os.environ.get("SANDBOX_API_KEY", "")
    if not sandbox_key:
        raise SystemExit("SANDBOX_API_KEY is required")
    if args.model_class == "trapi_response" and not os.environ.get("TRAPI_BEARER_TOKEN_FILE") \
            and not os.environ.get("TRAPI_BEARER_TOKEN") and not os.environ.get("TRAPI_ALLOW_AZ_CRED"):
        raise SystemExit("TRAPI bearer token is required")
    repair_config = repair_module.build_config(
        args.api_base or "unused",
        args.api_key,
        args.endpoint,
        sandbox_key,
        args.temperature,
        args.model,
        args.model_class,
        args.disable_thinking,
        top_p=args.top_p,
        reasoning_effort=args.reasoning_effort,
    )
    verify_env_config = {
        "base_url": args.endpoint,
        "api_key": sandbox_key,
        "sandbox_timeout": max(args.test_timeout, 600),
    }
    run_config = {
        "input": str(args.input),
        "input_trajectory_role": "not_used" if args.dataset_instances else (
            "selection_only" if args.fresh_solve_round0 else "patch_and_optional_seed"
        ),
        "allow_empty_source_patch": args.allow_empty_source_patch,
        "output": str(args.output),
        "candidate_unit": "dataset_instance" if args.dataset_instances else "source_trajectory_row",
        "source_row_limit": args.max_source_rows,
        "eligible_source_rows": manifest_summary["eligible_source_rows"],
        "rollouts_per_source": args.rollouts_per_source,
        "rollout_index_start": args.rollout_index_start,
        "rollout_indices": list(
            range(args.rollout_index_start, args.rollout_index_start + args.rollouts_per_source)
        ),
        "total_attempt_units": manifest_summary["eligible_source_rows"] * args.rollouts_per_source,
        "endpoint": args.endpoint,
        "model": args.model,
        "model_class": args.model_class,
        "reasoning_effort": args.reasoning_effort,
        "api_base": args.api_base or None,
        "disable_thinking": args.disable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "workers": args.workers,
        "max_rounds": args.max_rounds,
        "max_gate_checks": args.max_gate_checks or args.max_rounds,
        "continue_on_duplicate_patch": args.continue_on_duplicate_patch,
        "repair_turn_budget": args.max_rounds * args.max_turns,
        "max_turns_per_round": args.max_turns,
        "force_submit_on_turn_limit": args.force_submit_on_turn_limit,
        "round0_mode": "fresh_e2e_solve" if args.fresh_solve_round0 else "provided_patch",
        "round0_turns": args.round0_turns if args.fresh_solve_round0 else None,
        "round0_cost_limit": args.round0_cost_limit if args.fresh_solve_round0 else None,
        "repair_seed": "fresh_round0_e2e_trajectory" if args.fresh_solve_round0 else (
            "input_source_trajectory" if args.seed_trajectory else "none"
        ),
        "repair_gate": "generated_test" if tests_map is not None else "oracle_f2p",
        "full_eval_script_gate": args.full_eval_script_gate,
        "gentest_map": str(args.gentest_map) if args.gentest_map else None,
        "gentest_instances": len(tests_map) if tests_map is not None else None,
        "max_tests": args.max_tests if tests_map is not None else None,
        "apply_test_patch_at_start": args.apply_test_patch_at_start if tests_map is not None else None,
        "isolated_generated_test_gate": (
            args.isolated_generated_test_gate if tests_map is not None else None
        ),
        "n_f2p": 1,
        "seed_trajectory": args.seed_trajectory,
        "workspace_restore_mode": args.restore_mode if not args.fresh_solve_round0 else None,
        "replay_cmd_timeout": args.replay_cmd_timeout if args.restore_mode == "replay" else None,
        "replay_budget": args.replay_budget if args.restore_mode == "replay" else None,
    }
    atomic_json(args.output / "run_config.json", run_config)
    print("RUN_CONFIG " + json.dumps(run_config, sort_keys=True), flush=True)

    completed_path = args.output / "completed.jsonl"
    statuses = load_completed(completed_path)
    task_iterator = pending_tasks(
        manifest_path,
        set(statuses),
        args.rollouts_per_source,
        args.rollout_index_start,
    )
    # Keep one submitted task per worker. With hundreds of workers, pre-queuing
    # a second full wave retains unnecessary source/task state and can push the
    # long-lived pod into host memory pressure without increasing concurrency.
    max_in_flight = max(1, args.workers)
    submitted = 0
    started = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[concurrent.futures.Future, tuple[dict, int]] = {}

        def submit_next() -> bool:
            nonlocal submitted
            try:
                source_record, rollout_index = next(task_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                process_task,
                source_record,
                rollout_index,
                args,
                instances,
                repair_module,
                verify_module,
                repair_config,
                verify_env_config,
                tests_map,
            )
            futures[future] = (source_record, rollout_index)
            submitted += 1
            return True

        while len(futures) < max_in_flight and submit_next():
            pass

        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                source_record, rollout_index = futures.pop(future)
                try:
                    status = future.result()
                except Exception as exc:
                    status = {
                        "task_key": task_key(source_record, rollout_index),
                        "row_index": source_record["row_index"],
                        "instance_id": source_record["instance_id"],
                        "source_sample_idx": source_record["source_sample_idx"],
                        "rollout_index": rollout_index,
                        "orchestrator_error": f"{type(exc).__name__}: {exc}",
                    }
                statuses[status["task_key"]] = status
                with completed_path.open("a") as handle:
                    handle.write(json.dumps(status, ensure_ascii=False, sort_keys=True) + "\n")
                counters = count_statuses(statuses)
                progress = {
                    **run_config,
                    **counters,
                    "submitted_this_process": submitted,
                    "in_flight": len(futures),
                    "elapsed_seconds": round(time.time() - started, 3),
                    "last_task_key": status["task_key"],
                }
                atomic_json(args.output / "progress.json", progress)
                print(
                    f"done={counters['completed_attempt_units']}/{run_config['total_attempt_units']} "
                    f"f2p_rescued={counters['f2p_rescued_units']} "
                    f"model_final_submit={counters['model_final_submit_units']} "
                    f"official_resolved={counters['official_resolved_units']} "
                    f"errors={counters['repair_errors'] + counters['orchestrator_errors']} "
                    f"last={status['task_key']}",
                    flush=True,
                )
                while len(futures) < max_in_flight and submit_next():
                    pass

    final_summary = {**run_config, **count_statuses(statuses), "elapsed_seconds": round(time.time() - started, 3)}
    atomic_json(args.output / "final_summary.json", final_summary)
    print("FINAL_SUMMARY " + json.dumps(final_summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
