#!/usr/bin/env python3
"""Run the trainable F2P-repair rollout directly, without Ray or Megatron."""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import math
import os
import re
import signal
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from slime.utils.http_utils import init_http_client
from slime.utils.types import Sample


LOGGER = logging.getLogger("offline_f2p_repair_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sglang-host", default="127.0.0.1")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=6122)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("examples/mini_swe/configs/swe_agent_v2_qwen3.5.yaml"))
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N rows (0 means all).")
    parser.add_argument("--force", action="store_true", help="Rerun rows that already have a driver result or trajectory.")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    return args


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    ids = [str((row.get("metadata") or {}).get("instance_id") or row.get("instance_id") or "") for row in rows]
    if any(not instance_id for instance_id in ids):
        raise ValueError("every dataset row must contain metadata.instance_id or instance_id")
    duplicate_ids = sorted(instance_id for instance_id, count in collections.Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"dataset contains duplicate instance IDs: {duplicate_ids[:10]}")
    return rows


def instance_id(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("instance_id") or row["instance_id"])


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def stats(values: list[float]) -> dict[str, float | int]:
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p90": round(values[math.ceil(0.9 * len(values)) - 1], 2),
        "max": round(max(values), 2),
    }


def load_existing_results(output_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "results").glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("ignoring unreadable result file %s", path)
            continue
        if record.get("instance_id"):
            records[str(record["instance_id"])] = record
    return records


def load_existing_trajectories(output_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "trajectories").rglob("*.f2p_repair.json")):
        try:
            trajectory = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("ignoring unreadable trajectory file %s", path)
            continue
        iid = str(trajectory.get("instance_id") or "")
        if not iid:
            continue
        repair = trajectory.get("f2p_repair") or {}
        official = repair.get("official_verify") or {}
        records[iid] = {
            "instance_id": iid,
            "status": "trajectory_only",
            "exit_status": trajectory.get("exit_status"),
            "resolved": official.get("resolved") is True,
            "official_verify": official,
            "n_steps": trajectory.get("n_steps"),
            "round0_turns": repair.get("round0_turns"),
            "repair_turns": repair.get("repair_turns"),
            "repair_rounds_started": repair.get("repair_rounds_started"),
            "f2p_passed": repair.get("f2p_passed") is True,
            "trajectory_path": str(path),
            "elapsed_seconds": trajectory.get("total_time"),
            "recovered_from_trajectory": True,
        }
    return records


def build_summary(
    expected_ids: list[str], records: dict[str, dict[str, Any]], *, active: int, started_at: float
) -> dict[str, Any]:
    expected = set(expected_ids)
    relevant = {iid: record for iid, record in records.items() if iid in expected}
    resolved = sum(record.get("resolved") is True for record in relevant.values())
    status_counts = collections.Counter(str(record.get("status") or "unknown") for record in relevant.values())
    exit_status_counts = collections.Counter(
        str(record.get("exit_status") or "unknown") for record in relevant.values()
    )
    turns = [float(record["n_steps"]) for record in relevant.values() if isinstance(record.get("n_steps"), (int, float))]
    elapsed = [
        float(record["elapsed_seconds"])
        for record in relevant.values()
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    completed = len(relevant)
    return {
        "expected_samples": len(expected_ids),
        "completed": completed,
        "active": active,
        "remaining": len(expected_ids) - completed,
        "resolved": resolved,
        "accuracy_fixed_denominator": resolved / len(expected_ids) if expected_ids else 0.0,
        "accuracy_completed": resolved / completed if completed else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "exit_status_counts": dict(sorted(exit_status_counts.items())),
        "entered_repair": sum(int(record.get("repair_rounds_started") or 0) > 0 for record in relevant.values()),
        "f2p_passed": sum(record.get("f2p_passed") is True for record in relevant.values()),
        "official_error_count": sum(bool((record.get("official_verify") or {}).get("error")) for record in relevant.values()),
        "turns": stats(turns),
        "sample_wall_time_seconds": stats(elapsed),
        "run_wall_time_seconds": round(time.time() - started_at, 2),
        "missing_ids": [iid for iid in expected_ids if iid not in relevant],
        "resolved_ids": sorted(iid for iid, record in relevant.items() if record.get("resolved") is True),
    }


def rollout_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        hf_checkpoint=str(args.checkpoint),
        sglang_router_ip=args.sglang_host,
        sglang_router_port=args.sglang_port,
        sglang_server_concurrency=args.workers,
        rollout_num_engines=8,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=args.temperature,
        rollout_top_p=args.top_p,
        rollout_top_k=args.top_k,
        rollout_max_response_len=args.max_new_tokens,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        rollout_seed=args.seed,
        n_samples_per_prompt=1,
        sglang_dp_size=1,
        sglang_enable_deterministic_inference=False,
        use_distributed_post=False,
        use_rollout_routing_replay=False,
        swe_config_path=str(args.config),
    )


async def main_async(args: argparse.Namespace) -> int:
    # Import after setting SWE_TRAJECTORY_DIR so the rollout is fully isolated under this run root.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SWE_TRAJECTORY_DIR"] = str(args.output_dir / "trajectories")
    os.environ.setdefault("SWE_CONFIG_PATH", str(args.config))
    from examples.mini_swe.f2p_repair_rollout import GenerateState, generate

    rows = load_rows(args.dataset, args.limit)
    expected_ids = [instance_id(row) for row in rows]
    records = load_existing_results(args.output_dir)
    for iid, record in load_existing_trajectories(args.output_dir).items():
        records.setdefault(iid, record)
    if args.force:
        records = {}

    model_args = rollout_args(args)
    init_http_client(model_args)
    state = GenerateState(model_args)
    sampling_params = state.sampling_params.copy()
    semaphore = asyncio.Semaphore(args.workers)
    result_lock = asyncio.Lock()
    started_at = time.time()
    running = 0
    stopping = False

    pending_rows = [(position, row) for position, row in enumerate(rows) if instance_id(row) not in records]
    LOGGER.info(
        "offline F2P repair: expected=%d resumed=%d pending=%d workers=%d endpoint=http://%s:%d/generate",
        len(rows), len(rows) - len(pending_rows), len(pending_rows), args.workers, args.sglang_host, args.sglang_port,
    )

    async def persist_summary() -> dict[str, Any]:
        summary = build_summary(expected_ids, records, active=running, started_at=started_at)
        atomic_json(args.output_dir / "summary.json", summary)
        return summary

    async def run_one(position: int, row: dict[str, Any]) -> None:
        nonlocal running
        iid = instance_id(row)
        async with semaphore:
            running += 1
            sample_started = time.time()
            try:
                sample = Sample(
                    index=position,
                    group_index=position,
                    prompt=str(row.get("problem_statement") or (row.get("metadata") or {}).get("problem_statement") or ""),
                    label=row.get("patch"),
                    metadata=dict(row),
                )
                result = await generate(
                    model_args,
                    sample,
                    sampling_params.copy(),
                    evaluation=True,
                )
                repair = result.metadata.get("f2p_repair") or {}
                official = repair.get("official_verify") or {}
                record = {
                    "instance_id": iid,
                    "index": position,
                    "status": result.status.value,
                    "exit_status": result.metadata.get("exit_status"),
                    "resolved": official.get("resolved") is True,
                    "official_verify": official,
                    "n_steps": result.metadata.get("n_steps"),
                    "round0_turns": repair.get("round0_turns"),
                    "repair_turns": repair.get("repair_turns"),
                    "repair_rounds_started": repair.get("repair_rounds_started"),
                    "repair_rounds_used": repair.get("repair_rounds_used"),
                    "f2p_passed": repair.get("f2p_passed") is True,
                    "had_submission": repair.get("had_submission") is True,
                    "force_submit_on_turn_limit": repair.get("force_submit_on_turn_limit") is True,
                    "stop_reason": repair.get("stop_reason"),
                    "trajectory_path": result.metadata.get("trajectory_path"),
                    "error": result.metadata.get("f2p_repair_error"),
                    "elapsed_seconds": round(time.time() - sample_started, 2),
                }
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep the other 499 rows running and make the failure auditable
                LOGGER.exception("driver failure for %s", iid)
                record = {
                    "instance_id": iid,
                    "index": position,
                    "status": "driver_error",
                    "exit_status": "driver_error",
                    "resolved": False,
                    "official_verify": {},
                    "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                    "elapsed_seconds": round(time.time() - sample_started, 2),
                }
            finally:
                running -= 1

            async with result_lock:
                records[iid] = record
                atomic_json(args.output_dir / "results" / f"{position:04d}.{safe_name(iid)}.json", record)
                summary = await persist_summary()
                LOGGER.info(
                    "completed=%d/%d active=%d resolved=%d fixed_acc=%.4f last=%s status=%s turns=%s",
                    summary["completed"], summary["expected_samples"], summary["active"], summary["resolved"],
                    summary["accuracy_fixed_denominator"], iid, record["status"], record.get("n_steps"),
                )

    await persist_summary()
    tasks = [asyncio.create_task(run_one(position, row), name=instance_id(row)) for position, row in pending_rows]
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        LOGGER.warning("stop requested; cancelling %d tasks so active sandboxes can clean up", len(tasks))
        for task in tasks:
            if not task.done():
                task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, request_stop)

    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    cancelled = sum(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
    summary = await persist_summary()
    LOGGER.info(
        "finished=%d/%d resolved=%d fixed_acc=%.4f cancelled=%d summary=%s",
        summary["completed"], summary["expected_samples"], summary["resolved"],
        summary["accuracy_fixed_denominator"], cancelled, args.output_dir / "summary.json",
    )
    return 130 if stopping else 0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "offline.log"),
        ],
    )
    for path, label in ((args.checkpoint, "checkpoint"), (args.dataset, "dataset"), (args.config, "config")):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
