#!/usr/bin/env python3
"""Repair FAIL_SELF_TEST / TEST_UNKNOWN SWE-bench trajectories.

The script selects instances from a completed mini-swe-agent run by generation-time
self-test category, augments each original SWE-bench problem with prior trajectory
observations, and runs mini-swe-agent again. Optionally verifies the repaired
trajectories after generation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Make this script runnable from the copied harness tree without an editable install.
HARNESS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HARNESS_ROOT / "src"))
sys.path.insert(0, str(HARNESS_ROOT / "external" / "azure-modal"))

from datasets import load_dataset  # noqa: E402
from rich.live import Live  # noqa: E402

from minisweagent.config import get_config_from_spec  # noqa: E402
from minisweagent.run.benchmarks import swebench as swebench_runner  # noqa: E402
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager  # noqa: E402
from minisweagent.utils.log import add_file_handler, logger  # noqa: E402
from minisweagent.utils.serialize import UNSET, recursive_merge  # noqa: E402

DATASET_MAPPING = swebench_runner.DATASET_MAPPING


def load_inspector():
    path = HARNESS_ROOT / "scripts" / "score" / "inspect_self_test_patterns.py"
    spec = importlib.util.spec_from_file_location("inspect_self_test_patterns", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import inspector from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def short_text(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... <truncated {len(text) - len(head) - len(tail)} chars> ...\n{tail}"


def load_patch_from_traj(data: dict) -> str:
    return (
        (data.get("info", {}) or {}).get("submission", "")
        or (data.get("metadata", {}) or {}).get("model_patch", "")
        or ""
    )


def summarize_event(event: dict, *, command_limit: int, output_limit: int) -> str:
    command = short_text(event.get("command", ""), command_limit)
    output = short_text(event.get("output", ""), output_limit)
    return (
        f"- kind={event.get('kind')} status={event.get('status')} reason={event.get('reason')}\n"
        f"  command:\n{command}\n"
        f"  observation excerpt:\n{output}"
    )


def make_repair_context(
    *,
    iid: str,
    category: str,
    traj_data: dict,
    events: list[dict],
    verify_entry: dict | None,
    include_verify_results: bool,
    max_events: int,
    patch_limit: int,
    command_limit: int,
    output_limit: int,
) -> str:
    info = traj_data.get("info", {}) if isinstance(traj_data, dict) else {}
    patch = load_patch_from_traj(traj_data)
    event_lines = [
        summarize_event(event, command_limit=command_limit, output_limit=output_limit)
        for event in events[-max_events:]
    ]
    if not event_lines:
        event_lines = ["- No recognizable self-test command was detected in the previous trajectory."]

    verify_summary = ""
    if include_verify_results and verify_entry:
        sample = (verify_entry.get("samples") or [verify_entry])[0]
        f2p_failed = sample.get("fail_to_pass_failed", [])
        p2p_failed = sample.get("pass_to_pass_failed", [])
        verify_summary = f"""
+<previous_official_verify_result>
+resolved: {bool(verify_entry.get('resolved') or sample.get('resolved'))}
+patch_applied: {sample.get('patch_applied')}
+exit_code: {sample.get('exit_code')}
+FAIL_TO_PASS failed ({len(f2p_failed)}): {short_text(json.dumps(f2p_failed, ensure_ascii=False), output_limit)}
+PASS_TO_PASS failed ({len(p2p_failed)}): {short_text(json.dumps(p2p_failed, ensure_ascii=False), output_limit)}
+</previous_official_verify_result>
+""".strip()

    patch_summary = short_text(patch, patch_limit) if patch.strip() else "<empty previous patch>"

    return f"""
+
+<repair_context>
+You are making a second attempt for instance {iid}.
+
+The previous attempt was selected because its generation-time self-test category was: {category}.
+Previous exit_status: {info.get('exit_status')}
+Previous patch was non-empty: {bool(patch.strip())}
+
+<previous_self_test_observations>
+{chr(10).join(event_lines)}
+</previous_self_test_observations>
+
+<previous_patch_excerpt>
+{patch_summary}
+</previous_patch_excerpt>
+
+{verify_summary}
+
+Repair workflow to follow:
+1. Act -> Observe: start from the prior observations above; do not blindly repeat the same patch.
+2. Evaluate: decide whether the prior self-test actually proved the fix. If it failed or was ambiguous, name the missing evidence.
+3. Diagnose: identify the likely root cause and why the previous patch/test was insufficient.
+4. Generate an improvement target: make a smaller or more correct patch that addresses the diagnosis.
+5. Re-evaluate: run a meaningful targeted test or reproduction. If it fails or is ambiguous, iterate before submitting.
+6. Submit only a source-code patch once you have a meaningful passing self-test or a clearly justified fix.
+</repair_context>
+""".strip()


def select_instances(
    source_run: Path,
    categories: set[str],
    *,
    require_nonempty_patch: bool,
    include_ids: set[str] | None,
    limit: int,
) -> tuple[list[dict], dict[str, dict], object]:
    inspector = load_inspector()
    selected: list[dict] = []
    metadata: dict[str, dict] = {}
    for traj_path in sorted(source_run.rglob("*.traj.json")):
        data = json.loads(traj_path.read_text())
        iid = data.get("instance_id") or traj_path.parent.name
        if include_ids is not None and iid not in include_ids:
            continue
        events = list(inspector.iter_test_events(data))
        category = inspector.classify_last_event(events)
        patch = load_patch_from_traj(data)
        if category not in categories:
            continue
        if require_nonempty_patch and not patch.strip():
            continue
        selected.append({"instance_id": iid, "traj_path": str(traj_path), "category": category})
        metadata[iid] = {
            "traj_path": str(traj_path),
            "category": category,
            "events": events,
            "traj_data": data,
            "previous_patch_nonempty": bool(patch.strip()),
        }
        if limit and len(selected) >= limit:
            break
    return selected, metadata, inspector


def build_augmented_instances(
    selected: list[dict],
    metadata: dict[str, dict],
    *,
    subset: str,
    split: str,
    verify_results: dict | None,
    include_verify_results: bool,
    max_events: int,
    patch_limit: int,
    command_limit: int,
    output_limit: int,
) -> list[dict]:
    dataset_path = DATASET_MAPPING.get(subset, subset)
    instances_by_id = {instance["instance_id"]: dict(instance) for instance in load_dataset(dataset_path, split=split)}
    augmented: list[dict] = []
    missing: list[str] = []
    for row in selected:
        iid = row["instance_id"]
        if iid not in instances_by_id:
            missing.append(iid)
            continue
        instance = dict(instances_by_id[iid])
        meta = metadata[iid]
        verify_entry = verify_results.get(iid) if verify_results else None
        repair_context = make_repair_context(
            iid=iid,
            category=meta["category"],
            traj_data=meta["traj_data"],
            events=meta["events"],
            verify_entry=verify_entry,
            include_verify_results=include_verify_results,
            max_events=max_events,
            patch_limit=patch_limit,
            command_limit=command_limit,
            output_limit=output_limit,
        )
        instance["problem_statement"] = instance["problem_statement"].rstrip() + "\n\n" + repair_context
        augmented.append(instance)
    if missing:
        logger.warning("%d selected ids missing from dataset: %s", len(missing), missing[:10])
    return augmented


def run_generation(instances: list[dict], output_dir: Path, config: dict, *, workers: int, instance_timeout: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_dir / "repair_minisweagent.log")
    progress_manager = RunBatchProgressManager(len(instances), output_dir / f"repair_statuses_{time.time()}.yaml")
    live_ctx = Live(progress_manager.render_group, refresh_per_second=4) if workers <= 1 else progress_manager._main_progress_bar
    if workers > 1:
        progress_manager._main_progress_bar.live.redirect_stdout = False
        progress_manager._main_progress_bar.live.redirect_stderr = False
    with live_ctx:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    swebench_runner.process_instance,
                    instance,
                    output_dir,
                    config,
                    progress_manager,
                    0,
                    True,
                    instance_timeout,
                ): instance["instance_id"]
                for instance in instances
            }
            done: set = set()
            while len(done) < len(futures):
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=2.0):
                        done.add(future)
                        try:
                            future.result()
                        except Exception as exc:
                            iid = futures[future]
                            logger.error("repair generation failed for %s: %s", iid, exc, exc_info=True)
                            progress_manager.on_uncaught_exception(iid, exc)
                except concurrent.futures.TimeoutError:
                    pass


def run_verify(output_dir: Path, *, subset: str, split: str, workers: int, extra_args: list[str]) -> None:
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
    subprocess.run(cmd, check=True)


def run_tally(output_dir: Path, denom: int | None) -> None:
    tally = HARNESS_ROOT / "scripts" / "score" / "tally.py"
    cmd = [sys.executable, str(tally), str(output_dir)]
    if denom is not None:
        cmd.append(str(denom))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run", type=Path, help="Completed source run directory")
    parser.add_argument("--output", type=Path, default=None, help="Repair output directory")
    parser.add_argument("--categories", default="fail_self_test,test_unknown", help="Comma-separated self-test categories")
    parser.add_argument("--subset", default="verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=int(os.getenv("REPAIR_WORKERS", "32")))
    parser.add_argument("--verify-workers", type=int, default=int(os.getenv("REPAIR_VERIFY_WORKERS", "64")))
    parser.add_argument("--instance-timeout", type=int, default=int(os.getenv("REPAIR_INSTANCE_TIMEOUT", "900")))
    parser.add_argument("--temperature", default=os.getenv("TEMPERATURE", "1"))
    parser.add_argument("--api-base", default=os.getenv("MODEL_API_BASE", "http://127.0.0.1:{port}/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--sandbox-base-url", default=os.getenv("SANDBOX_BASE_URL", ""))
    parser.add_argument("--sandbox-api-key", default=os.getenv("SANDBOX_API_KEY", ""))
    parser.add_argument("--denom", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=1, help="Repeat repair on fail/unknown trajectories from the previous repair output")
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--require-nonempty-patch", action="store_true")
    parser.add_argument("--include-verify-results", action="store_true", help="Include previous official verify failures in repair prompt; this is oracle-guided")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--patch-limit", type=int, default=12000)
    parser.add_argument("--command-limit", type=int, default=2000)
    parser.add_argument("--output-limit", type=int, default=4000)
    args, passthrough = parser.parse_known_args()
    if not args.dry_run and not args.sandbox_base_url:
        parser.error("--sandbox-base-url or SANDBOX_BASE_URL must be set explicitly")

    source_run = args.source_run.resolve()
    if not source_run.exists():
        raise SystemExit(f"source run does not exist: {source_run}")
    output_dir = args.output or source_run.parent / f"{source_run.name}_repair_{time.strftime('%Y%m%d_%H%M%S')}"
    categories = {item.strip() for item in args.categories.split(",") if item.strip()}
    include_ids = None
    if args.ids_file:
        include_ids = {line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()}

    config_specs = [
        "swebench",
        "swebench_azure_modal",
        f"model.model_kwargs.temperature={args.temperature}",
        f"model.model_kwargs.api_key={args.api_key}",
        f"model.model_kwargs.api_base={args.api_base}",
        f"environment.base_url={args.sandbox_base_url}",
        f"environment.api_key={args.sandbox_api_key}",
    ]
    configs = [get_config_from_spec(spec) for spec in config_specs]
    configs.append({"environment": {"environment_class": UNSET}, "model": {"model_name": UNSET, "model_class": UNSET}})
    config = recursive_merge(*configs)

    current_source = source_run
    for iteration in range(1, max(args.max_iters, 1) + 1):
        iter_output = output_dir if args.max_iters <= 1 else output_dir / f"iter_{iteration}"
        selected, metadata, _ = select_instances(
            current_source,
            categories,
            require_nonempty_patch=args.require_nonempty_patch,
            include_ids=include_ids,
            limit=args.limit,
        )
        counts: dict[str, int] = {}
        for row in selected:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        print(f"source_run={current_source}")
        print(f"output_dir={iter_output}")
        print(f"iteration={iteration}/{max(args.max_iters, 1)} selected={len(selected)} categories={counts}")

        verify_results = None
        verify_path = current_source / "verify_results_azure_modal.json"
        if args.include_verify_results and verify_path.exists():
            verify_results = json.loads(verify_path.read_text())
            print(f"include_verify_results={verify_path}")
        elif args.include_verify_results:
            print(f"include_verify_results requested but missing: {verify_path}")

        if args.dry_run:
            for row in selected[:20]:
                print(row)
            return
        if not selected:
            print("no selected instances; stopping")
            break

        iter_output.mkdir(parents=True, exist_ok=True)
        (iter_output / "selected_repair_instances.json").write_text(json.dumps(selected, indent=2))

        instances = build_augmented_instances(
            selected,
            metadata,
            subset=args.subset,
            split=args.split,
            verify_results=verify_results,
            include_verify_results=args.include_verify_results,
            max_events=args.max_events,
            patch_limit=args.patch_limit,
            command_limit=args.command_limit,
            output_limit=args.output_limit,
        )

        print(f"running repair generation for {len(instances)} instances workers={args.workers}")
        run_generation(instances, iter_output, config, workers=args.workers, instance_timeout=args.instance_timeout)

        if not args.no_verify:
            print(f"running repair verification workers={args.verify_workers}")
            run_verify(iter_output, subset=args.subset, split=args.split, workers=args.verify_workers, extra_args=passthrough)
            run_tally(iter_output, args.denom)

        current_source = iter_output


if __name__ == "__main__":
    main()
