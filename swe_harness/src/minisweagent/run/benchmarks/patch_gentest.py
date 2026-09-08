#!/usr/bin/env python3
"""Generate patch-conditioned regression tests for patch-group classification.

This harness is a sibling of native gentest. It reads candidate patch groups,
shows one candidate patch to the generated-test agent as diagnostic context,
and records three validation layers:

- base validation: the generated test should cleanly fail on the buggy base;
- gold validation: the generated test should pass after the SWE-bench gold patch;
- candidate validation: the generated test is run after the candidate patch.

The candidate patch is prompt context only. The model still has to produce an
issue-backed behavioral regression test through GeneratedTestSubmitAgent.
"""

from __future__ import annotations

import concurrent.futures
import copy
import csv
import hashlib
import json
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any

import typer
from datasets import load_dataset

from minisweagent.agents import get_agent
from minisweagent.config import builtin_config_dir
from minisweagent.models import get_model
from minisweagent.run.benchmarks.gentest import (
    DEFAULT_CONFIG_FILE as GENTEST_CONFIG_FILE,
    TEST_FILE,
    ActivatingEnv,
    apply_test_patch_for_alignment,
    build_config,
    evaluate_oracle_contract,
    extract_example,
    generated_test_file_for_instance,
    infra_failure_info,
    list_from_json_or_obj,
    load_ids,
    read_test_file,
    reset_base,
    restore_env,
    run_generated_test,
    run_generated_test_official,
    shell,
    starter_code_for_sample,
    test_command_for_instance,
    test_style_guidance_for_instance,
    validate_generated_test,
    write_file,
    write_generated_test_file,
    _record_for_traj,
)
from minisweagent.run.benchmarks.swerebench import DATASET_MAPPING, _resolve_per_instance_api_base, get_sb_environment
from minisweagent.utils.log import add_file_handler, logger


app = typer.Typer(rich_markup_mode="rich", add_completion=False)
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "patch_gentest.yaml"
DEFAULT_GROUPS_FILE = "groups_with_gold.tsv"
DEFAULT_PATCH_MAX_CHARS = 12_000


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('instance_id', '')}::{row.get('class_id', '')}::{row.get('source_type', '')}"


def _safe_name(text: str, *, max_chars: int = 180) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    if len(safe) <= max_chars:
        return safe or "patch"
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]
    return f"{safe[: max_chars - 13]}_{digest}"


def _int_field(row: dict[str, Any], name: str) -> int:
    try:
        return int(row.get(name) or 0)
    except (TypeError, ValueError):
        return 0


def _resolve_patch_file(groups_file: Path, value: str) -> Path:
    path = Path(value or "")
    if path.is_absolute():
        return path
    return groups_file.parent / path


def _changed_files_from_patch(patch_text: str) -> list[str]:
    files = re.findall(r"(?m)^\+\+\+ b/(.+)$", patch_text or "")
    if not files:
        files = re.findall(r"(?m)^diff --git a/\S+ b/(\S+)$", patch_text or "")
    return list(dict.fromkeys(path.strip() for path in files if path.strip()))


def _shorten_middle(text: str, max_chars: int) -> tuple[str, bool]:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    marker = "\n\n[... candidate patch truncated by patch_gentest harness ...]\n\n"
    if max_chars <= len(marker) + 2:
        return text[:max_chars], True
    head_len = max(0, (max_chars - len(marker)) // 2)
    tail_len = max(0, max_chars - len(marker) - head_len)
    return f"{text[:head_len]}{marker}{text[-tail_len:]}", True


def load_patch_rows(
    groups_file: Path,
    *,
    id_filter: set[str] | None,
    source_types: set[str],
    include_gold: bool,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    missing_patch_files = 0
    want_gold = include_gold or "gold" in source_types
    with groups_file.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            iid = raw.get("instance_id") or ""
            if id_filter is not None and iid not in id_filter:
                continue
            source_type = raw.get("source_type") or ""
            if source_type == "gold" and not want_gold:
                continue
            if source_types and source_type not in source_types:
                continue
            patch_file = _resolve_patch_file(groups_file, raw.get("patch_file") or "")
            if not patch_file.exists():
                missing_patch_files += 1
                continue
            class_id = raw.get("class_id") or patch_file.stem
            oracle_resolved = True if source_type == "gold" else _int_field(raw, "resolved_count") > 0
            row = {
                "instance_id": iid,
                "class_id": class_id,
                "source_type": source_type,
                "patch_file": str(patch_file),
                "oracle_resolved": oracle_resolved,
                "resolved_count": _int_field(raw, "resolved_count"),
                "unresolved_count": _int_field(raw, "unresolved_count"),
                "attempt_count": _int_field(raw, "attempt_count"),
                "patch_chars": _int_field(raw, "patch_chars"),
            }
            row["key"] = row_key(row)
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows, missing_patch_files


def read_existing(results_jsonl: Path) -> set[tuple[str, int]]:
    existing: set[tuple[str, int]] = set()
    if not results_jsonl.exists():
        return existing
    for line in results_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("key") or row_key(row)
        try:
            sample_idx = int(row.get("sample_idx", row.get("patch_sample_idx", 0)))
        except (TypeError, ValueError):
            sample_idx = 0
        if key:
            existing.add((key, sample_idx))
    return existing


def _patch_prompt_metadata(row: dict[str, Any], patch_text: str, *, truncated: bool, patch_max_chars: int) -> str:
    metadata = {
        "key": row["key"],
        "instance_id": row["instance_id"],
        "class_id": row["class_id"],
        "source_type": row["source_type"],
        "candidate_patch_sha1": hashlib.sha1(patch_text.encode("utf-8", "replace")).hexdigest(),
        "candidate_patch_chars": len(patch_text),
        "candidate_patch_truncated": truncated,
        "candidate_patch_prompt_chars": min(
            len(patch_text),
            patch_max_chars if patch_max_chars > 0 else len(patch_text),
        ),
        "changed_files": _changed_files_from_patch(patch_text),
    }
    return json.dumps(metadata, ensure_ascii=False, indent=2)


def apply_candidate_patch_text(env, patch_text: str, *, timeout: int = 120) -> tuple[bool, str]:
    if not (patch_text or "").strip():
        return False, "empty patch"
    write_file(env, "/tmp/patch_gentest_candidate.diff", patch_text, timeout=30)
    errors = []
    for command in (
        "cd /testbed && git apply --whitespace=nowarn /tmp/patch_gentest_candidate.diff",
        "cd /testbed && git apply --3way --whitespace=nowarn /tmp/patch_gentest_candidate.diff",
        "cd /testbed && patch --batch --fuzz=5 -p1 -i /tmp/patch_gentest_candidate.diff",
    ):
        out = shell(env, command, timeout=timeout)
        text = (out.get("output") or "") + (out.get("exception_info") or "")
        if (out.get("returncode") or 0) == 0:
            return True, ""
        errors.append(text[-800:])
    return False, "\n".join(errors)[-1600:]


def validate_candidate_patch(
    env,
    instance: dict[str, Any],
    code: str,
    *,
    filename: str,
    test_command: str | None = None,
    patch_text: str,
    test_timeout: int,
    apply_test_patch_at_start: bool = False,
    official_verify_format: bool = False,
    official_skip_install: bool | None = None,
) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "has_test": bool((code or "").strip()),
        "candidate_patch_chars": len(patch_text or ""),
        "test_patch": {"enabled": bool(apply_test_patch_at_start)},
    }
    if not validation["has_test"]:
        validation["label"] = "missing_test"
        validation["candidate_pass"] = False
        return validation

    try:
        reset_base(env)
        applied, apply_error = apply_candidate_patch_text(env, patch_text)
        validation["candidate_patch_applied"] = applied
        if not applied:
            validation["candidate_apply_error"] = apply_error
            validation["label"] = "candidate_apply_failure"
            validation["candidate_pass"] = False
            return validation

        if official_verify_format:
            candidate_test_patch = {
                "enabled": bool(apply_test_patch_at_start),
                "phase": "candidate",
                "applied": bool(apply_test_patch_at_start),
                "reason": "",
                "error": "",
                "official_verify_format": True,
            }
        else:
            candidate_test_patch = apply_test_patch_for_alignment(
                env, instance, enabled=apply_test_patch_at_start, phase="candidate"
            )
        validation["test_patch"]["candidate"] = candidate_test_patch
        if candidate_test_patch["reason"]:
            validation["label"] = f"candidate_test_patch_{candidate_test_patch['reason']}"
            validation["candidate_pass"] = False
            validation["infra_failure"] = True
            return validation

        if official_verify_format:
            candidate = run_generated_test_official(
                env,
                instance,
                filename,
                code,
                test_timeout,
                test_command=test_command,
                include_dataset_test_patch=apply_test_patch_at_start,
                skip_install=official_skip_install,
            )
        else:
            write_generated_test_file(env, filename, code)
            candidate = run_generated_test(env, instance, filename, test_timeout)
        candidate.update(infra_failure_info(candidate))
        validation["candidate"] = candidate
        validation["candidate_pass"] = candidate.get("verdict") == "pass"
        validation["candidate_clean_fail"] = bool(candidate.get("clean_fail"))
        if candidate.get("infra_failure"):
            validation["label"] = "candidate_infra_failure"
        elif validation["candidate_pass"]:
            validation["label"] = "candidate_pass"
        elif validation["candidate_clean_fail"]:
            validation["label"] = "candidate_clean_fail"
        else:
            validation["label"] = "candidate_not_pass"
        return validation
    finally:
        try:
            reset_base(env)
        except Exception:
            pass


def patch_error_record(
    instance: dict[str, Any],
    row: dict[str, Any],
    sample_idx: int,
    opts: dict[str, Any],
    error: str,
    traceback_text: str = "",
) -> dict[str, Any]:
    return {
        "key": row.get("key") or row_key(row),
        "instance_id": row.get("instance_id") or instance.get("instance_id", ""),
        "class_id": row.get("class_id", ""),
        "source_type": row.get("source_type", ""),
        "patch_file": row.get("patch_file", ""),
        "oracle_resolved": bool(row.get("oracle_resolved")),
        "sample_idx": int(sample_idx),
        "patch_sample_idx": int(sample_idx),
        "test_filename": generated_test_file_for_instance(instance, opts.get("test_file", TEST_FILE)),
        "requested_test_filename": opts.get("test_file", TEST_FILE),
        "error": error,
        "traceback": traceback_text,
    }


def run_patch_sample_in_env(
    instance: dict[str, Any],
    row: dict[str, Any],
    config: dict[str, Any],
    opts: dict[str, Any],
    env,
    sample_idx: int,
) -> dict[str, Any]:
    instance_id = instance["instance_id"]
    key = row["key"]
    output_dir = Path(opts["output_dir"])
    requested_test_file = opts["test_file"]
    test_file = generated_test_file_for_instance(instance, requested_test_file)
    traj_dir = output_dir / instance_id
    traj_path = traj_dir / f"{_safe_name(key)}.sample_{sample_idx}.traj.json"
    traj_dir.mkdir(parents=True, exist_ok=True)
    condition_on_patch = bool(opts.get("condition_on_patch", True))

    agent = None
    record: dict[str, Any] = {
        "key": key,
        "instance_id": instance_id,
        "class_id": row["class_id"],
        "source_type": row["source_type"],
        "patch_file": row["patch_file"],
        "oracle_resolved": bool(row["oracle_resolved"]),
        "resolved_count": row.get("resolved_count", 0),
        "unresolved_count": row.get("unresolved_count", 0),
        "attempt_count": row.get("attempt_count", 0),
        "sample_idx": int(sample_idx),
        "patch_sample_idx": int(sample_idx),
        "test_filename": test_file,
        "requested_test_filename": requested_test_file,
        "prompt_mode": opts["prompt_mode"],
        "example_shot": opts["example_shot"],
        "starter": opts["starter"],
        "traj_path": str(traj_path),
    }
    try:
        reset_base(env)
        model = get_model(config=config.get("model", {}))
        patch_text = Path(row["patch_file"]).read_text(encoding="utf-8", errors="replace")
        prompt_patch, patch_truncated = _shorten_middle(patch_text, int(opts["patch_max_chars"]))
        patch_sha1 = hashlib.sha1(patch_text.encode("utf-8", "replace")).hexdigest()
        record["candidate_patch_sha1"] = patch_sha1
        record["candidate_patch_chars"] = len(patch_text)
        record["candidate_patch_truncated"] = patch_truncated
        record["candidate_patch_changed_files"] = _changed_files_from_patch(patch_text)

        all_f2p = list_from_json_or_obj(instance.get("FAIL_TO_PASS"))
        chosen = ""
        if all_f2p and (opts["prompt_mode"] == "one_shot_gold" or opts["starter"] == "gold_imports"):
            rng = random.Random(f"{opts['seed']}:{key}:{sample_idx}")
            chosen = str(rng.choice(all_f2p))
        example_node, example_src = extract_example(instance.get("test_patch", "") or "", chosen, opts["example_shot"])
        if opts["prompt_mode"] != "one_shot_gold":
            example_node, example_src = "", ""

        starter_text, starter_imports, starter_status = starter_code_for_sample(env, instance, opts, chosen)
        write_generated_test_file(env, test_file, starter_text)
        record["starter_example_node"] = chosen
        record["starter_status"] = starter_status
        record["starter_imports"] = starter_imports
        record["starter_code_chars"] = len(starter_text)

        agent_config = copy.deepcopy(config.get("agent", {}))
        agent_config.setdefault("agent_class", "generated_test_submit")
        agent_config["output_path"] = traj_path
        agent_config["test_file"] = test_file
        agent_config["allowed_generated_paths"] = [test_file]
        agent_config["self_test_command"] = f"cd /testbed && {test_command_for_instance(instance, test_file)}"
        agent_config["self_test_timeout"] = int(opts["test_timeout"])
        if condition_on_patch:
            agent_config["candidate_patch_text"] = patch_text
            agent_config["candidate_replay_command"] = agent_config["self_test_command"]
            agent_config["candidate_replay_timeout"] = int(opts["test_timeout"])
        agent_config["initial_test_hash"] = hashlib.sha256(starter_text.encode("utf-8", "replace")).hexdigest()
        agent = get_agent(model, ActivatingEnv(env), agent_config, default_type="generated_test_submit")
        run_kwargs = {
            "task": instance.get("problem_statement", ""),
            "instance_id": instance_id,
            "sample_idx": sample_idx,
            "prompt_mode": opts["prompt_mode"],
            "example_node": example_node,
            "example_src": example_src,
            "test_file": test_file,
            "test_style_guidance": test_style_guidance_for_instance(instance, test_file),
            "starter_status": starter_status,
            "starter_imports": starter_imports,
            "starter_code": starter_text,
        }
        if condition_on_patch:
            run_kwargs.update(
                {
                    "patch_key": key,
                    "patch_class_id": row["class_id"],
                    "patch_source_type": row["source_type"],
                    "candidate_patch": prompt_patch,
                    "candidate_patch_metadata": _patch_prompt_metadata(
                        row,
                        patch_text,
                        truncated=patch_truncated,
                        patch_max_chars=int(opts["patch_max_chars"]),
                    ),
                    "candidate_patch_file": row["patch_file"],
                    "candidate_patch_truncated": patch_truncated,
                }
            )
        info = agent.run(**run_kwargs)
        record["agent_exit_status"] = info.get("exit_status", "")
        record["agent_submission_chars"] = len(info.get("submission", "") or "")
        record["agent_calls"] = getattr(agent, "n_calls", 0)
        record["agent_cost"] = round(float(getattr(agent, "cost", 0.0)), 4)
        info_gate = info.get("gentest_gate") if isinstance(info.get("gentest_gate"), dict) else {}
        record["gate"] = getattr(agent, "gate_info", {})
        record["submitted_gate"] = info_gate
        record["candidate_replay"] = info_gate.get("latest_candidate_replay") if isinstance(info_gate, dict) else {}
        record["example_node"] = example_node

        code = read_test_file(env, test_file) or info.get("submission", "")
        oracle_quality = info_gate.get("oracle_quality") if isinstance(info_gate.get("oracle_quality"), dict) else {}
        oracle_contract = oracle_quality.get("contract") if isinstance(oracle_quality.get("contract"), dict) else {}
        record["oracle"] = oracle_contract
        record["oracle_quality"] = evaluate_oracle_contract(oracle_contract, code)
        record["test_code"] = code
        record["validation"] = validate_generated_test(
            env,
            instance,
            code,
            filename=test_file,
            test_timeout=int(opts["test_timeout"]),
            gold_eval=bool(opts["gold_eval"]),
            oracle=oracle_contract,
        )
        record["candidate_validation"] = validate_candidate_patch(
            env,
            instance,
            code,
            filename=test_file,
            patch_text=patch_text,
            test_timeout=int(opts["test_timeout"]),
        )
        record["generated_pred_resolved"] = bool(record["candidate_validation"].get("candidate_pass"))
        record["agreement"] = record["generated_pred_resolved"] == record["oracle_resolved"]
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        record["traceback"] = traceback.format_exc()[-4000:]
    finally:
        try:
            if agent is not None:
                agent.save(
                    traj_path,
                    {
                        "info": {
                            "instance_id": instance_id,
                            "patch_key": key,
                            "sample_idx": sample_idx,
                            "gentest_record": _record_for_traj(record),
                        },
                        "instance_id": instance_id,
                    },
                )
        except Exception:
            pass
    return record


def process_instance_work(
    instance_id: str,
    work_items: list[tuple[dict[str, Any], int]],
    instance: dict[str, Any],
    config: dict[str, Any],
    opts: dict[str, Any],
) -> list[dict[str, Any]]:
    env = None
    try:
        cfg = copy.deepcopy(config)
        _resolve_per_instance_api_base(cfg, instance_id)
        env = get_sb_environment(cfg, instance)
        reset_base(env)
        restore_env(env, instance, int(opts["install_timeout"]))
        return [run_patch_sample_in_env(instance, row, cfg, opts, env, sample_idx) for row, sample_idx in work_items]
    except Exception as exc:  # noqa: BLE001
        traceback_text = traceback.format_exc()[-4000:]
        return [
            patch_error_record(
                instance,
                row,
                sample_idx,
                opts,
                f"{type(exc).__name__}: {str(exc)[:300]}",
                traceback_text,
            )
            for row, sample_idx in work_items
        ]
    finally:
        try:
            if env is not None:
                env.cleanup()
        except Exception:
            pass


def summarize_record(record: dict[str, Any]) -> str:
    validation = record.get("validation") or {}
    candidate = record.get("candidate_validation") or {}
    return (
        f"{record.get('key')}#{record.get('sample_idx')} "
        f"exit={record.get('agent_exit_status')} "
        f"base={validation.get('base_clean_fail')} gold={validation.get('gold_pass')} "
        f"candidate={candidate.get('candidate_pass')} "
        f"pred={record.get('generated_pred_resolved')} oracle={record.get('oracle_resolved')} "
        f"label={validation.get('label')} candidate_label={candidate.get('label')} "
        f"error={record.get('error')}"
    )


def read_result_records(results_jsonl: Path) -> list[dict[str, Any]]:
    records = []
    if not results_jsonl.exists():
        return records
    for line in results_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _count_by(records: list[dict[str, Any]], getter) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(getter(record) or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    tp = sum(1 for r in records if r.get("generated_pred_resolved") and r.get("oracle_resolved"))
    fp = sum(1 for r in records if r.get("generated_pred_resolved") and not r.get("oracle_resolved"))
    tn = sum(1 for r in records if not r.get("generated_pred_resolved") and not r.get("oracle_resolved"))
    fn = sum(1 for r in records if not r.get("generated_pred_resolved") and r.get("oracle_resolved"))
    return {
        "total": total,
        "oracle_pass": sum(1 for r in records if r.get("oracle_resolved")),
        "oracle_fail": sum(1 for r in records if not r.get("oracle_resolved")),
        "generated_pred_pass": sum(1 for r in records if r.get("generated_pred_resolved")),
        "generated_pred_fail": sum(1 for r in records if not r.get("generated_pred_resolved")),
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": 0.5 * (tp / (tp + fn or 1) + tn / (tn + fp or 1)),
        "precision_pass": tp / (tp + fp or 1),
        "recall_pass": tp / (tp + fn or 1),
        "specificity_fail": tn / (tn + fp or 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    strict_gold = [r for r in records if (r.get("validation") or {}).get("label") == "gold_validated"]
    summary = {
        **_classification_metrics(records),
        "validation_label": _count_by(records, lambda r: (r.get("validation") or {}).get("label")),
        "candidate_label": _count_by(records, lambda r: (r.get("candidate_validation") or {}).get("label")),
        "source_type": _count_by(records, lambda r: r.get("source_type")),
        "agent_exit_status": _count_by(records, lambda r: r.get("agent_exit_status")),
        "gold_validated": len(strict_gold),
        "strict_gold_validated": _classification_metrics(strict_gold),
    }
    return summary


# fmt: off
@app.command()
def main(
    patch_groups_dir: Path = typer.Option(
        ...,
        "--patch-groups-dir",
        help="Directory containing groups_with_gold.tsv and patch files",
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output directory for patch-conditioned JSONL results and trajectories",
    ),
    groups_file: Path | None = typer.Option(
        None,
        "--groups-file",
        help="Override groups TSV path; defaults to <patch-groups-dir>/groups_with_gold.tsv",
    ),
    subset: str = typer.Option("verified", "--subset", help="Dataset subset or dataset path"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    ids: Path | None = typer.Option(None, "--ids", help="Optional JSON/JSONL file of instance ids"),
    limit: int = typer.Option(0, "--limit", help="Limit patch rows after filters"),
    workers: int = typer.Option(1, "-w", "--workers", help="Parallel instances"),
    n_samples: int = typer.Option(1, "-n", "--n-samples", help="Samples per patch row"),
    include_gold: bool = typer.Option(
        False,
        "--include-gold",
        help="Also generate tests conditioned on gold patch rows",
    ),
    source_type: list[str] = typer.Option(
        [],
        "--source-type",
        help="Only include these source_type values; repeatable",
    ),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Regenerate records already in results JSONL"),
    results_jsonl: Path | None = typer.Option(
        None,
        "--results-jsonl",
        help="Defaults to <output>/patch_gentest_results.jsonl",
    ),
    summary: Path | None = typer.Option(None, "--summary", help="Defaults to <output>/patch_gentest_summary.json"),
    config_spec: list[str] = typer.Option(
        [str(GENTEST_CONFIG_FILE), str(DEFAULT_CONFIG_FILE)],
        "-c",
        "--config",
        help="mini-swe-agent config specs; defaults to gentest.yaml plus patch_gentest.yaml",
    ),
    model: str | None = typer.Option(None, "-m", "--model", help="Override model name"),
    model_class: str | None = typer.Option(None, "--model-class", help="Override model class"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Override environment class"),
    prompt_mode: str = typer.Option("patch_conditioned", "--prompt-mode", help="patch_conditioned or one_shot_gold"),
    example_shot: str = typer.Option("node", "--example-shot", help="node, full, or none"),
    starter: str = typer.Option("gold_imports", "--starter", help="gold_imports or empty"),
    test_file: str = typer.Option(TEST_FILE, "--test-file", help="Generated test file name"),
    seed: int = typer.Option(0, "--seed", help="Seed for choosing starter/gold examples"),
    gold_eval: bool = typer.Option(True, "--gold-eval/--no-gold-eval", help="Run offline gold-patch validation"),
    test_timeout: int = typer.Option(240, "--test-timeout", help="Generated test timeout"),
    install_timeout: int = typer.Option(600, "--install-timeout", help="Install/restore timeout"),
    patch_max_chars: int = typer.Option(
        DEFAULT_PATCH_MAX_CHARS,
        "--patch-max-chars",
        help="Maximum candidate patch chars shown in prompt; 0 means no truncation",
    ),
    condition_on_patch: bool = typer.Option(
        True,
        "--condition-on-patch/--no-condition-on-patch",
        help="Show the candidate patch to the generator and enable candidate replay feedback",
    ),
) -> None:
    # fmt: on
    output.mkdir(parents=True, exist_ok=True)
    add_file_handler(output / "minisweagent_patch_gentest.log")
    results_path = results_jsonl or (output / "patch_gentest_results.jsonl")
    summary_path = summary or (output / "patch_gentest_summary.json")
    groups_path = groups_file or (patch_groups_dir / DEFAULT_GROUPS_FILE)

    id_list = load_ids(ids) if ids else None
    id_filter = set(id_list) if id_list is not None else None
    rows, missing_patch_files = load_patch_rows(
        groups_path,
        id_filter=id_filter,
        source_types=set(source_type),
        include_gold=include_gold,
        limit=limit,
    )

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path} split={split}")
    ds = {row["instance_id"]: dict(row) for row in load_dataset(dataset_path, split=split)}
    rows = [row for row in rows if row["instance_id"] in ds]

    done = set() if redo_existing else read_existing(results_path)
    work: list[tuple[dict[str, Any], int]] = []
    work_by_instance: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for row in rows:
        for sample_idx in range(n_samples):
            item_key = (row["key"], sample_idx)
            if item_key in done:
                continue
            work.append((row, sample_idx))
            work_by_instance.setdefault(row["instance_id"], []).append((row, sample_idx))

    config = build_config(config_spec, model=model, model_class=model_class, environment_class=environment_class)
    settings = {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset_path,
        "split": split,
        "patch_groups_dir": str(patch_groups_dir),
        "groups_file": str(groups_path),
        "patch_rows": len(rows),
        "missing_patch_files": missing_patch_files,
        "todo_generations": len(work),
        "todo_instances": len(work_by_instance),
        "n_samples": n_samples,
        "include_gold": include_gold,
        "source_type": source_type,
        "prompt_mode": prompt_mode,
        "example_shot": example_shot,
        "starter": starter,
        "test_file": test_file,
        "gold_eval": gold_eval,
        "patch_max_chars": patch_max_chars,
        "condition_on_patch": condition_on_patch,
        "workers": workers,
        "config_spec": config_spec,
    }
    (output / "patch_gentest_settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    print(
        f"dataset={dataset_path} rows={len(rows)} todo={len(work)} existing={len(done)} "
        f"missing_patch_files={missing_patch_files} output={output} results={results_path} "
        f"workers={workers} prompt_mode={prompt_mode} gold_eval={gold_eval}",
        flush=True,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    base_opts = {
        "output_dir": str(output),
        "prompt_mode": prompt_mode,
        "example_shot": example_shot,
        "starter": starter,
        "test_file": test_file,
        "seed": seed,
        "gold_eval": gold_eval,
        "test_timeout": test_timeout,
        "install_timeout": install_timeout,
        "patch_max_chars": patch_max_chars,
        "condition_on_patch": condition_on_patch,
    }
    with results_path.open("a", encoding="utf-8", buffering=1) as handle:
        completed = 0
        if workers <= 1:
            for iid, items in work_by_instance.items():
                records = process_instance_work(iid, items, ds[iid], config, base_opts)
                for record in records:
                    completed += 1
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"[{completed}/{len(work)}] {summarize_record(record)}", flush=True)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_instance_work, iid, items, ds[iid], config, base_opts): iid
                    for iid, items in work_by_instance.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    iid = futures[future]
                    try:
                        records = future.result()
                    except Exception as exc:  # noqa: BLE001
                        records = [
                            patch_error_record(
                                ds[iid],
                                row,
                                sample_idx,
                                base_opts,
                                f"{type(exc).__name__}: {str(exc)[:300]}",
                                traceback.format_exc()[-4000:],
                            )
                            for row, sample_idx in work_by_instance[iid]
                        ]
                    for record in records:
                        completed += 1
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        print(f"[{completed}/{len(work)}] {summarize_record(record)}", flush=True)

    summary_data = summarize_results(read_result_records(results_path))
    summary_data.update(
        {
            "patch_groups_dir": str(patch_groups_dir),
            "groups_file": str(groups_path),
            "results": str(results_path),
            "workers": workers,
            "n_samples": n_samples,
            "include_gold": include_gold,
            "condition_on_patch": condition_on_patch,
        }
    )
    summary_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary_data, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    app()
