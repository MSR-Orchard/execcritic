#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import traceback
from pathlib import Path
from typing import Any

from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks import gentest as gentest_module
from minisweagent.run.benchmarks.swerebench_verify_azure_modal import (
    DEFAULT_CONFIG_FILE,
    create_environment,
    evaluate_instance_in_environment,
    prepare_environment_for_evaluation,
)
from minisweagent.utils.serialize import recursive_merge


DEFAULT_DATASET = Path(
    os.environ.get(
        "GENTEST_PATCH_CLASSIFICATION_DATASET",
        "/data/prompts/patch_classification_train.jsonl",
    )
)


def _load_rows(path: Path, *, limit: int, indices: set[int] | None) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            if indices is not None and idx not in indices:
                continue
            rows.append((idx, json.loads(line)))
            if indices is not None and len(rows) == len(indices):
                break
            if indices is None and limit > 0 and len(rows) >= limit:
                break
    return rows


def _list(value: Any) -> list[str]:
    return [str(item) for item in gentest_module.list_from_json_or_obj(value) if str(item).strip()]


def _official_config_specs(args: argparse.Namespace) -> list[str]:
    if args.config:
        return list(args.config)
    specs = [os.environ.get("GENTEST_VERIFY_CONFIG", str(DEFAULT_CONFIG_FILE))]
    specs.extend(
        spec
        for spec in os.environ.get("GENTEST_VERIFY_CONFIG_SPECS", "").split(";;")
        if spec.strip()
    )
    return specs


def _load_official_env_config(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    specs = _official_config_specs(args)
    config = recursive_merge(*(get_config_from_spec(spec) for spec in specs))
    env_config = copy.deepcopy(config.get("environment") or {})
    env_config["environment_class"] = "azure_modal"
    env_config["block_network"] = False
    env_config["timeout"] = args.test_timeout
    if os.environ.get("SANDBOX_BASE_URL"):
        env_config["base_url"] = os.environ["SANDBOX_BASE_URL"]
    if os.environ.get("SANDBOX_API_KEY"):
        env_config["api_key"] = os.environ["SANDBOX_API_KEY"]
    overrides = {
        "sandbox_timeout": "GENTEST_VERIFY_SANDBOX_TIMEOUT",
        "request_timeout": "GENTEST_VERIFY_REQUEST_TIMEOUT",
        "cleanup_timeout": "GENTEST_VERIFY_CLEANUP_TIMEOUT",
        "cpu": "GENTEST_VERIFY_ENV_CPU",
        "memory": "GENTEST_VERIFY_ENV_MEMORY",
    }
    for key, name in overrides.items():
        if os.environ.get(name):
            value: Any = os.environ[name]
            if key.endswith("timeout"):
                value = int(value)
            env_config[key] = value
    return env_config, specs


def _official_result_is_healthy(result: dict[str, Any], *, expected_f2p: int) -> bool:
    accounted_f2p = len(result.get("fail_to_pass_passed") or []) + len(
        result.get("fail_to_pass_failed") or []
    )
    return bool(
        result.get("patch_applied")
        and not result.get("error")
        and int(result.get("parsed_tests_count") or 0) > 0
        and expected_f2p > 0
        and accounted_f2p >= expected_f2p
    )


def _result_label(*, base_healthy: bool, base_clean_fail: bool, gold_healthy: bool, gold_pass: bool) -> str:
    if not base_healthy:
        return "base_infra_failure"
    if not base_clean_fail:
        return "base_not_clean_fail"
    if not gold_healthy:
        return "gold_infra_failure"
    return "gold_validated" if gold_pass else "gold_failed"


def _run_one(task: tuple[int, dict[str, Any], argparse.Namespace]) -> dict[str, Any]:
    idx, row, args = task
    metadata = copy.deepcopy(row.get("metadata") or row)
    instance_id = metadata.get("instance_id") or row.get("instance_id") or row.get("id") or f"row_{idx}"
    instance = copy.deepcopy(metadata)
    instance.setdefault("instance_id", instance_id)
    env = None
    result: dict[str, Any] = {
        "idx": idx,
        "instance_id": instance_id,
        "repo": metadata.get("repo"),
        "base_commit": metadata.get("base_commit"),
        "status": "error",
        "error": "",
        "fail_to_pass": _list(metadata.get("FAIL_TO_PASS")),
        "test_patch_chars": len(metadata.get("test_patch") or ""),
        "gold_patch_chars": len(metadata.get("patch") or ""),
    }
    try:
        fail_to_pass = _list(instance.get("FAIL_TO_PASS"))
        result["selected_nodes"] = fail_to_pass
        if not fail_to_pass:
            result["status"] = "skipped"
            result["error"] = "no FAIL_TO_PASS nodes"
            return result
        install_config = instance.get("install_config") or {}
        result["command_info"] = {
            "mode": "canonical_official_eval_script",
            "test_cmd": install_config.get("test_cmd") if isinstance(install_config, dict) else None,
            "log_parser": install_config.get("log_parser") if isinstance(install_config, dict) else None,
            "fail_to_pass_count": len(fail_to_pass),
            "pass_to_pass_count": len(_list(instance.get("PASS_TO_PASS"))),
        }

        env_config, config_specs = _load_official_env_config(args)
        result["official_verifier"] = {
            "config_specs": config_specs,
            "environment_class": env_config.get("environment_class"),
            "block_network": env_config.get("block_network"),
            "setup_once": True,
            "environment_prepared": True,
        }
        env = create_environment(instance, env_config)
        setup = prepare_environment_for_evaluation(env, instance, timeout=args.install_timeout)
        result["official_verifier"]["setup"] = setup

        base_run = evaluate_instance_in_environment(
            instance,
            "",
            env,
            env_config,
            environment_prepared=True,
        )
        gold_run = evaluate_instance_in_environment(
            instance,
            instance.get("patch", "") or "",
            env,
            env_config,
            environment_prepared=True,
        )
        base_healthy = _official_result_is_healthy(base_run, expected_f2p=len(fail_to_pass))
        gold_healthy = _official_result_is_healthy(gold_run, expected_f2p=len(fail_to_pass))
        base_clean_fail = bool(
            base_healthy
            and not base_run.get("resolved")
            and base_run.get("fail_to_pass_failed")
        )
        gold_pass = bool(gold_healthy and gold_run.get("resolved"))
        result.update(
            {
                "base": base_run,
                "gold": gold_run,
                "base_healthy": base_healthy,
                "gold_healthy": gold_healthy,
                "base_clean_fail": base_clean_fail,
                "base_pass": bool(base_healthy and base_run.get("resolved")),
                "gold_pass": gold_pass,
                "f2p_gold_validated": bool(base_clean_fail and gold_pass),
                "validation_label": _result_label(
                    base_healthy=base_healthy,
                    base_clean_fail=base_clean_fail,
                    gold_healthy=gold_healthy,
                    gold_pass=gold_pass,
                ),
            }
        )
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        result["traceback"] = traceback.format_exc()[-3000:]
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception:
                pass
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    return {
        "samples": len(rows),
        "ok": len(ok),
        "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
        "errors": sum(1 for row in rows if row.get("status") == "error"),
        "base_clean_fail": sum(1 for row in ok if row.get("base_clean_fail") is True),
        "base_infra_failure": sum(1 for row in ok if not row.get("base_healthy")),
        "base_fail": sum(1 for row in ok if row.get("base_clean_fail") is True),
        "base_pass": sum(1 for row in ok if row.get("base_pass") is True),
        "gold_pass": sum(1 for row in ok if row.get("gold_pass") is True),
        "gold_fail": sum(1 for row in ok if row.get("gold_pass") is False),
        "gold_infra_failure": sum(1 for row in ok if not row.get("gold_healthy")),
        "f2p_gold_validated": sum(1 for row in ok if row.get("f2p_gold_validated") is True),
        "labels": {
            label: sum(1 for row in ok if row.get("validation_label") == label)
            for label in (
                "base_infra_failure",
                "base_not_clean_fail",
                "gold_infra_failure",
                "gold_failed",
                "gold_validated",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--indices", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Deprecated; canonical official verification always grades the full F2P set.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Official verifier config/spec; repeat to merge multiple specs.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--test-timeout", type=int, default=int(os.environ.get("SWE_PATCH_CLASSIFICATION_TEST_TIMEOUT", "240")))
    parser.add_argument("--install-timeout", type=int, default=int(os.environ.get("SWE_PATCH_CLASSIFICATION_INSTALL_TIMEOUT", "600")))
    parser.add_argument(
        "--environment-class",
        default="azure_modal",
        help="Compatibility option; only azure_modal is accepted by this official smoke.",
    )
    args = parser.parse_args()

    if args.max_nodes != 0:
        parser.error("--max-nodes is incompatible with canonical full official F2P verification")
    if args.environment_class != "azure_modal":
        parser.error("canonical official F2P verification requires --environment-class azure_modal")
    if args.install_timeout <= 0:
        parser.error("--install-timeout must be positive; official setup is mandatory")

    indices = {int(item) for item in args.indices.split(",") if item.strip()} if args.indices.strip() else None
    rows = _load_rows(args.dataset, limit=args.limit, indices=indices)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with args.out.open("w", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_one, (idx, row, args)) for idx, row in rows]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "idx": result.get("idx"),
                            "instance_id": result.get("instance_id"),
                            "status": result.get("status"),
                            "base_pass": result.get("base_pass"),
                            "base_clean_fail": result.get("base_clean_fail"),
                            "gold_pass": result.get("gold_pass"),
                            "f2p_gold_validated": result.get("f2p_gold_validated"),
                            "validation_label": result.get("validation_label"),
                            "error": result.get("error"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    summary = _summarize(results)
    summary.update({"dataset": str(args.dataset), "out": str(args.out), "indices": args.indices, "limit": args.limit})
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
