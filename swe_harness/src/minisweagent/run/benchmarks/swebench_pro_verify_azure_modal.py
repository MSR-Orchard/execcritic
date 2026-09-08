#!/usr/bin/env python3

"""Evaluate SWE-bench Pro patches in isolated Azure Modal sandboxes.

The benchmark's per-instance ``run_script.sh`` and ``parser.py`` files remain
the source of truth.  A focused repair gate changes only the selected test
targets; final verification uses the dataset's complete selected-test list.
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import os
import random
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Any


DOCKER_WORKDIR = "/app"
DATASET_NAME = "ScaleAI/SWE-bench_Pro"
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
DEFAULT_HARNESS_ROOT = Path("swe_harness/SWE-bench_Pro-os")
_B64WRITE_CHUNK_SIZE = 48_000
_BOOT_SEMAPHORE = threading.Semaphore(int(os.environ.get("SWEPRO_BOOT_CONCURRENCY", "8")))
logger = logging.getLogger(__name__)


def create_sandbox_with_retry(factory, instance_id: str):
    """Rate-limit sandbox boots and retry transient orchestrator failures."""
    attempts = int(os.environ.get("SWEPRO_BOOT_RETRIES", "5"))
    for attempt in range(1, attempts + 1):
        try:
            with _BOOT_SEMAPHORE:
                return factory()
        except Exception as exc:
            if attempt >= attempts:
                raise
            delay = min(30.0, 2.0**attempt) + random.random() * 2.0
            logger.warning(
                "%s: sandbox boot failed (%d/%d): %s; retrying in %.1fs",
                instance_id,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"{instance_id}: sandbox creation exhausted retries")


def _from_json_or_obj(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def prepare_instance(instance: dict) -> dict:
    """Normalize one public SWE-bench Pro row for the shared repair harness."""
    prepared = dict(instance)
    if not prepared.get("_swebench_pro_prompt_prepared"):
        prepared["problem_statement"] = (
            f"{prepared['problem_statement']}\n\n"
            f"Requirements:\n{prepared['requirements']}\n\n"
            f"New interfaces introduced:\n{prepared['interface']}"
        )
        prepared["_swebench_pro_prompt_prepared"] = True
    prepared["FAIL_TO_PASS"] = _from_json_or_obj(prepared.get("FAIL_TO_PASS", prepared.get("fail_to_pass")))
    prepared["PASS_TO_PASS"] = _from_json_or_obj(prepared.get("PASS_TO_PASS", prepared.get("pass_to_pass")))
    prepared["selected_test_files_to_run"] = _from_json_or_obj(prepared.get("selected_test_files_to_run"))
    docker_tag = str(prepared.get("dockerhub_tag") or "").strip()
    if not docker_tag:
        raise ValueError(f"{prepared.get('instance_id', '<unknown>')}: dockerhub_tag is missing")
    prepared["image_name"] = f"mirror.gcr.io/jefzda/sweap-images:{docker_tag}"
    prepared["_swebench_pro"] = True
    return prepared


def get_harness_root() -> Path:
    """Return the checkout containing the official per-instance run scripts."""
    return Path(os.environ.get("SWEPRO_HARNESS_ROOT", str(DEFAULT_HARNESS_ROOT))).expanduser()


def validate_harness_root(root: Path | None = None) -> Path:
    """Validate a SWE-bench Pro OS checkout before starting a batch."""
    root = (root or get_harness_root()).expanduser()
    if not (root / "run_scripts").is_dir():
        raise FileNotFoundError(
            f"SWE-bench Pro run_scripts not found under {root}; "
            "clone https://github.com/scaleapi/SWE-bench_Pro-os and set "
            "SWEPRO_HARNESS_ROOT or --swebench-pro-harness-root"
        )
    return root


def _harness_files(instance_id: str) -> tuple[str, str]:
    root = get_harness_root()
    instance_dir = root / "run_scripts" / instance_id
    run_script = instance_dir / "run_script.sh"
    parser_script = instance_dir / "parser.py"
    if not run_script.is_file() or not parser_script.is_file():
        raise FileNotFoundError(f"missing SWE-bench Pro scripts under {instance_dir}")
    return run_script.read_text(), parser_script.read_text()


def strip_binary_hunks(patch: str) -> str:
    """Drop binary diff sections, matching the official Pro evaluator."""
    sections = re.split(r"(?=^diff --git )", patch or "", flags=re.MULTILINE)
    return "".join(
        section
        for section in sections
        if section.strip()
        and not re.search(r"^Binary files .* differ$", section, re.MULTILINE)
        and not re.search(r"^GIT binary patch$", section, re.MULTILINE)
    )


def _b64write_cmd(path: str, content: bytes, mode: str) -> str:
    encoded = base64.b64encode(content).decode()
    return f"python3 -c \"import base64; open({path!r},{mode!r}).write(base64.b64decode({encoded!r}))\""


def _upload_file(env, path: str, content: str, *, timeout: int = 60) -> None:
    raw = content.encode()
    env.execute({"command": f"python3 -c \"open({path!r},'wb').close()\""}, cwd=DOCKER_WORKDIR, timeout=30)
    for offset in range(0, len(raw), _B64WRITE_CHUNK_SIZE):
        chunk = raw[offset : offset + _B64WRITE_CHUNK_SIZE]
        result = env.execute(
            {"command": _b64write_cmd(path, chunk, "ab")},
            cwd=DOCKER_WORKDIR,
            timeout=timeout,
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"failed to upload {path}: {result.get('exception_info') or result.get('output')}")


def _last_before_repo_command(instance: dict) -> str:
    commands = [line.strip() for line in str(instance.get("before_repo_set_cmd") or "").splitlines() if line.strip()]
    return commands[-1] if commands else ":"


def _selected_targets(instance: dict) -> list[str]:
    focused = instance.get("_swebench_pro_selected_nodes")
    if focused is not None:
        return _from_json_or_obj(focused)
    return _from_json_or_obj(instance.get("selected_test_files_to_run"))


def _grade_results(instance: dict, parsed: dict[str, str]) -> dict:
    fail_to_pass = _from_json_or_obj(instance.get("FAIL_TO_PASS", instance.get("fail_to_pass")))
    pass_to_pass = _from_json_or_obj(instance.get("PASS_TO_PASS", instance.get("pass_to_pass")))
    passed = {name for name, status in parsed.items() if status == "PASSED"}
    f2p_passed = sorted(set(fail_to_pass) & passed)
    f2p_failed = sorted(set(fail_to_pass) - passed)
    p2p_passed = sorted(set(pass_to_pass) & passed)
    p2p_failed = sorted(set(pass_to_pass) - passed)
    return {
        "resolved": bool(fail_to_pass) and not f2p_failed and not p2p_failed,
        "fail_to_pass_passed": f2p_passed,
        "fail_to_pass_failed": f2p_failed,
        "pass_to_pass_passed": len(p2p_passed),
        "pass_to_pass_failed": p2p_failed,
        "parsed_tests_count": len(parsed),
    }


def create_environment(instance: dict, env_config: dict):
    """Create, but do not evaluate or clean up, one official verifier sandbox."""
    from minisweagent.environments.extra.azure_modal import AzureModalEnvironment

    instance = prepare_instance(instance)
    cfg = {key: value for key, value in env_config.items() if key != "environment_class"}
    cfg.update(image=instance["image_name"], cwd=DOCKER_WORKDIR)
    cfg.setdefault("cpu", "4")
    cfg.setdefault("memory", "16Gi")
    cfg.setdefault("block_network", False)
    return create_sandbox_with_retry(
        lambda: AzureModalEnvironment(**cfg), str(instance["instance_id"])
    )


def evaluate_instance_in_environment(
    instance: dict,
    patch: str,
    env,
    env_config: dict,
    *,
    logs_dir: Path | None = None,
    sample_idx: int = 0,
) -> dict:
    """Run one focused or full SWE-bench Pro grade in a persistent sandbox."""
    instance = prepare_instance(instance)
    instance_id = str(instance["instance_id"])
    run_script, parser_script = _harness_files(instance_id)
    targets = _selected_targets(instance)
    cfg = {key: value for key, value in env_config.items() if key != "environment_class"}

    output_parts: list[str] = []
    test_exit_code = -1
    patch_applied = not bool((patch or "").strip())
    parsed: dict[str, str] = {}
    error = ""
    try:
        cleaned_patch = strip_binary_hunks(patch or "")
        if cleaned_patch != (patch or ""):
            logger.info("%s: stripped binary diff hunks before evaluation", instance_id)
        _upload_file(env, "/tmp/swepro_patch.diff", cleaned_patch)
        _upload_file(env, "/tmp/swepro_run.sh", run_script)
        _upload_file(env, "/tmp/swepro_parser.py", parser_script)

        base_commit = shlex.quote(
            str(instance.get("_persistent_baseline_commit") or instance["base_commit"])
        )
        reset = env.execute(
            {"command": f"git reset --hard {base_commit} && git clean -fdq"},
            cwd=DOCKER_WORKDIR,
            timeout=120,
        )
        output_parts.append(reset.get("output", ""))
        if reset["returncode"] != 0:
            error = "base reset failed"

        if not error and not patch_applied:
            applied = env.execute(
                {"command": "git apply -v /tmp/swepro_patch.diff"},
                cwd=DOCKER_WORKDIR,
                timeout=120,
            )
            output_parts.append(applied.get("output", ""))
            patch_applied = applied["returncode"] == 0
            if not patch_applied:
                error = "model patch failed to apply"

        if not error:
            before = env.execute(
                {"command": _last_before_repo_command(instance)},
                cwd=DOCKER_WORKDIR,
                timeout=300,
            )
            output_parts.append(before.get("output", ""))
            if before["returncode"] != 0:
                error = "before_repo_set_cmd failed"

        if not error:
            target_arg = shlex.quote(",".join(targets))
            test_result = env.execute(
                {
                    "command": (
                        f"echo {shlex.quote(START_TEST_OUTPUT)}; "
                        f"bash /tmp/swepro_run.sh {target_arg} > /tmp/swepro_stdout.log "
                        "2> /tmp/swepro_stderr.log; test_rc=$?; "
                        "python3 /tmp/swepro_parser.py /tmp/swepro_stdout.log "
                        "/tmp/swepro_stderr.log /tmp/swepro_result.json; parser_rc=$?; "
                        f"echo {shlex.quote(END_TEST_OUTPUT)}; "
                        "echo __SWEPRO_TEST_RC__=$test_rc; echo __SWEPRO_PARSER_RC__=$parser_rc"
                    )
                },
                cwd=DOCKER_WORKDIR,
                timeout=int(cfg.get("sandbox_timeout", 900)),
            )
            output_parts.append(test_result.get("output", ""))
            stdout_result = env.execute(
                {"command": "cat /tmp/swepro_stdout.log; cat /tmp/swepro_stderr.log >&2"},
                cwd=DOCKER_WORKDIR,
                timeout=60,
            )
            output_parts.append(stdout_result.get("output", ""))
            result_json = env.execute(
                {"command": "cat /tmp/swepro_result.json"},
                cwd=DOCKER_WORKDIR,
                timeout=60,
            )
            try:
                parsed_payload = json.loads(result_json.get("output", ""))
                parsed = {
                    str(item["name"]): str(item["status"])
                    for item in parsed_payload.get("tests", [])
                    if item.get("name") and item.get("status")
                }
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                error = f"parser output invalid: {type(exc).__name__}: {exc}"
            match = re.search(r"__SWEPRO_TEST_RC__=(-?\d+)", test_result.get("output", ""))
            test_exit_code = int(match.group(1)) if match else test_result["returncode"]
            output_parts.append("\n".join(f"{status} {name}" for name, status in parsed.items()))
    finally:
        try:
            env.execute(
                {
                    "command": (
                        "rm -f /tmp/swepro_patch.diff /tmp/swepro_run.sh "
                        "/tmp/swepro_parser.py /tmp/swepro_stdout.log "
                        "/tmp/swepro_stderr.log /tmp/swepro_result.json"
                    )
                },
                cwd=DOCKER_WORKDIR,
                timeout=30,
            )
        except Exception as exc:  # cleanup must not replace the evaluator result
            logger.warning("%s: transient verifier cleanup failed: %s", instance_id, exc)

    output = "\n".join(part for part in output_parts if part)
    if logs_dir:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
        (logs_dir / f"{instance_id}{suffix}_log.txt").write_text(output, errors="replace")

    grade = _grade_results(instance, parsed)
    if error:
        grade["resolved"] = False
    return {
        "instance_id": instance_id,
        **grade,
        "exit_code": test_exit_code,
        "patch_applied": patch_applied,
        "error": error,
    }


def evaluate_instance_azure_modal(
    instance: dict,
    patch: str,
    env_config: dict,
    *,
    logs_dir: Path | None = None,
    sample_idx: int = 0,
) -> dict:
    """Run one focused or full SWE-bench Pro grade in a fresh sandbox."""
    env = create_environment(instance, env_config)
    try:
        return evaluate_instance_in_environment(
            instance,
            patch,
            env,
            env_config,
            logs_dir=logs_dir,
            sample_idx=sample_idx,
        )
    finally:
        env.cleanup()
