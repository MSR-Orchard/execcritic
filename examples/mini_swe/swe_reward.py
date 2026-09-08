"""
Reward computation for SWE-bench tasks.

This module provides multiple reward strategies based on SWE-bench evaluation:
1. Binary: 1.0 if tests pass (FULL resolution), 0.0 otherwise
2. Multi-objective: Dict with multiple reward components
3. Shaped: Intermediate rewards for progress (optional)

The evaluation follows SWE-bench grading logic:
- Fail-to-Pass (F2P): Tests that should change from failing to passing
- Pass-to-Pass (P2P): Tests that should remain passing (maintenance)
- Resolution: FULL (F2P=1.0 & P2P=1.0), PARTIAL (0<F2P<1 & P2P=1.0), NO (otherwise)
"""

import asyncio
import base64
import copy
import csv
import hashlib
import json
import logging
import os
import re
import shlex
import sys
import subprocess
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from slime.utils.types import Sample

from .swe_wrapper import (
    create_environment,
    get_docker_image_name,
    load_config,
    stop_environment,
)

# Import SWE-bench harness components for proper evaluation
try:
    from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
    from swebench.harness.constants import LATEST, SWEbenchInstance
    from swebench.harness.grading import get_eval_report
    from swebench.harness.constants import (
        KEY_INSTANCE_ID,
        KEY_MODEL,
        KEY_PREDICTION,
        MAP_REPO_VERSION_TO_SPECS,
        MAP_REPO_TO_EXT,
        START_TEST_OUTPUT,
        END_TEST_OUTPUT,
    )
    from swebench.harness.test_spec.create_scripts import make_eval_script_list
    from swebench.harness.log_parsers.python import parse_log_pytest, parse_log_pytest_v2
    HAS_SWEBENCH_HARNESS = True
except ImportError:
    HAS_SWEBENCH_HARNESS = False
    LATEST = "latest"
    SWEbenchInstance = dict
    MAP_REPO_VERSION_TO_SPECS = {}
    MAP_REPO_TO_EXT = {}
    make_test_spec = None
    TestSpec = None
    get_eval_report = None
    make_eval_script_list = None
    parse_log_pytest = None
    parse_log_pytest_v2 = None
    KEY_INSTANCE_ID = "instance_id"
    KEY_MODEL = "model_name_or_path"
    KEY_PREDICTION = "model_patch"
    START_TEST_OUTPUT = ">>>>> Start Test Output"
    END_TEST_OUTPUT = ">>>>> End Test Output"

SWE_TIMEOUT_REWARD_CREATE_ENV = int(os.environ.get("SWE_TIMEOUT_CREATE_ENV", "480"))
SWE_TIMEOUT_REWARD_EXECUTE = int(os.environ.get("SWE_TIMEOUT_REWARD_EXECUTE", "360"))
SWE_TIMEOUT_REWARD_TOTAL = int(os.environ.get("SWE_TIMEOUT_REWARD_TOTAL", "1200"))
SWE_TIMEOUT_REWARD_STOP_ENV = int(os.environ.get("SWE_TIMEOUT_STOP_ENV", "120"))

_REWARD_EXECUTORS: dict[int, ThreadPoolExecutor] = {}


def _reward_worker_count(args) -> int:
    configured = os.environ.get("SWE_REWARD_WORKERS", "").strip()
    if configured:
        workers = int(configured)
        if workers <= 0:
            raise ValueError(f"SWE_REWARD_WORKERS must be positive, got {workers}")
        return workers

    rollout_batch_size = max(1, int(getattr(args, "rollout_batch_size", 1) or 1))
    samples_per_prompt = max(1, int(getattr(args, "n_samples_per_prompt", 1) or 1))
    return rollout_batch_size * samples_per_prompt


def _reward_executor(args) -> ThreadPoolExecutor:
    workers = _reward_worker_count(args)
    executor = _REWARD_EXECUTORS.get(workers)
    if executor is None:
        logger.info("[SWE_REWARD] creating dedicated reward executor workers=%s", workers)
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="swe-reward")
        _REWARD_EXECUTORS[workers] = executor
    return executor


async def _run_cleanup_owned_sync(executor: ThreadPoolExecutor, function, *args):
    """Do not detach a sandbox-owning worker when its caller is cancelled."""
    task = asyncio.get_running_loop().run_in_executor(executor, function, *args)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            logger.exception("Sandbox-owning reward worker failed while cancellation waited for cleanup")
        raise

# ==============================================================================
# Constants and Enums (from SWE-bench)
# ==============================================================================

class ResolvedStatus(Enum):
    """Resolution status for an instance."""
    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"


# Test category constants
FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"


# ==============================================================================
# Utility Functions
# ==============================================================================

def extract_patch_from_output(text: str) -> str:
    """
    Extract patch/diff from final output.

    Args:
        text: Final output text that may contain a git diff or patch

    Returns:
        Extracted patch string
    """
    # Look for git diff format
    diff_pattern = r"```diff\s*\n(.*?)\n```"
    diff_match = re.search(diff_pattern, text, re.DOTALL)
    if diff_match:
        return _with_patch_trailing_newline(diff_match.group(1))

    # Look for patch format
    patch_pattern = r"```patch\s*\n(.*?)\n```"
    patch_match = re.search(patch_pattern, text, re.DOTALL)
    if patch_match:
        return _with_patch_trailing_newline(patch_match.group(1))

    # Look for plain code blocks
    code_pattern = r"```\s*\n(.*?)\n```"
    code_match = re.search(code_pattern, text, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip("\r\n")
        # Check if it looks like a diff
        if content.startswith("diff ") or "@@" in content or content.startswith("---"):
            return _with_patch_trailing_newline(content)

    # If no code block, check if the whole text looks like a diff
    if text.startswith("diff ") or "@@" in text or text.startswith("---"):
        return _with_patch_trailing_newline(text)

    return ""


def _with_patch_trailing_newline(patch: str) -> str:
    """Remove fence-adjacent blank lines without stripping diff content."""
    patch = patch.strip("\r\n")
    return f"{patch}\n" if patch else ""


def is_valid_patch(patch: str) -> bool:
    """
    Check if a patch has valid syntax.

    Args:
        patch: Patch string

    Returns:
        True if patch looks valid
    """
    if not patch:
        return False

    # Check for basic diff markers
    has_diff_markers = any(
        marker in patch
        for marker in ["diff --git", "---", "+++", "@@", "Index:", "==="]
    )

    return has_diff_markers


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_swe_harness_on_path() -> None:
    harness_src = _repo_root() / "swe_harness" / "src"
    if harness_src.exists() and str(harness_src) not in sys.path:
        sys.path.insert(0, str(harness_src))
    azure_modal_client = _repo_root() / "swe_harness" / "external" / "azure-modal"
    if azure_modal_client.exists() and str(azure_modal_client) not in sys.path:
        sys.path.insert(0, str(azure_modal_client))


def _coerce_resolved_label(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "resolved", "pass", "passed", "success"}:
            return True
        if text in {"0", "false", "no", "n", "failed", "fail", "unresolved", "error"}:
            return False
    return None


def _strip_first_code_block(text: str) -> str:
    text = text or ""
    fenced = re.search(r"```(?:python|py|pytest|test)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def extract_generated_test_code_from_sample(sample: Sample) -> str:
    """Extract generated test code from common mini-SWE / gentest sample fields."""
    metadata = sample.metadata or {}
    keys = (
        "test_code",
        "generated_test_code",
        "generated_test",
        "submission",
        "final_output",
    )
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, dict):
            value = value.get("test_code") or value.get("submission") or value.get("content")
        if isinstance(value, str) and value.strip():
            return _strip_first_code_block(value)
    if isinstance(sample.response, str) and sample.response.strip():
        return _strip_first_code_block(sample.response)
    return ""


def _read_patch_text_from_candidate(row: dict[str, Any], *, base_dir: Path | None = None) -> str:
    for key in ("patch", "patch_text", "model_patch", "submission", "candidate_patch"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    patch_file = row.get("patch_file") or row.get("path")
    if isinstance(patch_file, str) and patch_file.strip():
        path = Path(patch_file)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _candidate_oracle_resolved(row: dict[str, Any]) -> bool | None:
    for key in ("oracle_resolved", "resolved", "actual_verify_resolved", "label", "verify_status", "actual_verify_status"):
        label = _coerce_resolved_label(row.get(key))
        if label is not None:
            return label
    if row.get("source_type") == "gold":
        return True
    try:
        if int(row.get("resolved_count") or 0) > 0:
            return True
        if int(row.get("unresolved_count") or 0) > 0:
            return False
    except (TypeError, ValueError):
        return None
    return None


def _normalize_patch_candidate(
    row: dict[str, Any],
    *,
    instance_id: str,
    base_dir: Path | None = None,
) -> dict[str, Any] | None:
    row_instance_id = row.get("instance_id")
    if row_instance_id and instance_id and row_instance_id != instance_id:
        return None

    patch_text = _read_patch_text_from_candidate(row, base_dir=base_dir)
    if not patch_text.strip():
        return None

    oracle_resolved = _candidate_oracle_resolved(row)
    if oracle_resolved is None:
        return None

    patch_sha256 = row.get("patch_sha256") or hashlib.sha256(patch_text.encode("utf-8", "replace")).hexdigest()
    return {
        "instance_id": row_instance_id or instance_id,
        "sample_idx": row.get("sample_idx"),
        "run_dir": row.get("run_dir"),
        "source_path": row.get("source_path") or row.get("patch_file") or row.get("path"),
        "patch_sha256": patch_sha256,
        "patch_chars": len(patch_text),
        "patch_text": patch_text,
        "oracle_resolved": bool(oracle_resolved),
    }


def _load_patch_candidates_from_samples_jsonl(metadata: dict[str, Any], instance_id: str) -> list[dict[str, Any]]:
    path_value = (
        metadata.get("patch_classification_samples_jsonl")
        or metadata.get("patch_samples_jsonl")
        or metadata.get("all_rebench_patch_samples_jsonl")
    )
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"patch classification samples jsonl not found: {path}")

    target_sha = (
        metadata.get("patch_sha256")
        or metadata.get("candidate_patch_sha256")
        or metadata.get("patch_classification_patch_sha256")
    )
    candidates = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("instance_id") != instance_id:
                continue
            if target_sha and row.get("patch_sha256") != target_sha:
                continue
            candidate = _normalize_patch_candidate(row, instance_id=instance_id, base_dir=path.parent)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _load_patch_candidates_from_groups_file(metadata: dict[str, Any], instance_id: str) -> list[dict[str, Any]]:
    groups_file = (
        metadata.get("patch_classification_groups_file")
        or metadata.get("patch_groups_file")
        or metadata.get("groups_file")
    )
    if not groups_file:
        return []

    path = Path(str(groups_file))
    if not path.exists():
        raise FileNotFoundError(f"patch groups file not found: {path}")

    source_types = metadata.get("patch_classification_source_types") or []
    if isinstance(source_types, str):
        source_types = [item.strip() for item in source_types.split(",") if item.strip()]

    candidates = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("instance_id") != instance_id:
                continue
            if source_types and row.get("source_type") not in set(source_types):
                continue
            candidate = _normalize_patch_candidate(row, instance_id=instance_id, base_dir=path.parent)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def load_patch_classification_candidates(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Load labeled patch candidates for generated-test binary classification reward."""
    instance_id = metadata.get("instance_id", "")
    candidates: list[dict[str, Any]] = []

    inline_rows = (
        metadata.get("patch_classification_patches")
        or metadata.get("patch_candidates")
        or metadata.get("candidate_patches")
        or metadata.get("patches")
        or []
    )
    if isinstance(inline_rows, str):
        try:
            inline_rows = json.loads(inline_rows)
        except json.JSONDecodeError:
            inline_rows = []
    if isinstance(inline_rows, dict):
        inline_rows = [inline_rows]
    for row in inline_rows:
        if isinstance(row, str):
            row = {"patch": row}
        if not isinstance(row, dict):
            continue
        candidate = _normalize_patch_candidate(row, instance_id=instance_id)
        if candidate is not None:
            candidates.append(candidate)

    candidates.extend(_load_patch_candidates_from_groups_file(metadata, instance_id))
    candidates.extend(_load_patch_candidates_from_samples_jsonl(metadata, instance_id))
    return candidates


def patch_classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
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


# ==============================================================================
# Docker Test Execution
# ==============================================================================



async def run_tests_in_docker(
    instance_id: str,
    patch: str,
    repo: str,
    base_commit: str,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    instance: dict | None = None,
    config: dict | None = None,
) -> dict:
    """
    Run SWE-bench tests in a Docker container and evaluate results.

    This function follows the same evaluation logic as run_instance_modal() from
    swebench.harness.modal_eval:
    1. Creates a TestSpec from the instance using make_test_spec()
    2. Applies the model's patch using git apply (with fallback to patch command)
    3. Runs the eval_script from the TestSpec
    4. Uses get_eval_report() for proper SWE-bench grading

    Args:
        instance_id: SWE-bench instance ID
        patch: The generated patch to test (model_patch)
        repo: Repository name
        base_commit: Base commit hash
        fail_to_pass: List of tests that should change from failing to passing
        pass_to_pass: List of tests that should remain passing
        instance: Full instance dict for TestSpec creation and Docker image resolution
        config: Optional environment config (loads default if not provided)

    Returns:
        Dict with test results including:
        - passed: bool (True if FULL resolution)
        - resolved: bool (same as passed)
        - resolution: str (FULL, PARTIAL, or NO)
        - fail_to_pass_rate: float
        - pass_to_pass_rate: float
        - tests_run: int
        - error: str
        - output: str
    """
    result = {
        "passed": False,
        "resolved": False,
        "resolution": ResolvedStatus.NO.value,
        "fail_to_pass_rate": 0.0,
        "pass_to_pass_rate": 0.0,
        "tests_run": 0,
        "tests_passed": 0,
        "f2p_total": 0,
        "f2p_passed": 0,
        "p2p_total": 0,
        "p2p_passed": 0,
        "error": "",
        "output": "",
    }

    # Validate patch first
    if not is_valid_patch(patch):
        result["error"] = "Invalid patch format"
        return result

    # Set defaults for test lists
    if fail_to_pass is None:
        fail_to_pass = []
    if pass_to_pass is None:
        pass_to_pass = []

    # Build instance dict for Docker image resolution if not provided
    if instance is None:
        instance = {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
        }

    use_official_swerebench = uses_official_swerebench_verifier(instance)

    # Standard SWE-bench and Scale-SWE still use the existing harness path.
    if not use_official_swerebench and not HAS_SWEBENCH_HARNESS:
        result["error"] = "Error: SWE-bench harness is required but not available. Please install swebench package."
        return result

    # Load config if not provided
    if config is None:
        config_path = os.environ["SWE_CONFIG_PATH"]
        try:
            config = load_config(config_path)
        except Exception as e:
            result["error"] = f"Failed to load config: {e}"
            return result

    MAX_ENV_UNAVAILABLE_RETRIES = int(os.environ.get("SWE_MAX_ENV_RETRIES", "2"))
    ENV_UNAVAILABLE_RETRY_WAIT = int(os.environ.get("SWE_ENV_RETRY_WAIT", "10"))

    resource_level = 1
    for env_attempt in range(1, MAX_ENV_UNAVAILABLE_RETRIES + 1):
        attempt_result = dict(result)  # Fresh copy for each attempt
        try:
            if use_official_swerebench:
                return await _run_official_swerebench_verifier(
                    instance=instance,
                    patch=patch,
                    config=config,
                    result=attempt_result,
                    fail_to_pass=fail_to_pass,
                    pass_to_pass=pass_to_pass,
                    resource_level=resource_level,
                )
            return await _run_tests_with_swebench_harness(
                instance=instance,
                patch=patch,
                config=config,
                result=attempt_result,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                resource_level=resource_level,
            )
        except EnvironmentUnavailable as e:
            logger.error(
                f"[SWE_REWARD] EnvironmentUnavailable during reward evaluation "
                f"(attempt {env_attempt}/{MAX_ENV_UNAVAILABLE_RETRIES}): {e}"
            )
            resource_level += 1
            if env_attempt < MAX_ENV_UNAVAILABLE_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying reward evaluation in {ENV_UNAVAILABLE_RETRY_WAIT}s..."
                )
                await asyncio.sleep(ENV_UNAVAILABLE_RETRY_WAIT)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_ENV_UNAVAILABLE_RETRIES} retry attempts exhausted"
                )
                attempt_result["error"] = (
                    f"Environment unavailable after {MAX_ENV_UNAVAILABLE_RETRIES} attempts: {e}"
                )
                return attempt_result
        except Exception as e:
            logger.error(
                f"[SWE_REWARD] Unexpected error during reward evaluation for "
                f"instance_id={instance.get('instance_id', 'unknown')}: {e}\n{traceback.format_exc()}"
            )
            if env_attempt < MAX_ENV_UNAVAILABLE_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying reward evaluation in {ENV_UNAVAILABLE_RETRY_WAIT}s..."
                )
                await asyncio.sleep(ENV_UNAVAILABLE_RETRY_WAIT)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_ENV_UNAVAILABLE_RETRIES} retry attempts exhausted"
                )
                attempt_result["error"] = f"Unexpected error: {e}"
                return attempt_result

from .azure_modal_docker import AzureDockerEnvironment
from .swe_wrapper_v2 import EnvironmentUnavailable, _is_environment_error
from .swe_prm import _parse_patch_files, parse_trajectory_from_sample

logger = logging.getLogger(__name__)


async def _execute(env, command: str, timeout: int | None = None) -> dict:
    """Execute a command with timeout handling.

    Args:
        env: The execution environment (AzureDockerEnvironment or sync docker env).
        command: The shell command to execute.

    Returns:
        Dict with at least 'output' and 'returncode' keys.

    Raises:
        asyncio.TimeoutError: If the command exceeds the timeout.
    """

    execute_timeout = timeout or SWE_TIMEOUT_REWARD_EXECUTE
    old_timeout = None
    if hasattr(env, "config") and hasattr(env.config, "exe_timeout"):
        old_timeout = env.config.exe_timeout
        env.config.exe_timeout = max(int(old_timeout), execute_timeout)
    try:
        if isinstance(env, AzureDockerEnvironment):
            result = await asyncio.wait_for(env.execute(command), timeout=execute_timeout)
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(env.execute, command), timeout=execute_timeout
            )
        # Detect irrecoverable environment errors (e.g., container deleted/expired)
        output_text = result.get("output", "")
        if result.get("returncode") != 0 and _is_environment_error(output_text):
            raise EnvironmentUnavailable(output_text)
        return result
    except asyncio.TimeoutError:
        cmd_preview = command[:200] + ("..." if len(command) > 200 else "")
        logger.error(
            "[SWE_REWARD] _execute TIMED OUT after %ds for command: %s",
            execute_timeout,
            cmd_preview,
        )
        return {
            "output": f"Command timed out after {execute_timeout}s",
            "returncode": -1,
            "timed_out": True,
        }
    finally:
        if old_timeout is not None:
            env.config.exe_timeout = old_timeout


# Maximum base64 payload size per shell command. The agent exec endpoint passes
# the entire command string as a single argv to bash, so this must stay well
# below the sandbox's ARG_MAX. Some sandboxes have very small effective limits
# (~128KB total argv+env), so we keep chunks conservative.
_B64_CHUNK_LIMIT = 32_000

async def _write_file_to_sandbox(env, path: str, content: str | bytes) -> dict:
    """Write content into the sandbox using chunked base64 writes.

    Always uses base64 + heredoc chunking so we can transport arbitrary bytes
    (patches with special characters, embedded delimiters, etc.) without ever
    exceeding the sandbox's per-command argv size limit. The chunk size is
    deliberately small to stay safely under ARG_MAX in restricted sandboxes.

    Returns the result dict from the final _execute call.
    """
    if isinstance(content, str):
        raw = content.encode()
    else:
        raw = content
    encoded = base64.b64encode(raw).decode()

    quoted_path = shlex.quote(path)
    quoted_b64 = shlex.quote(path + ".b64")

    # Clear any stale temp files from a previous run.
    await _execute(env, f"rm -f {quoted_path} {quoted_b64}")

    # Append base64 chunks via heredoc. Chunks are base64-only, so they cannot
    # contain the delimiter B64EOF.
    last_result: dict = {}
    for i in range(0, len(encoded), _B64_CHUNK_LIMIT):
        chunk = encoded[i:i + _B64_CHUNK_LIMIT]
        last_result = await _execute(
            env,
            f"cat >> {quoted_b64} << 'B64EOF'\n{chunk}\nB64EOF",
        )
        if last_result.get("returncode", 0) != 0:
            return last_result

    return await _execute(
        env,
        f"base64 -d {quoted_b64} > {quoted_path} && rm -f {quoted_b64}",
    )


def make_test_spec_scaleswe(
    instance: SWEbenchInstance,
    namespace: Optional[str] = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
    *,
    fail_to_pass_override: list[str] | None = None,
    pass_to_pass_override: list[str] | None = None,
    test_command_override: str | None = None,
) -> TestSpec:
    """Create a TestSpec for Scale-SWE instances that don't use MAP_REPO_VERSION_TO_SPECS.

    Scale-SWE instances use f2p_patch/f2p_script instead of the standard test_patch,
    and provide pre-built Docker images rather than repo version specs.
    """
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance.get("repo", "")
    version = instance.get("version")

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if key not in instance:
            # If P2P, F2P keys not found, it's a validation instance
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = (
        list(pass_to_pass_override)
        if pass_to_pass_override is not None
        else _from_json_or_obj("PASS_TO_PASS")
    )
    fail_to_pass = (
        list(fail_to_pass_override)
        if fail_to_pass_override is not None
        else _from_json_or_obj("FAIL_TO_PASS")
    )

    # For Scale-SWE, use f2p_patch (introduces failing test expectations);
    # fall back to test_patch for standard SWE-bench instances.
    test_patch = instance.get("f2p_patch") or instance.get("test_patch", "")

    repo_directory = instance.get("workdir", "/testbed")

    if test_command_override is not None:
        test_command = test_command_override
    else:
        test_command = build_prebuilt_test_command(
            instance,
            fail_to_pass,
            pass_to_pass,
            focused=False,
        )

    docker_specs = {}
    repo_script_list = []
    env_script_list = []
    HEREDOC_DELIMITER = "EOF_114329324912"

    eval_script_list = [
        f"cd {repo_directory}",
    ]
    # Only apply the test patch (f2p_patch) if non-empty
    if test_patch:
        eval_script_list.append(
            f"cat > /tmp/f2p_patch.diff <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
        eval_script_list.append(
            f"cd {repo_directory} && "
            f"(git apply --verbose /tmp/f2p_patch.diff || "
            f"git apply --verbose --reject /tmp/f2p_patch.diff || "
            f"patch --batch --forward --fuzz=5 -p1 -i /tmp/f2p_patch.diff || "
            f"true)"
        )
    eval_script_list.extend([
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
    ])

    arch = "x86_64"

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language='py',
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
    )


def make_test_spec_swerebench_v2(
    instance: SWEbenchInstance,
    namespace: Optional[str] = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
    *,
    fail_to_pass_override: list[str] | None = None,
    pass_to_pass_override: list[str] | None = None,
    test_command_override: str | None = None,
) -> TestSpec:
    """Create a TestSpec for SWE-rebench-V2 instances.

    Rebench-V2 instances provide pre-built Docker images via ``image_url``,
    a structured ``install_config`` with ``test_cmd`` and ``log_parser``,
    and a standard ``test_patch`` for introducing failing-test expectations.
    """
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"

    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance.get("repo", "")
    version = instance.get("version")

    def _from_json_or_obj(key: str) -> Any:
        if key not in instance:
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = (
        list(pass_to_pass_override)
        if pass_to_pass_override is not None
        else _from_json_or_obj("PASS_TO_PASS")
    )
    fail_to_pass = (
        list(fail_to_pass_override)
        if fail_to_pass_override is not None
        else _from_json_or_obj("FAIL_TO_PASS")
    )

    test_patch = instance.get("test_patch", "")
    repo_directory = instance.get("workdir", "/testbed")

    # Use install_config from instance (contains test_cmd, log_parser, etc.)
    inst_install_config = instance.get("install_config", {})
    test_command = test_command_override
    if test_command is None:
        test_command = build_prebuilt_test_command(
            instance,
            fail_to_pass,
            pass_to_pass,
            focused=False,
        )
    log_parser = inst_install_config.get("log_parser", "parse_log_pytest")

    install_config = {
        "env_vars": None,
        "env_yml_path": None,
        "install": inst_install_config.get("install", ""),
        "log_parser": log_parser,
        "no_use_env": None,
        "packages": "",
        "pip_packages": [],
        "pre_install": None,
        "python": "3.10",
        "reqs_path": [],
        "test_cmd": test_command,
    }

    docker_specs = inst_install_config.get("docker_specs", {})
    repo_script_list = [install_config.get("install", "")]
    env_script_list = []
    HEREDOC_DELIMITER = "EOF_114329324912"

    eval_script_list = [
        f"cd {repo_directory}",
    ]
    # Apply test_patch if non-empty
    if test_patch:
        eval_script_list.append(
            f"cat > /tmp/test_patch.diff <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
        eval_script_list.append(
            f"cd {repo_directory} && "
            f"(git apply --verbose /tmp/test_patch.diff || "
            f"git apply --verbose --reject /tmp/test_patch.diff || "
            f"patch --batch --forward --fuzz=5 -p1 -i /tmp/test_patch.diff || "
            f"true)"
        )
    eval_script_list.extend([
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
    ])

    arch = "x86_64"

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language="py",
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
    )


def _is_unlabeled_swerebench_instance(instance: dict[str, Any]) -> bool:
    """Recognize prebuilt SWE-rebench rows whose dataset provenance was dropped."""
    if instance.get("dataset"):
        return False
    install_config = instance.get("install_config")
    return (
        instance.get("repo") not in MAP_REPO_TO_EXT
        and bool(instance.get("image_url") or instance.get("image_name"))
        and isinstance(install_config, dict)
        and bool(install_config.get("test_cmd"))
    )


def uses_official_swerebench_verifier(instance: dict[str, Any]) -> bool:
    """Route only SWE-rebench V2 rows through its official verifier."""
    return instance.get("dataset") == "nebius/SWE-rebench-V2" or (
        _is_unlabeled_swerebench_instance(instance)
    )


def _load_official_swerebench_verifier():
    """Import the vendored official verifier only when a rebench row needs it."""
    _ensure_swe_harness_on_path()
    from minisweagent.run.benchmarks.swerebench_verify_azure_modal import (
        evaluate_instance_azure_modal,
    )

    return evaluate_instance_azure_modal


def _official_swerebench_env_config(
    config: dict[str, Any], resource_level: int
) -> dict[str, Any]:
    """Translate the rollout environment config to the official verifier API."""
    source = config.get("environment", {})
    accepted_keys = {
        "env",
        "forward_env",
        "sandbox_ready_poll_interval",
        "login_shell",
        "heartbeat_interval",
        "cleanup_retries",
        "cleanup_retry_wait",
    }
    env_config = {key: copy.deepcopy(source[key]) for key in accepted_keys if key in source}
    env_config["base_url"] = os.environ.get("SANDBOX_BASE_URL", source.get("base_url", "http://localhost:8000"))
    env_config["api_key"] = os.environ.get("SANDBOX_API_KEY", source.get("api_key", ""))
    env_config["block_network"] = False
    env_config["cpu"] = str(resource_level * int(os.environ.get("SWE_REWARD_CPU_PER_LEVEL", "4")))
    env_config["memory"] = (
        f"{resource_level * int(os.environ.get('SWE_REWARD_MEM_GB_PER_LEVEL', '8'))}Gi"
    )

    create_timeout = max(1, SWE_TIMEOUT_REWARD_CREATE_ENV)
    execute_timeout = max(1, SWE_TIMEOUT_REWARD_EXECUTE)
    env_config["sandbox_timeout"] = create_timeout
    env_config["timeout"] = execute_timeout
    env_config["cleanup_timeout"] = max(1, SWE_TIMEOUT_REWARD_STOP_ENV)
    env_config["request_timeout"] = max(
        int(source.get("request_timeout", 0)),
        create_timeout + 60,
        execute_timeout + 60,
        env_config["cleanup_timeout"],
    )
    return env_config


def _official_passed_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _map_official_swerebench_result(
    official: dict[str, Any],
    result: dict[str, Any],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> dict[str, Any]:
    """Convert the official verifier output to the reward result schema."""
    f2p_total = len(fail_to_pass)
    p2p_total = len(pass_to_pass)
    f2p_passed = _official_passed_count(official.get("fail_to_pass_passed"))
    p2p_passed = _official_passed_count(official.get("pass_to_pass_passed"))
    resolved = bool(official.get("resolved"))

    result.update(
        passed=resolved,
        resolved=resolved,
        resolution=(
            ResolvedStatus.FULL.value
            if resolved
            else ResolvedStatus.PARTIAL.value
            if f2p_passed and p2p_passed == p2p_total
            else ResolvedStatus.NO.value
        ),
        fail_to_pass_rate=f2p_passed / f2p_total if f2p_total else 1.0,
        pass_to_pass_rate=p2p_passed / p2p_total if p2p_total else 1.0,
        tests_run=f2p_total + p2p_total,
        tests_passed=f2p_passed + p2p_passed,
        f2p_total=f2p_total,
        f2p_passed=f2p_passed,
        p2p_total=p2p_total,
        p2p_passed=p2p_passed,
        error=str(official.get("error") or ""),
        output=str(official.get("output") or ""),
        exit_code=official.get("exit_code"),
        patch_applied=bool(official.get("patch_applied")),
        parsed_tests_count=official.get("parsed_tests_count", 0),
        official_swerebench_verifier=True,
    )
    if not result["patch_applied"]:
        result["passed"] = False
        result["resolved"] = False
        result["resolution"] = ResolvedStatus.NO.value
        result["error"] = result["error"] or "Official SWE-rebench verifier failed to apply model patch"
    return result


async def _run_official_swerebench_verifier(
    *,
    instance: dict[str, Any],
    patch: str,
    config: dict[str, Any],
    result: dict[str, Any],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    resource_level: int,
) -> dict[str, Any]:
    evaluator = _load_official_swerebench_verifier()
    official_instance = copy.deepcopy(instance)
    image_name = official_instance.get("image_name") or official_instance.get("image_url")
    if not image_name:
        result["error"] = "SWE-rebench instance is missing image_name/image_url"
        return result
    if image_name.startswith("mirror.gcr.io/"):
        image_name = image_name[len("mirror.gcr.io/") :]
    official_instance["image_name"] = image_name
    official_instance["FAIL_TO_PASS"] = list(fail_to_pass)
    official_instance["PASS_TO_PASS"] = list(pass_to_pass)
    env_config = _official_swerebench_env_config(config, resource_level)

    official = await asyncio.to_thread(evaluator, official_instance, patch, env_config)
    return _map_official_swerebench_result(
        official,
        result,
        fail_to_pass,
        pass_to_pass,
    )


async def run_official_swerebench_in_environment(
    env,
    instance: dict[str, Any],
    *,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    test_command_override: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Run the official SWE-rebench suite on a patch already applied in ``env``.

    The caller owns the sandbox and its model patch. This function uploads only
    the hidden test patch and official eval script, executes the same script and
    parser as the standalone verifier, and deliberately leaves sandbox cleanup
    to the caller.
    """
    _ensure_swe_harness_on_path()
    from minisweagent.run.benchmarks.swerebench_verify_azure_modal import (
        build_eval_script,
        grade_eval_output,
    )
    official_instance = copy.deepcopy(instance)
    official_instance["FAIL_TO_PASS"] = list(fail_to_pass)
    official_instance["PASS_TO_PASS"] = list(pass_to_pass)
    eval_script = build_eval_script(
        official_instance,
        test_command_override=test_command_override,
        environment_prepared=False,
    )
    await _write_file_to_sandbox(
        env,
        "/tmp/test_patch.diff",
        str(official_instance.get("test_patch") or ""),
    )
    await _write_file_to_sandbox(env, "/tmp/eval.sh", eval_script)

    started = time.monotonic()
    run = await _execute(env, "bash /tmp/eval.sh 2>&1", timeout=timeout)
    output = str(run.get("output") or "")
    official = grade_eval_output(
        official_instance,
        output,
        exit_code=int(run.get("returncode") if run.get("returncode") is not None else -1),
        patch_applied=True,
    )
    official["output"] = output

    result = {
        "passed": False,
        "resolved": False,
        "resolution": ResolvedStatus.NO.value,
        "fail_to_pass_rate": 0.0,
        "pass_to_pass_rate": 0.0,
        "tests_run": 0,
        "tests_passed": 0,
        "f2p_total": 0,
        "f2p_passed": 0,
        "p2p_total": 0,
        "p2p_passed": 0,
        "error": "",
        "output": "",
    }
    mapped = _map_official_swerebench_result(
        official,
        result,
        fail_to_pass,
        pass_to_pass,
    )
    mapped["reward_eval_seconds"] = time.monotonic() - started
    mapped["reused_rollout_verify_environment"] = True
    return mapped


async def run_official_tests_in_environment(
    env,
    instance: dict[str, Any],
    *,
    patch: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> dict[str, Any]:
    """Run the dataset's full official suite in a caller-owned verifier."""
    if uses_official_swerebench_verifier(instance):
        return await run_official_swerebench_in_environment(
            env,
            instance,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )

    started = time.monotonic()
    official_instance = copy.deepcopy(instance)
    official_instance["FAIL_TO_PASS"] = list(fail_to_pass)
    official_instance["PASS_TO_PASS"] = list(pass_to_pass)
    result = {
        "passed": False,
        "resolved": False,
        "resolution": ResolvedStatus.NO.value,
        "fail_to_pass_rate": 0.0,
        "pass_to_pass_rate": 0.0,
        "tests_run": 0,
        "tests_passed": 0,
        "f2p_total": 0,
        "f2p_passed": 0,
        "p2p_total": 0,
        "p2p_passed": 0,
        "error": "",
        "output": "",
        "patch_applied": True,
    }
    try:
        test_spec, uses_prebuilt_grading = make_test_spec_for_instance(official_instance)
        workdir = str(official_instance.get("workdir") or "/testbed")
        f2p_script = str(official_instance.get("f2p_script") or "")
        if f2p_script:
            await _write_file_to_sandbox(env, f"{workdir}/test_fail_to_pass.py", f2p_script)

        eval_script = test_spec.eval_script.replace("locale-gen", "locale-gen en_US.UTF-8")
        await _write_file_to_sandbox(env, "/tmp/slime_official_eval.sh", eval_script)
        command = f"cd {shlex.quote(workdir)}"
        if "pylint" in test_spec.instance_id:
            command += " && PYTHONPATH="
        command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
        command += " && /bin/bash /tmp/slime_official_eval.sh 2>&1"
        run = await _execute(env, command)
        output = str(run.get("output") or "")
        result["output"] = output

        if uses_prebuilt_grading:
            result = grade_prebuilt_pytest_output(
                official_instance,
                test_spec,
                output,
                result,
                returncode=run.get("returncode"),
            )
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as stream:
                stream.write(output)
                test_output_path = Path(stream.name)
            try:
                report = get_eval_report(
                    test_spec=test_spec,
                    prediction={
                        KEY_INSTANCE_ID: official_instance["instance_id"],
                        KEY_MODEL: "slime-swe-agent",
                        KEY_PREDICTION: patch,
                    },
                    test_log_path=test_output_path,
                    include_tests_status=True,
                )
            finally:
                test_output_path.unlink(missing_ok=True)

            instance_report = report.get(official_instance["instance_id"], {})
            tests_status = instance_report.get("tests_status", {})
            f2p_status = tests_status.get(FAIL_TO_PASS, {})
            p2p_status = tests_status.get(PASS_TO_PASS, {})
            f2p_passed = len(f2p_status.get("success", []))
            f2p_failed = len(f2p_status.get("failure", []))
            p2p_passed = len(p2p_status.get("success", []))
            p2p_failed = len(p2p_status.get("failure", []))
            resolved = bool(instance_report.get("resolved"))
            result.update(
                passed=resolved,
                resolved=resolved,
                resolution=ResolvedStatus.FULL.value if resolved else ResolvedStatus.NO.value,
                f2p_total=f2p_passed + f2p_failed,
                f2p_passed=f2p_passed,
                p2p_total=p2p_passed + p2p_failed,
                p2p_passed=p2p_passed,
                tests_run=f2p_passed + f2p_failed + p2p_passed + p2p_failed,
                tests_passed=f2p_passed + p2p_passed,
                fail_to_pass_rate=(
                    f2p_passed / (f2p_passed + f2p_failed)
                    if f2p_passed + f2p_failed
                    else 1.0
                ),
                pass_to_pass_rate=(
                    p2p_passed / (p2p_passed + p2p_failed)
                    if p2p_passed + p2p_failed
                    else 1.0
                ),
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"

    result["reward_eval_seconds"] = time.monotonic() - started
    result["reused_rollout_verify_environment"] = True
    return result


def uses_prebuilt_test_spec(instance: dict[str, Any]) -> bool:
    """Whether official evaluation uses a prebuilt Scale/SWE-rebench image spec."""
    return instance.get("dataset") in {"AweAI-Team/Scale-SWE", "nebius/SWE-rebench-V2"} or (
        _is_unlabeled_swerebench_instance(instance)
    )


def _patch_test_directives(instance: dict[str, Any]) -> set[str]:
    patch = str(instance.get("f2p_patch") or instance.get("test_patch") or "")
    return {
        match.group(1).strip()
        for match in re.finditer(r"(?m)^(?:---|\+\+\+) [ab]/(.+)$", patch)
        if match.group(1).strip()
    }


def build_prebuilt_test_command(
    instance: dict[str, Any],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    *,
    focused: bool,
) -> str:
    """Build the command used by both full official verify and focused repair.

    Full evaluation preserves the dataset command. Focused repair preserves its
    interpreter/environment/options while replacing only test-file directives
    with the selected oracle nodes.
    """
    dataset = instance.get("dataset", "")
    selected = [*fail_to_pass, *pass_to_pass]
    if dataset == "AweAI-Team/Scale-SWE":
        base = "pytest --no-header -rA --tb=line --color=no -p no:cacheprovider -W ignore::DeprecationWarning"
        return " ".join([base, *(shlex.quote(node) for node in selected)]).strip()

    install_config = instance.get("install_config") or {}
    if isinstance(install_config, str):
        try:
            install_config = json.loads(install_config)
        except json.JSONDecodeError:
            install_config = {}
    configured = install_config.get("test_cmd") if isinstance(install_config, dict) else None
    if isinstance(configured, (list, tuple)):
        commands = [str(command).strip() for command in configured if str(command).strip()]
        is_python = bool(install_config.get("packages") or install_config.get("python"))
        if not is_python:
            if focused:
                matching = [command for command in commands if any(node in command for node in selected)]
                commands = matching or commands
            return " && ".join(f"({command})" for command in commands)
        command = " ".join(commands)
        if not focused:
            return command
    else:
        command = str(configured or "python -m pytest -rA --tb=short -p no:cacheprovider --color=no").strip()
    if not focused or not selected:
        return command
    if not re.search(r"(?:^|\s)(?:(?:python|python3)\s+-m\s+)?(?:pytest|py\.test)(?:\s|$)", command):
        return command

    directives = _patch_test_directives(instance)
    command_parts = [
        part for part in shlex.split(command) if part.split("::", 1)[0] not in directives
    ]
    return shlex.join([*command_parts, *selected])


def make_test_spec_for_instance(
    instance: dict[str, Any],
    *,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    focused: bool = False,
) -> tuple[TestSpec, bool]:
    """Create the one TestSpec used by both repair gates and final verify."""
    dataset = instance.get("dataset", "")
    if dataset == "AweAI-Team/Scale-SWE":
        if fail_to_pass is None and pass_to_pass is None and not focused:
            return make_test_spec_scaleswe(instance), True
        command = None
        if focused:
            command = build_prebuilt_test_command(
                instance,
                fail_to_pass or [],
                pass_to_pass or [],
                focused=True,
            )
        return (
            make_test_spec_scaleswe(
                instance,
                fail_to_pass_override=fail_to_pass,
                pass_to_pass_override=pass_to_pass,
                test_command_override=command,
            ),
            True,
        )
    if dataset == "nebius/SWE-rebench-V2" or _is_unlabeled_swerebench_instance(instance):
        if fail_to_pass is None and pass_to_pass is None and not focused:
            return make_test_spec_swerebench_v2(instance), True
        command = None
        if focused:
            command = build_prebuilt_test_command(
                instance,
                fail_to_pass or [],
                pass_to_pass or [],
                focused=True,
            )
        return (
            make_test_spec_swerebench_v2(
                instance,
                fail_to_pass_override=fail_to_pass,
                pass_to_pass_override=pass_to_pass,
                test_command_override=command,
            ),
            True,
        )

    standard_instance = instance
    if fail_to_pass is not None or pass_to_pass is not None:
        standard_instance = dict(instance)
        if fail_to_pass is not None:
            standard_instance["FAIL_TO_PASS"] = list(fail_to_pass)
        if pass_to_pass is not None:
            standard_instance["PASS_TO_PASS"] = list(pass_to_pass)
    return make_test_spec(standard_instance), False


_PYTEST_FAILURE_PREFIXES = ("FAILED", "ERROR", "SUBFAILED", "SUBERROR")


def _pytest_failure_nodes(output: str) -> set[str]:
    """Collect failed nodes that the upstream parser may later overwrite/ignore."""
    failed: set[str] = set()
    for raw_line in output.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw_line).strip()
        if not line.startswith(_PYTEST_FAILURE_PREFIXES):
            continue
        matches = re.findall(r"(?:^|\s)([^\s]+\.py(?:::[^\s]+)+)\s*$", line)
        if matches:
            failed.add(matches[-1])
            continue
        parts = line.replace(" - ", " ").split()
        if len(parts) >= 2 and parts[0] in {"FAILED", "ERROR"}:
            failed.add(parts[1])
    return failed


def grade_prebuilt_pytest_output(
    instance: dict[str, Any],
    test_spec: TestSpec,
    output: str,
    result: dict[str, Any],
    *,
    returncode: int | None = None,
) -> dict[str, Any]:
    """Grade Scale-SWE/SWE-rebench pytest logs without standard repo maps."""
    if START_TEST_OUTPUT not in output or END_TEST_OUTPUT not in output:
        result["error"] = "Test output is missing START/END markers"
        return result

    test_output = output.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    parser_name = (instance.get("install_config") or {}).get("log_parser")
    if parser_name in (None, "", "parse_log_pytest_v2", "parse_log_pytest_scaleswe"):
        parser = parse_log_pytest_v2
    elif parser_name == "parse_log_pytest":
        parser = parse_log_pytest
    else:
        result["error"] = f"Unsupported prebuilt-image log parser: {parser_name}"
        return result

    if parser is None:
        result["error"] = "SWE-bench pytest log parser is unavailable"
        return result
    status_map = parser(test_output, test_spec)
    if not status_map:
        status_map = parser(output, test_spec)
    for failed_node in _pytest_failure_nodes(test_output):
        # Failure is monotonic: a parent PASSED line must never erase a
        # SUBFAILED/FAILED record for the same oracle node.
        status_map[failed_node] = "FAILED"
    if not status_map:
        result["error"] = "No pytest test statuses could be parsed"
        return result

    passing = {"PASSED", "XFAIL"}
    fail_to_pass = list(test_spec.FAIL_TO_PASS)
    pass_to_pass = list(test_spec.PASS_TO_PASS)
    f2p_passed = sum(status_map.get(test) in passing for test in fail_to_pass)
    p2p_passed = sum(status_map.get(test) in passing for test in pass_to_pass)

    result["f2p_total"] = len(fail_to_pass)
    result["f2p_passed"] = f2p_passed
    result["p2p_total"] = len(pass_to_pass)
    result["p2p_passed"] = p2p_passed
    result["tests_run"] = len(fail_to_pass) + len(pass_to_pass)
    result["tests_passed"] = f2p_passed + p2p_passed
    result["fail_to_pass_rate"] = f2p_passed / len(fail_to_pass) if fail_to_pass else 1.0
    result["pass_to_pass_rate"] = p2p_passed / len(pass_to_pass) if pass_to_pass else 1.0
    result["test_returncode"] = returncode
    resolved = (
        bool(fail_to_pass)
        and f2p_passed == len(fail_to_pass)
        and p2p_passed == len(pass_to_pass)
        and returncode in (None, 0)
    )
    result["passed"] = resolved
    result["resolved"] = resolved
    result["resolution"] = ResolvedStatus.FULL.value if resolved else ResolvedStatus.NO.value
    return result


async def _run_tests_with_swebench_harness(
    instance: dict,
    patch: str,
    config: dict,
    result: dict,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    resource_level: int = 1,
) -> dict:
    """
    Async implementation of SWE-bench harness evaluation.

    This follows the same approach as run_instance_modal():
    1. Create TestSpec from instance
    2. Apply patch with git apply (fallback to patch command)
    3. Run test_spec.eval_script
    4. Grade with get_eval_report()
    """
    import time
    reward_start_time = time.time()
    instance_id = instance.get("instance_id", "unknown")
    logger.info(f"[SWE_REWARD] Starting _run_tests_with_swebench_harness for instance_id={instance_id}")

    # Repair and final reward share this exact TestSpec routing. The repair
    # path only overrides the selected F2P/P2P lists.
    try:
        test_spec, uses_prebuilt_grading = make_test_spec_for_instance(instance)
    except Exception as e:
        result["error"] = f"Failed to create TestSpec: {e}"
        logger.error(f"[SWE_REWARD] Failed to create TestSpec for instance_id={instance_id}: {e}")
        return result


    # Get environment config
    env_config = config.get("environment", {}).copy()
    env_config["environment_class"] = env_config.get("environment_class", "docker")

    # Increase execution timeout for reward evaluation (original exe_timeout + 300s)
    env_config["exe_timeout"] = env_config.get("exe_timeout", 60) + 300

    # Scale CPU and memory based on resource level (only increased for EnvironmentUnavailable)
    cpu_per_level = int(os.environ.get("SWE_REWARD_CPU_PER_LEVEL", "4"))
    mem_per_level = int(os.environ.get("SWE_REWARD_MEM_GB_PER_LEVEL", "8"))
    env_config["cpu"] = str(resource_level * cpu_per_level)
    env_config["memory"] = f"{resource_level * mem_per_level}Gi"
    logger.info(f"[SWE_REWARD] Environment resources for resource_level {resource_level}: cpu={env_config['cpu']}, memory={env_config['memory']}")

    # Resolve workdir: use instance's workdir if available, otherwise default to /testbed
    workdir = instance.get("workdir", "/testbed")
    env_config.setdefault("cwd", workdir)
    logger.info(f"[SWE_REWARD] Using workdir={workdir} for instance_id={instance_id}")

    # Build startup command: append pre_commands from instance if available
    startup_command = config.get("run", {}).get("env_startup_command")

    # Timeout for creating the reward evaluation environment
    MAX_CREATE_ENV_RETRIES = 1
    RETRY_WAIT_SECONDS = 5
    env = None

    # Create Docker environment with correct SWE-bench image (with retry logic)
    logger.info(f"[SWE_REWARD] Creating test environment for instance_id={instance_id}")
    _env_create_start = time.time()
    for attempt in range(1, MAX_CREATE_ENV_RETRIES + 1):
        logger.info(f"[SWE_REWARD] Attempt {attempt}/{MAX_CREATE_ENV_RETRIES} to create environment for instance_id={instance_id}")
        try:
            env = await asyncio.wait_for(
                create_environment(
                    env_config,
                    instance_id=f"test-{instance_id}",
                    instance=instance,
                    startup_command=startup_command,
                ),
                timeout=SWE_TIMEOUT_REWARD_CREATE_ENV,
            )
            break  # Success, exit retry loop
        except asyncio.TimeoutError:
            logger.error(
                f"[SWE_REWARD] TIMEOUT creating test environment for instance_id={instance_id} "
                f"after {SWE_TIMEOUT_REWARD_CREATE_ENV}s (attempt {attempt}/{MAX_CREATE_ENV_RETRIES})"
            )
            if attempt < MAX_CREATE_ENV_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying in {RETRY_WAIT_SECONDS}s for instance_id={instance_id}"
                )
                await asyncio.sleep(RETRY_WAIT_SECONDS)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_CREATE_ENV_RETRIES} attempts exhausted for instance_id={instance_id}"
                )
                result["error"] = (
                    f"Timed out creating test environment after {SWE_TIMEOUT_REWARD_CREATE_ENV}s "
                    f"({MAX_CREATE_ENV_RETRIES} attempts)"
                )
                return result
        except Exception as e:
            logger.error(
                f"[SWE_REWARD] Failed to create test environment for instance_id={instance_id}: {e} "
                f"(attempt {attempt}/{MAX_CREATE_ENV_RETRIES})"
            )
            if attempt < MAX_CREATE_ENV_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying in {RETRY_WAIT_SECONDS}s for instance_id={instance_id}"
                )
                await asyncio.sleep(RETRY_WAIT_SECONDS)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_CREATE_ENV_RETRIES} attempts exhausted for instance_id={instance_id}"
                )
                result["error"] = f"Failed to create test environment: {e} ({MAX_CREATE_ENV_RETRIES} attempts)"
                return result
    assert env is not None, "Environment creation failed without raising an exception"
    
    logger.info(f"[SWE_REWARD] Test environment created for instance_id={instance_id}, elapsed={time.time() - _env_create_start:.2f}s")

    test_output_path = None
    try:
        # Write patch to a file in the container
        # Using base64 + chunking to handle arbitrarily large patches and any special chars
        logger.info(f"[SWE_REWARD] [{instance_id}] Writing patch to container...")
        # # Previous version used a heredoc to write the patch directly into the container
        # write_patch_cmd = f"cat > /tmp/patch.diff << 'SWEAGENT_PATCH_EOF'\n{patch}\nSWEAGENT_PATCH_EOF"
        # await _execute(env, write_patch_cmd)
        await _write_file_to_sandbox(env, "/tmp/patch.diff", patch)
        logger.info(f"[SWE_REWARD] [{instance_id}] Patch written, applying with git apply...")

        # Apply the patch using multiple fallback strategies (following scaleswe evaluator)
        patch_applied = False
        for git_apply_cmd in [
            f"cd {workdir} && git apply --verbose /tmp/patch.diff",
            f"cd {workdir} && git apply --verbose --reject /tmp/patch.diff",
            f"cd {workdir} && patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        ]:
            apply_result = await _execute(env, git_apply_cmd)
            if apply_result.get("returncode", 0) == 0:
                patch_applied = True
                break
            logger.warning("[SWE_REWARD] [%s] Patch apply failed with: %s", instance_id, git_apply_cmd)

        if not patch_applied:
            result["error"] = f"Failed to apply patch: {apply_result.get('output', '')}"
            return result

        # Get git diff before running eval script (for debugging)
        _git_diff_before = await _execute(env, f"cd {workdir} && git -c core.fileMode=false diff")

        f2p_script = instance.get("f2p_script", "")
        if f2p_script:
            write_f2p_cmd = f"cat > {workdir}/test_fail_to_pass.py << 'SWEAGENT_F2P_SCRIPT_EOF'\n{f2p_script}\nSWEAGENT_F2P_SCRIPT_EOF"
            await _execute(env, write_f2p_cmd)

        # Write and run the eval script from TestSpec
        eval_script = test_spec.eval_script
        # Django hack (same as run_instance_modal)
        eval_script = eval_script.replace("locale-gen", "locale-gen en_US.UTF-8")

        logger.debug("Eval script for %s:\n%s", instance_id, eval_script)

        write_eval_cmd = f"cat > /root/eval.sh << 'EVAL_EOF'\n{eval_script}\nEVAL_EOF"
        await _execute(env, write_eval_cmd)
        await _execute(env, "chmod +x /root/eval.sh")

        run_command = f"cd {workdir}"
        # pylint hack
        if "pylint" in test_spec.instance_id:
            run_command += " && PYTHONPATH="
        # increase recursion limit for testing
        run_command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
        # Redirect stderr to stdout so that set -x traces (which contain the
        # >>>>> Start/End Test Output markers) are interleaved with the actual
        # test output.  AzureDockerEnvironment captures stdout and stderr
        # separately and concatenates them (stdout first), which breaks the
        # marker-based splitting in get_logs_eval when 2>&1 is omitted.
        run_command += " && /bin/bash /root/eval.sh 2>&1"

        # run eval script
        _eval_start = time.time()
        logger.info(f"[SWE_REWARD] [{instance_id}] Running eval script...")

        test_result = await _execute(env, run_command)
        logger.info(f"[SWE_REWARD] [{instance_id}] Eval script finished in {time.time() - _eval_start:.2f}s")
        output = test_result.get("output", "")
        result["output"] = output

        # Get git diff after running eval script (for debugging)
        _git_diff_after = await _execute(env, f"cd {workdir} && git -c core.fileMode=false diff")

        # Write test output to a temporary file for get_eval_report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(output)
            test_output_path = Path(f.name)

        # Build prediction dict for get_eval_report (same format as run_instance_modal)
        pred = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: "slime-swe-agent",
            KEY_PREDICTION: patch,
        }

        if uses_prebuilt_grading:
            return grade_prebuilt_pytest_output(
                instance,
                test_spec,
                output,
                result,
                returncode=test_result.get("returncode"),
            )

        # Use get_eval_report for proper SWE-bench grading
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )

        # Extract results from report
        instance_report = report.get(instance_id, {})
        resolved = instance_report.get("resolved", False)

        result["passed"] = resolved
        result["resolved"] = resolved
        result["resolution"] = ResolvedStatus.FULL.value if resolved else ResolvedStatus.NO.value

        # Extract detailed test results if available
        tests_status = instance_report.get("tests_status", {})
        f2p_status = tests_status.get(FAIL_TO_PASS, {})
        p2p_status = tests_status.get(PASS_TO_PASS, {})

        f2p_passed = len(f2p_status.get("success", []))
        f2p_failed = len(f2p_status.get("failure", []))
        p2p_passed = len(p2p_status.get("success", []))
        p2p_failed = len(p2p_status.get("failure", []))

        result["f2p_total"] = f2p_passed + f2p_failed
        result["f2p_passed"] = f2p_passed
        result["p2p_total"] = p2p_passed + p2p_failed
        result["p2p_passed"] = p2p_passed
        result["tests_passed"] = f2p_passed + p2p_passed
        result["tests_run"] = result["f2p_total"] + result["p2p_total"]

        # Calculate rates
        if result["f2p_total"] > 0:
            result["fail_to_pass_rate"] = f2p_passed / result["f2p_total"]
        else:
            result["fail_to_pass_rate"] = 1.0  # No F2P tests means all "passing"

        if result["p2p_total"] > 0:
            result["pass_to_pass_rate"] = p2p_passed / result["p2p_total"]
        else:
            result["pass_to_pass_rate"] = 1.0  # No P2P tests means all "passing"

    except EnvironmentUnavailable:
        raise  # Propagate to retry loop in run_tests_in_docker()
    except Exception as e:
        result["error"] = f"Test execution error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[SWE_REWARD] Test execution error for instance_id={instance_id}: {e}")
    finally:
        # Always cleanup the environment
        if env is not None:
            logger.info(f"[SWE_REWARD] Stopping test environment for instance_id={instance_id}")
            try:
                await asyncio.wait_for(stop_environment(env), timeout=SWE_TIMEOUT_REWARD_EXECUTE)
                logger.info(f"[SWE_REWARD] Test environment stopped for instance_id={instance_id}")
            except asyncio.TimeoutError:
                logger.error(f"[SWE_REWARD] TIMEOUT stopping test environment for instance_id={instance_id}")
            except Exception as e:
                logger.error(f"[SWE_REWARD] Failed to stop test environment for instance_id={instance_id}: {e}")
        else:
            logger.warning(f"[SWE_REWARD] No environment to stop for instance_id={instance_id} (env was None)")
        # Cleanup temporary file
        if test_output_path is not None and test_output_path.exists():
            try:
                test_output_path.unlink()
            except Exception:
                pass

    result["reward_eval_time"] = time.time() - reward_start_time
    logger.info(
        "[SWE_REWARD] Test results for %s: resolved=%s f2p=%.3f p2p=%.3f tests=%s error=%s total_time=%.2fs",
        instance_id,
        result.get("resolved", False),
        result.get("fail_to_pass_rate", 0.0),
        result.get("pass_to_pass_rate", 0.0),
        result.get("tests_run", 0),
        bool(result.get("error")),
        result["reward_eval_time"],
    )
    return result


# ==============================================================================
# Process-shaping helpers (CODE-BASED, deterministic — section A + integrity gate G)
# ==============================================================================
#
# These power the ``process_shaped`` reward mode.  Design principle: outcome
# (resolved / F2P rate) stays primary; process terms only *densify* zero-variance
# GRPO groups and *anchor* shaping to ground truth.  Only signals validated to
# predict success are included (e.g. localization AUC≈0.58).  Anything that
# regressed to AUC≈0.5 in offline analysis (generic verify-before-submit,
# generic reproduce, raw patch-length penalty) is deliberately excluded.
#
# All helpers are pure / cheap (no Docker, no network): they read the already
# computed ``test_results`` dict and the parsed trajectory.  The trajectory is
# parsed once via ``swe_prm.parse_trajectory_from_sample`` (lazy import to avoid
# a hard dependency cycle and to keep import cost off the hot path).

# Heuristic: file paths that look like test / fixture files (used by A5 and G).
_TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|testing)(/|$)|(^|/)conftest\.py$|(^|/)test_[^/]*\.py$|[^/]*_test\.py$",
    re.IGNORECASE,
)

# Destructive shell commands that should trip the integrity gate G.
_DESTRUCTIVE_CMD_RE = re.compile(
    r"\brm\s+-rf\b|\bgit\s+reset\s+--hard\b|\bgit\s+checkout\s+--\s|\bgit\s+clean\s+-[a-z]*f"
    r"|\b(chmod|chown)\s+-R\b|>\s*/dev/sd|\bmkfs\b|\bdd\s+if=",
    re.IGNORECASE,
)

# Reward-hacking signatures in the *submitted patch* that should trip gate G:
# hardcoding an expected value / special-casing a specific failing input rather
# than fixing the root cause.  These are intentionally conservative (high
# precision, low recall) — the offline LLM judge (section B) catches the rest.
_HARDCODE_PATCH_RE = re.compile(
    r"^\+.*\b(pytest\.skip|pytest\.xfail|@pytest\.mark\.(skip|xfail))\b"
    r"|^\+\s*assert\s+True\b"
    r"|^\+\s*return\s+(True|None)\s*#",
    re.IGNORECASE | re.MULTILINE,
)


def _is_test_path(path: str) -> bool:
    """Return True if a repo-relative path looks like a test / fixture file."""
    return bool(_TEST_FILE_RE.search(path or ""))


def _localization_score(patch_files: list[str], gold_files: list[str]) -> float:
    """A2 (part 1): fraction of gold source files the submission actually touched.

    ``|submitted ∩ gold| / |gold|`` restricted to NON-test gold files (editing
    test files is graded by the integrity gate, not rewarded here).  Returns 0.0
    when there is no gold signal so the term is simply inert on datasets without
    a gold patch.
    """
    gold_src = [f for f in gold_files if not _is_test_path(f)]
    if not gold_src:
        return 0.0
    submitted = set(patch_files)
    hit = sum(1 for f in gold_src if f in submitted)
    return hit / len(gold_src)


def _read_before_edit_bonus(traj, gold_files: list[str]) -> float:
    """A2 (part 2): did the agent *read* a gold source file before its first edit?

    This is the genuine *process* half of the localization signal (grounding),
    as opposed to the outcome-correlated patch∩gold half.  Returns 1.0 if any
    gold source file appears in a read-style command that occurs strictly before
    the first edit-style command; else 0.0.  No gold signal -> 0.0 (inert).
    """
    gold_src = [f for f in gold_files if not _is_test_path(f)]
    if not gold_src or traj is None:
        return 0.0

    read_re = re.compile(r"\b(cat|less|head|tail|sed -n|grep|rg|nl|vi|view|open|read_file)\b")
    edit_re = re.compile(r"\b(sed -i|>|>>|tee|patch|apply_patch|git apply|python - <<|cat >)\b|edit_file")

    first_edit_idx = None
    for s in traj.steps:
        cmd = s.command or ""
        if edit_re.search(cmd):
            first_edit_idx = s.idx
            break

    for s in traj.steps:
        if first_edit_idx is not None and s.idx >= first_edit_idx:
            break
        cmd = s.command or ""
        if read_re.search(cmd) and any(g in cmd for g in gold_src):
            return 1.0
    return 0.0


def _extra_files_penalty(patch_files: list[str], gold_files: list[str]) -> float:
    """A5: fraction of submitted files that are outside the gold set (over-reach).

    Counts NON-test submitted files not present in the gold patch, normalized by
    the number of submitted source files.  This is a cleaner "over-modification"
    signal than raw patch length (length penalizes genuinely hard fixes).  When
    there is no gold signal, returns 0.0 (inert).
    """
    if not gold_files:
        return 0.0
    gold_set = set(gold_files)
    submitted_src = [f for f in patch_files if not _is_test_path(f)]
    if not submitted_src:
        return 0.0
    extra = sum(1 for f in submitted_src if f not in gold_set)
    return extra / len(submitted_src)


# Pytest result line patterns, e.g. ``tests/test_x.py::test_foo PASSED`` or
# ``FAILED tests/test_x.py::test_foo``.  Captures the nodeid and the status.
_PYTEST_LINE_RE = re.compile(
    r"(?:^|\s)(?P<node1>[\w./\-]+::[\w\[\]\-./ ]+?)\s+(?P<stat1>PASSED|FAILED|ERROR)\b"
    r"|(?:^|\s)(?P<stat2>PASSED|FAILED|ERROR)\s+(?P<node2>[\w./\-]+::[\w\[\]\-./ ]+)",
    re.MULTILINE,
)


def _test_states_in_text(text: str) -> dict[str, str]:
    """Extract a {test_nodeid: status} map from a chunk of test output."""
    states: dict[str, str] = {}
    if not text:
        return states
    for m in _PYTEST_LINE_RE.finditer(text):
        node = (m.group("node1") or m.group("node2") or "").strip()
        stat = (m.group("stat1") or m.group("stat2") or "").strip().upper()
        if node:
            states[node] = stat
    return states


def _detect_flip(traj, gold_f2p: list[str] | None = None) -> float:
    """A3: anchored reproduce→fix loop.

    Rewards a *flip* — a test observed FAILING earlier in the trajectory and
    PASSING later — NOT merely "ran a test" (that signal had AUC≈0.52 and is
    excluded).  Prefers tests whose nodeid matches a gold FAIL_TO_PASS name so
    the verification is anchored to the true target; falls back to any test that
    flips (agent-built self-check).  Returns 1.0 if such a flip exists else 0.0.
    """
    if traj is None:
        return 0.0
    gold_f2p = gold_f2p or []
    # Normalize gold F2P nodeids to their bare test-name tail for fuzzy matching.
    gold_names = set()
    for g in gold_f2p:
        gold_names.add(g)
        gold_names.add(g.split("::")[-1].split("[")[0])

    seen_fail: set[str] = set()
    any_flip = False
    for s in traj.steps:
        states = _test_states_in_text(s.observation or "")
        for node, stat in states.items():
            tail = node.split("::")[-1].split("[")[0]
            anchored = node in gold_names or tail in gold_names
            if stat in ("FAILED", "ERROR"):
                seen_fail.add(node)
            elif stat == "PASSED" and node in seen_fail:
                # A previously-failing test now passes -> flip.
                if anchored or not gold_names:
                    return 1.0
                any_flip = True
    # If we have gold names but only un-anchored flips, give partial credit.
    return 1.0 if any_flip and not gold_names else (0.5 if any_flip else 0.0)


def _error_rate(traj) -> float:
    """A7: fraction of executed steps whose command returned a non-zero code.

    Pure operational-hygiene loss signal (current baseline ≈14%).  Steps with an
    unknown returncode (None) are ignored from the denominator.
    """
    if traj is None:
        return 0.0
    rcs = [s.returncode for s in traj.steps if s.returncode is not None]
    if not rcs:
        return 0.0
    bad = sum(1 for rc in rcs if rc != 0)
    return bad / len(rcs)


def _no_progress_penalty(traj) -> float:
    """A6: penalize *patterns* of no-progress churn (NOT step count).

    Penalizes, in [0,1]:
      - exact repeated commands (same command string run again), and
      - repeated reads of the same file with no intervening edit.
    Step count itself is deliberately NOT penalized (it is inversely correlated
    with success — hard tasks need more steps).
    """
    if traj is None or not traj.steps:
        return 0.0
    cmds = [(s.command or "").strip() for s in traj.steps]
    nonempty = [c for c in cmds if c]
    if not nonempty:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for c in nonempty:
        if c in seen:
            repeats += 1
        else:
            seen.add(c)
    return min(1.0, repeats / len(nonempty))


def _integrity_gate(patch: str, traj, gold_test_files: list[str] | None = None) -> float:
    """Integrity gate G: multiplicative cap in {1.0 (clean), 0.0 (violation)}.

    Trips (returns 0.0) when the submission or trajectory shows reward-hacking /
    destructive behavior:
      - editing test / conftest / fixture files (incl. known gold test files),
      - hardcoded expected values / skip / xfail / ``assert True`` in the patch,
      - destructive shell commands (``rm -rf``, ``git reset --hard``, ...).
    This is a CAP, not a bonus: clean trajectories get 1.0 and are unaffected.
    """
    gold_test_files = set(gold_test_files or [])

    # 1) Edited test files (path heuristic OR known gold test file).
    for f in _parse_patch_files(patch):
        if f in gold_test_files or _is_test_path(f):
            return 0.0

    # 2) Reward-hacking signatures inside the patch body.
    if patch and _HARDCODE_PATCH_RE.search(patch):
        return 0.0

    # 3) Destructive commands anywhere in the trajectory.
    if traj is not None:
        for s in traj.steps:
            if s.command and _DESTRUCTIVE_CMD_RE.search(s.command):
                return 0.0

    return 1.0


# ==============================================================================
# Reward Computation Functions
# ==============================================================================

def _bool_from_metadata_or_env(metadata: dict[str, Any], key: str, env_key: str, default: bool) -> bool:
    value = metadata.get(key)
    if value is None:
        value = os.environ.get(env_key)
    label = _coerce_resolved_label(value)
    return default if label is None else label


def _int_from_metadata_or_env(metadata: dict[str, Any], key: str, env_key: str, default: int) -> int:
    value = metadata.get(key)
    if value is None:
        value = os.environ.get(env_key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def cap_patch_classification_candidates(
    candidates: list[dict[str, Any]], max_candidates: int
) -> list[dict[str, Any]]:
    """Cap candidates to ``max_candidates`` unique patches, balancing pos/neg.

    A single generated test is replayed once per unique candidate patch, so the
    reward cost scales with the number of distinct patches. Capping bounds that
    cost. Selection keeps oracle_resolved pos/neg roughly balanced (so
    balanced_accuracy stays meaningful) and is deterministic (ordered by
    patch_sha256) so the same sample yields the same subset across retries.
    """
    if max_candidates <= 0:
        return candidates
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda c: c.get("patch_sha256", "")):
        sha = candidate.get("patch_sha256", "")
        if sha in seen:
            continue
        seen.add(sha)
        unique.append(candidate)
    if len(unique) <= max_candidates:
        return unique
    pos = [c for c in unique if c.get("oracle_resolved")]
    neg = [c for c in unique if not c.get("oracle_resolved")]
    selected: list[dict[str, Any]] = []
    i = 0
    while len(selected) < max_candidates and (i < len(pos) or i < len(neg)):
        if i < len(pos) and len(selected) < max_candidates:
            selected.append(pos[i])
        if i < len(neg) and len(selected) < max_candidates:
            selected.append(neg[i])
        i += 1
    return selected


def _write_patch_classification_reward_details(sample: Sample, details: dict[str, Any]) -> None:
    if sample.metadata is not None:
        sample.metadata["patch_classification_reward"] = details

    trajectory_path = sample.metadata.get("trajectory_path", "unknown") if sample.metadata else "unknown"
    if trajectory_path == "unknown":
        return
    try:
        with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
            json.dump(details, f)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to write patch classification rewards to {trajectory_path}: {e}")


def _patch_classification_config_specs(metadata: dict[str, Any]) -> list[str] | None:
    raw = metadata.get("patch_classification_config_spec") or os.environ.get("SWE_PATCH_CLASSIFICATION_CONFIG_SPEC")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in text.split(",") if item.strip()]
    return None


def _apply_sandbox_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    env_config = config.setdefault("environment", {})
    if os.environ.get("SANDBOX_BASE_URL"):
        env_config["base_url"] = os.environ["SANDBOX_BASE_URL"]
    if os.environ.get("SANDBOX_API_KEY"):
        env_config["api_key"] = os.environ["SANDBOX_API_KEY"]
    configured_create_timeout = int(env_config.get("sandbox_timeout", SWE_TIMEOUT_REWARD_CREATE_ENV))
    if SWE_TIMEOUT_REWARD_CREATE_ENV > 0:
        env_config["sandbox_timeout"] = min(configured_create_timeout, SWE_TIMEOUT_REWARD_CREATE_ENV)
    env_config["cleanup_timeout"] = SWE_TIMEOUT_REWARD_STOP_ENV
    env_config["request_timeout"] = max(
        int(env_config.get("request_timeout", 0)),
        int(env_config.get("sandbox_timeout", SWE_TIMEOUT_REWARD_CREATE_ENV)) + 60,
        SWE_TIMEOUT_REWARD_STOP_ENV,
    )
    return config


def _build_patch_classification_config(args, metadata: dict[str, Any], gentest_module, patch_gentest_module) -> dict[str, Any]:
    raw_config = metadata.get("patch_classification_config") or metadata.get("minisweagent_config")
    if isinstance(raw_config, str):
        text = raw_config.strip()
        if text:
            try:
                raw_config = json.loads(text)
            except json.JSONDecodeError:
                path = Path(text)
                if path.exists():
                    raw_config = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_config, dict):
        return _apply_sandbox_runtime_config(copy.deepcopy(raw_config))

    specs = _patch_classification_config_specs(metadata)
    if specs is None:
        specs = [
            str(gentest_module.DEFAULT_CONFIG_FILE),
            str(patch_gentest_module.DEFAULT_CONFIG_FILE),
        ]

    environment_class = (
        metadata.get("patch_classification_environment_class")
        or getattr(args, "swe_patch_classification_environment_class", None)
        or os.environ.get("SWE_PATCH_CLASSIFICATION_ENVIRONMENT_CLASS")
    )
    config = gentest_module.build_config(
        specs,
        model=getattr(args, "swe_patch_classification_model", None),
        model_class=getattr(args, "swe_patch_classification_model_class", None),
        environment_class=environment_class,
    )
    return _apply_sandbox_runtime_config(config)


def _check_reward_deadline(start_time: float, phase: str) -> None:
    if SWE_TIMEOUT_REWARD_TOTAL > 0 and time.time() - start_time >= SWE_TIMEOUT_REWARD_TOTAL:
        raise TimeoutError(f"reward deadline exceeded before {phase} ({SWE_TIMEOUT_REWARD_TOTAL}s)")


def _patch_classification_metric_name(args, metadata: dict[str, Any]) -> str:
    return str(
        getattr(args, "swe_patch_classification_reward_metric", None)
        or os.environ.get("SWE_PATCH_CLASSIFICATION_REWARD_METRIC")
        or metadata.get("patch_classification_reward_metric")
        or "balanced_accuracy"
    )


def _base_gold_reward_from_validation(validation: dict[str, Any]) -> dict[str, float]:
    base_reward = 1.0 if validation.get("base_clean_fail") else 0.0
    gold_reward = 1.0 if validation.get("gold_pass") is True else 0.0
    patch_reward = 1.0 if validation.get("label") == "gold_validated" else 0.0
    return {
        "base_reward": base_reward,
        "gold_reward": gold_reward,
        "patch_reward": patch_reward,
    }


def _patch_classification_raw_reward(
    base_reward: float,
    gold_reward: float,
    patch_cls_reward: float,
    *,
    base_infra_failure: bool = False,
) -> float:
    try:
        threshold = float(os.environ.get("SWE_PATCH_CLASSIFICATION_POST_REWARD_BALANCED_ACC_THRESHOLD", "1.0"))
    except ValueError:
        threshold = 1.0
    try:
        partial_threshold = float(
            os.environ.get(
                "SWE_PATCH_CLASSIFICATION_POST_REWARD_PARTIAL_BALANCED_ACC_THRESHOLD",
                "1.0",
            )
        )
    except ValueError:
        partial_threshold = 1.0
    try:
        success_score = float(os.environ.get("SWE_PATCH_CLASSIFICATION_POST_REWARD_SUCCESS_SCORE", "1.0"))
    except ValueError:
        success_score = 1.0
    try:
        partial_score = float(os.environ.get("SWE_PATCH_CLASSIFICATION_POST_REWARD_PARTIAL_SCORE", "0.5"))
    except ValueError:
        partial_score = 0.5
    try:
        gold_score = float(os.environ.get("SWE_PATCH_CLASSIFICATION_POST_REWARD_GOLD_SCORE", "0.2"))
    except ValueError:
        gold_score = 0.2
    try:
        base_score = float(os.environ.get("SWE_PATCH_CLASSIFICATION_POST_REWARD_BASE_SCORE", "0.05"))
    except ValueError:
        base_score = 0.05

    clean_base_success = base_reward > 0.0 and patch_cls_reward >= threshold
    clean_base_partial = base_reward > 0.0 and patch_cls_reward >= partial_threshold
    base_infra_success = base_infra_failure and patch_cls_reward >= 1.0
    if gold_reward > 0.0 and (clean_base_success or base_infra_success):
        return success_score
    if gold_reward > 0.0 and clean_base_partial:
        return partial_score
    if base_reward > 0.0 and gold_reward > 0.0:
        return gold_score
    if base_reward > 0.0:
        return base_score
    return 0.0


def _patch_classification_reward_dict(
    base_reward: float,
    gold_reward: float,
    patch_cls_reward: float,
    *,
    base_infra_failure: bool = False,
) -> dict[str, float]:
    return {
        "base_reward": base_reward,
        "gold_reward": gold_reward,
        "patch_cls_reward": patch_cls_reward,
        "raw_reward": _patch_classification_raw_reward(
            base_reward,
            gold_reward,
            patch_cls_reward,
            base_infra_failure=base_infra_failure,
        ),
    }


def _patch_classification_error_reward(sample: Sample, details: dict[str, Any]) -> dict[str, float]:
    reward = _patch_classification_reward_dict(0.0, 0.0, 0.0)
    details["rewards"] = reward
    details["final_reward"] = reward
    _write_patch_classification_reward_details(sample, details)
    return reward


def _patch_classification_not_submitted_reward(sample: Sample, details: dict[str, Any]) -> dict[str, float]:
    """Fixed negative reward for trajectories that never reached Submitted.

    These are aborted / format-error / limit-exceeded rollouts. Giving them a
    explicit penalty (instead of the 0.0 an unresolved-but-submitted
    sample gets) prevents the group-normalizer from treating "gave up" and
    "tried but wrong" as identical, which otherwise lets a degenerate
    early-EOS / no-tool-call policy escape penalty and self-reinforce.
    """
    try:
        penalty = float(os.environ.get("SWE_PATCH_CLASSIFICATION_NOT_SUBMITTED_SCORE", "-1.0"))
    except ValueError:
        penalty = -1.0
    reward = {
        "base_reward": 0.0,
        "gold_reward": 0.0,
        "patch_cls_reward": 0.0,
        "raw_reward": penalty,
    }
    details["rewards"] = reward
    details["final_reward"] = reward
    _write_patch_classification_reward_details(sample, details)
    return reward


def _base_gold_error_reward(sample: Sample, details: dict[str, Any]) -> dict[str, float]:
    reward = {
        "base_reward": 0.0,
        "gold_reward": 0.0,
        "patch_reward": 0.0,
    }
    details["rewards"] = reward
    details["final_reward"] = reward
    _write_patch_classification_reward_details(sample, details)
    return reward


def _run_test_patch_verdict_pass(result: dict[str, Any]) -> bool:
    """True iff a run_test_patch_official result is a clean PASS (no fail/error)."""
    if result.get("verdict") == "pass":
        return True
    return bool(result.get("passed")) and not result.get("failed") and not result.get("errors")


def _compute_patch_classification_reward_testpatch(
    args,
    sample: Sample,
    *,
    existing_env,
    details: dict[str, Any],
    instance: dict[str, Any],
    instance_id: str,
    test_patch: str,
    test_command: str,
    candidates: list[dict[str, Any]],
    rollout_validation: dict[str, Any],
    reward_start_time: float,
) -> dict[str, float]:
    """Resolve-style reward: the generated artifact is a git-diff ``test_patch`` (may edit existing
    files), not a single-file test body. Reuses the SAME 0/0.05/0.2/0.5/1.0 raw_reward formula as the
    v6 single-file path; only the apply mechanism differs (official eval git-applies the diff).

    base/gold reuse: the resolve rollout already knows ``base_clean_fail`` (from the agent's own
    write_test_patch verify). We reuse it and only spend the verifier sandbox on gold + candidates.
    gold-fail short-circuits before any candidate eval. Uses the warm rollout verifier sandbox.
    """
    _ensure_swe_harness_on_path()
    from minisweagent.run.benchmarks.gentest import (
        run_shared_test_patch_official,
        run_test_patch_official,
    )

    install_timeout = int(
        sample.metadata.get("patch_classification_test_timeout")
        or os.environ.get("SWE_PATCH_CLASSIFICATION_TEST_TIMEOUT", "600")
    )
    include_gold = _bool_from_metadata_or_env(
        sample.metadata, "patch_classification_include_gold", "SWE_PATCH_CLASSIFICATION_INCLUDE_GOLD", True
    )
    require_gold_validated = _bool_from_metadata_or_env(
        sample.metadata,
        "patch_classification_require_gold_validated",
        "SWE_PATCH_CLASSIFICATION_REQUIRE_GOLD_VALIDATED",
        False,
    )
    details["mode"] = "patch_classification_testpatch"
    details["test_patch_chars"] = len(test_patch)
    details["test_command"] = test_command
    shared_verify = bool(sample.metadata.get("shared_verify"))
    details["shared_verify"] = shared_verify

    # base_reward from the rollout's own verify (no recompute). base_infra: unknown here -> False.
    base_reward = 1.0 if rollout_validation.get("base_clean_fail") else 0.0
    base_infra_failure = False
    details["base_clean_fail"] = bool(rollout_validation.get("base_clean_fail"))
    details["validation_reused_from_rollout"] = True

    warmed = {"done": bool(sample.metadata.get("patch_classification_verifier_warmed"))}

    def verify(patch: str) -> dict:
        if shared_verify:
            result = run_shared_test_patch_official(
                existing_env,
                instance,
                patch,
                test_command,
                install_timeout,
                skip_install=warmed["done"],
                restore_agent_workspace=False,
            )
        else:
            result = run_test_patch_official(
                existing_env,
                instance,
                patch,
                test_command,
                install_timeout,
                skip_install=warmed["done"],
            )
        warmed["done"] = True
        return result

    # gold: gold source fix + generated test_patch -> expect PASS. gold read ONLY in reward.
    gold_reward = 0.0
    gold_pass: bool | None = None
    gold_src = str(instance.get("patch") or "")
    if include_gold and gold_src.strip():
        combined = gold_src.rstrip("\n") + "\n" + test_patch.rstrip("\n") + "\n"
        try:
            gres = verify(combined)
            gold_pass = _run_test_patch_verdict_pass(gres)
            gold_reward = 1.0 if gold_pass else 0.0
            details["gold_pass"] = gold_pass
            details["gold_verdict"] = gres.get("verdict")
        except Exception as exc:  # noqa: BLE001
            details["gold_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    if require_gold_validated and gold_pass is False:
        rewards = _patch_classification_reward_dict(base_reward, gold_reward, 0.0, base_infra_failure=base_infra_failure)
        details["gold_validation_gate"] = "failed"
        details["patch_evaluation_skipped"] = "gold_failed"
        details["rewards"] = rewards
        details["final_reward"] = rewards
        details["reward_eval_time"] = time.time() - reward_start_time
        _write_patch_classification_reward_details(sample, details)
        return rewards

    # candidate classification: each candidate source fix + generated test_patch. PASS => the test
    # predicts "resolved" for that candidate.
    records: list[dict[str, Any]] = []
    by_patch: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        by_patch.setdefault(candidate["patch_sha256"], candidate)
    for patch_sha256, candidate in by_patch.items():
        combined = candidate["patch_text"].rstrip("\n") + "\n" + test_patch.rstrip("\n") + "\n"
        try:
            cres = verify(combined)
            pred_resolved = _run_test_patch_verdict_pass(cres)
        except Exception as exc:  # noqa: BLE001
            details.setdefault("candidate_errors", {})[patch_sha256[:12]] = f"{type(exc).__name__}: {str(exc)[:120]}"
            continue
        record = {
            "instance_id": candidate["instance_id"],
            "patch_sha256": patch_sha256,
            "oracle_resolved": candidate["oracle_resolved"],
            "generated_pred_resolved": pred_resolved,
        }
        record["agreement"] = record["generated_pred_resolved"] == record["oracle_resolved"]
        records.append(record)

    if include_gold and gold_pass is not None:
        gold_record = {
            "instance_id": instance_id,
            "patch_sha256": "gold",
            "oracle_resolved": True,
            "generated_pred_resolved": bool(gold_pass),
            "is_gold": True,
        }
        gold_record["agreement"] = gold_record["generated_pred_resolved"] == gold_record["oracle_resolved"]
        records.append(gold_record)
        details["gold_in_metrics"] = True

    metrics = patch_classification_metrics(records)
    metric_name = _patch_classification_metric_name(args, sample.metadata)
    if metric_name not in metrics:
        raise ValueError(f"unknown patch classification reward metric: {metric_name}")
    patch_cls_reward = float(metrics[metric_name])
    rewards = _patch_classification_reward_dict(
        base_reward, gold_reward, patch_cls_reward, base_infra_failure=base_infra_failure
    )
    details.update(metrics)
    details["reward_metric"] = metric_name
    details["rewards"] = rewards
    details["final_reward"] = rewards
    details["records"] = records
    details["reward_eval_time"] = time.time() - reward_start_time
    _write_patch_classification_reward_details(sample, details)
    logger.info(
        "[SWE_REWARD] patch_classification_testpatch %s: base=%.1f gold=%.1f patch_metric=%s "
        "patch_cls=%.4f raw=%.2f total=%s",
        instance_id, base_reward, gold_reward, metric_name, patch_cls_reward,
        rewards["raw_reward"], metrics["total"],
    )
    return rewards


def _compute_patch_classification_reward_sync(
    args,
    sample: Sample,
    *,
    existing_env=None,
) -> dict[str, float]:
    metadata = sample.metadata or {}
    instance_id = metadata.get("instance_id", "unknown")
    reward_start_time = time.time()
    details: dict[str, Any] = {
        "mode": "patch_classification",
        "instance_id": instance_id,
        "error": "",
        "sandbox_reused_from_rollout": existing_env is not None,
        "timings": {},
    }
    timings = details["timings"]

    gentest_record = metadata.get("gentest_record") if isinstance(metadata.get("gentest_record"), dict) else {}
    exit_status = str(metadata.get("exit_status") or gentest_record.get("agent_exit_status") or "Unknown")
    format_error_count = int(metadata.get("format_error_count") or gentest_record.get("format_error_count") or 0)
    require_submitted = _bool_from_metadata_or_env(
        metadata,
        "patch_classification_require_submitted",
        "SWE_PATCH_CLASSIFICATION_REQUIRE_SUBMITTED",
        False,
    )
    max_format_errors = _int_from_metadata_or_env(
        metadata,
        "patch_classification_max_format_errors",
        "SWE_PATCH_CLASSIFICATION_MAX_FORMAT_ERRORS",
        -1,
    )
    details.update(
        {
            "exit_status": exit_status,
            "format_error_count": format_error_count,
            "require_submitted": require_submitted,
            "max_format_errors": max_format_errors,
        }
    )
    if exit_status != SUBMITTED_EXIT_STATUS:
        details["error"] = f"trajectory did not submit: exit_status={exit_status}"
        return _patch_classification_not_submitted_reward(sample, details)
    if max_format_errors >= 0 and format_error_count > max_format_errors:
        details["error"] = f"trajectory exceeded format-error budget: {format_error_count}>{max_format_errors}"
        return _patch_classification_error_reward(sample, details)

    code = extract_generated_test_code_from_sample(sample)
    candidates = load_patch_classification_candidates(metadata)
    loaded_candidate_count = len(candidates)
    max_candidates = _int_from_metadata_or_env(
        metadata,
        "patch_classification_max_candidates",
        "SWE_PATCH_CLASSIFICATION_MAX_CANDIDATES",
        5,
    )
    candidates = cap_patch_classification_candidates(candidates, max_candidates)
    details["test_code_chars"] = len(code)
    details["candidate_rows"] = len(candidates)
    details["candidate_rows_loaded"] = loaded_candidate_count
    details["candidate_max_candidates"] = max_candidates
    details["candidate_unique_patches"] = len({c["patch_sha256"] for c in candidates})

    # Resolve-style git-diff test_patch: the artifact edits existing files, so it can't be written
    # as a single-file body. Route to the test_patch branch (same raw_reward formula, diff apply).
    resolve_test_patch = str(metadata.get("test_patch") or "").strip()
    resolve_mode = bool(resolve_test_patch) and _bool_from_metadata_or_env(
        metadata, "patch_classification_testpatch_mode", "SWE_PATCH_CLASSIFICATION_TESTPATCH_MODE", True
    )
    if resolve_mode:
        if existing_env is None:
            details["error"] = "resolve test_patch reward requires the warm rollout verifier env"
            return _patch_classification_error_reward(sample, details)
        if not candidates:
            details["error"] = "no labeled patch classification candidates"
            return _patch_classification_error_reward(sample, details)
        rollout_record = metadata.get("gentest_record") if isinstance(metadata.get("gentest_record"), dict) else {}
        rollout_validation = (
            rollout_record.get("validation") if isinstance(rollout_record.get("validation"), dict) else {}
        )
        instance = copy.deepcopy(metadata.get("instance") if isinstance(metadata.get("instance"), dict) else metadata)
        instance.setdefault("instance_id", instance_id)
        test_command = str(metadata.get("test_command") or metadata.get("generated_test_command") or "").strip()
        return _compute_patch_classification_reward_testpatch(
            args,
            sample,
            existing_env=existing_env,
            details=details,
            instance=instance,
            instance_id=instance_id,
            test_patch=resolve_test_patch,
            test_command=test_command,
            candidates=candidates,
            rollout_validation=rollout_validation,
            reward_start_time=reward_start_time,
        )

    if not code.strip():
        details["error"] = "missing generated test artifact"
        return _patch_classification_error_reward(sample, details)
    if not candidates:
        details["error"] = "no labeled patch classification candidates"
        return _patch_classification_error_reward(sample, details)

    _ensure_swe_harness_on_path()
    try:
        from minisweagent.run.benchmarks import gentest as gentest_module
        from minisweagent.run.benchmarks import patch_gentest as patch_gentest_module
        from minisweagent.run.benchmarks.swerebench import _resolve_per_instance_api_base, get_sb_environment
    except Exception as exc:  # noqa: BLE001
        details["error"] = f"failed to import swe_harness patch gentest logic: {type(exc).__name__}: {exc}"
        return _patch_classification_error_reward(sample, details)

    instance = copy.deepcopy(metadata.get("instance") if isinstance(metadata.get("instance"), dict) else metadata)
    instance.setdefault("instance_id", instance_id)
    rollout_record = metadata.get("gentest_record") if isinstance(metadata.get("gentest_record"), dict) else {}
    execution_contract_mode = bool(
        metadata.get("agent_inferred_execution_contract")
        or rollout_record.get("agent_inferred_execution_contract")
    )

    test_timeout = int(
        metadata.get("patch_classification_test_timeout")
        or getattr(args, "swe_patch_classification_test_timeout", 0)
        or os.environ.get("SWE_PATCH_CLASSIFICATION_TEST_TIMEOUT", "60")
    )
    requested_test_file = (
        metadata.get("patch_classification_test_file")
        or metadata.get("test_filename")
        or metadata.get("requested_test_filename")
        or gentest_module.TEST_FILE
    )
    test_file = (
        str(requested_test_file)
        if execution_contract_mode
        else gentest_module.generated_test_file_for_instance(instance, requested_test_file)
    )
    test_command = str(
        metadata.get("generated_test_command")
        or rollout_record.get("generated_test_command")
        or ""
    ).strip()
    validate_test = _bool_from_metadata_or_env(
        metadata,
        "patch_classification_validate_test",
        "SWE_PATCH_CLASSIFICATION_VALIDATE_TEST",
        True,
    )
    gold_eval = _bool_from_metadata_or_env(
        metadata,
        "patch_classification_gold_eval",
        "SWE_PATCH_CLASSIFICATION_GOLD_EVAL",
        True,
    )
    apply_test_patch_at_start = _bool_from_metadata_or_env(
        metadata,
        "apply_test_patch_at_start",
        "GENTEST_APPLY_TEST_PATCH_AT_START",
        False,
    )
    require_gold_validated = _bool_from_metadata_or_env(
        metadata,
        "patch_classification_require_gold_validated",
        "SWE_PATCH_CLASSIFICATION_REQUIRE_GOLD_VALIDATED",
        False,
    )

    # The rollout phase already ran validate_generated_test(gold_eval=True) and stored the
    # result on sample.metadata["gentest_record"]["validation"]. Reuse it instead of building a
    # fresh sandbox and recomputing base/gold: only candidate-patch classification needs a live
    # env, and gold-failed samples short-circuit before any sandbox is created.
    rollout_validation = (
        rollout_record.get("validation") if isinstance(rollout_record.get("validation"), dict) else {}
    )
    official_verify_format = bool(rollout_record.get("official_verify_format"))
    official_verifier = (
        rollout_record.get("official_verifier")
        if isinstance(rollout_record.get("official_verifier"), dict)
        else {}
    )
    official_skip_install = True if official_verify_format and official_verifier else None
    reuse_validation = bool(rollout_validation) and _bool_from_metadata_or_env(
        metadata,
        "patch_classification_reuse_rollout_validation",
        "SWE_PATCH_CLASSIFICATION_REUSE_ROLLOUT_VALIDATION",
        True,
    )

    def _base_infra_failure(validation: dict[str, Any]) -> bool:
        return validation.get("base_validation_label") == "base_infra_failure" or (
            "base_validation_label" not in validation and validation.get("label") == "base_infra_failure"
        )

    def _gold_failed_result(base_reward: float, gold_reward: float, base_infra_failure: bool) -> dict[str, float]:
        rewards = _patch_classification_reward_dict(
            base_reward,
            gold_reward,
            0.0,
            base_infra_failure=base_infra_failure,
        )
        details["gold_validation_gate"] = "failed"
        details["patch_evaluation_skipped"] = "gold_failed"
        details["rewards"] = rewards
        details["final_reward"] = rewards
        details["test_filename"] = test_file
        details["records"] = []
        details["reward_eval_time"] = time.time() - reward_start_time
        _write_patch_classification_reward_details(sample, details)
        logger.info(
            "[SWE_REWARD] patch_classification %s: base=%.1f gold=%.1f patch_cls=%.1f raw=%.2f skipped=gold_failed",
            instance_id,
            base_reward,
            gold_reward,
            0.0,
            rewards["raw_reward"],
        )
        return rewards

    # Fast path: gate on the reused rollout validation before spending a sandbox.
    if reuse_validation:
        validation = rollout_validation
        details["validation"] = validation
        details["validation_reused_from_rollout"] = True
        base_gold_rewards = _base_gold_reward_from_validation(validation)
        base_reward = base_gold_rewards["base_reward"]
        gold_reward = base_gold_rewards["gold_reward"]
        base_infra_failure = _base_infra_failure(validation)
        details["base_infra_failure"] = base_infra_failure
        if require_gold_validated and validation.get("gold_pass") is False:
            return _gold_failed_result(base_reward, gold_reward, base_infra_failure)

    owns_env = existing_env is None
    env = existing_env
    records: list[dict[str, Any]] = []
    try:
        if owns_env:
            config = _build_patch_classification_config(args, metadata, gentest_module, patch_gentest_module)
            _resolve_per_instance_api_base(config, instance_id)
            install_timeout = int(
                metadata.get("patch_classification_install_timeout")
                or getattr(args, "swe_patch_classification_install_timeout", 0)
                or os.environ.get("SWE_PATCH_CLASSIFICATION_INSTALL_TIMEOUT", "600")
            )
            _check_reward_deadline(reward_start_time, "sandbox creation")
            phase_started = time.perf_counter()
            env = get_sb_environment(config, instance)
            timings["sandbox_create_sec"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            gentest_module.reset_base(env)
            if install_timeout > 0:
                gentest_module.restore_env(env, instance, install_timeout)
            timings["sandbox_prepare_sec"] = time.perf_counter() - phase_started
        else:
            timings["sandbox_create_sec"] = 0.0
            timings["sandbox_prepare_sec"] = 0.0
        _check_reward_deadline(reward_start_time, "candidate evaluation")

        if not reuse_validation:
            validation = {}
            if validate_test:
                validation = gentest_module.validate_generated_test(
                    env,
                    instance,
                    code,
                    filename=test_file,
                    test_command=test_command or None,
                    test_timeout=test_timeout,
                    gold_eval=gold_eval,
                    oracle=None,
                    apply_test_patch_at_start=apply_test_patch_at_start,
                    official_verify_format=official_verify_format,
                    official_skip_install=official_skip_install,
                )
            details["validation"] = validation
            base_gold_rewards = _base_gold_reward_from_validation(validation)
            base_reward = base_gold_rewards["base_reward"]
            gold_reward = base_gold_rewards["gold_reward"]
            base_infra_failure = _base_infra_failure(validation)
            details["base_infra_failure"] = base_infra_failure

            if require_gold_validated and validation.get("gold_pass") is False:
                return _gold_failed_result(base_reward, gold_reward, base_infra_failure)

        by_patch: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            by_patch.setdefault(candidate["patch_sha256"], candidate)

        patch_predictions: dict[str, dict[str, Any]] = {}
        candidate_timings: dict[str, float] = {}
        candidate_started = time.perf_counter()
        for patch_sha256, candidate in by_patch.items():
            _check_reward_deadline(reward_start_time, f"candidate {patch_sha256[:12]}")
            patch_started = time.perf_counter()
            candidate_validation = patch_gentest_module.validate_candidate_patch(
                env,
                instance,
                code,
                filename=test_file,
                test_command=test_command or None,
                patch_text=candidate["patch_text"],
                test_timeout=test_timeout,
                apply_test_patch_at_start=apply_test_patch_at_start,
                official_verify_format=official_verify_format,
                official_skip_install=official_skip_install,
            )
            candidate_timings[patch_sha256] = time.perf_counter() - patch_started
            patch_predictions[patch_sha256] = {
                "generated_pred_resolved": bool(candidate_validation.get("candidate_pass")),
                "candidate_validation": candidate_validation,
            }
        timings["candidate_evaluation_sec"] = time.perf_counter() - candidate_started
        timings["candidate_sec_by_patch"] = candidate_timings

        for candidate in candidates:
            prediction = patch_predictions[candidate["patch_sha256"]]
            record = {
                "instance_id": candidate["instance_id"],
                "sample_idx": candidate.get("sample_idx"),
                "run_dir": candidate.get("run_dir"),
                "source_path": candidate.get("source_path"),
                "patch_sha256": candidate["patch_sha256"],
                "patch_chars": candidate["patch_chars"],
                "oracle_resolved": candidate["oracle_resolved"],
                "generated_pred_resolved": prediction["generated_pred_resolved"],
                "candidate_validation": prediction["candidate_validation"],
            }
            record["agreement"] = record["generated_pred_resolved"] == record["oracle_resolved"]
            records.append(record)

        include_gold = _bool_from_metadata_or_env(
            metadata,
            "patch_classification_include_gold",
            "SWE_PATCH_CLASSIFICATION_INCLUDE_GOLD",
            True,
        )
        if include_gold and validation.get("gold_applied") and validation.get("gold_pass") is not None:
            gold_record = {
                "instance_id": instance_id,
                "sample_idx": None,
                "run_dir": None,
                "source_path": "gold",
                "patch_sha256": "gold",
                "patch_chars": 0,
                "oracle_resolved": True,
                "generated_pred_resolved": bool(validation.get("gold_pass")),
                "is_gold": True,
            }
            gold_record["agreement"] = (
                gold_record["generated_pred_resolved"] == gold_record["oracle_resolved"]
            )
            records.append(gold_record)
            details["gold_in_metrics"] = True

        metrics = patch_classification_metrics(records)
        metric_name = _patch_classification_metric_name(args, metadata)
        if metric_name not in metrics:
            raise ValueError(f"unknown patch classification reward metric: {metric_name}")
        patch_cls_reward = float(metrics[metric_name])
        rewards = _patch_classification_reward_dict(
            base_reward,
            gold_reward,
            patch_cls_reward,
            base_infra_failure=base_infra_failure,
        )

        details.update(metrics)
        details["reward_metric"] = metric_name
        details["rewards"] = rewards
        details["final_reward"] = rewards
        details["test_filename"] = test_file
        details["records"] = records
        details["reward_eval_time"] = time.time() - reward_start_time
        timings["total_sec"] = details["reward_eval_time"]
        _write_patch_classification_reward_details(sample, details)
        logger.info(
            (
                "[SWE_REWARD] patch_classification %s: base=%.1f gold=%.1f "
                "patch_metric=%s patch_cls=%.4f raw=%.2f total=%s acc=%.4f bal_acc=%.4f"
            ),
            instance_id,
            base_reward,
            gold_reward,
            metric_name,
            patch_cls_reward,
            rewards["raw_reward"],
            metrics["total"],
            metrics["accuracy"],
            metrics["balanced_accuracy"],
        )
        return rewards
    except Exception as exc:  # noqa: BLE001
        details["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        details["traceback"] = traceback.format_exc()[-4000:]
        details["reward_eval_time"] = time.time() - reward_start_time
        logger.error(f"[SWE_REWARD] patch_classification ERROR for {instance_id}: {exc}\n{traceback.format_exc()}")
        return _patch_classification_error_reward(sample, details)
    finally:
        if owns_env and env is not None:
            cleanup_started = time.perf_counter()
            try:
                env.cleanup()
            except Exception as cleanup_exc:
                logger.warning("[SWE_REWARD] sandbox cleanup failed for %s: %s", instance_id, cleanup_exc)
            finally:
                timings["sandbox_cleanup_sec"] = time.perf_counter() - cleanup_started


def compute_patch_classification_reward_in_env(args, sample: Sample, env) -> dict[str, float]:
    """Compute patch-classification reward without taking ownership of ``env``."""
    if env is None:
        raise ValueError("existing rollout environment is required")
    try:
        return _compute_patch_classification_reward_sync(args, sample, existing_env=env)
    except Exception as exc:  # keep reward failures from retrying a completed rollout
        instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
        details = {
            "mode": "patch_classification",
            "instance_id": instance_id,
            "sandbox_reused_from_rollout": True,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "traceback": traceback.format_exc()[-4000:],
        }
        logger.error(
            "[SWE_REWARD] in-environment patch classification failed for %s: %s",
            instance_id,
            exc,
            exc_info=True,
        )
        return _patch_classification_error_reward(sample, details)


async def compute_patch_classification_reward(args, sample: Sample) -> dict[str, float]:
    """Reward a generated test by its binary-classification accuracy on labeled patches.

    The generated test is run on each corresponding candidate patch. Passing
    means the test predicts "resolved"; failing means it predicts "failed".
    Labels come from ``oracle_resolved`` / ``label`` / ``verify_status`` on the
    candidate rows. Returns four keys:
      - base_reward: 1.0 iff the test cleanly fails on the base repo;
      - gold_reward: 1.0 iff the test passes after the gold patch;
      - patch_cls_reward: candidate-patch classification accuracy;
      - raw_reward: state-derived scalar before group normalization. A
        gold-passing base-infra test earns the success score only when its
        patch-classification metric reaches the configured success threshold.
    """
    return await _run_cleanup_owned_sync(
        _reward_executor(args),
        _compute_patch_classification_reward_sync,
        args,
        sample,
    )


def _compute_base_gold_reward_sync(args, sample: Sample) -> dict[str, float]:
    metadata = sample.metadata or {}
    instance_id = metadata.get("instance_id", "unknown")
    reward_start_time = time.time()
    details: dict[str, Any] = {
        "mode": "base_gold",
        "instance_id": instance_id,
        "error": "",
    }

    code = extract_generated_test_code_from_sample(sample)
    details["test_code_chars"] = len(code)
    details["candidate_rows"] = 0
    details["candidate_unique_patches"] = 0

    gentest_record = metadata.get("gentest_record") if isinstance(metadata.get("gentest_record"), dict) else {}
    cached_validation = gentest_record.get("validation") if isinstance(gentest_record, dict) else None
    if isinstance(cached_validation, dict) and cached_validation:
        validation = copy.deepcopy(cached_validation)
        rewards = _base_gold_reward_from_validation(validation)
        test_file = (
            metadata.get("test_filename")
            or metadata.get("requested_test_filename")
            or gentest_record.get("test_filename")
            or gentest_record.get("requested_test_filename")
            or ""
        )
        details["validation"] = validation
        details["rewards"] = rewards
        details["final_reward"] = rewards
        details["test_filename"] = test_file
        details["base_clean_fail"] = bool(validation.get("base_clean_fail"))
        details["gold_pass"] = validation.get("gold_pass")
        details["validation_label"] = validation.get("label")
        details["validation_source"] = "gentest_record"
        details["reward_eval_time"] = time.time() - reward_start_time
        _write_patch_classification_reward_details(sample, details)
        logger.info(
            "[SWE_REWARD] base_gold cached %s: base=%.1f gold=%.1f patch_reward=%.1f label=%s",
            instance_id,
            rewards["base_reward"],
            rewards["gold_reward"],
            rewards["patch_reward"],
            validation.get("label"),
        )
        return rewards

    if not code.strip():
        details["error"] = "missing generated test code"
        return _base_gold_error_reward(sample, details)

    _ensure_swe_harness_on_path()
    try:
        from minisweagent.run.benchmarks import gentest as gentest_module
        from minisweagent.run.benchmarks import patch_gentest as patch_gentest_module
        from minisweagent.run.benchmarks.swerebench import _resolve_per_instance_api_base, get_sb_environment
    except Exception as exc:  # noqa: BLE001
        details["error"] = f"failed to import swe_harness gentest logic: {type(exc).__name__}: {exc}"
        return _base_gold_error_reward(sample, details)

    instance = copy.deepcopy(metadata.get("instance") if isinstance(metadata.get("instance"), dict) else metadata)
    instance.setdefault("instance_id", instance_id)
    config = _build_patch_classification_config(args, metadata, gentest_module, patch_gentest_module)
    _resolve_per_instance_api_base(config, instance_id)

    test_timeout = int(
        metadata.get("patch_classification_test_timeout")
        or getattr(args, "swe_patch_classification_test_timeout", 0)
        or os.environ.get("SWE_PATCH_CLASSIFICATION_TEST_TIMEOUT", "60")
    )
    install_timeout = int(
        metadata.get("patch_classification_install_timeout")
        or getattr(args, "swe_patch_classification_install_timeout", 0)
        or os.environ.get("SWE_PATCH_CLASSIFICATION_INSTALL_TIMEOUT", "600")
    )
    requested_test_file = (
        metadata.get("patch_classification_test_file")
        or metadata.get("test_filename")
        or metadata.get("requested_test_filename")
        or gentest_module.TEST_FILE
    )
    test_file = gentest_module.generated_test_file_for_instance(instance, requested_test_file)
    gold_eval = _bool_from_metadata_or_env(
        metadata,
        "patch_classification_gold_eval",
        "SWE_PATCH_CLASSIFICATION_GOLD_EVAL",
        True,
    )
    apply_test_patch_at_start = _bool_from_metadata_or_env(
        metadata,
        "apply_test_patch_at_start",
        "GENTEST_APPLY_TEST_PATCH_AT_START",
        False,
    )

    env = None
    try:
        _check_reward_deadline(reward_start_time, "sandbox creation")
        env = get_sb_environment(config, instance)
        gentest_module.reset_base(env)
        if install_timeout > 0:
            gentest_module.restore_env(env, instance, install_timeout)
        _check_reward_deadline(reward_start_time, "base/gold validation")

        validation = gentest_module.validate_generated_test(
            env,
            instance,
            code,
            filename=test_file,
            test_timeout=test_timeout,
            gold_eval=gold_eval,
            oracle=None,
            apply_test_patch_at_start=apply_test_patch_at_start,
        )
        rewards = _base_gold_reward_from_validation(validation)

        details["validation"] = validation
        details["rewards"] = rewards
        details["final_reward"] = rewards
        details["test_filename"] = test_file
        details["base_clean_fail"] = bool(validation.get("base_clean_fail"))
        details["gold_pass"] = validation.get("gold_pass")
        details["validation_label"] = validation.get("label")
        details["reward_eval_time"] = time.time() - reward_start_time
        _write_patch_classification_reward_details(sample, details)
        logger.info(
            "[SWE_REWARD] base_gold %s: base=%.1f gold=%.1f patch_reward=%.1f label=%s",
            instance_id,
            rewards["base_reward"],
            rewards["gold_reward"],
            rewards["patch_reward"],
            validation.get("label"),
        )
        return rewards
    except Exception as exc:  # noqa: BLE001
        details["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        details["traceback"] = traceback.format_exc()[-4000:]
        details["reward_eval_time"] = time.time() - reward_start_time
        logger.error(f"[SWE_REWARD] base_gold ERROR for {instance_id}: {exc}\n{traceback.format_exc()}")
        return _base_gold_error_reward(sample, details)
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception as cleanup_exc:
                logger.warning("[SWE_REWARD] sandbox cleanup failed for %s: %s", instance_id, cleanup_exc)


async def compute_base_gold_reward(args, sample: Sample) -> dict[str, float]:
    """Reward a generated test using only base clean-fail and gold-pass validation."""
    return await _run_cleanup_owned_sync(
        _reward_executor(args),
        _compute_base_gold_reward_sync,
        args,
        sample,
    )


async def compute_simple_reward(args, sample: Sample) -> float:
    """
    Compute simple binary reward based on SWE-bench resolution.

    Returns 1.0 if FULL resolution (all F2P tests pass and all P2P tests maintained),
    -1.0 otherwise.

    Args:
        args: Training arguments
        sample: Sample with response and metadata

    Returns:
        Binary reward (-1.0 or 1.0)
    """
    # Extract final output from metadata
    final_output = sample.metadata.get("final_output", "")

    if not final_output:
        return -1.0

    # Extract patch
    patch = extract_patch_from_output(final_output)

    if not is_valid_patch(patch):
        return -1.0

    # Get test information from metadata
    instance_id = sample.metadata.get("instance_id", "unknown")
    repo = sample.metadata.get("repo", "")
    base_commit = sample.metadata.get("base_commit", "")

    # Get SWE-bench test lists
    fail_to_pass = sample.metadata.get("FAIL_TO_PASS", [])
    pass_to_pass = sample.metadata.get("PASS_TO_PASS", [])

    # Handle JSON string format
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = []
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = json.loads(pass_to_pass)
        except json.JSONDecodeError:
            pass_to_pass = []

    # Run tests with full instance data for proper Docker image resolution
    test_results = await run_tests_in_docker(
        instance_id=instance_id,
        patch=patch,
        repo=repo,
        base_commit=base_commit,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        instance=sample.metadata,
    )

    # final_reward = 1.0 if test_results["resolved"] else 0.0
    # final_reward *= 10.0

    # Assign a positive reward for resolved tests and a configurable negative reward
    # for unresolved tests (controlled via SWE_UNRESOLVED_REWARD env var, default -1.0)
    unresolved_reward = float(os.environ.get("SWE_UNRESOLVED_REWARD", "-1.0"))
    final_reward = 1.0 if test_results["resolved"] else unresolved_reward

    # update test results with final reward for debugging
    test_results["final_reward"] = final_reward
    # write test results to trajectory path for debugging
    trajectory_path = sample.metadata.get("trajectory_path", "unknown")
    if trajectory_path != "unknown":
        try:
            with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                json.dump(test_results, f)
                logger.info(f"test_results: {json.dumps(test_results)}")
        except Exception as e:
            logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")
    
    return final_reward


# Only "Submitted" indicates the agent finished the trajectory on its own.
# Any other exit_status (LimitsExceeded, TimeExceeded, Timeout_LLM, Error_LLM,
# Unknown, ...) is treated as a truncated / incomplete trajectory.
SUBMITTED_EXIT_STATUS = "Submitted"


async def compute_simple_truncated_reward(args, sample: Sample) -> float:
    """
    Compute binary reward like ``compute_simple_reward`` but penalize truncated
    trajectories more strongly.

    Reward scheme:
    - -2.0 if the trajectory did not finish with exit_status == "Submitted"
      (covers LimitsExceeded, TimeExceeded, LLM errors, unknown, etc.)
    -  1.0 if the patch fully resolves the SWE-bench instance
    - -1.0 otherwise
    """
    exit_status = (
        sample.metadata.get("exit_status", "Unknown") if sample.metadata else "Unknown"
    )

    # Anything other than a clean "Submitted" is treated as truncated.
    if exit_status != SUBMITTED_EXIT_STATUS:
        instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
        logger.info(
            f"[SWE_REWARD] Trajectory not submitted (exit_status={exit_status}) "
            f"for instance_id={instance_id}, assigning reward=-2.0"
        )
        truncated_results = {
            "truncated": True,
            "exit_status": exit_status,
            "final_reward": -2.0,
        }
        trajectory_path = sample.metadata.get("trajectory_path", "unknown") if sample.metadata else "unknown"
        if trajectory_path != "unknown":
            try:
                with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                    json.dump(truncated_results, f)
                    logger.info(f"test_results: {json.dumps(truncated_results)}")
            except Exception as e:
                logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")
        return -2.0

    # Otherwise fall back to the standard simple binary reward (+1 / -1).
    return await compute_simple_reward(args, sample)


async def compute_simple_truncated_zero_reward(args, sample: Sample) -> float:
    """
    Variant of ``compute_simple_truncated_reward`` that uses 0.0 (instead of
    -1.0) as the default penalty for unresolved-but-submitted trajectories.

    Reward scheme:
    - -2.0 if the trajectory did not finish with exit_status == "Submitted"
      (covers LimitsExceeded, TimeExceeded, LLM errors, unknown, etc.)
    -  1.0 if the patch fully resolves the SWE-bench instance
    -  0.0 otherwise
    """
    exit_status = (
        sample.metadata.get("exit_status", "Unknown") if sample.metadata else "Unknown"
    )

    # Anything other than a clean "Submitted" is treated as truncated.
    if exit_status != SUBMITTED_EXIT_STATUS:
        instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
        logger.info(
            f"[SWE_REWARD] Trajectory not submitted (exit_status={exit_status}) "
            f"for instance_id={instance_id}, assigning reward=-2.0"
        )
        truncated_results = {
            "truncated": True,
            "exit_status": exit_status,
            "final_reward": -2.0,
        }
        trajectory_path = sample.metadata.get("trajectory_path", "unknown") if sample.metadata else "unknown"
        if trajectory_path != "unknown":
            try:
                with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                    json.dump(truncated_results, f)
                    logger.info(f"test_results: {json.dumps(truncated_results)}")
            except Exception as e:
                logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")
        return -2.0

    # Submitted: run the simple binary evaluation, then map -1.0 -> 0.0.
    simple_reward = await compute_simple_reward(args, sample)
    return 1.0 if simple_reward > 0 else 0.0


async def compute_simple_plus_llm_as_judges_reward(args, sample: Sample) -> float:
    """Binary outcome reward (``simple_truncated_zero``) PLUS an LLM-as-judge
    process reward applied to ALL paths.

    Reward scheme (with PRM_ALPHA = alpha):
      - truncated (R_bin = -2.0):              R' = -2.0 + alpha * J
      - submitted, unresolved (R_bin =  0.0):  R' =  0.0 + alpha * J
      - submitted, resolved   (R_bin =  1.0):  R' =  1.0 + alpha * J

    where J in [0, 1] is the aggregated judge score from
    ``swe_prm.score_sample``. With the default ``alpha = 0.5`` the three regions
    [-2.0, -1.5], [0.0, 0.5], [1.0, 1.5] do not overlap, so outcome ordering is
    preserved while GRPO groups gain dense intra-group signal.

    Short-circuits (no API call) when ``PRM_ALPHA == 0``: falls back to pure
    ``compute_simple_truncated_zero_reward``.

    On any PRM failure, falls back to the underlying binary reward unchanged.
    Configuration is read from ``PRM_*`` env vars (see ``swe_prm.py``).
    """
    instance_id = (
        sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
    )

    base_reward = await compute_simple_truncated_zero_reward(args, sample)

    # alpha == 0 -> short-circuit: behave exactly like simple_truncated_zero,
    # skipping the (slow, costly) judge API call entirely.
    try:
        alpha = float(os.environ.get("PRM_ALPHA", "0.5"))
    except ValueError:
        alpha = 0.5
    if alpha == 0.0:
        logger.info(
            f"[PRM] {instance_id}: PRM_ALPHA=0 -> skipping judge, R={base_reward:.3f}"
        )
        if sample.metadata is not None:
            sample.metadata["prm"] = {"called": False, "alpha": 0.0}
        return base_reward

    process_reward = 0.0
    error_str = ""
    scores: dict = {}
    harmful = False
    hallucinated = False
    try:
        from .swe_prm import score_sample

        prm_timeout = float(os.environ.get("PRM_TIMEOUT_TOTAL", "120"))
        grade = await asyncio.wait_for(
            score_sample(sample, timeout=max(prm_timeout * 0.8, 10.0)),
            timeout=prm_timeout,
        )
        process_reward = float(grade.process_reward)
        scores = grade.scores or {}
        harmful = bool(grade.harmful)
        hallucinated = bool(grade.hallucinated)
        error_str = grade.error or ""
        if error_str:
            logger.warning(
                f"[PRM] {instance_id}: judge returned error: {error_str}"
            )
    except asyncio.TimeoutError:
        error_str = "timeout"
        logger.warning(
            f"[PRM] {instance_id}: TIMEOUT after {os.environ.get('PRM_TIMEOUT_TOTAL', '120')}s; J=0.0"
        )
    except Exception as e:
        error_str = f"{type(e).__name__}: {e}"
        logger.warning(f"[PRM] {instance_id}: failed: {error_str}; J=0.0")

    final_reward = base_reward + alpha * process_reward
    logger.info(
        f"[PRM] {instance_id}: R_binary={base_reward:.3f}, J={process_reward:.3f}, "
        f"alpha={alpha}, R_final={final_reward:.3f}"
    )

    # Write structured PRM info onto sample.metadata so that
    # _compute_prm_metrics (rollout.py) can aggregate it into wandb.
    if sample.metadata is not None:
        sample.metadata["prm"] = {
            "called": True,
            "alpha": alpha,
            "base_binary_reward": base_reward,
            "process_reward": process_reward,
            "final_reward": final_reward,
            "contribution": alpha * process_reward,
            "scores": scores,           # {"D1": float, "D3": float, "D5": float, "D6": float}
            "harmful": harmful,
            "hallucinated": hallucinated,
            "error": error_str,         # "" == success, "timeout" == timed-out, else exception
        }

    # Augment the *_rewards.json that compute_simple_reward / the truncated
    # wrapper already wrote, so PRM info is auditable per trajectory.
    trajectory_path = (
        sample.metadata.get("trajectory_path", "unknown")
        if sample.metadata
        else "unknown"
    )
    if trajectory_path != "unknown":
        rewards_json_path = trajectory_path.replace(".json", "_rewards.json")
        try:
            existing: dict = {}
            if os.path.exists(rewards_json_path):
                with open(rewards_json_path) as f:
                    existing = json.load(f)
            existing.update(
                {
                    "base_binary_reward": base_reward,
                    "process_reward": process_reward,
                    "prm_alpha": alpha,
                    "final_reward": final_reward,
                    "judge_details": {
                        "scores": scores,
                        "harmful": harmful,
                        "hallucinated": hallucinated,
                        "error": error_str,
                    },
                }
            )
            with open(rewards_json_path, "w") as f:
                json.dump(existing, f)
        except Exception as e:
            logger.error(
                f"[PRM] Failed to update rewards json {rewards_json_path}: {e}"
            )

    return final_reward


async def compute_pass_rate_reward(args, sample: Sample) -> float:
    """
    Compute reward based on test pass rate.

    This provides a more granular reward signal based on:
    - Fail-to-Pass rate (weighted higher as it's the primary objective)
    - Pass-to-Pass rate (maintenance)

    Formula: 0.7 * F2P_rate + 0.3 * P2P_rate

    Args:
        args: Training arguments
        sample: Sample with response and metadata

    Returns:
        Pass rate reward (0.0 to 1.0)
    """
    final_output = sample.metadata.get("final_output", "")
    #final_output = sample.label # !!!!!!!!!!!!!!!! only for debugging

    if not final_output:
        return 0.0

    patch = extract_patch_from_output(final_output)

    if not is_valid_patch(patch):
        return 0.0

    instance_id = sample.metadata.get("instance_id", "unknown")
    repo = sample.metadata.get("repo", "")
    base_commit = sample.metadata.get("base_commit", "")

    fail_to_pass = sample.metadata.get("FAIL_TO_PASS", [])
    pass_to_pass = sample.metadata.get("PASS_TO_PASS", [])

    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = []
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = json.loads(pass_to_pass)
        except json.JSONDecodeError:
            pass_to_pass = []

    test_results = await run_tests_in_docker(
        instance_id=instance_id,
        patch=patch,
        repo=repo,
        base_commit=base_commit,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        instance=sample.metadata,
    )

    if test_results.get("error"):
        return 0.0

    # Weighted combination of F2P and P2P rates
    f2p_weight = getattr(args, "swe_f2p_weight", 0.3)
    p2p_weight = getattr(args, "swe_p2p_weight", 0.7)

    f2p_rate = test_results.get("fail_to_pass_rate", 0.0)
    p2p_rate = test_results.get("pass_to_pass_rate", 0.0)

    final_reward = f2p_weight * f2p_rate + p2p_weight * p2p_rate

    final_reward *= 10.0

    # update test results with final reward for debugging
    test_results["final_reward"] = final_reward
    # write test results to trajectory path for debugging
    trajectory_path = sample.metadata.get("trajectory_path", "unknown")
    if trajectory_path != "unknown":
        try:
            with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                json.dump(test_results, f)
                logger.info(f"test_results: {json.dumps(test_results)}")
        except Exception as e:
            logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")

    return final_reward


async def compute_multi_objective_reward(args, sample: Sample) -> dict[str, float]:
    """
    Compute multi-objective reward with multiple components.

    Components:
    - resolved: Binary reward for FULL resolution
    - fail_to_pass_rate: Rate of F2P test success
    - pass_to_pass_rate: Rate of P2P test success
    - patch_valid: Reward for valid patch syntax
    - step_efficiency: Reward for using fewer steps
    - token_efficiency: Reward for using fewer tokens

    Args:
        args: Training arguments
        sample: Sample with response and metadata

    Returns:
        Dict of reward components
    """
    rewards = {}

    final_output = sample.metadata.get("final_output", "")
    n_steps = sample.metadata.get("n_steps", 0)
    exit_status = sample.metadata.get("exit_status", "Unknown")
    max_steps = getattr(args, "swe_max_steps", 100)

    # Extract patch
    patch = extract_patch_from_output(final_output)
    rewards["patch_valid"] = 1.0 if is_valid_patch(patch) else 0.0

    # Completion reward
    rewards["completed"] = 1.0 if exit_status == "Submitted" else 0.0

    # Step efficiency
    if n_steps > 0:
        rewards["step_efficiency"] = max(0.0, 1.0 - (n_steps / max_steps))
    else:
        rewards["step_efficiency"] = 0.0

    # Token efficiency
    response_length = sample.response_length
    max_response_length = getattr(args, "rollout_max_response_len", 16384)
    if response_length > 0:
        rewards["token_efficiency"] = max(0.0, 1.0 - (response_length / max_response_length))
    else:
        rewards["token_efficiency"] = 0.0

    # Test-based rewards
    if not final_output or not is_valid_patch(patch):
        rewards["resolved"] = 0.0
        rewards["fail_to_pass_rate"] = 0.0
        rewards["pass_to_pass_rate"] = 0.0
    else:
        instance_id = sample.metadata.get("instance_id", "unknown")
        repo = sample.metadata.get("repo", "")
        base_commit = sample.metadata.get("base_commit", "")

        fail_to_pass = sample.metadata.get("FAIL_TO_PASS", [])
        pass_to_pass = sample.metadata.get("PASS_TO_PASS", [])

        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass = []
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except json.JSONDecodeError:
                pass_to_pass = []

        test_results = await run_tests_in_docker(
            instance_id=instance_id,
            patch=patch,
            repo=repo,
            base_commit=base_commit,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            instance=sample.metadata,
        )

        rewards["resolved"] = 1.0 if test_results["resolved"] else 0.0
        rewards["fail_to_pass_rate"] = test_results.get("fail_to_pass_rate", 0.0)
        rewards["pass_to_pass_rate"] = test_results.get("pass_to_pass_rate", 0.0)

        # update test results with final reward for debugging
        test_results["rewards"] = rewards
        # write test results to trajectory path for debugging
        trajectory_path = sample.metadata.get("trajectory_path", "unknown")
        if trajectory_path != "unknown":
            try:
                with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                    json.dump(test_results, f)
                    logger.info(f"test_results: {json.dumps(test_results)}")
            except Exception as e:
                logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")

    return rewards


# ==============================================================================
# Process-shaped reward (mode: "process_shaped")
# ==============================================================================
#
# Composite reward: outcome stays primary, validated process terms densify
# zero-variance GRPO groups and anchor shaping to ground truth, and a
# multiplicative integrity gate caps reward-hacking / destructive behavior.
#
#   R = ( w0·resolved + w1·f2p_rate + w2·localization + w3·anchored_flip
#         + w4·p2p_rate − p5·extra_files − p6·no_progress − p7·error_rate )
#       × integrity_gate            (gate ∈ {1.0, 0.0})
#
# Weight ordering enforced by defaults: w0 ≫ {w1,w2,w3} > {w4} > {p5,p6,p7}.
# All weights are configurable via ``getattr(args, ...)`` then ``PROCESS_*`` env
# vars (mirrors the existing ``swe_f2p_weight`` pattern) so terms can be retuned
# or disabled per run and re-validated by AUC without code edits.

_PROCESS_WEIGHT_DEFAULTS = {
    "resolved": 1.0,          # w0  outcome (primary)
    "f2p_rate": 0.5,          # w1  dense outcome (A1)
    "localization": 0.4,      # w2  validated grounding (A2, AUC≈0.58)
    "anchored_flip": 0.3,     # w3  reproduce→fix loop (A3)
    "read_before_edit": 0.1,  # w2b process half of localization
    "p2p_rate": 0.2,          # w4  non-regression (A4)
    "extra_files": 0.05,      # p5  over-reach penalty (A5)
    "no_progress": 0.05,      # p6  churn-pattern penalty (A6)
    "error_rate": 0.03,       # p7  operational hygiene (A7)
}

# Components that are subtracted (penalties) rather than added.
_PROCESS_PENALTY_KEYS = {"extra_files", "no_progress", "error_rate"}


def _process_weight(args, key: str) -> float:
    """Resolve a process-shaping weight: args attr > env var > default."""
    attr = getattr(args, f"swe_process_{key}_weight", None)
    if attr is not None:
        return float(attr)
    env = os.environ.get(f"PROCESS_{key.upper()}_WEIGHT")
    if env is not None:
        try:
            return float(env)
        except ValueError:
            pass
    return _PROCESS_WEIGHT_DEFAULTS.get(key, 0.0)


def combine_process_rewards(args, rewards: dict[str, float]) -> float:
    """Combine process-shaped components with a multiplicative integrity gate.

    Additive terms minus penalty terms, then multiplied by the gate (so a gate
    of 0.0 zeroes the whole reward), then scaled ×10 for training (matching the
    other reward modes).
    """
    total = 0.0
    for key, default in _PROCESS_WEIGHT_DEFAULTS.items():
        w = _process_weight(args, key)
        val = rewards.get(key, 0.0)
        if key in _PROCESS_PENALTY_KEYS:
            total -= w * val
        else:
            total += w * val

    gate = rewards.get("integrity_gate", 1.0)
    total *= gate
    total *= 10.0
    return total


async def compute_process_shaped_reward(args, sample: Sample) -> dict[str, float]:
    """Compute the validated process-shaped reward components (section A + gate G).

    Returns a dict of components (so it can be logged / AUC-validated per term).
    Use :func:`combine_process_rewards` to reduce it to the training scalar.

    Components:
      - resolved, f2p_rate, p2p_rate          (outcome / dense outcome, A1/A4)
      - localization, read_before_edit        (A2: patch∩gold + grounding)
      - anchored_flip                         (A3: reproduce→fix loop)
      - extra_files, no_progress, error_rate  (A5/A6/A7 penalties)
      - integrity_gate                        (G: multiplicative cap)
    """
    rewards: dict[str, float] = {
        "resolved": 0.0,
        "f2p_rate": 0.0,
        "p2p_rate": 0.0,
        "localization": 0.0,
        "read_before_edit": 0.0,
        "anchored_flip": 0.0,
        "extra_files": 0.0,
        "no_progress": 0.0,
        "error_rate": 0.0,
        "integrity_gate": 1.0,
    }

    final_output = sample.metadata.get("final_output", "")
    patch = extract_patch_from_output(final_output)

    # Parse trajectory once (cheap, no Docker). Tolerate any parsing failure so
    # the reward never crashes a rollout — fall back to outcome-only shaping.
    traj = None
    try:
        traj = parse_trajectory_from_sample(sample)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[SWE_REWARD] process_shaped: trajectory parse failed: {e}")

    gold_files = list(getattr(traj, "gold_changed_files", []) or [])
    gold_test_files = list(getattr(traj, "gold_test_files", []) or [])

    # Gold FAIL_TO_PASS names for anchoring the flip detector (A3).
    fail_to_pass = sample.metadata.get("FAIL_TO_PASS", [])
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = []

    # ---- Trajectory-derived terms (work even when outcome is all-fail) -------
    patch_files = _parse_patch_files(patch)
    rewards["localization"] = _localization_score(patch_files, gold_files)
    rewards["read_before_edit"] = _read_before_edit_bonus(traj, gold_files)
    rewards["anchored_flip"] = _detect_flip(traj, fail_to_pass)
    rewards["extra_files"] = _extra_files_penalty(patch_files, gold_files)
    rewards["no_progress"] = _no_progress_penalty(traj)
    rewards["error_rate"] = _error_rate(traj)
    rewards["integrity_gate"] = _integrity_gate(patch, traj, gold_test_files)

    # ---- Outcome terms (require a valid patch + Docker eval) -----------------
    test_results: dict = {}
    if final_output and is_valid_patch(patch):
        instance_id = sample.metadata.get("instance_id", "unknown")
        repo = sample.metadata.get("repo", "")
        base_commit = sample.metadata.get("base_commit", "")

        pass_to_pass = sample.metadata.get("PASS_TO_PASS", [])
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except json.JSONDecodeError:
                pass_to_pass = []

        test_results = await run_tests_in_docker(
            instance_id=instance_id,
            patch=patch,
            repo=repo,
            base_commit=base_commit,
            fail_to_pass=fail_to_pass if isinstance(fail_to_pass, list) else [],
            pass_to_pass=pass_to_pass if isinstance(pass_to_pass, list) else [],
            instance=sample.metadata,
        )

        rewards["resolved"] = 1.0 if test_results.get("resolved") else 0.0
        rewards["f2p_rate"] = test_results.get("fail_to_pass_rate", 0.0)
        rewards["p2p_rate"] = test_results.get("pass_to_pass_rate", 0.0)

    # ---- Debug dump ----------------------------------------------------------
    test_results = dict(test_results)
    test_results["rewards"] = rewards
    test_results["process_scalar"] = combine_process_rewards(args, rewards)
    trajectory_path = sample.metadata.get("trajectory_path", "unknown")
    if trajectory_path != "unknown":
        try:
            with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                json.dump(test_results, f)
                logger.info(f"test_results: {json.dumps(test_results)}")
        except Exception as e:
            logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")

    return rewards


async def compute_shaped_reward(args, sample: Sample) -> dict[str, float]:
    """
    Compute shaped reward with intermediate progress signals.

    This provides denser reward signal by rewarding:
    - File exploration (ls, find, cat)
    - File editing (vim, sed, patch)
    - Test execution (pytest, python -m test)
    - Valid patch generation
    - SWE-bench pass rates

    Args:
        args: Training arguments
        sample: Sample with response and metadata

    Returns:
        Dict of reward components including shaping signals
    """
    # Start with multi-objective rewards
    rewards = await compute_multi_objective_reward(args, sample)

    # Add shaping rewards based on trajectory
    trajectory = sample.metadata.get("trajectory", {})
    messages = trajectory.get("messages", [])

    # Track intermediate progress
    explored_files = False
    edited_files = False
    ran_tests = False

    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")

            # Check for file exploration
            if any(cmd in content for cmd in ["ls", "find", "cat", "grep"]):
                explored_files = True

            # Check for file editing
            if any(cmd in content for cmd in ["vim", "sed", "patch", "echo >", ">"]):
                edited_files = True

            # Check for test execution
            if any(cmd in content for cmd in ["pytest", "python -m test", "python -m unittest"]):
                ran_tests = True

    # Add intermediate rewards (small bonuses)
    rewards["explored_files"] = 0.1 if explored_files else 0.0
    rewards["edited_files"] = 0.2 if edited_files else 0.0
    rewards["ran_tests"] = 0.1 if ran_tests else 0.0

    return rewards


def combine_rewards(rewards: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """
    Combine multiple reward components into a single scalar.

    Default weights prioritize test success, with small bonuses for other factors.

    Args:
        rewards: Dict of reward components
        weights: Optional custom weights for each component

    Returns:
        Combined scalar reward
    """
    default_weights = {
        "resolved": 1.0,              # Primary: full resolution
        "fail_to_pass_rate": 0.5,     # F2P test pass rate
        "pass_to_pass_rate": 0.3,     # P2P test pass rate
        "patch_valid": 0.1,           # Valid patch bonus
        "completed": 0.1,             # Completion bonus
        "step_efficiency": 0.05,      # Efficiency bonus
        "token_efficiency": 0.05,     # Token efficiency bonus
        "explored_files": 0.02,       # Shaping: exploration
        "edited_files": 0.03,         # Shaping: editing
        "ran_tests": 0.02,            # Shaping: testing
    }

    if weights is None:
        weights = default_weights

    total = 0.0
    for key, value in rewards.items():
        weight = weights.get(key, 0.0)
        total += weight * value

    total *= 10.0  # Scale up for training

    return total


async def reward_func(
    args,
    sample: Sample | list[Sample],
    **kwargs,
) -> float | dict[str, float] | list[float | dict[str, float]]:
    """
    Main reward function for SWE-bench tasks.

    Supports multiple reward modes:
    - "simple": Binary reward (1.0 for FULL resolution, 0.0 otherwise)
    - "simple_truncated": Same as "simple", but trajectories truncated by
      step/token/time limits receive -2.0 instead of running the evaluator
    - "simple_truncated_zero": Same as "simple_truncated", but unresolved
      (yet submitted) trajectories get 0.0 instead of -1.0
    - "simple_plus_llm_as_judges": "simple_truncated_zero" binary R, plus an
      LLM-as-judge process reward J in [0,1] added to ALL paths:
      R' = R_binary + PRM_ALPHA * J. Configured via PRM_* env vars (see
      ``swe_prm.py``). When PRM_ALPHA == 0 it short-circuits to
      "simple_truncated_zero" (no API call). PRM failure falls back to R_binary.
    - "pass_rate": Weighted pass rate (F2P and P2P rates)
    - "multi_objective": Multiple reward components combined into scalar
    - "shaped": Includes intermediate progress rewards combined into scalar
    - "process_shaped": Outcome (resolved/F2P) primary, plus *validated*
      deterministic process shaping (localization A2, anchored reproduce->fix
      flip A3, P2P non-regression A4) minus small over-reach/churn/error
      penalties (A5/A6/A7), all multiplied by an integrity gate G that caps
      reward-hacking / destructive trajectories. See
      ``compute_process_shaped_reward`` / ``combine_process_rewards``.
    - "patch_classification": For generated-test tasks, run the generated test
      on the corresponding labeled candidate patches and reward binary
      classification accuracy. This mode returns a reward dict with
      base_reward, gold_reward, patch_cls_reward, and raw_reward. The first
      three fields are diagnostic state, while raw_reward is the scalar used
      for group normalization and training.
    - "base_gold": For generated-test tasks, run only base/gold validation
      and skip candidate-patch replay. This legacy mode retains its existing
      base_reward, gold_reward, and patch_reward schema.

    Eval-mode override:
      When called with ``evaluation=True`` (forwarded by slime's eval rollout
      via ``async_rm``), the active mode is replaced with the value of the
      ``SWE_EVAL_REWARD_MODE`` env var (default ``simple_truncated_zero``).
      This lets training use ``simple_plus_llm_as_judges`` while eval tracks
      pure solve-rate without paying for the PRM API.

    Args:
        args: Training arguments (should have swe_reward_mode attribute)
        sample: Sample or batch of samples to compute rewards for
        **kwargs: Additional keyword arguments

    Returns:
        Float reward value for normal SWE modes, or a reward dict for
        patch_classification. Configure --reward-key when training from a dict.
    """
    if isinstance(sample, list):
        # slime.rollout.rm_hub calls custom RM paths in batch mode.
        return await asyncio.gather(*(reward_func(args, item, **kwargs) for item in sample))

    instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
    logger.info(f"[SWE_REWARD] Starting reward_func for instance_id={instance_id}")
    
    # Get reward mode (priority: args > environment variable > default)
    reward_mode = getattr(args, "swe_reward_mode", None)
    if reward_mode is None:
        reward_mode = os.environ.get("SWE_REWARD_MODE", "simple")

    # During eval rollouts we want to track pure solve-rate, not the
    # PRM-augmented (mixed) reward used for training.  slime forwards
    # ``evaluation=True`` from sglang_rollout.eval_rollout_single_dataset
    # -> async_rm -> here via **kwargs.  When set, we override the active
    # reward mode with SWE_EVAL_REWARD_MODE (default "simple_truncated_zero"),
    # which is binary and never calls the PRM API.  Set the env var to the
    # same value as SWE_REWARD_MODE if you actually want PRM during eval.
    is_eval = bool(kwargs.get("evaluation", False))
    if is_eval:
        eval_mode = os.environ.get("SWE_EVAL_REWARD_MODE", "simple_truncated_zero")
        if eval_mode != reward_mode:
            logger.info(
                f"[SWE_REWARD] {instance_id}: eval rollout -> overriding "
                f"reward_mode '{reward_mode}' -> '{eval_mode}' (no PRM API call)"
            )
        reward_mode = eval_mode

    # Wrap the entire reward computation with a timeout to prevent indefinite hangs
    # (e.g. Azure Docker container creation or eval script execution stuck)
    try:
        if reward_mode == "simple":
            result = await asyncio.wait_for(compute_simple_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "simple_truncated":
            result = await asyncio.wait_for(compute_simple_truncated_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "simple_truncated_zero":
            result = await asyncio.wait_for(compute_simple_truncated_zero_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "simple_plus_llm_as_judges":
            result = await asyncio.wait_for(compute_simple_plus_llm_as_judges_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "pass_rate":
            result = await asyncio.wait_for(compute_pass_rate_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "multi_objective":
            rewards = await asyncio.wait_for(compute_multi_objective_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
            result = combine_rewards(rewards)
        elif reward_mode == "shaped":
            rewards = await asyncio.wait_for(compute_shaped_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
            result = combine_rewards(rewards)
        elif reward_mode == "process_shaped":
            rewards = await asyncio.wait_for(compute_process_shaped_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
            result = combine_process_rewards(args, rewards)
        elif reward_mode == "patch_classification":
            # This mode owns a synchronous sandbox client in a worker thread.
            # Cancelling asyncio.to_thread() only detaches the thread, so its
            # deadline is enforced cooperatively inside the worker instead.
            result = await compute_patch_classification_reward(args, sample)
        elif reward_mode in {"base_gold", "base_fail_gold"}:
            result = await compute_base_gold_reward(args, sample)
        else:
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        logger.info(f"[SWE_REWARD] Completed reward_func for instance_id={instance_id}, reward={result}")
        assert isinstance(result, (float, int, dict)), f"Reward function must return a scalar or reward dict, got {type(result)}"
        if isinstance(result, dict):
            for key, value in result.items():
                assert isinstance(value, (float, int)), f"Reward dict value for {key!r} must be numeric, got {type(value)}"
        return result
    except asyncio.TimeoutError:
        logger.error(
            f"[SWE_REWARD] reward_func TIMED OUT after {SWE_TIMEOUT_REWARD_TOTAL}s for instance_id={instance_id}. "
        )
        if reward_mode == "patch_classification":
            details = {
                "mode": reward_mode,
                "instance_id": instance_id,
                "error": f"reward_func timed out after {SWE_TIMEOUT_REWARD_TOTAL}s",
            }
            rewards = _patch_classification_reward_dict(0.0, 0.0, 0.0)
            details["rewards"] = rewards
            details["final_reward"] = rewards
            _write_patch_classification_reward_details(sample, details)
            return rewards
        if reward_mode in {"base_gold", "base_fail_gold"}:
            return _base_gold_error_reward(
                sample,
                {
                    "mode": reward_mode,
                    "instance_id": instance_id,
                    "error": f"reward_func timed out after {SWE_TIMEOUT_REWARD_TOTAL}s",
                },
            )
        return -1.0
    except Exception as e:
        logger.error(f"[SWE_REWARD] reward_func ERROR for instance_id={instance_id}: {e}\n{traceback.format_exc()}")
        if reward_mode == "patch_classification":
            return _patch_classification_error_reward(
                sample,
                {
                    "mode": reward_mode,
                    "instance_id": instance_id,
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "traceback": traceback.format_exc()[-4000:],
                },
            )
        if reward_mode in {"base_gold", "base_fail_gold"}:
            return _base_gold_error_reward(
                sample,
                {
                    "mode": reward_mode,
                    "instance_id": instance_id,
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "traceback": traceback.format_exc()[-4000:],
                },
            )
        return -1.0


# Synchronous wrapper for backward compatibility
def compute_reward_sync(args, sample: Sample, **kwargs) -> float | dict[str, Any]:
    """
    Synchronous wrapper for reward computation.

    Args:
        args: Training arguments
        sample: Sample to compute reward for
        **kwargs: Additional keyword arguments

    Returns:
        Reward value(s)
    """
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If already in async context, create task
        return asyncio.create_task(reward_func(args, sample, **kwargs))
    else:
        # Run in event loop
        return loop.run_until_complete(reward_func(args, sample, **kwargs))
