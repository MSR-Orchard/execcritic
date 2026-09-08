"""Trainable SWE solve -> oracle-F2P -> repair rollout.

One returned :class:`~slime.utils.types.Sample` contains the complete agent
trajectory:

1. solve the issue normally (round 0),
2. send each submitted patch, including the configured turn-limit fallback, to
   one prepared official verifier that resets to baseline before every grade,
3. append only the failing output (never the hidden test source), and
4. continue the same conversation until a repaired patch is submitted.

The model workspace is never reset for verification. All repair gates and the
final F2P + P2P grade reuse the same official verifier sandbox; only its Git
worktree is reset between complete submissions.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import random
import re
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .swe_reward import is_valid_patch
from .swe_wrapper_v2 import (
    TOOLS,
    EnvironmentUnavailable,
    FormatError,
    LimitsExceeded,
    SWEAgentV2,
    TimeExceeded,
    create_environment,
    load_config,
    stop_environment,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_ALL_TOKENS = 65536
RETURN_LOGPROB = True
_RUN_TIMESTAMP: str | None = None

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")

INITIAL_REPAIR_PROMPT = (
    "Your previous patch attempt has been applied to /testbed, but a hidden regression test still "
    "FAILS. You cannot see the test source, only its failure output below. Continue editing the "
    "SOURCE code so the described behavior is correct and the failing test passes. Do not edit or "
    "create test files. Keep the current source changes, make a minimal repair, then submit the full "
    "source diff again.\n\n<failing_test_output>\n{feedback}\n</failing_test_output>"
)

FOLLOWUP_REPAIR_PROMPT = (
    "The hidden regression test still FAILS after your last submission. Keep the existing source "
    "changes, use the updated failure output below, make the next minimal source-only edit, and "
    "submit again. Do not edit or create test files.\n\n<failing_test_output>\n{feedback}\n"
    "</failing_test_output>"
)

SUBMIT_CURRENT_PATCH_PROMPT = (
    "Round 0 ended without a valid submission, so the isolated verifier has not tested these "
    "changes. Do not make unrelated edits. Create the full source-only git diff and submit it now "
    "with COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


def _load_official_f2p_gate():
    # Keep the harness dependency lazy so importing the rollout for local unit
    # tests does not require a site-installed minisweagent package.
    from .swe_reward import _ensure_swe_harness_on_path

    _ensure_swe_harness_on_path()
    from minisweagent.run.benchmarks import official_f2p_gate

    return official_f2p_gate


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(item) for item in value if str(item).strip()]


def _problem_statement(sample: Sample) -> str:
    if isinstance(sample.metadata, dict) and sample.metadata.get("problem_statement"):
        return str(sample.metadata["problem_statement"])
    if isinstance(sample.prompt, str):
        return sample.prompt
    for message in reversed(sample.prompt or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return str(sample.prompt)


def _instance_from_sample(sample: Sample) -> dict[str, Any]:
    metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
    nested = metadata.pop("metadata", None)
    if isinstance(nested, dict):
        metadata = {**nested, **metadata}
    metadata.setdefault("instance_id", f"task_{sample.index}")
    instance_id = str(metadata["instance_id"])
    if not _SAFE_INSTANCE_ID.fullmatch(instance_id):
        raise ValueError(f"unsafe instance_id for trajectory path: {instance_id!r}")
    metadata["instance_id"] = instance_id
    metadata.setdefault("problem_statement", _problem_statement(sample))
    if sample.label is not None:
        metadata.setdefault("patch", sample.label)
    return metadata


def _f2p_repair_rollout_metrics(samples: list[Sample]) -> dict[str, float | int]:
    repair_records = [
        sample.metadata["f2p_repair"]
        for sample in samples
        if isinstance(sample.metadata, dict)
        and isinstance(sample.metadata.get("f2p_repair"), dict)
    ]
    entered_repair = [
        record
        for record in repair_records
        if int(record.get("repair_rounds_started", record.get("repair_rounds_used", 0)) or 0) > 0
    ]
    repair_rounds = [
        int(record.get("repair_rounds_started", record.get("repair_rounds_used", 0)) or 0)
        for record in entered_repair
    ]
    repair_turns = [int(record.get("repair_turns") or 0) for record in entered_repair]
    repair_successes = sum(bool(record.get("f2p_passed")) for record in entered_repair)

    return {
        "f2p_repair/entered_samples": len(entered_repair),
        "f2p_repair/repair_rounds_mean": (
            sum(repair_rounds) / len(entered_repair) if entered_repair else 0.0
        ),
        "f2p_repair/success_rate": (
            repair_successes / len(entered_repair) if entered_repair else 0.0
        ),
        # Keep the established metric key, but report turns per sample that
        # actually entered repair rather than a rollout-batch-size-dependent sum.
        "f2p_repair/repair_turns_total": (
            sum(repair_turns) / len(entered_repair) if entered_repair else 0.0
        ),
    }


def append_rollout_metrics(_rollout_id, _args, samples, rollout_extra_metrics, _rollout_time) -> bool:
    """Add F2P-repair metrics, then let slime's default rollout logger continue."""
    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(_f2p_repair_rollout_metrics(samples))
    return False


def select_f2p_nodes(args, sample: Sample, instance: dict[str, Any]) -> list[str]:
    """Pick the same oracle node for all siblings in one GRPO prompt group."""
    nodes = [
        node
        for node in _as_list(instance.get("FAIL_TO_PASS"))
        if node not in {"[", "]", ","} and node.count("[") == node.count("]")
    ]
    if not nodes:
        return ["<f2p_script>"] if instance.get("f2p_script") else []
    group_key = sample.group_index if sample.group_index is not None else instance["instance_id"]
    rng = random.Random(f"{getattr(args, 'rollout_seed', 42)}:{instance['instance_id']}:{group_key}")
    rng.shuffle(nodes)
    return nodes[: max(1, _env_int("SWE_F2P_REPAIR_N_TESTS", 1))]


async def _execute(env, command: str, timeout: int | None = None) -> dict[str, Any]:
    old_timeout = None
    if timeout and hasattr(env, "config") and hasattr(env.config, "exe_timeout"):
        old_timeout = env.config.exe_timeout
        env.config.exe_timeout = max(int(old_timeout), int(timeout))
    try:
        if inspect.iscoroutinefunction(env.execute):
            awaitable = env.execute(command)
        else:
            awaitable = asyncio.to_thread(env.execute, command)
        result = await asyncio.wait_for(awaitable, timeout=timeout + 45 if timeout else None)
    finally:
        if old_timeout is not None:
            env.config.exe_timeout = old_timeout

    output = str(result.get("output") or "")
    if result.get("returncode") != 0 and (
        "container exec failed" in output and ("404" in output or "Not Found" in output)
    ):
        raise EnvironmentUnavailable(output)
    return result


def _patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"(?m)^(?:---|\+\+\+) [ab]/(.+)$", patch or ""):
        path = match.group(1).strip()
        if path and path != "/dev/null" and path not in paths:
            paths.append(path)
    return paths


def _looks_like_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/")
    parts = normalized.lower().split("/")
    name = parts[-1] if parts else ""
    return bool(
        any(part in {"test", "tests", "testing"} for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith(("_test.py", "_tests.py", ".snap", ".golden"))
        or name == "conftest.py"
    )


def forbidden_test_paths(model_patch: str, hidden_test_patch: str) -> list[str]:
    hidden_paths = set(_patch_paths(hidden_test_patch))
    return [
        path
        for path in _patch_paths(model_patch)
        if path in hidden_paths or _looks_like_test_path(path)
    ]


async def _current_diff(env, workdir: str) -> str:
    # HEAD includes both staged and unstaged model edits. Plain `git diff` would
    # silently drop a patch after the model runs `git add`.
    result = await _execute(
        env,
        f"cd {shlex.quote(workdir)} && git -c core.fileMode=false diff HEAD --binary",
    )
    return str(result.get("output") or "") if result.get("returncode") == 0 else ""


class OfficialVerifySession:
    """One prepared official verifier reused for every grade in the rollout."""

    def __init__(
        self,
        instance: dict[str, Any],
        *,
        test_timeout: int,
        feedback_chars: int,
    ) -> None:
        self.instance = instance
        self.test_timeout = test_timeout
        self.feedback_chars = feedback_chars
        self._gate = None

    async def initialize(self, setup_timeout: int) -> None:
        official_f2p_gate = _load_official_f2p_gate()
        self._gate = await asyncio.to_thread(
            official_f2p_gate.PersistentOfficialGate,
            self.instance,
            timeout=max(self.test_timeout, setup_timeout),
        )

    async def _evaluate(
        self,
        patch: str,
        *,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        timeout: int,
    ) -> dict[str, Any]:
        if self._gate is None:
            raise RuntimeError("persistent official verifier was not initialized")
        return await asyncio.to_thread(
            self._gate.evaluate,
            patch,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            timeout=timeout,
        )

    async def check_f2p(self, submitted_patch: str, nodes: list[str]) -> dict[str, Any]:
        submitted_patch = str(submitted_patch or "")
        submitted_sha = (
            hashlib.sha256(submitted_patch.encode()).hexdigest() if submitted_patch else ""
        )
        hidden_patch = str(
            self.instance.get("f2p_patch") or self.instance.get("test_patch") or ""
        )
        forbidden = forbidden_test_paths(submitted_patch, hidden_patch)
        if forbidden:
            feedback = (
                "The submission modifies forbidden test files. Revert every test-file change "
                "and repair source only.\n"
                + "\n".join(f"- {path}" for path in forbidden)
            )
            return {
                "passed": False,
                "feedback": feedback,
                "returncode": 1,
                "forbidden_test_paths": forbidden,
                "command": "official_full_eval_script",
                "patch": submitted_patch,
                "submitted_patch_sha256": submitted_sha,
                "gate_patch_sha256": "",
                "worktree_patch_sha256": "",
            }
        if not submitted_patch.strip():
            return {
                "passed": False,
                "feedback": "SUBMITTED_PATCH_EMPTY",
                "returncode": 1,
                "forbidden_test_paths": [],
                "command": "official_full_eval_script",
                "patch": submitted_patch,
                "submitted_patch_apply_error": "SUBMITTED_PATCH_EMPTY",
                "submitted_patch_sha256": submitted_sha,
                "gate_patch_sha256": "",
                "worktree_patch_sha256": "",
            }

        official = await self._evaluate(
            submitted_patch,
            fail_to_pass=nodes,
            pass_to_pass=[],
            timeout=self.test_timeout,
        )
        output = str(official.get("output") or "")
        returncode = int(
            official.get("exit_code") if official.get("exit_code") is not None else -1
        )
        passed = bool(official.get("resolved"))
        feedback = _focused_feedback(
            output,
            nodes,
            returncode,
            self.feedback_chars,
            passed=passed,
        )
        apply_error = ""
        if not official.get("patch_applied"):
            apply_error = "SUBMITTED_PATCH_APPLY_FAILED"
            feedback = apply_error + "\n" + feedback
            passed = False
        official_error = str(official.get("error") or "")
        if official_error:
            feedback = f"OFFICIAL_EVAL_ERROR: {official_error}\n" + feedback
            passed = False

        return {
            "passed": passed,
            "feedback": feedback,
            "returncode": returncode,
            "forbidden_test_paths": [],
            "command": "official_full_eval_script",
            "output_tail": _ANSI_ESCAPE.sub("", output)[-2000:],
            "official_gate": {
                "resolved": bool(official.get("resolved")),
                "fail_to_pass_passed": official.get("fail_to_pass_passed", []),
                "fail_to_pass_failed": official.get("fail_to_pass_failed", []),
                "pass_to_pass_passed": official.get("pass_to_pass_passed", 0),
                "pass_to_pass_failed": official.get("pass_to_pass_failed", []),
                "exit_code": official.get("exit_code"),
                "parsed_tests_count": official.get("parsed_tests_count", 0),
                "patch_applied": bool(official.get("patch_applied")),
                "error": official_error,
                "official_verifier": official.get("official_verifier", ""),
            },
            "patch": submitted_patch,
            "submitted_patch_apply_error": apply_error,
            "submitted_patch_sha256": submitted_sha,
            "gate_patch_sha256": submitted_sha if official.get("patch_applied") else "",
            "worktree_patch_sha256": submitted_sha if official.get("patch_applied") else "",
            "persistent_verifier": True,
        }

    async def run_final_official(self, patch: str) -> dict[str, Any]:
        fail_to_pass = _as_list(self.instance.get("FAIL_TO_PASS"))
        pass_to_pass = _as_list(self.instance.get("PASS_TO_PASS"))
        started = time.monotonic()
        official_f2p_gate = _load_official_f2p_gate()
        official = await self._evaluate(
            patch,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            timeout=_env_int("SWE_TIMEOUT_REWARD_TOTAL", 900),
        )
        result = official_f2p_gate.normalize_result(
            official,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
        result["reward_eval_seconds"] = time.monotonic() - started
        result["persistent_verifier"] = True
        return result

    async def close(self) -> None:
        if self._gate is None:
            return
        gate, self._gate = self._gate, None
        await asyncio.to_thread(gate.close)


def _focused_feedback(
    output: str,
    nodes: list[str],
    returncode: int,
    max_chars: int,
    *,
    passed: bool | None = None,
) -> str:
    clean = _ANSI_ESCAPE.sub("", output or "")
    header = [f"test_return_code: {returncode}", "selected_f2p:"]
    gate_passed = returncode == 0 if passed is None else passed
    header.extend(f"- {node}: {'PASSED' if gate_passed else 'FAILED'}" for node in nodes)

    lines = clean.splitlines()
    failure_start = next(
        (i for i, line in enumerate(lines) if re.match(r"=+ (FAILURES|ERRORS) =+", line)),
        None,
    )
    failure = ""
    if failure_start is not None:
        failure = "\n".join(lines[failure_start : failure_start + 100])[-3000:]

    selected: set[int] = set()
    for node in nodes:
        terms = [node]
        if "::" in node:
            terms.append(node.rsplit("::", 1)[-1].split("[")[0])
        for index, line in enumerate(lines):
            if any(term and term in line for term in terms):
                selected.update(range(max(0, index - 10), min(len(lines), index + 25)))
    focused = "\n".join(lines[index] for index in sorted(selected)) if selected else clean[-3500:]
    feedback = "\n".join(header)
    if failure:
        feedback += "\n\nfailure_traceback:\n" + failure
    feedback += "\n\nfocused_test_output:\n" + focused
    return feedback[-max_chars:]


def get_token_delta(
    tokenizer,
    messages: list[dict],
    previous_len: int,
    *,
    tools: list[dict] | None = None,
) -> list[int]:
    """Tokenize a whole non-model suffix without inserting an interim assistant prompt.

    Repair adds ``tool -> user feedback`` before the next model turn. Rendering the
    tool message first with ``add_generation_prompt=True`` would append an assistant
    prefix that must later be retracted when the user feedback is added. Render the
    complete suffix once so the append-only training stream exactly matches the next
    SGLang prompt.
    """
    prefix = messages[:previous_len]
    if not prefix or prefix[-1].get("role") != "assistant":
        raise RuntimeError("non-model token suffix must start immediately after an assistant response")
    previous = tokenizer.apply_chat_template(
        prefix, tokenize=False, add_generation_prompt=False, tools=tools
    )
    current = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=messages[-1].get("role") != "assistant",
        tools=tools,
    )
    if not current.startswith(previous):
        raise RuntimeError("chat template rewrote the generated assistant prefix; cannot align repair tokens")
    return list(tokenizer.encode(current[len(previous) :]))


def _trajectory_root() -> Path:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = os.environ.get("WANDB_RANDOM_SUFFIX", "")
    dirname = _RUN_TIMESTAMP + (f"_{suffix}" if suffix else "")
    return Path(os.environ.get("SWE_TRAJECTORY_DIR", "trajectories")) / dirname


def _reset_sample(sample: Sample, prompt_ids: list[int], prompt_text: str) -> None:
    sample.prompt = prompt_text
    sample.tokens = list(prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.rollout_routed_experts = None
    sample.status = Sample.Status.PENDING


def _normalize_response_text(response_text: str, tokenizer) -> str:
    """Match the existing SWE rollout's multi-turn chat normalization."""
    eos_token = tokenizer.eos_token
    if eos_token and response_text.endswith(eos_token):
        response_text = response_text[: -len(eos_token)]
    if "</think>" in response_text and not response_text.startswith("<think>"):
        response_text = ("<think>\n" if "\n</think>" in response_text else "<think>") + response_text
    return response_text


async def _append_messages(
    args,
    sample: Sample,
    tokenizer,
    agent: SWEAgentV2,
    previous_len: int,
    *,
    max_all_tokens: int,
) -> bool:
    """Append masked context and return whether the complete suffix fit."""
    token_ids = get_token_delta(tokenizer, agent.messages, previous_len, tools=TOOLS)
    remaining = max(0, max_all_tokens - len(sample.tokens))
    if len(token_ids) > remaining:
        return False
    if token_ids:
        sample.append_response_tokens(args, tokens=token_ids, trainable=False)
    return True


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Slime custom-generate entrypoint with whole-environment retry."""
    # Custom generation normally replaces this with the full rendered agent
    # prompt in ``_generate_once``. Initialize it here as well so preflight
    # failures remain valid zero-loss training rows instead of zero-token rows.
    if not sample.tokens:
        state = GenerateState(args)
        tokenizer = state.tokenizer
        if isinstance(sample.prompt, str):
            prompt_text = sample.prompt
        else:
            prompt_text = tokenizer.apply_chat_template(
                sample.prompt,
                tokenize=False,
                add_generation_prompt=True,
                tools=sample.metadata.get("tools"),
            )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            prompt_ids = [getattr(tokenizer, "eos_token_id", None) or 0]
        _reset_sample(sample, list(prompt_ids), prompt_text)

    instance = _instance_from_sample(sample)
    if not any(instance.get(key) for key in ("test_patch", "f2p_patch", "f2p_script")):
        sample.status = Sample.Status.ABORTED
        sample.reward = None
        sample.metadata["f2p_repair_error"] = (
            "MissingF2POracle: instance has neither test_patch/f2p_patch nor f2p_script"
        )
        logger.error(
            "[F2P_REPAIR] skipping sample=%s instance=%s because its oracle F2P content is missing",
            sample.index,
            instance.get("instance_id", "unknown"),
        )
        return sample

    max_retries = _env_int("SWE_MAX_ENV_RETRIES", 2)
    retry_wait = _env_int("SWE_ENV_RETRY_WAIT", 10)
    resource_level = 1
    for attempt in range(1, max_retries + 1):
        try:
            return await _generate_once(args, sample, sampling_params, resource_level, evaluation=evaluation)
        except Exception as exc:  # noqa: BLE001 - a new sandbox is the recovery boundary
            logger.exception(
                "[F2P_REPAIR] rollout attempt %s/%s failed sample=%s: %s",
                attempt,
                max_retries,
                sample.index,
                exc,
            )
            resource_level += 1
            if attempt >= max_retries:
                sample.status = Sample.Status.ABORTED
                sample.reward = None
                sample.metadata["f2p_repair_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                return sample
            await asyncio.sleep(retry_wait)


async def _create_rollout_environment(
    env_config: dict[str, Any],
    instance: dict[str, Any],
    *,
    instance_id: str,
    startup_command: str | None,
    create_timeout: int,
    create_retries: int,
    create_retry_wait: int,
):
    for create_attempt in range(1, create_retries + 1):
        try:
            return await asyncio.wait_for(
                create_environment(
                    env_config,
                    instance_id=instance_id,
                    instance=instance,
                    startup_command=startup_command,
                ),
                timeout=create_timeout,
            )
        except Exception:
            if create_attempt >= create_retries:
                raise
            await asyncio.sleep(create_retry_wait)
    raise RuntimeError("environment creation returned no environment")


async def _stop_rollout_environments(environments: list[Any], timeout: int, instance_id: str) -> None:
    async def stop_one(env) -> None:
        try:
            await asyncio.wait_for(stop_environment(env), timeout=timeout)
        except BaseException as exc:  # cleanup failure must not discard a completed rollout
            logger.error("[F2P_REPAIR] failed to stop environment %s: %s", instance_id, exc)

    await asyncio.gather(*(stop_one(env) for env in environments if env is not None))


async def _generate_once(
    args,
    sample: Sample,
    sampling_params,
    resource_level: int,
    *,
    evaluation: bool,
) -> Sample:
    started = time.time()
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    instance = _instance_from_sample(sample)
    instance_id = str(instance["instance_id"])
    nodes = select_f2p_nodes(args, sample, instance)
    if not nodes:
        raise RuntimeError("instance has no FAIL_TO_PASS node or f2p_script")

    config_path = getattr(args, "swe_config_path", None) or os.environ["SWE_CONFIG_PATH"]
    config = load_config(config_path)
    agent_config = copy.deepcopy(config.get("agent", {}))
    env_config = copy.deepcopy(config.get("environment", {}))
    env_config["cpu"] = str(resource_level * 2)
    env_config["memory"] = f"{resource_level * 8}Gi"

    round0_steps = _env_int("SWE_F2P_REPAIR_ROUND0_STEPS", 200)
    repair_steps = _env_int("SWE_F2P_REPAIR_STEPS_PER_ROUND", 30)
    max_repair_rounds = _env_int("SWE_F2P_REPAIR_MAX_ROUNDS", 5)
    test_timeout = _env_int("SWE_F2P_REPAIR_TEST_TIMEOUT", 240)
    setup_timeout = _env_int("SWE_F2P_REPAIR_SETUP_TIMEOUT", 600)
    feedback_chars = _env_int("SWE_F2P_REPAIR_FEEDBACK_MAX_CHARS", 6000)
    llm_timeout = _env_int("SWE_TIMEOUT_LLM_INFERENCE", 60)
    observation_timeout = _env_int("SWE_TIMEOUT_GET_OBSERVATION", 90)
    create_timeout = _env_int("SWE_TIMEOUT_CREATE_ENV", 480)
    stop_timeout = _env_int("SWE_TIMEOUT_STOP_ENV", 60)
    create_retries = _env_int("SWE_MAX_CREATE_ENV_RETRIES", 1)
    create_retry_wait = _env_int("SWE_CREATE_ENV_RETRY_WAIT", 10)
    jitter_max = _env_int("SWE_ENV_CREATE_JITTER_MAX", 5)
    force_submit_on_turn_limit = _env_bool("SWE_F2P_REPAIR_FORCE_SUBMIT_ON_TURN_LIMIT", True)
    if min(round0_steps, repair_steps, max_repair_rounds, test_timeout, setup_timeout, feedback_chars) <= 0:
        raise ValueError(
            "F2P repair limits must be positive: "
            f"round0={round0_steps} repair_steps={repair_steps} rounds={max_repair_rounds} "
            f"test_timeout={test_timeout} setup_timeout={setup_timeout} feedback_chars={feedback_chars}"
        )

    multiplier = max(1, int(sampling_params.get("max_new_tokens", 4096)) // 4096)
    max_all_tokens = _env_int(
        "SWE_F2P_REPAIR_MAX_ALL_TOKENS",
        int(agent_config.get("max_all_tokens", DEFAULT_MAX_ALL_TOKENS)),
    ) * multiplier
    llm_timeout *= multiplier
    max_llm_attempts = 2 * multiplier
    agent_config["step_limit"] = 0  # phase-aware limits below replace the single global limit
    if agent_config.get("time_limit", 0) > 0:
        agent_config["time_limit"] = float(agent_config["time_limit"]) * multiplier

    workspace_env = None
    verify_session = OfficialVerifySession(
        instance,
        test_timeout=test_timeout,
        feedback_chars=feedback_chars,
    )
    if jitter_max > 0:
        await asyncio.sleep(random.uniform(0, jitter_max))
    try:
        startup_command = config.get("run", {}).get("env_startup_command")
        workdir = str(instance.get("workdir") or "/testbed")
        workspace_result, verifier_result = await asyncio.gather(
            _create_rollout_environment(
                env_config,
                instance,
                instance_id=f"{instance_id}:{sample.index}:workspace",
                startup_command=startup_command,
                create_timeout=create_timeout,
                create_retries=create_retries,
                create_retry_wait=create_retry_wait,
            ),
            verify_session.initialize(setup_timeout),
            return_exceptions=True,
        )
        if not isinstance(workspace_result, BaseException):
            workspace_env = workspace_result
        for result in (workspace_result, verifier_result):
            if isinstance(result, BaseException):
                raise result
        agent = SWEAgentV2(env=workspace_env, **agent_config)
        agent.setup(task=instance["problem_statement"], workdir=workdir)
        initial_prompt = tokenizer.apply_chat_template(
            agent.messages, tokenize=False, add_generation_prompt=True, tools=TOOLS
        )
        prompt_ids = tokenizer(initial_prompt, add_special_tokens=False)["input_ids"]
        _reset_sample(sample, list(prompt_ids), initial_prompt)
    except BaseException:
        try:
            await verify_session.close()
        except BaseException as exc:
            logger.error("[F2P_REPAIR] failed to close verifier %s: %s", instance_id, exc)
        await _stop_rollout_environments(
            [workspace_env],
            stop_timeout,
            instance_id,
        )
        raise

    env = workspace_env

    phase = "round0"
    phase_calls = 0
    round0_calls = 0
    repair_calls = 0
    repair_round = 0
    repair_records: list[dict[str, Any]] = []
    f2p_checks: list[dict[str, Any]] = []
    submitted_patch = ""
    last_submitted_patch = ""
    exit_status = "Unknown"
    stop_reason = ""
    had_submission = False
    llm_error = False
    pending_previous_len: int | None = None

    try:
        while True:
            phase_limit = round0_steps if phase == "round0" else repair_steps
            if phase_calls >= phase_limit:
                if phase == "repair":
                    repair_records.append(
                        {
                            "round": repair_round,
                            "turns": phase_calls,
                            "passed": False,
                            "submitted": False,
                            "forced_submission": False,
                            "patch_sha256": "",
                            "patch_chars": 0,
                            "forbidden_test_paths": [],
                        }
                    )
                    exit_status = "LimitsExceeded"
                    stop_reason = "repair_turn_limit_without_submit"
                    break
                previous_len = len(agent.messages)
                agent.add_messages(
                    {
                        "role": "tool",
                        "content": SUBMIT_CURRENT_PATCH_PROMPT,
                        "tool_call_id": "slime_f2p_repair",
                    }
                )
                if pending_previous_len is None:
                    pending_previous_len = previous_len
                phase = "repair"
                repair_round = 1
                phase_calls = 0
                continue

            if pending_previous_len is not None:
                suffix_complete = await _append_messages(
                    args,
                    sample,
                    tokenizer,
                    agent,
                    pending_previous_len,
                    max_all_tokens=max_all_tokens,
                )
                pending_previous_len = None
                if not suffix_complete:
                    exit_status = "LimitsExceeded"
                    stop_reason = "max_context_tokens_in_oracle_feedback"
                    break
            if len(sample.tokens) >= max_all_tokens:
                exit_status = "LimitsExceeded"
                stop_reason = "max_context_tokens"
                break

            try:
                agent.check_limits()  # cost + wall-clock limits remain active
            except TimeExceeded:
                exit_status = "TimeExceeded"
                stop_reason = "wall_clock_limit"
                break
            except LimitsExceeded:
                exit_status = "LimitsExceeded"
                stop_reason = "agent_limit"
                break

            current_prompt = tokenizer.apply_chat_template(
                agent.messages, tokenize=False, add_generation_prompt=True, tools=TOOLS
            )
            payload = {
                "text": current_prompt,
                "sampling_params": sampling_params,
                "return_logprob": RETURN_LOGPROB,
            }
            output = None
            for llm_attempt in range(1, max_llm_attempts + 1):
                try:
                    output = await asyncio.wait_for(post(url, payload), timeout=llm_timeout)
                    break
                except Exception:
                    if llm_attempt >= max_llm_attempts:
                        raise
                    await asyncio.sleep(5)
            if output is None:
                raise RuntimeError("SGLang returned no output")

            meta_info = output.get("meta_info") or {}
            finish_type = (meta_info.get("finish_reason") or {}).get("type")
            if finish_type == "abort":
                llm_error = True
                exit_status = "Error_LLM"
                stop_reason = "sglang_abort"
                break
            if "output_token_logprobs" not in meta_info:
                raise RuntimeError("SGLang response missing output_token_logprobs")

            response_text = str(output.get("text") or "")
            response_tokens = [int(item[1]) for item in meta_info["output_token_logprobs"]]
            response_log_probs = [float(item[0]) for item in meta_info["output_token_logprobs"]]
            if len(sample.tokens) + len(response_tokens) >= max_all_tokens:
                exit_status = "LimitsExceeded"
                stop_reason = "max_context_tokens"
                break

            # Keep the sampled token IDs/logprobs unchanged, but normalize the
            # text before the next chat-template render so it cannot add a
            # duplicate EOS or receive an invalid bare closing think tag.
            response_text = _normalize_response_text(response_text, tokenizer)

            sample.response += response_text
            sample.append_response_tokens(
                args,
                tokens=response_tokens,
                log_probs=response_log_probs,
                trainable=True,
                meta_info=meta_info,
                update_terminal_info=False,
            )
            agent.n_calls += 1
            phase_calls += 1
            if phase == "round0":
                round0_calls += 1
            else:
                repair_calls += 1
            assistant_message = {"role": "assistant", "content": response_text}
            agent.add_messages(assistant_message)
            previous_len = len(agent.messages)
            pending_previous_len = previous_len

            if finish_type == "length":
                if phase == "round0":
                    agent.add_messages(
                        {
                            "role": "tool",
                            "content": SUBMIT_CURRENT_PATCH_PROMPT,
                            "tool_call_id": "slime_f2p_repair",
                        }
                    )
                    phase = "repair"
                    repair_round = 1
                    phase_calls = 0
                    continue
                exit_status = "LimitsExceeded"
                stop_reason = "per_turn_generation_length"
                break

            submission_info = None
            forced_submission = False
            try:
                parsed = agent.parse_response(response_text)
                agent.messages[-1]["extra"] = parsed.get("extra", {})
                outputs, submission_info = await agent.execute_actions(
                    parsed.get("extra", {}).get("actions", []), timeout=observation_timeout
                )
                if submission_info is not None:
                    # Pair the submit tool call before appending hidden-oracle feedback.
                    outputs.append(
                        {
                            "output": "Submission received; running the hidden regression test.",
                            "returncode": 0,
                            "exception_info": "",
                        }
                    )
                for observation in agent.format_observation_messages(agent.messages[-1], outputs):
                    agent.add_messages(observation)
            except FormatError as exc:
                # Qwen3.5 strips historical assistant reasoning when a new
                # user turn appears. Parser feedback is environment feedback,
                # so keep it as a tool turn and preserve the sampled prefix.
                agent.add_messages(
                    *(
                        {
                            **message,
                            "role": "tool",
                            "tool_call_id": message.get("tool_call_id", "slime_format_error"),
                        }
                        for message in exc.messages
                    )
                )
            except EnvironmentUnavailable:
                raise
            except Exception as exc:  # keep ordinary command failures inside the trajectory
                agent.add_messages(
                    {"role": "tool", "content": f"Error: unexpected failure during action execution: {exc}"}
                )

            if len(sample.tokens) >= max_all_tokens:
                exit_status = "LimitsExceeded"
                stop_reason = "max_context_tokens"
                break
            if (
                submission_info is None
                and phase == "repair"
                and phase_calls >= phase_limit
                and force_submit_on_turn_limit
            ):
                submission_info = {
                    "submission": await _current_diff(env, workdir),
                    "tool_call_id": "slime_f2p_repair_turn_limit",
                }
                forced_submission = True
            if submission_info is None:
                continue

            had_submission = True
            current_patch = str(submission_info.get("submission") or "")
            gate = await verify_session.check_f2p(current_patch, nodes)
            submitted_patch = current_patch
            f2p_checks.append(
                {
                    **gate,
                    "patch": None,
                    "phase": phase,
                    "submitted": True,
                    "forced_submission": forced_submission,
                }
            )

            if gate["passed"]:
                exit_status = "Submitted"
                stop_reason = "f2p_passed"
                if phase == "repair":
                    repair_records.append(
                        {
                            "round": repair_round,
                            "turns": phase_calls,
                            "passed": True,
                            "submitted": True,
                            "forced_submission": forced_submission,
                            "patch_sha256": hashlib.sha256(current_patch.encode()).hexdigest() if current_patch else "",
                            "patch_chars": len(current_patch),
                        }
                    )
                break

            if phase == "round0":
                agent.add_messages(
                    {
                        "role": "tool",
                        "content": INITIAL_REPAIR_PROMPT.format(feedback=gate["feedback"]),
                        "tool_call_id": "slime_f2p_repair",
                    }
                )
                phase = "repair"
                repair_round = 1
                phase_calls = 0
                last_submitted_patch = current_patch
                continue

            no_progress = bool(current_patch and current_patch == last_submitted_patch)
            repair_records.append(
                {
                    "round": repair_round,
                    "turns": phase_calls,
                    "passed": False,
                    "submitted": True,
                    "forced_submission": forced_submission,
                    "no_progress": no_progress,
                    "patch_sha256": hashlib.sha256(current_patch.encode()).hexdigest() if current_patch else "",
                    "patch_chars": len(current_patch),
                    "forbidden_test_paths": gate.get("forbidden_test_paths") or [],
                }
            )
            if repair_round >= max_repair_rounds:
                # This was still a valid agent submission; official RM will score its final patch.
                exit_status = "Submitted"
                stop_reason = "max_repair_rounds"
                break

            last_submitted_patch = current_patch
            repair_round += 1
            phase_calls = 0
            agent.add_messages(
                {
                    "role": "tool",
                    "content": FOLLOWUP_REPAIR_PROMPT.format(feedback=gate["feedback"]),
                    "tool_call_id": "slime_f2p_repair",
                }
            )

        workspace_patch = await _current_diff(env, workdir)
        hidden_patch = str(instance.get("f2p_patch") or instance.get("test_patch") or "")
        last_gate_forbidden = (
            f2p_checks[-1].get("forbidden_test_paths") or []
            if f2p_checks and f2p_checks[-1].get("submitted")
            else []
        )
        final_forbidden = list(
            dict.fromkeys([*forbidden_test_paths(submitted_patch, hidden_patch), *last_gate_forbidden])
        )
        if final_forbidden:
            stop_reason = stop_reason or "forbidden_test_changes"
            final_output = ""
        else:
            # A model submission or configured turn-limit fallback may cross the
            # workspace/verifier boundary; unsent workspace state never does.
            final_output = submitted_patch if had_submission else ""

        official_verify: dict[str, Any] = {}
        if final_output and is_valid_patch(final_output):
            official_started = time.monotonic()
            try:
                # The synchronous official evaluator owns its command timeout;
                # cancelling its worker thread would race verifier cleanup.
                official_verify = await verify_session.run_final_official(final_output)
            except Exception as exc:  # retain the rollout; reward falls back to its repair tier
                official_verify = {
                    "resolved": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                }
            official_verify.setdefault(
                "reward_eval_seconds",
                time.monotonic() - official_started,
            )
            official_verify.pop("output", None)

        if exit_status == "Submitted":
            sample.status = Sample.Status.COMPLETED
        elif exit_status in {"LimitsExceeded", "TimeExceeded"}:
            sample.status = Sample.Status.TRUNCATED
        else:
            sample.status = Sample.Status.ABORTED if llm_error else Sample.Status.TRUNCATED

        rollout_id = getattr(state, "rollout_id", None)
        trajectory_dir = _trajectory_root()
        if rollout_id is not None:
            trajectory_dir /= f"rl_step_{rollout_id}"
        trajectory_dir /= instance_id
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = trajectory_dir / f"{instance_id}.sample_{int(sample.index or 0)}.f2p_repair.json"

        repair_metadata = {
            "selected_f2p": nodes,
            "round0_turns": round0_calls,
            "repair_turns": repair_calls,
            "repair_rounds": repair_records,
            "repair_rounds_started": repair_round,
            "repair_rounds_used": len(repair_records),
            "f2p_checks": f2p_checks,
            "f2p_passed": bool(
                f2p_checks
                and f2p_checks[-1].get("passed")
                and f2p_checks[-1].get("submitted")
            ),
            "had_submission": had_submission,
            "force_submit_on_turn_limit": force_submit_on_turn_limit,
            "stop_reason": stop_reason,
            "final_forbidden_test_paths": final_forbidden,
            "official_verify": official_verify,
            "persistent_verifier": True,
            "verify_environment_reused_for_official": bool(official_verify),
            "evaluation": bool(evaluation),
        }
        sample.metadata.update(instance)
        sample.metadata.update(
            {
                "instance_id": instance_id,
                "rl_step": rollout_id,
                "trajectory_path": str(trajectory_path),
                "trajectory": agent.serialize(),
                "n_steps": agent.n_calls,
                "exit_status": exit_status,
                "final_output": final_output,
                "f2p_repair": repair_metadata,
            }
        )

        trajectory_payload = {
            "instance_id": instance_id,
            "instance": instance,
            "rl_step": rollout_id,
            "messages": agent.messages,
            "model_patch": final_output,
            "raw_final_patch": workspace_patch,
            "exit_status": exit_status,
            "stop_reason": stop_reason,
            "n_steps": agent.n_calls,
            "response_length": sample.response_length,
            "loss_mask": sample.loss_mask,
            "f2p_repair": repair_metadata,
            "total_time": time.time() - started,
        }
        trajectory_path.write_text(json.dumps(trajectory_payload, ensure_ascii=False, indent=2, default=str))

        assert len(sample.loss_mask or []) == sample.response_length
        assert len(sample.rollout_log_probs or []) == sample.response_length
        sample._validate_response_metadata_lengths()
        logger.info(
            "[F2P_REPAIR] completed instance=%s sample=%s status=%s f2p=%s rounds=%s reason=%s",
            instance_id,
            sample.index,
            sample.status,
            repair_metadata["f2p_passed"],
            len(repair_records),
            stop_reason,
        )
        return sample
    finally:
        try:
            await verify_session.close()
        except BaseException as exc:
            logger.error("[F2P_REPAIR] failed to close verifier %s: %s", instance_id, exc)
        await _stop_rollout_environments(
            [workspace_env],
            stop_timeout,
            instance_id,
        )
