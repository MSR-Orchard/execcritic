"""Native direct-SGLang mini-swe gentest custom generator.

This keeps the native ``GeneratedTestSubmitAgent`` gate in charge of the
generated-test protocol while routing model calls through slime's SGLang
rollout engine so token ids, loss masks, and rollout logprobs remain trainable.
Repository exploration stays in a workspace sandbox; generated-test feedback,
validation, and in-rollout reward evaluation use a separate verifier sandbox.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import torch

from minisweagent.agents.extra.generated_test_submit import (
    GeneratedTestSubmitAgent,
    evaluate_oracle_contract,
    validate_execution_contract,
)
from minisweagent.config import builtin_config_dir
from minisweagent.exceptions import FormatError, LimitsExceeded
from minisweagent.models.utils.actions_toolcall import (
    format_toolcall_observation_messages,
    parse_toolcall_actions,
    tools_for_names,
)
from minisweagent.run.benchmarks.gentest import (
    _normalize_test_file_path,
    build_config,
    execution_contract_hash,
    extract_example,
    get_gentest_agent_environment,
    get_gentest_environment,
    generated_test_file_for_instance,
    list_from_json_or_obj,
    read_test_file,
    reset_base,
    run_generated_test_official,
    starter_code_for_sample,
    test_command_for_instance,
    test_style_guidance_for_instance,
    validate_generated_test,
    write_generated_test_file,
)
from minisweagent.run.benchmarks.gentest import _submitted_test_names
from minisweagent.run.benchmarks.swerebench import _resolve_per_instance_api_base
from minisweagent.run.benchmarks.swerebench_verify_azure_modal import (
    DEFAULT_CONFIG_FILE as OFFICIAL_VERIFY_CONFIG_FILE,
    prepare_environment_for_evaluation,
)

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.misc import decode_int32_meta_array
from slime.utils.types import Sample

from .gentest_reward import reward_from_record
from .swe_reward import SWE_TIMEOUT_REWARD_TOTAL, compute_patch_classification_reward_in_env
from .swe_wrapper_v2 import EnvironmentUnavailable, _is_environment_error

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "gentest.yaml"
DEFAULT_MAX_ALL_TOKENS = 65536
RETURN_LOGPROB = True

_RUN_TIMESTAMP: str | None = None
_PARSER_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}
_BLOCKING_EXECUTOR: ThreadPoolExecutor | None = None
_BLOCKING_EXECUTOR_WORKERS = 0
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
PATCH_CLASSIFICATION_REWARD_PATH = "examples.mini_swe.swe_reward.reward_func"
_RETRYABLE_INFRA_ERROR = re.compile(
    r"connection (?:reset|refused|aborted)|remote end closed|server disconnected|"
    r"(?:http|status)(?: code)?\s*5(?:02|03|04)|sandbox.*(?:expired|timeout|timed out|unavailable)|"
    r"container exec failed|execution timed out|sglang generation aborted|sglang query failed",
    re.IGNORECASE,
)


@contextmanager
def _record_timing(timings: dict[str, Any], phase: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[phase] = timings.get(phase, 0.0) + time.perf_counter() - started


def _is_retryable_infrastructure_error(exc: Exception) -> bool:
    if isinstance(exc, (EnvironmentUnavailable, httpx.TransportError, TimeoutError, ConnectionError)):
        return True
    text = f"{type(exc).__name__}: {exc}"
    return bool(_is_environment_error(text) or _RETRYABLE_INFRA_ERROR.search(text))


def configure_blocking_executor(max_workers: int) -> None:
    global _BLOCKING_EXECUTOR, _BLOCKING_EXECUTOR_WORKERS
    max_workers = max(1, max_workers)
    if _BLOCKING_EXECUTOR is not None and _BLOCKING_EXECUTOR_WORKERS == max_workers:
        return
    previous_executor = _BLOCKING_EXECUTOR
    _BLOCKING_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gentest-generate")
    _BLOCKING_EXECUTOR_WORKERS = max_workers
    if previous_executor is not None:
        previous_executor.shutdown(wait=False)


async def _run_blocking(function, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_BLOCKING_EXECUTOR, partial(function, *args, **kwargs))


async def _run_blocking_before_environment_cleanup(function, *args, **kwargs):
    """Wait for an env-using worker before allowing its caller to clean the env."""
    task = asyncio.create_task(_run_blocking(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            logger.exception("[GENTEST_GENERATE] in-environment worker failed during cancellation")
        raise


def _active_reward_mode(args, *, evaluation: bool) -> str:
    reward_mode = getattr(args, "swe_reward_mode", None) or os.environ.get("SWE_REWARD_MODE", "simple")
    if evaluation:
        reward_mode = os.environ.get("SWE_EVAL_REWARD_MODE", "simple_truncated_zero")
    return str(reward_mode)


async def _compute_reward_before_environment_cleanup(args, sample: Sample, env, *, evaluation: bool) -> bool:
    if (
        getattr(args, "custom_rm_path", None) != PATCH_CLASSIFICATION_REWARD_PATH
        or _active_reward_mode(args, evaluation=evaluation) != "patch_classification"
        or sample.status == Sample.Status.ABORTED
        or sample.reward is not None
        or _has_format_error(sample)
    ):
        return False

    try:
        # ``wait_for`` cancels the wrapper at the deadline. The wrapper deliberately waits for its
        # non-interruptible env-using thread before propagating cancellation, so this coroutine never
        # returns while a verifier worker can still race the caller's environment cleanup.
        sample.reward = await asyncio.wait_for(
            _run_blocking_before_environment_cleanup(
                compute_patch_classification_reward_in_env,
                args,
                sample,
                env,
            ),
            timeout=SWE_TIMEOUT_REWARD_TOTAL,
        )
    except asyncio.TimeoutError:
        record = sample.metadata.get("gentest_record") if isinstance(sample.metadata, dict) else None
        if isinstance(record, dict):
            record.setdefault("timings", {})
            record["reward_timeout"] = True
            record["infra_failure"] = True
            record["error"] = record.get("error") or f"reward_timeout after {SWE_TIMEOUT_REWARD_TOTAL}s"
        reward_key = getattr(args, "reward_key", None)
        sample.reward = {reward_key: 0.0} if reward_key else 0.0
        sample.status = Sample.Status.ABORTED
        sample.remove_sample = True
        _zero_loss_mask(sample)
        logger.warning(
            "[GENTEST_GENERATE] patch_classification reward timed out after %ss for %s sample=%s; "
            "worker drained and sample was excluded from training",
            SWE_TIMEOUT_REWARD_TOTAL,
            (record or {}).get("instance_id", "unknown") if isinstance(record, dict) else "unknown",
            sample.index,
        )
    return True


async def _create_environment(config: dict[str, Any], instance: dict[str, Any]):
    """Keep ownership until a blocking create either returns or cleans itself up."""
    create_task = asyncio.create_task(_run_blocking(get_gentest_environment, config, instance))
    try:
        return await asyncio.shield(create_task)
    except asyncio.CancelledError:
        try:
            env = await create_task
        except Exception:
            pass
        else:
            try:
                await _run_blocking(env.cleanup)
            except Exception as cleanup_exc:
                logger.warning(
                    "[GENTEST_GENERATE] cleanup after cancelled environment creation failed for %s: %s",
                    instance.get("instance_id", "unknown"),
                    cleanup_exc,
                )
        raise


async def _create_isolated_environments(
    workspace_config: dict[str, Any],
    verifier_config: dict[str, Any],
    instance: dict[str, Any],
    timings: dict[str, Any],
):
    """Create the fresh workspace and verifier concurrently without leaking either on failure."""

    async def create_one(phase: str, config: dict[str, Any]):
        with _record_timing(timings, phase):
            return await _create_environment(copy.deepcopy(config), instance)

    started = time.perf_counter()
    tasks = (
        asyncio.create_task(create_one("workspace_environment_create_sec", workspace_config)),
        asyncio.create_task(create_one("verify_environment_create_sec", verifier_config)),
    )
    try:
        workspace_env, verify_env = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for role, result in zip(("workspace", "verifier"), results, strict=True):
            if isinstance(result, BaseException):
                continue
            try:
                await _run_blocking(result.cleanup)
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning(
                    "[GENTEST_GENERATE] %s cleanup after parallel creation failure failed for %s: %s",
                    role,
                    instance.get("instance_id", "unknown"),
                    cleanup_exc,
                )
        raise
    timings["environment_create_sec"] = time.perf_counter() - started
    return workspace_env, verify_env


class _SGLangGentestModel:
    """mini-swe Model backed by slime's SGLang rollout endpoint."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        tokenizer: Any,
        url: str,
        sampling_params: dict[str, Any],
        llm_timeout: int,
        max_llm_attempts: int,
        max_all_tokens: int,
        use_rollout_routing_replay: bool,
        num_layers: int,
        moe_router_topk: int,
        num_experts: int,
        abort_check: Callable[[], bool] | None = None,
    ):
        self.config = copy.deepcopy(config)
        self.config.setdefault("model_name", "slime-sglang")
        self.config.setdefault("format_error_template", "{{ error }}")
        self.config.setdefault(
            "observation_template",
            "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
            "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>",
        )
        self.config.setdefault("extra_tools", ["grounding_plan", "write_generated_test"])
        self.config.setdefault("multimodal_regex", "")
        self.tokenizer = tokenizer
        self.url = url
        self.sampling_params = copy.deepcopy(sampling_params)
        self.llm_timeout = llm_timeout
        self.max_llm_attempts = max(1, max_llm_attempts)
        self.max_all_tokens = max_all_tokens
        self.use_rollout_routing_replay = use_rollout_routing_replay
        self.num_layers = num_layers
        self.moe_router_topk = moe_router_topk
        self.num_experts = num_experts
        self.abort_check = abort_check
        if self.use_rollout_routing_replay and min(self.num_layers, self.moe_router_topk, self.num_experts) <= 0:
            raise ValueError(
                "Rollout routing replay requires positive num_layers, moe_router_topk, and num_experts, got "
                f"{self.num_layers}, {self.moe_router_topk}, {self.num_experts}."
            )
        self.records: list[dict[str, Any]] = []
        self.prompt_text = ""
        self.prompt_token_ids: list[int] = []
        self.response_token_ids: list[int] = []
        self.loss_mask: list[int] = []
        self.rollout_log_probs: list[float] = []
        self.response_text = ""
        self._rendered_text: str | None = None
        self._latest_routing_record_idx: int | None = None
        self._last_structured_write_record_idx: int | None = None
        self._client = httpx.Client(timeout=httpx.Timeout(None), trust_env=False)
        self._call_count = 0
        self.timings: dict[str, float | int] = {
            "chat_template_sec": 0.0,
            "tokenization_sec": 0.0,
            "generation_sec": 0.0,
            "parse_actions_sec": 0.0,
            "generation_calls": 0,
        }

    @property
    def extra_tools(self) -> list[str]:
        return list(self.config.get("extra_tools") or [])

    @property
    def tools(self) -> list[dict]:
        return tools_for_names(self.extra_tools)

    def close(self) -> None:
        self._client.close()

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def query(self, messages: list[dict], **kwargs) -> dict:
        if self.abort_check is not None and self.abort_check():
            raise RuntimeError("Rollout generation aborted")
        token_stream_messages = []
        latest_record_idx = len(self.records) - 1
        latest_assistant_boundary = None
        latest_assistant_matches = 0
        if self.records:
            content_digest = hashlib.sha256(self.records[latest_record_idx]["content"].encode()).hexdigest()
            latest_assistant_boundary = f"__SLIME_GENTEST_ASSISTANT_BOUNDARY_{latest_record_idx}_{content_digest}__"
        for message in messages:
            extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
            record_idx = extra.get("record_idx")
            if message.get("role") == "assistant" and isinstance(record_idx, int) and 0 <= record_idx < len(self.records):
                # The exact raw assistant bytes are already represented by the
                # generated token IDs. Render them as content solely to locate
                # the new observation suffix; structured tool_calls would emit
                # a second copy of the same call.
                message = copy.deepcopy(message)
                message["content"] = self.records[record_idx]["content"]
                message.pop("tool_calls", None)
                if record_idx == latest_record_idx:
                    message["content"] = latest_assistant_boundary
                    latest_assistant_matches += 1
            token_stream_messages.append(message)
        render_started = time.perf_counter()
        current_prompt = _apply_chat_template(
            self.tokenizer,
            token_stream_messages,
            add_generation_prompt=True,
            tools=self.tools,
        )
        self.timings["chat_template_sec"] += time.perf_counter() - render_started
        if self._rendered_text is None:
            tokenization_started = time.perf_counter()
            current_prompt_ids = self.tokenizer(current_prompt, add_special_tokens=False)["input_ids"]
            self.timings["tokenization_sec"] += time.perf_counter() - tokenization_started
            self.prompt_text = current_prompt
            self.prompt_token_ids = list(current_prompt_ids)
            self._rendered_text = current_prompt
        else:
            if latest_assistant_matches != 1:
                raise RuntimeError(
                    "Expected exactly one message for the latest gentest assistant response, "
                    f"found {latest_assistant_matches} for record_idx={latest_record_idx}."
                )
            boundary_count = current_prompt.count(latest_assistant_boundary)
            if boundary_count != 1:
                raise RuntimeError(
                    "The gentest chat template did not preserve the latest assistant boundary exactly once: "
                    f"record_idx={latest_record_idx} boundary_count={boundary_count}."
                )
            # Generated token IDs already own every byte through the latest
            # assistant response. Only append the template framing and
            # observation text after that response. This remains exact even
            # when a chat template trims or rewrites historical assistant
            # content (for example Qwen3.5 thinking normalization).
            suffix = current_prompt.split(latest_assistant_boundary, 1)[1]
            eos_token = getattr(self.tokenizer, "eos_token", None)
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if (
                eos_token
                and eos_token_id is not None
                and self.records[latest_record_idx]["token_ids"]
                and self.records[latest_record_idx]["token_ids"][-1] == eos_token_id
                and suffix.startswith(eos_token)
            ):
                # SGLang includes the sampled assistant terminator in the
                # generated token IDs. The rendered history includes the same
                # terminator before the tool observation; retain it only once.
                suffix = suffix[len(eos_token) :]
            tokenization_started = time.perf_counter()
            suffix_token_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]
            self.timings["tokenization_sec"] += time.perf_counter() - tokenization_started
            self.response_token_ids.extend(int(token_id) for token_id in suffix_token_ids)
            self.loss_mask.extend([0] * len(suffix_token_ids))
            self.rollout_log_probs.extend([0.0] * len(suffix_token_ids))
            self.response_text += suffix
            self._rendered_text += suffix

        current_input_ids = self.prompt_token_ids + self.response_token_ids
        if len(current_input_ids) >= self.max_all_tokens:
            raise LimitsExceeded(_exit_message("LimitsExceeded"))

        payload = {
            "input_ids": current_input_ids,
            "sampling_params": copy.deepcopy(self.sampling_params),
            "return_logprob": RETURN_LOGPROB,
        }
        if self.use_rollout_routing_replay:
            payload["return_routed_experts"] = True
        generation_started = time.perf_counter()
        try:
            output = self._post_generate(payload)
        finally:
            self.timings["generation_sec"] += time.perf_counter() - generation_started
            self.timings["generation_calls"] += 1
        if self.abort_check is not None and self.abort_check():
            raise RuntimeError("Rollout generation aborted")
        finish_type = (output.get("meta_info") or {}).get("finish_reason", {}).get("type")
        if finish_type == "abort":
            raise RuntimeError("SGLang generation aborted")

        cur_response = output["text"]
        meta_info = output.get("meta_info") or {}
        if "output_token_logprobs" not in meta_info:
            raise RuntimeError("SGLang response missing output_token_logprobs")
        cur_token_ids = [int(item[1]) for item in meta_info["output_token_logprobs"]]
        cur_log_probs = [float(item[0]) for item in meta_info["output_token_logprobs"]]
        routed_experts = None
        if self.use_rollout_routing_replay:
            routed_experts = _decode_routed_experts(
                meta_info,
                token_count=len(current_input_ids) + len(cur_token_ids),
                num_layers=self.num_layers,
                moe_router_topk=self.moe_router_topk,
            )

        if len(current_input_ids) + len(cur_token_ids) >= self.max_all_tokens:
            raise LimitsExceeded(_exit_message("LimitsExceeded"))

        self.response_token_ids.extend(cur_token_ids)
        self.loss_mask.extend([1] * len(cur_token_ids))
        self.rollout_log_probs.extend(cur_log_probs)
        self.response_text += cur_response
        self._rendered_text += cur_response

        record_idx = self._append_record(
            cur_response,
            cur_token_ids,
            cur_log_probs,
            routed_experts,
            meta_info,
        )
        if self.use_rollout_routing_replay:
            previous_idx = self._latest_routing_record_idx
            if previous_idx is not None and previous_idx != self._last_structured_write_record_idx:
                self._release_routing_record(previous_idx)
            self._latest_routing_record_idx = record_idx
        self._call_count += 1
        message = self.format_message(
            role="assistant",
            content=cur_response,
            extra={
                "record_idx": record_idx,
                # content is cleared below for structured tool calls so the
                # chat template does not render the action twice. Keep the
                # exact sampled bytes in extra for trajectory inspection;
                # _strip_extra excludes them from subsequent model context.
                "raw_response": cur_response,
                "cost": 0.0,
                "timestamp": time.time(),
                "finish_reason": meta_info.get("finish_reason", {}),
            },
        )
        if finish_type == "length":
            raise LimitsExceeded(message, _exit_message("LimitsExceeded"))

        parse_started = time.perf_counter()
        try:
            actions, tool_calls = self.parse_text_actions(cur_response, f"call_{self._call_count}")
        except FormatError as exc:
            raise FormatError(message, *exc.messages) from exc
        finally:
            self.timings["parse_actions_sec"] += time.perf_counter() - parse_started
        message["tool_calls"] = tool_calls
        message["extra"]["actions"] = actions
        if tool_calls:
            # The structured tool_calls field renders the same tool call on the
            # next turn. Keeping the raw block in content would render it twice.
            message["content"] = ""
        if any(
            isinstance(action, dict)
            and (action.get("tool") == "write_generated_test" or ("test_code" in action and "command" not in action))
            for action in actions
        ):
            previous_idx = self._last_structured_write_record_idx
            if self.use_rollout_routing_replay and previous_idx is not None and previous_idx != record_idx:
                self._release_routing_record(previous_idx)
            self._last_structured_write_record_idx = record_idx
        return message

    def _post_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_llm_attempts + 1):
            if self.abort_check is not None and self.abort_check():
                raise RuntimeError("Rollout generation aborted")
            response = None
            try:
                response = self._client.post(self.url, json=payload, timeout=self.llm_timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.error(
                    "[GENTEST_GENERATE] SGLang query failed attempt=%s/%s: %s",
                    attempt,
                    self.max_llm_attempts,
                    exc,
                )
                if self.abort_check is not None and self.abort_check():
                    raise RuntimeError("Rollout generation aborted") from exc
                if attempt >= self.max_llm_attempts:
                    raise
                time.sleep(5)
            finally:
                if response is not None:
                    response.close()
        raise RuntimeError("SGLang query failed") from last_exc

    def _append_record(
        self,
        content: str,
        token_ids: list[int],
        log_probs: list[float],
        routed_experts: torch.Tensor | None,
        meta_info: dict[str, Any],
    ) -> int:
        record_idx = len(self.records)
        self.records.append(
            {
                "content": content,
                "token_ids": token_ids,
                "log_probs": log_probs,
                "response_length_after": len(self.response_token_ids),
                "response_text_length_after": len(self.response_text),
                "routed_experts": routed_experts,
                "finish_reason": meta_info.get("finish_reason", {}),
            }
        )
        return record_idx

    def _release_routing_record(self, record_idx: int) -> None:
        self.records[record_idx]["routed_experts"] = None

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.config["observation_template"],
            template_vars=template_vars,
            multimodal_regex=self.config.get("multimodal_regex", ""),
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return dict(self.config)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config,
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "model_stats": {
                    "recorded_sglang_calls": len(self.records),
                },
            }
        }

    def parse_text_actions(self, text: str, call_prefix: str) -> tuple[list[dict], list[dict]]:
        raw_calls = _parse_sglang_tool_calls(text, self.tools)
        tool_calls = []
        for idx, call in enumerate(raw_calls):
            arguments = call.parameters if isinstance(call.parameters, str) else json.dumps(call.parameters or {})
            tool_calls.append(
                SimpleNamespace(
                    id=f"{call_prefix}_{idx}",
                    function=SimpleNamespace(name=call.name or "", arguments=arguments),
                )
            )
        actions = parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config["format_error_template"],
            extra_tools=self.extra_tools,
        )
        formatted = [
            {
                "id": item.id,
                "type": "function",
                "function": {
                    "name": item.function.name,
                    "arguments": item.function.arguments,
                },
            }
            for item in tool_calls
        ]
        return actions, formatted


def _parse_sglang_tool_calls(text: str, tools: list[dict]) -> list[Any]:
    parser_name = os.environ.get("GENTEST_SGLANG_TOOL_CALL_PARSER", "qwen3_coder")
    cache_key = (parser_name, tuple(tool["function"]["name"] for tool in tools))
    parser = _PARSER_CACHE.get(cache_key)
    if parser is None:
        from sglang.srt.entrypoints.openai.protocol import Function, Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        sg_tools = [Tool(type="function", function=Function(**tool["function"])) for tool in tools]
        parser = FunctionCallParser(tools=sg_tools, tool_call_parser=parser_name)
        _PARSER_CACHE[cache_key] = parser
    try:
        _normal_text, calls = parser.parse_non_stream(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("SGLang tool parser failed, falling back to ChatML block parser: %s", exc)
    else:
        if calls:
            return calls
    return _parse_tool_call_blocks(text)


def _parse_tool_call_blocks(text: str) -> list[Any]:
    calls = []
    for idx, match in enumerate(_TOOL_CALL_BLOCK.finditer(text or "")):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name") or payload.get("tool")
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            arguments_text = arguments
        else:
            arguments_text = json.dumps(arguments or {})
        calls.append(SimpleNamespace(name=name, parameters=arguments_text, index=idx))
    return calls


def _exit_message(status: str) -> dict[str, Any]:
    return {
        "role": "exit",
        "content": status,
        "extra": {"exit_status": status, "submission": ""},
    }


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _config_specs() -> list[str]:
    specs = [os.environ.get("GENTEST_CONFIG", str(DEFAULT_CONFIG_FILE))]
    extra = os.environ.get("GENTEST_CONFIG_SPECS", "")
    specs.extend(spec for spec in extra.split(";;") if spec.strip())
    return specs


def _official_verifier_config_specs() -> list[str]:
    specs = [os.environ.get("GENTEST_VERIFY_CONFIG", str(OFFICIAL_VERIFY_CONFIG_FILE))]
    extra = os.environ.get("GENTEST_VERIFY_CONFIG_SPECS", "")
    specs.extend(spec for spec in extra.split(";;") if spec.strip())
    return specs


def _load_gentest_config(instance_id: str) -> dict:
    config = build_config(
        _config_specs(),
        model=os.environ.get("GENTEST_MODEL_NAME", "slime-sglang"),
        model_class=None,
        environment_class=os.environ.get("GENTEST_ENVIRONMENT_CLASS"),
    )
    env_config = config.setdefault("environment", {})
    if os.environ.get("SANDBOX_BASE_URL"):
        env_config["base_url"] = os.environ["SANDBOX_BASE_URL"]
    if os.environ.get("SANDBOX_API_KEY"):
        env_config["api_key"] = os.environ["SANDBOX_API_KEY"]
    if os.environ.get("GENTEST_SANDBOX_TIMEOUT"):
        env_config["sandbox_timeout"] = _env_int("GENTEST_SANDBOX_TIMEOUT", int(env_config.get("sandbox_timeout", 3600)))
    if os.environ.get("GENTEST_ENV_TIMEOUT"):
        env_config["timeout"] = _env_int("GENTEST_ENV_TIMEOUT", int(env_config.get("timeout", 60)))
    if os.environ.get("GENTEST_ENV_CPU"):
        env_config["cpu"] = os.environ["GENTEST_ENV_CPU"]
    if os.environ.get("GENTEST_ENV_MEMORY"):
        env_config["memory"] = os.environ["GENTEST_ENV_MEMORY"]
    _resolve_per_instance_api_base(config, instance_id)
    return config


def _load_official_verifier_config(instance_id: str) -> dict:
    """Load verifier transport and lifecycle settings from the official config."""
    config = build_config(
        _official_verifier_config_specs(),
        model=None,
        model_class=None,
        environment_class="azure_modal",
    )
    env_config = config.setdefault("environment", {})
    # Verification must match the official evaluator, which permits dependency
    # downloads during its per-instance setup.  Keep this independent from the
    # generator workspace's network policy and from arbitrary config overlays.
    env_config["block_network"] = False
    if os.environ.get("SANDBOX_BASE_URL"):
        env_config["base_url"] = os.environ["SANDBOX_BASE_URL"]
    if os.environ.get("SANDBOX_API_KEY"):
        env_config["api_key"] = os.environ["SANDBOX_API_KEY"]
    if os.environ.get("GENTEST_VERIFY_SANDBOX_TIMEOUT"):
        env_config["sandbox_timeout"] = _env_int(
            "GENTEST_VERIFY_SANDBOX_TIMEOUT", int(env_config.get("sandbox_timeout", 600))
        )
    if os.environ.get("GENTEST_VERIFY_REQUEST_TIMEOUT"):
        env_config["request_timeout"] = _env_int(
            "GENTEST_VERIFY_REQUEST_TIMEOUT", int(env_config.get("request_timeout", 660))
        )
    if os.environ.get("GENTEST_VERIFY_CLEANUP_TIMEOUT"):
        env_config["cleanup_timeout"] = _env_int(
            "GENTEST_VERIFY_CLEANUP_TIMEOUT", int(env_config.get("cleanup_timeout", 120))
        )
    if os.environ.get("GENTEST_VERIFY_ENV_CPU"):
        env_config["cpu"] = os.environ["GENTEST_VERIFY_ENV_CPU"]
    if os.environ.get("GENTEST_VERIFY_ENV_MEMORY"):
        env_config["memory"] = os.environ["GENTEST_VERIFY_ENV_MEMORY"]
    _resolve_per_instance_api_base(config, instance_id)
    return config


def _problem_statement_from_sample(sample: Sample) -> str:
    if isinstance(sample.metadata, dict) and sample.metadata.get("problem_statement"):
        return str(sample.metadata["problem_statement"])
    if isinstance(sample.prompt, str):
        return sample.prompt
    for message in reversed(sample.prompt or []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
    return str(sample.prompt)


def _instance_from_sample(sample: Sample) -> dict[str, Any]:
    metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
    if "metadata" in metadata and isinstance(metadata["metadata"], dict):
        metadata = {**metadata["metadata"], **{k: v for k, v in metadata.items() if k != "metadata"}}
    metadata.setdefault("problem_statement", _problem_statement_from_sample(sample))
    if sample.label is not None:
        metadata.setdefault("patch", sample.label)
    metadata.setdefault("instance_id", f"task_{sample.index}")
    instance_id = str(metadata["instance_id"])
    if not _SAFE_INSTANCE_ID.fullmatch(instance_id):
        raise ValueError(f"unsafe instance_id for trajectory path: {instance_id!r}")
    metadata["instance_id"] = instance_id
    return metadata


def _opts_for_sample(sample: Sample, trajectory_dir: Path) -> dict[str, Any]:
    return {
        "output_dir": str(trajectory_dir),
        "prompt_mode": os.environ.get("GENTEST_PROMPT_MODE", "zero_shot"),
        "example_shot": os.environ.get("GENTEST_EXAMPLE_SHOT", "node"),
        "starter": os.environ.get("GENTEST_STARTER", "gold_imports"),
        "test_file": os.environ.get("GENTEST_TEST_FILE", "test_model_gen.py"),
        "seed": _env_int("GENTEST_SEED", 0),
        "gold_eval": _env_flag("GENTEST_GOLD_EVAL", True),
        "apply_test_patch_at_start": _env_flag("GENTEST_APPLY_TEST_PATCH_AT_START", True),
        "test_timeout": _env_int("GENTEST_TEST_TIMEOUT", 240),
        "sample_idx": int(sample.index or 0),
    }


def _json_mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_tool_call_for_chat_template(tool_call: Any) -> Any:
    if not isinstance(tool_call, dict):
        return tool_call
    normalized = copy.deepcopy(tool_call)
    function = normalized.get("function")
    if isinstance(function, dict) and "arguments" in function:
        function["arguments"] = _json_mapping_or_empty(function.get("arguments"))
    elif "arguments" in normalized:
        normalized["arguments"] = _json_mapping_or_empty(normalized.get("arguments"))
    return normalized


def _strip_extra(messages: list[dict]) -> list[dict]:
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    normalized = []
    for message in messages:
        item = {key: value for key, value in message.items() if key in allowed}
        if isinstance(item.get("tool_calls"), list):
            item["tool_calls"] = [_normalize_tool_call_for_chat_template(tool_call) for tool_call in item["tool_calls"]]
        normalized.append(item)
    return normalized


def _apply_chat_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool, tools: list[dict]) -> str:
    template_kwargs = {}
    if os.environ.get("GENTEST_ENABLE_THINKING") not in (None, ""):
        template_kwargs["enable_thinking"] = _env_flag("GENTEST_ENABLE_THINKING", True)
    return tokenizer.apply_chat_template(
        _strip_extra(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        **template_kwargs,
    )


def _trajectory_root() -> Path:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = os.environ.get("WANDB_RANDOM_SUFFIX", "")
    name = _RUN_TIMESTAMP + (f"_{suffix}" if suffix else "")
    return Path(os.environ.get("GENTEST_TRAJECTORY_DIR") or os.environ.get("SWE_TRAJECTORY_DIR", "trajectories")) / name


def _record_error(instance: dict, sample: Sample, error: str, traceback_text: str = "") -> dict[str, Any]:
    return {
        "instance_id": instance.get("instance_id", ""),
        "sample_idx": int(sample.index or 0),
        "agent_exit_status": "Error",
        "error": error,
        "traceback": traceback_text[-4000:],
    }


def _build_rollout_tensors(
    model: _SGLangGentestModel,
    record_idx: int | None = None,
) -> tuple[list[int], str, list[int], list[int], list[float], torch.Tensor | None, str]:
    """Return a prefix of the exact append-only token stream sent to SGLang."""
    if not model.records:
        raise RuntimeError("No SGLang records were found for the gentest trajectory.")

    if record_idx is None:
        record_idx = len(model.records) - 1
    if not 0 <= record_idx < len(model.records):
        raise RuntimeError(f"Invalid SGLang record index: {record_idx}")
    record = model.records[record_idx]
    response_length = int(record["response_length_after"])
    response_text_length = int(record["response_text_length_after"])

    response_token_ids = list(model.response_token_ids[:response_length])
    loss_mask = list(model.loss_mask[:response_length])
    rollout_log_probs = list(model.rollout_log_probs[:response_length])
    if not (len(response_token_ids) == len(loss_mask) == len(rollout_log_probs) == response_length):
        raise RuntimeError(
            "The model-owned gentest token stream is internally misaligned: "
            f"tokens={len(response_token_ids)} mask={len(loss_mask)} "
            f"logprobs={len(rollout_log_probs)} expected={response_length}"
        )

    routed_experts = None
    if model.use_rollout_routing_replay:
        routed_experts = record["routed_experts"]
        if routed_experts is None:
            raise RuntimeError(f"No routed experts were retained for SGLang record {record_idx}.")
        expected_rows = len(model.prompt_token_ids) + response_length - 1
        if routed_experts.shape[0] != expected_rows:
            raise RuntimeError(
                "SGLang routed_experts/token length mismatch: "
                f"routed_rows={routed_experts.shape[0]} token_transitions={expected_rows}"
            )

    return (
        list(model.prompt_token_ids),
        model.prompt_text,
        response_token_ids,
        loss_mask,
        rollout_log_probs,
        routed_experts,
        model.response_text[:response_text_length],
    )


def _decode_routed_experts(
    meta_info: dict[str, Any],
    *,
    token_count: int,
    num_layers: int,
    moe_router_topk: int,
) -> torch.Tensor:
    routed_experts = decode_int32_meta_array(meta_info, "routed_experts")
    if routed_experts is None:
        raise RuntimeError("SGLang response missing routed_experts while rollout routing replay is enabled.")
    expected_shape = (token_count - 1, num_layers, moe_router_topk)
    expected_numel = expected_shape[0] * expected_shape[1] * expected_shape[2]
    if routed_experts.numel() != expected_numel:
        raise RuntimeError(
            "SGLang routed_experts size mismatch: "
            f"got={routed_experts.numel()} expected={expected_numel} shape={expected_shape}"
        )
    return routed_experts.reshape(expected_shape)


def _placeholder_routed_experts(args, token_count: int) -> torch.Tensor:
    shape = (token_count - 1, args.num_layers, args.moe_router_topk)
    routed_experts = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.int32).reshape(shape)
    routed_experts.remainder_(args.num_experts)
    return routed_experts


def _last_structured_write_message_index(messages: list[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        actions = (messages[index].get("extra") or {}).get("actions") or []
        if any(
            isinstance(action, dict)
            and (action.get("tool") == "write_generated_test" or ("test_code" in action and "command" not in action))
            for action in actions
        ):
            return index
    return None


async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    """Slime custom-generate entrypoint with whole-environment retry.

    The default ``sglang_rollout.generate_rollout`` owns batching, dynamic
    filtering, reward invocation, and eval orchestration.  This function owns
    one gentest agent trajectory and converts it to a trainable ``Sample``.
    """
    default_workers = max(
        2,
        2
        * int(getattr(args, "rollout_batch_size", 1) or 1)
        * int(getattr(args, "n_samples_per_prompt", 1) or 1),
    )
    configure_blocking_executor(_env_int("GENTEST_NATIVE_WORKERS", default_workers))

    resource_level = 1
    max_retries = max(1, _env_int("GENTEST_MAX_ENV_RETRIES", 2))
    retry_wait = _env_int("GENTEST_ENV_RETRY_WAIT", 10)
    for attempt in range(1, max_retries + 1):
        try:
            sample = await _generate_once(args, sample, sampling_params, resource_level, evaluation=evaluation)
            gentest_record = sample.metadata.get("gentest_record") if isinstance(sample.metadata, dict) else None
            if isinstance(gentest_record, dict):
                gentest_record["generate_attempts"] = attempt
                gentest_record["resource_level"] = resource_level
            if _has_format_error(sample):
                reward_key = getattr(args, "reward_key", None)
                sample.reward = {reward_key: 0.0} if reward_key else 0.0
            return sample
        except Exception as exc:  # noqa: BLE001
            retryable = _is_retryable_infrastructure_error(exc)
            logger.error(
                "[GENTEST_GENERATE] generate attempt %s/%s failed for sample=%s retryable=%s: %s",
                attempt,
                max_retries,
                sample.index,
                retryable,
                exc,
                exc_info=True,
            )
            if retryable and attempt < max_retries:
                resource_level += 1
                sample.reward = None
                await asyncio.sleep(retry_wait)
                continue

            instance = _instance_from_sample(sample)
            existing_record = sample.metadata.get("gentest_record") if isinstance(sample.metadata, dict) else None
            record = existing_record if isinstance(existing_record, dict) else _record_error(
                instance,
                sample,
                f"{type(exc).__name__}: {str(exc)[:300]}",
                traceback.format_exc(),
            )
            record["retryable_infrastructure_error"] = retryable
            record["generate_attempts"] = attempt
            sample.metadata["gentest_record"] = record
            sample.status = Sample.Status.ABORTED
            _fill_placeholder_trajectory(args, sample, instance)
            _zero_loss_mask(sample)
            # Aborted samples (dead sandbox / infra failure) carry no learning signal:
            # their loss mask is zeroed and they are excluded from the GRPO group
            # mean/std. Reward is 0.0 (not the -1.0 unsubmitted penalty) so nothing
            # downstream mistakes them for a genuinely bad trajectory.
            reward = 0.0
            reward_key = getattr(args, "reward_key", None)
            sample.reward = {reward_key: reward} if reward_key else reward
            return sample


def _fill_placeholder_trajectory(args, sample: Sample, instance: dict[str, Any]) -> None:
    """Give an exception-path aborted sample a minimal *trainable* token trajectory
    so it still occupies its batch slot (keeping GRPO group sizes / global_batch_size
    aligned). loss_mask is 1 so the sample's reward contributes gradient like any other
    aborted sample. Note: the exception fires before the model produces its own tokens,
    so the single masked response token is a placeholder (eos), not real model output."""
    tokenizer = GenerateState(args).tokenizer
    problem = str(instance.get("problem_statement", "") or "")[:2000] or "placeholder"
    try:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:  # noqa: BLE001
        prompt_text = problem
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if not prompt_token_ids:
        prompt_token_ids = [tokenizer.eos_token_id or 0]
    response_token_ids = [tokenizer.eos_token_id or prompt_token_ids[-1]]

    sample.prompt = prompt_text
    sample.tokens = list(prompt_token_ids) + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = ""
    sample.loss_mask = [1] * len(response_token_ids)
    sample.rollout_log_probs = [0.0] * len(response_token_ids)
    if getattr(args, "use_rollout_routing_replay", False):
        sample.rollout_routed_experts = _placeholder_routed_experts(args, len(sample.tokens))


async def _generate_once(
    args, sample: Sample, sampling_params: dict[str, Any], resource_level: int, evaluation: bool = False
) -> Sample:
    start_time = time.time()
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    rollout_id = getattr(state, "rollout_id", None)

    instance = _instance_from_sample(sample)
    instance_id = str(instance["instance_id"])
    config = _load_gentest_config(instance_id)
    execution_contract_mode = bool((config.get("agent") or {}).get("require_execution_contract"))
    verifier_config = _load_official_verifier_config(instance_id)
    env_config = config.setdefault("environment", {})
    verifier_env_config = verifier_config.setdefault("environment", {})
    if resource_level > 1:
        env_config["cpu"] = str(resource_level * 2)
        env_config["memory"] = f"{resource_level * 8}Gi"
        verifier_env_config["cpu"] = str(resource_level * 2)
        verifier_env_config["memory"] = f"{resource_level * 8}Gi"

    trajectory_dir = _trajectory_root()
    if rollout_id is not None:
        trajectory_dir = trajectory_dir / f"rl_step_{rollout_id}"
    trajectory_dir = trajectory_dir / instance_id
    traj_path = trajectory_dir / f"{instance_id}.sample_{int(sample.index or 0)}.traj.json"
    opts = _opts_for_sample(sample, trajectory_dir)

    env = None
    verify_env = None
    agent = None
    model = None
    record: dict[str, Any] = {
        "instance_id": instance_id,
        "sample_idx": int(sample.index or 0),
        "prompt_mode": opts["prompt_mode"],
        "example_shot": opts["example_shot"],
        "starter": opts["starter"],
        "agent_inferred_execution_contract": execution_contract_mode,
        "use_dataset_test_patch": not execution_contract_mode,
        "apply_test_patch_at_start": bool(opts["apply_test_patch_at_start"]),
        "official_verify_format": True,
        "official_verifier": {
            "config": os.environ.get("GENTEST_VERIFY_CONFIG", str(OFFICIAL_VERIFY_CONFIG_FILE)),
            "block_network": bool(verifier_env_config.get("block_network", False)),
            "setup_once": not _env_flag("GENTEST_VERIFY_ASSUME_PREPARED", False),
        },
        "rl_step": rollout_id,
        "timings": {},
    }
    phase_timings = record["timings"]
    try:
        if execution_contract_mode and (
            opts["prompt_mode"] != "zero_shot"
            or opts["example_shot"] != "none"
            or opts["starter"] != "empty"
            or opts["apply_test_patch_at_start"]
        ):
            raise ValueError(
                "agent-inferred execution-contract mode requires zero-shot, no example, empty starter, "
                "and no dataset test patch"
            )
        logger.info(
            "[GENTEST_GENERATE] creating isolated workspace/verifier environments for %s sample=%s",
            instance_id,
            sample.index,
        )
        env, verify_env = await _create_isolated_environments(
            config, verifier_config, instance, phase_timings
        )

        if not _env_flag("GENTEST_VERIFY_ASSUME_PREPARED", False):
            setup_timeout = _env_int("GENTEST_VERIFY_SETUP_TIMEOUT", 600)
            with _record_timing(phase_timings, "verify_environment_setup_sec"):
                setup_result = await _run_blocking(
                    prepare_environment_for_evaluation,
                    verify_env,
                    instance,
                    timeout=setup_timeout,
                )
            record["official_verifier"]["setup"] = setup_result

        requested_test_file = opts["test_file"]
        test_file = (
            _normalize_test_file_path(requested_test_file)
            if execution_contract_mode
            else generated_test_file_for_instance(instance, requested_test_file)
        )
        chosen = ""
        example_node, example_src = "", ""
        if execution_contract_mode:
            starter_text, starter_imports, starter_status = "", [], "disabled:agent_inferred_execution_contract"
        else:
            all_f2p = list_from_json_or_obj(instance.get("FAIL_TO_PASS"))
            if all_f2p and (opts["prompt_mode"] == "one_shot_gold" or opts["starter"] == "gold_imports"):
                chosen = str(random.Random(f"{opts['seed']}:{instance_id}:{sample.index}").choice(all_f2p))
            example_node, example_src = extract_example(
                instance.get("test_patch", "") or "", chosen, opts["example_shot"]
            )
            if opts["prompt_mode"] != "one_shot_gold":
                example_node, example_src = "", ""
            with _record_timing(phase_timings, "starter_sec"):
                starter_text, starter_imports, starter_status = await _run_blocking(
                    starter_code_for_sample, verify_env, instance, opts, chosen
                )
        test_patch_start = {
            "enabled": bool(opts["apply_test_patch_at_start"]),
            "phase": "generation",
            "applied": False,
            "reason": "",
            "error": "",
            "isolated_to_verifier": True,
        }
        record["test_patch_start"] = test_patch_start
        if starter_text or not execution_contract_mode:
            with _record_timing(phase_timings, "write_starter_sec"):
                await _run_blocking(write_generated_test_file, env, test_file, starter_text)
        record.update(
            {
                "test_filename": test_file,
                "requested_test_filename": requested_test_file,
                "starter_example_node": chosen,
                "starter_status": starter_status,
                "starter_imports": starter_imports,
                "starter_code_chars": len(starter_text),
                "example_node": example_node,
            }
        )

        agent_config = copy.deepcopy(config.get("agent", {}))
        agent_config.setdefault("agent_class", "generated_test_submit")
        agent_config["output_path"] = traj_path
        agent_config["test_file"] = test_file
        agent_config["allowed_generated_paths"] = [] if execution_contract_mode else [test_file]
        if not execution_contract_mode:
            agent_config["self_test_command"] = f"cd /testbed && {test_command_for_instance(instance, test_file)}"
        agent_config["self_test_timeout"] = int(opts["test_timeout"])
        agent_config["initial_test_hash"] = hashlib.sha256(starter_text.encode("utf-8", "replace")).hexdigest()
        agent_config["stop_on_no_tool_call_format_error"] = _env_flag(
            "GENTEST_TRUNCATE_ON_NO_TOOL_CALL", True
        )

        max_all_tokens = _env_int(
            "GENTEST_MAX_ALL_TOKENS", int(getattr(args, "rollout_max_context_len", None) or DEFAULT_MAX_ALL_TOKENS)
        )
        model = _SGLangGentestModel(
            config.get("model", {}),
            tokenizer=tokenizer,
            url=url,
            sampling_params=sampling_params,
            llm_timeout=_env_int("GENTEST_TIMEOUT_LLM_INFERENCE", 300),
            max_llm_attempts=_env_int("GENTEST_MAX_LLM_ATTEMPTS", 2),
            max_all_tokens=max_all_tokens,
            use_rollout_routing_replay=bool(getattr(args, "use_rollout_routing_replay", False)),
            num_layers=int(args.num_layers),
            moe_router_topk=int(getattr(args, "moe_router_topk", 0) or 0),
            num_experts=int(getattr(args, "num_experts", 0) or 0),
        )

        def run_isolated_self_test(
            test_artifact: str,
            selected_test_file: str = test_file,
            selected_test_command: str = "",
        ) -> dict[str, Any]:
            reset_base(verify_env)
            return run_generated_test_official(
                verify_env,
                instance,
                selected_test_file if execution_contract_mode else test_file,
                test_artifact,
                int(opts["test_timeout"]),
                test_command=selected_test_command if execution_contract_mode else None,
                include_dataset_test_patch=(
                    bool(opts["apply_test_patch_at_start"]) if not execution_contract_mode else False
                ),
                skip_install=True,
            )

        agent = GeneratedTestSubmitAgent(
            model,
            get_gentest_agent_environment(env, instance),
            self_test_runner=run_isolated_self_test,
            **agent_config,
        )
        run_kwargs = {
            "instance_id": instance_id,
            "sample_idx": int(sample.index or 0),
            "prompt_mode": opts["prompt_mode"],
            "example_node": example_node,
            "example_src": example_src,
            "test_file": test_file,
            "test_style_guidance": (
                "" if execution_contract_mode else test_style_guidance_for_instance(instance, test_file)
            ),
            "starter_status": starter_status,
            "starter_imports": starter_imports,
            "starter_code": starter_text,
        }
        with _record_timing(phase_timings, "agent_sec"):
            info = await _run_blocking(
                agent.run,
                str(instance.get("problem_statement", "")),
                **run_kwargs,
            )
        phase_timings["model"] = dict(model.timings)
        gate_metrics = getattr(agent, "gate_info", {})
        phase_timings["self_test_sec"] = sum(
            float(attempt.get("duration_sec") or 0.0)
            for attempt in gate_metrics.get("self_test_attempts", [])
            if isinstance(attempt, dict)
        )
        exit_status = str(info.get("exit_status") or "Unknown")
        format_error_messages = [
            message
            for message in agent.messages
            if message.get("role") == "user"
            and isinstance(message.get("extra"), dict)
            and message["extra"].get("interrupt_type") == "FormatError"
        ]
        no_tool_error_count = sum(
            "No tool calls found in the response" in str(message.get("content") or "")
            for message in format_error_messages
        )
        truncation = None
        training_record_idx = None
        if no_tool_error_count and agent_config["stop_on_no_tool_call_format_error"]:
            last_write_index = _last_structured_write_message_index(agent.messages)
            training_record_idx = model._last_structured_write_record_idx
            truncation = {
                "reason": "no_tool_call_format_error",
                "last_write_message_index": last_write_index,
                "original_message_count": len(agent.messages),
                "kept_message_count": (last_write_index + 1) if last_write_index is not None else len(agent.messages),
                "discarded_message_count": (
                    len(agent.messages) - last_write_index - 1 if last_write_index is not None else 0
                ),
            }
        with _record_timing(phase_timings, "build_rollout_tensors_sec"):
            (
                prompt_token_ids,
                init_prompt_text,
                response_token_ids,
                loss_mask,
                rollout_log_probs,
                rollout_routed_experts,
                response,
            ) = await _run_blocking(_build_rollout_tensors, model, training_record_idx)
        info_gate = info.get("gentest_gate") if isinstance(info.get("gentest_gate"), dict) else {}
        contract_error = ""
        if execution_contract_mode:
            execution_contract, contract_error = validate_execution_contract(
                info_gate.get("test_file"),
                info_gate.get("test_command"),
                info_gate.get("command_evidence"),
            )
            if contract_error:
                selected_test_file = test_file
                selected_test_command = ""
                command_evidence = ""
                record["error"] = f"missing_or_invalid_execution_contract: {contract_error}"
            else:
                selected_test_file = execution_contract["test_file_path"]
                selected_test_command = execution_contract["test_command"]
                command_evidence = execution_contract["command_evidence"]
        else:
            selected_test_file = test_file
            selected_test_command = ""
            command_evidence = ""
        code = str(info_gate.get("test_code") or info.get("submission") or "")
        if not code and not contract_error:
            with _record_timing(phase_timings, "read_test_file_sec"):
                code = await _run_blocking(read_test_file, env, selected_test_file)
        oracle_quality = info_gate.get("oracle_quality") if isinstance(info_gate.get("oracle_quality"), dict) else {}
        oracle_contract = oracle_quality.get("contract") if isinstance(oracle_quality.get("contract"), dict) else {}
        latest_self_test = info_gate.get("latest_self_test") if isinstance(info_gate.get("latest_self_test"), dict) else {}
        code_hash = hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()
        selected_execution_hash = (
            execution_contract_hash(code, selected_test_file, selected_test_command)
            if execution_contract_mode and not contract_error
            else ""
        )
        reused_base_result = (
            latest_self_test
            if latest_self_test.get("clean_fail")
            and latest_self_test.get("test_hash") == code_hash
            and (
                not execution_contract_mode
                or latest_self_test.get("execution_hash") == selected_execution_hash
            )
            else None
        )
        reused_base_test_patch = None
        if reused_base_result is not None:
            reused_base_test_patch = {
                "enabled": bool(opts["apply_test_patch_at_start"]),
                "phase": "base",
                "applied": bool(opts["apply_test_patch_at_start"]),
                "reason": "",
                "error": "",
                "isolated_to_verifier": True,
                "official_verify_format": True,
                "reused": True,
            }
        if contract_error:
            validation = {
                "label": "missing_or_invalid_execution_contract",
                "base_clean_fail": False,
                "gold_pass": False,
                "infra_failure": False,
            }
        else:
            with _record_timing(phase_timings, "validation_sec"):
                validation = await _run_blocking(
                    validate_generated_test,
                    verify_env,
                    instance,
                    code,
                    filename=selected_test_file,
                    test_command=selected_test_command or None,
                    test_timeout=int(opts["test_timeout"]),
                    gold_eval=bool(opts["gold_eval"]),
                    oracle=oracle_contract,
                    apply_test_patch_at_start=bool(opts["apply_test_patch_at_start"]),
                    official_verify_format=True,
                    official_skip_install=True,
                    base_result=reused_base_result,
                    base_test_patch=reused_base_test_patch,
                    generated_test_names=_submitted_test_names(info_gate, code),
                )

        record.update(
            {
                "agent_exit_status": exit_status,
                "agent_submission_chars": len(info.get("submission", "") or ""),
                "agent_calls": agent.n_calls,
                "format_error_count": len(format_error_messages),
                "no_tool_error_count": no_tool_error_count,
                "rollout_truncation": truncation,
                "agent_cost": round(float(agent.cost), 4),
                "gate": gate_metrics,
                "submitted_gate": info_gate,
                "test_filename": selected_test_file,
                "generated_test_path": selected_test_file,
                "generated_test_command": selected_test_command,
                "command_evidence": command_evidence,
                "execution_hash": selected_execution_hash,
                "oracle": oracle_contract,
                "oracle_quality": evaluate_oracle_contract(oracle_contract, code),
                "test_code": code,
                "validation": validation,
                "trajectory_path": str(traj_path),
                "total_time": time.time() - start_time,
            }
        )

        sample.tokens = prompt_token_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        sample.response = response
        sample.loss_mask = loss_mask
        sample.prompt = init_prompt_text
        sample.rollout_log_probs = rollout_log_probs
        sample.rollout_routed_experts = rollout_routed_experts
        if exit_status == "Submitted":
            sample.status = Sample.Status.COMPLETED
        elif exit_status in {"LimitsExceeded", "TimeExceeded"}:
            sample.status = Sample.Status.TRUNCATED
        else:
            sample.status = Sample.Status.ABORTED
        sample.metadata.update(
            {
                "instance_id": instance_id,
                "rl_step": rollout_id,
                "trajectory_path": str(traj_path),
                "test_code": code,
                "test_filename": selected_test_file,
                "requested_test_filename": requested_test_file,
                "generated_test_command": selected_test_command,
                "agent_inferred_execution_contract": execution_contract_mode,
                "gentest_record": record,
                "exit_status": exit_status,
                "format_error_count": len(format_error_messages),
                "no_tool_error_count": no_tool_error_count,
            }
        )
        reward_started = time.perf_counter()
        reward_sandbox_reused = await _compute_reward_before_environment_cleanup(
            args,
            sample,
            verify_env,
            evaluation=evaluation,
        )
        if reward_sandbox_reused:
            phase_timings["reward_sec"] = time.perf_counter() - reward_started
            record["reward_sandbox_reused"] = True
            record["reward_verifier_sandbox_reused"] = True

        assert len(sample.loss_mask) == sample.response_length
        assert sample.rollout_log_probs is None or len(sample.rollout_log_probs) == sample.response_length
        if sample.rollout_routed_experts is not None:
            assert sample.rollout_routed_experts.shape[0] == len(sample.tokens) - 1

        trajectory_dir.mkdir(parents=True, exist_ok=True)
        record["total_time"] = time.time() - start_time
        phase_timings["total_sec_before_cleanup"] = record["total_time"]
        with _record_timing(phase_timings, "trajectory_save_sec"):
            agent.save(
                traj_path,
                {
                    "info": {
                        "instance_id": instance_id,
                        "sample_idx": int(sample.index or 0),
                        "gentest_record": _compact_record(record),
                    },
                    "instance": instance,
                    "rl_step": rollout_id,
                    "prompt_token_ids": prompt_token_ids,
                    "response_token_ids": response_token_ids,
                    "loss_mask": loss_mask,
                },
            )
        logger.info(
            "[GENTEST_GENERATE] completed %s sample=%s status=%s reward=%s traj=%s timings=%s",
            instance_id,
            sample.index,
            sample.status.value,
            sample.reward,
            traj_path,
            phase_timings,
        )
        return sample
    except Exception as exc:
        record.update(_record_error(instance, sample, f"{type(exc).__name__}: {str(exc)[:300]}", traceback.format_exc()))
        sample.metadata.update({"instance_id": instance_id, "gentest_record": record})
        sample.reward = reward_from_record(record)
        sample.status = Sample.Status.ABORTED
        raise
    finally:
        if model is not None:
            model.close()
        if env is not None:
            try:
                with _record_timing(phase_timings, "workspace_environment_cleanup_sec"):
                    await _run_blocking(env.cleanup)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[GENTEST_GENERATE] workspace cleanup failed for %s: %s", instance_id, exc)
        if verify_env is not None:
            try:
                with _record_timing(phase_timings, "verify_environment_cleanup_sec"):
                    await _run_blocking(verify_env.cleanup)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[GENTEST_GENERATE] verifier cleanup failed for %s: %s", instance_id, exc)
        phase_timings["environment_cleanup_sec"] = (
            phase_timings.get("workspace_environment_cleanup_sec", 0.0)
            + phase_timings.get("verify_environment_cleanup_sec", 0.0)
        )
        phase_timings["total_sec"] = time.time() - start_time
        record["total_time"] = phase_timings["total_sec"]


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    if "test_code" in out:
        out["test_code_chars"] = len(out.get("test_code") or "")
        out.pop("test_code", None)
    validation = out.get("validation")
    if isinstance(validation, dict) and "test_code" in validation:
        validation = dict(validation)
        validation["test_code_chars"] = len(validation.get("test_code") or "")
        validation.pop("test_code", None)
        out["validation"] = validation
    return out


def _has_format_error(sample: Sample) -> bool:
    if not isinstance(sample.metadata, dict):
        return False
    return int(sample.metadata.get("format_error_count") or 0) > 0


def _zero_loss_mask(sample: Sample) -> None:
    if sample.loss_mask is not None:
        sample.loss_mask = [0] * len(sample.loss_mask)
    elif sample.response_length is not None:
        sample.loss_mask = [0] * sample.response_length


def _gentest_rollout_metrics(samples: list[Sample], rollout_time: float) -> dict[str, float | int]:
    samples_with_records = [
        sample
        for sample in samples
        if isinstance(sample.metadata, dict) and isinstance(sample.metadata.get("gentest_record"), dict)
    ]
    records = [sample.metadata["gentest_record"] for sample in samples_with_records]
    statuses: dict[str, int] = {}
    labels: dict[str, int] = {}
    calls: list[int] = []
    no_tool_samples = 0
    format_error_samples = 0
    env_errors = 0
    # Auto-collect every numeric timing key (incl. nested "model.*") instead of a fixed phase list,
    # so keys like total_sec_before_cleanup / workspace_environment_create_sec / model.generation_sec
    # are not silently dropped when the reader's hardcoded names drift from what the rollout writes.
    timings_by_phase: dict[str, list[float]] = {}

    def _num(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    # Discriminative-quality accumulators from the reward layer (sample.metadata["patch_classification_reward"]).
    submit_flags: list[int] = []
    base_fail_flags: list[int] = []
    gold_pass_flags: list[int] = []
    balanced_acc: list[float] = []
    raw_rewards: list[float] = []

    for sample, record in zip(samples_with_records, records):
        status = str(record.get("agent_exit_status") or ("Error" if record.get("error") else "Unknown"))
        statuses[status] = statuses.get(status, 0) + 1
        env_errors += int(
            status == "Error"
            or bool(record.get("retryable_infrastructure_error"))
            or bool(record.get("sandbox_infra_error"))
        )
        validation = record.get("validation") if isinstance(record.get("validation"), dict) else {}
        label = str(validation.get("label") or "NO_LABEL")
        labels[label] = labels.get(label, 0) + 1
        if record.get("agent_calls") is not None:
            calls.append(int(record.get("agent_calls") or 0))
        no_tool_samples += int(int(record.get("no_tool_error_count") or 0) > 0)
        format_error_samples += int(int(record.get("format_error_count") or 0) > 0)
        timings = record.get("timings") if isinstance(record.get("timings"), dict) else {}
        for phase, value in timings.items():
            if _num(value):
                timings_by_phase.setdefault(phase, []).append(float(value))
            elif isinstance(value, dict):  # nested block e.g. "model"
                for sub, subval in value.items():
                    if _num(subval):
                        timings_by_phase.setdefault(f"{phase}.{sub}", []).append(float(subval))

        pcr = sample.metadata.get("patch_classification_reward") if isinstance(sample.metadata, dict) else None
        if isinstance(pcr, dict):
            submit_flags.append(int(str(record.get("agent_exit_status")) == "Submitted"))
            if pcr.get("base_clean_fail") is not None:
                base_fail_flags.append(int(bool(pcr.get("base_clean_fail"))))
            if pcr.get("gold_pass") is not None:
                gold_pass_flags.append(int(bool(pcr.get("gold_pass"))))
            if _num(pcr.get("balanced_accuracy")):
                balanced_acc.append(float(pcr.get("balanced_accuracy")))
            fr = pcr.get("final_reward") if isinstance(pcr.get("final_reward"), dict) else {}
            if _num(fr.get("raw_reward")):
                raw_rewards.append(float(fr.get("raw_reward")))

    metrics: dict[str, float | int] = {
        "gentest_native/elapsed_sec": float(rollout_time),
        "gentest_native/num_samples": len(records),
        "gentest_native/env_error_count": env_errors,
        "gentest_native/env_error_rate": (env_errors / len(records)) if records else 0.0,
        "gentest_native/no_tool_sample_rate": (no_tool_samples / len(records)) if records else 0.0,
        "gentest_native/format_error_sample_rate": (format_error_samples / len(records)) if records else 0.0,
    }
    metrics.update({f"gentest_native/status/{status}": count for status, count in statuses.items()})
    metrics.update({f"gentest_native/label/{label}": count for label, count in labels.items()})
    if calls:
        metrics["gentest_native/agent_calls_mean"] = sum(calls) / len(calls)
        metrics["gentest_native/agent_calls_max"] = max(calls)
    for phase, values in timings_by_phase.items():
        if values:
            metrics[f"gentest_native/timing/{phase}_mean"] = sum(values) / len(values)
            metrics[f"gentest_native/timing/{phase}_max"] = max(values)

    # Discriminative-quality scalars (only over samples that reached the reward layer). A good
    # regression test is base_fail & gold_pass -> discriminative_rate is the headline signal.
    def _rate(flags: list[int]) -> float:
        return (sum(flags) / len(flags)) if flags else 0.0

    if submit_flags:
        metrics["gentest_native/reward_eval_samples"] = len(submit_flags)
        metrics["gentest_native/submit_rate"] = _rate(submit_flags)
    if base_fail_flags:
        metrics["gentest_native/base_fail_rate"] = _rate(base_fail_flags)
    if gold_pass_flags:
        metrics["gentest_native/gold_pass_rate"] = _rate(gold_pass_flags)
    if base_fail_flags and gold_pass_flags:
        disc = [int(b and g) for b, g in zip(base_fail_flags, gold_pass_flags)]
        metrics["gentest_native/discriminative_rate"] = _rate(disc)
    if balanced_acc:
        metrics["gentest_native/balanced_accuracy_mean"] = sum(balanced_acc) / len(balanced_acc)
    if raw_rewards:
        metrics["gentest_native/raw_reward_mean"] = sum(raw_rewards) / len(raw_rewards)

    # Turn-penalty coverage (flags set by post_process_rewards._apply_same_reward_turn_penalty).
    turn_penalty_eligible = sum(
        1 for s in samples if isinstance(s.metadata, dict) and s.metadata.get("turn_penalty_eligible")
    )
    if turn_penalty_eligible:
        turn_penalty_applied = sum(
            1 for s in samples if isinstance(s.metadata, dict) and s.metadata.get("turn_penalty_applied")
        )
        metrics["gentest_native/turn_penalty_eligible"] = turn_penalty_eligible
        metrics["gentest_native/turn_penalty_applied"] = turn_penalty_applied
        metrics["gentest_native/turn_penalty_applied_rate_over_eligible"] = turn_penalty_applied / turn_penalty_eligible
        if samples:
            metrics["gentest_native/turn_penalty_applied_rate_over_all"] = turn_penalty_applied / len(samples)
    return metrics


def append_rollout_metrics(_rollout_id, _args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Add gentest metrics, then let slime's default rollout logger continue."""
    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(_gentest_rollout_metrics(samples, rollout_time))
    return False
