"""TRAPI/Azure OpenAI Responses API model for mini-swe-agent.

This is intentionally small: it keeps mini-swe-agent's Responses API tool-call
message format, but obtains auth through a static token, Azure CLI, or managed
identity for an explicitly configured endpoint.
"""

from __future__ import annotations

import os
import random
import re
import time
from collections.abc import Callable
from typing import Any

from azure.identity import AzureCliCredential, ChainedTokenCredential, ManagedIdentityCredential, get_bearer_token_provider
from openai import APIConnectionError, APITimeoutError, AzureOpenAI, OpenAI

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModelConfig
from minisweagent.models.utils.actions_toolcall_response import (
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
    tools_for_names_response_api,
)


class TrapiResponseModelConfig(LitellmModelConfig):
    endpoint: str = os.getenv("TRAPI_ENDPOINT", "")
    openai_compatible: bool = os.getenv("TRAPI_OPENAI_COMPAT", "").lower() in {"1", "true", "yes"}
    api_version: str = os.getenv("TRAPI_API_VERSION", "2025-04-01-preview")
    scope: str = os.getenv("TRAPI_SCOPE", "")
    retry_attempts: int = int(os.getenv("TRAPI_RETRY_ATTEMPTS", "6"))
    retry_min_seconds: float = float(os.getenv("TRAPI_RETRY_MIN_SECONDS", "1"))
    retry_max_seconds: float = float(os.getenv("TRAPI_RETRY_MAX_SECONDS", "30"))
    retry_jitter_seconds: float = float(os.getenv("TRAPI_RETRY_JITTER_SECONDS", "1"))


class TrapiResponseModel:
    def __init__(self, *, config_class: Callable = TrapiResponseModelConfig, **kwargs: Any):
        self.config = config_class(**kwargs)
        endpoint = self.config.endpoint.strip().rstrip("/")
        if not endpoint:
            raise ValueError("TRAPI_ENDPOINT or model config endpoint must be set explicitly")
        static_token = self._static_bearer_token()
        provider = self._file_bearer_token_provider()
        if not static_token and provider is None:
            scope = self.config.scope.strip()
            if not scope:
                raise ValueError(
                    "TRAPI_SCOPE or model config scope must be set when Azure identity auth is used"
                )
            provider = get_bearer_token_provider(
                ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
                scope,
            )
        if self.config.openai_compatible or endpoint.endswith("/openai/v1") or endpoint.endswith("/v1"):
            self._client = OpenAI(
                base_url=endpoint,
                api_key=static_token or provider,
            )
        else:
            client_kwargs = {
                "azure_endpoint": self.config.endpoint,
                "api_version": self.config.api_version,
            }
            if static_token:
                client_kwargs["azure_ad_token"] = static_token
            else:
                client_kwargs["azure_ad_token_provider"] = provider
            self._client = AzureOpenAI(**client_kwargs)

    def _static_bearer_token(self) -> str:
        return os.getenv("TRAPI_BEARER_TOKEN", "").strip()

    def _file_bearer_token_provider(self) -> Callable[[], str] | None:
        token_file = os.getenv("TRAPI_BEARER_TOKEN_FILE", "").strip()
        if not token_file:
            return None

        def provider() -> str:
            with open(token_file, encoding="utf-8") as handle:
                return handle.read().strip()

        return provider

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """Flatten stored Response objects into stateless Responses API input."""
        result: list[dict] = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    clean = self._sanitize_replay_item(item)
                    if clean is not None:
                        result.append(clean)
            else:
                clean = self._sanitize_replay_item(msg)
                if clean is not None:
                    result.append(clean)
        return result

    def _sanitize_replay_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Return the subset of an item accepted as Responses API input.

        Some OpenAI-compatible endpoints reject output-only Response fields such
        as status/id/summary when prior output items are replayed as stateless
        input. Keep only the fields needed to preserve the tool loop.
        """
        item_type = item.get("type")
        if item_type == "function_call":
            clean = {
                "type": "function_call",
                "call_id": item.get("call_id") or item.get("id"),
                "name": item.get("name"),
                "arguments": item.get("arguments", "{}"),
            }
            if not clean["call_id"] or not clean["name"]:
                return None
            return clean
        if item_type == "function_call_output":
            return {
                "type": "function_call_output",
                "call_id": item.get("call_id"),
                "output": item.get("output", ""),
            }
        if item_type == "message":
            role = item.get("role", "assistant")
            return {
                "type": "message",
                "role": role,
                "content": self._sanitize_content(item.get("content", "")),
            }
        if item_type == "reasoning":
            return None
        if "role" in item:
            return {
                "role": item.get("role"),
                "content": self._sanitize_content(item.get("content", "")),
            }
        return None

    def _sanitize_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content
        clean_blocks = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in {"input_text", "output_text"}:
                clean_blocks.append({"type": block_type, "text": block.get("text", "")})
            elif block_type == "input_image":
                clean = {"type": "input_image"}
                for key in ("image_url", "file_id", "detail"):
                    if key in block:
                        clean[key] = block[key]
                clean_blocks.append(clean)
            elif block_type == "input_file":
                clean = {"type": "input_file"}
                for key in ("file_id", "filename", "file_data"):
                    if key in block:
                        clean[key] = block[key]
                clean_blocks.append(clean)
        return clean_blocks

    def _request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self.config.model_kwargs)
        merged.update(kwargs)
        # These are LiteLLM/OpenAI-chat style knobs that may be present when this
        # model is dropped into existing hosted_vllm configs.
        for key in ("api_base", "api_key", "base_url", "drop_params", "parallel_tool_calls"):
            merged.pop(key, None)
        max_tokens = merged.pop("max_tokens", None)
        if max_tokens is not None and "max_output_tokens" not in merged:
            merged["max_output_tokens"] = max_tokens
        # GPT-5.x TRAPI deployments do not expose sampling controls.
        merged.pop("temperature", None)
        merged.pop("top_p", None)
        return merged

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        response = self._responses_create_with_retry(messages, kwargs)
        cost_output = {"cost": 0.0}
        GLOBAL_MODEL_STATS.add(0.0)
        message = response.model_dump()
        message["extra"] = {
            "actions": parse_toolcall_actions_response(
                getattr(response, "output", []),
                format_error_template=self.config.format_error_template,
                extra_tools=self.config.extra_tools,
            ),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _responses_create_with_retry(self, messages: list[dict], kwargs: dict[str, Any]):
        last_exc: Exception | None = None
        attempts = max(1, int(self.config.retry_attempts))
        for attempt in range(attempts):
            try:
                return self._client.responses.create(
                    model=self.config.model_name,
                    input=self._prepare_messages_for_api(messages),
                    tools=tools_for_names_response_api(self.config.extra_tools),
                    **self._request_kwargs(kwargs),
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= attempts - 1 or not self._is_retryable(exc):
                    raise
                time.sleep(self._retry_sleep_seconds(exc, attempt))
        raise last_exc  # type: ignore[misc]

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        text = str(exc).lower()
        return "temporarily unavailable" in text or "retry after" in text or "timeout" in text

    def _retry_sleep_seconds(self, exc: Exception, attempt: int) -> float:
        retry_after = self._retry_after_seconds(exc)
        if retry_after is not None:
            base = retry_after
        else:
            base = min(
                float(self.config.retry_max_seconds),
                float(self.config.retry_min_seconds) * (2**attempt),
            )
        jitter = random.uniform(0, max(0.0, float(self.config.retry_jitter_seconds)))
        return min(float(self.config.retry_max_seconds), base + jitter)

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        if headers:
            value = headers.get("retry-after") or headers.get("Retry-After")
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        match = re.search(r"retry after\s+(\d+(?:\.\d+)?)\s+seconds?", str(exc), re.I)
        if match:
            return float(match.group(1))
        return None

    def format_message(self, **kwargs: Any) -> dict:
        return kwargs

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
