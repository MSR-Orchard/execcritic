"""mini-swe-agent model class for the GitHub Copilot API (Bonete port).

Differs from the gcr323 origin (probes/copilot_model.py) only in:
  - No gcr323 sys.path hack; this file lives INSIDE Baolin's fork, so the
    minisweagent imports resolve via the fork's own package layout.
  - Token loading: tries env $COPILOT_TOKEN, then the explicitly configured
    $COPILOT_TOKEN_FILE, then shell-out to `gh auth token` as a last resort.

Endpoint: https://api.githubcopilot.com/chat/completions
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModelConfig
from minisweagent.models.utils.actions_text import format_observation_messages, parse_regex_actions

from openai import OpenAI


COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_HEADERS = {
    "Copilot-Integration-Id": "copilot-cli",
    "X-GitHub-Api-Version": "2026-01-09",
    "Editor-Version": "copilot-cli/1.0.0",
}
def _gh_token() -> str:
    """Resolve a Copilot bearer token.

    Order of preference:
      1) $COPILOT_TOKEN (raw token in env)
      2) $COPILOT_TOKEN_FILE — file containing the token
      3) shell-out to `gh auth token` if a gh binary is on PATH or at $GH_BIN
    """
    tok = os.environ.get("COPILOT_TOKEN")
    if tok:
        return tok.strip()
    fp = os.environ.get("COPILOT_TOKEN_FILE")
    if fp and pathlib.Path(fp).is_file():
        return pathlib.Path(fp).read_text().strip()
    gh = os.environ.get("GH_BIN") or shutil.which("gh")
    if gh and pathlib.Path(gh).is_file():
        return subprocess.check_output([gh, "auth", "token"]).decode().strip()
    raise FileNotFoundError(
        "No Copilot token source found. Set $COPILOT_TOKEN or $COPILOT_TOKEN_FILE, "
        "or install and authenticate gh."
    )


class CopilotChatModelConfig(LitellmTextbasedModelConfig):
    base_url: str = COPILOT_BASE_URL


class CopilotChatModel(LitellmModel):
    """mini-swe-agent compatible model class for Copilot /chat/completions.

    Cost tracking short-circuited (Copilot Enterprise corporate quota, not $).
    """

    def __init__(self, **kwargs):
        mk = dict(kwargs.get("model_kwargs", {}) or {})
        mk.pop("drop_params", None)
        if "max_completion_tokens" not in mk and "max_tokens" not in mk:
            mk["max_completion_tokens"] = 32768
        kwargs["model_kwargs"] = mk
        super().__init__(config_class=CopilotChatModelConfig, **kwargs)
        self._build_client()

    def _build_client(self):
        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=_gh_token(),
            default_headers=COPILOT_HEADERS,
        )

    def _query(self, messages, **kwargs):
        call_kwargs = {**self.config.model_kwargs, **kwargs}
        for k in ("drop_params", "api_base", "api_key", "base_url",
                  "azure_endpoint", "api_version", "host"):
            call_kwargs.pop(k, None)
        if "max_tokens" in call_kwargs and "max_completion_tokens" not in call_kwargs:
            call_kwargs["max_completion_tokens"] = call_kwargs.pop("max_tokens")

        model = self.config.model_name
        for prefix in ("openai/", "trapi/", "hosted_vllm/", "azure/", "copilot/"):
            if model.startswith(prefix):
                model = model[len(prefix):]

        self._build_client()
        return self._client.chat.completions.create(
            model=model, messages=messages, **call_kwargs,
        )

    def _parse_actions(self, response):
        content = response.choices[0].message.content or ""
        return parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
        )

    def format_observation_messages(self, message, outputs, template_vars=None):
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def _calculate_cost(self, response):
        usage = getattr(response, "usage", None)
        in_tok = (getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) if usage else 0) or 0
        out_tok = (getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) if usage else 0) or 0
        return {
            "cost": 0.0,
            "n_input_tokens": in_tok,
            "n_output_tokens": out_tok,
        }


if __name__ == "__main__":
    m = CopilotChatModel(model_name="claude-opus-4.8")
    r = m._query(messages=[{"role": "user", "content": "say OK and nothing else"}])
    print("OK_smoke[chat]:", repr(r.choices[0].message.content))
    print("usage:", r.usage)
