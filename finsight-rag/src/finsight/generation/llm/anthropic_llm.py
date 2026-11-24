"""
Claude via the official Anthropic Python SDK.

- No temperature / top_p / top_k: current Claude models reject sampling parameters (400).
- Depth is controlled with output_config={"effort": ...} (low for judging, medium for answers).
- Claude Opus 5.5 thinks adaptively (always on); max_tokens leaves room for it.
- Opus requests opt into server-side refusal fallbacks (beta "server-side-fallback-2026-07-01",
  fallbacks="default"): if a safety classifier declines, the API re-runs the request on a
  fallback model inside the same call. A final stop_reason "refusal" means the chain declined;
  the pipeline then treats the answer as an abstention.
- The SDK retries 408/409/429/5xx itself (max_retries); typed errors become ProviderError.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anthropic

from finsight.config import LLMCfg
from finsight.errors import ProviderError
from finsight.generation.llm.base import LLM, LLMResponse

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept server-side fallbacks (Haiku has none).
FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-sonnet-5-5",
}
MIN_MAX_TOKENS = 16000  # thinking + answer; don't truncate mid-thought


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, cfg: LLMCfg, api_key: str | None, client: Any = None) -> None:
        self.cfg, self.model = cfg, cfg.model
        self.client = client or anthropic.Anthropic(
            api_key=api_key, timeout=cfg.timeout_s, max_retries=3
        )

    def _params(self, system: str, messages: list[dict], max_tokens: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(max_tokens, MIN_MAX_TOKENS),
            "system": system,
            "messages": messages,
            "output_config": {"effort": self.cfg.effort},
        }
        if self.model in FALLBACK_MODELS:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _api(self) -> Any:
        return self.client.beta.messages if self.model in FALLBACK_MODELS else self.client.messages

    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        if json_mode:
            system += "\n\nRespond with a single JSON object and nothing else."
        try:
            resp = self._api().create(**self._params(system, messages, max_tokens))
        except anthropic.RateLimitError as exc:
            raise ProviderError(f"anthropic rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"anthropic HTTP {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"anthropic connection error: {exc}") from exc
        text = (
            ""
            if resp.stop_reason == "refusal"
            else "".join(b.text for b in resp.content if b.type == "text")
        )
        return LLMResponse(
            text=text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason,
        )

    def stream(self, system: str, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        try:
            with self._api().stream(**self._params(system, messages, max_tokens)) as s:
                yield from s.text_stream
        except anthropic.APIError as exc:
            raise ProviderError(f"anthropic stream failed: {exc}") from exc
