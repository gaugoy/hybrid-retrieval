"""
Online LLMs over plain HTTP, sharing the embedders' rate limiter and retry policy.

  openai-style  POST {base}/chat/completions      Mistral, Groq (free tiers), OpenAI (paid)
  gemini        POST {base}/models/{m}:generateContent   (key in x-goog-api-key)
Responses are cached? No: answers change with the context; the P8 judge caches its own calls.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from finsight.config import LLMCfg
from finsight.generation.llm.base import LLM, LLMResponse
from finsight.providers.http import post_json
from finsight.providers.ratelimit import RateLimiter, UsageStore, estimate_tokens

BASE_URLS = {
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}
KEY_URLS = {
    "mistral": "https://console.mistral.ai/",
    "groq": "https://console.groq.com/keys",
    "openai": "https://platform.openai.com/api-keys",
    "gemini": "https://aistudio.google.com/apikey",
}
PAID_ONLY = {"openai"}


class _RemoteLLM(LLM):
    def __init__(
        self,
        cfg: LLMCfg,
        api_key: str,
        *,
        usage_dir: Path | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg, self.model, self.provider = cfg, cfg.model, cfg.provider
        self.api_key, self._sleep = api_key, sleep
        self.base = (cfg.base_url or BASE_URLS[cfg.provider]).rstrip("/")
        self.client = client or httpx.Client(timeout=cfg.timeout_s)
        rl = cfg.rate_limit
        self.limiter = RateLimiter(
            rpm=rl.rpm,
            tpm=rl.tpm,
            rpd=rl.rpd,
            rpmonth=rl.rpmonth,
            name=cfg.provider,
            usage=UsageStore(usage_dir / f"llm-{cfg.provider}.json" if usage_dir else None),
            sleep=sleep,
        )

    def _post(
        self, url: str, body: dict[str, Any], headers: dict[str, str], est_tokens: int
    ) -> dict[str, Any]:
        self.limiter.acquire(est_tokens)
        return post_json(
            self.client,
            url,
            body=body,
            headers=headers,
            provider=self.provider,
            key_hint=f"{self.provider.upper()}_API_KEY",
            max_retries=self.cfg.rate_limit.max_retries,
            sleep=self._sleep,
        )


class OpenAIChatLLM(_RemoteLLM):
    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        est = estimate_tokens(system) + sum(estimate_tokens(m["content"]) for m in messages)
        data = self._post(
            f"{self.base}/chat/completions",
            body,
            {"Authorization": f"Bearer {self.api_key}"},
            est + max_tokens,
        )
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=choice["message"].get("content") or "",
            model=data.get("model", self.model),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            stop_reason=choice.get("finish_reason"),
        )


class GeminiLLM(_RemoteLLM):
    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
        ]
        gen: dict[str, Any] = {"temperature": 0, "maxOutputTokens": max_tokens}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": gen,
        }
        est = estimate_tokens(system) + sum(estimate_tokens(m["content"]) for m in messages)
        data = self._post(
            f"{self.base}/models/{self.model}:generateContent",
            body,
            {"x-goog-api-key": self.api_key},
            est + max_tokens,
        )
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text="".join(p.get("text", "") for p in parts),
            model=data.get("modelVersion", self.model),
            input_tokens=int(usage.get("promptTokenCount") or 0),
            output_tokens=int(usage.get("candidatesTokenCount") or 0),
            stop_reason=cand.get("finishReason"),
        )
