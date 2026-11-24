"""
Ollama: a local LLM server (https://ollama.com), POST /api/chat.

num_ctx is sent on EVERY request: Ollama's default context window is small and it silently
drops the start of a long prompt (your retrieved context!) instead of raising an error.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from finsight.config import LLMCfg
from finsight.errors import ProviderError
from finsight.generation.llm.base import LLM, LLMResponse


class OllamaLLM(LLM):
    provider = "ollama"

    def __init__(self, cfg: LLMCfg, client: httpx.Client | None = None) -> None:
        self.cfg, self.model = cfg, cfg.model
        self.base = (cfg.base_url or cfg.ollama_base_url).rstrip("/")
        self.client = client or httpx.Client(timeout=cfg.timeout_s)

    def _body(self, system: str, messages: list[dict], max_tokens: int, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": stream,
            "options": {"temperature": 0, "num_ctx": self.cfg.num_ctx, "num_predict": max_tokens},
        }

    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        body = self._body(system, messages, max_tokens, stream=False)
        if json_mode:
            body["format"] = "json"
        try:
            r = self.client.post(f"{self.base}/api/chat", json=body)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"ollama at {self.base}: {exc} (is `ollama serve` running?)"
            ) from exc
        data = r.json()
        return LLMResponse(
            text=data["message"]["content"],
            model=self.model,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            stop_reason=data.get("done_reason"),
        )

    def stream(self, system: str, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        body = self._body(system, messages, max_tokens, stream=True)
        with self.client.stream("POST", f"{self.base}/api/chat", json=body) as r:
            for line in r.iter_lines():
                if line:
                    delta = json.loads(line).get("message", {}).get("content", "")
                    if delta:
                        yield delta


def ollama_ready(base_url: str, model: str, timeout: float = 1.0) -> bool:
    """True if the server answers and the model is pulled (used by provider "auto")."""
    try:
        tags = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout).json()
    except (httpx.HTTPError, ValueError):
        return False
    names = {m.get("name", "") for m in tags.get("models", [])}
    return model in names or f"{model}:latest" in names
