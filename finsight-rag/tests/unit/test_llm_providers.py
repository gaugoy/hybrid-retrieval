"""LLM providers: request shapes (mock transports / fake SDK client), auto selection, guards."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from finsight.config import LLMCfg, Secrets, load_settings
from finsight.errors import ConfigError, MissingAPIKeyError, PaidProviderBlockedError
from finsight.generation.llm import factory
from finsight.generation.llm.anthropic_llm import AnthropicLLM
from finsight.generation.llm.ollama import OllamaLLM
from finsight.generation.llm.remote_llm import GeminiLLM, OpenAIChatLLM


def mock(reply: dict[str, Any]) -> tuple[httpx.Client, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, json=reply)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def test_ollama_sends_num_ctx_and_reads_usage() -> None:
    client, seen = mock(
        {
            "message": {"content": "hi [1]"},
            "prompt_eval_count": 50,
            "eval_count": 3,
            "done_reason": "stop",
        }
    )
    llm = OllamaLLM(LLMCfg(provider="ollama", model="qwen2.5:7b-instruct", num_ctx=8192), client)
    out = llm.generate("sys", [{"role": "user", "content": "q"}], 100, json_mode=True)
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://localhost:11434/api/chat"
    assert body["options"]["num_ctx"] == 8192 and body["format"] == "json"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert (out.text, out.input_tokens, out.output_tokens) == ("hi [1]", 50, 3)


def test_openai_style_and_gemini_shapes() -> None:
    client, seen = mock(
        {
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 1},
            "model": "m",
        }
    )
    llm = OpenAIChatLLM(
        LLMCfg(provider="groq", model="m"), "k", client=client, sleep=lambda s: None
    )
    assert llm.generate("s", [{"role": "user", "content": "q"}], 10).input_tokens == 9
    assert str(seen[0].url) == "https://api.groq.com/openai/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer k"

    client, seen = mock(
        {
            "candidates": [{"content": {"parts": [{"text": "g"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
        }
    )
    gem = GeminiLLM(
        LLMCfg(provider="gemini", model="gemini-flash-latest"),
        "gk",
        client=client,
        sleep=lambda s: None,
    )
    out = gem.generate("sys", [{"role": "user", "content": "q"}], 10, json_mode=True)
    body = json.loads(seen[0].content)
    assert str(seen[0].url).endswith("/models/gemini-flash-latest:generateContent")
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert out.text == "g" and seen[0].headers["x-goog-api-key"] == "gk"


class FakeMessages:
    def __init__(self, stop: str = "end_turn") -> None:
        self.kwargs: dict[str, Any] = {}
        self.stop = stop

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            stop_reason=self.stop,
            model=kwargs["model"],
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="ok [1]")],
            usage=SimpleNamespace(input_tokens=11, output_tokens=2),
        )


def test_anthropic_params_effort_fallbacks_no_sampling() -> None:
    beta, plain = FakeMessages(), FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=beta), messages=plain)
    opus = AnthropicLLM(
        LLMCfg(provider="anthropic", model="claude-opus-5-5", effort="medium"), "k", client=client
    )
    assert opus.generate("sys", [{"role": "user", "content": "q"}], 1024).text == "ok [1]"
    kw = beta.kwargs
    assert kw["output_config"] == {"effort": "medium"} and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["max_tokens"] >= 16000
    assert not {"temperature", "top_p", "top_k"} & set(kw)

    haiku = AnthropicLLM(
        LLMCfg(provider="anthropic", model="claude-haiku-5-5", effort="low"), "k", client=client
    )
    haiku.generate("sys", [{"role": "user", "content": "q"}])
    assert "fallbacks" not in plain.kwargs and plain.kwargs["output_config"] == {"effort": "low"}

    refusing = SimpleNamespace(
        beta=SimpleNamespace(messages=FakeMessages("refusal")), messages=plain
    )
    out = AnthropicLLM(
        LLMCfg(provider="anthropic", model="claude-opus-5-5"), "k", client=refusing
    ).generate("s", [{"role": "user", "content": "q"}])
    assert out.text == "" and out.stop_reason == "refusal"


def test_auto_picks_first_runnable(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("finsight.generation.llm.ollama.ollama_ready", lambda *a, **k: False)
    s = load_settings(root=project_root, env={})
    assert s.llm.provider == "auto"
    none = factory.resolve_auto(s.llm, project_root, Secrets())
    assert none.provider == "hf"  # nothing else available: the local model
    gem = factory.resolve_auto(s.llm, project_root, Secrets.from_env({"GEMINI_API_KEY": "g"}))
    assert gem.provider == "gemini"
    claude = factory.resolve_auto(
        s.llm, project_root, Secrets.from_env({"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a"})
    )
    assert claude.model == "claude-opus-5-5"
    with pytest.raises(ConfigError, match="found nothing"):
        factory.resolve_auto(
            s.llm.model_copy(update={"auto_order": ["gemini-flash"]}), project_root, Secrets()
        )


def test_guards(project_root: Path) -> None:
    with pytest.raises(PaidProviderBlockedError):
        factory.build_llm(
            LLMCfg(provider="openai", model="gpt-4.1-mini"),
            root=project_root,
            secrets=Secrets.from_env({"OPENAI_API_KEY": "o"}),
        )
    with pytest.raises(MissingAPIKeyError, match="MISTRAL_API_KEY"):
        factory.build_llm(
            LLMCfg(provider="mistral", model="mistral-small-latest"),
            root=project_root,
            secrets=Secrets(),
        )
    s = load_settings(root=project_root, env={"LLM": "claude-haiku"})
    assert (s.llm.provider, s.llm.model, s.llm.effort) == ("anthropic", "claude-haiku-5-5", "low")
