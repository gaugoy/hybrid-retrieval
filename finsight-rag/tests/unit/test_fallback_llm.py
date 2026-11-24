"""FallbackLLM: provider failures move to the next preset; config errors don't."""

from __future__ import annotations

import pytest

from finsight.errors import MissingAPIKeyError, ProviderError
from finsight.generation.llm.base import LLM, LLMResponse
from finsight.generation.llm.fake import FakeLLM
from finsight.generation.llm.fallback import FallbackLLM

MSG = [{"role": "user", "content": "hi"}]


class Broken(LLM):
    provider, model = "broken", "b1"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        raise self.exc


def test_provider_error_falls_back_and_is_flagged() -> None:
    llm = FallbackLLM(Broken(ProviderError("503")), ["local"], lambda n: FakeLLM(["ok"], model=n))
    resp = llm.generate("s", MSG)
    assert resp.text == "ok" and resp.model == "local" and resp.fallback_from == "broken:b1"


def test_primary_success_is_not_flagged_and_fallback_not_built() -> None:
    built: list[str] = []
    llm = FallbackLLM(FakeLLM(["fine"]), ["local"], lambda n: built.append(n) or FakeLLM())  # type: ignore[func-returns-value]
    assert llm.generate("s", MSG).fallback_from is None and built == []


def test_config_error_is_raised_not_retried() -> None:
    llm = FallbackLLM(Broken(MissingAPIKeyError("no key")), ["local"], lambda n: FakeLLM())
    with pytest.raises(MissingAPIKeyError):
        llm.generate("s", MSG)


def test_unbuildable_fallback_is_skipped_and_all_failing_raises() -> None:
    def build(name: str) -> LLM:
        if name == "missing":
            raise RuntimeError("cannot build")
        return Broken(TimeoutError("slow"))

    llm = FallbackLLM(Broken(ProviderError("down")), ["missing", "slow"], build)
    with pytest.raises(ProviderError, match="every LLM failed"):
        llm.generate("s", MSG)


def test_stream_falls_back_before_first_token() -> None:
    llm = FallbackLLM(Broken(ProviderError("down")), ["local"], lambda n: FakeLLM(["streamed"]))
    assert "".join(llm.stream("s", MSG)) == "streamed"
