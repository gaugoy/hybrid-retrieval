"""The LLM interface (contract §4.2) and its response type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    fallback_from: str | None = None  # P10: set when a fallback LLM answered instead


class LLM(ABC):
    provider: str
    model: str

    @abstractmethod
    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse: ...

    def stream(self, system: str, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        """Token deltas. Default: one chunk with the whole answer (providers override)."""
        yield self.generate(system, messages, max_tokens).text
