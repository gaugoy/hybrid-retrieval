"""FakeLLM for tests: deterministic, no model. Cites block [1] when context exists, else abstains.

`replies` (optional) is a queue of canned texts returned in order, for testing parsers and loops.
"""

from __future__ import annotations

import re

from finsight.generation.llm.base import LLM, LLMResponse

ABSTAIN = "Not found in the provided filings."


class FakeLLM(LLM):
    provider = "fake"

    def __init__(self, replies: list[str] | None = None, model: str = "fake") -> None:
        self.model = model
        self.replies = list(replies or [])
        self.calls: list[dict] = []

    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "json_mode": json_mode})
        if self.replies:
            text = self.replies.pop(0)
        else:
            user = messages[-1]["content"]
            m = re.search(r"\[1\][^\n]*\n(.+)", user)
            text = f"The answer is in the filing: {m.group(1)[:60].strip()} [1]." if m else ABSTAIN
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=len(str(messages)) // 4,
            output_tokens=len(text) // 4,
            stop_reason="end_turn",
        )
