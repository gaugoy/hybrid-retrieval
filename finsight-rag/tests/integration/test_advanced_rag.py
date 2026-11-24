"""P7 techniques on the tiny corpus with scripted fake LLMs: transforms, corrective, agentic."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from finsight.config import Settings
from finsight.container import Container
from finsight.generation.llm.base import LLMResponse
from finsight.generation.llm.fake import FakeLLM
from finsight.retrieval.query_transform import transformed_retrieve

Q = "What is Acme's FY2020 capital expenditure?"


def container(tiny: Settings, **update: Any) -> Container:
    s = tiny.model_copy(update=update) if update else tiny
    s.root, s.secrets, s.paths = tiny.root, tiny.secrets, tiny.paths
    return Container(s)


@pytest.mark.parametrize("technique", ["multi_query", "decompose"])
def test_query_list_transforms(tiny: Settings, technique: str) -> None:
    c = container(tiny)
    try:
        llm = FakeLLM(
            ['{"queries": ["Acme purchases of property plant and equipment 2020", "Acme capex"]}']
        )
        res, trace = transformed_retrieve(c.retrieval, llm, Q, technique)
        assert trace.queries[0].startswith("Acme purchases") and trace.llm_calls == 1
        assert res.final and "fused" in res.stages and res.degraded == []
        assert ("ACME_2020_10K", 59) in [
            (r.chunk.doc_name, r.chunk.page_start) for r in res.final[:3]
        ]

        bad, t2 = transformed_retrieve(
            c.retrieval, FakeLLM(["no json", "still none"]), Q, technique
        )
        assert f"{technique}_parse_failed" in bad.degraded and t2.llm_calls == 2 and bad.final
    finally:
        c.close()


def test_hyde_embeds_the_passage(tiny: Settings) -> None:
    c = container(tiny)
    try:
        llm = FakeLLM(
            ["Acme's purchases of property, plant and equipment were 1,577 million in FY2020."]
        )
        res, trace = transformed_retrieve(c.retrieval, llm, Q, "hyde")
        assert trace.hyde_passage.startswith("Acme's purchases") and res.final
    finally:
        c.close()


def test_corrective_pass_rewrite_and_abstain(tiny: Settings) -> None:
    c = container(tiny, pipeline="corrective")
    try:
        c.settings.crag.min_score = -1e9  # always sufficient
        ok = c.pipeline.answer(Q)
        assert c.pipeline.branch == "pass" and not ok.abstained
        c.settings.crag.min_score = 1e9  # never sufficient: rewrite, then abstain
        c.pipeline.llm = FakeLLM(['{"query": "Acme purchases of property plant and equipment"}'])
        ans = c.pipeline.answer(Q)
        assert c.pipeline.branch == "abstain" and ans.abstained and "crag_abstained" in ans.warnings
        assert len(c.pipeline.llm.calls) == 1  # only the rewrite: the generator was never called
    finally:
        c.close()


class ScriptedAgent(FakeLLM):
    """search -> read the first id it saw -> answer citing it."""

    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        self.calls.append({"messages": messages})
        hist = messages[-1]["content"]
        ids = re.findall(r"(ACME_2020_10K::p\d+-\d+::c\d+)", hist)
        if "read(" in hist:
            act = {
                "tool": "answer",
                "args": {"text": "Capex was 1,577 million.", "citations": ids[:1]},
            }
        elif ids:
            act = {"tool": "read", "args": {"chunk_id": ids[0]}}
        else:
            act = {
                "tool": "search",
                "args": {"query": "Acme capital expenditure", "company": "Acme"},
            }
        return LLMResponse(text=json.dumps(act), model="scripted")


def test_agentic_json_protocol_and_fallback(tiny: Settings) -> None:
    c = container(tiny, pipeline="agentic")
    try:
        c.pipeline.llm = ScriptedAgent()
        ans = c.pipeline.answer(Q)
        assert [t["tool"] for t in c.pipeline.trace] == ["search", "read", "answer"]
        assert ans.citations and ans.citations[0].doc_name == "ACME_2020_10K" and not ans.abstained
        assert ans.usage["llm_calls"] == 3

        c.pipeline.llm = FakeLLM(["not json", "nope", "The answer is in the filing [1]."])
        fell = c.pipeline.answer(Q)
        assert "agent_fell_back" in fell.degraded and fell.pipeline == "agentic"
    finally:
        c.close()


class FakeClaude:
    """Mimics client.beta.messages.create for a tool-use loop: search, then answer."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if len(self.calls) == 1:
            block = SimpleNamespace(
                type="tool_use",
                id="t1",
                name="search",
                input={"query": "Acme capex", "company": "Acme"},
            )
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        results = kw["messages"][-1]["content"]
        first_id = results[0]["content"].split(" | ")[0]
        if len(self.calls) == 2:
            block = SimpleNamespace(
                type="tool_use", id="t2", name="read", input={"chunk_id": first_id}
            )
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        block = SimpleNamespace(
            type="tool_use",
            id="t3",
            name="answer",
            input={"text": "1,577 million", "citations": [self.read_id(kw)]},
        )
        return SimpleNamespace(stop_reason="tool_use", content=[block])

    @staticmethod
    def read_id(kw: dict) -> str:
        for m in kw["messages"]:
            for b in m["content"] if isinstance(m["content"], list) else []:
                if getattr(b, "name", None) == "read":
                    return str(b.input["chunk_id"])
        return ""


def test_agentic_native_claude_loop(tiny: Settings) -> None:
    from finsight.config import LLMCfg
    from finsight.generation.llm.anthropic_llm import AnthropicLLM

    c = container(tiny, pipeline="agentic")
    try:
        fake = FakeClaude()
        client = SimpleNamespace(beta=SimpleNamespace(messages=fake), messages=fake)
        c.pipeline.llm = AnthropicLLM(
            LLMCfg(provider="anthropic", model="claude-opus-5-5"), "k", client=client
        )
        ans = c.pipeline.answer(Q)
        assert [t["tool"] for t in c.pipeline.trace] == ["search", "read", "answer"]
        assert ans.citations and fake.calls[0]["tool_choice"] == {"type": "auto"}
        assert {t["name"] for t in fake.calls[0]["tools"]} == {"search", "read", "answer"}
    finally:
        c.close()
