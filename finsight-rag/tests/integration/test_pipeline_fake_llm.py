"""Both pipelines end to end on the tiny corpus with FakeLLM: answer, citations, abstention."""

from __future__ import annotations

import pytest

from finsight.config import Settings
from finsight.container import Container
from finsight.generation.llm.fake import FakeLLM


@pytest.mark.parametrize("pipeline", ["hybrid_rerank", "naive"])
def test_cited_answer(tiny: Settings, pipeline: str) -> None:
    c = Container(tiny.model_copy(update={"pipeline": pipeline}))
    c.settings.root, c.settings.secrets = tiny.root, tiny.secrets
    try:
        ans = c.pipeline.answer("What is Acme's FY2020 capital expenditure?")
        assert ans.pipeline == pipeline and ans.provider == "fake"
        assert ans.citations and ans.citations[0].marker == 1 and not ans.abstained
        assert {"generate", "total"} <= set(ans.timings_ms) and ans.request_id
        assert ans.usage["input_tokens"] > 0 and ans.warnings == []
    finally:
        c.close()


def test_hallucinated_citation_and_abstention_are_flagged(tiny: Settings) -> None:
    c = Container(tiny)
    try:
        c.__dict__["llm"] = FakeLLM(
            ["The capex was large [9].", "Not found in the provided filings."]
        )
        bad = c.pipeline.answer("Acme capex FY2020?")
        assert bad.citations == [] and set(bad.warnings) == {
            "hallucinated_citation",
            "uncited_answer",
        }
        off = c.pipeline.answer("Who won the 2018 World Cup?")
        assert off.abstained and off.citations == []
    finally:
        c.close()
