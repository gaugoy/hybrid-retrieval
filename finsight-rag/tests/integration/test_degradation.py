"""P10 fallbacks end to end on the tiny corpus: dense down -> BM25; LLM down -> fallback LLM;
the API caches answers (X-Cache) but never a degraded one."""

from __future__ import annotations

from fastapi.testclient import TestClient

from finsight.api.main import create_app
from finsight.config import Settings
from finsight.container import Container
from finsight.errors import ProviderError
from finsight.generation.llm.fake import FakeLLM
from finsight.generation.llm.fallback import FallbackLLM
from finsight.retrieval.base import Retriever
from finsight.schemas import QueryFilters, RetrievedChunk

Q = "What is Acme's FY2020 capital expenditure?"


class Down(Retriever):
    name = "dense"

    def retrieve(
        self, query: str, k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]:
        raise ConnectionError("qdrant unreachable")


def _down_llm() -> FakeLLM:
    llm = FakeLLM()
    llm.generate = lambda *a, **k: (_ for _ in ()).throw(ProviderError("503"))  # type: ignore[method-assign]
    return llm


def test_dense_failure_falls_back_to_bm25(tiny: Settings) -> None:
    for mode in ("dense", "hybrid"):
        s = tiny.model_copy(update={"retrieval": tiny.retrieval.model_copy(update={"mode": mode})})
        s.root, s.secrets = tiny.root, tiny.secrets
        c = Container(s)
        try:
            c.__dict__["dense_retriever"] = Down()
            ans = c.pipeline.answer(Q)
            assert "dense_failed" in ans.degraded, mode
            assert ans.citations and ans.retrieved[0].chunk.doc_name == "ACME_2020_10K", mode
        finally:
            c.close()


def test_llm_failure_falls_back_and_is_flagged(tiny: Settings) -> None:
    c = Container(tiny)
    try:
        c.__dict__["llm"] = FallbackLLM(_down_llm(), ["local"], lambda n: FakeLLM(model=n))
        ans = c.pipeline.answer(Q)
        assert "llm_fallback" in ans.degraded and ans.model == "local" and ans.citations
    finally:
        c.close()


def test_api_answer_cache_hits_but_skips_degraded(tiny: Settings) -> None:
    c = Container(tiny)
    with TestClient(create_app(container=c)) as client:
        first = client.post("/v1/query", json={"question": Q})
        second = client.post("/v1/query", json={"question": Q})
        assert first.headers["X-Cache"] == "miss" and second.headers["X-Cache"] == "hit"
        assert first.json()["answer"] == second.json()["answer"]
        assert first.json()["request_id"] != second.json()["request_id"]
        other = client.post("/v1/query", json={"question": Q, "pipeline": "naive"})
        assert other.headers["X-Cache"] == "miss"

        c.__dict__["llm"] = FallbackLLM(_down_llm(), ["local"], lambda n: FakeLLM(model=n))
        c.__dict__.pop("pipeline", None)
        q2 = "Beta debt maturing next year?"
        for _ in range(2):
            r = client.post("/v1/query", json={"question": q2})
            assert r.headers["X-Cache"] == "miss" and "llm_fallback" in r.json()["degraded"]
