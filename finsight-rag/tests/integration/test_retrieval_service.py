"""RetrievalService and the retrieval eval runner on the tiny indexed corpus (offline)."""

from __future__ import annotations

from finsight.config import Settings
from finsight.container import Container
from finsight.errors import ProviderError
from finsight.evaluation.runner import run_retrieval_eval
from finsight.retrieval.base import Reranker
from finsight.retrieval.service import RetrievalService
from finsight.schemas import QueryFilters, RetrievedChunk


def test_hybrid_with_company_filter_finds_the_gold_page(tiny: Settings) -> None:
    c = Container(tiny)
    try:
        res = c.retrieval.retrieve("What is Acme's FY2020 capital expenditure?")
        assert res.extracted.companies == ["Acme"] and res.applied == QueryFilters(
            companies=["Acme"]
        )
        assert {"dense", "sparse", "hybrid"} <= set(res.stages)
        assert {r.chunk.company for r in res.final} == {"Acme"}
        gold = ("ACME_2020_10K", 59)
        sparse_top = res.stages["sparse"][0].chunk
        assert (sparse_top.doc_name, sparse_top.page_start) == gold  # exact words: BM25 wins
        assert gold in [
            (r.chunk.doc_name, r.chunk.page_start) for r in res.final[:3]
        ]  # RRF keeps it
        assert res.final[0].stage == "final" and "rrf" in res.final[0].scores_by_stage
        assert set(res.timings_ms) >= {"filter", "dense", "sparse", "fuse"}

        beta = c.retrieval.retrieve("Beta debt maturing next year", mode="sparse")
        assert set(beta.stages) == {"sparse"} and beta.final[0].chunk.doc_name == "BETA_2019_10Q"
    finally:
        c.close()


class Boom(Reranker):
    name = "boom"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        raise ProviderError("down")


def test_fallbacks_reranker_failure_and_empty_filter(tiny: Settings) -> None:
    c = Container(tiny)
    try:
        svc = RetrievalService(
            tiny,
            dense=c.dense_retriever,
            sparse=c.sparse_retriever,
            reranker=Boom(),
            extractor=c.filter_extractor,
        )
        res = svc.retrieve("Acme capital expenditure")
        assert "reranker_failed" in res.degraded and res.final  # RRF order kept
        none = svc.retrieve("capex", filters=QueryFilters(companies=["Nobody"]), rerank=False)
        assert "filter_returned_nothing" in none.degraded and none.final and none.applied is None
    finally:
        c.close()


def test_runner_writes_report(tiny: Settings) -> None:
    experiment = {
        "name": "tiny_ablation",
        "k": [1, 5, 10],
        "variants": [
            {"name": "dense", "retrieval": {"mode": "dense", "use_metadata_filter": False}},
            {"name": "hybrid+filter", "retrieval": {"mode": "hybrid", "use_metadata_filter": True}},
        ],
    }
    out = run_retrieval_eval(tiny, experiment, progress=False)
    report = (out / "report.md").read_text()
    assert "hybrid+filter" in report and "± " in report
    assert (out / "per_question.jsonl").read_text().count("\n") == 2 * 3  # 2 variants × 3 questions
