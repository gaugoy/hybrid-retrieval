"""HybridRerankPipeline: the configured RetrievalService (filters, dense/BM25/RRF, rerank) -> LLM.

The default pipeline. Despite the name it runs whatever `retrieval`/`rerank` say (P5 default:
dense + company/year filter, no reranker)."""

from __future__ import annotations

from finsight.pipelines.base import RAGPipeline, Retrieved
from finsight.pipelines.registry import register_pipeline
from finsight.retrieval.service import RetrievalResult, RetrievalService
from finsight.schemas import QueryFilters


@register_pipeline("hybrid_rerank")
class HybridRerankPipeline(RAGPipeline):
    def __init__(self, *args: object, retrieval: RetrievalService, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.retrieval = retrieval
        self.last: RetrievalResult | None = (
            None  # the last RetrievalResult (all stages), for the UI and scripts
        )

    def _retrieve(self, question: str, filters: QueryFilters | None) -> Retrieved:
        res = self.retrieval.retrieve(question, filters=filters)
        self.last = res
        return Retrieved(res.final, dict(res.timings_ms), list(res.degraded), res.applied)
