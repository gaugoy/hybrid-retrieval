"""HybridRerankPipeline: the configured RetrievalService (filters, dense/BM25/RRF, rerank) -> LLM.

The default pipeline. Despite the name it runs whatever `retrieval`/`rerank` say (P5 default:
dense + company/year filter, no reranker). With `retrieval.query_transform` = multi_query |
hyde | decompose (P7), the LLM rewrites the query before retrieval (see query_transform.py).
"""

from __future__ import annotations

from finsight.pipelines.base import RAGPipeline, Retrieved
from finsight.pipelines.registry import register_pipeline
from finsight.retrieval.query_transform import TransformTrace, transformed_retrieve
from finsight.retrieval.service import RetrievalResult, RetrievalService
from finsight.schemas import QueryFilters


@register_pipeline("hybrid_rerank")
class HybridRerankPipeline(RAGPipeline):
    def __init__(self, *args: object, retrieval: RetrievalService, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.retrieval = retrieval
        self.last: RetrievalResult | None = None  # the last RetrievalResult (all stages)
        self.last_trace: TransformTrace | None = None
        self.llm_calls = 0  # LLM calls before generation (query transforms, graders, agents)

    def _retrieve(self, question: str, filters: QueryFilters | None) -> Retrieved:
        technique = self.settings.retrieval.query_transform
        if technique == "none":
            res = self.retrieval.retrieve(question, filters=filters)
            self.llm_calls = 0
        else:
            res, trace = transformed_retrieve(
                self.retrieval, self.llm, question, technique, filters
            )
            self.last_trace, self.llm_calls = trace, trace.llm_calls
        self.last = res
        return Retrieved(res.final, dict(res.timings_ms), list(res.degraded), res.applied)
