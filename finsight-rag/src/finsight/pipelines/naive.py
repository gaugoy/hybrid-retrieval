"""NaivePipeline: dense top-k straight into the LLM (no filters, no BM25, no reranker).

The baseline every improvement is measured against."""

from __future__ import annotations

import time

from finsight.pipelines.base import RAGPipeline, Retrieved
from finsight.pipelines.registry import register_pipeline
from finsight.retrieval.base import Retriever
from finsight.schemas import QueryFilters


@register_pipeline("naive")
class NaivePipeline(RAGPipeline):
    def __init__(self, *args: object, dense: Retriever, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.dense = dense

    def _retrieve(self, question: str, filters: QueryFilters | None) -> Retrieved:
        t0 = time.perf_counter()
        chunks = self.dense.retrieve(question, self.settings.rerank.top_n, filters)
        return Retrieved(
            chunks, {"dense": round((time.perf_counter() - t0) * 1000, 1)}, filters=filters
        )
