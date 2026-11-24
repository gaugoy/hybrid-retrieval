"""
RetrievalService: the configured retrieval stages, with every intermediate ranking recorded.

    question ─► filters (rule-based) ─► dense ┐
                                      ─► BM25  ┴► RRF (hybrid) ─► rerank ─► top_k
    retrieval.mode = dense | sparse | hybrid;  rerank.provider = none | local | cohere | ...

Each stage keeps `candidate_k` chunks (default 50); the reranker re-orders all of them and the
result is cut to `top_k`. The generator later uses the first `rerank.top_n` as context.

SAFETY NETS
  - The company filter returned nothing (an alias matched a company with no filing in the
    corpus)? Retry without filters, and say so in `degraded`.
  - The reranker failed (API down, quota)? Keep the RRF order (rerank.on_error=fallback).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from finsight.config import Settings
from finsight.errors import FinsightError
from finsight.observability.logging import get_logger
from finsight.retrieval.base import Reranker, Retriever
from finsight.retrieval.filters import FilterExtractor, apply_fields
from finsight.retrieval.hybrid import rrf_fuse
from finsight.retrieval.reranker import NoopReranker
from finsight.schemas import QueryFilters, RetrievedChunk

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    question: str
    extracted: QueryFilters
    applied: QueryFilters | None
    stages: dict[str, list[RetrievedChunk]] = field(default_factory=dict)
    final: list[RetrievedChunk] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        *,
        dense: Retriever | None,
        sparse: Retriever | None,
        reranker: Reranker | None = None,
        extractor: FilterExtractor | None = None,
        sparse_fallback: Callable[[], Retriever] | None = None,
    ) -> None:
        self.settings, self.dense, self.sparse = settings, dense, sparse
        # P10: BM25, built only if the dense search fails in dense-only mode
        self.sparse_fallback = sparse_fallback
        self.reranker = reranker or NoopReranker()
        self.extractor = extractor

    def retrieve(
        self,
        question: str,
        *,
        filters: QueryFilters | None = None,
        mode: str | None = None,
        use_filter: bool | None = None,
        rerank: bool | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        cfg = self.settings.retrieval
        mode = mode or cfg.mode
        use_filter = cfg.use_metadata_filter if use_filter is None else use_filter
        do_rerank = (self.reranker.name != "none") if rerank is None else rerank
        top_k = top_k or cfg.top_k

        t0 = time.perf_counter()
        extracted = self.extractor.extract(question) if self.extractor else QueryFilters()
        applied = (
            filters
            if filters is not None
            else (apply_fields(extracted, cfg.filter_fields) if use_filter else None)
        )
        result = RetrievalResult(question, extracted, applied)
        result.timings_ms["filter"] = _ms(t0)

        candidates = self._candidates(question, mode, applied, result)
        if not candidates and applied is not None:
            result.degraded.append("filter_returned_nothing")
            result.applied = None
            candidates = self._candidates(question, mode, None, result)

        final = candidates
        if do_rerank and candidates:
            t0 = time.perf_counter()
            try:
                final = self.reranker.rerank(question, candidates, top_n=len(candidates))
                result.stages["rerank"] = final
            except FinsightError as exc:
                if self.settings.rerank.on_error == "raise":
                    raise
                log.warning("reranker failed, keeping RRF order", extra={"error": str(exc)})
                result.degraded.append("reranker_failed")
            result.timings_ms["rerank"] = _ms(t0)
        result.final = [
            rc.model_copy(update={"rank": i + 1, "stage": "final"})
            for i, rc in enumerate(final[:top_k])
        ]
        return result

    def _candidates(
        self, question: str, mode: str, filters: QueryFilters | None, result: RetrievalResult
    ) -> list[RetrievedChunk]:
        k = self.settings.retrieval.candidate_k
        lists: list[list[RetrievedChunk]] = []
        for name, retriever in (("dense", self.dense), ("sparse", self.sparse)):
            if retriever is None or (mode != "hybrid" and mode != name):
                continue
            t0 = time.perf_counter()
            try:
                hits = retriever.retrieve(question, k, filters)
            except Exception as exc:
                if name != "dense":
                    raise
                log.warning("dense search failed, using BM25", extra={"error": str(exc)})
                if "dense_failed" not in result.degraded:
                    result.degraded.append("dense_failed")
                if mode == "dense" and self.sparse_fallback is not None:
                    mode, retriever, name = "sparse", self.sparse_fallback(), "sparse"
                    hits = retriever.retrieve(question, k, filters)
                else:
                    continue
            result.timings_ms[name] = _ms(t0)
            result.stages[name] = hits
            lists.append(hits)
        if mode != "hybrid" or len(lists) == 1:
            return lists[0] if lists else []
        t0 = time.perf_counter()
        fused = rrf_fuse(lists, self.settings.retrieval.rrf_k)[:k]
        result.timings_ms["fuse"] = _ms(t0)
        result.stages["hybrid"] = fused
        return fused


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
