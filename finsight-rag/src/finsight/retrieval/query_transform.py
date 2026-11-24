"""
Query transformations: change WHAT we search for, not how we rank.

  multi_query  the LLM writes 3 rewrites in filing vocabulary; retrieve each (and the original)
               with hybrid search, fuse everything with RRF, rerank against the ORIGINAL question
  hyde         the LLM writes a hypothetical filing passage; embed THAT for the dense side
               (documents look like documents), keep the original question for BM25
  decompose    the LLM splits a multi-company / multi-year question into ≤ 3 sub-questions;
               retrieve each, fuse, rerank against the original
Each costs one extra LLM call (more with a retry) before retrieval. Invalid JSON from the LLM
falls back to plain retrieval (recorded in `degraded`). Motivation (P4/P5): FinanceBench asks
for "capital expenditure", 3M's statement says "Purchases of property, plant and equipment".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from finsight.generation import prompt_loader
from finsight.generation.jsonout import ask_json
from finsight.generation.llm.base import LLM
from finsight.retrieval.filters import apply_fields
from finsight.retrieval.hybrid import rrf_fuse
from finsight.retrieval.service import RetrievalResult, RetrievalService
from finsight.schemas import QueryFilters, RetrievedChunk

TECHNIQUES = ("none", "multi_query", "hyde", "decompose")


class Queries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)


@dataclass
class TransformTrace:
    technique: str
    queries: list[str] = field(default_factory=list)
    hyde_passage: str = ""
    llm_calls: int = 0
    llm_ms: float = 0.0


def transformed_retrieve(
    service: RetrievalService,
    llm: LLM,
    question: str,
    technique: str,
    filters: QueryFilters | None = None,
    top_k: int | None = None,
) -> tuple[RetrievalResult, TransformTrace]:
    trace = TransformTrace(technique)
    if technique == "none":
        return service.retrieve(question, filters=filters, top_k=top_k), trace

    t0 = time.perf_counter()
    queries: list[str] = []
    if technique in ("multi_query", "decompose"):
        parsed, trace.llm_calls = ask_json(
            llm, prompt_loader.render(technique, question=question), Queries
        )
        queries = [q.strip() for q in (parsed.queries if parsed else []) if q.strip()][:3]
    elif technique == "hyde":
        resp = llm.generate(
            "You write realistic SEC filing excerpts.",
            [{"role": "user", "content": prompt_loader.render("hyde", question=question)}],
            300,
        )
        trace.llm_calls, trace.hyde_passage = 1, resp.text.strip()
    trace.llm_ms = round((time.perf_counter() - t0) * 1000, 1)
    trace.queries = queries

    cfg = service.settings.retrieval
    extracted = service.extractor.extract(question) if service.extractor else QueryFilters()
    applied = (
        filters
        if filters is not None
        else (apply_fields(extracted, cfg.filter_fields) if cfg.use_metadata_filter else None)
    )
    result = RetrievalResult(question, extracted, applied)
    result.timings_ms["transform_llm"] = trace.llm_ms

    lists: list[list[RetrievedChunk]] = []
    t1 = time.perf_counter()
    if technique == "hyde":
        if trace.hyde_passage and service.dense is not None:
            lists.append(service.dense.retrieve(trace.hyde_passage, cfg.candidate_k, applied))
        if service.sparse is not None:
            lists.append(service.sparse.retrieve(question, cfg.candidate_k, applied))
        if not trace.hyde_passage:
            result.degraded.append("hyde_empty")
    else:
        if not queries:
            result.degraded.append(f"{technique}_parse_failed")
        for q in [question, *queries]:
            lists.append(service._candidates(q, cfg.mode, applied, result))
    result.timings_ms["retrieve"] = round((time.perf_counter() - t1) * 1000, 1)
    fused = rrf_fuse(lists, cfg.rrf_k)[: cfg.candidate_k]
    result.stages["fused"] = fused
    final = fused
    if service.reranker.name != "none" and fused:
        t2 = time.perf_counter()
        final = service.reranker.rerank(question, fused, top_n=len(fused))
        result.stages["rerank"] = final
        result.timings_ms["rerank"] = round((time.perf_counter() - t2) * 1000, 1)
    k = top_k or cfg.top_k
    result.final = [
        rc.model_copy(update={"rank": i + 1, "stage": "final"}) for i, rc in enumerate(final[:k])
    ]
    return result, trace
