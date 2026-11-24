"""
CorrectiveRAGPipeline (after CRAG, arXiv 2401.15884), in three cheap steps:

  1. retrieve + rerank (hybrid_rerank)
  2. grade: insufficient if the top rerank score < crag.min_score (calibrated on P5 data), or,
     with crag.llm_grader, if the LLM says the blocks don't contain the answer
  3. insufficient -> the LLM rewrites the query in filing vocabulary -> retrieve again -> fuse with
     the first results -> rerank against the original question -> grade by score again;
     still insufficient -> ABSTAIN without calling the generator (saves a call, avoids guessing)
The branch taken is recorded in Answer.timings_ms ("crag_rewrite", ...) and Answer.warnings.
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from finsight.generation import prompt_loader
from finsight.generation.context import build_context
from finsight.generation.jsonout import ask_json
from finsight.pipelines.base import Retrieved
from finsight.pipelines.hybrid_rerank import HybridRerankPipeline
from finsight.pipelines.registry import register_pipeline
from finsight.retrieval.hybrid import rrf_fuse
from finsight.schemas import Answer, QueryFilters, RetrievedChunk


class Grade(BaseModel):
    sufficient: bool
    missing: str = ""


class Rewrite(BaseModel):
    query: str


def top_score(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return float("-inf")
    s = chunks[0].scores_by_stage
    for key in ("local", "cohere", "jina", "voyage"):
        if key in s:
            return float(s[key])
    return float(chunks[0].score)


@register_pipeline("corrective")
class CorrectiveRAGPipeline(HybridRerankPipeline):
    def _grade(self, question: str, chunks: list[RetrievedChunk]) -> tuple[bool, str, int]:
        cfg = self.settings.crag
        if top_score(chunks) < cfg.min_score:
            return False, "low rerank score", 0
        if not cfg.llm_grader:
            return True, "", 0
        context, _ = build_context(chunks[:4], max_tokens=2500, count_tokens=self.tokenizer.count)
        g, calls = ask_json(
            self.llm,
            prompt_loader.render("grade", question=question, context=context),
            Grade,
            max_tokens=150,
        )
        if g is None:
            return True, "", calls  # grader failed: don't block the answer
        return g.sufficient, g.missing, calls

    def _retrieve(self, question: str, filters: QueryFilters | None) -> Retrieved:
        first = super()._retrieve(question, filters)
        t0 = time.perf_counter()
        ok, missing, calls = self._grade(question, first.chunks)
        first.timings_ms["crag_grade"] = round((time.perf_counter() - t0) * 1000, 1)
        self.llm_calls = calls
        self.branch = "pass"
        if ok:
            return first
        t1 = time.perf_counter()
        rw, c2 = ask_json(
            self.llm,
            prompt_loader.render(
                "rewrite", question=question, missing=missing or "the exact figure"
            ),
            Rewrite,
            max_tokens=120,
        )
        self.llm_calls += c2
        if rw is None or not rw.query.strip():
            first.degraded.append("crag_rewrite_failed")
            self.branch = "abstain"
            return first
        second = self.retrieval.retrieve(rw.query, filters=filters, rerank=False)
        merged = rrf_fuse([first.chunks, second.final], self.settings.retrieval.rrf_k)
        if self.retrieval.reranker.name != "none":
            merged = self.retrieval.reranker.rerank(question, merged, top_n=len(merged))
        merged = merged[: self.settings.retrieval.top_k]
        first.timings_ms["crag_rewrite"] = round((time.perf_counter() - t1) * 1000, 1)
        self.branch = "rewrite" if top_score(merged) >= self.settings.crag.min_score else "abstain"
        return Retrieved(merged, first.timings_ms, first.degraded, first.filters)

    def answer(
        self, question: str, filters: QueryFilters | None = None, request_id: str | None = None
    ) -> Answer:
        t_all = time.perf_counter()
        got = self._retrieve(question, filters)
        if self.branch == "abstain":
            g = self.settings.generation
            ans = Answer(
                question=question,
                answer=g.abstain_phrase,
                citations=[],
                abstained=True,
                retrieved=got.chunks,
                pipeline=self.name,
                provider=self.llm.provider,
                model=self.llm.model,
                timings_ms={
                    **got.timings_ms,
                    "total": round((time.perf_counter() - t_all) * 1000, 1),
                },
                request_id=request_id or "",
                warnings=["crag_abstained"],
                degraded=got.degraded,
            )
            return ans
        ans = self.generate(question, got, t_all, request_id)
        if self.branch == "rewrite":
            ans.warnings.append("crag_rewrote_query")
        return ans
