"""
RAGPipeline interface (contract §4.2) and the shared generation step.

    retrieve  (pipeline-specific)  ->  build_context  ->  LLM  ->  parse_citations  ->  Answer

Every Answer carries: the answer text, validated citations, abstained flag, the retrieved chunks,
provider/model, per-stage timings (ms), token usage, warnings, degraded fallbacks, request_id.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from finsight.chunking.tokenizer import Tokenizer
from finsight.config import Settings
from finsight.generation import prompt_loader
from finsight.generation.citations import parse_citations
from finsight.generation.context import build_context
from finsight.generation.llm.base import LLM
from finsight.retrieval.base import ChunkStore
from finsight.schemas import Answer, QueryFilters, RetrievedChunk


@dataclass
class Retrieved:
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    filters: QueryFilters | None = None


class RAGPipeline(ABC):
    name: str

    def __init__(
        self, settings: Settings, llm: LLM, tokenizer: Tokenizer, store: ChunkStore | None
    ) -> None:
        self.settings, self.llm, self.tokenizer, self.store = settings, llm, tokenizer, store

    @abstractmethod
    def _retrieve(self, question: str, filters: QueryFilters | None) -> Retrieved: ...

    def retrieve_only(
        self, question: str, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]:
        """Used by the retrieval eval (no LLM cost)."""
        return self._retrieve(question, filters).chunks

    def answer(
        self, question: str, filters: QueryFilters | None = None, request_id: str | None = None
    ) -> Answer:
        t_all = time.perf_counter()
        got = self._retrieve(question, filters)
        return self.generate(question, got, t_all, request_id)

    def prompt(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, str, dict]:
        g = self.settings.generation
        context, marker_map = build_context(
            chunks,
            max_tokens=g.max_context_tokens,
            count_tokens=self.tokenizer.count,
            store=self.store,
            max_blocks=self.settings.rerank.top_n,
        )
        system = prompt_loader.render("answer_system", abstain_phrase=g.abstain_phrase)
        user = prompt_loader.render(
            "answer_user",
            context=context or "(no context)",
            question=question,
            abstain_phrase=g.abstain_phrase,
        )
        return system, user, marker_map

    def generate(
        self, question: str, got: Retrieved, t_all: float, request_id: str | None = None
    ) -> Answer:
        g = self.settings.generation
        timings = dict(got.timings_ms)
        system, user, marker_map = self.prompt(question, got.chunks)
        t0 = time.perf_counter()
        resp = self.llm.generate(system, [{"role": "user", "content": user}], g.max_answer_tokens)
        timings["generate"] = round((time.perf_counter() - t0) * 1000, 1)
        text = resp.text.strip() or g.abstain_phrase
        citations, abstained, warnings = parse_citations(text, marker_map, g.abstain_phrase)
        if resp.stop_reason == "refusal":
            warnings.append("model_refusal")
            abstained = True
        timings["total"] = round((time.perf_counter() - t_all) * 1000, 1)
        return Answer(
            question=question,
            answer=text,
            citations=citations,
            abstained=abstained,
            retrieved=got.chunks,
            pipeline=self.name,
            provider=self.llm.provider,
            model=resp.model,
            timings_ms=timings,
            usage={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
            request_id=request_id or uuid.uuid4().hex[:12],
            warnings=warnings,
            degraded=got.degraded,
        )
