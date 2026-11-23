"""
Data contracts shared by every stage of the pipeline.

WHY
  Each stage (ingestion → chunking → indexing → retrieval → generation → evaluation) hands data
  to the next one. Pydantic models make those hand-offs explicit and validated, and give the
  FastAPI response format for free. Changing a model here is a contract change: record it in
  docs/DEVELOPMENT_GUIDE.md §7.

PAGE CONVENTION
  ``page_idx`` is always 0-based, matching PyMuPDF and FinanceBench's ``evidence_page_num``.
  Add 1 only when showing a page number to a human.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    doc_name: str  # "3M_2018_10K": primary key, also the PDF file stem
    company: str  # "3M"
    doc_type: str  # "10k" | "10q" | "8k" | "Earnings"
    doc_period: int  # 2018
    gics_sector: str | None = None
    source_url: str | None = None


class Page(BaseModel):
    doc_name: str
    page_idx: int = Field(ge=0)  # 0-based
    text: str  # cleaned text; tables rendered as pipe rows
    n_chars: int = Field(ge=0)
    has_table: bool = False


class Chunk(BaseModel):
    chunk_id: str  # f"{doc_name}::p{page_start}-{page_end}::c{seq}"
    doc_name: str
    company: str
    doc_type: str
    doc_period: int
    page_start: int = Field(ge=0)  # 0-based, inclusive
    page_end: int = Field(ge=0)  # 0-based, inclusive
    text: str  # what the LLM and the user see
    embed_text: str  # what gets embedded / BM25-indexed (may include a contextual header)
    n_tokens: int = Field(ge=0)
    strategy: str  # "fixed" | "recursive" | "page_aware" | "parent_child"
    parent_id: str | None = None  # parent_child only


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int = Field(ge=1)  # 1-based within this stage
    stage: Literal["dense", "sparse", "hybrid", "rerank", "final"]
    # Per-stage scores, for the UI's "under the hood" panel.
    scores_by_stage: dict[str, float] = Field(default_factory=dict)


class QueryFilters(BaseModel):
    companies: list[str] = Field(default_factory=list)
    doc_periods: list[int] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.companies or self.doc_periods or self.doc_types)


class Citation(BaseModel):
    marker: int  # the [n] used in the answer text
    chunk_id: str
    doc_name: str
    page_idx: int = Field(ge=0)
    snippet: str  # first ~300 chars of the chunk


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    abstained: bool
    retrieved: list[RetrievedChunk]
    pipeline: str
    provider: str
    model: str
    # stage -> ms: filter, dense, sparse, fuse, rerank, generate, total
    timings_ms: dict[str, float]
    usage: dict[str, int] = Field(default_factory=dict)  # input_tokens, output_tokens
    request_id: str


class Evidence(BaseModel):
    doc_name: str
    page_idx: int = Field(ge=0)  # 0-based
    text: str


class EvalRecord(BaseModel):
    qid: str  # financebench_id
    question: str
    gold_answer: str | None  # None = unanswerable (MultiHop null queries)
    question_type: str
    company: str | None
    doc_name: str | None
    evidence: list[Evidence]
    dataset: Literal["financebench", "multihop"]
