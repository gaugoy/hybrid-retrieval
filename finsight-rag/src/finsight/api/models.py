"""Request/response models of the HTTP API (validation lives here, not in the routes)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from finsight.schemas import Answer, QueryFilters

Pipeline = Literal["naive", "hybrid_rerank", "corrective", "agentic"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    pipeline: Pipeline | None = None  # default: config
    llm: str | None = Field(None, description="LLM preset (configs/llms.yaml); default: config")
    filters: QueryFilters | None = None  # default: extracted from the question
    top_k: int | None = Field(None, ge=1, le=20)
    include_retrieved: bool = False


class QueryResponse(Answer):
    """An Answer; `retrieved` is trimmed to [] unless include_retrieved=true."""


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Ready(BaseModel):
    ready: bool
    checks: dict[str, str]
