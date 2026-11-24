"""
Endpoints.

  GET  /health            the process is alive
  GET  /ready             indexes loaded and an LLM available (503 + reasons otherwise)
  GET  /v1/config         resolved non-secret config + fingerprint
  POST /v1/query          an Answer (retrieved trimmed unless include_retrieved)
  POST /v1/query/stream   Server-Sent Events: `retrieved` (context preview), `token` (deltas),
                          `done` (final answer with citations, timings)
Per-request `pipeline` / `llm` overrides build a request-scoped pipeline from the SAME shared
components (embedder, indexes, retrieval), so they never reload the models.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse

from finsight.api.deps import ContainerDep
from finsight.api.models import Health, QueryRequest, QueryResponse, Ready
from finsight.container import Container
from finsight.errors import ConfigError, FinsightError
from finsight.generation.citations import parse_citations
from finsight.pipelines.base import RAGPipeline

router = APIRouter()


def _pipeline(c: Container, req: QueryRequest) -> RAGPipeline:
    if not req.pipeline and not req.llm and not req.top_k:
        return c.pipeline
    updates: dict = {}
    if req.pipeline:
        updates["pipeline"] = req.pipeline
    if req.top_k:
        updates["retrieval"] = c.settings.retrieval.model_copy(update={"top_k": req.top_k})
    s = c.settings.model_copy(update=updates)
    s.root, s.secrets, s.paths, s.config_path = (
        c.settings.root,
        c.settings.secrets,
        c.settings.paths,
        c.settings.config_path,
    )
    scoped = Container(s)
    # share the heavy, already-loaded components
    for name in (
        "tokenizer",
        "embedder",
        "qdrant_client",
        "qdrant",
        "collection",
        "bm25",
        "chunk_store",
        "reranker",
        "filter_extractor",
    ):
        if name in c.__dict__:
            scoped.__dict__[name] = c.__dict__[name]
    if not req.llm:
        scoped.__dict__["llm"] = c.llm
    else:
        from finsight.config import LLMCfg, resolve_llm_preset
        from finsight.generation.llm.factory import build_llm

        try:
            cfg = LLMCfg.model_validate(resolve_llm_preset(s.root, {"preset": req.llm}))
            scoped.__dict__["llm"] = build_llm(cfg, root=s.root, secrets=s.secrets)
        except (ConfigError, ValueError) as exc:
            raise HTTPException(422, f"llm {req.llm!r}: {exc}") from exc
    return scoped.pipeline


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@router.get("/health", response_model=Health)
def health() -> Health:
    return Health()


@router.get("/ready", response_model=Ready)
def ready(c: ContainerDep) -> JSONResponse:
    checks: dict[str, str] = {}
    try:
        n = c.qdrant.count(c.collection)
        checks["dense_index"] = f"ok ({n} points)" if n else "empty: run make index"
    except Exception as exc:
        checks["dense_index"] = f"error: {exc}"
    try:
        checks["llm"] = f"ok ({c.llm.provider}:{c.llm.model})"
    except FinsightError as exc:
        checks["llm"] = f"error: {exc}"
    ok = all(v.startswith("ok") for v in checks.values())
    return JSONResponse(Ready(ready=ok, checks=checks).model_dump(), status_code=200 if ok else 503)


@router.get("/v1/config")
def config(c: ContainerDep) -> dict:
    return {
        **c.describe(),
        "settings": c.settings.model_dump(mode="json", exclude={"paths", "logging", "api"}),
    }


@router.post("/v1/query", response_model=QueryResponse)
def query(
    req: QueryRequest, request: Request, response: Response, c: ContainerDep
) -> QueryResponse:
    cache = c.answer_cache
    key = (
        cache.key(
            c.settings.fingerprint(),
            question=req.question.strip(),
            filters=req.filters.model_dump() if req.filters else None,
            pipeline=req.pipeline or c.settings.pipeline,
            llm=req.llm or c.settings.llm.preset,
            top_k=req.top_k,
        )
        if cache
        else ""
    )
    ans = cache.get(key) if cache else None
    response.headers["X-Cache"] = "hit" if ans else "miss"
    if ans is not None:
        ans = ans.model_copy(update={"request_id": request.state.request_id})
    else:
        pipe = _pipeline(c, req)
        try:
            ans = pipe.answer(
                req.question, filters=req.filters, request_id=request.state.request_id
            )
        except FinsightError as exc:
            raise HTTPException(503, f"{type(exc).__name__}: {exc}") from exc
        if cache and not ans.degraded:  # never cache an answer produced during an outage
            cache.put(key, ans)
    data = ans.model_dump()
    if not req.include_retrieved:
        data["retrieved"] = []
    return QueryResponse(**data)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/v1/query/stream")
def query_stream(req: QueryRequest, request: Request, c: ContainerDep) -> StreamingResponse:
    pipe = _pipeline(c, req)
    rid = request.state.request_id

    def events() -> Iterator[str]:
        t0 = time.perf_counter()
        got = pipe._retrieve(req.question, req.filters)
        system, user, marker_map = pipe.prompt(req.question, got.chunks)
        yield _sse(
            "retrieved",
            [
                {"marker": n, "doc": rc.chunk.doc_name, "page": rc.chunk.page_start + 1}
                for n, rc in marker_map.items()
            ],
        )
        parts = []
        try:
            for delta in pipe.llm.stream(
                system, [{"role": "user", "content": user}], c.settings.generation.max_answer_tokens
            ):
                parts.append(delta)
                yield _sse("token", delta)
        except FinsightError as exc:
            yield _sse("error", str(exc))
            return
        text = "".join(parts).strip() or c.settings.generation.abstain_phrase
        cites, abstained, warnings = parse_citations(
            text, marker_map, c.settings.generation.abstain_phrase
        )
        yield _sse(
            "done",
            {
                "answer": text,
                "citations": [ci.model_dump() for ci in cites],
                "abstained": abstained,
                "warnings": warnings,
                "request_id": rid,
                "timings_ms": {
                    **got.timings_ms,
                    "total": round((time.perf_counter() - t0) * 1000, 1),
                },
            },
        )

    return StreamingResponse(events(), media_type="text/event-stream")
