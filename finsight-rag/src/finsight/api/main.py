"""
FastAPI application factory.

    uvicorn finsight.api.main:app --port 8000      (make serve)

The lifespan builds ONE Container from config; with api.warmup=true it loads the embedder, the
indexes and the LLM at startup (first request fast, /ready honest). Embedded Qdrant allows one
process: run a single worker, or point QDRANT_URL at a Qdrant server (Docker, P10).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finsight.api.middleware import RequestContextMiddleware
from finsight.api.routes import router
from finsight.config import Settings, load_settings
from finsight.container import Container
from finsight.observability.logging import configure_logging, get_logger

log = get_logger("finsight.api")

app: FastAPI  # created on first access by __getattr__ below (nothing loads at import time)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else load_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.logging.level, settings.logging.json_format)
        c = container or Container(settings)
        app.state.container = c
        if settings.api.warmup:
            log.info("warming up", extra=c.describe())
            _ = c.embedder.dim, c.qdrant.count(c.collection), c.llm.model, c.pipeline
        yield
        c.close()

    app = FastAPI(
        title="FinSight RAG",
        version="1.0.0",
        lifespan=lifespan,
        description="Grounded, cited Q&A over SEC filings (FinanceBench).",
    )
    app.add_middleware(
        RequestContextMiddleware,
        max_body_bytes=settings.api.max_question_chars * 4 + 4096,
        per_minute=settings.api.rate_limit_per_minute,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


def __getattr__(name: str) -> FastAPI:
    """`finsight.api.main:app` is created on first access (nothing loads at import time)."""
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
