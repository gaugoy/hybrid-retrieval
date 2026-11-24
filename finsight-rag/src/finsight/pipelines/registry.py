"""Pipeline registry: @register_pipeline("name"); config `pipeline:` picks one."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from finsight.pipelines.base import RAGPipeline

PIPELINES: dict[str, type[RAGPipeline]] = {}


def register_pipeline(name: str) -> Callable[[type[RAGPipeline]], type[RAGPipeline]]:
    def deco(cls: type[RAGPipeline]) -> type[RAGPipeline]:
        PIPELINES[name] = cls
        cls.name = name
        return cls

    return deco


def load_all() -> None:
    from finsight.pipelines import hybrid_rerank, naive  # noqa: F401 (register)

    with contextlib.suppress(ImportError):
        from finsight.pipelines import agentic, corrective  # type: ignore[attr-defined]  # noqa: F401
