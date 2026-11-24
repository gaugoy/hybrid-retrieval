"""
Dependency container: builds pipeline components from ``Settings``, lazily, once.

WHY
  The CLI, the learning scripts, the FastAPI app and the eval runner all need the same objects
  (embedder, retrievers, reranker, LLM, pipeline). If each built them its own way, the
  pipeline you *evaluate* could quietly differ from the one you *serve*. Building everything
  here, from one ``Settings``, keeps them identical.

  Components are ``cached_property``: nothing heavy (models, indexes, connections) is created
  until something asks for it, and it is created only once per container.

GROWTH PLAN
  P3: tokenizer, chunker (done) · P4: embedder, qdrant, bm25 · P5: retrievers, reranker ·
  P6: llm, pipeline · P8: judge.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from finsight.config import Settings
from finsight.observability.logging import get_logger

if TYPE_CHECKING:
    from finsight.chunking.base import Chunker
    from finsight.chunking.tokenizer import Tokenizer

log = get_logger(__name__)


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        log.debug("container created", extra={"config_fingerprint": settings.fingerprint()})

    @cached_property
    def tokenizer(self) -> Tokenizer:
        """The embedding model's tokenizer (P3): used to size chunks."""
        from finsight.chunking.tokenizer import get_tokenizer

        return get_tokenizer(self.settings.embedding.model)

    @cached_property
    def chunker(self) -> Chunker:
        """The configured chunking strategy (P3)."""
        from finsight.chunking.pipeline import make_chunker

        return make_chunker(self.settings, self.tokenizer)

    def describe(self) -> dict[str, str]:
        """A short summary of what this container would build: handy for logs and /v1/config."""
        s = self.settings
        return {
            "config": str(
                s.config_path.relative_to(s.root)
                if s.config_path and s.config_path.is_relative_to(s.root)
                else s.config_path
            ),
            "fingerprint": s.fingerprint(),
            "dataset": s.dataset,
            "chunking": f"{s.chunking.strategy}/{s.chunking.chunk_tokens}",
            "embedding": s.embedding.model,
            "retrieval": f"{s.retrieval.mode} (candidate_k={s.retrieval.candidate_k})",
            "rerank": f"{s.rerank.provider}:{s.rerank.model}"
            if s.rerank.provider != "none"
            else "none",
            "pipeline": s.pipeline,
            "llm": f"{s.llm.provider}:{_llm_model(s)}",
        }


def _llm_model(s: Settings) -> str:
    return s.llm.ollama_model if s.llm.provider == "ollama" else s.llm.anthropic_model
