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
  P3: tokenizer, chunker · P4: embedder, qdrant, bm25 · P5: retrievers, reranker (done) ·
  P6: llm, pipeline · P8: judge.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from finsight.config import Settings
from finsight.observability.logging import get_logger

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from finsight.chunking.base import Chunker
    from finsight.chunking.tokenizer import Tokenizer
    from finsight.embedding.base import Embedder
    from finsight.indexing.bm25_store import BM25Index
    from finsight.indexing.qdrant_store import QdrantStore
    from finsight.retrieval.base import ChunkStore, Reranker, Retriever
    from finsight.retrieval.filters import FilterExtractor
    from finsight.retrieval.service import RetrievalService

log = get_logger(__name__)


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        log.debug("container created", extra={"config_fingerprint": settings.fingerprint()})

    @cached_property
    def tokenizer(self) -> Tokenizer:
        """The embedding model's tokenizer (P3): used to size chunks."""
        from finsight.chunking.tokenizer import get_tokenizer

        return get_tokenizer(self.settings.chunking.tokenizer)

    @cached_property
    def chunker(self) -> Chunker:
        """The configured chunking strategy (P3)."""
        from finsight.chunking.pipeline import make_chunker

        return make_chunker(self.settings, self.tokenizer)

    @cached_property
    def embedder(self) -> Embedder:
        """The configured embedder (P4): local model or online API, from embedding.preset."""
        from finsight.indexing.pipeline import make_embedder

        return make_embedder(self.settings)

    @cached_property
    def qdrant_client(self) -> QdrantClient:
        """One Qdrant client per container: embedded mode allows one per process (P4)."""
        from finsight.indexing.qdrant_store import open_client

        return open_client(self.settings)

    @cached_property
    def qdrant(self) -> QdrantStore:
        from finsight.indexing.qdrant_store import QdrantStore

        return QdrantStore(self.qdrant_client)

    @cached_property
    def collection(self) -> str:
        """Qdrant collection for the configured dataset, chunking and embedding model."""
        from finsight.indexing.pipeline import collection_name

        return collection_name(self.settings, self.embedder.slug)

    @cached_property
    def bm25(self) -> BM25Index:
        """The BM25 index for the configured chunking (P4)."""
        from finsight.indexing.pipeline import load_bm25

        return load_bm25(self.settings)

    @cached_property
    def chunk_store(self) -> ChunkStore:
        """chunk_id -> Chunk for the configured chunking (parents too, for parent_child)."""
        from finsight.chunking import pipeline as cp
        from finsight.retrieval.base import ChunkStore

        return ChunkStore.from_files(cp.chunks_path(self.settings), cp.parents_path(self.settings))

    @cached_property
    def dense_retriever(self) -> Retriever:
        from finsight.retrieval.dense import DenseRetriever

        return DenseRetriever(self.qdrant, self.embedder, self.collection)

    @cached_property
    def sparse_retriever(self) -> Retriever:
        from finsight.retrieval.sparse import SparseRetriever

        return SparseRetriever(self.bm25, self.chunk_store)

    @cached_property
    def reranker(self) -> Reranker:
        """The configured reranker (P5): none, local cross-encoder, or an online API."""
        from finsight.indexing.pipeline import usage_dir
        from finsight.retrieval.reranker import build_reranker

        s = self.settings
        return build_reranker(
            s.rerank, secrets=s.secrets, device=s.embedding.device, usage_dir=usage_dir(s)
        )

    @cached_property
    def filter_extractor(self) -> FilterExtractor:
        from finsight.data import financebench as fb
        from finsight.retrieval.filters import FilterExtractor

        companies = {m.company for m in fb.load_document_meta(self.settings).values()}
        return FilterExtractor(companies)

    @cached_property
    def retrieval(self) -> RetrievalService:
        """Filters → dense/BM25 → RRF → rerank, as configured (P5)."""
        from finsight.retrieval.service import RetrievalService

        mode = self.settings.retrieval.mode
        return RetrievalService(
            self.settings,
            dense=self.dense_retriever if mode in ("dense", "hybrid") else None,
            sparse=self.sparse_retriever if mode in ("sparse", "hybrid") else None,
            reranker=self.reranker,
            extractor=self.filter_extractor,
        )

    def close(self) -> None:
        """Release the Qdrant folder lock (embedded mode)."""
        if "qdrant_client" in self.__dict__:
            self.qdrant_client.close()
            del self.__dict__["qdrant_client"]
            self.__dict__.pop("qdrant", None)

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
            "embedding": f"{s.embedding.preset or '-'} "
            f"({s.embedding.provider}:{s.embedding.model})",
            "retrieval": f"{s.retrieval.mode} (candidate_k={s.retrieval.candidate_k})",
            "rerank": f"{s.rerank.provider}:{s.rerank.model}"
            if s.rerank.provider != "none"
            else "none",
            "pipeline": s.pipeline,
            "llm": f"{s.llm.provider}:{_llm_model(s)}",
        }


def _llm_model(s: Settings) -> str:
    return s.llm.ollama_model if s.llm.provider == "ollama" else s.llm.anthropic_model
