"""
Local bi-encoder embeddings with sentence-transformers (default: BAAI/bge-small-en-v1.5).

BI-ENCODER
  The query and each chunk are encoded SEPARATELY into one vector each (384 dims for bge-small);
  relevance is their cosine similarity. Chunks are encoded once at index time, so a search costs
  one query encoding plus a nearest-neighbour lookup. (A cross-encoder, used for reranking in P5,
  reads query and chunk together: more accurate, but far too slow to run over the whole corpus.)

DEVICE
  "auto" picks Apple's GPU (MPS) when available, else CUDA, else CPU.

QUERY INSTRUCTION
  bge v1.5 was trained with "Represent this sentence for searching relevant passages: " in front
  of short queries. The bge presets put it in embedding.query_template (queries only; set the
  template to "{text}" to switch it off). P5 measures retrieval with and without it.

The model is loaded on first use, not at import time.
More info: https://sbert.net/ · https://huggingface.co/BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from finsight.embedding.base import Embedder, model_slug
from finsight.observability.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from finsight.embedding.cache import EmbeddingCache

log = get_logger(__name__)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 64,
        query_template: str = "{text}",
        document_template: str = "{text}",
    ) -> None:
        import hashlib

        # identity: the model, plus the document template when it changes what gets embedded
        self.model_name = model_name
        if document_template != "{text}":
            self.model_name += "#" + hashlib.sha256(document_template.encode()).hexdigest()[:8]
        self.slug = model_slug(model_name)
        self._model_id = model_name
        self._device = device
        self.batch_size = batch_size
        self.query_template = query_template
        self.document_template = document_template
        # attached by make_embedder; caches query vectors (free + reproducible re-runs)
        self.query_cache: EmbeddingCache | None = None

    @cached_property
    def model(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        device = resolve_device(self._device)
        log.info("loading embedding model", extra={"model": self._model_id, "device": device})
        return SentenceTransformer(self._model_id, device=device)

    @property
    def device(self) -> str:
        return str(self.model.device)

    @property
    def dim(self) -> int:  # type: ignore[override]
        get_dim = getattr(self.model, "get_embedding_dimension", None)  # sentence-transformers ≥ 6
        dim = get_dim() if get_dim else self.model.get_sentence_embedding_dimension()
        assert dim is not None
        return int(dim)

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length or 0)

    def embed_documents(self, texts: list[str], progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.document_template != "{text}":
            texts = [self.document_template.format(text=t) for t in texts]
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        from finsight.embedding.cache import cached_query_vectors

        formatted = [self.query_template.format(text=t) for t in texts]
        ident = f"{self.model_name}|query|{self.query_template}"
        vecs, _ = cached_query_vectors(self.query_cache, ident, formatted, self._encode, self.dim)
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]
