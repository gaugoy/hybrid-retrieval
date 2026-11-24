"""
The Embedder interface, and the factory that builds one from config (local or online).

WHAT AN EMBEDDER PROMISES
  embed_documents(texts) -> (n, dim) float32, every row L2-normalized (length 1)
  embed_query(text)      -> (dim,)   float32, L2-normalized
  embed_queries(texts)   -> (n, dim) batch version (online providers: one request, cached)
  model_name             identity of the vectors (cache keys, index manifest)
  slug                   short name for files and Qdrant collections
  Because every vector has length 1, cosine similarity is just a dot product.

WHY QUERIES AND DOCUMENTS ARE SEPARATE
  Retrieval models are often asymmetric: bge v1.5 wants an instruction in front of the QUERY,
  gemini-embedding-2 wants "task: search result | query: ..." vs "title: none | text: ...",
  Cohere/Voyage/Jina take an input type. config: query_template / document_template /
  query_params / document_params. Keeping two calls makes it impossible to mix them up.

CHOOSING ONE
  embedding.preset (configs/embedders.yaml) fills provider + model + templates + rate limits.
  build_embedder() checks the API key and refuses paid-only providers unless allowed.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from finsight.config import EmbeddingCfg
from finsight.errors import MissingAPIKeyError, PaidProviderBlockedError

if TYPE_CHECKING:
    from finsight.config import Secrets


class Embedder(ABC):
    model_name: str
    slug: str
    dim: int
    provider: str = "local"

    @abstractmethod
    def embed_documents(self, texts: list[str], progress: bool = False) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_query(t) for t in texts]).astype(np.float32)

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def preferred_batch(self) -> int:
        """Texts per embed/cache round in embed_with_cache (small for rate-limited APIs)."""
        return 1024


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def model_slug(model_name: str) -> str:
    """'BAAI/bge-small-en-v1.5' -> 'bge-small-en-v1-5' (safe in file and collection names)."""
    return slugify(model_name.split("/")[-1])


def embedder_slug(cfg: EmbeddingCfg) -> str:
    """The slug an embedder built from `cfg` will have (no model load, no API key needed)."""
    if cfg.provider == "fake" or cfg.model.startswith("fake"):
        return f"fake-{cfg.dimensions or 32}"
    if cfg.provider == "local":
        return model_slug(cfg.model)
    return slugify(f"{cfg.provider}-{cfg.model}" + (f"-{cfg.dimensions}" if cfg.dimensions else ""))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def check_allowed(cfg: EmbeddingCfg, secrets: Secrets | None) -> str | None:
    """Return the API key to use; raise if it is missing or the provider is paid-only."""
    if cfg.provider in ("local", "fake"):
        return None
    from finsight.embedding.remote import PROVIDERS

    spec = PROVIDERS[cfg.provider]
    if spec.paid_only and not cfg.allow_paid:
        price = spec.price_per_mtok.get(cfg.model)
        cost = f" (${price}/1M tokens)" if price else ""
        raise PaidProviderBlockedError(
            f"{cfg.provider} has no free tier{cost}. To spend money on purpose set "
            "embedding.allow_paid: true (or FINSIGHT__EMBEDDING__ALLOW_PAID=true). Free "
            "online options: gemini-001, gemini-2, mistral-embed, cohere-v4 "
            "(list: uv run finsight embedders)."
        )
    if not spec.needs_key:
        return None
    key = secrets.for_provider(cfg.provider) if secrets else None
    if key is None:
        raise MissingAPIKeyError(
            f"Embedder {cfg.preset or cfg.model!r} needs {' or '.join(spec.key_env)} in .env "
            f"(free key: {spec.key_url})."
        )
    return key.get_secret_value()


def build_embedder(
    cfg: EmbeddingCfg,
    *,
    secrets: Secrets | None = None,
    usage_dir: Path | None = None,
) -> Embedder:
    """The configured embedder: local sentence-transformers, the test fake, or an online API."""
    if cfg.provider == "fake" or cfg.model.startswith("fake"):
        from finsight.embedding.fake import FakeEmbedder

        return FakeEmbedder(dim=cfg.dimensions or 32)
    if cfg.provider == "local":
        from finsight.embedding.sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            cfg.model,
            device=cfg.device,
            batch_size=cfg.batch_size,
            query_template=cfg.query_template,
            document_template=cfg.document_template,
        )
    from finsight.embedding.remote import PROVIDERS, WIRES

    api_key = check_allowed(cfg, secrets)
    spec = PROVIDERS[cfg.provider]
    return WIRES[spec.wire](cfg, spec, api_key, usage_dir=usage_dir)
