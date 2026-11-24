"""Embedding cache + FakeEmbedder: only cache misses reach the model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from finsight.config import EmbeddingCfg
from finsight.embedding.base import build_embedder, model_slug
from finsight.embedding.cache import EmbeddingCache, cache_key, embed_with_cache
from finsight.embedding.fake import FakeEmbedder


def test_fake_embedder_is_deterministic_normalized_and_lexical() -> None:
    emb = FakeEmbedder(dim=64)
    docs = emb.embed_documents(["capital expenditure 2018", "capital expenditure", "net sales"])
    assert docs.shape == (3, 64) and docs.dtype == np.float32
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert np.array_equal(
        docs,
        FakeEmbedder(dim=64).embed_documents(
            ["capital expenditure 2018", "capital expenditure", "net sales"]
        ),
    )
    q = emb.embed_query("capital expenditure")
    assert q @ docs[0] > q @ docs[2]  # shared words -> closer


def test_second_pass_is_all_hits_and_vectors_match(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    texts = ["alpha beta", "gamma", "alpha beta", "delta"]  # one duplicate
    v1, s1 = embed_with_cache(emb, texts, cache, batch_size=2)
    assert (s1.hits, s1.misses) == (0, 4)
    assert emb.calls == 3  # the duplicate is embedded once
    v2, s2 = embed_with_cache(emb, texts, cache)
    assert (s2.hits, s2.misses) == (4, 0) and emb.calls == 3
    assert np.array_equal(v1, v2)
    assert np.array_equal(v1[0], v1[2])
    assert len(cache) == 3

    _, s3 = embed_with_cache(emb, [*texts, "new text"], cache)
    assert (s3.hits, s3.misses) == (4, 1) and emb.calls == 4
    cache.close()
    assert len(EmbeddingCache(tmp_path / "c.sqlite")) == 4  # persisted on disk


def test_keys_depend_on_model_and_empty_input() -> None:
    assert cache_key("m1", "text") != cache_key("m2", "text")
    vecs, stats = embed_with_cache(FakeEmbedder(dim=8), [], None)
    assert vecs.shape == (0, 8) and stats.misses == 0


def test_factory_and_slug() -> None:
    assert isinstance(build_embedder(EmbeddingCfg(model="fake")), FakeEmbedder)
    assert model_slug("BAAI/bge-small-en-v1.5") == "bge-small-en-v1-5"
