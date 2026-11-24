"""Query-embedding and reranker caches: a second run embeds/scores nothing and is identical.

These lock in the fix for 'embedding happens again and again' and 'rerank does not reproduce':
document vectors are skipped by the index manifest (tested in test_index_roundtrip), query vectors
and cross-encoder scores are cached here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _helpers import rc

from finsight.config import RerankCfg
from finsight.embedding.cache import EmbeddingCache
from finsight.embedding.fake import FakeEmbedder
from finsight.retrieval.reranker import LocalCrossEncoderReranker, ScoreCache, build_reranker


class CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that records how many texts it actually embedded."""

    def __init__(self) -> None:
        super().__init__(dim=32)
        self.query_cache: EmbeddingCache | None = None
        self.embedded = 0

    def _encode(self, texts: list[str]) -> np.ndarray:
        self.embedded += len(texts)
        return self.embed_documents(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        from finsight.embedding.cache import cached_query_vectors

        vecs, _ = cached_query_vectors(self.query_cache, "m|query|{text}", texts, self._encode, 32)
        return vecs


def test_query_cache_avoids_recompute_and_is_identical(tmp_path: Path) -> None:
    emb = CountingEmbedder()
    emb.query_cache = EmbeddingCache(tmp_path / "q.sqlite")
    v1 = emb.embed_queries(["capex 2018", "revenue"])
    assert emb.embedded == 2
    v2 = emb.embed_queries(["capex 2018", "revenue"])  # second run: no embedding
    assert emb.embedded == 2 and np.array_equal(v1, v2)
    # a fresh embedder sharing the same cache file also recomputes nothing
    emb2 = CountingEmbedder()
    emb2.query_cache = EmbeddingCache(tmp_path / "q.sqlite")
    assert np.array_equal(emb2.embed_queries(["revenue"]), v1[1:2]) and emb2.embedded == 0


class CountingReranker(LocalCrossEncoderReranker):
    def __init__(self, cache: ScoreCache) -> None:
        super().__init__("fake-reranker", cache=cache)
        self.scored = 0

    @property
    def model(self):  # type: ignore[override]
        parent = self

        class _M:
            def predict(self, pairs, **kw):
                parent.scored += len(pairs)
                # deterministic score: longer chunk text ranks higher
                return [float(len(p[1])) for p in pairs]

        return _M()


def test_reranker_cache_avoids_recompute_and_is_reproducible(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path / "rr.sqlite", "fake-reranker")
    r = CountingReranker(cache)
    cands = [
        rc("a", 1, text="short"),
        rc("b", 2, text="a much longer chunk body"),
        rc("c", 3, text="mid"),
    ]
    first = [x.chunk.chunk_id for x in r.rerank("q", cands, top_n=3)]
    assert r.scored == 3 and first[0] == "b"  # longest text first
    second = [x.chunk.chunk_id for x in r.rerank("q", cands, top_n=3)]
    assert r.scored == 3 and second == first  # nothing re-scored, same order
    r2 = CountingReranker(ScoreCache(tmp_path / "rr.sqlite", "fake-reranker"))
    assert [x.chunk.chunk_id for x in r2.rerank("q", cands, top_n=3)] == first and r2.scored == 0


def test_build_reranker_attaches_cache(tmp_path: Path) -> None:
    r = build_reranker(RerankCfg(provider="local"), device="cpu", cache_dir=tmp_path)
    assert isinstance(r, LocalCrossEncoderReranker) and r.cache is not None
    assert build_reranker(RerankCfg(provider="local"), device="cpu").cache is None
