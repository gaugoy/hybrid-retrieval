"""
FakeEmbedder: deterministic, dependency-free vectors for tests and CI.

Each word and word pair is hashed (sha1, not Python's salted hash()) to a dimension and a sign;
the counts are L2-normalized. Texts sharing words therefore get similar vectors, which is enough
to test that indexing, search and filters are wired correctly, without downloading a model.
"""

from __future__ import annotations

import hashlib
import itertools
import re

import numpy as np

from finsight.embedding.base import Embedder, l2_normalize

_WORD = re.compile(r"\w+")


class FakeEmbedder(Embedder):
    def __init__(self, dim: int = 32) -> None:
        self.model_name = f"fake-{dim}"
        self.slug = self.model_name
        self.provider = "fake"
        self.dim = dim
        self.calls = 0  # how many texts were embedded (tests check the cache with it)

    def _vector(self, text: str) -> np.ndarray:
        words = _WORD.findall(text.lower())
        grams = words + [f"{a} {b}" for a, b in itertools.pairwise(words)]
        v = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            h = int.from_bytes(hashlib.sha1(g.encode()).digest()[:4], "little")
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        if not v.any():
            v[0] = 1.0
        return v

    def embed_documents(self, texts: list[str], progress: bool = False) -> np.ndarray:
        self.calls += len(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text))
