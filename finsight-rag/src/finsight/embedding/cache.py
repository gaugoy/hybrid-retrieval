"""
Embedding cache: never embed the same text with the same model twice.

    key = sha256(model_name + "\\x00" + text)   ->   float32 vector bytes

A SQLite file per model (indexes/embed_cache/<model_slug>.sqlite). Re-chunking with new settings
creates mostly new texts, but whatever is unchanged (and every re-run after an interruption)
comes from the cache: only the misses go to the model. Misses are written batch by batch, so an
interrupted run keeps the work it finished.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from finsight.embedding.base import Embedder

_SQL_BATCH = 500  # keys per SELECT ... IN (...)


def cache_key(model_name: str, text: str) -> str:
    return hashlib.sha256(f"{model_name}\x00{text}".encode()).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    embed_s: float = 0.0


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), _SQL_BATCH):
            part = keys[i : i + _SQL_BATCH]
            rows = self._db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part
            )
            out.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
            [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()],
        )
        self._db.commit()

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM emb").fetchone()[0])

    def close(self) -> None:
        self._db.close()


def cached_query_vectors(
    cache: EmbeddingCache | None,
    ident: str,
    formatted: list[str],
    embed: Callable[[list[str]], np.ndarray],
    dim: int,
) -> tuple[np.ndarray, int]:
    """Query vectors for `formatted`, reusing `cache` (key = sha256(ident + text)).

    Only texts missing from the cache are embedded, and the misses are stored. Caching queries
    makes re-running an evaluation free AND reproducible: the same cached bytes come back, so a
    non-deterministic backend (e.g. Apple MPS) can't change the ranking between runs. Returns
    (vectors in order, number of cache hits).
    """
    if not formatted:
        return np.zeros((0, dim), dtype=np.float32), 0
    keys = [cache_key(ident, t) for t in formatted]
    found = cache.get_many(list(dict.fromkeys(keys))) if cache is not None else {}
    hits = sum(k in found for k in keys)
    todo = list(dict.fromkeys(k for k in keys if k not in found))
    if todo:
        text_of = dict(zip(keys, formatted, strict=True))
        vecs = embed([text_of[k] for k in todo])
        new = dict(zip(todo, vecs, strict=True))
        found.update(new)
        if cache is not None:
            cache.put_many(new)
    return np.stack([found[k] for k in keys]).astype(np.float32), hits


def embed_with_cache(
    embedder: Embedder,
    texts: list[str],
    cache: EmbeddingCache | None,
    *,
    batch_size: int | None = None,
    progress: bool = False,
) -> tuple[np.ndarray, CacheStats]:
    """Vectors for `texts` in order; only texts missing from the cache are embedded."""
    import time

    stats = CacheStats()
    keys = [cache_key(embedder.model_name, t) for t in texts]
    found = cache.get_many(list(dict.fromkeys(keys))) if cache is not None else {}
    todo = list(dict.fromkeys(k for k in keys if k not in found))  # unique misses, in order
    text_of = dict(zip(keys, texts, strict=True))
    stats.hits = sum(k in found for k in keys)
    stats.misses = len(keys) - stats.hits

    batch_size = batch_size or embedder.preferred_batch
    start = time.perf_counter()
    bar = tqdm(total=len(todo), desc="Embedding", unit="chunk", disable=not progress or not todo)
    for i in range(0, len(todo), batch_size):
        part = todo[i : i + batch_size]
        vecs = embedder.embed_documents([text_of[k] for k in part])
        new = dict(zip(part, vecs, strict=True))
        found.update(new)
        if cache is not None:
            cache.put_many(new)
        bar.update(len(part))
    bar.close()
    stats.embed_s = round(time.perf_counter() - start, 2)

    if not keys:
        return np.zeros((0, embedder.dim), dtype=np.float32), stats
    return np.stack([found[k] for k in keys]).astype(np.float32), stats
