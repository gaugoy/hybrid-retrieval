"""
Answer cache: the same question, filters, pipeline and LLM on the same config -> the same Answer.

    key = sha256(question, filters, pipeline, llm, top_k, config fingerprint)

The config fingerprint covers everything that changes answers (chunking, embedding, retrieval,
rerank, generation, llm), so changing any of them misses the cache instead of serving stale
answers. Entries expire after api.answer_cache_ttl_s. SQLite, shared by the API's threads.

Query vectors and rerank scores are cached one level down (P7: embedding/cache.py,
retrieval/reranker.py), so even a cache miss skips the repeated embedding work.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from finsight.schemas import Answer


class AnswerCache:
    def __init__(self, path: Path, ttl_s: int, clock: object = time.time) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_s, self._clock = ttl_s, clock
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, created REAL, answer TEXT)"
        )

    @staticmethod
    def key(fingerprint: str, **parts: object) -> str:
        blob = json.dumps({"fp": fingerprint, **parts}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _now(self) -> float:
        return float(self._clock())  # type: ignore[operator]

    def get(self, key: str) -> Answer | None:
        with self._lock:
            row = self._db.execute(
                "SELECT created, answer FROM answers WHERE key = ?", (key,)
            ).fetchone()
        if row is None or self._now() - row[0] > self.ttl_s:
            return None
        return Answer.model_validate_json(row[1])

    def put(self, key: str, answer: Answer) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO answers VALUES (?, ?, ?)",
                (key, self._now(), answer.model_dump_json()),
            )
            self._db.commit()
