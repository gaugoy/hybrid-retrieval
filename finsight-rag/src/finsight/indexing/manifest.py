"""
Index manifest: indexes/manifest.json records what each index was built from.

    {
      "financebench__page_aware_384__bge-small-en-v1-5": {
        "kind": "dense", "chunks_sha256": "...", "embed_model": "BAAI/bge-small-en-v1.5",
        "n_chunks": 31114, "dim": 384, "built_at": "...", "config_fingerprint": "...",
        "version": "v1"
      },
      "bm25__page_aware_384": {...}
    }

IDEMPOTENCY
  An index is rebuilt only when what it was built FROM changed: the chunk content (a hash of
  chunk_id + embed_text), the embedding model, or the index code version. Changing the LLM or
  top_k changes the config fingerprint but not the index, so nothing is rebuilt. The fingerprint
  is stored for traceability only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Keys whose change forces a rebuild (everything else in an entry is informational).
REBUILD_KEYS = ("chunks_sha256", "embed_model", "version")


def chunks_sha256(chunk_ids: Iterable[str], embed_texts: Iterable[str]) -> str:
    """Hash of what gets indexed: ids and the exact text that is embedded / BM25-tokenized."""
    h = hashlib.sha256()
    for cid, text in zip(chunk_ids, embed_texts, strict=True):
        h.update(cid.encode())
        h.update(b"\x00")
        h.update(text.encode())
        h.update(b"\x01")
    return h.hexdigest()


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        data: dict[str, dict[str, Any]] = json.loads(self.path.read_text())
        return data

    def entry(self, name: str) -> dict[str, Any] | None:
        return self.load().get(name)

    def needs_rebuild(self, name: str, **expected: Any) -> bool:
        """True if `name` was never built, or any REBUILD_KEYS value differs from `expected`."""
        entry = self.entry(name)
        if entry is None:
            return True
        return any(entry.get(k) != expected[k] for k in REBUILD_KEYS if k in expected)

    def record(self, name: str, **fields: Any) -> dict[str, Any]:
        data = self.load()
        entry = {**fields, "built_at": datetime.now(UTC).isoformat(timespec="seconds")}
        data[name] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)  # atomic: a crash never leaves a half-written manifest
        return entry

    def remove(self, name: str) -> None:
        data = self.load()
        if data.pop(name, None) is not None:
            self.path.write_text(json.dumps(data, indent=2, sort_keys=True))
