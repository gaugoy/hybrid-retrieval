"""
Sparse (lexical) index: BM25 over chunk.embed_text, with bm25s.

BM25 IN ONE LINE
  score(q, d) = Σ_w IDF(w) · tf(w,d)·(k1+1) / (tf(w,d) + k1·(1 - b + b·|d|/avgdl))
  Rare words (high IDF) matter more; repeating a word helps with diminishing returns (k1);
  long chunks are penalized (b). It matches EXACT words: tickers, line-item names, years, which
  dense embeddings tend to blur. That is why hybrid search (P5) combines the two.

TOKENIZATION (what BM25 actually sees)
  bm25s.tokenize: lowercase, words of ≥ 2 letters/digits, English stopwords removed, no stemming.
  "Purchases of property, plant and equipment (PP&E) | (1,577)" ->
      ['purchases', 'property', 'plant', 'equipment', 'pp', '577']
  Note "1,577" becomes "577": numbers are split at the comma. Questions rarely contain the
  answer's digits, so this hardly matters for retrieval; years ("2018") survive intact.
  Queries are tokenized the same way and looked up in the index's own vocabulary; unknown
  words simply score 0.

ON DISK  indexes/bm25/<strategy>_<size>/  = bm25s files + chunk_ids.txt (row i = document i)

FILTERS
  BM25 has no metadata, so `search` takes an optional boolean mask over documents (True = may
  be returned). bm25s applies it before ranking, so k matching results come back. P5's sparse
  retriever builds the mask from the chunk table and QueryFilters.

More info: https://github.com/xhluca/bm25s · https://en.wikipedia.org/wiki/Okapi_BM25
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np

STOPWORDS = "en"
CHUNK_IDS_FILE = "chunk_ids.txt"


def tokenize(texts: list[str]) -> list[list[str]]:
    """The exact tokenization used for both the corpus and the queries."""
    out = bm25s.tokenize(texts, stopwords=STOPWORDS, return_ids=False, show_progress=False)
    return out  # type: ignore[return-value]


@dataclass
class SparseHit:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, chunk_ids: list[str]) -> None:
        self.retriever = retriever
        self.chunk_ids = chunk_ids

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str]) -> BM25Index:
        retriever = bm25s.BM25()  # k1=1.5, b=0.75 (Lucene variant)
        retriever.index(
            bm25s.tokenize(texts, stopwords=STOPWORDS, show_progress=False), show_progress=False
        )
        return cls(retriever, list(chunk_ids))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory), show_progress=False)
        (directory / CHUNK_IDS_FILE).write_text("\n".join(self.chunk_ids) + "\n")

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        retriever = bm25s.BM25.load(str(directory), show_progress=False)
        chunk_ids = (directory / CHUNK_IDS_FILE).read_text().splitlines()
        return cls(retriever, chunk_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.retriever.vocab_dict)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, k: int, mask: np.ndarray | None = None) -> list[SparseHit]:
        """Top-k chunks by BM25 score (zero-score results dropped). `mask`: allowed documents."""
        tokens = tokenize([query])
        if not tokens[0] or not self.chunk_ids:
            return []
        k = min(k, len(self.chunk_ids))
        weight_mask = None if mask is None else np.asarray(mask, dtype=np.float32)
        docs, scores = self.retriever.retrieve(
            tokens, k=k, show_progress=False, weight_mask=weight_mask, n_threads=1
        )
        return [
            SparseHit(self.chunk_ids[int(d)], float(s))
            for d, s in zip(docs[0], scores[0], strict=True)
            if s > 0
        ]
