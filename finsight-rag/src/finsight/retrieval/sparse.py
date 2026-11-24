"""Sparse retrieval: BM25 over embed_text; filters as a document mask applied before ranking."""

from __future__ import annotations

import numpy as np

from finsight.indexing.bm25_store import BM25Index
from finsight.retrieval.base import ChunkStore, Retriever
from finsight.schemas import QueryFilters, RetrievedChunk


class SparseRetriever(Retriever):
    name = "sparse"

    def __init__(self, index: BM25Index, chunks: ChunkStore) -> None:
        if index.chunk_ids != chunks.df["chunk_id"].tolist():
            raise ValueError("BM25 index and chunk table are out of sync: rebuild with --force")
        self.index, self.chunks = index, chunks

    def retrieve(
        self, query: str, k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]:
        mask = self.chunks.mask(filters)
        hits = self.index.search(query, k, mask=None if mask is None else np.asarray(mask))
        return [
            RetrievedChunk(
                chunk=self.chunks.get(h.chunk_id),
                score=h.score,
                rank=i + 1,
                stage="sparse",
                scores_by_stage={"sparse": h.score},
            )
            for i, h in enumerate(hits)
        ]
