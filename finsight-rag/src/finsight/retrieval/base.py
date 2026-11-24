"""Retriever and Reranker interfaces (contract §4.2), plus the chunk lookup they share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from finsight.schemas import Chunk, QueryFilters, RetrievedChunk


class Retriever(ABC):
    name: str

    @abstractmethod
    def retrieve(
        self, query: str, k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]: ...


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class ChunkStore:
    """chunk_id -> Chunk, from the chunks parquet (and parents for parent_child)."""

    def __init__(self, chunks_df: pd.DataFrame, parents_df: pd.DataFrame | None = None) -> None:
        self.df = chunks_df.reset_index(drop=True)
        self._pos = {cid: i for i, cid in enumerate(self.df["chunk_id"])}
        self._parents = (
            {r["chunk_id"]: r for r in parents_df.to_dict("records")}
            if parents_df is not None
            else {}
        )

    @classmethod
    def from_files(cls, chunks: Path, parents: Path | None = None) -> ChunkStore:
        pdf = pd.read_parquet(parents) if parents and parents.is_file() else None
        return cls(pd.read_parquet(chunks), pdf)

    def __len__(self) -> int:
        return len(self.df)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._pos

    def get(self, chunk_id: str) -> Chunk:
        return to_chunk(self.df.iloc[self._pos[chunk_id]].to_dict())

    def parent(self, parent_id: str | None) -> Chunk | None:
        row = self._parents.get(parent_id) if parent_id else None
        return to_chunk(row) if row is not None else None

    def mask(self, filters: QueryFilters | None) -> list[bool] | None:
        """Boolean mask in chunk-table order (= BM25 document order) for the filters."""
        if filters is None:
            return None
        m = pd.Series(True, index=self.df.index)
        if filters.companies:
            m &= self.df["company"].isin(filters.companies)
        if filters.doc_periods:
            m &= self.df["doc_period"].isin(filters.doc_periods)
        if filters.doc_types:
            m &= self.df["doc_type"].isin(filters.doc_types)
        return m.tolist()


def to_chunk(record: dict) -> Chunk:
    clean = {k: (None if isinstance(v, float) and v != v else v) for k, v in record.items()}
    clean = {k: v for k, v in clean.items() if k in Chunk.model_fields}
    for key in ("page_start", "page_end", "doc_period", "n_tokens"):
        clean[key] = int(clean[key])  # type: ignore[arg-type]
    return Chunk(**clean)
