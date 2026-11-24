"""Dense retrieval: embed the query, nearest neighbours in Qdrant (filters applied in-search)."""

from __future__ import annotations

from finsight.embedding.base import Embedder
from finsight.indexing.qdrant_store import QdrantStore
from finsight.retrieval.base import Retriever, to_chunk
from finsight.schemas import QueryFilters, RetrievedChunk


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, store: QdrantStore, embedder: Embedder, collection: str) -> None:
        self.store, self.embedder, self.collection = store, embedder, collection

    def retrieve(
        self, query: str, k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]:
        hits = self.store.search(self.collection, self.embedder.embed_query(query), k, filters)
        return [
            RetrievedChunk(
                chunk=to_chunk(h.payload),
                score=h.score,
                rank=i + 1,
                stage="dense",
                scores_by_stage={"dense": h.score},
            )
            for i, h in enumerate(hits)
        ]
