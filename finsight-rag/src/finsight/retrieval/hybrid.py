"""
Hybrid retrieval: run dense and sparse, fuse with Reciprocal Rank Fusion.

    RRF(d) = Σ over rankings r of  1 / (k + rank_r(d))        (k = 60, Cormack et al. 2009)

WHY RRF
  BM25 scores (0..~30) and cosine scores (0..1) live on different scales; adding them needs
  per-query normalization that is fragile. RRF uses only RANKS: a chunk ranked high by either
  list rises, one ranked high by both rises most. k damps the advantage of the very top ranks.
Ties are broken by chunk_id so the order is deterministic. Each fused chunk keeps its original
per-stage scores in scores_by_stage (shown in the UI's "under the hood" panel).
"""

from __future__ import annotations

from finsight.retrieval.base import Retriever
from finsight.schemas import QueryFilters, RetrievedChunk


def rrf_fuse(rankings: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    stage_scores: dict[str, dict[str, float]] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rc in ranking:
            cid = rc.chunk.chunk_id
            if cid in seen:  # a duplicate inside one list counts once (its best rank)
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rc.rank)
            best.setdefault(cid, rc)
            stage_scores.setdefault(cid, {}).update(rc.scores_by_stage)
    order = sorted(scores, key=lambda c: (-scores[c], c))
    return [
        RetrievedChunk(
            chunk=best[cid].chunk,
            score=scores[cid],
            rank=i + 1,
            stage="hybrid",
            scores_by_stage={**stage_scores[cid], "rrf": scores[cid]},
        )
        for i, cid in enumerate(order)
    ]


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, dense: Retriever, sparse: Retriever, rrf_k: int = 60) -> None:
        self.dense, self.sparse, self.rrf_k = dense, sparse, rrf_k
        self.last: dict[str, list[RetrievedChunk]] = {}  # per-stage lists of the last call

    def retrieve(
        self, query: str, k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]:
        d = self.dense.retrieve(query, k, filters)
        s = self.sparse.retrieve(query, k, filters)
        fused = rrf_fuse([d, s], self.rrf_k)[:k]
        self.last = {"dense": d, "sparse": s, "hybrid": fused}
        return fused
