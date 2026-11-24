"""
Retrieval metrics against gold evidence pages.

RELEVANCE
  A retrieved chunk is relevant to an evidence item if it comes from the same document and its
  page range contains the evidence page:  chunk.doc_name == ev.doc_name and
  page_start ≤ ev.page_idx ≤ page_end.  Duplicate (doc, page) evidence items count once.

PER QUESTION (ranked = the retrieved chunks, best first)
  hit@k      1 if any of the top k is relevant to any evidence item
  recall@k   share of the question's evidence pages covered by the top k
  mrr        1 / rank of the first relevant chunk (0 if none is retrieved)
  ndcg@k     binary gains; a chunk gains only if it covers an evidence page not covered by a
             higher-ranked chunk (so 3 chunks of the same gold page don't count 3 times);
             ideal = all evidence pages at the top ranks
  doc_hit@k  the right DOCUMENT is in the top k (separates "wrong company/year" from "right
             filing, wrong page")

More info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Ranked = Sequence[tuple[str, int, int]]  # (doc_name, page_start, page_end), best first
Gold = Iterable[tuple[str, int]]  # (doc_name, page_idx)


def _gold(gold: Gold) -> list[tuple[str, int]]:
    return sorted(set(gold))


def _covers(item: tuple[str, int, int], ev: tuple[str, int]) -> bool:
    doc, start, end = item
    return doc == ev[0] and start <= ev[1] <= end


def hit_at_k(ranked: Ranked, gold: Gold, k: int) -> float:
    g = _gold(gold)
    return float(any(_covers(it, ev) for it in ranked[:k] for ev in g))


def recall_at_k(ranked: Ranked, gold: Gold, k: int) -> float:
    g = _gold(gold)
    if not g:
        return 0.0
    return sum(any(_covers(it, ev) for it in ranked[:k]) for ev in g) / len(g)


def mrr(ranked: Ranked, gold: Gold) -> float:
    g = _gold(gold)
    for rank, it in enumerate(ranked, start=1):
        if any(_covers(it, ev) for ev in g):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Ranked, gold: Gold, k: int) -> float:
    g = _gold(gold)
    if not g:
        return 0.0
    covered: set[tuple[str, int]] = set()
    dcg = 0.0
    for rank, it in enumerate(ranked[:k], start=1):
        new = {ev for ev in g if ev not in covered and _covers(it, ev)}
        if new:
            covered |= new
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(g), k) + 1))
    return dcg / ideal


def doc_hit_at_k(ranked: Ranked, gold: Gold, k: int) -> float:
    docs = {d for d, _ in _gold(gold)}
    return float(any(it[0] in docs for it in ranked[:k]))


def score_question(ranked: Ranked, gold: Gold, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    g = _gold(gold)
    out: dict[str, float] = {"mrr": mrr(ranked, g)}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranked, g, k)
        out[f"recall@{k}"] = recall_at_k(ranked, g, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, g, k)
        out[f"doc_hit@{k}"] = doc_hit_at_k(ranked, g, k)
    return out
