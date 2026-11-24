"""Retrieval metrics on hand-built rankings."""

from __future__ import annotations

import math

import pytest

from finsight.evaluation.retrieval_metrics import (
    doc_hit_at_k,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_question,
)

GOLD = [("3M_2018_10K", 59), ("3M_2018_10K", 59), ("3M_2018_10K", 60)]  # duplicate counts once
RANKED = [
    ("AMD_2022_10K", 10, 10),
    ("3M_2018_10K", 59, 59),
    ("3M_2018_10K", 59, 59),  # same gold page again: no extra gain
    ("3M_2018_10K", 58, 60),  # range covers page 60
]


def test_hit_recall_mrr() -> None:
    assert hit_at_k(RANKED, GOLD, 1) == 0.0 and hit_at_k(RANKED, GOLD, 2) == 1.0
    assert recall_at_k(RANKED, GOLD, 2) == 0.5 and recall_at_k(RANKED, GOLD, 4) == 1.0
    assert mrr(RANKED, GOLD) == 0.5
    assert mrr(RANKED[:1], GOLD) == 0.0


def test_ndcg_counts_each_evidence_page_once() -> None:
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, GOLD, 10) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([("3M_2018_10K", 59, 60)], GOLD, 10) == pytest.approx(1 / ideal)


def test_doc_hit_and_empty_cases() -> None:
    assert doc_hit_at_k([("3M_2018_10K", 1, 1)], GOLD, 1) == 1.0
    assert recall_at_k(RANKED, [], 5) == 0.0 and ndcg_at_k(RANKED, [], 5) == 0.0
    s = score_question(RANKED, GOLD, ks=(1, 5))
    assert s["hit@1"] == 0.0 and s["recall@5"] == 1.0 and set(s) >= {"mrr", "ndcg@5", "doc_hit@1"}
