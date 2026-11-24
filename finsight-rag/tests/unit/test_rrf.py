"""Reciprocal Rank Fusion: math, ties, duplicates, per-stage scores."""

from __future__ import annotations

import pytest
from _helpers import rc

from finsight.retrieval.hybrid import rrf_fuse


def test_rrf_scores_and_order() -> None:
    dense = [rc("a", 1), rc("b", 2), rc("c", 3)]
    sparse = [rc("c", 1, "sparse", 9.0), rc("a", 2, "sparse", 5.0)]
    fused = rrf_fuse([dense, sparse], k=60)
    assert [f.chunk.chunk_id for f in fused] == ["a", "c", "b"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0].scores_by_stage == {"dense": 1.0, "sparse": 5.0, "rrf": fused[0].score}
    assert [f.rank for f in fused] == [1, 2, 3] and {f.stage for f in fused} == {"hybrid"}


def test_ties_break_by_chunk_id_and_duplicates_count_once() -> None:
    fused = rrf_fuse([[rc("z", 1)], [rc("y", 1)]])
    assert [f.chunk.chunk_id for f in fused] == ["y", "z"]  # equal score: deterministic order
    dup = rrf_fuse([[rc("a", 1), rc("a", 2)]])
    assert len(dup) == 1 and dup[0].score == pytest.approx(1 / 61)


def test_empty_inputs() -> None:
    assert rrf_fuse([]) == [] and rrf_fuse([[], []]) == []
