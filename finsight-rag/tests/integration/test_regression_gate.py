"""CI regression gate: hybrid retrieval with the company filter on 10 frozen FinanceBench
questions (gold + 9 distractor chunks each, tests/fixtures/regression_set.jsonl), FakeEmbedder +
BM25, no downloads. Fails if hit@5 drops below tests/fixtures/regression_thresholds.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from finsight.config import load_settings
from finsight.embedding.fake import FakeEmbedder
from finsight.evaluation.retrieval_metrics import hit_at_k
from finsight.indexing.bm25_store import BM25Index
from finsight.indexing.qdrant_store import QdrantStore, open_client
from finsight.retrieval.base import ChunkStore
from finsight.retrieval.dense import DenseRetriever
from finsight.retrieval.filters import FilterExtractor
from finsight.retrieval.service import RetrievalService
from finsight.retrieval.sparse import SparseRetriever

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def test_hybrid_hit_at_5_meets_threshold(project_root: Path) -> None:
    cases = [json.loads(line) for line in (FIX / "regression_set.jsonl").read_text().splitlines()]
    thresholds = json.loads((FIX / "regression_thresholds.json").read_text())
    rows = {c["chunk_id"]: {**c, "parent_id": None} for case in cases for c in case["chunks"]}
    df = pd.DataFrame(list(rows.values()))
    s = load_settings(
        root=project_root,
        env={},
        overrides={
            "retrieval": {"mode": "hybrid", "filter_fields": ["company"], "candidate_k": 20}
        },
    )
    emb = FakeEmbedder(dim=64)
    client = open_client(s)
    try:
        store = QdrantStore(client)
        store.recreate("gate", emb.dim)
        store.upsert_chunks("gate", df, emb.embed_documents(df["embed_text"].tolist()))
        chunks = ChunkStore(df)
        bm25 = BM25Index.build(df["chunk_id"].tolist(), df["embed_text"].tolist())
        svc = RetrievalService(
            s,
            dense=DenseRetriever(store, emb, "gate"),
            sparse=SparseRetriever(bm25, chunks),
            extractor=FilterExtractor(set(df["company"])),
        )
        hits = []
        for case in cases:
            res = svc.retrieve(case["question"], top_k=5)
            ranked = [(r.chunk.doc_name, r.chunk.page_start, r.chunk.page_end) for r in res.final]
            hits.append(hit_at_k(ranked, [(g["doc_name"], g["page_idx"]) for g in case["gold"]], 5))
    finally:
        client.close()
    score = sum(hits) / len(hits)
    assert len(cases) == 10
    assert score >= thresholds["hybrid_hit@5"], f"regression: hit@5 {score:.2f}"
