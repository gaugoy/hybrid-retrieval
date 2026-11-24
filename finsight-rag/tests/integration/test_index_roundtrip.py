"""chunks -> Qdrant (embedded, tmp dir) + BM25 -> search, with FakeEmbedder: no model download."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from finsight.chunking import pipeline as cp
from finsight.chunking.tokenizer import WhitespaceTokenizer
from finsight.config import Settings, load_settings
from finsight.data import financebench as fb
from finsight.embedding.fake import FakeEmbedder
from finsight.indexing import pipeline as ixp
from finsight.indexing.qdrant_store import QdrantStore, open_client, point_id
from finsight.schemas import QueryFilters

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "financebench"
TOPICS = {
    0: "Purchases of property plant and equipment capital expenditures | (1,577) | (1,373)",
    1: "Net sales by business segment increased in the year",
    2: "Dividends paid to shareholders and share repurchases",
}


@pytest.fixture
def settings(project_root: Path) -> Settings:
    s = load_settings(
        root=project_root,
        env={},
        overrides={"chunking": {"strategy": "page_aware", "chunk_tokens": 60, "overlap_tokens": 0}},
    )
    fb.raw_dir(s).mkdir(parents=True)
    for name in (fb.QA_FILE, fb.DOCS_FILE):
        shutil.copy(FIXTURES / name, fb.raw_dir(s) / name)
    rows = []
    for doc in ("ACME_2020_10K", "ACME_2021_10K", "BETA_2019_10Q"):
        for i in range(6):
            text = f"{TOPICS[i % 3]} for {doc} page {i}. " + " ".join(
                f"filler{j}" for j in range(20)
            )
            rows.append({"doc_name": doc, "page_idx": i, "text": text, "n_chars": len(text)})
    cp.chunk_corpus(s, tokenizer=WhitespaceTokenizer(), progress=False, pages_df=pd.DataFrame(rows))
    return s


def test_build_search_filter_and_skip(settings: Settings) -> None:
    emb = FakeEmbedder()
    first = ixp.index_corpus(settings, embedder=emb, progress=False)
    n = first.n_chunks
    assert n == 18 and first.qdrant_points == n and first.bm25_docs == n
    assert first.cache_misses == n and not first.dense_skipped and not first.bm25_skipped
    assert first.collection == "financebench__page_aware_60__fake-32"

    client = open_client(settings)
    try:
        store = QdrantStore(client)
        q = emb.embed_query("property plant and equipment capital expenditures")
        hits = store.search(first.collection, q, k=3)
        assert all(h.payload["page_start"] % 3 == 0 for h in hits)  # the capex pages
        assert {"text", "chunk_id", "company", "doc_period"} <= set(hits[0].payload)

        only_beta = store.search(
            first.collection, q, k=10, filters=QueryFilters(companies=["Beta"])
        )
        assert only_beta and {h.payload["doc_name"] for h in only_beta} == {"BETA_2019_10Q"}
        y2021 = store.search(first.collection, q, k=10, filters=QueryFilters(doc_periods=[2021]))
        assert {h.payload["doc_period"] for h in y2021} == {2021}

        point = client.retrieve(first.collection, [point_id(hits[0].payload["chunk_id"])])
        assert point and point[0].payload["chunk_id"] == hits[0].payload["chunk_id"]
    finally:
        client.close()

    bm25 = ixp.load_bm25(settings)
    sparse = bm25.search("dividends repurchases", k=3)
    assert len(sparse) == 3 and all(
        "::p2-2::" in h.chunk_id or "::p5-5::" in h.chunk_id for h in sparse
    )
    assert bm25.search("zzzunknownword", k=3) == []

    again = ixp.index_corpus(settings, embedder=emb, progress=False)
    assert again.dense_skipped and again.bm25_skipped and emb.calls == n

    forced = ixp.index_corpus(settings, embedder=emb, progress=False, force=True)
    assert not forced.dense_skipped and forced.cache_hits == n  # rebuilt from the cache
    assert forced.qdrant_points == n and emb.calls == n


def test_missing_collection_triggers_rebuild(settings: Settings) -> None:
    emb = FakeEmbedder()
    report = ixp.index_corpus(settings, embedder=emb, progress=False, only="dense")
    client = open_client(settings)
    client.delete_collection(report.collection)
    client.close()
    again = ixp.index_corpus(settings, embedder=emb, progress=False, only="dense")
    assert not again.dense_skipped and again.qdrant_points == report.n_chunks


def test_bm25_mask_restricts_results(settings: Settings) -> None:
    ixp.index_corpus(settings, embedder=FakeEmbedder(), progress=False, only="sparse")
    bm25 = ixp.load_bm25(settings)
    mask = [cid.startswith("BETA") for cid in bm25.chunk_ids]
    hits = bm25.search("dividends repurchases", k=5, mask=mask)  # type: ignore[arg-type]
    assert hits and all(h.chunk_id.startswith("BETA") for h in hits)
