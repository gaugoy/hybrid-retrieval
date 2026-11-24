"""Real-index check (slow): the default indexes are complete and searchable.

Run locally after `make index`:  uv run pytest -m slow

Retrieval QUALITY is measured in P5. Here we only check that the gold page for FinanceBench's
3M capex question is findable when the query uses the filing's own wording (both indexes rank it
#2 for this query). With the question's wording ("capital expenditure") it ranks ~55 dense and
not at all in BM25: the vocabulary gap P5/P7 address (docs/evaluation.md, P4).
"""

from __future__ import annotations

import pytest

from finsight.chunking import pipeline as cp
from finsight.config import load_settings
from finsight.container import Container
from finsight.indexing import pipeline as ixp
from finsight.schemas import QueryFilters

pytestmark = pytest.mark.slow

QUERY = "3M 2018 consolidated statement of cash flows purchases of property, plant and equipment"
GOLD = ("3M_2018_10K", 59)  # FinanceBench evidence page (0-based) for financebench_id_03029


def test_default_indexes_are_complete_and_find_the_gold_page() -> None:
    settings = load_settings()
    if ixp.manifest(settings).entry(ixp.collection_name(settings)) is None:
        pytest.skip("indexes not built yet (run make index)")
    n_chunks = len(cp.load_chunks_df(settings))
    c = Container(settings)
    try:
        assert c.qdrant.count(c.collection) == n_chunks
        assert len(c.bm25) == n_chunks
        dense = c.qdrant.search(
            c.collection,
            c.embedder.embed_query(QUERY),
            k=10,
            filters=QueryFilters(companies=["3M"]),
        )
        dense_pages = [(h.payload["doc_name"], h.payload["page_start"]) for h in dense]
        assert {h.payload["company"] for h in dense} == {"3M"}
        sparse_ids = [h.chunk_id for h in c.bm25.search(QUERY, k=10)]
    finally:
        c.close()
    assert GOLD in dense_pages, dense_pages[:5]
    assert any(cid.startswith(f"{GOLD[0]}::p{GOLD[1]}-") for cid in sparse_ids), sparse_ids[:5]
