"""MultiHop-RAG loader on a 2-article / 2-question fixture (file shapes as downloaded)."""

from __future__ import annotations

import json
from pathlib import Path

from finsight.config import load_settings
from finsight.data import datasets as ds
from finsight.data import multihop


def test_loader_maps_articles_and_null_queries(project_root: Path) -> None:
    s = load_settings("configs/multihop.yaml", root=project_root, env={})
    assert s.dataset == "multihop" and s.paths.chunks_dir.name == "multihop"
    raw = multihop.raw_dir(s)
    raw.mkdir(parents=True)
    corpus = [
        {
            "title": "FTX trial",
            "author": "A",
            "source": "The Verge",
            "published_at": "2023-09-28T12:00:00+00:00",
            "category": "technology",
            "url": "https://v.com/ftx",
            "body": "Sam Bankman-Fried faces trial.",
        },
        {
            "title": "Deals",
            "author": "B",
            "source": "Mashable",
            "published_at": "2023-11-27T08:45:59+00:00",
            "category": "entertainment",
            "url": "https://m.com/deals",
            "body": "Cyber Monday.",
        },
    ]
    queries = [
        {
            "query": "Who is on trial?",
            "answer": "Sam Bankman-Fried",
            "question_type": "inference_query",
            "evidence_list": [
                {
                    "title": "FTX trial",
                    "url": "https://v.com/ftx",
                    "source": "The Verge",
                    "fact": "Sam Bankman-Fried faces trial.",
                }
            ],
        },
        {
            "query": "Unknowable?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
    ]
    (raw / "corpus.json").write_text(json.dumps(corpus))
    (raw / "MultiHopRAG.json").write_text(json.dumps(queries))

    meta = ds.document_meta(s)
    did = multihop.doc_id("https://v.com/ftx")
    assert meta[did].company == "The Verge" and meta[did].doc_period == 2023
    pages = ds.pages_df(s)
    assert list(pages["page_idx"]) == [0, 0] and pages.iloc[0]["text"].startswith("FTX trial")
    recs = ds.eval_records(s)
    assert recs[0].evidence[0].doc_name == did and recs[0].gold_answer == "Sam Bankman-Fried"
    assert recs[1].gold_answer is None and recs[1].question_type == "null_query"
