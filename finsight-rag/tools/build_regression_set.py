"""
Build tests/fixtures/regression_set.jsonl: 10 FinanceBench questions with their gold chunk and 9
distractors each (4 from the same company's filings, 5 from other companies), taken from the real
page-aware chunks. SEC filing text is public; FinanceBench questions are CC-BY-NC (data card).

The CI regression gate indexes these 100 chunks with FakeEmbedder + BM25 and checks hit@5:
it catches pipeline regressions (fusion, filtering, metrics) without models or data downloads.

Run (after make ingest index):  uv run python tools/build_regression_set.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from finsight.chunking import pipeline as cp
from finsight.config import load_settings
from finsight.data import financebench as fb

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "regression_set.jsonl"


def main() -> None:
    s = load_settings(overrides={"chunking": {"strategy": "page_aware", "chunk_tokens": 384}})
    df = cp.load_chunks_df(s)
    rng = random.Random(11)
    picked = 0
    lines = []
    for rec in fb.load_eval_records(s):
        if picked == 10 or rec.question_type != "metrics-generated":
            continue
        ev = rec.evidence[0]
        gold = df[(df.doc_name == ev.doc_name) & (df.page_start == ev.page_idx)]
        if gold.empty:
            continue
        g = gold.iloc[0]
        same = df[(df.company == g.company) & (df.page_start != ev.page_idx)]
        other = df[df.company != g.company]
        distract = same.sample(4, random_state=rng.randint(0, 10**6)).to_dict(
            "records"
        ) + other.sample(5, random_state=rng.randint(0, 10**6)).to_dict("records")
        cols = [
            "chunk_id",
            "doc_name",
            "company",
            "doc_type",
            "doc_period",
            "page_start",
            "page_end",
            "text",
            "embed_text",
            "n_tokens",
            "strategy",
        ]
        lines.append(
            json.dumps(
                {
                    "qid": rec.qid,
                    "question": rec.question,
                    "company": rec.company,
                    "gold": [{"doc_name": ev.doc_name, "page_idx": ev.page_idx}],
                    "chunks": [
                        {
                            k: (
                                int(v)
                                if k in ("doc_period", "page_start", "page_end", "n_tokens")
                                else v
                            )
                            for k, v in c.items()
                            if k in cols
                        }
                        for c in [g.to_dict(), *distract]
                    ],
                }
            )
        )
        picked += 1
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {picked} questions to {OUT}")


if __name__ == "__main__":
    main()
