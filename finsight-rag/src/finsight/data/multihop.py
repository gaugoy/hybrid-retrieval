"""
MultiHop-RAG (Tang & Yang, COLM 2024; ODC-BY): 2,556 questions over 609 news articles (2023).

Files (Hugging Face dataset yixuantt/MultiHopRAG): corpus.json (title, author, source,
published_at, category, url, body) and MultiHopRAG.json (query, answer, question_type,
evidence_list[{title, url, source, fact, ...}]). Question types: inference 816, comparison 856,
temporal 583, null 301 (answer "Insufficient information." -> gold_answer=None: the right
behaviour is to abstain).

Mapping to our schemas: one article = one DocumentMeta (doc_name "mh_<sha1(url)[:10]>",
company = the news source, doc_type "news", doc_period = publication year) with a single Page
(page_idx 0, text = title + body). Evidence = (article, page 0), so the FinanceBench metrics
(relevant = same doc, page in range) work unchanged at article level.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pandas as pd

from finsight.config import Settings
from finsight.errors import DataNotFoundError
from finsight.schemas import DocumentMeta, EvalRecord, Evidence

BASE_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"
FILES = ("corpus.json", "MultiHopRAG.json")


def raw_dir(settings: Settings) -> Path:
    return settings.paths.raw_dir / "multihop"


def doc_id(url: str) -> str:
    return "mh_" + hashlib.sha1(url.encode()).hexdigest()[:10]


def download(settings: Settings, force: bool = False) -> list[Path]:
    out = raw_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FILES:
        p = out / name
        if force or not p.is_file():
            r = httpx.get(f"{BASE_URL}/{name}", follow_redirects=True, timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
        paths.append(p)
    return paths


def _read(settings: Settings, name: str) -> list[dict]:
    p = raw_dir(settings) / name
    if not p.is_file():
        raise DataNotFoundError(f"{p} missing. Run: uv run finsight download --dataset multihop")
    data: list[dict] = json.loads(p.read_text())
    return data


def load_document_meta(settings: Settings) -> dict[str, DocumentMeta]:
    out = {}
    for a in _read(settings, "corpus.json"):
        did = doc_id(a["url"])
        out[did] = DocumentMeta(
            doc_name=did,
            company=a["source"],
            doc_type="news",
            doc_period=int(str(a["published_at"])[:4]),
            source_url=a["url"],
        )
    return out


def load_pages_df(settings: Settings) -> pd.DataFrame:
    rows = []
    for a in _read(settings, "corpus.json"):
        text = f"{a['title']}\n\n{a['body']}"
        rows.append(
            {
                "doc_name": doc_id(a["url"]),
                "page_idx": 0,
                "text": text,
                "n_chars": len(text),
                "has_table": False,
            }
        )
    return pd.DataFrame(rows)


def load_eval_records(settings: Settings) -> list[EvalRecord]:
    out = []
    for i, q in enumerate(_read(settings, "MultiHopRAG.json")):
        null = q["question_type"] == "null_query"
        evidence = [
            Evidence(doc_name=doc_id(e["url"]), page_idx=0, text=e.get("fact", ""))
            for e in q.get("evidence_list", [])
        ]
        out.append(
            EvalRecord(
                qid=f"mh_{i:04d}",
                question=q["query"],
                gold_answer=None if null else q["answer"],
                question_type=q["question_type"],
                company=None,
                doc_name=None,
                evidence=evidence,
                dataset="multihop",
            )
        )
    return out
