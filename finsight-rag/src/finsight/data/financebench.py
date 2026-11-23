"""
FinanceBench: where it lives, and how it becomes our schemas.

WHAT A BENCHMARK IS
  Every retrieval/QA benchmark has three parts:
    corpus     the documents to search (here: 84 SEC filings as PDFs)
    queries    the questions (here: 150 analyst questions)
    judgments  what counts as correct: the gold answer AND the evidence (document + page)
  The evidence lets us score retrieval ("did we fetch the right page?") separately from
  generation ("did we answer correctly?"). That separation is what makes debugging possible.

FILES (verified 2026-11-23, see docs/DEVELOPMENT_GUIDE.md §1)
  financebench_open_source.jsonl           150 Q&A rows, each with an `evidence` list
  financebench_document_information.jsonl  361 rows of doc metadata (360 unique names: one
                                           duplicate, FOOTLOCKER_2023_annualreport, unused by Q&A)
  pdfs/{doc_name}.pdf                      the filings themselves

PAGE CONVENTION
  `evidence_page_num` is 0-based and matches PyMuPDF's page index. We keep it as `page_idx`.

More info: https://arxiv.org/abs/2311.11944 · https://github.com/patronus-ai/financebench
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from finsight.config import Settings
from finsight.errors import DataNotFoundError
from finsight.observability.logging import get_logger
from finsight.schemas import DocumentMeta, EvalRecord, Evidence

log = get_logger(__name__)

REPO_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QA_FILE = "financebench_open_source.jsonl"
DOCS_FILE = "financebench_document_information.jsonl"
QA_URL = f"{REPO_RAW}/data/{QA_FILE}"
DOCS_URL = f"{REPO_RAW}/data/{DOCS_FILE}"
PDF_URL_TEMPLATE = f"{REPO_RAW}/pdfs/{{doc_name}}.pdf"

_DOWNLOAD_HINT = "Run: uv run python scripts/01_download_data.py"


def raw_dir(settings: Settings) -> Path:
    """Folder for the two FinanceBench JSONL files."""
    return settings.paths.raw_dir / "financebench"


def pdf_path(settings: Settings, doc_name: str) -> Path:
    return settings.paths.pdf_dir / f"{doc_name}.pdf"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DataNotFoundError(f"{path} not found. {_DOWNLOAD_HINT}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataNotFoundError(f"{path}:{line_no} is not valid JSON ({exc}).") from exc
    return rows


def referenced_doc_names(qa_rows: Iterable[dict[str, Any]]) -> list[str]:
    """Every document a question points to, via `doc_name` or any evidence item. Sorted."""
    names: set[str] = set()
    for row in qa_rows:
        names.add(row["doc_name"])
        names.update(ev["doc_name"] for ev in row.get("evidence") or [])
    return sorted(names)


def parse_document_meta(rows: Iterable[dict[str, Any]]) -> dict[str, DocumentMeta]:
    """Doc-info rows → {doc_name: DocumentMeta}. On duplicate names, keep the first and warn."""
    out: dict[str, DocumentMeta] = {}
    for row in rows:
        name = row["doc_name"]
        if name in out:
            log.info(
                "duplicate doc_name in document information; keeping first", extra={"doc": name}
            )
            continue
        out[name] = DocumentMeta(
            doc_name=name,
            company=row["company"],
            doc_type=row["doc_type"],
            doc_period=int(row["doc_period"]),
            gics_sector=row.get("gics_sector"),
            source_url=row.get("doc_link"),
        )
    return out


def parse_eval_records(rows: Iterable[dict[str, Any]]) -> list[EvalRecord]:
    """Q&A rows → EvalRecord. Page numbers stay 0-based."""
    records = []
    for row in rows:
        evidence = [
            Evidence(
                doc_name=ev["doc_name"],
                page_idx=int(ev["evidence_page_num"]),
                text=ev.get("evidence_text") or "",
            )
            for ev in row.get("evidence") or []
        ]
        records.append(
            EvalRecord(
                qid=row["financebench_id"],
                question=row["question"],
                gold_answer=row.get("answer"),
                question_type=row.get("question_type") or "unknown",
                question_reasoning=row.get("question_reasoning"),
                company=row.get("company"),
                doc_name=row.get("doc_name"),
                evidence=evidence,
                dataset="financebench",
            )
        )
    return records


def load_document_meta(settings: Settings) -> dict[str, DocumentMeta]:
    return parse_document_meta(read_jsonl(raw_dir(settings) / DOCS_FILE))


def load_eval_records(settings: Settings) -> list[EvalRecord]:
    return parse_eval_records(read_jsonl(raw_dir(settings) / QA_FILE))


def validate_dataset(
    records: list[EvalRecord], meta: dict[str, DocumentMeta], pdf_dir: Path | None = None
) -> list[str]:
    """Consistency checks between questions, doc metadata and (optionally) the PDFs on disk.

    Returns human-readable problems; an empty list means the dataset is consistent.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for rec in records:
        if rec.qid in seen:
            problems.append(f"duplicate question id {rec.qid}")
        seen.add(rec.qid)
        if not (rec.gold_answer or "").strip():
            problems.append(f"{rec.qid}: empty gold answer")
        if not rec.evidence:
            problems.append(f"{rec.qid}: no evidence items")
        for ev in rec.evidence:
            if ev.doc_name not in meta:
                problems.append(f"{rec.qid}: evidence doc {ev.doc_name} has no metadata")
            if pdf_dir is not None and not (pdf_dir / f"{ev.doc_name}.pdf").is_file():
                problems.append(f"{rec.qid}: PDF missing for {ev.doc_name}")
        if (
            rec.doc_name
            and rec.company
            and rec.doc_name in meta
            and meta[rec.doc_name].company != rec.company
        ):
            problems.append(f"{rec.qid}: company mismatch with doc metadata")
    return problems
