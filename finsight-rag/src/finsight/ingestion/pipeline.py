"""
Ingestion pipeline: every referenced PDF → cleaned pages → data/processed/*.parquet.

OUTPUTS
  pages.parquet      one row per PDF page: doc_name, page_idx (0-based), text, n_chars, has_table
  documents.parquet  one row per document: metadata + n_pages, n_empty_pages, n_table_pages, ...
  ingest_manifest.json  what produced these files (PDF checksums, ingestion settings, parser
                        version), so a re-run with identical inputs is skipped

PARALLELISM (workers)
  workers=1 (default) parses documents one after another: ~12k pages in about a minute, and it is
  safe everywhere, including notebooks and cell-based scripts.
  workers>1 spreads documents over processes (ProcessPoolExecutor). On macOS, worker processes
  are started by re-importing the caller's main module; a script without an
  `if __name__ == "__main__":` guard (like our cell-based step scripts) would re-run itself in
  every worker. So only use workers>1 from guarded entry points such as `finsight ingest`.
  A document never spans two workers: boilerplate detection needs all of its pages.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from finsight.config import IngestionCfg, Settings
from finsight.data import financebench as fb
from finsight.errors import DataNotFoundError, FinsightError
from finsight.ingestion.cleaner import clean_document
from finsight.ingestion.pdf_parser import parse_pdf
from finsight.ingestion.quality import count_table_rows
from finsight.observability.logging import get_logger
from finsight.schemas import DocumentMeta, Page

log = get_logger(__name__)

PARSER_VERSION = "rows-v3"  # bump when parsing/cleaning logic changes, to force a rebuild
PAGES_FILE = "pages.parquet"
DOCUMENTS_FILE = "documents.parquet"
MANIFEST_FILE = "ingest_manifest.json"


@dataclass
class IngestReport:
    n_docs: int = 0
    n_pages: int = 0
    n_empty_pages: int = 0
    n_table_pages: int = 0
    total_chars: int = 0
    boilerplate_lines: int = 0
    elapsed_s: float = 0.0
    skipped: bool = False
    failures: dict[str, str] = field(default_factory=dict)
    pages_path: Path | None = None
    documents_path: Path | None = None


def ingest_document(
    pdf_path: Path, meta: DocumentMeta, cfg: IngestionCfg
) -> tuple[list[Page], dict[str, Any]]:
    """Parse + clean one PDF. Top-level function so worker processes can run it."""
    start = time.perf_counter()
    raw_pages = parse_pdf(pdf_path, cfg)
    texts, boilerplate = clean_document([p.lines for p in raw_pages], cfg)
    pages = [
        Page(
            doc_name=meta.doc_name,
            page_idx=raw.page_idx,
            text=text,
            n_chars=len(text),
            has_table=count_table_rows(text) >= cfg.min_table_rows,
        )
        for raw, text in zip(raw_pages, texts, strict=True)
    ]
    stats = {
        **meta.model_dump(),
        "n_pages": len(pages),
        "n_empty_pages": sum(p.n_chars == 0 for p in pages),
        "n_table_pages": sum(p.has_table for p in pages),
        "n_chars": sum(p.n_chars for p in pages),
        "n_boilerplate_lines": len(boilerplate),
        "boilerplate_examples": sorted(boilerplate)[:5],
        "parse_s": round(time.perf_counter() - start, 3),
    }
    return pages, stats


def processed_dir(settings: Settings) -> Path:
    return settings.paths.processed_dir


def _manifest_inputs(settings: Settings, doc_names: list[str]) -> dict[str, Any]:
    """Everything that determines the output: PDF checksums + settings + parser version."""
    dl_manifest = fb.raw_dir(settings) / "download_manifest.json"
    checksums = {}
    if dl_manifest.is_file():
        files = json.loads(dl_manifest.read_text())["files"]
        checksums = {n: files.get(f"raw/pdfs/{n}.pdf", {}).get("sha256") for n in doc_names}
    blob = json.dumps(
        {"pdfs": checksums, "ingestion": settings.ingestion.model_dump(), "parser": PARSER_VERSION},
        sort_keys=True,
    )
    return {
        "inputs_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "parser_version": PARSER_VERSION,
        "ingestion": settings.ingestion.model_dump(),
        "n_docs": len(doc_names),
    }


def _is_up_to_date(settings: Settings, inputs: dict[str, Any]) -> bool:
    out = processed_dir(settings)
    manifest = out / MANIFEST_FILE
    if not (
        manifest.is_file() and (out / PAGES_FILE).is_file() and (out / DOCUMENTS_FILE).is_file()
    ):
        return False
    return bool(json.loads(manifest.read_text()).get("inputs_sha256") == inputs["inputs_sha256"])


def referenced_docs(settings: Settings, limit_docs: int | None = None) -> list[DocumentMeta]:
    """The documents the questions reference (sorted by name), with metadata."""
    meta = fb.load_document_meta(settings)
    records = fb.load_eval_records(settings)
    names = sorted(
        {ev.doc_name for r in records for ev in r.evidence}
        | {r.doc_name for r in records if r.doc_name}
    )
    docs = [meta[n] for n in names if n in meta]
    return docs[:limit_docs] if limit_docs is not None else docs


def ingest_all(
    settings: Settings,
    *,
    limit_docs: int | None = None,
    workers: int | None = None,
    force: bool = False,
    progress: bool = True,
) -> IngestReport:
    """Parse and clean every referenced PDF; write pages.parquet and documents.parquet."""
    start = time.perf_counter()
    report = IngestReport()
    docs = referenced_docs(settings, limit_docs)
    missing = [d.doc_name for d in docs if not fb.pdf_path(settings, d.doc_name).is_file()]
    if missing:
        raise DataNotFoundError(
            f"{len(missing)} PDFs missing (e.g. {missing[0]}). "
            "Run: uv run python scripts/01_download_data.py"
        )

    out = processed_dir(settings)
    inputs = _manifest_inputs(settings, [d.doc_name for d in docs])
    report.pages_path, report.documents_path = out / PAGES_FILE, out / DOCUMENTS_FILE
    if not force and _is_up_to_date(settings, inputs):
        documents = pd.read_parquet(report.documents_path)
        _fill_report(report, documents)
        report.skipped = True
        report.elapsed_s = round(time.perf_counter() - start, 2)
        return report

    all_pages: list[Page] = []
    doc_stats: list[dict[str, Any]] = []

    def collect(doc: DocumentMeta, result: tuple[list[Page], dict[str, Any]] | Exception) -> None:
        if isinstance(result, Exception):  # one bad PDF must not stop the batch
            report.failures[doc.doc_name] = f"{type(result).__name__}: {result}"
            log.warning("pdf parse failed", extra={"doc": doc.doc_name, "error": str(result)})
        else:
            all_pages.extend(result[0])
            doc_stats.append(result[1])

    bar = tqdm(total=len(docs), desc="Parsing PDFs", unit="doc", disable=not progress)
    n_workers = workers if workers is not None else 1
    if n_workers <= 0:
        n_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    if n_workers == 1:
        for d in docs:
            try:
                collect(
                    d, ingest_document(fb.pdf_path(settings, d.doc_name), d, settings.ingestion)
                )
            except Exception as exc:
                collect(d, exc)
            bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    ingest_document, fb.pdf_path(settings, d.doc_name), d, settings.ingestion
                ): d
                for d in docs
            }
            for fut in as_completed(futures):
                try:
                    collect(futures[fut], fut.result())
                except Exception as exc:
                    collect(futures[fut], exc)
                bar.update(1)
    bar.close()
    if not all_pages:
        first = next(iter(report.failures.items()), ("-", "no documents"))
        raise FinsightError(
            f"No pages were parsed ({len(report.failures)} failures). First: {first}"
        )

    out.mkdir(parents=True, exist_ok=True)
    pages_df = pd.DataFrame([p.model_dump() for p in all_pages]).sort_values(
        ["doc_name", "page_idx"]
    )
    documents = pd.DataFrame(doc_stats).sort_values("doc_name").reset_index(drop=True)
    pages_df.to_parquet(report.pages_path, index=False)
    documents.to_parquet(report.documents_path, index=False)
    if not report.failures:  # only a complete run counts as up to date
        manifest = {**inputs, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
        (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    _fill_report(report, documents)
    report.elapsed_s = round(time.perf_counter() - start, 2)
    for _, row in documents[documents["n_empty_pages"] > 0].iterrows():
        log.info(
            "document has empty pages (scanned/image-only?)",
            extra={"doc": row["doc_name"], "empty_pages": int(row["n_empty_pages"])},
        )
    return report


def _fill_report(report: IngestReport, documents: pd.DataFrame) -> None:
    report.n_docs = len(documents)
    report.n_pages = int(documents["n_pages"].sum())
    report.n_empty_pages = int(documents["n_empty_pages"].sum())
    report.n_table_pages = int(documents["n_table_pages"].sum())
    report.total_chars = int(documents["n_chars"].sum())
    report.boilerplate_lines = int(documents["n_boilerplate_lines"].sum())


def load_pages_df(settings: Settings) -> pd.DataFrame:
    path = processed_dir(settings) / PAGES_FILE
    if not path.is_file():
        raise DataNotFoundError(f"{path} not found. Run: uv run python scripts/03_parse_pdfs.py")
    return pd.read_parquet(path)


def load_documents_df(settings: Settings) -> pd.DataFrame:
    path = processed_dir(settings) / DOCUMENTS_FILE
    if not path.is_file():
        raise DataNotFoundError(f"{path} not found. Run: uv run python scripts/03_parse_pdfs.py")
    return pd.read_parquet(path)


def evidence_check(settings: Settings, pages_df: pd.DataFrame) -> pd.DataFrame:
    """One row per FinanceBench evidence item: how well does our parsed page contain it?

    Also finds the best-matching page of the document when the gold page scores below 0.8,
    which tells an off-by-one page problem apart from a parsing problem.
    """
    from finsight.ingestion.quality import evidence_overlap

    pages = {(r.doc_name, r.page_idx): r.text for r in pages_df.itertuples()}
    by_doc: dict[str, list[tuple[int, str]]] = {}
    for (doc, idx), text in pages.items():
        by_doc.setdefault(doc, []).append((idx, text))

    rows = []
    for rec in fb.load_eval_records(settings):
        for ev in rec.evidence:
            text = pages.get((ev.doc_name, ev.page_idx))
            if text is None:  # document not ingested (e.g. --limit-docs)
                continue
            score = evidence_overlap(ev.text, text)
            best_page, best_score = ev.page_idx, score
            if score < 0.8:
                for idx, other in by_doc.get(ev.doc_name, []):
                    s = evidence_overlap(ev.text, other)
                    if s > best_score:
                        best_page, best_score = idx, s
            rows.append(
                {
                    "qid": rec.qid,
                    "doc_name": ev.doc_name,
                    "page_idx": ev.page_idx,
                    "overlap": round(score, 3),
                    "best_page_idx": best_page,
                    "best_overlap": round(best_score, 3),
                    "evidence_tokens": len(ev.text.split()),
                }
            )
    return pd.DataFrame(rows)
