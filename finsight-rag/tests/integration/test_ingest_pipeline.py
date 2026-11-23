"""End-to-end ingestion on small generated PDFs that mimic a filing's layout."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pymupdf
import pytest

from finsight.config import Settings, load_settings
from finsight.data import financebench as fb
from finsight.errors import DataNotFoundError
from finsight.ingestion import pipeline as ip
from finsight.ingestion.pdf_parser import parse_pdf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "financebench"

# Fixture questions point at: ACME_2020_10K p59, ACME_2021_10K p40 + p0, BETA_2019_10Q p12.
EVIDENCE_PAGES = {"ACME_2020_10K": {59}, "ACME_2021_10K": {0, 40}, "BETA_2019_10Q": {12}}
EVIDENCE_TEXT = {
    ("ACME_2020_10K", 59): ("Purchases of property, plant and equipment (PP&E)", "(1,577)"),
    ("ACME_2021_10K", 40): ("Operating income 2021", "2,100"),
    ("ACME_2021_10K", 0): ("Net sales 2021", "9,800"),
    ("BETA_2019_10Q", 12): ("Notes due 2020:", "$250"),
}


WORDS = [
    "revenue", "margin", "liquidity", "capital", "pension",
    "leases", "debt", "goodwill", "inventory", "tax",
]  # fmt: skip


def make_filing(path: Path, doc_name: str, n_pages: int, empty_page: int | None = None) -> None:
    """A fake filing: repeated header, page-number footer, varied body text, a table row and a
    hyphenated word. Body text differs per page, as in real filings."""
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page(width=612, height=792)
        if i == empty_page:
            continue  # an image-only/scanned page has no text layer
        topic, other = WORDS[i % len(WORDS)], WORDS[(i * 3 + 1) % len(WORDS)]
        page.insert_text(
            (72, 40), f"{doc_name.split('_')[0]} Corporation Annual Report", fontsize=9
        )
        page.insert_text(
            (72, 120), f"Management discussed {topic} and capital expendi-", fontsize=10
        )
        page.insert_text((72, 133), f"tures and {other} for the period.", fontsize=10)
        label, value = EVIDENCE_TEXT.get(
            (doc_name, i), (f"{other.title()} line item", f"{i * 7:,}")
        )
        page.insert_text((72, 200), label, fontsize=10)
        page.insert_text((400, 200), value, fontsize=10)
        page.insert_text(
            (72, 260), f"Further notes on {topic} follow in section {other}.", fontsize=10
        )
        page.insert_text((300, 770), str(i + 1), fontsize=9)
    doc.save(path)
    doc.close()


@pytest.fixture
def settings(project_root: Path) -> Settings:
    s = load_settings(root=project_root, env={})
    fb.raw_dir(s).mkdir(parents=True)
    for name in (fb.QA_FILE, fb.DOCS_FILE):
        shutil.copy(FIXTURES / name, fb.raw_dir(s) / name)
    s.paths.pdf_dir.mkdir(parents=True)
    for doc, pages in EVIDENCE_PAGES.items():
        make_filing(fb.pdf_path(s, doc), doc, max(pages) + 3, empty_page=1)
    return s


def test_parse_keeps_every_page_including_empty(settings: Settings) -> None:
    raw = parse_pdf(fb.pdf_path(settings, "BETA_2019_10Q"), settings.ingestion)
    assert [p.page_idx for p in raw] == list(range(15))  # 0-based, none dropped
    assert raw[1].lines == [] and raw[1].n_words == 0  # the empty page is still there


def test_ingest_all_writes_clean_pages(settings: Settings) -> None:
    report = ip.ingest_all(settings, progress=False)
    assert report.n_docs == 3 and not report.failures and not report.skipped
    pages = ip.load_pages_df(settings)
    assert len(pages) == 62 + 43 + 15 == report.n_pages
    assert report.n_empty_pages == 3

    page = pages.set_index(["doc_name", "page_idx"]).loc[("ACME_2020_10K", 59), "text"]
    assert "Purchases of property, plant and equipment (PP&E) | (1,577)" in page  # row kept
    assert "capital expenditures and" in page  # hyphenation joined
    assert "Corporation Annual Report" not in page  # repeated header removed
    assert not any(line.strip().isdigit() for line in page.splitlines())  # page number removed

    docs = ip.load_documents_df(settings)
    assert set(docs.columns) >= {"doc_name", "company", "n_pages", "n_empty_pages", "parse_s"}


def test_rerun_is_skipped_and_force_rebuilds(settings: Settings) -> None:
    ip.ingest_all(settings, progress=False)
    assert ip.ingest_all(settings, progress=False).skipped
    assert not ip.ingest_all(settings, progress=False, force=True).skipped
    changed = load_settings(
        root=settings.root, env={}, overrides={"ingestion": {"column_gap_factor": 3.0}}
    )
    assert not ip.ingest_all(changed, progress=False).skipped  # settings changed → rebuild


def test_evidence_check_finds_every_evidence_page(settings: Settings) -> None:
    ip.ingest_all(settings, progress=False)
    ev = ip.evidence_check(settings, ip.load_pages_df(settings))
    assert len(ev) == 4
    assert (ev["overlap"] >= 0.8).all()
    assert (ev["best_page_idx"] == ev["page_idx"]).all()


def test_missing_pdf_names_the_download_step(settings: Settings) -> None:
    fb.pdf_path(settings, "BETA_2019_10Q").unlink()
    with pytest.raises(DataNotFoundError, match=r"01_download_data\.py"):
        ip.ingest_all(settings, progress=False)


def test_pages_parquet_round_trips_schema(settings: Settings) -> None:
    ip.ingest_all(settings, progress=False)
    pages = pd.read_parquet(settings.paths.processed_dir / ip.PAGES_FILE)
    assert list(pages.columns) == ["doc_name", "page_idx", "text", "n_chars", "has_table"]
    assert pages["page_idx"].min() == 0
