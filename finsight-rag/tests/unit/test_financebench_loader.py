"""FinanceBench loaders: schema mapping, page convention, duplicates, consistency checks."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from finsight.config import Settings, load_settings
from finsight.data import financebench as fb
from finsight.errors import DataNotFoundError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "financebench"


@pytest.fixture
def settings(project_root: Path) -> Settings:
    s = load_settings(root=project_root, env={})
    fb.raw_dir(s).mkdir(parents=True)
    for name in (fb.QA_FILE, fb.DOCS_FILE):
        shutil.copy(FIXTURES / name, fb.raw_dir(s) / name)
    return s


def test_eval_records_map_fields_and_keep_zero_based_pages(settings: Settings) -> None:
    records = fb.load_eval_records(settings)
    assert [r.qid for r in records] == ["fb_0001", "fb_0002", "fb_0003"]
    first = records[0]
    assert first.gold_answer == "$1,577.00"
    assert first.question_type == "metrics-generated"
    assert first.question_reasoning == "Information extraction"
    assert first.dataset == "financebench"
    assert first.evidence[0].page_idx == 59  # stored exactly as FinanceBench gives it (0-based)
    assert len(records[1].evidence) == 2
    assert records[1].evidence[1].page_idx == 0  # page 0 is valid (the cover page)


def test_document_meta_keeps_first_duplicate(settings: Settings) -> None:
    meta = fb.load_document_meta(settings)
    assert len(meta) == 4  # 5 rows, one duplicate name
    assert meta["UNUSED_2018_annualreport"].doc_period == 2018  # first row wins
    acme = meta["ACME_2020_10K"]
    assert (acme.company, acme.doc_type, acme.doc_period) == ("Acme", "10k", 2020)
    assert acme.source_url == "https://example.com/acme2020.pdf"


def test_referenced_doc_names_includes_evidence_docs() -> None:
    rows = [
        {"doc_name": "A", "evidence": [{"doc_name": "A"}, {"doc_name": "C"}]},
        {"doc_name": "B", "evidence": []},
    ]
    assert fb.referenced_doc_names(rows) == ["A", "B", "C"]


def test_validate_dataset_clean_without_pdf_check(settings: Settings) -> None:
    records = fb.load_eval_records(settings)
    meta = fb.load_document_meta(settings)
    assert fb.validate_dataset(records, meta) == []


def test_validate_dataset_reports_missing_pdfs_and_metadata(settings: Settings) -> None:
    records = fb.load_eval_records(settings)
    meta = fb.load_document_meta(settings)
    del meta["BETA_2019_10Q"]
    problems = fb.validate_dataset(records, meta, settings.paths.pdf_dir)
    assert any("BETA_2019_10Q has no metadata" in p for p in problems)
    assert sum("PDF missing" in p for p in problems) == 4  # 4 evidence items, no PDFs on disk


def test_missing_file_tells_you_what_to_run(project_root: Path) -> None:
    s = load_settings(root=project_root, env={})
    with pytest.raises(DataNotFoundError, match=r"01_download_data\.py"):
        fb.load_eval_records(s)


def test_corrupt_jsonl_line_is_reported(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\n{not json}\n')
    with pytest.raises(DataNotFoundError, match=r"bad\.jsonl:2"):
        fb.read_jsonl(bad)
