"""Row reconstruction, cleaning and quality metrics on synthetic inputs (no real PDFs)."""

from __future__ import annotations

import pytest

from finsight.config import IngestionCfg
from finsight.ingestion.cleaner import (
    clean_page,
    find_boilerplate,
    join_hyphenation,
    normalize_line,
)
from finsight.ingestion.pdf_parser import Word, group_rows, page_to_lines, render_row
from finsight.ingestion.quality import (
    count_numbers,
    count_table_rows,
    evidence_overlap,
    label_number_adjacency,
)

CFG = IngestionCfg()


def w(x0: float, y0: float, text: str, width: float = 30, height: float = 10) -> Word:
    return Word(x0, y0, x0 + width, y0 + height, text)


# ---------------------------------------------------------------- rows


def test_group_rows_clusters_by_vertical_position_and_sorts_left_to_right() -> None:
    words = [w(400, 100.5, "1,577"), w(72, 100, "Capex"), w(72, 130, "Next"), w(110, 101, "spend")]
    rows = group_rows(words, tolerance_factor=0.5)
    assert [[x.text for x in r] for r in rows] == [["Capex", "spend", "1,577"], ["Next"]]


def test_render_row_inserts_column_separator_on_wide_gaps() -> None:
    row = [w(72, 0, "Net", 20), w(95, 0, "sales", 25), w(400, 0, "11,303", 30)]
    assert render_row(row, column_gap=15) == "Net sales | 11,303"


def test_render_row_attaches_currency_to_following_number() -> None:
    # "$" left-aligned in its column: far from its number, close to the previous number
    row = [
        w(72, 0, "Net", 20),
        w(95, 0, "income", 30),
        w(300, 0, "$", 5),
        w(360, 0, "5,363", 25),
        w(390, 0, "$", 5),
        w(450, 0, "4,869", 25),
    ]
    assert render_row(row, column_gap=15) == "Net income | $ 5,363 | $ 4,869"


def test_page_to_lines_marks_paragraph_gaps() -> None:
    words = [w(72, 100, "First"), w(72, 112, "line"), w(72, 160, "New"), w(110, 160, "para")]
    assert page_to_lines(words, CFG) == ["First", "line", "", "New para"]


def test_empty_page_gives_no_lines() -> None:
    assert page_to_lines([], CFG) == []


# ---------------------------------------------------------------- cleaning


def test_normalize_line_maps_digits() -> None:
    assert (
        normalize_line("  Page 12  of 160 ") == normalize_line("Page 13 of 160") == "page ## of ###"
    )


def test_find_boilerplate_only_at_page_edges_and_repeated() -> None:
    pages = [
        ["ACME Corp 10-K", "Total | 1", "body text", "more body", "Page 1"],
        ["ACME Corp 10-K", "Total | 2", "other body", "more body", "Page 2"],
        ["ACME Corp 10-K", "Revenue | 3", "body", "more body", "Page 3"],
        ["Something else", "Total | 4", "x", "more body", "Page 4"],
    ]
    found = find_boilerplate(pages, edge_lines=1, min_share=0.5)
    assert "acme corp ##-k" in found  # header on 3/4 pages
    assert "page #" in found  # footer page numbers on 4/4
    assert "total | #" not in found  # repeated, but not at a page edge


def test_long_repeated_sentence_is_not_boilerplate() -> None:
    sentence = (
        "The following discussion should be read together with the consolidated financial "
        "statements and notes below"
    )
    pages = [[sentence, f"body {i}", "x", "y", "Page 1"] for i in range(4)]
    found = find_boilerplate(pages, edge_lines=1, min_share=0.5)
    assert "page #" in found
    assert normalize_line(sentence) not in found  # > MAX_BOILERPLATE_WORDS words: content


def test_clean_page_removes_boilerplate_but_keeps_body_repeats() -> None:
    lines = ["ACME Corp 10-K", "Total | 1", "", "", "Total | 1", "Page 9"]
    text = clean_page(lines, {"acme corp ##-k", "page #"}, edge_lines=1)
    assert text == "Total | 1\n\nTotal | 1"  # body kept, blank lines collapsed to one


def test_join_hyphenation_only_before_lowercase() -> None:
    assert join_hyphenation(["capital expendi-", "tures were high"]) == [
        "capital expenditures were high"
    ]
    assert join_hyphenation(["Coca-", "Cola Company"]) == ["Coca-", "Cola Company"]  # proper noun
    assert join_hyphenation(["a | b-", "c"]) == ["a | b-", "c"]  # never inside table rows


# ---------------------------------------------------------------- quality


def test_evidence_overlap_ignores_missing_spaces() -> None:
    evidence = "Netsales 11,303 Costofsales 7,772"  # how FinanceBench's old extractor wrote it
    page = "Net sales | $ 11,303\nCost of sales | 7,772"
    assert evidence_overlap(evidence, page) == 1.0
    assert evidence_overlap("Operating income 999", page) == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Net sales | $ 11,303 | $ 11,503", 1.0),  # numbers next to their label
        ("Net sales\n$\n11,303\n$\n11,503", 0.0),  # plain extraction: numbers alone
        ("no numbers here", 1.0),
    ],
)
def test_label_number_adjacency(text: str, expected: float) -> None:
    assert label_number_adjacency(text) == expected


def test_count_table_rows() -> None:
    text = (
        "Header line\nNet sales | 11,303 | 11,503\nCost | (7,772)\nPlain sentence with 2020 in it"
    )
    assert count_table_rows(text) == 2


def test_count_numbers() -> None:
    assert count_numbers("Net sales | $ 11,303 | (1,577) | 12.5% | FY2020") == 3
