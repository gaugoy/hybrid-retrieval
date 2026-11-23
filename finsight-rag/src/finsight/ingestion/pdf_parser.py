"""
PDF page → lines of text that keep table rows together.

THE PROBLEM
  A PDF has no "rows" or "paragraphs", only glyphs placed at x/y coordinates. PyMuPDF's plain
  `page.get_text("text")` emits text block by block, and in financial statements every number is
  its own block. So "Purchases of property, plant and equipment" and "(1,577)" end up on different
  lines, far apart. A chunker, an embedding model or an LLM then can't tell which number belongs
  to which label.

WHAT WE DO INSTEAD (layout-aware line reconstruction)
  1. `page.get_text("words")` gives every word with its bounding box (≈3 ms/page).
  2. Group words into visual ROWS: words whose vertical centres are within
     `row_tolerance_factor × line height` of each other sit on the same row.
  3. Sort each row left → right. A horizontal gap wider than `column_gap_factor × line height`
     means a new column, so we insert " | ":
        Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373) | (1,420)
  4. A vertical gap larger than `paragraph_gap_factor × line height` becomes a blank line,
     which keeps paragraph boundaries for the chunker (P3).

WHY NOT page.find_tables()?
  Measured on 3M_2018_10K page 59: 1.5 s per page (≈5 h for our 12,013 pages), and its cells lost
  their spaces ("CashFlowsfromOperatingActivities") and split numbers ("5363\\n,"). Row
  reconstruction is ~500× faster and keeps the text intact. Trade-off: no explicit cell
  structure, and two-column prose can merge into one row (rare in SEC filings).

More info: https://pymupdf.readthedocs.io/en/latest/app1.html (text extraction flavours)
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from finsight.config import IngestionCfg


@dataclass(frozen=True)
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class RawPage:
    """One parsed page before cleaning. ``lines`` may contain "" for paragraph breaks."""

    page_idx: int  # 0-based
    lines: list[str] = field(default_factory=list)
    n_words: int = 0


def extract_words(page: pymupdf.Page) -> list[Word]:
    """Every word on the page with its box. PyMuPDF splits words on whitespace."""
    out = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        text = text.strip()
        if text:
            out.append(Word(x0, y0, x1, y1, text))
    return out


def group_rows(words: list[Word], tolerance_factor: float = 0.5) -> list[list[Word]]:
    """Cluster words into visual rows (top → bottom), each sorted left → right."""
    if not words:
        return []
    line_height = statistics.median(w.height for w in words) or 1.0
    tol = tolerance_factor * line_height
    rows: list[list[Word]] = []
    row_center = 0.0
    for w in sorted(words, key=lambda w: (w.y_center, w.x0)):
        if rows and abs(w.y_center - row_center) <= tol:
            rows[-1].append(w)
            # running mean keeps slightly baseline-shifted words (e.g. "$") in the row
            row_center += (w.y_center - row_center) / len(rows[-1])
        else:
            rows.append([w])
            row_center = w.y_center
    return [sorted(r, key=lambda w: w.x0) for r in rows]


CURRENCY = {"$", "€", "£", "¥"}


def render_row(row: list[Word], column_gap: float) -> str:
    """Join a row's words with spaces, or " | " where the gap marks a new column.

    Currency symbols belong to the number AFTER them. In statements the "$" is left-aligned in
    its column, far from its number but close to the previous column's number, so gaps alone would
    give "$ | 5,363 $ | 4,869". The rules below give "| $ 5,363 | $ 4,869".
    """
    parts = [row[0].text]
    for prev, cur in itertools.pairwise(row):
        gap = cur.x0 - prev.x1
        if prev.text in CURRENCY:
            sep = " "  # "$" always attaches to the number that follows
        elif cur.text in CURRENCY:
            sep = " | " if gap > 0.25 * column_gap else " "  # a "$" after a value opens a new cell
        else:
            sep = " | " if gap > column_gap else " "
        parts.append(sep)
        parts.append(cur.text)
    return "".join(parts)


def page_to_lines(words: list[Word], cfg: IngestionCfg) -> list[str]:
    """Rows → text lines, with "" inserted at paragraph gaps."""
    rows = group_rows(words, cfg.row_tolerance_factor)
    if not rows:
        return []
    line_height = statistics.median(w.height for w in words) or 1.0
    lines: list[str] = []
    prev_bottom: float | None = None
    for row in rows:
        top = min(w.y0 for w in row)
        if prev_bottom is not None and top - prev_bottom > cfg.paragraph_gap_factor * line_height:
            lines.append("")
        lines.append(render_row(row, cfg.column_gap_factor * line_height))
        prev_bottom = max(w.y1 for w in row)
    return lines


def parse_pdf(path: Path, cfg: IngestionCfg) -> list[RawPage]:
    """Parse every page. Page order and count match the PDF exactly (empty pages kept)."""
    pages = []
    with pymupdf.open(path) as doc:
        for idx, page in enumerate(doc):
            words = extract_words(page)
            pages.append(RawPage(page_idx=idx, lines=page_to_lines(words, cfg), n_words=len(words)))
    return pages


def plain_text(path: Path, page_idx: int) -> str:
    """PyMuPDF's default extraction for one page: the baseline we compare against."""
    with pymupdf.open(path) as doc:
        return str(doc[page_idx].get_text("text"))
