"""
Clean parsed pages: remove repeated headers/footers, re-join hyphenated words, tidy whitespace.

WHY
  Every page of a 10-K repeats things like "Table of Contents", the company name, or a page
  number. Left in, they (a) add the same noise tokens to every chunk, which weakens BM25's IDF and
  pulls embeddings together, and (b) waste LLM context.

HOW BOILERPLATE IS FOUND (per document)
  A line counts as header/footer if, after normalising digits to "#" (so "Page 12" and "Page 13"
  match), it appears in the top or bottom `edge_lines` lines of at least `min_share` of the
  document's pages. Only page edges are considered, so a label that repeats in the body
  ("Total", "Net income") is never removed, and only short lines (≤ MAX_BOILERPLATE_WORDS words)
  qualify: real headers/footers are short ("Table of Contents", "Part II", "60"), while a repeated
  sentence near a page edge is content.

WHAT IS NEVER DROPPED
  Pages. Even an empty page stays, so page_idx keeps matching the PDF and the ground truth.
"""

from __future__ import annotations

import re
from collections import Counter

from finsight.config import IngestionCfg

_DIGITS = re.compile(r"\d")
_SPACES = re.compile(r"[ \t]+")
_HYPHEN_END = re.compile(r"[A-Za-z]-$")
MAX_BOILERPLATE_WORDS = 12


def normalize_line(line: str) -> str:
    """Comparison key for boilerplate detection: lowercase, digits → #, single spaces."""
    return _SPACES.sub(" ", _DIGITS.sub("#", line.lower())).strip()


def _edge_lines(lines: list[str], n: int) -> list[str]:
    content = [ln for ln in lines if ln.strip()]
    if len(content) <= 2 * n:
        return content
    return content[:n] + content[-n:]


def find_boilerplate(pages: list[list[str]], edge_lines: int, min_share: float) -> set[str]:
    """Normalised lines that repeat at the page edges on ≥ min_share of pages."""
    if edge_lines == 0 or len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({normalize_line(ln) for ln in _edge_lines(lines, edge_lines)})
    threshold = max(2, min_share * len(pages))
    return {
        key
        for key, n in counts.items()
        if key and n >= threshold and len(key.split()) <= MAX_BOILERPLATE_WORDS
    }


def join_hyphenation(lines: list[str]) -> list[str]:
    """'expendi-' + 'tures were' → 'expenditures were'. Only when the next line starts lowercase."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if _HYPHEN_END.search(line) and nxt[:1].islower() and " | " not in line:
            head, _, rest = nxt.partition(" ")
            out.append(line[:-1] + head + (" " + rest if rest else ""))
            i += 2
            continue
        out.append(line)
        i += 1
    return out


def clean_page(lines: list[str], boilerplate: set[str], edge_lines: int) -> str:
    """Drop boilerplate at the page edges, fix hyphenation, collapse blank lines."""
    content_idx = [i for i, ln in enumerate(lines) if ln.strip()]
    edge = set(content_idx[:edge_lines] + content_idx[-edge_lines:]) if edge_lines else set()
    kept = [
        _SPACES.sub(" ", ln).strip()
        for i, ln in enumerate(lines)
        if not (i in edge and normalize_line(ln) in boilerplate)
    ]
    kept = join_hyphenation(kept)
    text_lines: list[str] = []
    for ln in kept:
        if ln or (text_lines and text_lines[-1]):  # at most one blank line in a row
            text_lines.append(ln)
    return "\n".join(text_lines).strip()


def clean_document(pages: list[list[str]], cfg: IngestionCfg) -> tuple[list[str], set[str]]:
    """Clean all pages of one document. Returns (page texts, boilerplate lines found)."""
    boilerplate = find_boilerplate(pages, cfg.boilerplate_edge_lines, cfg.boilerplate_min_share)
    return [
        clean_page(lines, boilerplate, cfg.boilerplate_edge_lines) for lines in pages
    ], boilerplate
