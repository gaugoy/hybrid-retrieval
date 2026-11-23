"""
Measurable parsing quality: does the parsed page contain the evidence, and do numbers stay next
to their labels?

EVIDENCE OVERLAP (does the gold evidence appear on the parsed page?)
  The share of the evidence's word tokens (≥ 3 characters) found on the page. We compare
  space-insensitively: both texts are lowercased, stripped to letters and digits, and each
  evidence token is searched as a substring of the page. This matters because FinanceBench's
  evidence text came from an older extractor that sometimes dropped spaces ("Netsales"), while
  our parser keeps them ("Net sales"). A space-sensitive comparison would wrongly penalise us.

LABEL–NUMBER ADJACENCY (are table rows intact?)
  The share of numeric tokens that sit on a line together with a word of ≥ 3 letters. Plain
  block extraction puts most numbers on lines of their own (low score); row reconstruction keeps
  "Net sales | $ 11,303" together (high score). This is what lets a chunk, an embedding or an LLM
  know which number belongs to which line item.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"^\(?-?\$?\d[\d,]*(\.\d+)?%?\)?$")
_WORD = re.compile(r"[A-Za-z]{3,}")


def compact(text: str) -> str:
    """Lowercase letters and digits only: 'Net sales $ 11,303' → 'netsales11303'."""
    return "".join(_TOKEN.findall(text.lower()))


def evidence_tokens(text: str, min_len: int = 3) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) >= min_len]


def evidence_overlap(evidence_text: str, page_text: str) -> float:
    """Share of evidence tokens found (space-insensitively) in the page. 1.0 = all found."""
    tokens = evidence_tokens(evidence_text)
    if not tokens:
        return 1.0
    haystack = compact(page_text)
    return sum(t in haystack for t in tokens) / len(tokens)


def label_number_adjacency(text: str) -> float:
    """Share of numeric tokens that share a line with a real word. NaN-free: no numbers → 1.0."""
    total = with_label = 0
    for line in text.splitlines():
        tokens = line.replace("|", " ").split()
        n_numbers = sum(bool(_NUMBER.match(t)) for t in tokens)
        if not n_numbers:
            continue
        total += n_numbers
        if _WORD.search(line):
            with_label += n_numbers
    return with_label / total if total else 1.0


def count_numbers(text: str) -> int:
    """How many numeric tokens a text has. Adjacency is only meaningful when there are several."""
    return sum(bool(_NUMBER.match(t)) for t in text.replace("|", " ").split())


def count_table_rows(text: str) -> int:
    """Lines with ≥ 2 ' | ' cells that contain at least one number."""
    return sum(
        1
        for line in text.splitlines()
        if line.count(" | ") >= 1 and any(_NUMBER.match(t) for t in line.replace("|", " ").split())
    )
