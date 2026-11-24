"""
Reference-based answer metrics (no LLM needed).

numeric_match(pred, gold)   FinanceBench "metrics-generated" answers are numbers ("$1577.00",
                            "24.26%", "(0.13)"). Every number in both strings is parsed
                            ($, commas, %, "(1,234)" = negative, million/billion/thousand); the
                            match is true if ANY predicted number is within rel_tol (1%) of the
                            gold's main (first) number, either as written or after applying the
                            unit words (so "1.58 billion" matches "$1,577 million").
contains_match(pred, gold)  normalized substring match for short text answers ("Yes", "No",
                            "Sam Bankman-Fried").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUM = re.compile(
    r"(?P<neg>\()?\s*(?P<sign>-)?\$?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*\)?"
    r"\s*(?P<pct>%)?\s*(?P<unit>thousand|million|billion|trillion|mn|bn|m|b|k)?\b",
    re.I,
)
_MULT = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
}


@dataclass(frozen=True)
class Number:
    value: float  # as written (sign applied)
    scaled: float  # with unit words applied
    pct: bool


def parse_numbers(text: str) -> list[Number]:
    out = []
    for m in _NUM.finditer(text.replace("\u2212", "-")):
        raw = float(m.group("num").replace(",", ""))
        neg = bool(m.group("sign")) or (bool(m.group("neg")) and ")" in m.group(0))
        v = -raw if neg else raw
        unit = (m.group("unit") or "").lower()
        out.append(Number(v, v * _MULT.get(unit, 1.0), bool(m.group("pct"))))
    return out


def _close(a: float, b: float, rel_tol: float) -> bool:
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= rel_tol


def numeric_match(pred: str, gold: str, rel_tol: float = 0.01) -> bool | None:
    """True/False, or None when the gold answer has no number (not a numeric question)."""
    golds = parse_numbers(gold)
    if not golds:
        return None
    g = golds[0]
    for p in parse_numbers(pred):
        for pv in (p.value, p.scaled, abs(p.value)):
            for gv in (g.value, g.scaled):
                if _close(pv, gv, rel_tol):
                    return True
    return False


def contains_match(pred: str, gold: str) -> bool:
    words = re.sub(r"[^a-z0-9 ]+", " ", gold.lower()).split()
    hay = " " + " ".join(re.sub(r"[^a-z0-9 ]+", " ", pred.lower()).split()) + " "
    return bool(words) and f" {' '.join(words)} " in hay
