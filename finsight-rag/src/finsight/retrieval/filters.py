"""
Rule-based metadata filters from the question text.

    "What is the FY2018 capital expenditure ... for 3M?"  ->  companies=["3M"], doc_periods=[2018]

COMPANY  case-insensitive match of each company's aliases: the name from the document metadata,
         the name without legal suffixes ("AES Corporation" -> "AES"), and a small hand-written
         map of tickers / nicknames (MMM, J&J, AMEX, ...). Longest alias wins on overlap.
YEAR     \\b(?:FY\\s?)?(20\\d{2})\\b
Which fields are APPLIED is config (retrieval.filter_fields, default ["company"]): a FY2017
figure is often answered from the FY2018 10-K, which includes prior-year comparisons, so a hard
year filter can remove the right document. P5 measures both.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from finsight.schemas import QueryFilters

_SUFFIXES = re.compile(
    r"\b(corporation|corp\.?|incorporated|inc\.?|company|co\.?|plc|ltd\.?|group)\s*$", re.I
)
_YEAR = re.compile(r"\b(?:FY\s?)?(20\d{2})\b", re.I)

EXTRA_ALIASES: dict[str, list[str]] = {
    "3M": ["mmm", "3m company"],
    "Johnson & Johnson": ["j&j", "jnj", "johnson and johnson"],
    "American Express": ["amex"],
    "JPMorgan": ["jp morgan", "jpmorgan chase", "jpm"],
    "Coca-Cola": ["coca cola", "coke"],
    "PG&E Corporation": ["pg&e", "pacific gas and electric"],
    "McDonalds": ["mcdonald's", "mcdonald"],
    "Paypal": ["paypal holdings"],
    "Block": ["block inc", "square"],
    "Activision Blizzard": ["activision"],
    "Kraft Heinz": ["kraft"],
    "MGM Resorts": ["mgm"],
    "CVS Health": ["cvs"],
    "Ulta Beauty": ["ulta"],
    "Lockheed Martin": ["lockheed"],
    "Foot Locker": ["footlocker"],
    "AMD": ["advanced micro devices"],
}


def build_aliases(companies: Iterable[str]) -> dict[str, str]:
    """alias (lowercase) -> canonical company name."""
    out: dict[str, str] = {}
    for name in companies:
        for alias in {name, _SUFFIXES.sub("", name).strip(), *EXTRA_ALIASES.get(name, [])}:
            if len(alias) >= 2:
                out[alias.lower()] = name
    return out


class FilterExtractor:
    def __init__(self, companies: Iterable[str]) -> None:
        aliases = build_aliases(companies)
        # longest first, so "american express" beats a shorter overlapping alias
        self._patterns = [
            (re.compile(rf"(?<![\w&]){re.escape(a)}(?![\w&])", re.I), c)
            for a, c in sorted(aliases.items(), key=lambda kv: -len(kv[0]))
        ]

    def extract(self, question: str) -> QueryFilters:
        companies: list[str] = []
        taken: list[tuple[int, int]] = []
        for pattern, company in self._patterns:
            for m in pattern.finditer(question):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append(m.span())
                if company not in companies:
                    companies.append(company)
        years = sorted({int(y) for y in _YEAR.findall(question)})
        return QueryFilters(companies=companies, doc_periods=years)


def apply_fields(filters: QueryFilters, fields: Iterable[str]) -> QueryFilters | None:
    """Keep only the configured filter fields; None when nothing is left."""
    fields = set(fields)
    kept = QueryFilters(
        companies=filters.companies if "company" in fields else [],
        doc_periods=filters.doc_periods if "doc_period" in fields else [],
        doc_types=filters.doc_types if "doc_type" in fields else [],
    )
    return kept if (kept.companies or kept.doc_periods or kept.doc_types) else None
