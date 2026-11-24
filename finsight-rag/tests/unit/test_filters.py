"""Rule-based company/year extraction from questions."""

from __future__ import annotations

import pytest

from finsight.retrieval.filters import FilterExtractor, apply_fields, build_aliases
from finsight.schemas import QueryFilters

COMPANIES = [
    "3M",
    "Amazon",
    "American Express",
    "AES Corporation",
    "Johnson & Johnson",
    "PG&E Corporation",
    "Coca-Cola",
    "Apple",
    "Block",
]
X = FilterExtractor(COMPANIES)


@pytest.mark.parametrize(
    ("question", "companies", "years"),
    [
        ("What is the FY2018 capital expenditure amount for 3M?", ["3M"], [2018]),
        ("MMM net PPNE at year end FY 2018?", ["3M"], [2018]),
        ("Did American Express grow card spend in 2022?", ["American Express"], [2022]),
        (
            "Compare Amazon and Apple revenue in FY2019 and FY2020",
            ["Amazon", "Apple"],
            [2019, 2020],
        ),
        ("What drove AES's results?", ["AES Corporation"], []),
        ("J&J talc litigation reserve", ["Johnson & Johnson"], []),
        ("PG&E wildfire costs", ["PG&E Corporation"], []),
        ("What is the cash flow statement total?", [], []),  # no company, "Block" not matched
    ],
)
def test_extract(question: str, companies: list[str], years: list[int]) -> None:
    f = X.extract(question)
    assert f.companies == companies and f.doc_periods == years


def test_aliases_strip_legal_suffixes() -> None:
    assert build_aliases(["AES Corporation"])["aes"] == "AES Corporation"


def test_apply_fields() -> None:
    f = QueryFilters(companies=["3M"], doc_periods=[2018])
    assert apply_fields(f, ["company"]) == QueryFilters(companies=["3M"])
    assert apply_fields(f, ["company", "doc_period"]) == f
    assert apply_fields(QueryFilters(), ["company"]) is None
