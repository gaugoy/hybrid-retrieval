"""Numeric and text answer matching."""

from __future__ import annotations

import pytest

from finsight.evaluation.generation_metrics import contains_match, numeric_match, parse_numbers


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        ("3M's FY2018 capex was $1,577 million [2].", "$1577.00", True),
        ("Capex was 1.58 billion dollars.", "$1577.00 million", True),  # unit scaling
        ("The ratio is 24.3%.", "24.26%", True),  # within 1%
        ("Net loss of (0.13) per share", "-0.13", True),  # parentheses = negative
        ("Capex was $1,400 million.", "$1577.00", False),
        ("Not found in the provided filings.", "$1577.00", False),
    ],
)
def test_numeric_match(pred: str, gold: str, ok: bool) -> None:
    assert numeric_match(pred, gold) is ok


def test_non_numeric_gold_and_parsing() -> None:
    assert numeric_match("Yes, it is.", "Yes") is None
    nums = parse_numbers("($1,234) and 5.5% and 2 billion")
    assert [n.value for n in nums] == [-1234.0, 5.5, 2.0] and nums[2].scaled == 2e9 and nums[1].pct


def test_contains_match() -> None:
    assert contains_match(
        "The individual is Sam Bankman-Fried, per both outlets.", "Sam Bankman-Fried"
    )
    assert contains_match("Yes. 3M is capital intensive.", "Yes")
    assert not contains_match("Yesterday's results", "Yes")
