"""Citation parsing and abstention detection."""

from __future__ import annotations

from _helpers import rc

from finsight.generation.citations import parse_citations

ABSTAIN = "Not found in the provided filings."
MAP = {1: rc("a", 1, doc="3M_2018_10K", page=59), 2: rc("b", 2, doc="3M_2018_10K", page=60)}


def test_valid_duplicate_and_hallucinated_markers() -> None:
    cites, abstained, warnings = parse_citations(
        "Capex was $1,577M [1][7], see [1] and [2].", MAP, ABSTAIN
    )
    assert [c.marker for c in cites] == [1, 2] and not abstained
    assert cites[0].doc_name == "3M_2018_10K" and cites[0].page_idx == 59
    assert warnings == ["hallucinated_citation"]


def test_uncited_and_abstained() -> None:
    assert parse_citations("Capex was $1,577M.", MAP, ABSTAIN)[2] == ["uncited_answer"]
    cites, abstained, warnings = parse_citations(
        '"Not found in the provided filings."', MAP, ABSTAIN
    )
    assert abstained and cites == [] and warnings == []
    assert parse_citations("", MAP, ABSTAIN)[1]  # empty = abstain


def test_auto_cite_matches_figures_verbatim() -> None:
    mm = {
        1: rc("a", 1, text="Net sales | 32,765 | 31,657"),
        2: rc("b", 2, text="Purchases of PP&E | (1,577) | (1,373)"),
    }
    cites, _, warnings = parse_citations("Capex was $1,577 million in 2018.", mm, ABSTAIN)
    assert [c.marker for c in cites] == [2] and warnings == ["auto_cited"]
    _, _, w2 = parse_citations("Capex was $9,999 million.", mm, ABSTAIN)
    assert w2 == ["uncited_answer"]  # a figure in no block: still ungrounded
