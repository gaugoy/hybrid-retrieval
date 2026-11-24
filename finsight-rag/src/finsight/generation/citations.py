"""
Citation checking: every [n] must point at a block that was actually shown.

  hallucinated_citation  the answer cites [7] but there were 5 blocks (small models do this)
  uncited_answer         a non-abstaining answer with no valid citation (ungrounded)
  abstained              the answer starts with the abstain phrase
  auto_cited             the model gave no [n], but every figure it stated was found verbatim in a
                         block, so those blocks are cited (auto_cite); small local models often
                         answer correctly without following the citation format
"""

from __future__ import annotations

import re

from finsight.schemas import Citation, RetrievedChunk

MARKER = re.compile(r"\[(\d+)\]")
# figures worth matching: 1,577 / 32,765.5 / 12.3 / 2018 is excluded (years match everything)
FIGURE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])\d+\.\d+")


def auto_cite(answer: str, marker_map: dict[int, RetrievedChunk]) -> list[int]:
    """Blocks that contain each figure stated in the answer (first block per figure), in order."""
    found: list[int] = []
    for fig in dict.fromkeys(FIGURE.findall(answer)):
        for n, rc in marker_map.items():
            if fig in rc.chunk.text:
                if n not in found:
                    found.append(n)
                break
    return found


def is_abstention(answer: str, abstain_phrase: str) -> bool:
    norm = answer.strip().strip('"').lower()
    return norm.startswith(abstain_phrase.lower().rstrip(".")) or not norm


def parse_citations(
    answer: str, marker_map: dict[int, RetrievedChunk], abstain_phrase: str
) -> tuple[list[Citation], bool, list[str]]:
    citations: list[Citation] = []
    warnings: list[str] = []
    seen: set[int] = set()
    for m in MARKER.finditer(answer):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        rc = marker_map.get(n)
        if rc is None:
            if "hallucinated_citation" not in warnings:
                warnings.append("hallucinated_citation")
            continue
        citations.append(
            Citation(
                marker=n,
                chunk_id=rc.chunk.chunk_id,
                doc_name=rc.chunk.doc_name,
                page_idx=rc.chunk.page_start,
                snippet=rc.chunk.text[:300],
            )
        )
    abstained = is_abstention(answer, abstain_phrase)
    if not abstained and not citations:
        repaired = auto_cite(answer, marker_map)
        if repaired:
            warnings.append("auto_cited")
            for n in repaired:
                rc = marker_map[n]
                citations.append(
                    Citation(
                        marker=n,
                        chunk_id=rc.chunk.chunk_id,
                        doc_name=rc.chunk.doc_name,
                        page_idx=rc.chunk.page_start,
                        snippet=rc.chunk.text[:300],
                    )
                )
        else:
            warnings.append("uncited_answer")
    return citations, abstained, warnings
