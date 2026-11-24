"""
Page-aware chunking: recursive splitting, but never across a page boundary, plus table headers.

WHY PAGE BOUNDARIES
  Every chunk belongs to exactly one page (page_start == page_end), so a citation is exact:
  "3M 10-K FY2018, page 60". Pages in a filing are also natural units: one statement, one note.
  The cost: a sentence that continues onto the next page is split in two.

CARRYING THE TABLE HEADER (finance-specific)
  A balance sheet chunk that starts halfway down the table reads
      Inventories | (509) | (387) | 57
  Which column is 2018? The column header "(Millions) | 2018 | 2017 | 2016" is in an earlier
  chunk. With `carry_table_header`, a chunk that contains table rows but not their column header
  gets that header line prepended, so every chunk knows which fiscal year each number is for.
  A column header is a " | " row whose value cells are all years or periods.
"""

from __future__ import annotations

import re

from finsight.chunking.base import Chunker, DocText, make_chunk, merge_spans, split_spans, trim
from finsight.chunking.registry import register_chunker
from finsight.chunking.tokenizer import Tokenizer
from finsight.schemas import Chunk, DocumentMeta, Page

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_PERIOD = re.compile(
    r"^(?:FY\s*|fiscal\s+(?:year\s+)?)?(?:19|20)\d{2}$"  # 2018, FY2018, Fiscal 2018
    rf"|^{_MONTH}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}$"  # December 31, 2020 / Dec. 31 2020
    r"|^(?:Q[1-4]|[1-4]Q)\s*(?:FY\s*)?(?:19|20)?\d{2}$"  # Q4 2023, 4Q23
    r"|^(?:19|20)\d{2}\s*Q[1-4]$",  # 2023 Q4
    re.IGNORECASE,
)


def is_column_header(line: str) -> bool:
    """'(Millions) | 2018 | 2017 | 2016' → True. A ' | ' row whose value cells are periods."""
    cells = [c.strip() for c in line.split(" | ")]
    values = [c for c in cells[1:] if c]
    if _PERIOD.match(cells[0]):
        values = [c for c in cells if c]  # header row with no label cell: "2018 | 2017 | 2016"
    return len(values) >= 2 and all(_PERIOD.match(v) for v in values)


@register_chunker("page_aware")
class PageAwareChunker(Chunker):
    def __init__(
        self, tokenizer: Tokenizer, max_tokens: int, overlap: int, carry_table_header: bool = True
    ) -> None:
        super().__init__(tokenizer, max_tokens, overlap)
        self.carry_table_header = carry_table_header

    def chunk(self, pages: list[Page], meta: DocumentMeta) -> list[Chunk]:
        doc = DocText.from_pages(pages, self.tokenizer)
        chunks: list[Chunk] = []
        for page_idx in doc.page_idx:
            chunks.extend(self.chunk_page(doc, page_idx, meta, seq_start=len(chunks)))
        return chunks

    def chunk_page(
        self, doc: DocText, page_idx: int, meta: DocumentMeta, seq_start: int = 0
    ) -> list[Chunk]:
        p_start, p_end = doc.page_span(page_idx)
        headers = self._column_headers(doc, p_start, p_end) if self.carry_table_header else []
        # Reserve room for the longest column header on this page, so carrying it always fits.
        reserve = max((doc.tokens(a, b) for a, b in headers), default=0)
        budget = max(self.max_tokens - reserve, self.overlap + 1)
        pieces = split_spans(doc, p_start, p_end, budget)
        out: list[Chunk] = []
        for s, e in merge_spans(doc, pieces, budget, self.overlap):
            s, e = trim(doc.text, s, e)
            if s >= e:
                continue
            text, n_tokens = doc.text[s:e], doc.tokens(s, e)
            if self.carry_table_header and " | " in text:
                header = next((h for h in reversed(headers) if h[1] <= s), None)
                if header is not None:
                    h_text = doc.text[header[0] : header[1]]
                    h_tokens = doc.tokens(header[0], header[1])
                    if n_tokens + h_tokens <= self.max_tokens:
                        text, n_tokens = f"{h_text}\n{text}", n_tokens + h_tokens
            out.append(
                make_chunk(
                    meta,
                    seq=seq_start + len(out),
                    page_start=page_idx,
                    page_end=page_idx,
                    text=text,
                    n_tokens=n_tokens,
                    strategy=self.name,
                )
            )
        return out

    @staticmethod
    def _column_headers(doc: DocText, start: int, end: int) -> list[tuple[int, int]]:
        """Spans of the column-header lines on a page, top to bottom."""
        spans = []
        pos = start
        for line in doc.text[start:end].split("\n"):
            if is_column_header(line):
                spans.append((pos, pos + len(line)))
            pos += len(line) + 1
        return spans
