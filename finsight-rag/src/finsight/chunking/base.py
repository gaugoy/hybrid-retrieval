"""
Building blocks shared by every chunker.

DocText
  A document's pages joined into one string, remembering where each page starts. It answers two
  questions fast:
    tokens(start, end)  how many tokens a character span holds (binary search over token offsets,
                        so the document is tokenized only once)
    page_of(char)       which page a character belongs to (for page_start / page_end)

split_spans / merge_spans: the "recursive" splitting idea
  1. split: if a span is too long, cut it at the strongest boundary available, paragraph
     ("\\n\\n") → line ("\\n") → sentence (". ") → word (" "), recursively, until every piece fits.
     If nothing works (one giant "word"), cut at token boundaries.
  2. merge: walk the pieces in order and pack them greedily into chunks of ≤ max_tokens. When a
     chunk is full, the next one starts with the trailing pieces that fit in `overlap` tokens,
     so a fact sitting on a boundary appears in both chunks.
  Spans are (start, end) character offsets, so text never gets copied or altered.
"""

from __future__ import annotations

import bisect
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass

from finsight.chunking.tokenizer import Tokenizer
from finsight.schemas import Chunk, DocumentMeta, Page

PAGE_SEPARATOR = "\n\n"
SEPARATORS = ("\n\n", "\n", ". ", " ")

Span = tuple[int, int]


@dataclass
class DocText:
    text: str
    page_starts: list[int]  # char offset where each page begins (sorted)
    page_idx: list[int]  # page_idx for each entry of page_starts
    token_starts: list[int]  # char offset of each token (sorted)

    @classmethod
    def from_pages(cls, pages: list[Page], tokenizer: Tokenizer) -> DocText:
        parts: list[str] = []
        starts: list[int] = []
        idxs: list[int] = []
        pos = 0
        for page in sorted(pages, key=lambda p: p.page_idx):
            if not page.text.strip():
                continue  # empty pages contribute no text (their page_idx simply never appears)
            if parts:
                parts.append(PAGE_SEPARATOR)
                pos += len(PAGE_SEPARATOR)
            starts.append(pos)
            idxs.append(page.page_idx)
            parts.append(page.text)
            pos += len(page.text)
        text = "".join(parts)
        token_starts = [s for s, _ in tokenizer.token_spans(text)] if text else []
        return cls(text, starts, idxs, token_starts)

    def tokens(self, start: int, end: int) -> int:
        """Tokens that start inside [start, end)."""
        return bisect.bisect_left(self.token_starts, end) - bisect.bisect_left(
            self.token_starts, start
        )

    def page_of(self, char: int) -> int:
        i = bisect.bisect_right(self.page_starts, char) - 1
        return self.page_idx[max(i, 0)]

    def page_span(self, page_idx: int) -> Span:
        i = self.page_idx.index(page_idx)
        end = (
            self.page_starts[i + 1] - len(PAGE_SEPARATOR)
            if i + 1 < len(self.page_starts)
            else len(self.text)
        )
        return self.page_starts[i], end


def _split_on(text: str, start: int, end: int, sep: str) -> list[Span]:
    """Cut [start, end) after each occurrence of sep (the separator stays on the left piece)."""
    out: list[Span] = []
    pos = start
    while True:
        hit = text.find(sep, pos, end)
        if hit == -1:
            break
        cut = hit + len(sep)
        if cut >= end:
            break
        out.append((pos, cut))
        pos = cut
    out.append((pos, end))
    return out


def split_spans(
    doc: DocText, start: int, end: int, max_tokens: int, seps: tuple[str, ...] = SEPARATORS
) -> list[Span]:
    """Recursively split [start, end) at the strongest separator until every piece fits."""
    if doc.tokens(start, end) <= max_tokens:
        return [(start, end)]
    for i, sep in enumerate(seps):
        parts = _split_on(doc.text, start, end, sep)
        if len(parts) > 1:
            out: list[Span] = []
            for ps, pe in parts:
                out.extend(split_spans(doc, ps, pe, max_tokens, seps[i:]))
            return out
    return _hard_split(doc, start, end, max_tokens)


def _hard_split(doc: DocText, start: int, end: int, max_tokens: int) -> list[Span]:
    """Last resort: cut every max_tokens tokens."""
    lo = bisect.bisect_left(doc.token_starts, start)
    hi = bisect.bisect_left(doc.token_starts, end)
    cuts = [doc.token_starts[k] for k in range(lo + max_tokens, hi, max_tokens)]
    bounds = [start, *cuts, end]
    return [(a, b) for a, b in itertools.pairwise(bounds) if b > a]


def merge_spans(doc: DocText, pieces: list[Span], max_tokens: int, overlap: int) -> list[Span]:
    """Greedily pack consecutive pieces into chunks of ≤ max_tokens, with token overlap."""
    chunks: list[Span] = []
    current: list[Span] = []
    for piece in pieces:
        if current and doc.tokens(current[0][0], piece[1]) > max_tokens:
            chunks.append((current[0][0], current[-1][1]))
            keep: list[Span] = []
            for q in reversed(current):  # trailing pieces that fit in the overlap budget
                if overlap and doc.tokens(q[0], current[-1][1]) <= overlap:
                    keep.insert(0, q)
                else:
                    break
            current = keep
            while current and doc.tokens(current[0][0], piece[1]) > max_tokens:
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append((current[0][0], current[-1][1]))
    return chunks


def trim(text: str, start: int, end: int) -> Span:
    """Shrink a span so it doesn't start or end with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def make_chunk(
    meta: DocumentMeta,
    *,
    seq: int,
    page_start: int,
    page_end: int,
    text: str,
    n_tokens: int,
    strategy: str,
    parent_id: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{meta.doc_name}::p{page_start}-{page_end}::c{seq}",
        doc_name=meta.doc_name,
        company=meta.company,
        doc_type=meta.doc_type,
        doc_period=meta.doc_period,
        page_start=page_start,
        page_end=page_end,
        text=text,
        embed_text=text,
        n_tokens=n_tokens,
        strategy=strategy,
        parent_id=parent_id,
    )


class Chunker(ABC):
    """Turns one document's pages into chunks. Implementations register by name (registry.py)."""

    name: str = "base"

    def __init__(self, tokenizer: Tokenizer, max_tokens: int, overlap: int) -> None:
        if overlap >= max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap

    @abstractmethod
    def chunk(self, pages: list[Page], meta: DocumentMeta) -> list[Chunk]: ...

    def _from_spans(self, doc: DocText, spans: list[Span], meta: DocumentMeta) -> list[Chunk]:
        chunks: list[Chunk] = []
        for s, e in spans:
            s, e = trim(doc.text, s, e)
            if s >= e:
                continue
            chunks.append(
                make_chunk(
                    meta,
                    seq=len(chunks),
                    page_start=doc.page_of(s),
                    page_end=doc.page_of(e - 1),
                    text=doc.text[s:e],
                    n_tokens=doc.tokens(s, e),
                    strategy=self.name,
                )
            )
        return chunks
