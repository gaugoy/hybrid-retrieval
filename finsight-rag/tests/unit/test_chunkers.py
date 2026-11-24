"""Chunker invariants with a whitespace tokenizer (1 token = 1 word): no model download."""

from __future__ import annotations

import itertools

import pytest

from finsight.chunking.base import DocText, merge_spans, split_spans
from finsight.chunking.contextual import add_context_header
from finsight.chunking.fixed import FixedTokenChunker
from finsight.chunking.page_aware import PageAwareChunker, is_column_header
from finsight.chunking.parent_child import ParentChildChunker
from finsight.chunking.recursive import RecursiveChunker
from finsight.chunking.registry import CHUNKERS, build_chunker
from finsight.chunking.tokenizer import WhitespaceTokenizer
from finsight.config import ChunkingCfg
from finsight.schemas import DocumentMeta, Page

TOK = WhitespaceTokenizer()
META = DocumentMeta(doc_name="ACME_2020_10K", company="Acme", doc_type="10k", doc_period=2020)


def page(idx: int, text: str) -> Page:
    return Page(doc_name=META.doc_name, page_idx=idx, text=text, n_chars=len(text))


def prose(n_words: int, tag: str) -> str:
    words = [f"{tag}{i}" for i in range(n_words)]
    sentences = [" ".join(words[i : i + 10]) + "." for i in range(0, n_words, 10)]
    return "\n\n".join(" ".join(sentences[i : i + 3]) for i in range(0, len(sentences), 3))


PAGES = [
    page(0, prose(120, "a")),
    page(1, ""),  # empty page: contributes nothing, never appears in a chunk
    page(2, prose(90, "b")),
    page(
        3,
        "Consolidated Statement of Cash Flows\n(Millions) | 2018 | 2017 | 2016\n"
        + "\n".join(f"Line item {i} | {i} | {i + 1} | {i + 2}" for i in range(40)),
    ),
]


# ---------------------------------------------------------------- building blocks


def test_doctext_maps_chars_to_pages_and_skips_empty_pages() -> None:
    doc = DocText.from_pages(PAGES, TOK)
    assert doc.page_idx == [0, 2, 3]
    assert doc.page_of(0) == 0
    start2, _ = doc.page_span(2)
    assert doc.page_of(start2) == 2
    assert doc.tokens(0, len(doc.text)) == TOK.count(doc.text)


def test_split_and_merge_respect_budget_and_overlap() -> None:
    doc = DocText.from_pages(PAGES[:1], TOK)
    pieces = split_spans(doc, 0, len(doc.text), max_tokens=25)
    assert all(doc.tokens(s, e) <= 25 for s, e in pieces)
    chunks = merge_spans(doc, pieces, max_tokens=40, overlap=10)
    assert all(doc.tokens(s, e) <= 40 for s, e in chunks)
    assert all(b[0] < a[1] for a, b in itertools.pairwise(chunks))  # overlapping spans


# ---------------------------------------------------------------- strategies


@pytest.mark.parametrize("cls", [FixedTokenChunker, RecursiveChunker, PageAwareChunker])
def test_no_chunk_exceeds_budget_and_ids_are_deterministic(cls: type) -> None:
    a = cls(TOK, max_tokens=50, overlap=10).chunk(PAGES, META)
    b = cls(TOK, max_tokens=50, overlap=10).chunk(PAGES, META)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert all(c.n_tokens <= 50 for c in a)
    assert all(1 not in (c.page_start, c.page_end) for c in a)  # the empty page is never cited


def test_fixed_windows_overlap_and_cross_pages() -> None:
    chunks = FixedTokenChunker(TOK, max_tokens=100, overlap=20).chunk(PAGES, META)
    assert all(c.n_tokens == 100 for c in chunks[:-1])  # every window but the last is full
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]  # exactly `overlap` tokens repeated
    assert any(c.page_end > c.page_start for c in chunks)  # windows ignore page boundaries


def test_recursive_prefers_paragraph_boundaries() -> None:
    chunks = RecursiveChunker(TOK, max_tokens=70, overlap=0).chunk(PAGES[:1], META)
    assert all(c.text.endswith(".") for c in chunks)  # ends on a sentence/paragraph, not mid-way


def test_page_aware_never_crosses_pages() -> None:
    chunks = PageAwareChunker(TOK, max_tokens=60, overlap=10).chunk(PAGES, META)
    assert all(c.page_start == c.page_end for c in chunks)
    assert {c.page_start for c in chunks} == {0, 2, 3}


def test_page_aware_carries_column_header_into_later_table_chunks() -> None:
    chunks = PageAwareChunker(TOK, max_tokens=60, overlap=0).chunk(PAGES[3:], META)
    assert len(chunks) > 2
    for c in chunks:
        assert any(is_column_header(line) for line in c.text.splitlines()), c.text[:80]
        assert c.n_tokens <= 60
    off = PageAwareChunker(TOK, max_tokens=60, overlap=0, carry_table_header=False).chunk(
        PAGES[3:], META
    )
    assert sum(any(is_column_header(ln) for ln in c.text.splitlines()) for c in off) == 1


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("(Millions) | 2018 | 2017 | 2016", True),
        ("Year ended | December 31, 2020 | December 31, 2019", True),
        ("Fiscal 2023 | Fiscal 2022", True),
        ("Q4 2023 | Q4 2022 | Q3 2023", True),
        ("Net sales | 11,303 | 11,503", False),
        ("Revenue grew in 2020 and 2021", False),
        ("(Millions) | 2018", False),  # one period is not a header row
    ],
)
def test_is_column_header(line: str, expected: bool) -> None:
    assert is_column_header(line) is expected


def test_parent_child_links_children_to_page_parents() -> None:
    chunker = ParentChildChunker(TOK, max_tokens=30, overlap=5, parent_tokens=60)
    children, parents = chunker.chunk(PAGES, META), chunker.parents(PAGES, META)
    parent_ids = {p.chunk_id for p in parents}
    assert [p.page_start for p in parents] == [0, 2, 3]  # one parent per non-empty page
    assert all(c.parent_id in parent_ids for c in children)
    assert all(c.n_tokens <= 30 for c in children)
    assert all(p.n_tokens <= 60 for p in parents)  # truncated to parent_tokens
    assert all(c.strategy == "parent_child" for c in children)


# ---------------------------------------------------------------- registry & contextual


def test_registry_builds_every_strategy_with_reserve() -> None:
    assert set(CHUNKERS) >= {"fixed", "recursive", "page_aware", "parent_child"}
    for strategy in CHUNKERS:
        cfg = ChunkingCfg(strategy=strategy, chunk_tokens=100, overlap_tokens=10, child_tokens=50)
        chunker = build_chunker(cfg, TOK, reserve_tokens=20)
        assert chunker.max_tokens == (50 if strategy == "parent_child" else 100) - 20


def test_context_header_goes_into_embed_text_only() -> None:
    chunk = PageAwareChunker(TOK, max_tokens=60, overlap=0).chunk(PAGES[:1], META)[0]
    original = chunk.text
    add_context_header(chunk)
    assert chunk.embed_text.startswith("Acme | 10k FY2020 | page 1\n")
    assert chunk.text == original
