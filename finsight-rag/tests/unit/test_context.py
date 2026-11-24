"""Context assembly: numbering, headers, token budget, block cap, parent de-duplication."""

from __future__ import annotations

import pandas as pd
from _helpers import rc

from finsight.chunking.tokenizer import WhitespaceTokenizer
from finsight.generation.context import build_context
from finsight.retrieval.base import ChunkStore

TOK = WhitespaceTokenizer()


def test_numbering_headers_and_budget() -> None:
    chunks = [
        rc("a", 1, text="alpha " * 20, page=59, company="3M", period=2018),
        rc("b", 2, text="beta " * 200),
        rc("c", 3, text="gamma " * 10),
    ]
    ctx, mm = build_context(chunks, max_tokens=60, count_tokens=TOK.count)
    assert ctx.startswith("[1] 3M · 10k FY2018 · page 60\nalpha")
    assert list(mm) == [1, 2] and mm[2].chunk.chunk_id == "c"  # 'b' is too big, 'c' still fits
    _ctx2, mm2 = build_context(chunks, max_tokens=10_000, count_tokens=TOK.count, max_blocks=1)
    assert list(mm2) == [1]


def test_parent_text_replaces_children_once() -> None:
    a, b = rc("child-a", 1), rc("child-b", 2)
    a.chunk.parent_id = b.chunk.parent_id = "P1"
    parents = pd.DataFrame(
        [{**a.chunk.model_dump(), "chunk_id": "P1", "text": "THE WHOLE PAGE", "parent_id": None}]
    )
    store = ChunkStore(pd.DataFrame([a.chunk.model_dump(), b.chunk.model_dump()]), parents)
    ctx, mm = build_context([a, b], max_tokens=1000, count_tokens=TOK.count, store=store)
    assert "THE WHOLE PAGE" in ctx and list(mm) == [1]
