"""
Context assembly: numbered blocks the LLM must cite.

    [1] 3M · 10k FY2018 · page 60
    <chunk text>

Blocks follow the rerank order. For parent_child, the block text is the PARENT (the page),
de-duplicated when several children hit the same page. Blocks are added until
generation.max_context_tokens is reached (counted with the chunking tokenizer); a block that
doesn't fit is skipped, the next smaller one may still fit.
"""

from __future__ import annotations

from collections.abc import Callable

from finsight.retrieval.base import ChunkStore
from finsight.schemas import RetrievedChunk


def block_header(n: int, rc: RetrievedChunk) -> str:
    c = rc.chunk
    pages = f"page {c.page_start + 1}" + (f"-{c.page_end + 1}" if c.page_end > c.page_start else "")
    return f"[{n}] {c.company} · {c.doc_type} FY{c.doc_period} · {pages}"


def build_context(
    retrieved: list[RetrievedChunk],
    *,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    store: ChunkStore | None = None,
    max_blocks: int | None = None,
) -> tuple[str, dict[int, RetrievedChunk]]:
    blocks: list[str] = []
    marker_map: dict[int, RetrievedChunk] = {}
    used, seen_parents = 0, set()
    for rc in retrieved:
        if max_blocks is not None and len(marker_map) >= max_blocks:
            break
        text = rc.chunk.text
        if rc.chunk.parent_id and store is not None:
            if rc.chunk.parent_id in seen_parents:
                continue
            parent = store.parent(rc.chunk.parent_id)
            if parent is not None:
                seen_parents.add(rc.chunk.parent_id)
                text = parent.text
        n = len(marker_map) + 1
        block = f"{block_header(n, rc)}\n{text}"
        cost = count_tokens(block)
        if used + cost > max_tokens:
            continue
        blocks.append(block)
        marker_map[n] = rc
        used += cost
    return "\n\n".join(blocks), marker_map
