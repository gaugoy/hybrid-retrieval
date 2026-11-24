"""
Contextual chunk headers (cheap version of "contextual retrieval"; evaluated in P7).

A chunk saying "Revenue grew 5%" doesn't say whose revenue or which year. Prepending
"3M | 10k FY2018 | page 60" to the text that gets EMBEDDED and BM25-indexed (`embed_text`) puts
that context into the vector, while `text` (what the LLM and user see) stays unchanged.
Enable with `chunking.contextual_headers: true`; the chunk budget is reduced by HEADER_RESERVE
tokens so embed_text still fits the model's limit.

More info: https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

from finsight.schemas import Chunk

HEADER_RESERVE = 24  # tokens kept free for the header


def context_header(chunk: Chunk) -> str:
    pages = (
        f"page {chunk.page_start + 1}"
        if chunk.page_start == chunk.page_end
        else f"pages {chunk.page_start + 1}-{chunk.page_end + 1}"
    )
    return f"{chunk.company} | {chunk.doc_type} FY{chunk.doc_period} | {pages}"


def add_context_header(chunk: Chunk) -> Chunk:
    chunk.embed_text = f"{context_header(chunk)}\n{chunk.text}"
    return chunk
