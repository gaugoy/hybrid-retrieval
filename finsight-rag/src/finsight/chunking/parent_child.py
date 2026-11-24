"""
Parent-child ("small-to-big") chunking: search with small chunks, answer with the whole page.

THE IDEA
  Small chunks (children, ~200 tokens) make precise embeddings: one topic, one table block. But
  a small chunk is often too little context for the LLM (the row without the statement's title,
  or the note it refers to). So:
    children  page-aware chunks of `child_tokens`, each with parent_id → INDEXED (P4)
    parents   the whole page, truncated to `parent_tokens` → stored, NOT indexed
  At query time (P6), retrieval finds children and the generator receives their parents,
  de-duplicated. That separates "what matches well" from "what reads well".

More info: "small-to-big" / parent document retrieval, e.g.
https://python.langchain.com/docs/how_to/parent_document_retriever/
"""

from __future__ import annotations

import bisect

from finsight.chunking.base import DocText, make_chunk
from finsight.chunking.page_aware import PageAwareChunker
from finsight.chunking.registry import register_chunker
from finsight.chunking.tokenizer import Tokenizer
from finsight.schemas import Chunk, DocumentMeta, Page


def parent_id_for(doc_name: str, page_idx: int) -> str:
    return f"{doc_name}::p{page_idx}-{page_idx}::parent"


@register_chunker("parent_child")
class ParentChildChunker(PageAwareChunker):
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_tokens: int,
        overlap: int,
        parent_tokens: int = 1500,
        carry_table_header: bool = True,
    ) -> None:
        super().__init__(tokenizer, max_tokens, overlap, carry_table_header)
        self.parent_tokens = parent_tokens

    def chunk(self, pages: list[Page], meta: DocumentMeta) -> list[Chunk]:
        """The children (what gets indexed). Each points to its page's parent."""
        children = super().chunk(pages, meta)
        for c in children:
            c.parent_id = parent_id_for(meta.doc_name, c.page_start)
            c.strategy = self.name
        return children

    def parents(self, pages: list[Page], meta: DocumentMeta) -> list[Chunk]:
        """One parent per non-empty page: the page text, truncated to parent_tokens."""
        doc = DocText.from_pages(pages, self.tokenizer)
        out = []
        for page_idx in doc.page_idx:
            start, end = doc.page_span(page_idx)
            first = bisect.bisect_left(doc.token_starts, start)
            if doc.tokens(start, end) > self.parent_tokens:
                end = doc.token_starts[first + self.parent_tokens]
            parent = make_chunk(
                meta,
                seq=0,
                page_start=page_idx,
                page_end=page_idx,
                text=doc.text[start:end].rstrip(),
                n_tokens=doc.tokens(start, end),
                strategy="parent_child_parent",
            )
            parent.chunk_id = parent_id_for(meta.doc_name, page_idx)
            out.append(parent)
        return out
