"""
Recursive chunking: split at the strongest natural boundary that makes pieces fit, then pack.

Paragraph → line → sentence → word, recursively (see base.split_spans), then greedy packing with
overlap (base.merge_spans). Chunks end at natural boundaries, so they read cleanly. But they
still cross page boundaries: the whole document is treated as one stream of text.
This is the idea behind LangChain's RecursiveCharacterTextSplitter, written out by hand, and
counted in tokens rather than characters.
"""

from __future__ import annotations

from finsight.chunking.base import Chunker, DocText, merge_spans, split_spans
from finsight.chunking.registry import register_chunker
from finsight.schemas import Chunk, DocumentMeta, Page


@register_chunker("recursive")
class RecursiveChunker(Chunker):
    def chunk(self, pages: list[Page], meta: DocumentMeta) -> list[Chunk]:
        doc = DocText.from_pages(pages, self.tokenizer)
        if not doc.text:
            return []
        pieces = split_spans(doc, 0, len(doc.text), self.max_tokens)
        return self._from_spans(doc, merge_spans(doc, pieces, self.max_tokens, self.overlap), meta)
