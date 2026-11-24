"""
Fixed-size chunking: slide a window of N tokens over the whole document, step N - overlap.

THE BASELINE
  Simple, predictable sizes, and the most common default in tutorials. Its weaknesses are the
  point of comparing it: windows ignore structure, so a chunk can start mid-sentence or mid-word
  ("57" | "##7"), cut a table row in half, and span two pages, which makes citations vague
  ("pages 59–60").

More info: https://www.pinecone.io/learn/chunking-strategies/
"""

from __future__ import annotations

from finsight.chunking.base import Chunker, DocText
from finsight.chunking.registry import register_chunker
from finsight.schemas import Chunk, DocumentMeta, Page


@register_chunker("fixed")
class FixedTokenChunker(Chunker):
    def chunk(self, pages: list[Page], meta: DocumentMeta) -> list[Chunk]:
        doc = DocText.from_pages(pages, self.tokenizer)
        starts = doc.token_starts
        if not starts:
            return []
        stride = self.max_tokens - self.overlap
        spans = []
        for k in range(0, len(starts), stride):
            end = (
                starts[k + self.max_tokens] if k + self.max_tokens < len(starts) else len(doc.text)
            )
            spans.append((starts[k], end))
            if k + self.max_tokens >= len(starts):
                break
        return self._from_spans(doc, spans, meta)
