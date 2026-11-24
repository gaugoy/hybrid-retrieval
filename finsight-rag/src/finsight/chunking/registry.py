"""
Chunker registry: chunkers register by name, and config picks one.

    @register_chunker("page_aware")
    class PageAwareChunker(Chunker): ...

    build_chunker(settings.chunking, tokenizer)   # strategy name comes from config

Adding a strategy = one new class + one decorator; no if/else chains anywhere else.
"""

from __future__ import annotations

from collections.abc import Callable

from finsight.chunking.base import Chunker
from finsight.chunking.tokenizer import Tokenizer
from finsight.config import ChunkingCfg

CHUNKERS: dict[str, type[Chunker]] = {}


def register_chunker(name: str) -> Callable[[type[Chunker]], type[Chunker]]:
    def deco(cls: type[Chunker]) -> type[Chunker]:
        CHUNKERS[name] = cls
        cls.name = name
        return cls

    return deco


def build_chunker(cfg: ChunkingCfg, tokenizer: Tokenizer, reserve_tokens: int = 0) -> Chunker:
    """Instantiate the configured chunker. ``reserve_tokens`` leaves room for a context header."""
    from finsight.chunking import fixed, page_aware, parent_child, recursive  # noqa: F401 (register)

    cls = CHUNKERS[cfg.strategy]
    if cfg.strategy == "parent_child":
        return cls(  # type: ignore[call-arg]
            tokenizer,
            max_tokens=cfg.child_tokens - reserve_tokens,
            overlap=min(cfg.overlap_tokens, cfg.child_tokens // 4),
            parent_tokens=cfg.parent_tokens,
            carry_table_header=cfg.carry_table_header,
        )
    kwargs = {"carry_table_header": cfg.carry_table_header} if cfg.strategy == "page_aware" else {}
    return cls(
        tokenizer,
        max_tokens=cfg.chunk_tokens - reserve_tokens,
        overlap=cfg.overlap_tokens,
        **kwargs,
    )
