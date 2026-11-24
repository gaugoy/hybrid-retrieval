"""
Token counting with the EMBEDDING model's own tokenizer.

WHY THIS TOKENIZER
  A chunk's size limit comes from the embedding model: bge accepts 512 tokens, including two
  special tokens, so 510 tokens of text. Anything longer is silently truncated: the end of the
  chunk never makes it into the vector. Counting with a different tokenizer (tiktoken, words)
  gives different numbers, so we count with the model's own WordPiece tokenizer.
  Numbers are expensive: "(1,577)" is 6 tokens ( ( 1 , 57 ##7 ) ), so financial tables fill the
  budget much faster than prose does.

HOW
  `token_spans(text)` returns the (start, end) character offsets of every token. Chunkers
  tokenize a whole document once and then count any span with a binary search, see
  `finsight.chunking.base.DocText`.

More info: https://huggingface.co/docs/transformers/main_classes/tokenizer
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol


class Tokenizer(Protocol):
    name: str

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """(start, end) character offsets of each token, special tokens excluded."""
        ...

    def count(self, text: str) -> int:
        """Number of tokens, special tokens excluded."""
        ...


class HFTokenizer:
    """A Hugging Face fast tokenizer (the embedding model's). Loaded lazily, once per model."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self.name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._tok.model_max_length = 10**9  # we tokenize whole documents; no length warnings

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(s), int(e)) for s, e in enc["offset_mapping"]]

    def count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)["input_ids"])


class WhitespaceTokenizer:
    """One token per whitespace-separated word. For tests: no model download needed."""

    name = "whitespace"
    _WORD = re.compile(r"\S+")

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in self._WORD.finditer(text)]

    def count(self, text: str) -> int:
        return len(self._WORD.findall(text))


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> Tokenizer:
    """Shared tokenizer instance per model ('whitespace' gives the test tokenizer)."""
    if model_name == "whitespace":
        return WhitespaceTokenizer()
    return HFTokenizer(model_name)
