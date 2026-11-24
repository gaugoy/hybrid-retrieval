"""Ask an LLM for JSON, validate it with pydantic, retry once, else return None (callers fall back).

Small local models often wrap JSON in prose or ```json fences: we extract the first {...} block.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from finsight.generation.llm.base import LLM

T = TypeVar("T", bound=BaseModel)
_OBJ = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict | None:
    m = _OBJ.search(text)
    if not m:
        return None
    try:
        value = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def ask_json[T: BaseModel](
    llm: LLM,
    prompt: str,
    model: type[T],
    *,
    system: str = "You return JSON only.",
    max_tokens: int = 400,
    retries: int = 1,
) -> tuple[T | None, int]:
    """(parsed or None, number of LLM calls made)."""
    calls = 0
    for _ in range(retries + 1):
        calls += 1
        resp = llm.generate(
            system, [{"role": "user", "content": prompt}], max_tokens, json_mode=True
        )
        data = extract_json(resp.text)
        if data is not None:
            try:
                return model.model_validate(data), calls
            except ValidationError:
                pass
    return None, calls
