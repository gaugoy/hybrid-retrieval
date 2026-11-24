"""Prompt templates live in generation/prompts/*.md and use $placeholders (string.Template).

Not str.format: retrieved filing text is full of braces and percent signs, and Template only
substitutes $names that we pass. A missing placeholder raises, so a typo can't ship silently.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

PROMPTS = Path(__file__).parent / "prompts"


@lru_cache(maxsize=32)
def _template(name: str) -> Template:
    return Template((PROMPTS / f"{name}.md").read_text().strip())


def render(name: str, **values: object) -> str:
    return _template(name).substitute({k: str(v) for k, v in values.items()})
