"""
LLM-as-judge with a disk cache.

  correctness(question, gold, prediction) -> correct | partially_correct | incorrect (+ reason)
  faithfulness(answer, cited texts)       -> supported_claims / total_claims
JSON is validated with pydantic; on failure: one retry, then verdict "judge_error" (counted
separately, never as correct). Calls are cached in SQLite keyed by sha256(judge model + prompt),
so re-running a report costs nothing.

BIASES TO REMEMBER: a judge tends to favour its own model family, longer answers, and the first
option it sees. Prefer a judge from a different family than the generator, report which judge
produced each number, and validate it against hand labels (scripts/10b_label_judge_sample.py).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from finsight.generation import prompt_loader
from finsight.generation.jsonout import ask_json
from finsight.generation.llm.base import LLM


class Correctness(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    reason: str = ""


class Faithfulness(BaseModel):
    supported_claims: int = Field(ge=0)
    total_claims: int = Field(ge=0)
    unsupported: list[str] = Field(default_factory=list)


class Judge:
    def __init__(self, llm: LLM, cache_path: Path | None = None) -> None:
        self.llm = llm
        self.calls = 0
        self._db = None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(cache_path)
            self._db.execute("CREATE TABLE IF NOT EXISTS j (key TEXT PRIMARY KEY, value TEXT)")

    @property
    def name(self) -> str:
        return f"{self.llm.provider}:{self.llm.model}"

    def _cached(self, prompt: str, model: type[BaseModel]) -> BaseModel | None:
        key = hashlib.sha256(f"{self.name}\x00{prompt}".encode()).hexdigest()
        if self._db is not None:
            row = self._db.execute("SELECT value FROM j WHERE key = ?", (key,)).fetchone()
            if row:
                return model.model_validate(json.loads(row[0]))
        parsed, n = ask_json(
            self.llm,
            prompt,
            model,
            system="You are a strict grader. JSON only.",
            max_tokens=300,
            retries=1,
        )
        self.calls += n
        if parsed is not None and self._db is not None:
            self._db.execute(
                "INSERT OR REPLACE INTO j VALUES (?, ?)", (key, parsed.model_dump_json())
            )
            self._db.commit()
        return parsed

    def correctness(self, question: str, gold: str, prediction: str) -> tuple[str, str]:
        prompt = prompt_loader.render(
            "judge_correctness", question=question, gold=gold, prediction=prediction
        )
        out = self._cached(prompt, Correctness)
        if out is None:
            return "judge_error", ""
        assert isinstance(out, Correctness)
        return out.verdict, out.reason

    def faithfulness(self, answer: str, sources: list[str]) -> float | None:
        if not sources:
            return None
        joined = "\n\n".join(f"[{i + 1}] {s[:1500]}" for i, s in enumerate(sources))
        out = self._cached(
            prompt_loader.render("judge_faithfulness", answer=answer, sources=joined), Faithfulness
        )
        if out is None or not isinstance(out, Faithfulness) or out.total_claims == 0:
            return None
        return min(1.0, out.supported_claims / out.total_claims)
