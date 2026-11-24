"""Judge: parsing, retry/error, cache (no second LLM call)."""

from __future__ import annotations

from pathlib import Path

from finsight.evaluation.judge import Judge
from finsight.generation.llm.fake import FakeLLM


def test_correctness_cached(tmp_path: Path) -> None:
    llm = FakeLLM(['{"verdict": "correct", "reason": "same number"}'])
    j = Judge(llm, tmp_path / "j.sqlite")
    assert j.correctness("q", "$1,577", "1,577 million") == ("correct", "same number")
    again = Judge(FakeLLM([]), tmp_path / "j.sqlite")  # new process, same cache
    assert again.correctness("q", "$1,577", "1,577 million")[0] == "correct" and again.calls == 0


def test_judge_error_after_retry_and_faithfulness() -> None:
    j = Judge(FakeLLM(["not json", "still not"]))
    assert j.correctness("q", "g", "p") == ("judge_error", "") and j.calls == 2
    f = Judge(FakeLLM(['{"supported_claims": 2, "total_claims": 4, "unsupported": ["x"]}']))
    assert f.faithfulness("answer", ["source text"]) == 0.5
    assert f.faithfulness("answer", []) is None
