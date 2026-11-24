"""Small builders shared by unit tests (tests/unit is on sys.path under pytest's prepend mode)."""

from __future__ import annotations

from finsight.schemas import Chunk, RetrievedChunk


def chunk(
    cid: str,
    doc: str = "D",
    page: int = 0,
    text: str | None = None,
    company: str = "C",
    period: int = 2020,
) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_name=doc,
        company=company,
        doc_type="10k",
        doc_period=period,
        page_start=page,
        page_end=page,
        text=text or cid,
        embed_text=text or cid,
        n_tokens=1,
        strategy="x",
    )


def rc(
    cid: str, rank: int, stage: str = "dense", score: float = 1.0, **kw: object
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk(cid, **kw),
        score=score,
        rank=rank,  # type: ignore[arg-type]
        stage=stage,
        scores_by_stage={stage: score},
    )  # type: ignore[arg-type]
