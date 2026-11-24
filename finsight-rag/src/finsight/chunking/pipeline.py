"""
Chunking pipeline: pages.parquet → data/chunks/<strategy>_<tokens>.parquet, plus quality stats.

ONE FILE PER CONFIGURATION
  fixed_384.parquet, recursive_384.parquet, page_aware_384.parquet, parent_child_200.parquet
  (+ parent_child_200_parents.parquet), with a "_ctx" suffix when contextual headers are on.
  Each file has a sidecar manifest (inputs hash = ingestion output + chunking settings +
  tokenizer + CHUNKER_VERSION), so re-running an unchanged configuration is skipped.

HOW WE JUDGE A CHUNKING STRATEGY (before any retrieval exists)
  chunk_stats()        sizes (p50/p95/max tokens), share of chunks spanning >1 page, and share
                       over the model limit (recounted with the real tokenizer: must be 0)
  evidence_coverage()  for each FinanceBench evidence item, among the chunks that cover its page:
                         best_chunk_overlap  can ONE chunk carry the evidence? (small chunks: no)
                         union_overlap       is any evidence text lost? (must be ~1.0)
  These bound what retrieval can achieve: if the evidence is spread over 3 chunks, a top-1
  retriever can't return all of it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from finsight.chunking.base import Chunker
from finsight.chunking.contextual import HEADER_RESERVE, add_context_header
from finsight.chunking.parent_child import ParentChildChunker
from finsight.chunking.registry import build_chunker
from finsight.chunking.tokenizer import Tokenizer, get_tokenizer
from finsight.config import MAX_EMBED_TOKENS, Settings
from finsight.data import datasets as ds
from finsight.errors import DataNotFoundError
from finsight.ingestion import pipeline as ingest
from finsight.ingestion.quality import evidence_overlap
from finsight.schemas import Chunk, DocumentMeta, Page

CHUNKER_VERSION = "v2"  # bump when chunking logic changes, to force a rebuild


@dataclass
class ChunkReport:
    name: str = ""
    n_docs: int = 0
    n_chunks: int = 0
    n_parents: int = 0
    elapsed_s: float = 0.0
    skipped: bool = False
    path: Path | None = None
    parents_path: Path | None = None
    stats: dict[str, Any] = field(default_factory=dict)


def config_name(settings: Settings) -> str:
    c = settings.chunking
    size = c.child_tokens if c.strategy == "parent_child" else c.chunk_tokens
    return f"{c.strategy}_{size}" + ("_ctx" if c.contextual_headers else "")


def chunks_path(settings: Settings) -> Path:
    return settings.paths.chunks_dir / f"{config_name(settings)}.parquet"


def parents_path(settings: Settings) -> Path:
    return settings.paths.chunks_dir / f"{config_name(settings)}_parents.parquet"


def _inputs_hash(settings: Settings, tokenizer: Tokenizer) -> str:
    ingest_manifest = settings.paths.processed_dir / ingest.MANIFEST_FILE
    ingest_inputs = (
        json.loads(ingest_manifest.read_text()).get("inputs_sha256")
        if ingest_manifest.is_file()
        else None
    )
    blob = json.dumps(
        {
            "pages": ingest_inputs,
            "chunking": settings.chunking.model_dump(),
            "tokenizer": tokenizer.name,
            "version": CHUNKER_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def make_chunker(settings: Settings, tokenizer: Tokenizer) -> Chunker:
    reserve = HEADER_RESERVE if settings.chunking.contextual_headers else 0
    return build_chunker(settings.chunking, tokenizer, reserve_tokens=reserve)


def chunk_document(
    chunker: Chunker, pages: list[Page], meta: DocumentMeta, contextual: bool = False
) -> list[Chunk]:
    chunks = chunker.chunk(pages, meta)
    return [add_context_header(c) for c in chunks] if contextual else chunks


def chunk_corpus(
    settings: Settings,
    *,
    tokenizer: Tokenizer | None = None,
    force: bool = False,
    progress: bool = True,
    pages_df: pd.DataFrame | None = None,
) -> ChunkReport:
    """Chunk every document with the configured strategy and save the result."""
    start = time.perf_counter()
    tokenizer = tokenizer or get_tokenizer(settings.chunking.tokenizer)
    out_path = chunks_path(settings)
    report = ChunkReport(name=config_name(settings), path=out_path)
    is_pc = settings.chunking.strategy == "parent_child"
    if is_pc:
        report.parents_path = parents_path(settings)
    manifest = out_path.with_suffix(".manifest.json")
    inputs = _inputs_hash(settings, tokenizer)

    if (
        not force
        and manifest.is_file()
        and out_path.is_file()
        and json.loads(manifest.read_text()).get("inputs_sha256") == inputs
    ):
        report.skipped = True
        existing = pd.read_parquet(out_path, columns=["doc_name"])
        report.n_chunks, report.n_docs = len(existing), existing["doc_name"].nunique()
        if is_pc and report.parents_path and report.parents_path.is_file():
            report.n_parents = len(pd.read_parquet(report.parents_path, columns=["doc_name"]))
        report.elapsed_s = round(time.perf_counter() - start, 2)
        return report

    pages_df = pages_df if pages_df is not None else ds.pages_df(settings)
    meta = ds.document_meta(settings)
    chunker = make_chunker(settings, tokenizer)
    chunks: list[Chunk] = []
    parents: list[Chunk] = []
    groups = list(pages_df.groupby("doc_name", sort=True))
    for doc_name, group in tqdm(
        groups, desc=f"Chunking {report.name}", unit="doc", disable=not progress
    ):
        if doc_name not in meta:
            continue
        pages = [Page(**row) for row in group.to_dict("records")]
        chunks.extend(
            chunk_document(chunker, pages, meta[doc_name], settings.chunking.contextual_headers)
        )
        if isinstance(chunker, ParentChildChunker):
            parents.extend(chunker.parents(pages, meta[doc_name]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_df = pd.DataFrame([c.model_dump() for c in chunks])
    chunks_df.to_parquet(out_path, index=False)
    if is_pc and report.parents_path:
        pd.DataFrame([p.model_dump() for p in parents]).to_parquet(report.parents_path, index=False)
    manifest.write_text(
        json.dumps(
            {
                "inputs_sha256": inputs,
                "config": settings.chunking.model_dump(),
                "tokenizer": tokenizer.name,
                "version": CHUNKER_VERSION,
                "n_chunks": len(chunks),
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    report.n_chunks, report.n_parents = len(chunks), len(parents)
    report.n_docs = chunks_df["doc_name"].nunique() if len(chunks_df) else 0
    report.elapsed_s = round(time.perf_counter() - start, 2)
    return report


def load_chunks_df(settings: Settings, parents: bool = False) -> pd.DataFrame:
    path = parents_path(settings) if parents else chunks_path(settings)
    if not path.is_file():
        raise DataNotFoundError(f"{path} not found. Run: uv run python scripts/04_chunk.py")
    return pd.read_parquet(path)


def chunk_stats(chunks_df: pd.DataFrame, tokenizer: Tokenizer | None = None) -> dict[str, Any]:
    """Size and boundary statistics. With a tokenizer, recounts embed_text to verify the limit."""
    tokens = chunks_df["n_tokens"]
    stats: dict[str, Any] = {
        "chunks": len(chunks_df),
        "chunks_per_doc": round(len(chunks_df) / max(chunks_df["doc_name"].nunique(), 1), 1),
        "tokens_p50": int(tokens.median()),
        "tokens_p95": int(tokens.quantile(0.95)),
        "tokens_max": int(tokens.max()),
        "multi_page_share": round(
            float((chunks_df["page_end"] > chunks_df["page_start"]).mean()), 3
        ),
    }
    if tokenizer is not None:
        recount = chunks_df["embed_text"].map(tokenizer.count)
        stats["over_limit_share"] = round(float((recount > MAX_EMBED_TOKENS).mean()), 4)
    return stats


def evidence_coverage(settings: Settings, chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Per evidence item: how well do the chunks covering its page contain it?"""
    by_page: dict[tuple[str, int], list[str]] = {}
    for row in chunks_df[["doc_name", "page_start", "page_end", "text"]].itertuples(index=False):
        for p in range(row.page_start, row.page_end + 1):
            by_page.setdefault((row.doc_name, p), []).append(row.text)
    rows = []
    for rec in ds.eval_records(settings):
        for ev in rec.evidence:
            texts = by_page.get((ev.doc_name, ev.page_idx), [])
            rows.append(
                {
                    "qid": rec.qid,
                    "doc_name": ev.doc_name,
                    "page_idx": ev.page_idx,
                    "chunks_on_page": len(texts),
                    "best_chunk_overlap": round(
                        max((evidence_overlap(ev.text, t) for t in texts), default=0.0), 3
                    ),
                    "union_overlap": round(evidence_overlap(ev.text, "\n".join(texts)), 3)
                    if texts
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def coverage_summary(coverage: pd.DataFrame) -> dict[str, float]:
    return {
        "evidence_in_one_chunk": round(float((coverage["best_chunk_overlap"] >= 0.8).mean()), 3),
        "best_chunk_overlap_mean": round(float(coverage["best_chunk_overlap"].mean()), 3),
        "union_overlap_mean": round(float(coverage["union_overlap"].mean()), 3),
        "chunks_per_evidence_page": round(float(coverage["chunks_on_page"].mean()), 1),
    }


# ------------------------------------------------------------------------------------------
# Sweep: compare strategies and sizes on the same pages
# ------------------------------------------------------------------------------------------

DEFAULT_SWEEP: list[dict[str, Any]] = [
    {"strategy": "fixed", "chunk_tokens": 256},
    {"strategy": "fixed", "chunk_tokens": 384},
    {"strategy": "fixed", "chunk_tokens": 480},
    {"strategy": "recursive", "chunk_tokens": 256},
    {"strategy": "recursive", "chunk_tokens": 384},
    {"strategy": "recursive", "chunk_tokens": 480},
    {"strategy": "page_aware", "chunk_tokens": 256},
    {"strategy": "page_aware", "chunk_tokens": 384},
    {"strategy": "page_aware", "chunk_tokens": 480},
    {"strategy": "parent_child", "child_tokens": 128},
    {"strategy": "parent_child", "child_tokens": 200},
    {"strategy": "parent_child", "child_tokens": 256},
]


def run_sweep(
    settings: Settings,
    configs: list[dict[str, Any]] | None = None,
    *,
    tokenizer: Tokenizer | None = None,
    force: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Chunk the corpus with each configuration and return one row of statistics per config."""
    from finsight.config import load_settings

    tokenizer = tokenizer or get_tokenizer(settings.chunking.tokenizer)
    pages_df = ds.pages_df(settings)
    rows = []
    for overrides in configs or DEFAULT_SWEEP:
        s = load_settings(settings.config_path, {"chunking": overrides}, root=settings.root)
        report = chunk_corpus(
            s, tokenizer=tokenizer, force=force, progress=progress, pages_df=pages_df
        )
        df = load_chunks_df(s)
        rows.append(
            {
                "config": report.name,
                "strategy": s.chunking.strategy,
                "size": s.chunking.child_tokens
                if s.chunking.strategy == "parent_child"
                else s.chunking.chunk_tokens,
                **chunk_stats(df, tokenizer),
                **coverage_summary(evidence_coverage(s, df)),
                "build_s": report.elapsed_s,
            }
        )
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """A GitHub-flavoured Markdown table (no extra dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save_sweep_report(settings: Settings, sweep: pd.DataFrame) -> Path:
    """results/<date>_chunking_sweep/{report.md, sweep.csv}: committed, so numbers are traceable."""
    out = settings.paths.results_dir / f"{datetime.now().astimezone():%Y-%m-%d}_chunking_sweep"
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "sweep.csv", index=False)
    (out / "report.md").write_text(
        "# Chunking sweep\n\n"
        f"- Tokenizer: `{settings.chunking.tokenizer}` (limit {MAX_EMBED_TOKENS} text tokens)\n"
        f"- Overlap: {settings.chunking.overlap_tokens} tokens · carry_table_header: "
        f"{settings.chunking.carry_table_header} · base fingerprint `{settings.fingerprint()}`\n"
        "- `evidence_in_one_chunk`: share of the 189 evidence items that a single chunk "
        "covering the evidence page contains (overlap ≥ 0.8).\n"
        "- `union_overlap_mean`: all chunks of that page together (text lost if < 1).\n\n"
        + to_markdown(sweep)
        + "\n"
    )
    return out
