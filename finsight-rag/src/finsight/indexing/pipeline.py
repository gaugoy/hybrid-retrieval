"""
Indexing pipeline: data/chunks/<config>.parquet -> Qdrant collection + BM25 index.

    chunks ─embed_text─► embedder (+ SQLite cache) ─vectors─► Qdrant "<dataset>__<config>__<model>"
           └embed_text─► bm25s tokenize + index ────────────► indexes/bm25/<config>/

Both indexes are built from chunk.embed_text (which equals chunk.text unless contextual headers
are on), so dense and sparse search see exactly the same words.

IDEMPOTENT
  Each index has a manifest entry (manifest.py). A run is skipped when the chunk content, the
  model and INDEX_VERSION are unchanged AND the index on disk is complete (point count / file
  present). `force=True` rebuilds. Even a forced rebuild is cheap the second time: vectors come
  from the embedding cache.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from qdrant_client import QdrantClient

from finsight.chunking import pipeline as cp
from finsight.config import Settings
from finsight.data import datasets as ds
from finsight.embedding.base import Embedder, build_embedder, embedder_slug
from finsight.embedding.cache import EmbeddingCache, embed_with_cache
from finsight.indexing.bm25_store import CHUNK_IDS_FILE, BM25Index
from finsight.indexing.manifest import Manifest, chunks_sha256
from finsight.indexing.qdrant_store import QdrantStore, open_client
from finsight.observability.logging import get_logger

log = get_logger(__name__)

INDEX_VERSION = "v1"  # bump when indexing logic changes, to force a rebuild
MANIFEST_FILE = "manifest.json"

Only = Literal["all", "dense", "sparse"]


def collection_name(settings: Settings, slug: str | None = None, tag: str | None = None) -> str:
    """'<dataset>__<chunking>__<embedder>[__<tag>]'.

    e.g. financebench__page_aware_384__bge-small-en-v1-5

    `tag` marks an index over a SUBSET of documents (e.g. 'q30' = the filings of the first 30
    questions), used to compare rate-limited free-tier embedders cheaply.
    """
    slug = slug or embedder_slug(settings.embedding)
    name = f"{settings.dataset}__{cp.config_name(settings)}__{slug}"
    return f"{name}__{tag}" if tag else name


def bm25_name(settings: Settings, tag: str | None = None) -> str:
    return f"bm25__{cp.config_name(settings)}" + (f"__{tag}" if tag else "")


def bm25_dir(settings: Settings, tag: str | None = None) -> Path:
    return settings.paths.bm25_dir / (cp.config_name(settings) + (f"__{tag}" if tag else ""))


def manifest(settings: Settings) -> Manifest:
    return Manifest(settings.paths.index_dir / MANIFEST_FILE)


def usage_dir(settings: Settings) -> Path:
    """Per-provider request/token counts (daily/monthly caps survive restarts)."""
    return settings.paths.index_dir / "usage"


def embed_cache_path(settings: Settings, slug: str) -> Path:
    return settings.paths.embed_cache_dir / f"{slug}.sqlite"


def make_embedder(settings: Settings) -> Embedder:
    """The configured embedder with keys, usage tracking and (online) a query-vector cache."""
    embedder = build_embedder(
        settings.embedding, secrets=settings.secrets, usage_dir=usage_dir(settings)
    )
    if hasattr(embedder, "query_cache"):
        embedder.query_cache = EmbeddingCache(embed_cache_path(settings, embedder.slug))
    return embedder


def subset_for_questions(settings: Settings, n_questions: int) -> tuple[list[str], str]:
    """Doc names needed by the first n questions, and the tag for their subset index."""
    records = ds.eval_records(settings)[:n_questions]
    docs = sorted({ev.doc_name for r in records for ev in r.evidence})
    return docs, f"q{n_questions}"


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return round(sum(p.stat().st_size for p in files) / 1e6, 1)


@dataclass
class IndexReport:
    collection: str = ""
    bm25_dir: str = ""
    n_chunks: int = 0
    embed_model: str = ""
    device: str = ""
    dim: int = 0
    dense_skipped: bool = False
    bm25_skipped: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    embed_s: float = 0.0
    chunks_per_s: float = 0.0
    upsert_s: float = 0.0
    bm25_s: float = 0.0
    qdrant_points: int = 0
    bm25_docs: int = 0
    bm25_vocab: int = 0
    qdrant_mb: float = 0.0
    bm25_mb: float = 0.0
    cache_mb: float = 0.0
    api_requests: int = 0  # online providers: requests sent this run
    api_est_tokens: int = 0  # ... and estimated tokens
    subset: str = ""  # e.g. "q30": only the filings of the first 30 questions
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dense(
    settings: Settings,
    chunks_df: pd.DataFrame,
    embedder: Embedder,
    store: QdrantStore,
    report: IndexReport,
    *,
    content_sha: str,
    force: bool = False,
    progress: bool = True,
) -> None:
    name = report.collection
    m = manifest(settings)
    expected = {
        "chunks_sha256": content_sha,
        "embed_model": embedder.model_name,
        "version": INDEX_VERSION,
    }
    if not force and not m.needs_rebuild(name, **expected) and store.count(name) == len(chunks_df):
        report.dense_skipped = True
        report.dim = int((m.entry(name) or {}).get("dim", 0))
    else:
        cache_file = embed_cache_path(settings, embedder.slug)
        cache = EmbeddingCache(cache_file)
        try:
            vectors, stats = embed_with_cache(
                embedder, chunks_df["embed_text"].tolist(), cache, progress=progress
            )
        finally:
            cache.close()
        report.cache_hits, report.cache_misses, report.embed_s = (
            stats.hits,
            stats.misses,
            stats.embed_s,
        )
        if stats.misses and stats.embed_s:
            report.chunks_per_s = round(stats.misses / stats.embed_s, 1)
        report.dim = int(vectors.shape[1])

        start = time.perf_counter()
        store.recreate(name, report.dim)
        store.upsert_chunks(name, chunks_df, vectors, progress=progress)
        report.upsert_s = round(time.perf_counter() - start, 2)
        m.record(
            name,
            kind="dense",
            n_chunks=len(chunks_df),
            dim=report.dim,
            chunks_file=str(cp.chunks_path(settings).relative_to(settings.root)),
            config_fingerprint=settings.fingerprint(),
            **expected,
        )
    report.qdrant_points = store.count(name)
    report.device = embedder.device if not report.dense_skipped else "(skipped)"
    api_stats = getattr(embedder, "stats", None)
    if api_stats is not None:
        report.api_requests, report.api_est_tokens = api_stats.requests, api_stats.est_tokens


def build_sparse(
    settings: Settings,
    chunks_df: pd.DataFrame,
    report: IndexReport,
    *,
    content_sha: str,
    force: bool = False,
    tag: str | None = None,
) -> BM25Index:
    name, directory = bm25_name(settings, tag), bm25_dir(settings, tag)
    m = manifest(settings)
    expected = {"chunks_sha256": content_sha, "version": INDEX_VERSION}
    ids_file = directory / CHUNK_IDS_FILE
    complete = ids_file.is_file() and len(ids_file.read_text().splitlines()) == len(chunks_df)
    if not force and complete and not m.needs_rebuild(name, **expected):
        report.bm25_skipped = True
        index = BM25Index.load(directory)
    else:
        start = time.perf_counter()
        index = BM25Index.build(chunks_df["chunk_id"].tolist(), chunks_df["embed_text"].tolist())
        index.save(directory)
        report.bm25_s = round(time.perf_counter() - start, 2)
        m.record(
            name,
            kind="sparse",
            n_chunks=len(chunks_df),
            vocab_size=index.vocab_size,
            path=str(directory.relative_to(settings.root)),
            config_fingerprint=settings.fingerprint(),
            **expected,
        )
    report.bm25_docs, report.bm25_vocab = len(index), index.vocab_size
    return index


def index_corpus(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    client: QdrantClient | None = None,
    chunks_df: pd.DataFrame | None = None,
    force: bool = False,
    progress: bool = True,
    only: Only = "all",
    doc_names: list[str] | None = None,
    tag: str | None = None,
) -> IndexReport:
    """Build (or skip) the dense and sparse indexes for the configured chunking and embedder.

    `doc_names` + `tag` index only those documents (collection/BM25 names get "__<tag>").
    """
    start = time.perf_counter()
    chunks_df = chunks_df if chunks_df is not None else cp.load_chunks_df(settings)
    if doc_names is not None:
        if not tag:
            raise ValueError("a document subset needs a tag (it becomes part of the index name)")
        chunks_df = chunks_df[chunks_df["doc_name"].isin(doc_names)].reset_index(drop=True)
    embedder = embedder or make_embedder(settings)
    report = IndexReport(
        collection=collection_name(settings, embedder.slug, tag),
        bm25_dir=str(bm25_dir(settings, tag).relative_to(settings.root)),
        n_chunks=len(chunks_df),
        embed_model=embedder.model_name,
        subset=tag or "",
    )
    content_sha = chunks_sha256(chunks_df["chunk_id"], chunks_df["embed_text"])

    if only in ("all", "dense"):
        own_client = client is None
        client = client or open_client(settings)
        try:
            build_dense(
                settings,
                chunks_df,
                embedder,
                QdrantStore(client),
                report,
                content_sha=content_sha,
                force=force,
                progress=progress,
            )
        finally:
            if own_client:
                client.close()
        report.qdrant_mb = dir_size_mb(settings.paths.qdrant_dir / "collection" / report.collection)
        report.cache_mb = dir_size_mb(embed_cache_path(settings, embedder.slug))
    if only in ("all", "sparse"):
        build_sparse(settings, chunks_df, report, content_sha=content_sha, force=force, tag=tag)
        report.bm25_mb = dir_size_mb(bm25_dir(settings, tag))

    report.elapsed_s = round(time.perf_counter() - start, 2)
    log.info("indexing done", extra={k: v for k, v in report.as_dict().items() if v})
    return report


def load_bm25(settings: Settings, tag: str | None = None) -> BM25Index:
    return BM25Index.load(bm25_dir(settings, tag))
