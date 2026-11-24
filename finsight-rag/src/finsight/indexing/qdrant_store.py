"""
Dense index: chunk vectors + metadata in Qdrant.

EMBEDDED VS SERVER
  vector_store.qdrant_url = null  ->  embedded ("local") mode: Qdrant runs inside this Python
                                      process and stores data under indexes/qdrant/. No Docker.
  vector_store.qdrant_url = URL   ->  a Qdrant server (Docker, P10, or Qdrant Cloud).
  Embedded mode LOCKS its folder: only one process can open it at a time (don't run the API and
  a script together). It also searches by brute force and ignores payload indexes: fine for
  ~30k chunks (a few ms per query), but a server with HNSW is the production setup.

WHAT IS STORED PER CHUNK (a "point")
  id       uuid5(chunk_id): Qdrant ids must be an unsigned int or a UUID; uuid5 is deterministic,
           so re-upserting the same chunk overwrites it instead of duplicating it
  vector   the L2-normalized embedding of chunk.embed_text (distance: COSINE)
  payload  the whole Chunk (text, doc_name, company, doc_period, pages, ...), so a search
           result is a complete chunk, and company / doc_period / doc_type can be FILTERED on

FILTERING
  build_filter(QueryFilters) -> a Qdrant Filter: (company IN [...]) AND (doc_period IN [...])...
  Qdrant applies it DURING the search, so you still get k results that all match. (Filtering
  after the search could leave you with fewer than k.)

More info: https://qdrant.tech/documentation/concepts/ · .../concepts/filtering/
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from finsight.config import Settings
from finsight.observability.logging import get_logger
from finsight.schemas import QueryFilters

log = get_logger(__name__)

UPSERT_BATCH = 256
FILTER_FIELDS = {  # QueryFilters attribute -> (payload key, payload index type)
    "companies": ("company", models.PayloadSchemaType.KEYWORD),
    "doc_periods": ("doc_period", models.PayloadSchemaType.INTEGER),
    "doc_types": ("doc_type", models.PayloadSchemaType.KEYWORD),
}


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


@contextmanager
def _quiet_local_warnings() -> Iterator[None]:
    """Embedded mode warns on every large upsert/index call; we log the limitation once instead."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Local mode is not recommended")
        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        yield


def open_client(settings: Settings) -> QdrantClient:
    """Embedded Qdrant at paths.qdrant_dir, or a server when vector_store.qdrant_url is set."""
    url = settings.vector_store.qdrant_url
    if url:
        return QdrantClient(url=url)
    settings.paths.qdrant_dir.mkdir(parents=True, exist_ok=True)
    with _quiet_local_warnings():  # it warns on open when a collection has > 20k points
        return QdrantClient(path=str(settings.paths.qdrant_dir))


def is_local(client: QdrantClient) -> bool:
    return type(client._client).__name__ == "QdrantLocal"


def build_filter(filters: QueryFilters | None) -> models.Filter | None:
    if filters is None:
        return None
    must: list[models.Condition] = []
    for attr, (key, _) in FILTER_FIELDS.items():
        values = getattr(filters, attr)
        if values:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
    return models.Filter(must=must) if must else None


@dataclass
class DenseHit:
    payload: dict[str, Any]
    score: float


class QdrantStore:
    def __init__(self, client: QdrantClient) -> None:
        self.client = client

    def recreate(self, name: str, dim: int) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )
        if is_local(self.client):
            log.info(
                "embedded Qdrant: brute-force search, payload indexes ignored, one process at "
                "a time (use a Qdrant server for production, P10)"
            )
            return
        for key, schema in FILTER_FIELDS.values():
            self.client.create_payload_index(name, key, schema)

    def upsert_chunks(
        self, name: str, chunks_df: pd.DataFrame, vectors: np.ndarray, progress: bool = False
    ) -> None:
        records = chunks_df.to_dict("records")
        with _quiet_local_warnings():
            for i in tqdm(
                range(0, len(records), UPSERT_BATCH),
                desc="Upserting to Qdrant",
                unit="batch",
                disable=not progress,
            ):
                batch = records[i : i + UPSERT_BATCH]
                self.client.upsert(
                    name,
                    points=[
                        models.PointStruct(
                            id=point_id(r["chunk_id"]),
                            vector=vectors[i + j].tolist(),
                            payload=_clean(r),
                        )
                        for j, r in enumerate(batch)
                    ],
                    wait=True,
                )

    def count(self, name: str) -> int:
        if not self.client.collection_exists(name):
            return 0
        return int(self.client.count(name, exact=True).count)

    def search(
        self, name: str, query_vector: np.ndarray, k: int, filters: QueryFilters | None = None
    ) -> list[DenseHit]:
        with _quiet_local_warnings():
            points = self.client.query_points(
                name,
                query=query_vector.tolist(),
                limit=k,
                query_filter=build_filter(filters),
                with_payload=True,
            ).points
        return [DenseHit(payload=dict(p.payload or {}), score=float(p.score)) for p in points]


def _clean(record: dict[str, Any]) -> dict[str, Any]:
    """Payload values must be JSON types: numpy ints and NaN (missing parent_id) are converted."""
    out: dict[str, Any] = {}
    for k, v in record.items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and v != v:  # NaN
            v = None
        out[k] = v
    return out
