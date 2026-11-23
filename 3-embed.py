"""
Dense retrieval. Embed every document once, cache the matrix as numpy,
then answer queries by cosine similarity.

Supports three embedding providers (set via env var or CLI arg):
  EMBEDDING_PROVIDER = "local" (default) | "openai" | "gemini"
  EMBEDDING_MODEL    = optional model-name override

Local (default):  sentence-transformers, no API key needed.
OpenAI:           requires OPENAI_API_KEY   (default model: text-embedding-3-small)
Gemini:           requires GEMINI_API_KEY   (default model: gemini-embedding-001)

Usage:
  python 3-embed.py                        # local (default)
  python 3-embed.py --provider openai      # OpenAI
  python 3-embed.py --provider gemini      # Gemini
  EMBEDDING_PROVIDER=gemini python 3-embed.py
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# --------------------------------------------------------------
# Configuration: provider + model selection
# --------------------------------------------------------------

# Priority: CLI arg > env var > default ("local")
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--provider", default=None)
_parser.add_argument("--model", default=None)
_args, _ = _parser.parse_known_args()

EMBEDDING_PROVIDER = (
    _args.provider
    or os.getenv("EMBEDDING_PROVIDER", "local")
).lower().strip()

EMBEDDING_MODEL = _args.model or os.getenv("EMBEDDING_MODEL")

# Default models per provider
_DEFAULT_MODELS = {
    "local": "all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
}
if EMBEDDING_PROVIDER not in _DEFAULT_MODELS:
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}'. "
        f"Choose from: {list(_DEFAULT_MODELS.keys())}"
    )
if not EMBEDDING_MODEL:
    EMBEDDING_MODEL = _DEFAULT_MODELS[EMBEDDING_PROVIDER]

print(f"🔧 Embedding provider: {EMBEDDING_PROVIDER}  |  model: {EMBEDDING_MODEL}")

DATA_DIR = Path(__file__).parent / "data" / "fiqa"
INDEX_DIR = Path(__file__).parent / "indexes" / "dense"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------
# Provider: LOCAL (sentence-transformers, no API key needed)
# --------------------------------------------------------------

def _init_local():
    """Load sentence-transformers model once. Uses GPU if available."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📦 Loading local model '{EMBEDDING_MODEL}' on {device}...")
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def _embed_local(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Embed texts locally with sentence-transformers."""
    model = _get_local_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings.astype(np.float32)


# Lazy-loaded singleton
_local_model = None

def _get_local_model():
    global _local_model
    if _local_model is None:
        _local_model = _init_local()
    return _local_model


# --------------------------------------------------------------
# Provider: OPENAI
# --------------------------------------------------------------

def _init_openai():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Add it to .env or export it.")
    return OpenAI(api_key=api_key)


def _embed_openai(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Embed texts via OpenAI API in batches."""
    client = _init_openai()
    chunks = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (OpenAI)"):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        chunks.append(
            np.array([d.embedding for d in resp.data], dtype=np.float32)
        )
    return np.vstack(chunks)


# --------------------------------------------------------------
# Provider: GEMINI
# --------------------------------------------------------------

def _init_gemini():
    from openai import OpenAI

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINIAI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Add it to .env or export it.")
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def _embed_gemini_batch(client, texts: list[str], max_retries: int = 5) -> np.ndarray:
    """Embed a single batch with retry on rate limit."""
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return np.array([d.embedding for d in resp.data], dtype=np.float32)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** attempt * 10
                print(f"\n⏳ Rate limited. Waiting {wait}s ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded for Gemini embedding batch")


def _embed_gemini(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Embed texts via Gemini API (max 100 per batch, rate-limit pacing)."""
    client = _init_gemini()
    chunks = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (Gemini)"):
        batch = texts[i : i + batch_size]
        chunks.append(_embed_gemini_batch(client, batch))
        time.sleep(1.2)  # stay under free-tier 100 req/min
    return np.vstack(chunks)


# --------------------------------------------------------------
# Unified embed function
# --------------------------------------------------------------

_PROVIDER_FN = {
    "local": _embed_local,
    "openai": _embed_openai,
    "gemini": _embed_gemini,
}


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed texts using the configured provider."""
    return _PROVIDER_FN[EMBEDDING_PROVIDER](texts)


def build_index(doc_texts: list[str]) -> np.ndarray:
    """Embed the full corpus using the selected provider."""
    return embed_batch(doc_texts)


# --------------------------------------------------------------
# Step 2: Build or load the cached embedding matrix
# --------------------------------------------------------------

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
doc_ids = corpus["_id"].tolist()

# Some FiQA docs have blank text; swap in a placeholder so row order
# stays aligned with the BM25 index.
doc_texts = [t.strip() or "[empty document]" for t in corpus["text"].tolist()]

# Cache file is per-provider so switching providers rebuilds embeddings.
embeddings_path = INDEX_DIR / f"embeddings_{EMBEDDING_PROVIDER}.npy"
metadata_path = INDEX_DIR / f"embeddings_{EMBEDDING_PROVIDER}_meta.json"

# Build a fingerprint of the corpus so we skip re-embedding if docs haven't changed.
corpus_hash = hashlib.sha256("".join(doc_texts).encode()).hexdigest()[:16]
corpus_count = len(doc_texts)


def _cache_is_valid() -> bool:
    """Check if cached embeddings match the current corpus."""
    if not embeddings_path.exists():
        return False

    # Check row count quickly without reading full file into RAM
    try:
        cached = np.load(embeddings_path, mmap_mode="r")
        if cached.shape[0] != corpus_count:
            print(f"⚠️  Cached embeddings count ({cached.shape[0]}) != docs count ({corpus_count}) — will re-embed.")
            return False
    except Exception as e:
        print(f"⚠️  Cached embeddings unreadable ({e}) — will re-embed.")
        return False

    # Check metadata if available
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            if meta.get("corpus_hash") != corpus_hash:
                print("⚠️  Corpus content changed — will re-embed.")
                return False
            if meta.get("num_docs") != corpus_count:
                print("⚠️  Corpus size changed — will re-embed.")
                return False
        except Exception:
            pass
    else:
        # File exists and shape matches; generate missing metadata now
        metadata_path.write_text(
            json.dumps(
                {
                    "corpus_hash": corpus_hash,
                    "num_docs": corpus_count,
                    "provider": EMBEDDING_PROVIDER,
                    "model": EMBEDDING_MODEL,
                },
                indent=2,
            )
        )

    return True


if _cache_is_valid():
    print(f"✅ Loading cached embeddings from {embeddings_path} ({corpus_count} docs, hash={corpus_hash})")
    doc_embeddings = np.load(embeddings_path)
else:
    print(f"🚀 Embedding {corpus_count} docs with '{EMBEDDING_PROVIDER}'...")
    doc_embeddings = build_index(doc_texts)
    np.save(embeddings_path, doc_embeddings)
    metadata_path.write_text(json.dumps({
        "corpus_hash": corpus_hash,
        "num_docs": corpus_count,
        "provider": EMBEDDING_PROVIDER,
        "model": EMBEDDING_MODEL,
    }, indent=2))
    print(f"💾 Saved embeddings + metadata to {INDEX_DIR}")

# Pre-normalize once so cosine similarity becomes a single dot product later.
doc_embeddings_normed = doc_embeddings / np.linalg.norm(
    doc_embeddings, axis=1, keepdims=True
)

# --------------------------------------------------------------
# Step 3: Query by cosine similarity
# --------------------------------------------------------------


def search_dense(query: str, k: int = 10) -> list[tuple[str, float]]:
    """Return the top-k (doc_id, similarity) pairs for a query."""
    fn = _PROVIDER_FN[EMBEDDING_PROVIDER]
    query_vec = fn([query])[0]
    query_vec /= np.linalg.norm(query_vec)
    scores = doc_embeddings_normed @ query_vec
    top_k = np.argsort(-scores)[:k]
    return [(doc_ids[i], float(scores[i])) for i in top_k]


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"\nQuery: {query}\n")
    for i, (doc_id, score) in enumerate(search_dense(query, k=5), 1):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"{i}. [{score:.3f}] {doc_id} {text}\n")
