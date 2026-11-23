"""
Polished versions of the BM25 and dense retrievers built in 2-bm25.py and
3-embed.py. The later files import from here so each one can focus on the new
idea it introduces (fusion, reranking, evaluation) instead of re-loading state.

This is the same pattern Dave uses in agents/agentic-search: numbered files
teach the construction, then the cleaned-up version graduates to a helper.
"""

import os
from pathlib import Path
import bm25s
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "fiqa"
BM25_DIR = ROOT / "indexes" / "bm25"
DENSE_DIR = ROOT / "indexes" / "dense"


def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "corpus.parquet")


# --------------------------------------------------------------
# BM25
# --------------------------------------------------------------


class BM25Retriever:
    def __init__(self) -> None:
        self._retriever = bm25s.BM25.load(str(BM25_DIR))
        self._doc_ids = (BM25_DIR / "doc_ids.txt").read_text().splitlines()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([query], stopwords="en")
        indices, scores = self._retriever.retrieve(tokens, k=k)
        return [(self._doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())]


# --------------------------------------------------------------
# Dense
# --------------------------------------------------------------


class DenseRetriever:
    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        corpus = load_corpus()
        self._doc_ids = corpus["_id"].tolist()

        # Determine provider: explicitly passed > env var > auto-detect
        env_provider = os.getenv("EMBEDDING_PROVIDER")
        has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

        if provider:
            self._provider = provider.lower()
        elif env_provider:
            self._provider = env_provider.lower()
        elif has_openai_key and (DENSE_DIR / "embeddings_openai.npy").exists():
            self._provider = "openai"
        elif (DENSE_DIR / "embeddings_local.npy").exists() or not has_openai_key:
            self._provider = "local"
        else:
            self._provider = "openai"

        # Determine default model for provider
        default_models = {
            "local": "all-MiniLM-L6-v2",
            "openai": "text-embedding-3-small",
            "gemini": "gemini-embedding-001",
        }
        self._model = model or os.getenv("EMBEDDING_MODEL") or default_models.get(self._provider, "all-MiniLM-L6-v2")

        # Find the embeddings file
        candidates = [
            DENSE_DIR / f"embeddings_{self._provider}.npy",
            DENSE_DIR / "embeddings_local.npy",
            DENSE_DIR / "embeddings.npy",
        ]
        embeddings_path = next((p for p in candidates if p.exists()), None)
        if embeddings_path is None:
            raise FileNotFoundError(
                f"No embeddings file found in {DENSE_DIR}. Run 3-embed.py first to build the index."
            )

        raw = np.load(embeddings_path)
        self._embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        self._local_model = None

    def _embed_query(self, query: str) -> np.ndarray:
        if self._provider == "local":
            if self._local_model is None:
                import torch
                from sentence_transformers import SentenceTransformer

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._local_model = SentenceTransformer(self._model, device=device)

            vec = self._local_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=False,
            )[0].astype(np.float32)
            return vec / np.linalg.norm(vec)

        elif self._provider == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set. Set it in .env or switch to local provider.")
            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(model=self._model, input=[query])
            vec = np.array(response.data[0].embedding, dtype=np.float32)
            return vec / np.linalg.norm(vec)

        elif self._provider == "gemini":
            from openai import OpenAI

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINIAI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set. Set it in .env or switch to local provider.")
            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            response = client.embeddings.create(model=self._model, input=[query])
            vec = np.array(response.data[0].embedding, dtype=np.float32)
            return vec / np.linalg.norm(vec)

        else:
            raise ValueError(f"Unknown embedding provider: {self._provider}")

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self._embeddings @ self._embed_query(query)
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
