"""
Polished version of the Cohere cross-encoder rerank built in 5-rerank.py.
6-evaluate.py imports from here so the eval script can focus on the metric.
"""

import os
import time
import cohere
from cohere.errors import TooManyRequestsError
import pandas as pd
from dotenv import load_dotenv
from utils.fusion import hybrid_candidates
from utils.retrievers import BM25Retriever, DenseRetriever

load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
RERANK_MODEL = "rerank-v4.0-fast"


def rerank_with_cohere(
    query: str,
    candidate_ids: list[str],
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    max_retries: int = 5,
) -> list[tuple[str, float]]:
    documents = [corpus_by_id.loc[d, "text"] for d in candidate_ids]
    for attempt in range(max_retries):
        try:
            response = co.rerank(model=RERANK_MODEL, query=query, documents=documents, top_n=k)
            return [(candidate_ids[r.index], r.relevance_score) for r in response.results]
        except (TooManyRequestsError, Exception) as e:
            if "429" in str(e) or isinstance(e, TooManyRequestsError) or "trial" in str(e).lower():
                wait = 12.0 * (attempt + 1)
                print(f"\n⏳ Cohere rate limit reached (Trial key: 10 calls/min). Waiting {wait:.1f}s (retry {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded for Cohere rerank call.")


def search_reranked(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    candidates = hybrid_candidates(query, bm25, dense, candidate_k=candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]
    return rerank_with_cohere(query, candidate_ids, corpus_by_id, k=k)