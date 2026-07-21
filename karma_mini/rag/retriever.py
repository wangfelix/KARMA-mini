"""
Hybrid retrieval: BM25 + cosine, combined average score.

Every chunk gets a BM25 score and an embedding cosine similarity;
both are min-max normalized over the collection and the
final ranking score is their plain average.
"""

from typing import Dict, List

import numpy as np

from .bm25 import tokenize
from .embedder import Embedder
from .index import RAGIndex


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def hybrid_search(index: RAGIndex, embedder: Embedder, query: str,
                  k: int = 5) -> List[Dict]:
    """Return the top-k chunks by the combined, averaged hybrid score.

    Each hit: {"chunk", "score", "bm25", "cosine"} where score is the average
    of the two min-max-normalized components and bm25/cosine are raw values.
    """
    bm25_raw = np.asarray(index.bm25.scores(tokenize(query)), dtype=np.float32)
    query_vec = embedder.embed_query(query)
    cosine_raw = index.embeddings @ query_vec  # rows normalized -> cosine

    combined = (_minmax(bm25_raw) + _minmax(cosine_raw)) / 2.0

    top = np.argsort(-combined)[:k]
    return [
        {
            "chunk": index.chunks[int(i)],
            "score": float(combined[int(i)]),
            "bm25": float(bm25_raw[int(i)]),
            "cosine": float(cosine_raw[int(i)]),
        }
        for i in top
    ]
