"""Configurable BM25, dense, and hybrid retrieval.

For hybrid retrieval, BM25 and embedding cosine scores are independently
min-max normalized over the candidate set before interpolation.
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


def retrieve(index: RAGIndex, embedder: Embedder | None, query: str,
             k: int = 5, mode: str = "hybrid",
             dense_weight: float = 0.5,
             paper_ids: set[str] | None = None) -> List[Dict]:
    """Return top-k chunks under a reproducible retrieval configuration.

    ``mode`` is ``bm25``, ``dense``, or ``hybrid``. In hybrid mode,
    ``dense_weight`` is alpha in
    ``(1 - alpha) * normalized_bm25 + alpha * normalized_cosine``.
    """
    if mode not in {"bm25", "dense", "hybrid"}:
        raise ValueError("mode must be one of: bm25, dense, hybrid")
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError("dense_weight must lie in [0, 1]")
    if mode in {"dense", "hybrid"} and embedder is None:
        raise ValueError(f"An embedder is required for {mode} retrieval")

    candidate_indices = np.arange(len(index.chunks), dtype=np.int64)
    if paper_ids is not None:
        normalized_paper_ids = {paper_id.lower() for paper_id in paper_ids}
        candidate_indices = np.asarray([
            i for i, chunk in enumerate(index.chunks)
            if str(chunk.get("paper_id", "")).lower() in normalized_paper_ids
        ], dtype=np.int64)
        if not candidate_indices.size:
            raise ValueError(
                "No indexed chunks matched paper_ids="
                f"{sorted(normalized_paper_ids)}"
            )

    bm25_all = np.asarray(index.bm25.scores(tokenize(query)), dtype=np.float32)
    bm25_raw = bm25_all[candidate_indices]
    bm25_normalized = _minmax(bm25_raw)

    cosine_raw = None
    cosine_normalized = None
    if mode in {"dense", "hybrid"}:
        query_vec = embedder.embed_query(query)
        cosine_raw = (index.embeddings @ query_vec)[candidate_indices]
        cosine_normalized = _minmax(cosine_raw)

    if mode == "bm25":
        combined = bm25_normalized
    elif mode == "dense":
        combined = cosine_normalized
    else:
        combined = (
            (1.0 - dense_weight) * bm25_normalized
            + dense_weight * cosine_normalized
        )

    top = np.argsort(-combined, kind="stable")[:k]
    return [
        {
            "chunk": index.chunks[int(candidate_indices[int(i)])],
            "score": float(combined[int(i)]),
            "bm25": float(bm25_raw[int(i)]),
            "bm25_normalized": float(bm25_normalized[int(i)]),
            "cosine": (
                float(cosine_raw[int(i)]) if cosine_raw is not None else None
            ),
            "cosine_normalized": (
                float(cosine_normalized[int(i)])
                if cosine_normalized is not None else None
            ),
        }
        for i in top
    ]


def hybrid_search(index: RAGIndex, embedder: Embedder, query: str,
                  k: int = 5) -> List[Dict]:
    """Return the top-k chunks by the combined, averaged hybrid score.

    Each hit: {"chunk", "score", "bm25", "cosine"} where score is the average
    of the two min-max-normalized components and bm25/cosine are raw values.
    """
    return retrieve(
        index,
        embedder,
        query,
        k=k,
        mode="hybrid",
        dense_weight=0.5,
    )
