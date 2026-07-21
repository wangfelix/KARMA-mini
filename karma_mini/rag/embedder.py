"""
Embedding client: text windows -> L2-normalized vectors.

Rows are L2-normalized so a dot product with a normalized query vector is the cosine similarity.
"""

import time
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "kit.qwen3-embedding-8b"
BATCH_SIZE = 32
MAX_RETRIES = 3


class Embedder:
    def __init__(self, client, model: str = DEFAULT_EMBED_MODEL,
                 batch_size: int = BATCH_SIZE):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts (batched). Returns float32 array [n, dim]."""

        rows: List[List[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            rows.extend(self._embed_batch(batch))
            done = start + len(batch)
            if done % (self.batch_size * 10) < self.batch_size or done == len(texts):
                logger.info(f"embedded {done}/{len(texts)} chunks")

        matrix = np.asarray(rows, dtype=np.float32)

        return _l2_normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query. Returns a normalized float32 vector [dim]."""
        vec = np.asarray(self._embed_batch([text])[0], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                return [d.embedding for d in resp.data]

            except Exception as e:  # transient API failures: back off and retry
                last_err = e
                logger.warning(f"embedding batch failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(2 * attempt)
        raise RuntimeError(f"embedding batch failed after {MAX_RETRIES} attempts: {last_err}")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
