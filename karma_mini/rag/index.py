"""
The "vector database": build, persist, and load the RAG index.

On disk (default data/rag/):
    chunks.jsonl      one chunk per line (id, paper_id, lines, section, text)
    embeddings.npy    float32 [n_chunks, dim], rows L2-normalized
    meta.json         embed model, chunking params, counts

BM25 is rebuilt from the chunk texts at load time (cheap, avoids pickling).
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .corpus import build_chunks, WINDOW, STRIDE
from .bm25 import BM25Index, tokenize
from .embedder import Embedder, DEFAULT_EMBED_MODEL

logger = logging.getLogger(__name__)


@dataclass
class RAGIndex:
    chunks: List[Dict]
    embeddings: np.ndarray   # [n_chunks, dim], L2-normalized
    bm25: BM25Index
    meta: Dict


def build_index(client, trial_root: str, out_dir: str,
                embed_model: str = DEFAULT_EMBED_MODEL) -> RAGIndex:
    """Chunk the corpus, embed every chunk, and persist the index."""
    chunks = build_chunks(trial_root)
    if not chunks:
        raise RuntimeError(f"No chunks produced from {trial_root}")

    embedder = Embedder(client, model=embed_model)
    embeddings = embedder.embed_texts([c["text"] for c in chunks])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    meta = {
        "embed_model": embed_model,
        "dim": int(embeddings.shape[1]),
        "n_chunks": len(chunks),
        "window": WINDOW,
        "stride": STRIDE,
        "trial_root": trial_root,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Index built: {len(chunks)} chunks, dim={meta['dim']} -> {out_dir}")
    return RAGIndex(chunks, embeddings,
                    BM25Index([tokenize(c["text"]) for c in chunks]), meta)


def load_index(out_dir: str) -> RAGIndex:
    """Load a persisted index and rebuild the BM25 stats."""
    chunks_path = os.path.join(out_dir, "chunks.jsonl")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"No index at {out_dir} — build it first: python rag.py index")
    chunks = [json.loads(line) for line in open(chunks_path, encoding="utf-8")]
    embeddings = np.load(os.path.join(out_dir, "embeddings.npy"))
    meta = json.load(open(os.path.join(out_dir, "meta.json"), encoding="utf-8"))
    if len(chunks) != embeddings.shape[0]:
        raise RuntimeError(f"Index corrupt: {len(chunks)} chunks vs "
                           f"{embeddings.shape[0]} embedding rows")
    return RAGIndex(chunks, embeddings,
                    BM25Index([tokenize(c["text"]) for c in chunks]), meta)
