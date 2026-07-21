"""RAG over the NCG trial papers (baseline system for the GraphRAG comparison).

Corpus: the raw *-Stanza-out.txt files of all papers under a trial root.
Retrieval: hybrid BM25 + embedding-cosine, combined by averaging the two
min-max-normalized scores. Generation: an LLM answering from the retrieved
excerpts with [paper:lines] citations.
"""

from .corpus import build_chunks
from .bm25 import BM25Index, tokenize
from .embedder import Embedder
from .index import build_index, load_index, RAGIndex
from .retriever import hybrid_search
from .generator import answer

__all__ = [
    "build_chunks", "BM25Index", "tokenize", "Embedder",
    "build_index", "load_index", "RAGIndex", "hybrid_search", "answer",
]
