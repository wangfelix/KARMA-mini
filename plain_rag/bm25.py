"""
Okapi BM25 (the lexical half of the hybrid retriever). Pure Python, no deps.
"""

import math
import re
from collections import Counter
from typing import Dict, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# BM25 constants.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Inverted-index BM25 over a fixed list of documents.

    Scores are computed against all documents (score 0 when no term overlaps),
    so the result aligns with the embedding matrix regarding the index
    """

    def __init__(self, docs_tokens: List[List[str]]):
        self.n_docs = len(docs_tokens)
        self.doc_len = [len(toks) for toks in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        # term -> {doc_index: term frequency}
        self.postings: Dict[str, Dict[int, int]] = {}
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, {})[i] = tf

    def _idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: List[str]) -> List[float]:
        """BM25 score of the query against every document (dense list)."""
        out = [0.0] * self.n_docs
        if not self.n_docs:
            return out
        for term in query_tokens:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for doc_id, tf in self.postings[term].items():
                denom = tf + K1 * (1.0 - B + B * self.doc_len[doc_id] / self.avgdl)
                out[doc_id] += idf * tf * (K1 + 1.0) / denom
        return out
