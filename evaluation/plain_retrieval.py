"""Shared helpers for Plain-RAG retrieval and evidence records."""

from __future__ import annotations

import re
from typing import Any


PAPER_ID_PATTERN = re.compile(r"\b[a-z][a-z-]+/\d+\b", re.IGNORECASE)


def explicit_paper_scope(question: str) -> set[str]:
    """Return paper ids explicitly named in a question.

    Retained for validation only. Benchmark generator v3 emits open-corpus
    questions that never name a paper, so Plain RAG no longer derives a
    retrieval filter from question text. Filtering on a named identifier made
    paper-level retrieval correct by string matching rather than by ranking,
    which is the behaviour :func:`assert_open_corpus` now guards against.
    """
    return {match.lower() for match in PAPER_ID_PATTERN.findall(question)}


def assert_open_corpus(questions: list[dict[str, Any]]) -> None:
    """Fail loudly if any question names a paper.

    A question containing an identifier would silently reintroduce the scope
    filter and make the retrieval metrics uninterpretable, so this is checked
    before an experiment runs rather than discovered in the results.
    """
    offenders = [
        (question.get("id", "?"), sorted(named))
        for question in questions
        if (named := explicit_paper_scope(question.get("question", "")))
    ]
    if offenders:
        listed = "; ".join(f"{qid} names {ids}" for qid, ids in offenders[:5])
        raise RuntimeError(
            f"{len(offenders)} benchmark question(s) name a paper identifier: "
            f"{listed}. Open-corpus evaluation requires the retrieval system to "
            "locate the relevant papers itself."
        )


def format_retrieval_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert retriever hits to the stable Plain-RAG evidence schema."""
    evidence = []
    for rank, hit in enumerate(hits, start=1):
        chunk = hit["chunk"]
        evidence.append({
            "rank": rank,
            "chunk_id": chunk["id"],
            "paper_id": chunk["paper_id"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "section": chunk.get("section", ""),
            "text": chunk["text"],
            "retrieval_score": hit["score"],
            "bm25": hit["bm25"],
            "bm25_normalized": hit["bm25_normalized"],
            "cosine": hit["cosine"],
            "cosine_normalized": hit["cosine_normalized"],
        })
    return evidence
