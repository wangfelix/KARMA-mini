"""Run deterministic Plain-RAG retrieval sensitivity on a benchmark split.

The evaluator embeds each question once, then evaluates every BM25 and hybrid
configuration locally. It intentionally does not call an answer model or an
LLM judge; its output measures retrieval and provenance only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from plain_rag import Embedder, load_index, retrieve

from .plain_retrieval import assert_open_corpus, format_retrieval_hits
from .score import load_jsonl, retrieval_metrics


GRAPH_NATIVE_CATEGORIES = {
    "cross_paper_entity_list",
    "cross_paper_entity_count",
}


@dataclass(frozen=True)
class RetrievalConfig:
    """One Plain-RAG retrieval setting in the sensitivity grid."""

    name: str
    mode: str
    top_k: int
    dense_weight: float | None = None


class PrecomputedQueryEmbedder:
    """Expose precomputed query vectors through the retriever interface."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self._vectors = vectors

    def embed_query(self, query: str) -> np.ndarray:
        return self._vectors[query]


def build_configurations(k_values: Iterable[int],
                         dense_weights: Iterable[float]) -> list[RetrievalConfig]:
    """Build a deterministic BM25 and hybrid sensitivity grid."""
    unique_k = sorted(set(k_values))
    unique_weights = sorted(set(dense_weights))
    if not unique_k or any(k <= 0 for k in unique_k):
        raise ValueError("top-k values must be positive")
    if not unique_weights or any(not 0.0 <= weight <= 1.0
                                 for weight in unique_weights):
        raise ValueError("dense weights must lie in [0, 1]")

    configurations = [
        RetrievalConfig(name=f"bm25-k{k}", mode="bm25", top_k=k)
        for k in unique_k
    ]
    for weight in unique_weights:
        weight_label = f"{round(weight * 100):03d}"
        configurations.extend(
            RetrievalConfig(
                name=f"hybrid-a{weight_label}-k{k}",
                mode="hybrid",
                top_k=k,
                dense_weight=weight,
            )
            for k in unique_k
        )
    return configurations


def question_stratum(question: dict[str, Any]) -> str:
    """Separate shared-answerability from graph-native questions."""
    if question["category"] in GRAPH_NATIVE_CATEGORIES:
        return "graph_native"
    return "shared_answerability"


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question retrieval rows without imputing missing metrics."""
    collected: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row["metrics"].items():
            collected[name].append(float(value))
    return {
        "question_count": len(rows),
        "metric_means": {
            name: {"mean": _mean(values), "n": len(values)}
            for name, values in sorted(collected.items())
        },
        "mean_retrieved_chunks": _mean([
            float(row["retrieved_chunk_count"]) for row in rows
        ]),
        "mean_unique_papers": _mean([
            float(row["unique_paper_count"]) for row in rows
        ]),
        "mean_latency_ms": _mean([
            float(row["latency_ms"]) for row in rows
        ]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_questions(question_path: Path, split: str,
                      allow_draft: bool) -> list[dict[str, Any]]:
    questions = [
        question for question in load_jsonl(question_path)
        if split == "all" or question["split"] == split
    ]
    if not questions:
        raise RuntimeError("No questions matched the requested split")
    unverified = [
        question["id"] for question in questions
        if question.get("review_status") != "human_verified"
    ]
    if unverified and not allow_draft:
        raise RuntimeError(
            f"Refusing to run {len(unverified)} unverified questions. "
            "Pass --allow-draft for an explicitly exploratory analysis."
        )
    return questions


def _compact_hits(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain exact rankings and scores without duplicating corpus text."""
    fields = (
        "rank",
        "chunk_id",
        "paper_id",
        "start_line",
        "end_line",
        "section",
        "retrieval_score",
        "bm25",
        "bm25_normalized",
        "cosine",
        "cosine_normalized",
    )
    return [{field: item[field] for field in fields} for item in evidence]


def run_sensitivity(question_path: Path, index_path: Path, split: str,
                    k_values: list[int], dense_weights: list[float],
                    timeout: float, allow_draft: bool) -> dict[str, Any]:
    """Evaluate the retrieval grid and return a reproducible JSON report."""
    questions = _select_questions(question_path, split, allow_draft)
    assert_open_corpus(questions)
    configurations = build_configurations(k_values, dense_weights)
    index = load_index(str(index_path))

    api_key = os.getenv("KIT_API_KEY")
    base_url = os.getenv("KIT_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("KIT_API_KEY and KIT_BASE_URL must be set")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    embedder = Embedder(client, model=index.meta["embed_model"])

    question_texts = [question["question"] for question in questions]
    matrix = embedder.embed_texts(question_texts)
    query_embedder = PrecomputedQueryEmbedder(dict(zip(question_texts, matrix)))

    rows_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        for configuration in configurations:
            started = time.perf_counter()
            hits = retrieve(
                index,
                query_embedder if configuration.mode == "hybrid" else None,
                question["question"],
                k=configuration.top_k,
                mode=configuration.mode,
                dense_weight=configuration.dense_weight or 0.0,
                paper_ids=None,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            evidence = format_retrieval_hits(hits)
            paper_ids = {item["paper_id"] for item in evidence}
            rows_by_config[configuration.name].append({
                "question_id": question["id"],
                "category": question["category"],
                "stratum": question_stratum(question),
                "metrics": retrieval_metrics(question, {"evidence": evidence}),
                "retrieved_chunk_count": len(evidence),
                "unique_paper_count": len(paper_ids),
                "latency_ms": latency_ms,
                "paper_scope_filter": sorted(scope),
                "hits": _compact_hits(evidence),
            })

    summaries = {}
    for configuration in configurations:
        rows = rows_by_config[configuration.name]
        strata = sorted({row["stratum"] for row in rows})
        categories = sorted({row["category"] for row in rows})
        summaries[configuration.name] = {
            "configuration": asdict(configuration),
            "overall": summarize(rows),
            "by_stratum": {
                stratum: summarize([
                    row for row in rows if row["stratum"] == stratum
                ])
                for stratum in strata
            },
            "by_category": {
                category: summarize([
                    row for row in rows if row["category"] == category
                ])
                for category in categories
            },
        }

    return {
        "status": (
            "exploratory_auto_generated_questions"
            if allow_draft else "human_verified_questions"
        ),
        "analysis": "retrieval_only_sensitivity",
        "question_file": str(question_path.resolve()),
        "question_file_sha256": _file_sha256(question_path),
        "split": split,
        "question_count": len(questions),
        "index": str(index_path.resolve()),
        "index_metadata": index.meta,
        "embedding_model": index.meta["embed_model"],
        "configurations": [asdict(config) for config in configurations],
        "summaries": summaries,
        "per_configuration": rows_by_config,
        "notes": [
            "No answer model or LLM judge was called.",
            "Reference-item recall is strict normalized lexical coverage.",
            "Gold paper provenance metrics are omitted for questions with no "
            "positive evidence, including unanswerable questions.",
            "Generation quality must be evaluated separately before freezing k.",
        ],
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/questions.jsonl"),
    )
    parser.add_argument("--index", type=Path, default=Path("data/rag"))
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 5, 8, 10])
    parser.add_argument(
        "--dense-weight",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = run_sensitivity(
        args.questions,
        args.index,
        args.split,
        args.top_k,
        args.dense_weight,
        args.timeout,
        args.allow_draft,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote retrieval sensitivity report to {args.out}")


if __name__ == "__main__":
    main()
