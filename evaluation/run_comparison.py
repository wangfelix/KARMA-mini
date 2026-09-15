"""Run the complete frozen Plain-RAG versus Gold-GraphRAG evaluation.

This command orchestrates response generation, pointwise judging, aggregate
scoring, and paired system comparison. The JSONL stages resume compatible
partial runs unless ``--overwrite`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .compare_systems import compare_systems, validate_response_file
from .config import (
    FINAL_ANSWER_MODEL,
    FINAL_DENSE_WEIGHT,
    FINAL_JUDGE_MODEL,
    FINAL_RETRIEVAL_MODE,
    FINAL_TOP_K,
)
from .judge import run as run_judge
from .paired_statistics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_RANDOMIZATION_SAMPLES,
    DEFAULT_SEED,
)
from .run_experiment import SYSTEMS, run_experiment
from .score import build_report, load_jsonl


@dataclass(frozen=True)
class ComparisonPaths:
    responses: Path
    judgments: Path
    scores: Path
    comparison: Path
    manifest: Path

    @classmethod
    def under(cls, output_dir: Path, split: str) -> "ComparisonPaths":
        return cls(
            responses=output_dir / f"{split}-responses.jsonl",
            judgments=output_dir / f"{split}-judgments.jsonl",
            scores=output_dir / f"{split}-scores.json",
            comparison=output_dir / f"{split}-comparison.json",
            manifest=output_dir / f"{split}-run-manifest.json",
        )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_pipeline(args: argparse.Namespace) -> dict:
    """Execute all comparison stages and return the final run manifest."""
    paths = ComparisonPaths.under(args.output_dir, args.split)
    print(
        "Frozen Plain-RAG configuration: "
        f"{FINAL_RETRIEVAL_MODE}, dense_weight={FINAL_DENSE_WEIGHT}, "
        f"top_k={FINAL_TOP_K}, answer_model={args.answer_model}"
    )

    print(f"[1/4] Generating responses -> {paths.responses}")
    experiment_args = argparse.Namespace(
        questions=args.questions,
        systems=list(SYSTEMS),
        split=args.split,
        question_id=None,
        limit=None,
        out=paths.responses,
        model=args.answer_model,
        timeout=args.timeout,
        index=args.index,
        retrieval_mode=FINAL_RETRIEVAL_MODE,
        dense_weight=FINAL_DENSE_WEIGHT,
        top_k=FINAL_TOP_K,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        graph_source_label=args.graph_source_label,
        expected_graph_papers=args.expected_graph_papers,
        allow_graph_count_mismatch=args.allow_graph_count_mismatch,
        allow_draft=args.allow_unverified_questions,
        overwrite=args.overwrite,
    )
    run_experiment(experiment_args)
    response_validation = validate_response_file(
        args.questions,
        paths.responses,
        split=args.split,
    )
    print(
        "Validated "
        f"{response_validation['response_count']} responses for "
        f"{response_validation['question_count']} matched questions."
    )

    print(f"[2/4] Judging responses -> {paths.judgments}")
    run_judge(
        args.questions,
        paths.responses,
        paths.judgments,
        args.judge_model,
        args.timeout,
        overwrite=args.overwrite,
        allow_draft=args.allow_unverified_questions,
    )

    print(f"[3/4] Aggregating system metrics -> {paths.scores}")
    score_report = build_report(
        args.questions,
        paths.responses,
        paths.judgments,
    )
    _write_json(paths.scores, score_report)

    print(f"[4/4] Calculating paired comparison -> {paths.comparison}")
    comparison_report = compare_systems(
        args.questions,
        paths.responses,
        paths.judgments,
        split=args.split,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
    )
    _write_json(paths.comparison, comparison_report)

    manifest = {
        "status": comparison_report["status"],
        "split": args.split,
        "question_count": response_validation["question_count"],
        "response_count": response_validation["response_count"],
        "judgment_count": sum(
            system["judge"]["judged_count"]
            for system in score_report["systems"].values()
        ),
        "experiment_id": response_validation["experiment_id"],
        "frozen_plain_rag": {
            "retrieval_mode": FINAL_RETRIEVAL_MODE,
            "dense_weight": FINAL_DENSE_WEIGHT,
            "top_k": FINAL_TOP_K,
            "answer_model": args.answer_model,
        },
        "judge_model": args.judge_model,
        "benchmark_review_statuses": comparison_report[
            "benchmark_review_statuses"
        ],
        "outputs": {
            "responses": str(paths.responses),
            "judgments": str(paths.judgments),
            "scores": str(paths.scores),
            "comparison": str(paths.comparison),
        },
    }
    _write_json(paths.manifest, manifest)
    print(f"Comparison complete -> {paths.manifest}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/questions.jsonl"),
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_comparison"),
    )
    parser.add_argument("--index", type=Path, default=Path("data/rag"))
    parser.add_argument("--answer-model", default=FINAL_ANSWER_MODEL)
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", FINAL_JUDGE_MODEL),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USER", "neo4j"),
    )
    parser.add_argument("--graph-source-label", default="semeval_gold")
    parser.add_argument("--expected-graph-papers", type=int, default=50)
    parser.add_argument(
        "--allow-graph-count-mismatch",
        action="store_true",
        help="Diagnostic only. Do not use for the reported gold-graph run.",
    )
    parser.add_argument(
        "--allow-unverified-questions",
        action="store_true",
        help=(
            "Acknowledge that the automatically curated benchmark has no expert "
            "human verification. This status is preserved in every report."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--randomization-samples",
        type=int,
        default=DEFAULT_RANDOMIZATION_SAMPLES,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing response and judgment files instead of resuming them.",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.randomization_samples <= 0:
        parser.error("--randomization-samples must be positive")
    selected_questions = [
        question
        for question in load_jsonl(args.questions)
        if question.get("split") == args.split
    ]
    unverified_questions = [
        question["id"]
        for question in selected_questions
        if question.get("review_status") != "human_verified"
    ]
    if unverified_questions and not args.allow_unverified_questions:
        parser.error(
            f"{len(unverified_questions)} selected questions are not human "
            "verified. Pass --allow-unverified-questions to acknowledge and "
            "preserve this limitation in the reports."
        )
    run_pipeline(args)


if __name__ == "__main__":
    main()
