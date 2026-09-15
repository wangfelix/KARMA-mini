"""Compare two RAG systems on the same frozen question set.

The command validates that every selected question has exactly one response
and one judgment from each system before calculating aggregate and paired
statistics. Differences are always reported as comparison minus baseline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .paired_statistics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_RANDOMIZATION_SAMPLES,
    DEFAULT_SEED,
    bootstrap_mean_ci,
    paired_randomization_p,
)
from .score import (
    build_report,
    deterministic_metrics,
    load_jsonl,
    retrieval_metrics,
)


DEFAULT_BASELINE = "plain_rag"
DEFAULT_COMPARISON = "graph_rag_gold"
LABEL_UTILITY = {
    "fully_correct": 1.0,
    "partially_correct": 0.5,
    "incorrect": 0.0,
}


def _index_records(
    records: Iterable[dict[str, Any]],
    record_type: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        question_id = record.get("question_id")
        system = record.get("system")
        if not isinstance(question_id, str) or not isinstance(system, str):
            raise ValueError(
                f"Every {record_type} must contain string question_id and system fields"
            )
        key = (question_id, system)
        if key in indexed:
            raise ValueError(f"Duplicate {record_type} for {key}")
        indexed[key] = record
    return indexed


def _selected_questions(
    question_path: Path,
    split: str,
) -> dict[str, dict[str, Any]]:
    questions = load_jsonl(question_path)
    selected = {
        row["id"]: row
        for row in questions
        if split == "all" or row.get("split") == split
    }
    if not selected:
        raise ValueError(f"No questions found for split {split!r}")
    if len(selected) != sum(
        split == "all" or row.get("split") == split for row in questions
    ):
        raise ValueError("Selected questions contain duplicate ids")
    return selected


def _expected_keys(
    question_ids: Iterable[str],
    systems: tuple[str, str],
) -> set[tuple[str, str]]:
    return {
        (question_id, system)
        for question_id in question_ids
        for system in systems
    }


def _format_keys(keys: set[tuple[str, str]], limit: int = 8) -> str:
    ordered = sorted(keys)
    rendered = ", ".join(f"{question_id}/{system}" for question_id, system in ordered[:limit])
    if len(ordered) > limit:
        rendered += f", ... ({len(ordered)} total)"
    return rendered


def _validate_complete_pairs(
    questions: dict[str, dict[str, Any]],
    responses: dict[tuple[str, str], dict[str, Any]],
    judgments: dict[tuple[str, str], dict[str, Any]],
    systems: tuple[str, str],
    split: str,
    allow_errors: bool,
) -> None:
    expected = _expected_keys(questions, systems)
    _validate_response_pairs_records(
        questions,
        responses,
        systems,
        split,
        allow_errors,
    )
    actual_judgments = set(judgments)
    missing_judgments = expected - actual_judgments
    extra_judgments = actual_judgments - expected
    if missing_judgments:
        raise ValueError(f"Missing judgments: {_format_keys(missing_judgments)}")
    if extra_judgments:
        raise ValueError(f"Unexpected judgments: {_format_keys(extra_judgments)}")

    judge_settings = set()
    for key in sorted(expected):
        question_id, _ = key
        judgment = judgments[key]
        question = questions[question_id]

        if judgment.get("split") != question.get("split"):
            raise ValueError(f"Judgment split mismatch for {key}")
        if judgment.get("category") != question.get("category"):
            raise ValueError(f"Judgment category mismatch for {key}")
        if judgment.get("label") not in LABEL_UTILITY:
            raise ValueError(f"Unknown judgment label for {key}")

        judge_settings.add((
            judgment.get("judge_model"),
            judgment.get("prompt_version"),
        ))

    if len(judge_settings) != 1 or None in next(iter(judge_settings)):
        raise ValueError("Judgments must use one recorded judge model and prompt")


def _validate_response_pairs_records(
    questions: dict[str, dict[str, Any]],
    responses: dict[tuple[str, str], dict[str, Any]],
    systems: tuple[str, str],
    split: str,
    allow_errors: bool,
) -> str:
    expected = _expected_keys(questions, systems)
    actual = set(responses)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"Missing responses: {_format_keys(missing)}")
    if extra:
        raise ValueError(f"Unexpected responses: {_format_keys(extra)}")

    experiment_ids = set()
    experiment_configs = set()
    failures = []
    for key in sorted(expected):
        question_id, _ = key
        response = responses[key]
        question = questions[question_id]
        if response.get("split") != question.get("split"):
            raise ValueError(f"Response split mismatch for {key}")
        if response.get("category") != question.get("category"):
            raise ValueError(f"Response category mismatch for {key}")
        if split != "all" and response.get("split") != split:
            raise ValueError(f"Unexpected split for {key}: {response.get('split')!r}")
        experiment_id = response.get("experiment", {}).get("experiment_id")
        if not experiment_id:
            raise ValueError(f"Response is missing experiment metadata for {key}")
        experiment_ids.add(experiment_id)
        experiment_configs.add(json.dumps(
            response["experiment"],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ))
        if response.get("review_status") != question.get("review_status"):
            raise ValueError(f"Response review status mismatch for {key}")
        if response.get("status") != "ok":
            failures.append((key, response.get("error")))

    if len(experiment_ids) != 1:
        raise ValueError(
            "Responses contain multiple experiment ids; do not mix runs in one comparison"
        )
    if len(experiment_configs) != 1:
        raise ValueError(
            "Responses contain inconsistent experiment metadata"
        )
    experiment = responses[next(iter(expected))]["experiment"]
    if experiment.get("split") != split:
        raise ValueError("Experiment metadata does not match the selected split")
    if set(experiment.get("systems", [])) != set(systems):
        raise ValueError("Experiment metadata does not match the compared systems")
    if failures and not allow_errors:
        preview = ", ".join(
            f"{question_id}/{system}: {error}"
            for (question_id, system), error in failures[:5]
        )
        raise RuntimeError(
            f"Refusing to continue with {len(failures)} failed responses. {preview}"
        )
    return next(iter(experiment_ids))


def validate_response_file(
    question_path: Path,
    response_path: Path,
    split: str = "test",
    baseline: str = DEFAULT_BASELINE,
    comparison: str = DEFAULT_COMPARISON,
    allow_errors: bool = False,
) -> dict[str, Any]:
    """Validate a complete matched response file before LLM judging."""
    questions = _selected_questions(question_path, split)
    responses = _index_records(load_jsonl(response_path), "response")
    experiment_id = _validate_response_pairs_records(
        questions,
        responses,
        (baseline, comparison),
        split,
        allow_errors,
    )
    return {
        "question_count": len(questions),
        "response_count": len(responses),
        "experiment_id": experiment_id,
    }


def _paired_summary(
    baseline_values: list[float],
    comparison_values: list[float],
    seed: int,
    bootstrap_samples: int,
    randomization_samples: int,
) -> dict[str, Any]:
    baseline_vector = np.asarray(baseline_values, dtype=float)
    comparison_vector = np.asarray(comparison_values, dtype=float)
    differences = comparison_vector - baseline_vector
    return {
        "n": len(differences),
        "baseline_mean": float(baseline_vector.mean()),
        "comparison_mean": float(comparison_vector.mean()),
        "difference": float(differences.mean()),
        "bootstrap_95_ci": bootstrap_mean_ci(
            differences,
            seed=seed,
            samples=bootstrap_samples,
        ),
        "paired_randomization_p": paired_randomization_p(
            differences,
            seed=seed,
            samples=randomization_samples,
        ),
    }


def _paired_metric_summaries(
    questions: dict[str, dict[str, Any]],
    responses: dict[tuple[str, str], dict[str, Any]],
    systems: tuple[str, str],
    seed: int,
    bootstrap_samples: int,
    randomization_samples: int,
) -> dict[str, dict[str, Any]]:
    baseline, comparison = systems
    values: dict[str, tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for question_id, question in sorted(questions.items()):
        baseline_response = responses[(question_id, baseline)]
        comparison_response = responses[(question_id, comparison)]
        metric_groups = (
            (
                "answer",
                deterministic_metrics(question, baseline_response),
                deterministic_metrics(question, comparison_response),
            ),
            (
                "retrieval",
                retrieval_metrics(question, baseline_response),
                retrieval_metrics(question, comparison_response),
            ),
        )
        for prefix, baseline_metrics, comparison_metrics in metric_groups:
            for metric in sorted(baseline_metrics.keys() & comparison_metrics.keys()):
                baseline_values, comparison_values = values[f"{prefix}.{metric}"]
                baseline_values.append(float(baseline_metrics[metric]))
                comparison_values.append(float(comparison_metrics[metric]))

    return {
        metric: _paired_summary(
            baseline_values,
            comparison_values,
            seed,
            bootstrap_samples,
            randomization_samples,
        )
        for metric, (baseline_values, comparison_values) in sorted(values.items())
    }


def _category_comparisons(
    questions: dict[str, dict[str, Any]],
    judgments: dict[tuple[str, str], dict[str, Any]],
    systems: tuple[str, str],
) -> dict[str, dict[str, Any]]:
    baseline, comparison = systems
    category_ids: dict[str, list[str]] = defaultdict(list)
    for question_id, question in questions.items():
        category_ids[question["category"]].append(question_id)

    summaries = {}
    for category, question_ids in sorted(category_ids.items()):
        baseline_labels = [
            judgments[(question_id, baseline)]["label"]
            for question_id in question_ids
        ]
        comparison_labels = [
            judgments[(question_id, comparison)]["label"]
            for question_id in question_ids
        ]
        differences = [
            LABEL_UTILITY[right] - LABEL_UTILITY[left]
            for left, right in zip(baseline_labels, comparison_labels)
        ]
        summaries[category] = {
            "n": len(question_ids),
            "baseline_fully_correct_rate": (
                baseline_labels.count("fully_correct") / len(question_ids)
            ),
            "comparison_fully_correct_rate": (
                comparison_labels.count("fully_correct") / len(question_ids)
            ),
            "fully_correct_rate_difference": (
                comparison_labels.count("fully_correct")
                - baseline_labels.count("fully_correct")
            ) / len(question_ids),
            "comparison_wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "comparison_losses": sum(value < 0 for value in differences),
        }
    return summaries


def compare_systems(
    question_path: Path,
    response_path: Path,
    judgment_path: Path,
    split: str = "test",
    baseline: str = DEFAULT_BASELINE,
    comparison: str = DEFAULT_COMPARISON,
    seed: int = DEFAULT_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    randomization_samples: int = DEFAULT_RANDOMIZATION_SAMPLES,
    allow_errors: bool = False,
) -> dict[str, Any]:
    """Build a validated aggregate and paired comparison report."""
    if baseline == comparison:
        raise ValueError("Baseline and comparison systems must be different")
    if bootstrap_samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    if randomization_samples <= 0:
        raise ValueError("Randomization samples must be positive")
    systems = (baseline, comparison)
    questions = _selected_questions(question_path, split)
    responses = _index_records(load_jsonl(response_path), "response")
    judgments = _index_records(load_jsonl(judgment_path), "judgment")
    _validate_complete_pairs(
        questions,
        responses,
        judgments,
        systems,
        split,
        allow_errors,
    )

    question_ids = sorted(questions)
    baseline_labels = [
        judgments[(question_id, baseline)]["label"] for question_id in question_ids
    ]
    comparison_labels = [
        judgments[(question_id, comparison)]["label"] for question_id in question_ids
    ]
    baseline_utility = [LABEL_UTILITY[label] for label in baseline_labels]
    comparison_utility = [LABEL_UTILITY[label] for label in comparison_labels]
    baseline_fully_correct = [
        float(label == "fully_correct") for label in baseline_labels
    ]
    comparison_fully_correct = [
        float(label == "fully_correct") for label in comparison_labels
    ]
    utility_differences = np.asarray(comparison_utility) - np.asarray(baseline_utility)

    aggregate = build_report(question_path, response_path, judgment_path)
    review_statuses = sorted({
        question.get("review_status", "unknown") for question in questions.values()
    })
    experiment = responses[(question_ids[0], baseline)]["experiment"]
    judge_model = judgments[(question_ids[0], baseline)]["judge_model"]
    judge_prompt = judgments[(question_ids[0], baseline)]["prompt_version"]

    transitions = Counter(
        f"{left}_to_{right}"
        for left, right in zip(baseline_labels, comparison_labels)
    )
    return {
        "status": (
            "complete_human_verified"
            if review_statuses == ["human_verified"]
            else "complete_unverified_questions"
        ),
        "question_file": str(question_path),
        "response_file": str(response_path),
        "judgment_file": str(judgment_path),
        "split": split,
        "question_count": len(question_ids),
        "baseline": baseline,
        "comparison": comparison,
        "difference_direction": f"{comparison}_minus_{baseline}",
        "benchmark_review_statuses": review_statuses,
        "experiment": {
            "experiment_id": experiment["experiment_id"],
            "answer_model": experiment["answer_model"],
            "answer_prompt_version": experiment["answer_prompt_version"],
            "answer_temperature": experiment["answer_temperature"],
            "plain_rag": experiment["plain_rag"],
            "graph_rag": experiment["graph_rag"],
            "judge_model": judge_model,
            "judge_prompt_version": judge_prompt,
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "randomization_samples": randomization_samples,
        },
        "systems": aggregate["systems"],
        "by_category": aggregate["by_category"],
        "primary_paired_comparison": {
            "fully_correct_rate": _paired_summary(
                baseline_fully_correct,
                comparison_fully_correct,
                seed,
                bootstrap_samples,
                randomization_samples,
            ),
            "ordinal_utility": {
                **_paired_summary(
                    baseline_utility,
                    comparison_utility,
                    seed,
                    bootstrap_samples,
                    randomization_samples,
                ),
                "scale": {
                    "fully_correct": 1.0,
                    "partially_correct": 0.5,
                    "incorrect": 0.0,
                },
                "interpretation": "Secondary descriptive score, not a validated interval scale.",
            },
            "comparison_wins": int(np.sum(utility_differences > 0)),
            "ties": int(np.sum(utility_differences == 0)),
            "comparison_losses": int(np.sum(utility_differences < 0)),
            "label_transitions": dict(sorted(transitions.items())),
        },
        "paired_metric_comparisons": _paired_metric_summaries(
            questions,
            responses,
            systems,
            seed,
            bootstrap_samples,
            randomization_samples,
        ),
        "category_comparisons": _category_comparisons(
            questions,
            judgments,
            systems,
        ),
        "per_question": [
            {
                "question_id": question_id,
                "category": questions[question_id]["category"],
                "baseline_label": judgments[(question_id, baseline)]["label"],
                "comparison_label": judgments[(question_id, comparison)]["label"],
            }
            for question_id in question_ids
        ],
        "notes": [
            "All reported differences are comparison minus baseline.",
            "The fully-correct rate is the primary automated answer metric.",
            "Question-category results are secondary diagnostics.",
            "Judge probabilities are raw and uncalibrated.",
            "Gold GraphRAG is an oracle structured-retrieval condition.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/questions.jsonl"),
    )
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
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
        "--allow-errors",
        action="store_true",
        help="Include failed responses as zero-score diagnostics.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = compare_systems(
        args.questions,
        args.responses,
        args.judgments,
        split=args.split,
        baseline=args.baseline,
        comparison=args.comparison,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        allow_errors=args.allow_errors,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"Wrote system comparison to {args.out}")


if __name__ == "__main__":
    main()
