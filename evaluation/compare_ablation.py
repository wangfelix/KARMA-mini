"""Compare matched retrieval ablations on the same question set."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .paired_statistics import (
    bootstrap_mean_ci as _bootstrap_ci,
    paired_randomization_p as _paired_randomization_p,
)
from .score import deterministic_metrics, load_jsonl, retrieval_metrics


LABEL_UTILITY = {
    "fully_correct": 1.0,
    "partially_correct": 0.5,
    "incorrect": 0.0,
}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    collected: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row.items():
            collected[name].append(float(value))
    return {
        name: {"mean": _mean(values), "n": len(values)}
        for name, values in sorted(collected.items())
    }


def _validate_matched_configs(
    configs: dict[str, dict[str, Any]],
    allow_top_k_variation: bool = False,
) -> None:
    fixed_paths = (
        "question_file_sha256",
        "split",
        "answer_model",
        "answer_prompt_version",
        "answer_postprocessing",
        "answer_temperature",
    )
    first_name = next(iter(configs))
    baseline = configs[first_name]
    for name, config in configs.items():
        for key in fixed_paths:
            if config.get(key) != baseline.get(key):
                raise ValueError(
                    f"Run {name!r} differs from {first_name!r} on fixed setting {key!r}"
                )
        if (
            not allow_top_k_variation
            and config["plain_rag"]["top_k"] != baseline["plain_rag"]["top_k"]
        ):
            raise ValueError("All runs must use the same top_k")
        if config["plain_rag"]["index"] != baseline["plain_rag"]["index"]:
            raise ValueError("All runs must use the same index")


def compare(question_path: Path,
            runs: list[tuple[str, Path, Path]],
            seed: int = 20260817,
            allow_top_k_variation: bool = False) -> dict[str, Any]:
    questions = {row["id"]: row for row in load_jsonl(question_path)}
    responses_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    judgments_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    configs = {}
    judge_settings = {}

    for name, response_path, judgment_path in runs:
        responses = load_jsonl(response_path)
        judgments = load_jsonl(judgment_path)
        responses_by_run[name] = {row["question_id"]: row for row in responses}
        judgments_by_run[name] = {row["question_id"]: row for row in judgments}
        if len(responses_by_run[name]) != len(responses):
            raise ValueError(f"Duplicate response question ids in {response_path}")
        if len(judgments_by_run[name]) != len(judgments):
            raise ValueError(f"Duplicate judgment question ids in {judgment_path}")
        if set(responses_by_run[name]) != set(judgments_by_run[name]):
            raise ValueError(f"Response/judgment mismatch for {name}")
        settings = {
            (row["judge_model"], row["prompt_version"])
            for row in judgments
        }
        if len(settings) != 1:
            raise ValueError(
                f"Run {name!r} contains inconsistent judge settings"
            )
        judge_settings[name] = settings.pop()
        configs[name] = responses[0]["experiment"]

    _validate_matched_configs(configs, allow_top_k_variation)
    if len(set(judge_settings.values())) != 1:
        raise ValueError("Ablation runs use different judge settings")
    question_sets = [set(rows) for rows in responses_by_run.values()]
    if any(ids != question_sets[0] for ids in question_sets[1:]):
        raise ValueError("Ablation runs do not contain identical question ids")
    question_ids = sorted(question_sets[0])

    summaries = {}
    utility_by_run = {}
    fully_correct_by_run = {}
    category_summaries = {}
    for name in responses_by_run:
        responses = responses_by_run[name]
        judgments = judgments_by_run[name]
        labels = Counter(judgments[qid]["label"] for qid in question_ids)
        utilities = {
            qid: LABEL_UTILITY[judgments[qid]["label"]]
            for qid in question_ids
        }
        utility_by_run[name] = utilities
        fully_correct_by_run[name] = {
            qid: float(judgments[qid]["label"] == "fully_correct")
            for qid in question_ids
        }
        rubric: dict[str, list[float]] = defaultdict(list)
        for qid in question_ids:
            for metric, value in judgments[qid]["scores"].items():
                rubric[metric].append(float(value))
        summaries[name] = {
            "retrieval_mode": configs[name]["plain_rag"]["retrieval_mode"],
            "dense_weight": configs[name]["plain_rag"]["dense_weight"],
            "top_k": configs[name]["plain_rag"]["top_k"],
            "question_count": len(question_ids),
            "error_count": sum(
                responses[qid].get("status") == "error" for qid in question_ids
            ),
            "mean_latency_seconds": _mean([
                float(responses[qid]["latency_seconds"]) for qid in question_ids
            ]),
            "label_counts": {label: labels[label] for label in LABEL_UTILITY},
            "label_rates": {
                label: labels[label] / len(question_ids) for label in LABEL_UTILITY
            },
            "mean_ordinal_utility": _mean(list(utilities.values())),
            "mean_rubric_scores": {
                metric: _mean(values) for metric, values in sorted(rubric.items())
            },
            "answer_metrics": _mean_metrics([
                deterministic_metrics(questions[qid], responses[qid])
                for qid in question_ids
            ]),
            "retrieval_metrics": _mean_metrics([
                retrieval_metrics(questions[qid], responses[qid])
                for qid in question_ids
            ]),
        }

        categories: dict[str, list[str]] = defaultdict(list)
        for qid in question_ids:
            categories[questions[qid]["category"]].append(qid)
        category_summaries[name] = {}
        for category, ids in sorted(categories.items()):
            category_labels = Counter(judgments[qid]["label"] for qid in ids)
            category_summaries[name][category] = {
                "n": len(ids),
                "fully_correct_rate": category_labels["fully_correct"] / len(ids),
                "mean_ordinal_utility": _mean([utilities[qid] for qid in ids]),
                "reference_item_recall": _mean([
                    retrieval_metrics(questions[qid], responses[qid])[
                        "reference_item_recall"
                    ]
                    for qid in ids
                    if "reference_item_recall" in retrieval_metrics(
                        questions[qid], responses[qid]
                    )
                ]),
            }

    pairwise = {}
    names = list(responses_by_run)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            differences = np.asarray([
                utility_by_run[left][qid] - utility_by_run[right][qid]
                for qid in question_ids
            ], dtype=float)
            fully_correct_differences = np.asarray([
                fully_correct_by_run[left][qid]
                - fully_correct_by_run[right][qid]
                for qid in question_ids
            ], dtype=float)
            pairwise[f"{left}_minus_{right}"] = {
                "wins": int(np.sum(differences > 0)),
                "ties": int(np.sum(differences == 0)),
                "losses": int(np.sum(differences < 0)),
                "fully_correct_rate_difference": float(
                    fully_correct_differences.mean()
                ),
                "fully_correct_bootstrap_95_ci": _bootstrap_ci(
                    fully_correct_differences,
                    seed,
                ),
                "fully_correct_paired_randomization_p": (
                    _paired_randomization_p(fully_correct_differences)
                ),
                "mean_ordinal_utility_difference": float(differences.mean()),
                "bootstrap_95_ci": _bootstrap_ci(differences, seed),
                "paired_randomization_p": _paired_randomization_p(differences),
            }

    per_question = []
    for qid in question_ids:
        per_question.append({
            "question_id": qid,
            "category": questions[qid]["category"],
            "labels": {
                name: judgments_by_run[name][qid]["label"]
                for name in names
            },
        })

    fixed_settings = {
        "split": configs[names[0]]["split"],
        "answer_model": configs[names[0]]["answer_model"],
        "answer_prompt_version": configs[names[0]]["answer_prompt_version"],
        "answer_temperature": configs[names[0]]["answer_temperature"],
        "index": configs[names[0]]["plain_rag"]["index"],
        "judge_model": judge_settings[names[0]][0],
        "judge_prompt_version": judge_settings[names[0]][1],
    }
    if not allow_top_k_variation:
        fixed_settings["top_k"] = configs[names[0]]["plain_rag"]["top_k"]

    return {
        "status": "exploratory_draft_questions",
        "question_file": str(question_path),
        "seed": seed,
        "fixed_settings": fixed_settings,
        "top_k_varied": allow_top_k_variation,
        "ordinal_utility_note": (
            "Exploratory mapping only: fully_correct=1, partially_correct=0.5, "
            "incorrect=0. Do not treat as a validated interval scale."
        ),
        "summaries": summaries,
        "by_category": category_summaries,
        "pairwise": pairwise,
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/questions.jsonl"),
    )
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("NAME", "RESPONSES", "JUDGMENTS"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--allow-top-k-variation",
        action="store_true",
        help="Compare sensitivity runs whose top-k values differ.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(
        args.questions,
        [(name, Path(response), Path(judgment))
         for name, response, judgment in args.run],
        args.seed,
        args.allow_top_k_variation,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is None:
        print(rendered)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote ablation comparison to {args.out}")


if __name__ == "__main__":
    main()
