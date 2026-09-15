"""Aggregate deterministic answer metrics and pointwise judge results."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .judge import LABELS


PAPER_ID_PATTERN = re.compile(r"\b[a-z][a-z-]+/\d+\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![/\w])\d+(?![\w])")
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
ABSTENTION_PHRASES = (
    "the supplied evidence does not contain",
    "does not contain the answer",
    "does not contain enough evidence",
    "not enough evidence",
    "no matching results",
    "cannot answer",
    "can't answer",
    "not mentioned",
    "does not report",
    "no evidence",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(_tokens(prediction))
    expected = Counter(_tokens(reference))
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _set_metrics(predicted: set[str], expected: set[str]) -> dict[str, float]:
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact": 1.0}
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": float(predicted == expected),
    }


def _integer_match(answer: str, expected: int) -> float:
    numbers = {int(value) for value in NUMBER_PATTERN.findall(answer)}
    lowered_tokens = set(_tokens(answer))
    numbers.update(value for word, value in NUMBER_WORDS.items() if word in lowered_tokens)
    return float(expected in numbers)


def item_recall(answer: str, reference_items: list[Any]) -> float:
    """Fraction of reference terms that appear anywhere in the answer.

    Open-ended aggregation questions ask what a collection reports about an
    entity. Their reference is a set of terms drawn from the annotation, and a
    fluent answer will mention some of them while phrasing the rest its own
    way. Token F1 punishes that answer for its own wording, so recall of the
    reference terms is the more meaningful measure of how much of the expected
    content was recovered.
    """
    haystack = " ".join(_tokens(answer))
    needles = [" ".join(_tokens(str(item))) for item in reference_items]
    needles = [needle for needle in needles if needle]
    if not needles:
        return 0.0
    return sum(needle in haystack for needle in needles) / len(needles)


def deterministic_metrics(question: dict[str, Any],
                          response: dict[str, Any]) -> dict[str, float]:
    answer = str(response.get("answer", ""))
    answer_type = question["answer_type"]
    if response.get("status") == "error":
        if answer_type == "paper_id_set":
            return {
                "set_precision": 0.0,
                "set_recall": 0.0,
                "set_f1": 0.0,
                "set_exact": 0.0,
            }
        if answer_type == "integer":
            return {"integer_accuracy": 0.0}
        if answer_type == "unanswerable":
            return {"abstention_accuracy": 0.0}
        if answer_type == "list":
            return {"token_f1": 0.0, "item_recall": 0.0}
        return {"token_f1": 0.0}

    if answer_type == "paper_id_set":
        predicted = {value.lower() for value in PAPER_ID_PATTERN.findall(answer)}
        expected = {str(value).lower() for value in question["reference_items"]}
        values = _set_metrics(predicted, expected)
        return {f"set_{name}": score for name, score in values.items()}
    if answer_type == "integer":
        return {
            "integer_accuracy": _integer_match(
                answer,
                int(question["reference_items"][0]),
            )
        }
    if answer_type == "unanswerable":
        lowered = answer.lower()
        return {
            "abstention_accuracy": float(
                any(phrase in lowered for phrase in ABSTENTION_PHRASES)
            )
        }
    if answer_type == "list":
        return {
            "token_f1": token_f1(answer, question["reference_answer"]),
            "item_recall": item_recall(answer, question.get("reference_items", [])),
        }
    return {"token_f1": token_f1(answer, question["reference_answer"])}


def _evidence_paper_ids(evidence: Any) -> set[str]:
    """Collect paper ids from structured evidence without reading answer text."""
    paper_ids: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                visit(nested_value, str(nested_key).lower())
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, str) and (
            "paper_id" in key or key in {"paper", "pid", "paper_ids"}
        ):
            paper_ids.update(match.lower() for match in PAPER_ID_PATTERN.findall(value))

    visit(evidence)
    return paper_ids


def _reference_item_evidence_metrics(question: dict[str, Any],
                                     evidence: Any) -> dict[str, float]:
    """Measure exact normalized reference-item coverage in retrieved evidence.

    This is a deliberately strict lexical retrieval diagnostic, not a semantic
    relevance judgment. It applies to textual list and short-answer questions;
    paper-list/count questions use paper provenance metrics instead.
    """
    if question.get("answer_type") not in {"list", "short_answer"}:
        return {}
    items = [
        " ".join(_tokens(str(item)))
        for item in question.get("reference_items", [])
        if _tokens(str(item))
    ]
    if not items:
        return {}
    evidence_text = " ".join(_tokens(json.dumps(
        evidence,
        ensure_ascii=False,
        default=str,
    )))
    matched = sum(item in evidence_text for item in items)
    return {
        "reference_item_recall": matched / len(items),
        "reference_item_all": float(matched == len(items)),
    }


def retrieval_metrics(question: dict[str, Any],
                      response: dict[str, Any]) -> dict[str, float]:
    """Paper-level evidence retrieval metrics against gold provenance.

    Questions without positive gold evidence (currently the unanswerable
    category) are omitted because an empty result is not proof that the
    requested fact is absent.
    """
    expected = {
        str(item["paper_id"]).lower()
        for item in question.get("evidence", [])
        if item.get("paper_id")
    }
    evidence = response.get("evidence", [])
    metrics = _reference_item_evidence_metrics(question, evidence)
    if expected:
        predicted = _evidence_paper_ids(evidence)
        values = _set_metrics(predicted, expected)
        metrics.update({f"paper_{name}": score for name, score in values.items()})
    return metrics


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _summarize(records: list[tuple[dict[str, Any], dict[str, Any],
                                   dict[str, Any] | None]]) -> dict[str, Any]:
    answer_metric_values: dict[str, list[float]] = defaultdict(list)
    retrieval_metric_values: dict[str, list[float]] = defaultdict(list)
    labels = Counter()
    probabilities: dict[str, list[float]] = defaultdict(list)
    rubric_scores: dict[str, list[float]] = defaultdict(list)
    latencies = []
    error_count = 0
    judged_count = 0

    for question, response, judgment in records:
        for name, value in deterministic_metrics(question, response).items():
            answer_metric_values[name].append(value)
        for name, value in retrieval_metrics(question, response).items():
            retrieval_metric_values[name].append(value)
        latencies.append(float(response.get("latency_seconds", 0.0)))
        error_count += int(response.get("status") == "error")
        if judgment is None:
            continue
        judged_count += 1
        labels[judgment["label"]] += 1
        for label in LABELS:
            probabilities[label].append(float(judgment["probabilities"][label]))
        for name, value in judgment["scores"].items():
            rubric_scores[name].append(float(value))

    response_count = len(records)
    return {
        "response_count": response_count,
        "error_count": error_count,
        "error_rate": error_count / response_count if response_count else None,
        "mean_latency_seconds": _mean(latencies),
        "answer_metrics": {
            name: {"mean": _mean(values), "n": len(values)}
            for name, values in sorted(answer_metric_values.items())
        },
        "retrieval_metrics": {
            name: {"mean": _mean(values), "n": len(values)}
            for name, values in sorted(retrieval_metric_values.items())
        },
        "judge": {
            "judged_count": judged_count,
            "label_rates": {
                label: labels[label] / judged_count if judged_count else None
                for label in LABELS
            },
            "mean_probabilities_uncalibrated": {
                label: _mean(probabilities[label]) for label in LABELS
            },
            "mean_rubric_scores_0_to_4": {
                name: _mean(values) for name, values in sorted(rubric_scores.items())
            },
            "calibration": "not_computed_without_human_labels",
        },
    }


def build_report(question_path: Path, response_path: Path,
                 judgment_path: Path | None) -> dict[str, Any]:
    questions = {record["id"]: record for record in load_jsonl(question_path)}
    responses = load_jsonl(response_path)
    judgments = {}
    if judgment_path is not None:
        for judgment in load_jsonl(judgment_path):
            key = (judgment["question_id"], judgment["system"])
            if key in judgments:
                raise ValueError(f"Duplicate judgment for {key}")
            judgments[key] = judgment

    joined = []
    for response in responses:
        question_id = response["question_id"]
        if question_id not in questions:
            raise KeyError(f"Unknown question_id in responses: {question_id}")
        key = (question_id, response["system"])
        joined.append((questions[question_id], response, judgments.get(key)))

    by_system: dict[str, list] = defaultdict(list)
    by_system_category: dict[tuple[str, str], list] = defaultdict(list)
    for item in joined:
        question, response, _ = item
        by_system[response["system"]].append(item)
        by_system_category[(response["system"], question["category"])].append(item)

    review_statuses = sorted({
        questions[response["question_id"]].get("review_status", "unknown")
        for response in responses
    })
    return {
        "question_file": str(question_path),
        "response_file": str(response_path),
        "judgment_file": str(judgment_path) if judgment_path else None,
        "benchmark_review_statuses": review_statuses,
        "systems": {
            system: _summarize(items)
            for system, items in sorted(by_system.items())
        },
        "by_category": {
            system: {
                category: _summarize(by_system_category[(system, category)])
                for category in sorted({
                    key[1] for key in by_system_category if key[0] == system
                })
            }
            for system in sorted(by_system)
        },
        "notes": [
            "Judge probabilities in this report are raw and uncalibrated.",
            "Token F1 is diagnostic for free-form answers, not a semantic correctness metric.",
            "Disclose benchmark review statuses; unverified questions are not an externally validated benchmark.",
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
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = build_report(args.questions, args.responses, args.judgments)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is None:
        print(rendered)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote score report to {args.out}")


if __name__ == "__main__":
    main()
