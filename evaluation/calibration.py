"""Calibration utilities for verbalized categorical judge probabilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np


LABELS = ("fully_correct", "partially_correct", "incorrect")
EPSILON = 1e-12


def _probability_array(probabilities: Iterable[Iterable[float]]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError(f"Expected an N x {len(LABELS)} probability array")
    if len(values) == 0:
        raise ValueError("At least one probability row is required")
    if np.any(values < 0) or np.any(~np.isfinite(values)):
        raise ValueError("Probabilities must be finite and non-negative")
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each probability row needs positive mass")
    return values / row_sums


def inverse_softmax_logits(probabilities: Iterable[Iterable[float]]) -> np.ndarray:
    """Return centered log-probability proxies for inaccessible model logits."""
    values = np.clip(_probability_array(probabilities), EPSILON, 1.0)
    logits = np.log(values)
    return logits - logits.mean(axis=1, keepdims=True)


def apply_temperature(probabilities: Iterable[Iterable[float]],
                      temperature: float) -> np.ndarray:
    """Apply temperature scaling to inverse-softmax logit proxies."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    logits = inverse_softmax_logits(probabilities) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def negative_log_likelihood(probabilities: Iterable[Iterable[float]],
                            labels: Iterable[int]) -> float:
    values = _probability_array(probabilities)
    targets = np.asarray(list(labels), dtype=np.int64)
    if targets.shape != (len(values),):
        raise ValueError("labels must contain one integer per probability row")
    if np.any(targets < 0) or np.any(targets >= values.shape[1]):
        raise ValueError("label index out of range")
    chosen = np.clip(values[np.arange(len(values)), targets], EPSILON, 1.0)
    return float(-np.log(chosen).mean())


def fit_temperature(probabilities: Iterable[Iterable[float]],
                    labels: Iterable[int]) -> float:
    """Fit one scalar temperature by minimizing NLL on held-out labels.

    A dense log-spaced search is deterministic, dependency-free, and adequate
    for the single scalar optimized here.
    """
    values = _probability_array(probabilities)
    targets = list(labels)
    candidates = np.geomspace(0.05, 20.0, num=1200)
    losses = np.asarray([
        negative_log_likelihood(apply_temperature(values, float(t)), targets)
        for t in candidates
    ])
    return float(candidates[int(losses.argmin())])


def expected_calibration_error(probabilities: Iterable[Iterable[float]],
                               labels: Iterable[int], n_bins: int = 10) -> float:
    values = _probability_array(probabilities)
    targets = np.asarray(list(labels), dtype=np.int64)
    predictions = values.argmax(axis=1)
    confidences = values.max(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences >= lower) & (
            confidences <= upper if index == n_bins - 1 else confidences < upper
        )
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(
            float(correct[mask].mean()) - float(confidences[mask].mean())
        )
    return ece


def _binary_auc(scores: np.ndarray, targets: np.ndarray) -> float | None:
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[targets.astype(bool)].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def calibration_metrics(probabilities: Iterable[Iterable[float]],
                        labels: Iterable[int], n_bins: int = 10) -> dict[str, float | None]:
    values = _probability_array(probabilities)
    targets = np.asarray(list(labels), dtype=np.int64)
    if targets.shape != (len(values),):
        raise ValueError("labels must contain one integer per probability row")
    one_hot = np.eye(values.shape[1], dtype=np.float64)[targets]
    predictions = values.argmax(axis=1)
    return {
        "accuracy": float((predictions == targets).mean()),
        "nll": negative_log_likelihood(values, targets),
        "brier": float(np.square(values - one_hot).sum(axis=1).mean()),
        "ece": expected_calibration_error(values, targets, n_bins=n_bins),
        "auroc_fully_correct": _binary_auc(values[:, 0], targets == 0),
    }

