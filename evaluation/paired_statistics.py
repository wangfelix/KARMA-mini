"""Reusable paired statistical tests for matched evaluation results."""

from __future__ import annotations

import itertools

import numpy as np


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_RANDOMIZATION_SAMPLES = 100_000
DEFAULT_SEED = 20260817


def _as_finite_vector(values: np.ndarray | list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("Paired differences must be a one-dimensional vector")
    if not len(vector):
        raise ValueError("Paired differences must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Paired differences must contain only finite values")
    return vector


def bootstrap_mean_ci(
    differences: np.ndarray | list[float],
    seed: int = DEFAULT_SEED,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = 0.95,
) -> list[float]:
    """Return a percentile bootstrap interval for a paired mean difference."""
    vector = _as_finite_vector(differences)
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie strictly between zero and one")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(samples, len(vector)))
    means = vector[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return [float(value) for value in np.quantile(means, [tail, 1.0 - tail])]


def paired_randomization_p(
    differences: np.ndarray | list[float],
    seed: int = DEFAULT_SEED,
    samples: int = DEFAULT_RANDOMIZATION_SAMPLES,
    exact_max_nonzero: int = 20,
) -> float:
    """Two-sided paired randomization test for a mean difference.

    Exact sign enumeration is used for up to ``exact_max_nonzero`` non-zero
    pairs. Larger comparisons use a deterministic Monte Carlo approximation,
    which avoids the exponential runtime of exact enumeration on the 60-item
    test set.
    """
    vector = _as_finite_vector(differences)
    nonzero = vector[vector != 0.0]
    if not len(nonzero):
        return 1.0
    if exact_max_nonzero < 0:
        raise ValueError("exact_max_nonzero must be non-negative")

    observed = abs(float(nonzero.mean()))
    magnitudes = np.abs(nonzero)
    tolerance = 1e-12

    if len(nonzero) <= exact_max_nonzero:
        extreme = 0
        total = 2 ** len(nonzero)
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            permuted = float(np.mean(magnitudes * np.asarray(signs)))
            extreme += abs(permuted) >= observed - tolerance
        return extreme / total

    if samples <= 0:
        raise ValueError("Randomization samples must be positive")
    rng = np.random.default_rng(seed)
    extreme = 0
    processed = 0
    batch_size = min(10_000, samples)
    while processed < samples:
        current_batch = min(batch_size, samples - processed)
        signs = rng.integers(
            0,
            2,
            size=(current_batch, len(nonzero)),
            dtype=np.int8,
        )
        signs = signs * 2 - 1
        permuted_means = (signs * magnitudes).mean(axis=1)
        extreme += int(np.count_nonzero(
            np.abs(permuted_means) >= observed - tolerance
        ))
        processed += current_batch

    # The plus-one correction prevents a zero Monte Carlo p-value.
    return (extreme + 1) / (samples + 1)
