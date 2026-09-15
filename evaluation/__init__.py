"""Shared evaluation for both top-level retrieval systems.

The plain_rag and graph_rag packages implement retrieval. This package owns
the benchmark, common answer generation, judging, and paired comparison so
both systems are evaluated under the same conditions.
"""

from .calibration import (
    apply_temperature,
    calibration_metrics,
    fit_temperature,
)

__all__ = [
    "apply_temperature",
    "calibration_metrics",
    "fit_temperature",
]
