"""Shared helper functions for math operations.

Provides:
- argmin, argmax: index of min/max value
- logprob_to_prob, prob_to_logprob: scalar conversions
- normalize, normalize_pair: normalization utilities
"""

from __future__ import annotations

import math
from typing import Sequence

from .entropy_diversity.core import _EPS


def argmin(xs: Sequence[float]) -> int:
    """Index of the minimum value, or 0 for an empty list."""
    return min(range(len(xs)), key=lambda i: xs[i]) if xs else 0


def argmax(xs: Sequence[float]) -> int:
    """Index of the maximum value, or 0 for an empty list."""
    return max(range(len(xs)), key=lambda i: xs[i]) if xs else 0


def logprob_to_prob(logprob: float) -> float:
    """Convert a single log-probability to probability."""
    return math.exp(logprob)


def prob_to_logprob(prob: float) -> float:
    """Convert a single probability to log-probability."""
    if prob < _EPS:
        return float("-inf")
    return math.log(prob)


def normalize(values: Sequence[float]) -> list[float]:
    """Normalise non-negative values to sum to 1 using log-sum-exp."""
    if not values:
        return []

    if any(v < 0 for v in values):
        raise ValueError("values must be non-negative")

    total_linear = sum(values)
    if total_linear < _EPS:
        n = len(values)
        return [1.0 / n] * n

    logs = [(-math.inf if v == 0.0 else math.log(v)) for v in values]
    m = max(logs)

    # If any +inf slipped in (e.g., v was inf), allocate equally among them
    if math.isinf(m) and m > 0:
        inf_count = sum(1 for v in values if math.isinf(v))
        return [(1.0 / inf_count) if math.isinf(v) else 0.0 for v in values]

    log_total = m + math.log(sum(math.exp(l - m) for l in logs))
    return [math.exp(l - log_total) for l in logs]


def normalize_pair(a: float, b: float) -> tuple[float, float]:
    """Normalize two non-negative values to sum to 1 using log-sum-exp."""
    if a < 0 or b < 0:
        raise ValueError("a and b must be non-negative")

    if a + b < _EPS:
        return 0.5, 0.5

    loga = -math.inf if a == 0.0 else math.log(a)
    logb = -math.inf if b == 0.0 else math.log(b)

    m = max(loga, logb)
    log_total = m + math.log(math.exp(loga - m) + math.exp(logb - m))

    return math.exp(loga - log_total), math.exp(logb - log_total)
