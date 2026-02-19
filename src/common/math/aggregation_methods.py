"""Aggregation methods for combining values."""

from __future__ import annotations

from enum import Enum
from typing import Sequence


class AggregationMethod(Enum):
    """Method for aggregating values across a collection."""

    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    MEDIAN = "median"
    MIN = "min"


def aggregate(values: Sequence[float], method: AggregationMethod) -> float:
    """Aggregate a sequence of values using the specified method.

    Args:
        values: Sequence of float values to aggregate
        method: Aggregation method to use

    Returns:
        Aggregated value, or -inf for empty sequences
    """
    if not values:
        return float("-inf")

    if method == AggregationMethod.MEAN:
        return sum(values) / len(values)
    elif method == AggregationMethod.MAX:
        return max(values)
    elif method == AggregationMethod.MIN:
        return min(values)
    elif method == AggregationMethod.SUM:
        return sum(values)
    elif method == AggregationMethod.MEDIAN:
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        return sorted_vals[n // 2]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
