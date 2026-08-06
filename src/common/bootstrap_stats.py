"""Seeded percentile-bootstrap intervals, shared by any strand that reports a null or a gap.

A point estimate stated without an interval invites the "underpowered" objection, and the
projects's claims are frequently *differences* (a gap between two token roles, an accuracy
difference between two context-fill halves). The bootstrap here resamples **rows** of the value
matrix, so the caller controls the resampling unit: pass one row per independent case and the
interval respects that clustering. Everything is seeded (:data:`SEED`), so reported numbers are
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.base_schema import BaseSchema

SEED = 42
N_BOOT = 10000


@dataclass
class Interval(BaseSchema):
    """A point estimate with a 95% interval and the n it rests on."""

    estimate: float
    lo: float
    hi: float
    n: int

    def excludes_zero(self) -> bool:
        return (self.lo > 0) or (self.hi < 0)

    def render(self, digits: int = 3) -> str:
        return f"{self.estimate:+.{digits}f} [{self.lo:+.{digits}f}, {self.hi:+.{digits}f}] (n={self.n})"


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def bootstrap_interval(values: np.ndarray, statistic, alpha: float = 0.05) -> Interval:
    """Percentile bootstrap interval for ``statistic`` over rows of ``values``.

    Rows are the resampling unit. When rows are *clusters* (all of one problem's tokens summed
    into one row) the interval carries the within-cluster dependence that a per-observation
    interval would ignore.
    """
    rng = _rng()
    n = len(values)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        draws[b] = statistic(values[idx])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(statistic(values)), float(lo), float(hi), n)


def pooled_rate_gap(rows: np.ndarray) -> float:
    """Pooled rate difference ``a - b`` from rows of ``(n_a, hits_a, n_b, hits_b)`` counts.

    Pooled, not averaged over clusters: a cluster contributing more observations weighs more, so
    the point estimate equals the flat per-observation difference and only the *interval* changes.
    """
    n_a, hits_a, n_b, hits_b = np.asarray(rows, dtype=float).sum(0)
    if n_a == 0 or n_b == 0:
        return float("nan")
    return hits_a / n_a - hits_b / n_b


def clustered_rate_gap(counts_a: np.ndarray, counts_b: np.ndarray, alpha: float = 0.05) -> Interval:
    """Interval on a rate difference where whole clusters, not observations, are resampled.

    ``counts_a``/``counts_b`` are aligned ``(n_clusters, 2)`` arrays of per-cluster
    ``(n, hits)``. Use when observations within a cluster are dependent — successive tokens of one
    generated chain, repeated trials of one problem — so that the interval reflects the number of
    independent cases rather than the (much larger) number of observations.
    """
    counts_a, counts_b = np.asarray(counts_a, dtype=float), np.asarray(counts_b, dtype=float)
    if counts_a.shape != counts_b.shape or counts_a.ndim != 2 or counts_a.shape[1] != 2:
        raise ValueError(f"expected aligned (n_clusters, 2) count arrays, got "
                         f"{counts_a.shape} and {counts_b.shape}")
    return bootstrap_interval(np.hstack([counts_a, counts_b]), pooled_rate_gap, alpha)
