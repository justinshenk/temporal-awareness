"""Analysis for the dilution-localization experiments (E1 distance, E3 competition).

E1's claim is a comparison of coefficients, not a single number: with accuracy regressed jointly on
fill and distance, **distance should carry the coefficient and fill should not**. That is only
meaningful if the fit can be shown to recover a planted effect *and* to leave a planted null null,
so both directions are tested before any real data is fitted.

A linear probability model is used rather than logistic regression: the outcome is a rate, the
quantity the paper quotes is a difference in accuracy (percentage points), and OLS coefficients on
a 0/1 outcome are already on that scale — no odds-ratio translation stands between the fit and the
claim. Intervals come from a bootstrap that resamples **cases**, per brief §5, which is also what
keeps the interval honest under the model's misspecification at the probability bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.base_schema import BaseSchema
from src.probes.context_fatigue.null_statistics import RESULTS, final_bin_stats

# The pooled deep-fill stream: the original 12 sessions plus the 2026-08-17 top-bin batch.
# `final_bin_stats()` with no argument reads a *different*, smaller artifact, so every caller
# here passes the path explicitly and reports it alongside the number.
POOLED_TURNS = RESULTS / "random_context_topbin" / "turns_pooled.csv"


@dataclass
class Coefficient(BaseSchema):
    """One fitted coefficient with its bootstrap interval."""

    name: str
    estimate: float
    lo: float
    hi: float

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0


def _design(data, predictors):
    columns = [np.asarray(data[p], dtype=float) for p in predictors]  # KeyError on a bad name
    return np.column_stack([np.ones(len(columns[0])), *columns])


def _ols(design, outcome):
    return np.linalg.lstsq(design, outcome, rcond=None)[0]


def joint_fit(data, predictors=("fill", "distance"), n_boot: int = 2000,
              seed: int = 42, alpha: float = 0.05) -> dict[str, Coefficient]:
    """Regress ``correct`` on ``predictors`` jointly; bootstrap over cases for intervals.

    Returns one :class:`Coefficient` per predictor. E1 confirms if ``distance`` excludes zero and
    ``fill`` does not; that comparison is the reason the fit is joint rather than two marginals.
    """
    outcome = np.asarray(data["correct"], dtype=float)
    design = _design(data, predictors)
    point = _ols(design, outcome)

    rng = np.random.default_rng(seed)
    n = len(outcome)
    draws = np.empty((n_boot, design.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)  # resample cases, not turns
        draws[b] = _ols(design[idx], outcome[idx])

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        name: Coefficient(name=name,
                          estimate=float(point[i + 1]),
                          lo=float(np.percentile(draws[:, i + 1], lo_q)),
                          hi=float(np.percentile(draws[:, i + 1], hi_q)))
        for i, name in enumerate(predictors)
    }


def arm_accuracy_gap(arm_a, arm_b, n_boot: int = 10000, seed: int = 42,
                     alpha: float = 0.05) -> Coefficient:
    """Accuracy of ``arm_a`` minus ``arm_b``, with a case-resampled bootstrap interval."""
    a = np.asarray(arm_a, dtype=float)
    b = np.asarray(arm_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both arms need at least one case to compare")

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = (a[rng.integers(0, a.size, a.size)].mean()
                    - b[rng.integers(0, b.size, b.size)].mean())

    return Coefficient(name="accuracy_gap",
                       estimate=float(a.mean() - b.mean()),
                       lo=float(np.percentile(draws, 100 * alpha / 2)),
                       hi=float(np.percentile(draws, 100 * (1 - alpha / 2))))


def paired_accuracy_gap(arm_a, arm_b, n_boot: int = 10000, seed: int = 42,
                        alpha: float = 0.05) -> Coefficient:
    """Accuracy of ``arm_a`` minus ``arm_b`` when both arms scored the **same items**.

    The clamp and dissociation designs (E1c/E1d/E1e/E1f) measure one item under several
    conditions, so the two arms share item difficulty exactly. :func:`arm_accuracy_gap`
    resamples the arms independently, which is correct for genuinely independent arms and
    *throws the pairing away* here — it charges the interval for between-item variance that
    cancels in the contrast, inflating the CI roughly 2.5x on this data. Resample item indices
    once and take the difference within the resampled items.

    ``arm_a[i]`` and ``arm_b[i]`` must be the same item; equal lengths are necessary, not
    sufficient, so callers must align the arms themselves (pivoting on the item id).
    """
    a = np.asarray(arm_a, dtype=float)
    b = np.asarray(arm_b, dtype=float)
    if a.size != b.size:
        raise ValueError(f"paired arms must be aligned item-for-item, got {a.size} and {b.size}")
    if a.size == 0:
        raise ValueError("paired arms need at least one item to compare")

    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    draws = diff[idx].mean(axis=1)

    return Coefficient(name="paired_accuracy_gap",
                       estimate=float(diff.mean()),
                       lo=float(np.percentile(draws, 100 * alpha / 2)),
                       hi=float(np.percentile(draws, 100 * (1 - alpha / 2))))


def final_bin_regression(path: Path | None = None, mode: str = "random") -> dict:
    """The published top-bin dip, read from a **named** artifact.

    Wraps :func:`final_bin_stats` for one purpose: to make the artifact explicit. Called with no
    path, ``final_bin_stats`` falls back to ``results/random_context/turns.csv`` (n=31,
    −0.187) rather than the pooled stream behind the paper's −0.141 at n=91 — the same call
    returning two different headline numbers depending on which files happen to exist.
    """
    artifact = Path(path or POOLED_TURNS)
    stats = final_bin_stats(artifact)[mode]
    diff = stats["diff_top_minus_rest"]
    return {
        "artifact": str(artifact),
        "mode": mode,
        "n_top_bin": stats["n_top_bin"],
        "accuracy_top_bin": stats["accuracy_top_bin"],
        "accuracy_rest": stats["accuracy_rest"],
        "estimate": diff["estimate"],
        "lo": diff["lo"],
        "hi": diff["hi"],
        "significant": stats["significant"],
    }
