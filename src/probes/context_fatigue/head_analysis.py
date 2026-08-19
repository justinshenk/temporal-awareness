"""Per-head attention analysis for the dilution program.

Every share the paper reports is a mean over attention heads (:func:`span_share`). That is the
right summary for a dose-response, but it leaves one objection open on the competition result:
a mean can hold perfectly still while heads redistribute underneath it, so "the evidence's
attention mass did not move" could mean "no head moved" or "the heads cancelled".

This module answers that with the unreduced per-head shares. :func:`redistribution_test` reports
the head-averaged contrast the paper already quotes *and* the mean absolute per-head contrast
beside it; their ratio is ~1 when every head moves together and diverges when they cancel.
:func:`head_concentration` describes how many heads carry the evidence's mass at all, which sets
how blunt the averaged summary is in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.base_schema import BaseSchema
from src.probes.context_fatigue.dilution_analysis import Coefficient, paired_accuracy_gap


@dataclass
class HeadConcentration(BaseSchema):
    """How unevenly a span's attention mass is spread over heads."""

    n_heads: int
    effective_heads: float
    top4_fraction: float


@dataclass
class DrainShape(BaseSchema):
    """How well one uniform odds-scale reproduces the observed per-head drain.

    The clamp applies a single additive bias to every head, so ``r2`` is the fraction of the
    per-head target pattern that instrument can reach in principle. A low ``r2`` supports reading
    a partial-necessity result as instrument limitation; a high one withdraws that excuse.
    """

    best_bias_nats: float
    r2: float
    implied_bias_sd: float
    implied_bias_min: float
    implied_bias_max: float
    fractional_drain_mean: float
    fractional_drain_sd: float
    corr_level_with_fractional_drain: float


@dataclass
class HeadContrast(BaseSchema):
    """One head's paired between-arm difference in span share."""

    head: int
    delta: Coefficient


@dataclass
class Redistribution(BaseSchema):
    """Whether a head-averaged null hides per-head movement.

    ``redistribution_ratio`` is ``mean_abs_delta / |mean_delta|``: 1.0 when every head moves the
    same way and by the same amount, large when heads move in opposite directions and cancel.
    """

    n_heads: int
    mean_delta: float
    mean_abs_delta: float
    redistribution_ratio: float
    max_abs_delta: float
    n_heads_excluding_zero: int
    null_mean_abs_delta: float = 0.0
    p_value: float = 1.0


def head_concentration(shares) -> HeadConcentration:
    """Effective number of heads carrying ``shares``, and the top-4 fraction.

    ``effective_heads`` is the exponential of the entropy of the normalized per-head shares: it is
    the head count when the mass is spread evenly and 1 when a single head holds all of it. Both
    statistics are scale-free, so a uniform drain leaves them unchanged — draining and
    concentrating are different things and must not be reported as one number.
    """
    s = np.asarray(shares, dtype=float)
    if s.size == 0:
        raise ValueError("need at least one head")
    if (s < 0).any():
        raise ValueError("attention shares cannot be negative")
    total = s.sum()
    if total <= 0:
        raise ValueError("per-head shares sum to zero; nothing to describe")
    p = s / total
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    k = min(4, s.size)
    return HeadConcentration(n_heads=int(s.size),
                             effective_heads=float(np.exp(entropy)),
                             top4_fraction=float(np.sort(s)[::-1][:k].sum() / total))


def _paired_panel(df: pd.DataFrame, arm_a: str, arm_b: str, value: str, head: int):
    """The two arms' values for one head, aligned probe-for-probe, or raise."""
    d = df[df["head"] == head]
    wide = d.pivot_table(index="probe", columns="arm", values=value)
    if arm_a not in wide.columns or arm_b not in wide.columns:
        raise ValueError(f"arms {arm_a!r} and {arm_b!r} must both appear for head {head}")
    if wide[[arm_a, arm_b]].isna().any().any():
        missing = int(wide[[arm_a, arm_b]].isna().any(axis=1).sum())
        raise ValueError(
            f"head {head} is not paired: {missing} probes are missing one arm. Drop them "
            f"upstream so every head is compared on the same probe set.")
    return wide[arm_a].to_numpy(float), wide[arm_b].to_numpy(float)


def paired_head_contrasts(df: pd.DataFrame, arm_a: str, arm_b: str,
                          value: str = "evidence_share", n_boot: int = 10000,
                          seed: int = 42, alpha: float = 0.05) -> list[HeadContrast]:
    """``arm_a`` minus ``arm_b`` in ``value``, per head, paired over probes.

    ``df`` is the long per-head frame written by the drivers: one row per probe x arm x head.
    Pairing is not optional here — the arms score the same probes by construction, and
    resampling them independently charges the interval for between-probe variance that cancels.

    ``alpha`` is per-head, so counting how many heads exclude zero at 0.05 over 32 heads expects
    about 1.6 false positives; pass ``alpha / n_heads`` for a family-wise statement.
    """
    return [HeadContrast(head=int(head),
                         delta=paired_accuracy_gap(*_paired_panel(df, arm_a, arm_b, value, head),
                                                   n_boot=n_boot, seed=seed, alpha=alpha))
            for head in sorted(df["head"].unique())]


def _null_mean_abs(diff: np.ndarray, n_perm: int, seed: int):
    """Paired sign-flip null for ``mean|per-head delta|``.

    ``diff`` is ``[n_probes, n_heads]`` of per-probe, per-head differences. Under the null that
    the arms are exchangeable, flipping the sign of a probe's *whole* head vector is a valid
    relabelling; flipping heads independently would destroy the cross-head structure that the
    statistic is about. ``mean|delta|`` is positive under any noise, so it needs this floor before
    a ratio above 1 means anything.
    """
    rng = np.random.default_rng(seed)
    observed = float(np.abs(diff.mean(axis=0)).mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.shape[0], 1))
    draws = np.abs((signs * diff[None, :, :]).mean(axis=1)).mean(axis=1)
    # +1 in numerator and denominator: the observed labelling is one of the possible ones.
    p = float((np.sum(draws >= observed) + 1) / (n_perm + 1))
    return float(draws.mean()), p


def redistribution_test(df: pd.DataFrame, arm_a: str, arm_b: str,
                        value: str = "evidence_share", n_boot: int = 10000,
                        seed: int = 42, alpha: float = 0.05,
                        n_perm: int = 2000) -> Redistribution:
    """Does the head-averaged contrast hide per-head movement?

    ``mean_delta`` reproduces the head-averaged number the paper reports. ``mean_abs_delta`` is
    what that number would be if the heads could not cancel. A null that survives both is a null
    about attention mass; a null that survives only the first is a null about the *average*.
    """
    contrasts = paired_head_contrasts(df, arm_a, arm_b, value, n_boot=n_boot, seed=seed,
                                  alpha=alpha)
    deltas = np.array([c.delta.estimate for c in contrasts], dtype=float)
    mean_delta = float(deltas.mean())
    mean_abs = float(np.abs(deltas).mean())
    if mean_abs == 0.0:
        ratio = 0.0
    elif mean_delta == 0.0:
        ratio = float("inf")
    else:
        ratio = mean_abs / abs(mean_delta)
    heads = sorted(df["head"].unique())
    diff = np.column_stack([np.subtract(*_paired_panel(df, arm_a, arm_b, value, h)) for h in heads])
    null_abs, p_value = _null_mean_abs(diff, n_perm=n_perm, seed=seed)

    return Redistribution(
        n_heads=len(contrasts),
        mean_delta=mean_delta,
        mean_abs_delta=mean_abs,
        redistribution_ratio=ratio,
        max_abs_delta=float(np.abs(deltas).max()),
        n_heads_excluding_zero=sum(c.delta.excludes_zero() for c in contrasts),
        null_mean_abs_delta=null_abs,
        p_value=p_value,
    )


def drain_shape(level: np.ndarray, drained: np.ndarray) -> DrainShape:
    """Can one uniform odds-scale turn ``level`` into ``drained``, head for head?

    ``level`` and ``drained`` are per-head shares in two arms. The clamp multiplies every head's
    *odds* by the same factor, so the question is whether a single bias reproduces the observed
    per-head pattern. ``corr_level_with_fractional_drain`` is reported alongside because the
    correlation of level with *absolute* drain is near-vacuous: if every head loses a similar
    fraction, absolute loss tracks level by arithmetic and says nothing about specialization.
    """
    level = np.asarray(level, dtype=float)
    drained = np.asarray(drained, dtype=float)
    if level.shape != drained.shape:
        raise ValueError(f"per-head vectors must align, got {level.shape} and {drained.shape}")
    if ((level <= 0) | (level >= 1) | (drained <= 0) | (drained >= 1)).any():
        raise ValueError("shares must lie strictly inside (0, 1) to take odds")

    odds = lambda s: s / (1 - s)  # noqa: E731 - local, and reads better inline here
    implied = np.log(odds(drained) / odds(level))
    best = float(implied.mean())
    pred_odds = odds(level) * np.exp(best)
    pred = pred_odds / (1 + pred_odds)
    ss_res = float(((drained - pred) ** 2).sum())
    ss_tot = float(((drained - drained.mean()) ** 2).sum())

    frac = (level - drained) / level
    # A constant fractional drain has no variance, so the correlation is undefined and numpy
    # returns whatever the floating-point residue implies (0.447 on an exactly-uniform drain in
    # testing). Semantically a constant drain *is* "level does not predict it", so report 0.0
    # rather than a number manufactured from rounding error.
    corr = (0.0 if frac.std() < 1e-12 or level.std() < 1e-12
            else float(np.corrcoef(level, frac)[0, 1]))
    return DrainShape(
        best_bias_nats=best,
        r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        implied_bias_sd=float(implied.std()),
        implied_bias_min=float(implied.min()),
        implied_bias_max=float(implied.max()),
        fractional_drain_mean=float(frac.mean()),
        fractional_drain_sd=float(frac.std()),
        corr_level_with_fractional_drain=corr,
    )


def evidence_head_profile(df: pd.DataFrame, arm: str, span_fraction: float) -> pd.DataFrame:
    """Per-head evidence and question shares in ``arm``, with a specificity control.

    ``enrichment`` is the head's evidence share divided by ``span_fraction``, the evidence span's
    share of the context tokens: 1.0 means the head attends to the evidence exactly in proportion
    to its size, so anything below 1.0 is *under*-weighting it. Without this normalization a long
    span looks like a head specialty. ``ev_over_q`` separates heads that load on the evidence
    specifically from heads that simply load on all recent content.
    """
    if span_fraction <= 0:
        raise ValueError("span_fraction must be positive")
    per_head = df[df["arm"] == arm].groupby("head")[["evidence_share", "question_share"]].mean()
    per_head["enrichment"] = per_head["evidence_share"] / span_fraction
    per_head["ev_over_q"] = per_head["evidence_share"] / per_head["question_share"]
    return per_head.sort_values("evidence_share", ascending=False)


def span_profile_by_fill(df: pd.DataFrame, value: str, low_fill: float = 0.25,
                         high_fill: float = 0.6) -> pd.DataFrame:
    """Per (layer, head) change in ``value`` from cold start to full context.

    Rows are ordered by the **absolute** change, because that is the quantity a reader can judge:
    a head going 0.010 to 0.005 halves, and so does one going 0.400 to 0.200, but only the second
    moves attention anyone would notice. ``fold_change`` is reported alongside and never sets the
    ordering.
    """
    low = df[df["context_fill"] < low_fill]
    high = df[df["context_fill"] > high_fill]
    if low.empty:
        raise ValueError(f"no cold-start rows below fill {low_fill}; this profile needs a sweep "
                         f"that starts near empty context")
    if high.empty:
        raise ValueError(f"no rows above fill {high_fill}")

    keys = ["layer", "head"]
    a = low.groupby(keys)[value].mean().rename("cold")
    b = high.groupby(keys)[value].mean().rename("full")
    out = pd.concat([a, b], axis=1).dropna().reset_index()
    out["delta"] = out["full"] - out["cold"]
    out["fold_change"] = out["full"] / out["cold"].replace(0.0, np.nan)
    corr = (df.groupby(keys)[[value, "context_fill"]]
            .corr().unstack().iloc[:, 1].rename("corr_with_fill").reset_index())
    out = out.merge(corr, on=keys, how="left")
    return out.reindex(out["delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)
