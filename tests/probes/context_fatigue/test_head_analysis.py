"""Tests for per-head attention analysis.

The paper's competition result rests on a head-*averaged* null: the evidence's attention share
does not move between arms. The obvious objection is that a mean can hold still while heads
redistribute underneath it. These tests pin the statistics that answer it, on constructed data
where the right answer is known by hand.
"""

import numpy as np
import pandas as pd
import pytest

from src.probes.context_fatigue.head_analysis import (
    head_concentration,
    paired_head_contrasts,
    redistribution_test,
)


def _frame(shares_by_arm, n_probes=40, noise=0.0, seed=0):
    """Long per-head frame with a known per-head mean in every arm."""
    rng = np.random.default_rng(seed)
    rows = []
    for arm, shares in shares_by_arm.items():
        for probe in range(n_probes):
            for head, mu in enumerate(shares):
                rows.append({"probe": probe, "arm": arm, "head": head,
                             "evidence_share": mu + noise * rng.standard_normal()})
    return pd.DataFrame(rows)


# ── concentration ───────────────────────────────────────────────────────

def test_uniform_heads_have_effective_count_equal_to_head_count():
    """When every head carries the same mass, all of them are 'doing the work'."""
    out = head_concentration([0.02] * 8)
    assert out.effective_heads == pytest.approx(8.0)
    assert out.top4_fraction == pytest.approx(0.5)


def test_one_dominant_head_collapses_the_effective_count():
    shares = [1.0] + [0.0] * 7
    out = head_concentration(shares)
    assert out.effective_heads == pytest.approx(1.0)
    assert out.top4_fraction == pytest.approx(1.0)


def test_concentration_ignores_overall_scale():
    """Halving every head's share is a drain, not a change in how concentrated it is."""
    a = head_concentration([0.04, 0.02, 0.01, 0.01])
    b = head_concentration([0.02, 0.01, 0.005, 0.005])
    assert a.effective_heads == pytest.approx(b.effective_heads)


# ── per-head contrasts ──────────────────────────────────────────────────

def test_paired_contrast_recovers_a_planted_per_head_difference():
    df = _frame({"a": [0.05, 0.03, 0.01], "b": [0.04, 0.03, 0.02]})
    out = paired_head_contrasts(df, "a", "b")
    assert [c.head for c in out] == [0, 1, 2]
    assert [c.delta.estimate for c in out] == pytest.approx([0.01, 0.0, -0.01])


def test_paired_contrast_requires_the_same_probes_in_both_arms():
    df = _frame({"a": [0.05], "b": [0.04]})
    df = df.drop(df[(df["arm"] == "b") & (df["probe"] == 0)].index)
    with pytest.raises(ValueError, match="paired"):
        paired_head_contrasts(df, "a", "b")


# ── the redistribution test ─────────────────────────────────────────────

def test_pure_redistribution_is_caught_even_though_the_mean_does_not_move():
    """Half the heads gain exactly what the other half lose.

    This is the failure mode the head-averaged null cannot see, and the reason this module
    exists: mean_delta is 0 while every individual head moved.
    """
    df = _frame({"a": [0.06, 0.02, 0.06, 0.02], "b": [0.02, 0.06, 0.02, 0.06]})
    out = redistribution_test(df, "a", "b")
    assert out.mean_delta == pytest.approx(0.0, abs=1e-12)
    assert out.mean_abs_delta == pytest.approx(0.04)
    assert out.redistribution_ratio > 100
    assert out.n_heads_excluding_zero == 4


def test_a_genuine_null_leaves_both_statistics_at_zero():
    df = _frame({"a": [0.05, 0.03, 0.01], "b": [0.05, 0.03, 0.01]})
    out = redistribution_test(df, "a", "b")
    assert out.mean_delta == pytest.approx(0.0, abs=1e-12)
    assert out.mean_abs_delta == pytest.approx(0.0, abs=1e-12)
    assert out.n_heads_excluding_zero == 0


def test_a_uniform_drain_is_not_reported_as_redistribution():
    """Every head losing the same amount is dilution, and the ratio must stay near 1."""
    df = _frame({"a": [0.05, 0.04, 0.03], "b": [0.03, 0.02, 0.01]})
    out = redistribution_test(df, "a", "b")
    assert out.mean_delta == pytest.approx(0.02)
    assert out.redistribution_ratio == pytest.approx(1.0)


def test_mean_delta_equals_the_head_averaged_contrast():
    """The per-head machinery must reproduce the number the paper already reports."""
    df = _frame({"a": [0.05, 0.03, 0.01], "b": [0.04, 0.033, 0.02]}, noise=0.002, seed=7)
    out = redistribution_test(df, "a", "b")
    pooled = df.groupby(["arm", "probe"])["evidence_share"].mean().unstack(0)
    assert out.mean_delta == pytest.approx(float((pooled["a"] - pooled["b"]).mean()), abs=1e-12)


def test_alpha_tightens_the_per_head_significance_count():
    """32 heads at alpha=0.05 expect ~1.6 false positives, so the count needs a family-wise option."""
    df = _frame({"a": [0.05, 0.03], "b": [0.047, 0.03]}, noise=0.01, seed=3)
    loose = redistribution_test(df, "a", "b", alpha=0.05)
    strict = redistribution_test(df, "a", "b", alpha=0.05 / 32)
    assert strict.n_heads_excluding_zero <= loose.n_heads_excluding_zero
    assert strict.mean_delta == pytest.approx(loose.mean_delta)


def test_the_null_floor_absorbs_pure_noise():
    """With no real per-head difference, mean|delta| must sit at its own permutation floor."""
    df = _frame({"a": [0.05, 0.03, 0.01], "b": [0.05, 0.03, 0.01]}, noise=0.01, seed=11)
    out = redistribution_test(df, "a", "b", n_perm=500)
    assert out.mean_abs_delta == pytest.approx(out.null_mean_abs_delta, rel=0.6)
    assert out.p_value > 0.05


def test_real_redistribution_beats_the_null_floor():
    df = _frame({"a": [0.06, 0.02, 0.06, 0.02], "b": [0.02, 0.06, 0.02, 0.06]},
                noise=0.002, seed=5)
    out = redistribution_test(df, "a", "b", n_perm=500)
    assert out.mean_abs_delta > 5 * out.null_mean_abs_delta
    assert out.p_value < 0.01


# ── drain shape and head identity ───────────────────────────────────────

def test_a_true_uniform_odds_scale_is_recovered_exactly():
    """If the drain really was one uniform bias, the clamp can reach it and r2 must be 1."""
    from src.probes.context_fatigue.head_analysis import drain_shape
    level = np.array([0.09, 0.07, 0.05, 0.03, 0.01])
    odds = level / (1 - level)
    drained_odds = odds * np.exp(-1.4)
    drained = drained_odds / (1 + drained_odds)
    out = drain_shape(level, drained)
    assert out.best_bias_nats == pytest.approx(-1.4)
    assert out.r2 == pytest.approx(1.0, abs=1e-9)
    assert out.implied_bias_sd == pytest.approx(0.0, abs=1e-9)


def test_a_head_specific_drain_is_not_reachable_by_one_bias():
    from src.probes.context_fatigue.head_analysis import drain_shape
    level = np.array([0.09, 0.07, 0.05, 0.03, 0.01])
    drained = np.array([0.001, 0.068, 0.048, 0.029, 0.0098])  # one head collapses, rest barely move
    out = drain_shape(level, drained)
    assert out.r2 < 0.5
    assert out.implied_bias_sd > 1.0


def test_fractional_drain_correlation_is_reported_not_the_absolute_one():
    """Uniform fractional drain must report ~0 correlation with level, not the arithmetic ~1."""
    from src.probes.context_fatigue.head_analysis import drain_shape
    level = np.array([0.09, 0.07, 0.05, 0.03, 0.01])
    drained = level * 0.3  # every head loses exactly 70%
    out = drain_shape(level, drained)
    assert out.fractional_drain_mean == pytest.approx(0.7)
    assert abs(out.corr_level_with_fractional_drain) < 1e-6


def test_enrichment_flags_a_head_that_only_looks_big_because_the_span_is_big():
    from src.probes.context_fatigue.head_analysis import evidence_head_profile
    df = pd.DataFrame([
        {"arm": "local", "probe": p, "head": h, "evidence_share": s, "question_share": 0.2}
        for p in range(5) for h, s in enumerate([0.10, 0.30])
    ])
    prof = evidence_head_profile(df, "local", span_fraction=0.10)
    assert prof.loc[0, "enrichment"] == pytest.approx(1.0)   # exactly proportional to span size
    assert prof.loc[1, "enrichment"] == pytest.approx(3.0)   # genuinely concentrating


def test_drain_varying_but_unrelated_to_level_correlates_near_zero():
    """The guard must not mask a genuine measurement: here the drain varies and is unrelated."""
    from src.probes.context_fatigue.head_analysis import drain_shape
    level = np.array([0.09, 0.07, 0.05, 0.03, 0.01])
    frac = np.array([0.60, 0.80, 0.60, 0.80, 0.70])  # varies, uncorrelated with level
    out = drain_shape(level, level * (1 - frac))
    assert out.fractional_drain_sd > 0.05
    assert abs(out.corr_level_with_fractional_drain) < 0.35


# ── span profile across context fill ────────────────────────────────────

def _fill_frame(per_head_low_high, n=30, seed=0):
    """Per-head frame with a known cold-start and full-context value for one span."""
    rng = np.random.default_rng(seed)
    rows = []
    for (layer, head), (lo_v, hi_v) in per_head_low_high.items():
        for i in range(n):
            for fill, v in ((0.05, lo_v), (0.85, hi_v)):
                rows.append({"layer": layer, "head": head, "context_fill": fill,
                             "frac_system": v + 0.001 * rng.standard_normal()})
    return pd.DataFrame(rows)


def test_span_profile_reports_the_absolute_change_not_just_the_ratio():
    """A head going 0.010 -> 0.005 must not outrank one going 0.400 -> 0.200.

    Both halve. Only the second moves attention mass anyone would notice, and ranking by
    fraction would invert them.
    """
    from src.probes.context_fatigue.head_analysis import span_profile_by_fill
    df = _fill_frame({(24, 0): (0.010, 0.005), (16, 1): (0.400, 0.200)})
    prof = span_profile_by_fill(df, "frac_system")
    assert list(prof["head"])[0] == 1          # ranked by absolute loss
    assert prof.iloc[0]["delta"] == pytest.approx(-0.200, abs=0.01)
    assert prof.iloc[1]["delta"] == pytest.approx(-0.005, abs=0.01)
    # the ratio is still available, it just does not drive the ordering
    assert prof.iloc[0]["fold_change"] == pytest.approx(0.5, abs=0.05)
    assert prof.iloc[1]["fold_change"] == pytest.approx(0.5, abs=0.05)


def test_span_profile_keeps_layer_and_head_identity():
    from src.probes.context_fatigue.head_analysis import span_profile_by_fill
    df = _fill_frame({(3, 7): (0.2, 0.1), (30, 11): (0.05, 0.04)})
    prof = span_profile_by_fill(df, "frac_system")
    assert set(zip(prof["layer"], prof["head"])) == {(3, 7), (30, 11)}


def test_span_profile_rejects_a_frame_with_no_cold_start():
    from src.probes.context_fatigue.head_analysis import span_profile_by_fill
    df = _fill_frame({(1, 1): (0.2, 0.1)})
    df = df[df["context_fill"] > 0.5]
    with pytest.raises(ValueError, match="cold-start"):
        span_profile_by_fill(df, "frac_system")
