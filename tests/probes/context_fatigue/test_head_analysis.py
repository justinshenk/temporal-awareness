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
