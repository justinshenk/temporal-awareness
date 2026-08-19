"""Tests for the E1/E3 analysis, written before the runs (brief §7.3).

The point of testing an analysis on synthetic data with a *planted* effect is that a fit which
cannot recover a known effect cannot be trusted to report an unknown one — and, just as
importantly, one that "recovers" an effect from data with none is worse than useless. Both
directions are checked here.

Bootstraps resample **cases**, per §5: no turn- or head-level pseudo-replication.
"""

import numpy as np
import pytest

from src.probes.context_fatigue.dilution_analysis import (
    paired_accuracy_gap,
    arm_accuracy_gap,
    final_bin_regression,
    joint_fit,
)


def _planted(n_cases=4000, beta_fill=0.0, beta_distance=-0.01, base=0.75, seed=0):
    """Bernoulli outcomes from a known linear probability model."""
    rng = np.random.default_rng(seed)
    fill = rng.uniform(0.0, 1.0, n_cases)
    distance = rng.choice([0, 2, 5, 10, 20], n_cases).astype(float)
    p = np.clip(base + beta_fill * fill + beta_distance * distance, 0.01, 0.99)
    correct = (rng.random(n_cases) < p).astype(float)
    return {"fill": fill, "distance": distance, "correct": correct}


# ── joint fit: distance vs fill ─────────────────────────────────────────

def test_joint_fit_recovers_a_planted_distance_effect():
    """§7.3: a distance effect of known size must be recovered with a CI covering it."""
    data = _planted(beta_distance=-0.01, beta_fill=0.0)
    fit = joint_fit(data, predictors=("fill", "distance"), n_boot=400, seed=42)

    dist = fit["distance"]
    assert dist.lo <= -0.01 <= dist.hi, f"CI {dist.lo:.4f}..{dist.hi:.4f} missed the planted -0.01"
    assert dist.excludes_zero()


def test_joint_fit_leaves_a_null_predictor_null():
    """The claim is that *distance* carries the coefficient and fill does not — so a fit that
    invents a fill effect where none was planted would fake E1's headline result."""
    data = _planted(beta_distance=-0.01, beta_fill=0.0)
    fit = joint_fit(data, predictors=("fill", "distance"), n_boot=400, seed=42)

    fill = fit["fill"]
    assert fill.lo <= 0.0 <= fill.hi
    assert not fill.excludes_zero()


def test_joint_fit_recovers_a_planted_fill_effect_when_there_is_one():
    """The converse guard: the fit must not be structurally blind to fill."""
    data = _planted(beta_distance=0.0, beta_fill=-0.20)
    fit = joint_fit(data, predictors=("fill", "distance"), n_boot=400, seed=42)

    assert fit["fill"].lo <= -0.20 <= fit["fill"].hi
    assert fit["fill"].excludes_zero()
    assert not fit["distance"].excludes_zero()


def test_joint_fit_finds_nothing_in_pure_noise():
    data = _planted(beta_distance=0.0, beta_fill=0.0)
    fit = joint_fit(data, predictors=("fill", "distance"), n_boot=400, seed=42)
    assert not fit["distance"].excludes_zero()
    assert not fit["fill"].excludes_zero()


def test_joint_fit_is_seed_stable():
    data = _planted()
    a = joint_fit(data, predictors=("fill", "distance"), n_boot=200, seed=7)
    b = joint_fit(data, predictors=("fill", "distance"), n_boot=200, seed=7)
    assert a["distance"].to_dict() == b["distance"].to_dict()


def test_joint_fit_rejects_unknown_predictor():
    with pytest.raises(KeyError):
        joint_fit(_planted(), predictors=("fill", "phase_of_moon"), n_boot=50, seed=1)


# ── arm gaps ────────────────────────────────────────────────────────────

def test_arm_gap_recovers_a_planted_ten_point_difference():
    """§8: planted +0.10 accuracy difference recovered, 95% CI containing 0.10 and excluding 0."""
    rng = np.random.default_rng(11)
    n = 3000
    local = (rng.random(n) < 0.70).astype(float)
    back = (rng.random(n) < 0.60).astype(float)

    gap = arm_accuracy_gap(local, back, n_boot=2000, seed=42)

    assert gap.lo <= 0.10 <= gap.hi, f"CI {gap.lo:.4f}..{gap.hi:.4f} missed the planted 0.10"
    assert gap.excludes_zero()
    assert gap.estimate == pytest.approx(0.10, abs=0.03)


def test_arm_gap_is_seed_stable_across_two_runs():
    """§8 asks for seed stability explicitly."""
    rng = np.random.default_rng(5)
    a_arm = (rng.random(500) < 0.7).astype(float)
    b_arm = (rng.random(500) < 0.6).astype(float)
    first = arm_accuracy_gap(a_arm, b_arm, n_boot=500, seed=42)
    second = arm_accuracy_gap(a_arm, b_arm, n_boot=500, seed=42)
    assert first.to_dict() == second.to_dict()


def test_arm_gap_on_identical_arms_covers_zero():
    rng = np.random.default_rng(3)
    same = (rng.random(2000) < 0.65).astype(float)
    other = (rng.random(2000) < 0.65).astype(float)
    gap = arm_accuracy_gap(same, other, n_boot=1000, seed=42)
    assert not gap.excludes_zero()


def test_arm_gap_rejects_empty_arm():
    with pytest.raises(ValueError):
        arm_accuracy_gap(np.array([]), np.array([1.0, 0.0]), n_boot=10, seed=1)


# ── regression against the committed artifact ───────────────────────────

def test_reproduces_the_published_deep_fill_dip():
    """§8: the existing deep-fill artifact must still give -0.141 [-0.249, -0.031], n=91.

    This is the number the paper's one positive result rests on, so it is pinned exactly rather
    than approximately, at the real N_BOOT — the fast-bootstrap fixture used elsewhere would not
    reproduce the published interval.
    """
    stats = final_bin_regression()

    assert stats["n_top_bin"] == 91
    assert stats["estimate"] == pytest.approx(-0.141, abs=5e-4)
    assert stats["lo"] == pytest.approx(-0.249, abs=5e-4)
    assert stats["hi"] == pytest.approx(-0.031, abs=5e-4)
    assert stats["significant"]


def test_final_bin_regression_names_its_artifact():
    """A provenance hazard: `final_bin_stats()` with no path silently falls back to a *different*
    artifact (results/random_context/turns.csv, n=31, -0.187) than the one behind the published
    number (turns_pooled.csv, n=91, -0.141). The analyzer must name the file it read, so a number
    can never be quoted without knowing which stream produced it.
    """
    stats = final_bin_regression()
    assert stats["artifact"].endswith("random_context_topbin/turns_pooled.csv")


# ── paired contrasts ────────────────────────────────────────────────────

def test_paired_gap_recovers_a_planted_within_item_effect():
    """The clamp designs measure the *same* item twice; the estimator must use that."""
    rng = np.random.default_rng(0)
    n = 200
    base = rng.random(n) < 0.5                      # item difficulty, shared by both conditions
    flip = rng.random(n) < 0.20                     # the planted within-item cost
    treated = base & ~flip
    gap = paired_accuracy_gap(base.astype(float), treated.astype(float), n_boot=2000)
    assert gap.lo < base.mean() - treated.mean() < gap.hi
    assert gap.excludes_zero()


def test_pairing_is_tighter_than_independent_resampling_on_paired_data():
    """The bug this replaces: independent resampling throws the pairing away and inflates the CI.

    Both estimators are unbiased for the difference; only the paired one uses the fact that item
    difficulty is common to the two conditions, so its interval must be strictly narrower.
    """
    rng = np.random.default_rng(1)
    n = 300
    base = (rng.random(n) < 0.5).astype(float)
    treated = base * (rng.random(n) > 0.15)
    paired = paired_accuracy_gap(base, treated, n_boot=4000)
    independent = arm_accuracy_gap(base, treated, n_boot=4000)
    assert (paired.hi - paired.lo) < (independent.hi - independent.lo)


def test_paired_gap_rejects_misaligned_arms():
    """Unequal lengths mean the rows are not the same items -- pairing them would be a lie."""
    with pytest.raises(ValueError, match="paired"):
        paired_accuracy_gap(np.zeros(10), np.zeros(9))


def test_paired_gap_finds_nothing_in_pure_noise():
    rng = np.random.default_rng(2)
    a = (rng.random(400) < 0.5).astype(float)
    b = (rng.random(400) < 0.5).astype(float)
    assert not paired_accuracy_gap(a, b, n_boot=2000).excludes_zero()
