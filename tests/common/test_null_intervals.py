"""Tests for exact binomial bounds on the procedure nulls.

Written before the implementation. The central requirement is the one the bootstrap cannot
meet: a run that scored 0 out of n must yield a *positive* upper bound, because "we saw no
successes in 30 tries" bounds the true rate, it does not establish that the rate is zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.bootstrap_stats import bootstrap_interval
from src.common.null_intervals import (
    BoundedNull,
    bounded_null,
    bounded_null_from_rate,
    clopper_pearson,
)


class TestClopperPearson:
    def test_zero_successes_gives_positive_upper_bound(self):
        """The property the whole module exists for."""
        lo, hi = clopper_pearson(0, 30)
        assert lo == 0.0
        assert hi > 0.0

    @pytest.mark.parametrize("n, expected", [(20, 0.168433), (30, 0.115703)])
    def test_zero_successes_matches_closed_form(self, n, expected):
        """For 0 hits the exact two-sided upper bound is 1 - (alpha/2)^(1/n)."""
        _, hi = clopper_pearson(0, n)
        assert hi == pytest.approx(expected, abs=1e-6)
        assert hi == pytest.approx(1 - 0.025 ** (1 / n), abs=1e-12)

    def test_upper_bound_tightens_with_n(self):
        bounds = [clopper_pearson(0, n)[1] for n in (10, 20, 30, 100, 500)]
        assert bounds == sorted(bounds, reverse=True)
        assert bounds[-1] < 0.01

    def test_all_successes_is_mirror_image(self):
        lo, hi = clopper_pearson(7, 7)
        assert hi == 1.0
        assert lo == pytest.approx(0.025 ** (1 / 7), abs=1e-12)

    def test_interior_case_brackets_the_estimate(self):
        lo, hi = clopper_pearson(5, 20)
        assert 0.0 < lo < 0.25 < hi < 1.0

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError):
            clopper_pearson(31, 30)
        with pytest.raises(ValueError):
            clopper_pearson(-1, 30)
        with pytest.raises(ValueError):
            clopper_pearson(0, 0)


class TestBoundedNull:
    def test_accuracy_scale_only_without_references(self):
        b = bounded_null(hits=0, n=30)
        assert isinstance(b, BoundedNull)
        assert b.rate == 0.0
        assert b.rate_hi > 0.0
        assert b.recovery is None and b.recovery_hi is None

    def test_recovery_scales_by_the_budget(self):
        """base 0.0 / donor 0.5 halves the budget, so the recovery bound is twice the accuracy bound."""
        b = bounded_null(hits=0, n=30, base_acc=0.0, lora_acc=0.5)
        assert b.recovery == pytest.approx(0.0)
        assert b.recovery_hi == pytest.approx(b.rate_hi / 0.5, rel=1e-9)

    def test_recovery_offsets_by_a_nonzero_base(self):
        b = bounded_null(hits=3, n=30, base_acc=0.1, lora_acc=0.6)
        assert b.recovery == pytest.approx((0.1 - 0.1) / 0.5)
        assert b.recovery_lo == pytest.approx((b.rate_lo - 0.1) / 0.5, rel=1e-9)
        assert b.recovery_hi == pytest.approx((b.rate_hi - 0.1) / 0.5, rel=1e-9)

    def test_zero_budget_is_rejected(self):
        with pytest.raises(ValueError):
            bounded_null(hits=0, n=30, base_acc=0.5, lora_acc=0.5)

    def test_negative_budget_is_rejected(self):
        """A donor worse than base has no budget to recover, and dividing by it inverts the
        interval. Caught a real extraction bug: short_arithmetic's `direct` arm is base 0.767 ->
        LoRA 0.600, and the writeup's claim rests on the `cot` arm instead."""
        with pytest.raises(ValueError, match="donor scores below base"):
            bounded_null(hits=0, n=30, base_acc=0.767, lora_acc=0.600)

    def test_accepts_a_rate_instead_of_a_count(self):
        """Artifacts store accuracies, not hit counts; the rate must round-trip to an integer."""
        assert bounded_null_from_rate(0.0, 30).hits == 0
        assert bounded_null_from_rate(0.2, 30).hits == 6

    def test_from_rate_rejects_a_non_integral_count(self):
        with pytest.raises(ValueError):
            bounded_null_from_rate(0.137, 30)

    def test_render_shows_the_bound_and_the_n(self):
        text = bounded_null(hits=0, n=30, base_acc=0.0, lora_acc=0.5).render()
        assert "0.00" in text and "n=30" in text


class TestWhyNotBootstrap:
    """Documents the reason this module exists rather than reusing bootstrap_stats."""

    def test_bootstrap_of_an_all_zero_sample_is_degenerate(self):
        zeros = np.zeros((30, 1))
        boot = bootstrap_interval(zeros, lambda rows: float(rows.mean()))
        assert boot.lo == 0.0 and boot.hi == 0.0, "bootstrap cannot see unobserved successes"

        exact = bounded_null(hits=0, n=30)
        assert exact.rate_hi > 0.0
        assert exact.rate_hi > boot.hi
