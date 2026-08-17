"""Tests for the context-fatigue interval estimates.

The extended abstract's nulls are now bounded claims; these tests pin the interval machinery
(seeded, so exactly reproducible) and the headline numbers against the committed per-case data.
"""

import numpy as np
import pytest

from src.probes.context_fatigue import null_statistics as ns


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(ns, "N_BOOT", 1000)


def test_bootstrap_interval_recovers_mean_and_is_deterministic():
    values = np.arange(200, dtype=float).reshape(-1, 1)
    mean = lambda a: float(a.mean())
    first = ns.bootstrap_interval(values, mean)
    again = ns.bootstrap_interval(values, mean)
    assert first == again  # seeded: identical draws every call
    assert first.estimate == pytest.approx(99.5)
    assert first.lo < first.estimate < first.hi
    assert first.excludes_zero()


def test_fisher_z_interval_width_shrinks_with_n():
    narrow = ns.fisher_z_interval(0.5, n=400)
    wide = ns.fisher_z_interval(0.5, n=20)
    assert narrow.lo < 0.5 < narrow.hi
    assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)
    assert np.isnan(ns.fisher_z_interval(0.5, n=3).lo)


def test_fill_slope_bounds_the_null():
    out = ns.fill_slope_stats()
    assert set(out) == {"coherent", "random"}
    for stats in out.values():
        # the flat-accuracy null holds, and with power: declines beyond ~10 points excluded
        assert stats["flat_supported"]
        assert abs(stats["corr_fill"]["estimate"]) < 0.05
        assert stats["max_decline_excluded_at_95"] < 0.12
    assert out["coherent"]["accuracy"] == pytest.approx(0.841, abs=0.005)
    assert out["random"]["accuracy"] == pytest.approx(0.622, abs=0.005)


def test_fill_slope_max_fill_restricts_rows():
    full = ns.fill_slope_stats()
    scoped = ns.fill_slope_stats(max_fill=0.8)
    for mode in ("coherent", "random"):
        assert scoped[mode]["n"] < full[mode]["n"]
    # a max_fill above every observed fill is a no-op
    assert ns.fill_slope_stats(max_fill=1.01) == full


def test_final_bin_dip_only_significant_on_random_stream():
    out = ns.final_bin_stats()
    assert out["random"]["diff_top_minus_rest"]["estimate"] < -0.1
    assert not out["coherent"]["significant"]


def test_attention_inversion_is_directional_but_marginal():
    inv = ns.attention_inversion_stats("instruct", layer=24)
    deltas = {label: b["delta_wrong_minus_correct"]["estimate"] for label, b in inv["bins"].items()}
    # wrong answers over-attend the current query in every fill bin (never the neglect sign)
    assert all(d > 0 for d in deltas.values())
    assert inv["bins"]["75-100%"]["significant"]
    # ...but the pooled effect is marginal, and the paper must say so
    assert inv["pooled_delta"]["estimate"] == pytest.approx(0.045, abs=0.01)
    assert inv["pooled_delta"]["n"] == sum(b["n_cases"] for b in inv["bins"].values())


def test_layer_generality_sign_holds_across_layers():
    out = ns.layer_generality("instruct", layers=(8, 16, 24, 31))
    assert set(out) == {"8", "16", "24", "31"}
    assert all(s["pooled_delta"]["estimate"] > 0 for s in out.values())


def test_calibration_gap_widens_at_flat_accuracy():
    cal = ns.calibration_by_fill()
    assert cal["n_total"] == 154
    conf = [cal["bins"][lab]["confidence"] for lab in ns.FILL_LABELS]
    acc = [cal["bins"][lab]["accuracy"] for lab in ns.FILL_LABELS]
    # confidence rises monotonically with fill while accuracy stays in a flat band
    assert conf == sorted(conf)
    assert conf[-1] - conf[0] > 0.05
    assert max(acc) - min(acc) < 0.3
    # the rise is carried by wrong answers too — the confidently-wrong gap
    assert cal["corr_confidence_fill"]["lo"] > 0.5
    assert cal["corr_confidence_fill_wrong_only"]["lo"] > 0.4
