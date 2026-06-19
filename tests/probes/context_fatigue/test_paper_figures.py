"""Tests for the context-fatigue paper figures.

These guard the extended abstract against drift: the values plotted must match the
committed source data, and the figure files must actually render.
"""

import re
from pathlib import Path

import pytest

from src.probes.context_fatigue import paper_figures as pf

RESULTS = pf.RESULTS
WC_DYNAMICS = RESULTS / "context_fatigue" / "wildchat_dynamics" / "WILDCHAT_DYNAMICS.md"
WC_HOMOGENEITY = RESULTS / "context_fatigue" / "wildchat_homogeneity" / "WILDCHAT_HOMOGENEITY.md"


def test_dose_response_matches_source():
    df = pf.load_dose_response()
    assert list(df["stage"]) == pf.DOSE_ORDER
    # baseline (early-context) confidence collapses monotonically across post-training
    early = df["entropy_early"].tolist()
    assert early == sorted(early, reverse=True)
    assert early[0] == pytest.approx(1.0578, abs=1e-3)
    assert early[-1] == pytest.approx(0.2015, abs=1e-3)
    # within-context ratio falls from collapse (>1, base) to reversal (<1, instruct)
    ratios = dict(zip(df["stage"], df["entropy_ratio"]))
    assert ratios["base"] == pytest.approx(1.64, abs=0.02)
    assert ratios["sft"] == pytest.approx(1.26, abs=0.02)
    assert ratios["dpo"] == pytest.approx(0.81, abs=0.02)
    assert ratios["instruct"] == pytest.approx(0.47, abs=0.02)


def test_random_context_flat_with_fill():
    overall = pf.random_context_overall()
    assert overall["random"] == pytest.approx(0.62, abs=0.02)
    assert overall["coherent"] == pytest.approx(0.84, abs=0.02)
    # random is consistently the lower-accuracy stream (hard subjects), not a slope
    df = pf.load_random_context()
    piv = df.pivot(index="fill_bin", columns="mode", values="accuracy")
    assert (piv["coherent"] > piv["random"]).all()


def test_attention_signature_strong_accuracy_flat():
    by_fill = pf.load_attention_by_fill("instruct", layer=24)
    sys_mass = by_fill["frac_system"].tolist()
    recent_mass = by_fill["frac_recent_cases"].tolist()
    # system erodes, recency grows monotonically across fill bins
    assert sys_mass == sorted(sys_mass, reverse=True)
    assert recent_mass == sorted(recent_mass)
    # but accuracy does not fall across fill bins (rot != score drop)
    acc = by_fill["accuracy"].dropna()
    assert acc.max() - acc.min() < 0.30
    assert by_fill["accuracy"].iloc[-1] >= by_fill["accuracy"].iloc[0]

    corr = pf.attention_corr_at_layer("instruct", layer=24)
    assert corr["system"] < -0.6  # strong erosion
    assert corr["recent"] > 0.6   # strong recency
    assert corr["entropy"] > 0.6  # attention diffuses


def test_wildchat_constants_match_reports():
    s = pf.WILDCHAT_SUMMARY
    # organic dialogue does NOT collapse; the repeating task does
    assert s["wildchat_late_over_early"] == pytest.approx(0.99, abs=0.02)
    assert s["ddxplus_late_over_early"] < 0.5
    # entropy slope tracks homogeneity, not length
    assert s["homogeneous_late_over_early"] < s["heterogeneous_late_over_early"]
    assert s["partial_corr_homogeneity_entropy_slope"] < 0

    # the headline numbers actually appear in the source reports
    dyn = WC_DYNAMICS.read_text()
    assert "0.99 median" in dyn
    homo = WC_HOMOGENEITY.read_text()
    assert "0.897" in homo and "1.001" in homo
    assert "-0.151" in homo or "−0.151" in homo


def test_make_figures_writes_pdfs(tmp_path):
    paths = pf.make_figures(tmp_path)
    assert set(paths) == {"dose_response", "random_context", "attention_rot", "wildchat_homogeneity"}
    for p in paths.values():
        assert p.exists() and p.suffix == ".pdf" and p.stat().st_size > 0
