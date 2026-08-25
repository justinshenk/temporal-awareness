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
    assert set(paths) == {"distance_ladder", "mass_dose", "competition",
                          "dose_response", "random_context", "attention_rot",
                          "wildchat_homogeneity", "format_erosion",
                          "format_enrichment", "format_recovery"}
    for p in paths.values():
        assert p.exists() and p.suffix == ".pdf" and p.stat().st_size > 0


# ── the dilution program (E1 / E1b / E1f / E3) ──────────────────────────

E1_REPORT = RESULTS / "context_fatigue" / "E1_DISTANCE_SWEEP.md"
MECHANISM_REPORT = RESULTS / "context_fatigue" / "E1_MECHANISM.md"


def test_distance_ladder_matches_the_committed_report():
    """The plotted ladder must be the one E1_DISTANCE_SWEEP.md and E1_MECHANISM.md quote."""
    df = pf.load_distance_sweep()
    assert list(df["arm"]) == pf.DISTANCE_ORDER
    acc = dict(zip(df["arm"], df["accuracy"]))
    assert acc["local"] == pytest.approx(0.464, abs=1e-3)
    assert acc["back_5"] == pytest.approx(0.292, abs=1e-3)
    assert acc["back_20"] == pytest.approx(0.276, abs=1e-3)
    # fill is the variable held fixed -- if it ever drifts, the ladder is confounded
    assert df["fill"].nunique() == 1
    assert df["fill"].iloc[0] == pytest.approx(0.688, abs=1e-3)
    # evidence attention falls monotonically with distance (E1b, r = -0.83)
    share = df["evidence_share"].tolist()
    assert share == sorted(share, reverse=True)
    assert share[0] == pytest.approx(0.0408, abs=1e-3)
    assert share[-1] == pytest.approx(0.0124, abs=1e-3)


def test_share_dose_uses_the_balanced_panel_and_has_no_knee():
    """Raw per-level means confound dose with item set; the balanced panel is the comparison."""
    df = pf.load_share_dose()
    assert df["n"].nunique() == 1 and df["n"].iloc[0] == 131
    assert df["share"].is_monotonic_increasing
    assert df["accuracy"].iloc[0] == pytest.approx(0.275, abs=1e-3)   # share 0.012
    assert df["accuracy"].iloc[-1] == pytest.approx(0.473, abs=1e-3)  # natural
    # No threshold. "Smooth" is the claim E1f replaced the "knee" with, so the test is that no
    # single step carries the decline: the largest is a small fraction of the total drop. (An
    # earlier version of this test asserted <= 0.038, copied from a sentence in E1_MECHANISM.md
    # that its own table contradicted -- the real maximum is the natural -> 0.036 step at 0.053,
    # which is also the largest dose step in share terms.)
    steps = df["accuracy"].diff().dropna().abs()
    total = df["accuracy"].iloc[-1] - df["accuracy"].iloc[0]
    assert steps.max() <= 0.06
    assert steps.max() < total / 3


def test_competition_arms_are_matched_on_everything_but_confusability():
    df = pf.load_competition()
    assert list(df["arm"]) == pf.COMPETITION_ORDER
    assert df["n"].nunique() == 1, "paired design: every arm must see the same probes"
    # Fill is the variable held fixed. The arms differ by 2.3% because DDXPlus vignettes vary in
    # length and each arm draws a different subset; the guard that matters is that length does not
    # *order* the arms, which is checked below.
    assert (df["fill"].max() - df["fill"].min()) / df["fill"].mean() < 0.03


def test_competition_result_is_not_a_context_length_artifact():
    """Shorter context would be the obvious confound for near_dup, so it must not track accuracy."""
    df = pf.load_competition().set_index("arm")
    assert df.loc["near_dup", "fill"] == df["fill"].min(), "near_dup should be the shortest arm"
    assert df.loc["near_dup", "accuracy"] == df["accuracy"].min()
    # ...and the *longest* arm is not the most accurate, so length does not order the arms at all
    assert df["fill"].idxmax() != df["accuracy"].idxmax()


def test_format_erosion_ladders_match_reports():
    """E6 ladders: erosion orders by applicability, and accuracy is the corrected grade."""
    mm = pf.load_format_erosion("mmlu").set_index("depth")
    # total, immediate collapse (E6_FORMAT_EROSION.md)
    assert mm.loc[0, "compliance"] == pytest.approx(0.875)
    assert mm.loc[3, "compliance"] == 0.0
    assert (mm.loc[3:, "compliance"] <= 0.025).all()
    # the corrected accuracy, not the CSV's stale pre-fix grade (0.075/0.275)
    assert mm.loc[3, "accuracy"] == pytest.approx(0.500, abs=1e-3)
    assert mm.loc[7, "accuracy"] == pytest.approx(0.525, abs=1e-3)
    code = pf.load_format_erosion("code").set_index("depth")
    assert (code["compliance"] >= 0.875).all()
    assert code.loc[15, "n"] == 8  # overflow-starved cell, annotated in the figure
    gsm = pf.load_format_erosion("gsm8k").set_index("depth")
    assert gsm.loc[15, "compliance"] == pytest.approx(0.600)
    # enrichment never falls with erosion: every arm ends at or above its cold-start value
    for arm in pf.EROSION_ARMS:
        df = pf.load_format_erosion(arm)
        assert df["enrichment"].iloc[-1] >= df["enrichment"].iloc[0] - 0.05


def test_format_recovery_matches_report():
    df = pf.load_format_recovery().set_index("arm")
    assert list(df.index) == pf.RECOVERY_ORDER
    assert df.loc["natural", "compliance"] == 0.0
    assert df.loc["natural", "accuracy"] == pytest.approx(0.675)
    assert (df.loc[["upclamp", "refresh", "both"], "compliance"] == 1.0).all()
    assert df.loc["upclamp", "accuracy"] == pytest.approx(0.425)
    assert df.loc["refresh", "accuracy"] == pytest.approx(0.500)
    assert df.loc["both", "accuracy"] == pytest.approx(0.275)


def _write_row_npz(rows_dir, name, seq_len, ev_span, arm, distance):
    import json

    import numpy as np

    rng = np.random.default_rng(0)
    row = rng.dirichlet(np.ones(seq_len)).astype(np.float16)
    np.savez_compressed(
        rows_dir / name, row=row,
        input_ids=np.arange(seq_len, dtype=np.int32),
        meta=json.dumps({"session": 0, "depth": 21, "probe": 5, "arm": arm,
                         "distance": distance, "evidence_span": list(ev_span),
                         "question_span": [seq_len - 40, seq_len - 5],
                         "pathology": "Bronchitis"}))


def _fake_decode(tid):
    """Offline stand-in for the tokenizer: short words with spaces and a few newlines."""
    return "\nturn:" if tid % 60 == 0 else f" tok{tid}"


def test_token_heatmap_builds_from_stored_rows(tmp_path):
    """Stage 4 of the per-token capture program: the transcript's own text, each token
    shaded by its measured attention — built from stored rows, offline via a fake decode."""
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    _write_row_npz(rows_dir, "s0_d21_p5_local.npz", 300, (200, 260), "local", 0)
    _write_row_npz(rows_dir, "s0_d21_p5_back_10.npz", 300, (40, 100), "back_10", 10)

    out = tmp_path / "figs"
    out.mkdir()
    path = pf.fig_token_heatmap(out, rows_dir=rows_dir, decode=_fake_decode)
    assert path.exists() and path.suffix == ".pdf" and path.stat().st_size > 0


def test_token_heatmap_requires_a_matched_pair(tmp_path):
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    _write_row_npz(rows_dir, "s0_d21_p5_local.npz", 300, (200, 260), "local", 0)
    _write_row_npz(rows_dir, "s0_d21_p6_back_10.npz", 300, (40, 100), "back_10", 10)
    with pytest.raises(FileNotFoundError, match="pair"):
        pf.fig_token_heatmap(tmp_path, rows_dir=rows_dir)
