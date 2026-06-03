"""Partition WildChat by its OWN homogeneity → isolate heterogeneity as the driver.

Holds format and dataset constant (all open-ended WildChat chat) and asks two questions
from one extraction:

  Q1 (output signature): does the own-confidence entropy collapse track *homogeneity*?
     Prediction: homogeneous conversations collapse (negative entropy-vs-depth slope),
     heterogeneous ones stay flat → corr(homogeneity, entropy_slope) < 0, surviving a
     length control (partial correlation on tokens; homogeneous chats may be shorter).

  Q2 (attention signature): does current-query dilution track *length* independent of
     homogeneity? Prediction: frac_current falls with fill regardless of homogeneity →
     corr(homogeneity, dilution_slope) ≈ 0 while dilution itself is non-zero.

If entropy collapse tracks homogeneity but dilution tracks length, the output and
attention signatures dissociate cleanly within a single dataset.

    uv run python -m scripts.context_fatigue.analyze_wildchat_homogeneity \
        --in-dir results/context_fatigue/wildchat_homogeneity
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.probes.context_fatigue.instruction_checks import pearson

DILUTION_LAYER = 14
DEPTH_MIN_ATTN = 2  # exclude depth 0/1 (frac_current mechanically ~1 with no prior context)


def _finite(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def corr(x, y):
    x, y = _finite(x, y)
    return pearson(x.tolist(), y.tolist()) if len(x) >= 3 else float("nan")


def residualize(y, z):
    y, z = _finite(y, z)
    b = np.polyfit(z, y, 1)
    return y - (b[0] * z + b[1])


def partial_corr(x, y, z):
    """corr(x, y) controlling for z (linear residualization)."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 4:
        return float("nan")
    return pearson(residualize(x[m], z[m]).tolist(), residualize(y[m], z[m]).tolist())


def per_conversation(turns, attn):
    """One row per conversation: homogeneity, length, entropy slope, dilution slope."""
    rows = []
    dil = attn[(attn["layer"] == DILUTION_LAYER) & (attn["depth"] >= DEPTH_MIN_ATTN)]
    dil_by_conv = {c: g for c, g in dil.groupby("conv")}
    for c, g in turns.groupby("conv"):
        if g["depth"].nunique() < 4:
            continue
        e = g[g["depth"] <= g["depth"].quantile(0.3)]["probe_entropy"].mean()
        l = g[g["depth"] >= g["depth"].quantile(0.7)]["probe_entropy"].mean()
        dg = dil_by_conv.get(c)
        rows.append({
            "conv": c,
            "homogeneity": g["homogeneity"].iloc[0],
            "n_boundaries": len(g),
            "max_tokens": g["context_tokens"].max(),
            "entropy_slope": corr(g["depth"], g["probe_entropy"]),
            "entropy_late_over_early": (l / e) if e and np.isfinite(e) and e > 1e-6 else np.nan,
            "dilution_slope": corr(dg["context_fill"], dg["frac_current"]) if dg is not None and len(dg) >= 4 else np.nan,
        })
    return pd.DataFrame(rows)


def tertile_contrast(cdf):
    lo, hi = cdf["homogeneity"].quantile(1 / 3), cdf["homogeneity"].quantile(2 / 3)
    homo = cdf[cdf["homogeneity"] >= hi]
    hetero = cdf[cdf["homogeneity"] <= lo]
    def desc(d):
        return {"n": len(d), "mean_homogeneity": float(d["homogeneity"].mean()),
                "median_entropy_slope": float(d["entropy_slope"].median()),
                "median_late_over_early": float(d["entropy_late_over_early"].median()),
                "median_max_tokens": float(d["max_tokens"].median())}
    return {"homogeneous_top3rd": desc(homo), "heterogeneous_bot3rd": desc(hetero)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/context_fatigue/wildchat_homogeneity")
    args = ap.parse_args()
    d = Path(args.in_dir)

    turns = pd.read_csv(d / "turns.csv")
    attn = pd.read_csv(d / "attention.csv")
    cdf = per_conversation(turns, attn)
    cdf.to_csv(d / "per_conversation.csv", index=False)

    # Q1: heterogeneity drives entropy collapse
    q1_corr = corr(cdf["homogeneity"], cdf["entropy_slope"])
    q1_partial = partial_corr(cdf["entropy_slope"], cdf["homogeneity"], cdf["max_tokens"])
    # Q2: dilution tracks length, not homogeneity
    pooled_dilution = corr(attn[(attn.layer == DILUTION_LAYER) & (attn.depth >= DEPTH_MIN_ATTN)]["context_fill"],
                           attn[(attn.layer == DILUTION_LAYER) & (attn.depth >= DEPTH_MIN_ATTN)]["frac_current"])
    q2_corr = corr(cdf["homogeneity"], cdf["dilution_slope"])

    result = {
        "n_conversations": len(cdf),
        "homogeneity_range": [float(cdf["homogeneity"].min()), float(cdf["homogeneity"].max())],
        "Q1_entropy_collapse_vs_homogeneity": {
            "corr_homogeneity_entropy_slope": q1_corr,
            "partial_corr_controlling_tokens": q1_partial,
            "note": "negative = homogeneous conversations collapse (entropy falls with depth)",
        },
        "Q2_dilution_vs_length": {
            "pooled_corr_fracCurrent_fill_L14_depthge2": pooled_dilution,
            "corr_homogeneity_dilution_slope": q2_corr,
            "note": "dilution (neg pooled corr) present but ~independent of homogeneity (q2 corr ~0) = length-driven",
        },
        "tertile_contrast": tertile_contrast(cdf),
    }
    (d / "homogeneity_analysis.json").write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
