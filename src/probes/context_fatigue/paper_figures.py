"""Figure generation for the context-fatigue extended abstract.

Loads the consolidated context-fatigue result data and renders the four headline
figures used in ``context_fatigue_paper/context_fatigue.tex``:

1. OLMo-2 post-training dose-response (baseline-confidence collapse).
2. Per-case accuracy vs context fill, random vs coherent stream.
3. OLMo-2-Instruct attention reallocation vs context fill, with flat accuracy.
4. WildChat-vs-DDXPlus entropy and the homogeneity (not length) driver.

Figures 1--3 are computed from the committed result files under ``results/``.
Figure 4 uses summary values reported in the WildChat result markdown
(``results/context_fatigue/wildchat_*/``); no raw WildChat data is committed, so
those numbers are kept here as documented constants (see ``WILDCHAT_SUMMARY``)
and asserted against the source reports in the tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.bootstrap_stats import bootstrap_interval
from src.probes.context_fatigue.instruction_checks import check_clinical_format

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results"

# Quarter-width context-fill bins shared by the attention figure.
FILL_EDGES = [0.0, 0.25, 0.50, 0.75, 1.01]
FILL_LABELS = ["0-25%", "25-50%", "50-75%", "75-100%"]

# Post-training stages in dose order.
DOSE_ORDER = ["base", "sft", "dpo", "instruct"]
DOSE_LABELS = {"base": "base", "sft": "+SFT", "dpo": "+DPO", "instruct": "+Instruct\n(RLVR)"}

# E6 format-erosion filler arms, ordered by the applicability of their demonstrated answer shape.
EROSION_ARMS = ["code", "gsm8k", "mmlu"]
EROSION_LABELS = {
    "code": "code (inapplicable shape)",
    "gsm8k": "gsm8k (loosely applicable)",
    "mmlu": "mmlu (drop-in shape)",
}
EROSION_COLORS = {"code": "#55a868", "gsm8k": "#dd8452", "mmlu": "#c44e52"}
RECOVERY_ORDER = ["natural", "upclamp", "refresh", "both"]

# Reported summaries from results/context_fatigue/wildchat_*/ (see module docstring).
WILDCHAT_SUMMARY = {
    # within-conversation late/early entropy ratio (1.0 = flat, <1 = collapse)
    "ddxplus_late_over_early": 0.30,  # 0.13 / 0.47, the synthetic repeating-task collapse
    "wildchat_late_over_early": 0.99,  # WILDCHAT_DYNAMICS.md median
    # homogeneity tertiles: within-conversation late/early entropy ratio
    "homogeneous_late_over_early": 0.897,  # WILDCHAT_HOMOGENEITY.md (top third)
    "heterogeneous_late_over_early": 1.001,  # WILDCHAT_HOMOGENEITY.md (bottom third)
    # entropy-slope correlation with homogeneity, controlling for length
    "partial_corr_homogeneity_entropy_slope": -0.151,
}


def load_dose_response() -> pd.DataFrame:
    """OLMo-2 base->SFT->DPO->Instruct dose-response (results/olmo_gradient)."""
    path = RESULTS / "olmo_gradient" / "gradient.json"
    df = pd.DataFrame(json.loads(path.read_text()))
    df["stage"] = pd.Categorical(df["stage"], categories=DOSE_ORDER, ordered=True)
    return df.sort_values("stage").reset_index(drop=True)


def load_random_context() -> pd.DataFrame:
    """Per-case accuracy by fill bin, random vs coherent (results/random_context)."""
    path = RESULTS / "random_context" / "accuracy_by_fill.csv"
    return pd.read_csv(path)


def random_context_overall() -> dict[str, float]:
    """n-weighted overall accuracy per stream mode."""
    df = load_random_context()
    out = {}
    for mode, g in df.groupby("mode"):
        out[mode] = float((g["accuracy"] * g["n"]).sum() / g["n"].sum())
    return out


def load_attention_by_fill(model: str = "instruct", layer: int = 24) -> pd.DataFrame:
    """Mean attention mass + entropy + accuracy by fill bin at one layer.

    Reads results/olmo_attention_{model}/attention_stats.csv (per session/case/
    layer/head). Attention mass is averaged over heads and cases within each bin;
    accuracy is computed once per (session, case) so heads/layers do not inflate n.
    """
    path = RESULTS / f"olmo_attention_{model}" / "attention_stats.csv"
    df = pd.read_csv(path)
    df["fill_bin"] = pd.cut(
        df["context_fill"], bins=FILL_EDGES, labels=FILL_LABELS, include_lowest=True
    )

    layer_df = df[df["layer"] == layer]
    mass = layer_df.groupby("fill_bin", observed=True)[
        ["frac_system", "frac_early_cases", "frac_recent_cases", "frac_current_query", "attention_entropy"]
    ].mean()

    # one row per case for an unbiased accuracy / n
    per_case = df.drop_duplicates(["session", "case"])
    acc = per_case.groupby("fill_bin", observed=True)["correct"].agg(["mean", "count"])
    acc.columns = ["accuracy", "n_cases"]

    out = mass.join(acc).reindex(FILL_LABELS)
    return out.reset_index()


def attention_corr_at_layer(model: str = "instruct", layer: int = 24) -> dict[str, float]:
    """Reported attention<->fill correlations at a layer (attention_performance.csv)."""
    path = RESULTS / f"olmo_attention_{model}" / "attention_performance.csv"
    df = pd.read_csv(path)
    row = df[df["layer"] == layer].iloc[0]
    return {
        "system": float(row["corr_system_fill"]),
        "recent": float(row["corr_recent_fill"]),
        "current": float(row["corr_current_fill"]),
        "entropy": float(row["corr_entropy_fill"]),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

# ── the dilution program (E1 / E1b / E1f / E3) ──────────────────────────

CF = RESULTS / "context_fatigue"
DISTANCE_ORDER = ["local", "back_2", "back_5", "back_10", "back_20"]
DISTANCE_LABELS = {"local": "local\n(0)", "back_2": "2", "back_5": "5",
                   "back_10": "10", "back_20": "20"}
COMPETITION_ORDER = ["disjoint", "random", "near_dup"]
COMPETITION_LABELS = {"disjoint": "disjoint\n(0 shared)", "random": "random\n(0.8 shared)",
                      "near_dup": "near-dup\n(3.7 shared)"}


def _arm_accuracy(path: Path, order: list[str]) -> pd.DataFrame:
    """Per-arm accuracy from a driver's ``turns.csv``, with case-resampled CIs, in arm order."""
    df = pd.read_csv(path)
    out = (df.groupby("arm")
             .agg(n=("correct", "size"), accuracy=("correct", "mean"),
                  fill=("context_fill", "mean"))
             .reindex(order).reset_index())
    los, his = [], []
    for arm in out["arm"]:
        vals = df.loc[df["arm"] == arm, "correct"].to_numpy(dtype=float)
        interval = bootstrap_interval(vals, np.mean)
        los.append(interval.lo)
        his.append(interval.hi)
    return out.assign(lo=los, hi=his)


def load_distance_sweep() -> pd.DataFrame:
    """E1 accuracy by evidence distance, with E1b's evidence attention share at L24.

    Accuracy and share come from two runs (``e1_distance_sweep`` and ``e1_with_attention``); the
    second reproduced the first's accuracies exactly, which is why they can be shown on one axis.
    """
    acc = _arm_accuracy(CF / "e1_distance_sweep" / "turns.csv", DISTANCE_ORDER)
    attn = pd.read_csv(CF / "e1_with_attention" / "turns.csv")
    share = (attn.groupby("arm")["evidence_share"].mean().reindex(DISTANCE_ORDER)
             .reset_index(name="evidence_share"))
    return acc.merge(share, on="arm")


def load_share_dose() -> pd.DataFrame:
    """E1f: accuracy against clamped evidence share, on the balanced item panel.

    Levels at or above an item's natural share cannot be reached, so per-level raw ``n`` varies
    131--192. Only the subset present at *every* level is a like-for-like comparison; using the
    raw per-level means would confound the dose with the item set.
    """
    df = pd.read_csv(CF / "e1f_share_knee" / "turns.csv")
    n_levels = df["level"].nunique()
    counts = df.groupby("probe")["level"].nunique()
    balanced = df[df["probe"].isin(counts[counts == n_levels].index)]
    out = (balanced.groupby("level")
                   .agg(n=("correct", "size"), accuracy=("correct", "mean"),
                        share=("achieved_share", "mean"))
                   .reset_index()
                   .sort_values("share").reset_index(drop=True))
    return out


def load_competition() -> pd.DataFrame:
    """E3: accuracy by context confusability at fixed distance and fill."""
    return _arm_accuracy(CF / "e3_competition" / "turns.csv", COMPETITION_ORDER)


def _corrected_accuracy(df: pd.DataFrame) -> pd.Series:
    """Per-row accuracy with the lead-line grader fix applied to stored replies.

    The committed E6 CSVs predate the fix for bare-letter-line replies ("B\\n<prose>"), which
    the original grader scored unparsed *and* wrong -- fabricating an accuracy collapse exactly
    where compliance collapses. Rows the original grader parsed keep their options-aware grade
    (its bug was only ever a failure to parse, never a mis-parse); rows it failed on are
    re-graded with the current checker. Reproduces the corrected report numbers
    (mmlu depths 3/7: 0.500/0.525, E6_FORMAT_EROSION.md).
    """
    regraded = [check_clinical_format(str(r) if pd.notna(r) else "", "")["answer"]
                for r in df["response"]]
    return pd.Series(
        [bool(oc) if op else (rg == g)
         for oc, op, rg, g in zip(df["correct"], df["parsed"], regraded, df["gold"])],
        index=df.index,
    )


def load_format_erosion(arm: str) -> pd.DataFrame:
    """E6: per-depth compliance, corrected accuracy, and system enrichment for one filler arm."""
    df = pd.read_csv(CF / f"e6_{arm}" / "turns.csv")
    df["corrected"] = _corrected_accuracy(df)
    return (df.groupby("depth")
              .agg(fill=("fill", "mean"), compliance=("fully_compliant", "mean"),
                   accuracy=("corrected", "mean"), enrichment=("system_enrichment", "mean"),
                   n=("fill", "size"))
              .reset_index())


def load_format_recovery() -> pd.DataFrame:
    """E6 recovery arms at depth 42, with the same run's natural cell as the baseline."""
    df = pd.read_csv(CF / "e6_mmlu_recovery" / "turns.csv")
    df = df[df["depth"] == 42].copy()
    df["arm"] = df["recovery_arm"].fillna("natural")
    out = (df.groupby("arm")
             .agg(compliance=("fully_compliant", "mean"), accuracy=("correct", "mean"),
                  share=("system_share", "mean"), n=("fill", "size"))
             .reindex(RECOVERY_ORDER).reset_index())
    return out


def fig_format_erosion(ax: plt.Axes) -> None:
    """E6: compliance by fill per filler arm, with mmlu accuracy unharmed through its collapse."""
    for arm in EROSION_ARMS:
        df = load_format_erosion(arm)
        ax.plot(df["fill"], df["compliance"], marker="o", color=EROSION_COLORS[arm],
                label=EROSION_LABELS[arm])
        thin = df[df["n"] < 20]
        if not thin.empty:
            ax.annotate(f"n={int(thin['n'].iloc[0])}", (thin["fill"].iloc[0],
                        thin["compliance"].iloc[0] - 0.04), ha="center", va="top",
                        fontsize=7, color="#555555")
    mm = load_format_erosion("mmlu")
    ax.plot(mm["fill"], mm["accuracy"], marker="D", markersize=4, color="#000000",
            linestyle="--", linewidth=1, label="accuracy (mmlu)")
    ax.set_xlabel("context fill")
    ax.set_ylabel("format compliance / accuracy")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("(a) An applicable shape erodes the format;\naccuracy never pays", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.62, 0.12))


def fig_format_enrichment(ax: plt.Axes) -> None:
    """E6: the system prompt's per-token attention is flat-to-rising in every arm."""
    for arm in EROSION_ARMS:
        df = load_format_erosion(arm)
        ax.plot(df["fill"], df["enrichment"], marker="o", color=EROSION_COLORS[arm],
                label=EROSION_LABELS[arm])
    ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1)
    ax.text(0.02, 1.0, "token-share parity", transform=ax.get_yaxis_transform(), fontsize=7,
            va="bottom", color="#555555")
    ax.set_xlabel("context fill")
    ax.set_ylabel("system-prompt enrichment\n(share / token fraction)")
    ax.set_ylim(0, 3.4)
    ax.set_title("(b) The instruction's per-token attention\nnever falls", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")


def fig_format_recovery(ax: plt.Axes) -> None:
    """E6: at depth 42, re-weighting and re-presenting each fully restore compliance."""
    df = load_format_recovery()
    x = np.arange(len(df))
    colors = ["#8fb8de", "#4c72b0", "#55a868", "#dd8452"]
    ax.bar(x, df["compliance"], 0.6, color=colors, label="compliance")
    ax.plot(x, df["accuracy"], marker="D", color="#000000", linestyle="none",
            label="accuracy")
    for i, r in df.iterrows():
        ax.annotate(f"{r['accuracy']:.2f}", (i, r["accuracy"] + 0.045), ha="center",
                    fontsize=7.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(df["arm"], fontsize=8)
    ax.set_ylabel("format compliance / accuracy")
    ax.set_ylim(0, 1.18)
    ax.set_title("(c) Either lever restores compliance;\nre-weighting pays in accuracy", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")


def fig_dose_response(ax: plt.Axes) -> None:
    df = load_dose_response()
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["entropy_early"], w, label="early context", color="#3b6ea5")
    ax.bar(x + w / 2, df["entropy_late"], w, label="late context", color="#c44e52")
    for i, r in df.iterrows():
        ax.annotate(
            f"{r['entropy_ratio']:.2f}×",
            (i, max(r["entropy_early"], r["entropy_late"]) + 0.03),
            ha="center", fontsize=8, color="#333333",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DOSE_LABELS[s] for s in df["stage"]])
    ax.set_ylabel("next-token entropy (nats)")
    ax.set_title("(a) Post-training collapses baseline confidence", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.margins(y=0.18)


def _wilson_interval(acc: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    """Wilson 95% score interval for a binomial proportion, per bin."""
    center = (acc + z**2 / (2 * n)) / (1 + z**2 / n)
    half = (z / (1 + z**2 / n)) * np.sqrt(acc * (1 - acc) / n + z**2 / (4 * n**2))
    return center - half, center + half


def fig_random_context(ax: plt.Axes) -> None:
    df = load_random_context()
    centers = [10, 30, 50, 70, 90]
    styles = {
        "coherent": dict(color="#3b6ea5", marker="o", label="coherent stream (ICL works)"),
        "random": dict(color="#dd8452", marker="s", label="random stream (ICL deterred)"),
    }
    for mode, st in styles.items():
        g = df[df["mode"] == mode]
        lo, hi = _wilson_interval(g["accuracy"], g["n"])
        yerr = np.vstack([g["accuracy"] - lo, hi - g["accuracy"]])
        ax.errorbar(centers, g["accuracy"], yerr=yerr, capsize=2.5,
                    linewidth=1.4, elinewidth=0.9, **st)
        for x, acc, n in zip(centers, g["accuracy"], g["n"]):
            ax.annotate(f"{n}", (x, acc), textcoords="offset points",
                        xytext=(0, -11), ha="center", fontsize=6, color="#888888")
    ax.set_xlabel("context fill (%)")
    ax.set_ylabel("per-case accuracy")
    ax.set_ylim(0.25, 1.0)
    ax.set_title("(b) Accuracy is flat as context fills", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    overall = random_context_overall()
    ax.text(
        0.03, 0.97,
        f"overall: coherent {overall['coherent']:.2f}, random {overall['random']:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="#555555",
    )


def fig_attention(ax: plt.Axes) -> None:
    df = load_attention_by_fill("instruct", layer=24)
    x = np.arange(len(df))
    series = [
        ("frac_system", "system prompt", "#55a868"),
        ("frac_current_query", "current query", "#4c72b0"),
        ("frac_recent_cases", "recent cases", "#c44e52"),
    ]
    for col, lab, c in series:
        ax.plot(x, df[col], marker="o", color=c, label=lab)
    ax.plot(x, df["accuracy"], marker="D", color="#000000", linestyle="--", label="accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(df["fill_bin"])
    ax.set_xlabel("context fill")
    ax.set_ylabel("attention mass / accuracy")
    ax.set_title("(c) Attention reallocates; accuracy holds (L24)", fontsize=10)
    ax.set_ylim(0, 0.92)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center")


def fig_wildchat(ax: plt.Axes) -> None:
    s = WILDCHAT_SUMMARY
    labels = ["DDXPlus\n(repeating)", "WildChat\n(organic)", "homog.\ntertile", "heterog.\ntertile"]
    vals = [
        s["ddxplus_late_over_early"],
        s["wildchat_late_over_early"],
        s["homogeneous_late_over_early"],
        s["heterogeneous_late_over_early"],
    ]
    colors = ["#c44e52", "#3b6ea5", "#dd8452", "#8fb8de"]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.6)
    ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1)
    ax.text(0.02, 1.0, "flat", transform=ax.get_yaxis_transform(), fontsize=7,
            va="bottom", color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("within-conv. late/early entropy")
    ax.set_ylim(0, 1.15)
    ax.set_title("(d) Collapse needs a homogeneous task", fontsize=10)


def fig_distance_ladder(ax: plt.Axes) -> None:
    """E1/E1b: displacing the evidence drains its attention and costs accuracy, at fixed fill."""
    df = load_distance_sweep()
    x = np.arange(len(df))
    err = np.vstack([df["accuracy"] - df["lo"], df["hi"] - df["accuracy"]])
    ax.bar(x, df["accuracy"], 0.6, color="#4c72b0", yerr=err, capsize=2.5,
           error_kw=dict(lw=0.9, ecolor="#2a3f5f"), label="accuracy")
    ax.axhline(0.200, color="#555555", linestyle=":", linewidth=1)
    ax.text(0.98, 0.205, "chance", transform=ax.get_yaxis_transform(), fontsize=7,
            va="bottom", ha="right", color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([DISTANCE_LABELS[a] for a in df["arm"]])
    ax.set_xlabel("evidence distance (user turns back)")
    ax.set_ylabel("per-case accuracy")
    ax.set_ylim(0, 0.62)

    rhs = ax.twinx()
    rhs.plot(x, df["evidence_share"], marker="o", color="#c44e52", label="evidence attention")
    rhs.set_ylabel("evidence attention share @L24", color="#c44e52")
    rhs.tick_params(axis="y", colors="#c44e52")
    rhs.set_ylim(0, 0.052)

    ax.set_title("(a) Displace the evidence: mass and accuracy fall together", fontsize=9.5)
    ax.text(0.97, 0.94, f"fill = {df['fill'].iloc[0]:.3f} in every arm",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#555555")


def fig_mass_dose(ax: plt.Axes) -> None:
    """E1f: the share->accuracy dose-response is smooth and shallow -- no threshold."""
    df = load_share_dose()
    ax.plot(df["share"], df["accuracy"], marker="o", color="#4c72b0",
            label=f"clamped at local (n={int(df['n'].iloc[0])})")
    nat = df[df["level"] == "natural"].iloc[0]
    ax.scatter([nat["share"]], [nat["accuracy"]], s=70, facecolor="white",
               edgecolor="#4c72b0", zorder=5, label="natural share (unclamped)")
    # E1c measured the same endpoint contrast in a separate run; the agreement is the check.
    ax.scatter([0.0125], [0.333], marker="D", s=36, color="#c44e52", zorder=5,
               label="separate clamp run (agrees to 0.004)")
    ax.set_xlabel("evidence attention share @L24 (clamped)")
    ax.set_ylabel("per-case accuracy")
    ax.set_title("(b) Accuracy is graded in attention mass", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.margins(y=0.14)


def fig_competition(ax: plt.Axes) -> None:
    """E3: confusability of the accumulated context, at fixed distance and fill."""
    df = load_competition()
    x = np.arange(len(df))
    colors = ["#55a868", "#4c72b0", "#c44e52"]
    err = np.vstack([df["accuracy"] - df["lo"], df["hi"] - df["accuracy"]])
    ax.bar(x, df["accuracy"], 0.6, color=colors, yerr=err, capsize=2.5,
           error_kw=dict(lw=0.9, ecolor="#2a3f5f"))
    ax.axhline(0.200, color="#555555", linestyle=":", linewidth=1)
    ax.text(0.98, 0.205, "chance", transform=ax.get_yaxis_transform(), fontsize=7,
            va="bottom", ha="right", color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([COMPETITION_LABELS[a] for a in df["arm"]], fontsize=8)
    ax.set_xlabel("context confusability")
    ax.set_ylabel("per-case accuracy")
    ax.set_ylim(0, 0.72)
    ax.set_title("(c) Competition costs accuracy at distance 0", fontsize=9.5)
    ax.text(0.97, 0.94, f"n={int(df['n'].iloc[0])}/arm, fill {df['fill'].iloc[0]:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#555555")


def make_figures(outdir: str | Path) -> dict[str, Path]:
    """Render all four figures as PDFs into ``outdir``; return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    builders = {
        "distance_ladder": fig_distance_ladder,
        "mass_dose": fig_mass_dose,
        "competition": fig_competition,
        "dose_response": fig_dose_response,
        "random_context": fig_random_context,
        "attention_rot": fig_attention,
        "wildchat_homogeneity": fig_wildchat,
        "format_erosion": fig_format_erosion,
        "format_enrichment": fig_format_enrichment,
        "format_recovery": fig_format_recovery,
    }
    paths: dict[str, Path] = {}
    for name, builder in builders.items():
        fig, ax = plt.subplots(figsize=(4.2, 3.1))
        builder(ax)
        fig.tight_layout()
        path = outdir / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths[name] = path
    return paths


# Documented constants for the appendix figures. The OLMo raw run directories are
# not retained under results/, so the OLMo side plots the values recorded in
# E1_DISTANCE_SWEEP.md / E1_MECHANISM.md / E3_COMPETITION.md / E3C_COMPETITOR_CLOSE.md
# and papers' numbers.md; the Qwen side matches QWEN_*.md (raw turns.csv retained).
# Per-point whiskers are Wilson 95% intervals from (accuracy, n); the paired
# bootstrap CIs quoted in the text are the inferential intervals.
FAMILY_COLORS = {"olmo": "#3b6ea5", "qwen": "#dd8452"}
FAMILY_LABELS = {"olmo": "OLMo-2-7B", "qwen": "Qwen2.5-7B"}

DISTANCE_LADDERS = {
    "olmo": dict(distances=[0, 2, 5, 10, 20], accuracy=[0.464, 0.359, 0.292, 0.250, 0.276], n=192),
    "qwen": dict(distances=[0, 2, 5, 10, 20], accuracy=[0.630, 0.531, 0.516, 0.505, 0.469], n=192),
}

# Share->accuracy sweeps; OLMo is L24-indexed (common subset n=131), Qwen is
# all-layer pooled (n=192; the 0.0335 level has n=181). x is share / natural share.
SHARE_SWEEPS = {
    "olmo": dict(
        shares=[0.0441, 0.0360, 0.0320, 0.0290, 0.0250, 0.0200, 0.0160, 0.0120],
        accuracy=[0.473, 0.420, 0.427, 0.389, 0.351, 0.313, 0.313, 0.275],
        n=[131] * 8,
    ),
    "qwen": dict(
        shares=[0.0388, 0.0335, 0.0240, 0.0172, 0.0123, 0.0088, 0.0070],
        accuracy=[0.667, 0.646, 0.641, 0.573, 0.474, 0.432, 0.380],
        n=[192, 181, 192, 192, 192, 192, 192],
    ),
}

# Competition divergence: paired bootstrap point estimates and 95% CIs.
COMPETITION_GAPS = {  # accuracy, random - near_dup
    "olmo": (0.085, 0.030, 0.140),
    "qwen": (0.030, -0.016, 0.074),
}
COMPETITION_SHARE_DELTAS = {  # evidence share, near_dup - random
    "olmo": (-0.00027, -0.00088, 0.00035),
    "qwen": (0.0071, 0.0062, 0.0080),
}

# E3c competitor closure (OLMo, paired n=365) and Appendix H exemplar closure
# (OLMo mmlu depth 42, n=40 per arm).
CLOSURE_COMPETITION = {
    "competitor closure": (0.0548, 0.0055, 0.1041),
    "random-token closure": (-0.0055, -0.0356, 0.0247),
}
CLOSURE_COMPETITION_FULL_GAP = 0.0932  # random - near_dup, same run
CLOSURE_PRECEDENT = {  # compliance after generation-time closure at depth 42 (natural 0.000)
    "all answers": 0.000,
    "partial (dose-matched)": 0.000,
    "filler questions": 0.132,
    "random tokens": 0.000,
}

# Qwen Q5 neutral-context system clamp (n=120/level), QWEN_E5_SYSTEM_CLAMP.md.
QWEN_SYSTEM_CLAMP = dict(
    shares=[0.191, 0.150, 0.120, 0.090, 0.070, 0.0486, 0.038],
    prefix=[1.000, 0.983, 0.100, 0.000, 0.000, 0.000, 0.000],
    suffix=[1.000, 1.000, 0.917, 0.233, 0.017, 0.000, 0.000],
    forbid=[1.000, 1.000, 1.000, 0.975, 0.983, 1.000, 1.000],
    accuracy=[0.567, 0.583, 0.583, 0.617, 0.625, 0.633, 0.658],
    n=120,
)


def _wilson_yerr(acc, n) -> np.ndarray:
    acc = pd.Series(acc, dtype=float)
    n = pd.Series(n, dtype=float) if np.ndim(n) else pd.Series([float(n)] * len(acc))
    lo, hi = _wilson_interval(acc, n)
    return np.vstack([acc - lo, hi - acc])


def fig_qwen_replication(outdir: Path) -> Path:
    """2x2 cross-family panel: ladders, dose-responses, and the competition divergence."""
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6))
    ax_ladder, ax_dose, ax_gap, ax_share = axes.flat

    for fam, d in DISTANCE_LADDERS.items():
        ax_ladder.errorbar(d["distances"], d["accuracy"], yerr=_wilson_yerr(d["accuracy"], d["n"]),
                           color=FAMILY_COLORS[fam], marker="o", capsize=2.5,
                           label=FAMILY_LABELS[fam])
    ax_ladder.set_xlabel("evidence distance (user turns back)")
    ax_ladder.set_ylabel("per-case accuracy")
    ax_ladder.set_title("(a) Displacement replicates", fontsize=10)
    ax_ladder.legend(fontsize=8, frameon=False)

    for fam, d in SHARE_SWEEPS.items():
        frac = np.array(d["shares"]) / d["shares"][0]
        ax_dose.errorbar(frac, d["accuracy"], yerr=_wilson_yerr(d["accuracy"], d["n"]),
                         color=FAMILY_COLORS[fam], marker="o", capsize=2.5,
                         label=FAMILY_LABELS[fam])
    ax_dose.set_xlabel("clamped share / natural share")
    ax_dose.set_ylabel("per-case accuracy")
    ax_dose.set_title("(b) Dose-response replicates: graded, no knee", fontsize=10)
    ax_dose.invert_xaxis()
    ax_dose.legend(fontsize=8, frameon=False, loc="lower left")

    for ax, data, title, xlabel in [
        (ax_gap, COMPETITION_GAPS, "(c) Competition penalty",
         "accuracy gap (random $-$ near_dup)"),
        (ax_share, COMPETITION_SHARE_DELTAS, "(d) The attention inversion",
         "evidence-share $\\Delta$ (near_dup $-$ random)"),
    ]:
        ys = np.arange(len(data))[::-1]
        for y, (fam, (mid, lo, hi)) in zip(ys, data.items()):
            ax.errorbar([mid], [y], xerr=[[mid - lo], [hi - mid]],
                        color=FAMILY_COLORS[fam], marker="o", capsize=3, markersize=6)
        ax.axvline(0.0, color="#555555", linestyle=":", linewidth=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([FAMILY_LABELS[f] for f in data])
        ax.set_ylim(-0.6, len(data) - 0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    path = outdir / "qwen_replication.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_closure_dissociation(outdir: Path) -> Path:
    """Same scale-0 closure, opposite outcomes: competition rescued, precedent untouched."""
    fig, (ax_comp, ax_prec) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    labels = list(CLOSURE_COMPETITION)
    mids = [CLOSURE_COMPETITION[k][0] for k in labels]
    los = [CLOSURE_COMPETITION[k][1] for k in labels]
    his = [CLOSURE_COMPETITION[k][2] for k in labels]
    x = np.arange(len(labels))
    ax_comp.bar(x, mids, width=0.55, color=["#55a868", "#8fb8de"])
    ax_comp.errorbar(x, mids, yerr=[np.subtract(mids, los), np.subtract(his, mids)],
                     fmt="none", ecolor="#333333", capsize=3, linewidth=1)
    ax_comp.axhline(0.0, color="#555555", linewidth=0.8)
    ax_comp.axhline(CLOSURE_COMPETITION_FULL_GAP, color="#c44e52", linestyle="--", linewidth=1)
    ax_comp.text(0.98, CLOSURE_COMPETITION_FULL_GAP, "full penalty", ha="right", va="bottom",
                 fontsize=7.5, color="#c44e52", transform=ax_comp.get_yaxis_transform())
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(labels, fontsize=8)
    ax_comp.set_ylabel("accuracy recovered")
    ax_comp.set_title("(a) Competition: closure rescues 59%", fontsize=10)

    labels = list(CLOSURE_PRECEDENT)
    vals = [CLOSURE_PRECEDENT[k] for k in labels]
    x = np.arange(len(labels))
    ax_prec.bar(x, vals, width=0.6, color="#8fb8de")
    for xi, v in zip(x, vals):
        ax_prec.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                     color="#555555")
    ax_prec.axhline(1.0, color="#55a868", linestyle="--", linewidth=1)
    ax_prec.text(0.02, 1.0, "restatement restores", ha="left", va="bottom", fontsize=7.5,
                 color="#55a868", transform=ax_prec.get_yaxis_transform())
    ax_prec.set_xticks(x)
    ax_prec.set_xticklabels(labels, fontsize=7.5, rotation=12)
    ax_prec.set_ylim(0, 1.12)
    ax_prec.set_ylabel("compliance after closure")
    ax_prec.set_title("(b) Precedent: the same closure restores nothing", fontsize=10)

    fig.tight_layout()
    path = outdir / "closure_dissociation.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_qwen_system_clamp(outdir: Path) -> Path:
    """Qwen Q5: canary-ordered compliance collapse under the system clamp, accuracy intact."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    d = QWEN_SYSTEM_CLAMP
    styles = {
        "prefix": dict(color="#c44e52", marker="o", label="prefix canary"),
        "suffix": dict(color="#dd8452", marker="s", label="suffix canary"),
        "forbid": dict(color="#3b6ea5", marker="^", label="forbidden-phrase canary"),
    }
    for key, st in styles.items():
        ax.errorbar(d["shares"], d[key], yerr=_wilson_yerr(d[key], d["n"]), capsize=2.5, **st)
    ax.errorbar(d["shares"], d["accuracy"], yerr=_wilson_yerr(d["accuracy"], d["n"]),
                color="#555555", marker="D", markersize=4, linestyle="--", capsize=2.5,
                label="accuracy")
    ax.set_xlabel("clamped system-span share")
    ax.set_ylabel("rate")
    ax.set_title("Compliance collapses duty by duty; accuracy does not", fontsize=10)
    ax.invert_xaxis()
    ax.legend(fontsize=7.5, frameon=False, loc="center left")
    fig.tight_layout()
    path = outdir / "qwen_system_clamp.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


HEATMAP_ROWS_DIR = RESULTS / "context_fatigue" / "e1_rows" / "rows"


def fig_token_heatmap(outdir: Path, rows_dir: Path | None = None,
                      arms=("local", "back_10")) -> Path:
    """Per-token attention heatmap of one displacement pair, from stored capture rows.

    The honest version of the attention-explainer's span-tinted mock: each strip is the
    final position's measured all-layer/head-mean attention over the whole transcript, for
    the same probe with its evidence local vs displaced. The evidence span is bracketed and
    its share quoted, so the figure shows the drain the paper measures — not an illustration
    of it. Rows come from ``run_distance_sweep.py --attention-only --store-rows``.
    """
    rows_dir = HEATMAP_ROWS_DIR if rows_dir is None else Path(rows_dir)
    files = {}
    for f in sorted(rows_dir.glob("*.npz")):
        for arm in arms:
            if f.stem.endswith(f"_{arm}"):
                files.setdefault(f.stem[: -len(arm) - 1], {})[arm] = f
    pair = next((v for v in files.values() if len(v) == len(arms)), None)
    if pair is None:
        raise FileNotFoundError(
            f"no probe in {rows_dir} has a stored row for every arm in {arms} — "
            "no displacement pair to draw")

    fig, axes = plt.subplots(len(arms), 1, figsize=(7.4, 1.25 * len(arms) + 0.7),
                             sharex=True)
    vmin = 1e-5
    for ax, arm in zip(np.atleast_1d(axes), arms):
        z = np.load(pair[arm])
        row = z["row"].astype(np.float32)
        meta = json.loads(str(z["meta"]))
        ax.imshow(np.log10(np.maximum(row, vmin))[None, :], aspect="auto",
                  cmap="magma", vmin=np.log10(vmin), vmax=np.log10(row.max()),
                  interpolation="nearest", extent=(0, len(row), 0, 1))
        for span, color, label in ((meta["evidence_span"], "#55a868", "evidence"),
                                   (meta["question_span"], "#8fb8de", "question")):
            a, b = span
            ax.plot([a, b], [-0.18, -0.18], color=color, linewidth=3,
                    clip_on=False, solid_capstyle="butt")
            ax.text((a + b) / 2, -0.32, label, ha="center", va="top", fontsize=7,
                    color=color, clip_on=False)
        share = float(row[meta["evidence_span"][0]:meta["evidence_span"][1]].sum())
        ax.set_yticks([])
        ax.set_ylabel(arm, rotation=0, ha="right", va="center", fontsize=9)
        ax.text(0.995, 0.78, f"evidence share {share:.4f}", ha="right", va="center",
                fontsize=7.5, color="white", transform=ax.transAxes)
    np.atleast_1d(axes)[-1].set_xlabel("token position in transcript")
    np.atleast_1d(axes)[0].set_title(
        "Final-position attention over the transcript (all-layer mean, log color)",
        fontsize=10)

    fig.tight_layout()
    path = Path(outdir) / "token_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_appendix_figures(outdir: str | Path) -> dict[str, Path]:
    """Render the appendix figures (documented constants; no raw OLMo runs needed).

    The token heatmap is included only when its stored-row artifacts exist — they live in
    gitignored results/, so a fresh clone still renders the constant-based figures.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figs = {
        "qwen_replication": fig_qwen_replication(outdir),
        "closure_dissociation": fig_closure_dissociation(outdir),
        "qwen_system_clamp": fig_qwen_system_clamp(outdir),
    }
    if HEATMAP_ROWS_DIR.exists():
        figs["token_heatmap"] = fig_token_heatmap(outdir)
    return figs
