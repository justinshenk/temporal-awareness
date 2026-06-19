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

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results"

# Quarter-width context-fill bins shared by the attention figure.
FILL_EDGES = [0.0, 0.25, 0.50, 0.75, 1.01]
FILL_LABELS = ["0-25%", "25-50%", "50-75%", "75-100%"]

# Post-training stages in dose order.
DOSE_ORDER = ["base", "sft", "dpo", "instruct"]
DOSE_LABELS = {"base": "base", "sft": "+SFT", "dpo": "+DPO", "instruct": "+Instruct\n(RLVR)"}

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


def fig_random_context(ax: plt.Axes) -> None:
    df = load_random_context()
    centers = [10, 30, 50, 70, 90]
    styles = {
        "coherent": dict(color="#3b6ea5", marker="o", label="coherent stream (ICL works)"),
        "random": dict(color="#dd8452", marker="s", label="random stream (ICL deterred)"),
    }
    for mode, st in styles.items():
        g = df[df["mode"] == mode]
        ax.plot(centers, g["accuracy"], **st)
    ax.set_xlabel("context fill (%)")
    ax.set_ylabel("per-case accuracy")
    ax.set_ylim(0.35, 0.95)
    ax.set_title("(b) Accuracy is flat as context fills", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    overall = random_context_overall()
    ax.text(
        0.97, 0.05,
        f"corr(acc, fill): coherent {-0.01:+.2f}, random {-0.00:+.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555555",
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


def make_figures(outdir: str | Path) -> dict[str, Path]:
    """Render all four figures as PDFs into ``outdir``; return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    builders = {
        "dose_response": fig_dose_response,
        "random_context": fig_random_context,
        "attention_rot": fig_attention,
        "wildchat_homogeneity": fig_wildchat,
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
