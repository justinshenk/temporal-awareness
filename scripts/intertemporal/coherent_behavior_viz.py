#!/usr/bin/env python
"""
Polished visualizations for behavioral coherence analysis.

Reads responses.json and generates publication-quality plots
highlighting the key findings from the analysis.

Usage:
    python scripts/intertemporal/coherent_behavior_viz.py out/behavioral/investment_behave_full/
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style & Constants
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "Qwen2.5-3B-Instruct",
    "Qwen3-4B",
    "Qwen3-4B-Instruct-2507",
    "claude-haiku-4-5-20251001",
]

MODEL_SHORT = {
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen3-4B": "Qwen3-4B",
    "Qwen3-4B-Instruct-2507": "Qwen3-4B-Inst",
    "claude-haiku-4-5-20251001": "Claude Haiku",
}

MODEL_COLORS = {
    "Qwen2.5-3B-Instruct": "#4C72B0",
    "Qwen3-4B": "#55A868",
    "Qwen3-4B-Instruct-2507": "#DD8452",
    "claude-haiku-4-5-20251001": "#C44E52",
}

HORIZON_MONTHS_ORDER = [None, 1, 3, 6, 12, 24, 60, 120, 240, 600]
HORIZON_LABELS = {
    None: "None", 1: "1mo", 3: "3mo", 6: "6mo",
    12: "1y", 24: "2y", 60: "5y", 120: "10y", 240: "20y", 600: "50y",
}

# Zones for coherence
BEFORE_ANCHOR = {1, 3}       # Before short-term delivery (6mo)
EXACT_SHORT = {6}            # Exact match short-term
BETWEEN_ANCHOR = {12, 24, 60}  # Between 6mo and 10y
EXACT_LONG = {120}           # Exact match long-term
BEYOND_ANCHOR = {240, 600}   # Beyond long-term delivery (10y)


def _apply_style():
    """Set global matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _th_to_months(th):
    if th is None:
        return None
    v, u = th["value"], th["unit"]
    return v if u == "months" else v * 12


def _get_choice(row, mk):
    return row.get(f"{mk}_choice")


def _get_label_style(row):
    labels = row["labels"]
    return "ab" if "a)" in labels[0] or "b)" in labels[0] else "xy"


def _detect_models(data):
    """Return model keys present in data, in canonical order."""
    sample = data[0]
    present = [k.replace("_choice", "") for k in sample if k.endswith("_choice")]
    return [m for m in MODEL_ORDER if m in present] or present


def _add_thm(data):
    for d in data:
        d["thm"] = _th_to_months(d["time_horizon"])


def _save(fig, output_dir, name):
    path = output_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 1: Coherence Curve (line plot with shaded zones)
# ---------------------------------------------------------------------------

def plot_coherence_curve(data, models, output_dir):
    """Line plot of % LT by horizon, with shaded zones for anchor regions."""
    # Exclude no-horizon
    horizons = [h for h in HORIZON_MONTHS_ORDER if h is not None]
    h_labels = [HORIZON_LABELS[h] for h in horizons]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Shaded zones
    zone_colors = {
        "Before\nST anchor": ("#DBEAFE", [0, 1]),     # light blue
        "Exact\nmatch": ("#E5E7EB", [2]),              # gray for 6mo
        "Between\nanchors": ("#FEF3C7", [3, 4, 5]),    # light yellow
        "Exact\nmatch ": ("#E5E7EB", [6]),             # gray for 10y (trailing space for unique key)
        "Beyond\nLT anchor": ("#DCFCE7", [7, 8]),      # light green
    }
    for label, (color, indices) in zone_colors.items():
        if indices:
            ax.axvspan(min(indices) - 0.4, max(indices) + 0.4, alpha=0.5, color=color, zorder=0)
            mid = (min(indices) + max(indices)) / 2
            ax.text(mid, 103, label, ha="center", va="bottom", fontsize=7.5,
                    color="#6B7280", style="italic")

    for mk in models:
        pcts = []
        for h in horizons:
            subset = [d for d in data if d["thm"] == h]
            lt = sum(1 for d in subset if _get_choice(d, mk) == "long_term")
            pcts.append(100 * lt / len(subset) if subset else 0)
        ax.plot(range(len(horizons)), pcts, "o-", label=MODEL_SHORT[mk],
                color=MODEL_COLORS[mk], linewidth=2, markersize=6, zorder=3)

    ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1, zorder=1)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(h_labels)
    ax.set_xlabel("Time Horizon")
    ax.set_ylabel("% Choosing Long-Term")
    ax.set_ylim(-3, 118)
    ax.set_title("Temporal Coherence: Does Choice Follow Time Horizon?")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.subplots_adjust(right=0.82)
    _save(fig, output_dir, "01_coherence_curve")


# ---------------------------------------------------------------------------
# Plot 2: Order Bias Decomposition (the critical finding)
# ---------------------------------------------------------------------------

def plot_order_bias(data, models, output_dir):
    """For each model, show % LT when ST-first vs LT-first, by horizon.
    This reveals where ~50% rates are order bias artifacts."""
    horizons = [h for h in HORIZON_MONTHS_ORDER if h is not None]
    h_labels = [HORIZON_LABELS[h] for h in horizons]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.flatten()

    for idx, mk in enumerate(models):
        ax = axes[idx]
        st_first_pcts = []
        lt_first_pcts = []
        for h in horizons:
            for order, pcts_list in [(True, st_first_pcts), (False, lt_first_pcts)]:
                subset = [d for d in data if d["thm"] == h and d["short_term_first"] == order]
                lt = sum(1 for d in subset if _get_choice(d, mk) == "long_term")
                pcts_list.append(100 * lt / len(subset) if subset else 0)

        x = np.arange(len(horizons))
        w = 0.35
        bars1 = ax.bar(x - w/2, st_first_pcts, w, label="ST presented first",
                       color=MODEL_COLORS[mk], alpha=0.55, edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + w/2, lt_first_pcts, w, label="LT presented first",
                       color=MODEL_COLORS[mk], alpha=1.0, edgecolor="white", linewidth=0.5)

        # Mark severe order bias (gap > 60pp)
        for i in range(len(horizons)):
            gap = abs(st_first_pcts[i] - lt_first_pcts[i])
            if gap > 60:
                top = max(st_first_pcts[i], lt_first_pcts[i])
                ax.plot(i, top + 6, marker="v", color="#DC2626", markersize=8, zorder=5)

        ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(h_labels, fontsize=9)
        ax.set_ylim(-3, 112)
        ax.set_title(MODEL_SHORT[mk], fontsize=12, fontweight="bold",
                     color=MODEL_COLORS[mk])
        if idx >= 2:
            ax.set_xlabel("Time Horizon")
        if idx % 2 == 0:
            ax.set_ylabel("% Choosing Long-Term")
        ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, 1),
                  handlelength=1.2, handletextpad=0.4)

    fig.suptitle("Order Bias Decomposition: ST-First vs LT-First Presentation",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output_dir, "02_order_bias_decomposition")


# ---------------------------------------------------------------------------
# Plot 3: Order Stability Heatmap
# ---------------------------------------------------------------------------

def plot_order_stability_heatmap(data, models, output_dir):
    """Heatmap of order stability (%) per model x horizon."""
    horizons = HORIZON_MONTHS_ORDER
    h_labels = [HORIZON_LABELS[h] for h in horizons]

    matrix = []
    for mk in models:
        row = []
        for h in horizons:
            subset = [d for d in data if d["thm"] == h]
            groups = defaultdict(dict)
            for d in subset:
                ls = _get_label_style(d)
                key = (d["long_term_reward"], d["context_id"], ls)
                groups[key][d["short_term_first"]] = _get_choice(d, mk)
            total = stable = 0
            for key, orders in groups.items():
                if True in orders and False in orders:
                    total += 1
                    if orders[True] == orders[False]:
                        stable += 1
            row.append(100 * stable / total if total else float("nan"))
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    # Text annotations
    for i in range(len(models)):
        for j in range(len(horizons)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val < 30 or val > 85 else "black"
            fontweight = "bold" if val < 15 else "normal"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight=fontweight)

    h_display = [("No horizon" if h is None else HORIZON_LABELS[h]) for h in horizons]
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(h_display)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_SHORT[m] for m in models])
    ax.set_xlabel("Time Horizon")
    ax.set_title("Order Stability: % Pairs with Same Choice Regardless of Order")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("Stability %")
    cbar.set_ticks([0, 25, 50, 75, 100])

    fig.tight_layout(pad=1.5)
    _save(fig, output_dir, "03_order_stability_heatmap")


# ---------------------------------------------------------------------------
# Plot 4: Instruct vs Base Comparison (Qwen3-4B)
# ---------------------------------------------------------------------------

def plot_instruct_vs_base(data, models, output_dir):
    """Side-by-side comparison of Qwen3-4B (base) vs Qwen3-4B-Instruct-2507."""
    base = "Qwen3-4B"
    inst = "Qwen3-4B-Instruct-2507"
    if base not in models or inst not in models:
        print("  Missing Qwen3-4B or Instruct-2507 -- skipping instruct_vs_base.")
        return

    horizons = [h for h in HORIZON_MONTHS_ORDER if h is not None]
    h_labels = [HORIZON_LABELS[h] for h in horizons]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, mk, title in [(axes[0], base, "Qwen3-4B (Base)"),
                           (axes[1], inst, "Qwen3-4B-Instruct-2507")]:
        st_first = []
        lt_first = []
        for h in horizons:
            for order, lst in [(True, st_first), (False, lt_first)]:
                subset = [d for d in data if d["thm"] == h and d["short_term_first"] == order]
                lt = sum(1 for d in subset if _get_choice(d, mk) == "long_term")
                lst.append(100 * lt / len(subset) if subset else 0)

        x = np.arange(len(horizons))
        ax.fill_between(x, st_first, lt_first, alpha=0.15, color=MODEL_COLORS[mk])
        ax.plot(x, st_first, "s--", label="ST presented first",
                color=MODEL_COLORS[mk], alpha=0.6, markersize=5, linewidth=1.5)
        ax.plot(x, lt_first, "o-", label="LT presented first",
                color=MODEL_COLORS[mk], alpha=1.0, markersize=5, linewidth=1.5)

        ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(h_labels)
        ax.set_xlabel("Time Horizon")
        ax.set_title(title, fontweight="bold", color=MODEL_COLORS[mk])
        ax.set_ylim(-3, 108)
        ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0, 1))

    axes[0].set_ylabel("% Choosing Long-Term")

    fig.suptitle("Instruct Fine-Tuning Effect: Smooth Gradient vs 3-Mode Step Function",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, "04_instruct_vs_base")


# ---------------------------------------------------------------------------
# Plot 5: No-Horizon Order Bias (the ~60% artifact)
# ---------------------------------------------------------------------------

def plot_no_horizon_order(data, models, output_dir):
    """No-horizon: show that ~60% LT is just order bias averaging out."""
    no_h = [d for d in data if d["thm"] is None]
    if not no_h:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(models))
    w = 0.25

    overall = []
    st_first_vals = []
    lt_first_vals = []
    for mk in models:
        lt_all = sum(1 for d in no_h if _get_choice(d, mk) == "long_term")
        overall.append(100 * lt_all / len(no_h))
        for order, lst in [(True, st_first_vals), (False, lt_first_vals)]:
            subset = [d for d in no_h if d["short_term_first"] == order]
            lt = sum(1 for d in subset if _get_choice(d, mk) == "long_term")
            lst.append(100 * lt / len(subset) if subset else 0)

    colors = [MODEL_COLORS[m] for m in models]
    short_names = [MODEL_SHORT[m] for m in models]

    # Bars: ST-first, overall, LT-first
    hatches = ["//", "", ""]
    alphas = [0.5, 0.8, 1.0]
    labels_legend = ["ST presented first", "Overall", "LT presented first"]
    for i in range(len(models)):
        for j, (vals, a, hatch) in enumerate(zip(
                [st_first_vals, overall, lt_first_vals], alphas, hatches)):
            bar = ax.bar(x[i] + (j-1)*w, vals[i], w, color=colors[i], alpha=a,
                         edgecolor="white", linewidth=0.5, hatch=hatch)

        # Value labels
        for val, xpos in [(st_first_vals[i], x[i]-w), (overall[i], x[i]), (lt_first_vals[i], x[i]+w)]:
            ax.text(xpos, val + 2, f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("% Choosing Long-Term")
    ax.set_ylim(0, 120)
    ax.set_title("No-Horizon Default Preference: Overall Rate Is an Order Bias Artifact")

    # Manual legend outside plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888888", alpha=0.5, hatch="//", label="ST presented first"),
        Patch(facecolor="#888888", alpha=0.8, label="Overall"),
        Patch(facecolor="#888888", alpha=1.0, label="LT presented first"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)

    fig.subplots_adjust(right=0.78)
    _save(fig, output_dir, "05_no_horizon_order_bias")


# ---------------------------------------------------------------------------
# Plot 6: Context Sensitivity
# ---------------------------------------------------------------------------

CONTEXT_SHORT = {
    290464886: "Base",
    2129047351: "Step-by-step",
    2015021528: "Brief justify",
    974652745: "Tradeoff emph.",
    1171817341: "LT-thinking emph.",
    288782122: "Individual+step",
    1476481225: "Personal choice",
    251547231: "Committee",
}


def plot_context_sensitivity(data, models, output_dir):
    """Dot plot of % LT per context, per model."""
    contexts = sorted(set(d["context_id"] for d in data))
    if len(contexts) <= 1:
        print("  Only one context -- skipping context plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for mk_idx, mk in enumerate(models):
        pcts = []
        for ctx in contexts:
            subset = [d for d in data if d["context_id"] == ctx]
            lt = sum(1 for d in subset if _get_choice(d, mk) == "long_term")
            pcts.append(100 * lt / len(subset))

        y_positions = np.arange(len(contexts)) + mk_idx * 0.18 - 0.27
        ax.scatter(pcts, y_positions, color=MODEL_COLORS[mk], s=60, zorder=3,
                   label=MODEL_SHORT[mk], edgecolors="white", linewidth=0.5)

        # Connect min to max with a line
        ax.plot([min(pcts), max(pcts)],
                [y_positions[pcts.index(min(pcts))], y_positions[pcts.index(max(pcts))]],
                color=MODEL_COLORS[mk], alpha=0.3, linewidth=6, solid_capstyle="round")

    ctx_labels = [CONTEXT_SHORT.get(c, str(c)) for c in contexts]
    ax.set_yticks(range(len(contexts)))
    ax.set_yticklabels(ctx_labels)
    ax.set_xlabel("% Choosing Long-Term")
    ax.set_xlim(15, 85)
    ax.set_title("Context Sensitivity: How Framing Shifts Temporal Preference")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.axvline(50, color="#9CA3AF", linestyle=":", linewidth=1)

    fig.subplots_adjust(right=0.78, left=0.18)
    _save(fig, output_dir, "06_context_sensitivity")


# ---------------------------------------------------------------------------
# Plot 7: Reward Sensitivity (No-Horizon Only) x Order
# ---------------------------------------------------------------------------

def plot_reward_sensitivity(data, models, output_dir):
    """No-horizon: reward x order interaction."""
    no_h = [d for d in data if d["thm"] is None]
    if not no_h:
        return

    rewards = sorted(set(d["long_term_reward"] for d in no_h))
    if len(rewards) <= 1:
        print("  Only one reward level -- skipping reward sensitivity plot.")
        return

    reward_labels = [f"${r/1000:.0f}K" for r in rewards]

    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models) + 0.5, 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, mk in zip(axes, models):
        st_vals = []
        lt_vals = []
        for r in rewards:
            for order, lst in [(True, st_vals), (False, lt_vals)]:
                subset = [d for d in no_h if d["long_term_reward"] == r and d["short_term_first"] == order]
                lt_count = sum(1 for d in subset if _get_choice(d, mk) == "long_term")
                lst.append(100 * lt_count / len(subset) if subset else 0)

        x = np.arange(len(rewards))
        w = 0.3
        b1 = ax.bar(x - w/2, st_vals, w, color=MODEL_COLORS[mk], alpha=0.5,
                     edgecolor="white", hatch="//", label="ST first")
        b2 = ax.bar(x + w/2, lt_vals, w, color=MODEL_COLORS[mk], alpha=1.0,
                     edgecolor="white", label="LT first")

        for i in range(len(rewards)):
            ax.text(x[i] - w/2, st_vals[i] + 2, f"{st_vals[i]:.0f}",
                    ha="center", fontsize=7.5)
            ax.text(x[i] + w/2, lt_vals[i] + 2, f"{lt_vals[i]:.0f}",
                    ha="center", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(reward_labels, fontsize=9)
        ax.set_title(MODEL_SHORT[mk], fontweight="bold", color=MODEL_COLORS[mk], fontsize=11)
        ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1)
        ax.set_ylim(0, 118)
        if ax == axes[0]:
            ax.set_ylabel("% Choosing Long-Term")
        ax.set_xlabel("LT Reward")

    # Shared legend below all subplots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888888", alpha=0.5, hatch="//", label="ST presented first"),
        Patch(facecolor="#888888", alpha=1.0, label="LT presented first"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=True)

    fig.suptitle("Reward Sensitivity (No-Horizon): Models Ignore Reward Magnitude",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    _save(fig, output_dir, "07_reward_sensitivity")


# ---------------------------------------------------------------------------
# Plot 8: Label Stability Heatmap
# ---------------------------------------------------------------------------

def plot_label_stability_heatmap(data, models, output_dir):
    """Heatmap of label stability (%) per model x horizon."""
    horizons = HORIZON_MONTHS_ORDER
    h_labels = [HORIZON_LABELS[h] for h in horizons]

    matrix = []
    for mk in models:
        row = []
        for h in horizons:
            subset = [d for d in data if d["thm"] == h]
            groups = defaultdict(dict)
            for d in subset:
                ls = _get_label_style(d)
                key = (d["long_term_reward"], d["context_id"], d["short_term_first"])
                groups[key][ls] = _get_choice(d, mk)
            total = stable = 0
            for key, styles in groups.items():
                if "ab" in styles and "xy" in styles:
                    total += 1
                    if styles["ab"] == styles["xy"]:
                        stable += 1
            row.append(100 * stable / total if total else float("nan"))
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    for i in range(len(models)):
        for j in range(len(horizons)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val < 30 or val > 85 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=text_color)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(h_labels)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_SHORT[m] for m in models])
    ax.set_xlabel("Time Horizon")
    ax.set_title("Label Stability: % Pairs with Same Choice Under Different Labels (a/b vs x/y)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Stability %")

    fig.tight_layout()
    _save(fig, output_dir, "08_label_stability_heatmap")


# ---------------------------------------------------------------------------
# Plot 9: Claude Step-Function Detail
# ---------------------------------------------------------------------------

def plot_claude_step_function(data, models, output_dir):
    """Claude's binary step function with order decomposition."""
    mk = "claude-haiku-4-5-20251001"
    if mk not in models:
        return

    horizons = HORIZON_MONTHS_ORDER  # include None
    h_labels = [HORIZON_LABELS[h] for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 5))

    st_first = []
    lt_first = []
    overall = []
    for h in horizons:
        subset = [d for d in data if d["thm"] == h]
        lt_all = sum(1 for d in subset if _get_choice(d, mk) == "long_term")
        overall.append(100 * lt_all / len(subset) if subset else 0)
        for order, lst in [(True, st_first), (False, lt_first)]:
            s = [d for d in subset if d["short_term_first"] == order]
            lt = sum(1 for d in s if _get_choice(d, mk) == "long_term")
            lst.append(100 * lt / len(s) if s else 0)

    x = np.arange(len(horizons))

    ax.fill_between(x, st_first, lt_first, alpha=0.12, color=MODEL_COLORS[mk])
    ax.plot(x, st_first, "s--", color=MODEL_COLORS[mk], alpha=0.5,
            markersize=6, linewidth=1.5, label="ST presented first")
    ax.plot(x, lt_first, "o-", color=MODEL_COLORS[mk], alpha=1.0,
            markersize=6, linewidth=1.5, label="LT presented first")
    ax.plot(x, overall, "D:", color="#6B7280", alpha=0.7,
            markersize=5, linewidth=1, label="Overall")

    # Annotate the zones - positioned to avoid data
    ax.annotate("Always picks short-term\n(0% LT, stable across orders)",
                xy=(3, 5), fontsize=8.5,
                color="#6B7280", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#D1D5DB"))
    ax.annotate("Pure order bias\n(~50% = picks first option)",
                xy=(9, 30), fontsize=8.5,
                color="#DC2626", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#FCA5A5"))

    ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels)
    ax.set_xlabel("Time Horizon")
    ax.set_ylabel("% Choosing Long-Term")
    ax.set_ylim(-5, 112)
    ax.set_title("Claude Haiku: Binary Step Function with Order Bias Beyond 10y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.subplots_adjust(right=0.78)
    _save(fig, output_dir, "09_claude_step_function")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python coherent_behavior_viz.py <output_dir>")
        print("  e.g.: python coherent_behavior_viz.py out/behavioral/investment_behave_full/")
        return 1

    output_dir = Path(sys.argv[1])
    responses_path = output_dir / "responses.json"
    if not responses_path.exists():
        print(f"Error: {responses_path} not found")
        return 1

    _apply_style()

    print(f"Loading {responses_path}")
    with open(responses_path) as f:
        data = json.load(f)

    _add_thm(data)
    models = _detect_models(data)
    print(f"Models: {[MODEL_SHORT.get(m, m) for m in models]}")
    print(f"Samples: {len(data)}")
    print()

    plot_coherence_curve(data, models, output_dir)
    plot_order_bias(data, models, output_dir)
    plot_order_stability_heatmap(data, models, output_dir)
    plot_instruct_vs_base(data, models, output_dir)
    plot_no_horizon_order(data, models, output_dir)
    plot_context_sensitivity(data, models, output_dir)
    plot_reward_sensitivity(data, models, output_dir)
    plot_label_stability_heatmap(data, models, output_dir)
    plot_claude_step_function(data, models, output_dir)

    print("\nDone! 9 plots generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
