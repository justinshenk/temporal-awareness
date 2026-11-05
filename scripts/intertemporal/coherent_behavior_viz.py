#!/usr/bin/env python
"""
Polished visualizations for behavioral coherence analysis.

Reads responses.json and generates publication-quality plots
highlighting the key findings from the analysis. Supports arbitrary
model counts via a family-aware model registry.

Usage:
    python scripts/intertemporal/coherent_behavior_viz.py out/behavioral/investment_behave_all/
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str            # key used in responses.json (e.g. "Qwen3-4B" or "claude-opus-4-7")
    short: str          # display name used in legends
    compact: str        # tight display name used in heatmap y-ticks
    family: str         # qwen2.5, qwen3, qwen3.5, claude
    size_b: float | None  # parameter count in B (None for API)
    variant: str        # base, instruct, thinking, api
    color: str          # hex color


MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec("Qwen2.5-3B-Instruct",          "Qwen2.5-3B-Inst",      "Qwen2.5-3B",       "qwen2.5", 3.0,  "instruct", "#4C72B0"),
    ModelSpec("Qwen3-0.6B",                   "Qwen3-0.6B",           "Qwen3-0.6B",       "qwen3",   0.6,  "hybrid",     "#A8DBA8"),
    ModelSpec("Qwen3-0.6B-thinking",          "Qwen3-0.6B-Think",     "Qwen3-0.6B-T",     "qwen3",   0.6,  "thinking", "#7FBF7F"),
    ModelSpec("Qwen3-1.7B",                   "Qwen3-1.7B",           "Qwen3-1.7B",       "qwen3",   1.7,  "hybrid",     "#5BA87A"),
    ModelSpec("Qwen3-1.7B-thinking",          "Qwen3-1.7B-Think",     "Qwen3-1.7B-T",     "qwen3",   1.7,  "thinking", "#3E8E5C"),
    ModelSpec("Qwen3-4B",                     "Qwen3-4B",             "Qwen3-4B",         "qwen3",   4.0,  "hybrid",     "#2D7A55"),
    ModelSpec("Qwen3-4B-thinking",            "Qwen3-4B-Think",       "Qwen3-4B-T",       "qwen3",   4.0,  "thinking", "#1B5E3F"),
    ModelSpec("Qwen3-4B-Instruct-2507",       "Qwen3-4B-Inst-2507",   "Qwen3-4B-I",       "qwen3",   4.0,  "instruct", "#FF1493"),
    ModelSpec("Qwen3-8B",                     "Qwen3-8B",             "Qwen3-8B",         "qwen3",   8.0,  "hybrid",     "#053C5E"),
    ModelSpec("Qwen3-14B",                    "Qwen3-14B",            "Qwen3-14B",        "qwen3",   14.0, "hybrid",     "#0A2540"),
    ModelSpec("Qwen3-32B-4bit",               "Qwen3-32B",            "Qwen3-32B",        "qwen3",   32.0, "hybrid",     "#000814"),
    ModelSpec("Qwen3.5-0.8B",                 "Qwen3.5-0.8B",         "Qwen3.5-0.8B",     "qwen3.5", 0.8,  "instruct", "#C4A8E8"),
    ModelSpec("Qwen3.5-2B",                   "Qwen3.5-2B",           "Qwen3.5-2B",       "qwen3.5", 2.0,  "instruct", "#8172B3"),
    ModelSpec("Qwen3.5-4B",                   "Qwen3.5-4B",           "Qwen3.5-4B",       "qwen3.5", 4.0,  "instruct", "#5E4280"),
    ModelSpec("Qwen3.5-9B",                   "Qwen3.5-9B",           "Qwen3.5-9B",       "qwen3.5", 9.0,  "instruct", "#4B2E83"),
    ModelSpec("Qwen3.5-27B-4bit",             "Qwen3.5-27B",          "Qwen3.5-27B",      "qwen3.5", 27.0, "instruct", "#3B1F66"),
    ModelSpec("Qwen3.5-35B-A3B-4bit",         "Qwen3.5-35B-A3B",      "Qwen3.5-35B",      "qwen3.5", 35.0, "instruct", "#2D1747"),
    ModelSpec("claude-haiku-4-5-20251001",    "Claude Haiku 4.5",     "Claude Haiku 4.5", "claude",   70.0,  "api",      "#FCA5A5"),
    ModelSpec("claude-sonnet-4-6",            "Claude Sonnet 4.6",    "Claude Sonnet 4.6","claude",  400.0,  "api",      "#DC2626"),
    ModelSpec("claude-opus-4-7",              "Claude Opus 4.7",      "Claude Opus 4.7",  "claude", 2500.0,  "api",      "#7F1D1D"),
    ModelSpec("gpt-5.4-nano",                 "GPT-5.4 Nano",         "GPT-5.4 Nano",     "openai",    8.0,  "api",      "#FCD34D"),
    ModelSpec("gpt-5.4-mini",                 "GPT-5.4 Mini",         "GPT-5.4 Mini",     "openai",  150.0,  "api",      "#F59E0B"),
    ModelSpec("gpt-5.4",                      "GPT-5.4",              "GPT-5.4",          "openai", 1800.0,  "api",      "#B45309"),
    ModelSpec("o3",                           "OpenAI o3",            "o3",               "openai",  800.0,  "api",      "#78350F"),
    ModelSpec("gemini-2.5-flash",             "Gemini 2.5 Flash",     "Gemini 2.5 Flash", "gemini",   40.0,  "api",      "#A78BFA"),
    ModelSpec("gemini-2.5-pro",               "Gemini 2.5 Pro",       "Gemini 2.5 Pro",   "gemini", 1200.0,  "api",      "#5B21B6"),
    ModelSpec("Llama-3.2-3B-Instruct-4bit",   "Llama-3.2-3B",         "Llama-3.2-3B",     "llama",   3.0,  "instruct", "#06B6D4"),
    ModelSpec("Mistral-7B-Instruct-v0.3-4bit","Mistral-7B",           "Mistral-7B",       "mistral", 7.0,  "instruct", "#0891B2"),
    ModelSpec("gemma-3-4b-it-4bit-DWQ",       "Gemma-3-4B",           "Gemma-3-4B",       "gemma",   4.0,  "instruct", "#22C55E"),
    ModelSpec("Phi-4-mini-instruct-4bit",     "Phi-4-mini",           "Phi-4-mini",       "phi",     3.8,  "instruct", "#EC4899"),
]

_BY_KEY: dict[str, ModelSpec] = {m.key: m for m in MODEL_REGISTRY}

FAMILY_ORDER = ["qwen2.5", "qwen3", "qwen3.5", "llama", "phi", "gemma", "mistral", "claude", "openai", "gemini"]

FAMILY_DISPLAY = {
    "qwen2.5": "Qwen2.5",
    "qwen3":   "Qwen3",
    "qwen3.5": "Qwen3.5",
    "llama":   "Llama",
    "mistral": "Mistral",
    "gemma":   "Gemma",
    "phi":     "Phi",
    "claude":  "Claude",
    "openai":  "OpenAI",
    "gemini":  "Gemini",
}

HORIZON_MONTHS_ORDER = [None, 1, 3, 6, 12, 24, 60, 120, 240, 600]
HORIZON_LABELS = {
    None: "None", 1: "1mo", 3: "3mo", 6: "6mo",
    12: "1y", 24: "2y", 60: "5y", 120: "10y", 240: "20y", 600: "50y",
}

# Zones (in months) for coherence
BEFORE_ANCHOR = {1, 3}
EXACT_SHORT = {6}
BETWEEN_ANCHOR = {12, 24, 60}       # 1y, 2y, 5y  -- the "temporal reasoning zone"
EXACT_LONG = {120}
BEYOND_ANCHOR = {240, 600}

# Zone column indices within HORIZON_MONTHS_ORDER (0-indexed)
ZONE_COLS = {
    "none":     [0],
    "before":   [1, 2],
    "exact_st": [3],
    "between":  [4, 5, 6],
    "exact_lt": [7],
    "beyond":   [8, 9],
}


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.85",
        "figure.dpi": 140,
        "savefig.dpi": 140,
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


def _get_choice(row, key: str):
    return row.get(f"{key}_choice")


def _get_label_style(row):
    labels = row["labels"]
    return "ab" if "a)" in labels[0] or "b)" in labels[0] else "xy"


def _detect_models(data) -> list[ModelSpec]:
    """Return registered ModelSpecs present in data, ordered by family+size."""
    sample = data[0]
    present = {k[: -len("_choice")] for k in sample if k.endswith("_choice")}

    specs = []
    for key in present:
        spec = _BY_KEY.get(key)
        if spec is None:
            print(f"  WARN: model key '{key}' not in MODEL_REGISTRY -- skipping")
            continue
        specs.append(spec)

    def _sort_key(s: ModelSpec):
        fam_idx = FAMILY_ORDER.index(s.family) if s.family in FAMILY_ORDER else 99
        size = s.size_b if s.size_b is not None else 0
        # Thinking > instruct > hybrid (non-thinking) at same nominal size:
        # thinking variants exercise more compute at inference and are treated
        # as larger in every sort.
        variant_bump = {"hybrid": 0.00, "instruct": 0.01, "thinking": 0.02, "api": 0.03}.get(s.variant, 0.0)
        return (size + variant_bump, fam_idx)

    return sorted(specs, key=_sort_key)


def _add_thm(data):
    for d in data:
        d["thm"] = _th_to_months(d["time_horizon"])


def _save(fig, output_dir, name):
    path = output_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def _family_models(models: list[ModelSpec], family: str) -> list[ModelSpec]:
    return [m for m in models if m.family == family]


def _family_block_rows(models: list[ModelSpec]) -> list[tuple[str, int, int]]:
    """Return list of (family, row_start, row_end_inclusive) in the model ordering."""
    blocks = []
    i = 0
    while i < len(models):
        fam = models[i].family
        j = i
        while j + 1 < len(models) and models[j + 1].family == fam:
            j += 1
        blocks.append((fam, i, j))
        i = j + 1
    return blocks


def _blocks_are_meaningful(blocks) -> bool:
    """True only when the ordering groups families contiguously.

    With global size-sort, families get scattered into mostly-singleton blocks,
    so drawing a divider between every row is noise. We gate all family-block
    decorations (dividers, sidebars) on this predicate.
    """
    return any((end - start + 1) >= 3 for _, start, end in blocks)


def _paired_responses(data, spec_key: str):
    """Return only responses that belong to a valid ST-first/LT-first pair for
    this model: both orderings present for the same (horizon, reward, context,
    label_style) key, and both produced a parseable choice.
    """
    groups = defaultdict(dict)
    for d in data:
        key = (d["thm"], d["long_term_reward"], d["context_id"], _get_label_style(d))
        groups[key][d["short_term_first"]] = d
    out = []
    for g in groups.values():
        if True in g and False in g:
            c_t = _get_choice(g[True], spec_key)
            c_f = _get_choice(g[False], spec_key)
            if c_t in ("long_term", "short_term") and c_f in ("long_term", "short_term"):
                out.append(g[True])
                out.append(g[False])
    return out


_UNSET = object()


def _lt_pct(data, spec_key: str, *, horizon=_UNSET, st_first=None,
            reward=None, context_id=None, label_style=None):
    """Return (pct LT, n) over the paired subset matching the given filters.
    All analyses are restricted to prompt pairs where both ST-first and LT-first
    orderings produced a valid choice (single denominator across heatmaps).

    horizon: pass _UNSET (default) to include all horizons; pass None to
    restrict to no-horizon responses; pass a number to restrict to that horizon.
    """
    subset = _paired_responses(data, spec_key)
    if horizon is not _UNSET:
        subset = [d for d in subset if d["thm"] == horizon]
    if st_first is not None:
        subset = [d for d in subset if d["short_term_first"] == st_first]
    if reward is not None:
        subset = [d for d in subset if d["long_term_reward"] == reward]
    if context_id is not None:
        subset = [d for d in subset if d["context_id"] == context_id]
    if label_style is not None:
        subset = [d for d in subset if _get_label_style(d) == label_style]
    if not subset:
        return 0.0, 0
    lt = sum(1 for d in subset if _get_choice(d, spec_key) == "long_term")
    return 100 * lt / len(subset), len(subset)


def _horizon_lt_pct(data, spec_key, horizons):
    """Vector of %LT across horizons, restricted to paired responses."""
    paired = _paired_responses(data, spec_key)
    out = []
    for h in horizons:
        subset = [d for d in paired if d["thm"] == h]
        if not subset:
            out.append(0.0)
            continue
        lt = sum(1 for d in subset if _get_choice(d, spec_key) == "long_term")
        out.append(100 * lt / len(subset))
    return out


def _shade_incoherence_zone(ax, cols=(4, 5, 6), alpha=0.10, color="#FCA5A5"):
    """Shade the between-anchor (1y-5y) temporal reasoning zone on x-axis columns."""
    ax.axvspan(min(cols) - 0.5, max(cols) + 0.5, alpha=alpha, color=color, zorder=0)


# Horizon-keyed rational target for %LT.
# - None / before ST anchor / reasoning zone / exact ST: rational answer is ST,
#   so target %LT = 0, and incoherence = %LT.
# - Exact LT anchor / beyond: rational answer is LT, so target %LT = 100, and
#   incoherence = 100 - %LT.
# The convention for "no horizon" is that no temporal constraint is given, so
# we treat any LT preference as "non-rational default" — incoherence = %LT.
def _rational_lt_target(h):
    """Return the rational %LT target in [0, 100] for a given horizon (months)."""
    if h is None:
        return 0.0                   # no horizon -> no LT justification
    if h < 120:                      # before LT anchor
        return 0.0
    return 100.0                     # at/beyond LT anchor


def _incoherence_from_lt(lt_pct, h):
    """Convert %LT at horizon h into an incoherence score in [0, 100]."""
    target = _rational_lt_target(h)
    return abs(lt_pct - target)


def _horizon_incoherence(data, spec_key, horizons):
    """Vector of incoherence-% across horizons, restricted to paired responses."""
    paired = _paired_responses(data, spec_key)
    out = []
    for h in horizons:
        subset = [d for d in paired if d["thm"] == h]
        if not subset:
            out.append(0.0)
            continue
        lt = 100.0 * sum(1 for d in subset if _get_choice(d, spec_key) == "long_term") / len(subset)
        out.append(_incoherence_from_lt(lt, h))
    return out


TARGET_KEY = "Qwen3-4B-Instruct-2507"


def _add_family_sidebar(ax, blocks, side="left", pad_frac=0.22):
    """No-op: global size-sort fragments families into scattered rows, so family
    sidebar labels are visual noise rather than useful grouping information.
    The y-tick labels already identify each model by name.
    """
    return


def _highlight_target_row(ax, models, n_cols, *, color="#FF1493"):
    """Draw a thick outline rectangle around the Qwen3-4B-Instruct-2507 row."""
    idx = next((i for i, m in enumerate(models) if m.key == TARGET_KEY), None)
    if idx is None:
        return
    from matplotlib.patches import Rectangle
    rect = Rectangle((-0.5, idx - 0.5), n_cols, 1.0,
                     linewidth=2.2, edgecolor=color, facecolor="none",
                     zorder=7)
    ax.add_patch(rect)


def _apply_target_row_styling(ax, models, n_cols, *, color="#FF1493"):
    """Bold the target's y-tick label and draw the outline rectangle."""
    idx = next((i for i, m in enumerate(models) if m.key == TARGET_KEY), None)
    if idx is None:
        return
    labels = ax.get_yticklabels()
    if idx < len(labels):
        labels[idx].set_fontweight("bold")
        labels[idx].set_color(color)
    _highlight_target_row(ax, models, n_cols, color=color)


def _target_yticklabel(m: ModelSpec) -> str:
    """Render model compact name with a star prefix for the target model."""
    return f"★ {m.compact}" if m.key == TARGET_KEY else m.compact


# ---------------------------------------------------------------------------
# Plot 1: Coherence Curve  (family small-multiples + all-model envelope)
# ---------------------------------------------------------------------------

def _variant_linestyle(v: str):
    return {"hybrid": "--", "thinking": ":", "instruct": "-", "api": "-"}.get(v, "-")


def _panel_families(models):
    """Group small open-source families into a combined panel, keep big ones separate."""
    big = ["qwen3", "qwen3.5", "claude", "openai", "gemini"]
    groups = []
    for fam in big:
        fam_models = _family_models(models, fam)
        if fam_models:
            groups.append((FAMILY_DISPLAY[fam], fam_models))
    other = [m for m in models if m.family not in big]
    if other:
        groups.append(("Other open", other))
    return groups


def plot_coherence_curve(data, models, output_dir):
    """Per-horizon %LT across all 30 models, in per-family panels.

    This is the raw preference curve, NOT a coherence metric. Coherence is
    defined only in the temporal reasoning zone (1-5y, shaded red), where
    picking ST is the rational choice. Use plot 15 for the coherence score
    restricted to that zone.
    """
    horizons = [h for h in HORIZON_MONTHS_ORDER if h is not None]
    h_labels = [HORIZON_LABELS[h] for h in horizons]

    groups = _panel_families(models)
    n_panels = len(groups)
    rows, cols = _grid(n_panels, max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 3.8 * rows),
                              sharey=True, sharex=True, squeeze=False)
    axes_flat = axes.flatten()

    all_curves = np.array([_horizon_lt_pct(data, m.key, horizons) for m in models])
    env_lo = np.percentile(all_curves, 10, axis=0)
    env_hi = np.percentile(all_curves, 90, axis=0)
    env_med = np.median(all_curves, axis=0)

    between_idx = [3, 4, 5]  # 1y, 2y, 5y within 9-column horizon axis

    for idx, (title, fam_models) in enumerate(groups):
        ax = axes_flat[idx]
        x = np.arange(len(horizons))
        ax.fill_between(x, env_lo, env_hi, color="#D1D5DB", alpha=0.35, zorder=1,
                         label="All models P10-P90")
        ax.plot(x, env_med, color="#6B7280", linewidth=1.0, linestyle=":",
                 alpha=0.9, zorder=1, label="All-models median")

        ax.axvspan(min(between_idx) - 0.5, max(between_idx) + 0.5,
                   color="#FEE2E2", alpha=0.55, zorder=0)

        for spec in fam_models:
            pcts = _horizon_lt_pct(data, spec.key, horizons)
            ls = _variant_linestyle(spec.variant)
            is_target = spec.key == TARGET_KEY
            ax.plot(x, pcts, marker="o", linestyle=ls,
                    label=(f"\u2605 {spec.short}" if is_target else spec.short),
                    color=spec.color,
                    linewidth=(3.2 if is_target else 1.8),
                    markersize=(7 if is_target else 4),
                    markeredgecolor=("white" if is_target else None),
                    markeredgewidth=(1.4 if is_target else 0),
                    zorder=(6 if is_target else 3))

        ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(h_labels, fontsize=9)
        ax.set_ylim(-4, 118)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ncol = 2 if len(fam_models) >= 6 else 1
        leg = ax.legend(loc="upper left", fontsize=6.5, ncol=ncol, frameon=True,
                        handlelength=1.4, borderpad=0.3, labelspacing=0.18,
                        columnspacing=0.6, framealpha=0.92,
                        bbox_to_anchor=(0.005, 0.93))
        for t in leg.get_texts():
            if t.get_text().startswith("\u2605"):
                t.set_fontweight("bold")
                t.set_color("#FF1493")
        ax.text(np.mean(between_idx), 115, "temporal reasoning zone (1-5y)",
                ha="center", va="top", fontsize=7.5, color="#B91C1C",
                style="italic")

    for r in range(rows):
        axes[r, 0].set_ylabel("% Long-Term choice")
    for c in range(cols):
        axes[rows - 1, c].set_xlabel("Time Horizon")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle("Per-Horizon %LT: Does the Model Track the Stated Deadline?   (raw preference curve)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, -0.012,
             "Shaded red band = temporal reasoning zone (1-5y) where only the 6-month option can deliver and picking ST is rational. "
             "Coherence is defined only in this zone; see plot 15 for the coherence score. "
             "Linestyle: solid = instruct/API, dashed = hybrid-thinking (non-thinking run), dotted = thinking-only.",
             ha="center", fontsize=8.5, color="#374151")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, output_dir, "01_coherence_curve")


# ---------------------------------------------------------------------------
# Plot 2: Order Bias Gap Heatmap
# ---------------------------------------------------------------------------

def plot_order_bias(data, models, output_dir):
    horizons = HORIZON_MONTHS_ORDER
    h_display = [("None" if h is None else HORIZON_LABELS[h]) for h in horizons]

    matrix = np.full((len(models), len(horizons)), np.nan)
    for i, spec in enumerate(models):
        for j, h in enumerate(horizons):
            st_first, _ = _lt_pct(data, spec.key, horizon=h, st_first=True)
            lt_first, _ = _lt_pct(data, spec.key, horizon=h, st_first=False)
            matrix[i, j] = lt_first - st_first  # signed gap (LT-first minus ST-first)

    blocks = _family_block_rows(models)

    # Diverging colormap centered at 0.
    fig, ax = plt.subplots(figsize=(12, 0.38 * len(models) + 2.5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")

    for i in range(len(models)):
        for j in range(len(horizons)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            if abs(val) < 15:
                continue  # don't clutter small gaps
            text_color = "white" if abs(val) > 55 else "black"
            ax.text(j, i, f"{val:+.0f}", ha="center", va="center",
                    fontsize=7.5, color=text_color,
                    fontweight=("bold" if abs(val) > 60 else "normal"))

    # Family dividers and group labels (labels rendered via right-side axis).
    if _blocks_are_meaningful(blocks):
        for _, _, end in blocks[:-1]:
            ax.axhline(end + 0.5, color="white", linewidth=2.5, zorder=5)
            ax.axhline(end + 0.5, color="#374151", linewidth=0.6, zorder=6)
    _add_family_sidebar(ax, blocks, side="right")

    # Zone dividers on x-axis.
    zone_boundaries = [0.5, 2.5, 3.5, 6.5, 7.5]  # after None, before, exactST, between, exactLT
    for b in zone_boundaries:
        ax.axvline(b, color="white", linewidth=1.8, zorder=4)
        ax.axvline(b, color="#6B7280", linewidth=0.4, zorder=5, alpha=0.5)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(h_display, fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_target_yticklabel(m) for m in models], fontsize=8.5)
    _apply_target_row_styling(ax, models, len(horizons))
    ax.set_xlabel("Time Horizon")
    ax.set_title("Position Bias: Does the Model Favor Whichever Option Is Listed First?\n"
                 "Red = picks first-listed option (primacy); Blue = picks last-listed (recency); 0 = no position effect",
                 fontsize=12, loc="center")

    # Zone labels on top.
    zone_top_labels = [
        ("No horizon", 0),
        ("Before\nST anchor", 1.5),
        ("Exact\nST", 3),
        ("Temporal reasoning\nzone (1-5y)", 5),
        ("Exact\nLT", 7),
        ("Beyond\nLT anchor", 8.5),
    ]
    for lab, xp in zone_top_labels:
        ax.text(xp, -1.5, lab, ha="center", va="bottom",
                fontsize=8.5, color="#B91C1C" if "reasoning" in lab else "#374151",
                fontweight=("bold" if "reasoning" in lab else "normal"),
                style=("italic" if "reasoning" in lab else "normal"))

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("LT-first %LT  -  ST-first %LT")
    cbar.set_ticks([-100, -50, 0, 50, 100])

    fig.tight_layout()
    _save(fig, output_dir, "02_order_bias_decomposition")


# ---------------------------------------------------------------------------
# Plot 3: Order Stability Heatmap
# ---------------------------------------------------------------------------

def _order_stability_matrix(data, models):
    horizons = HORIZON_MONTHS_ORDER
    matrix = []
    for spec in models:
        paired = _paired_responses(data, spec.key)
        row = []
        for h in horizons:
            subset = [d for d in paired if d["thm"] == h]
            groups = defaultdict(dict)
            for d in subset:
                ls = _get_label_style(d)
                key = (d["long_term_reward"], d["context_id"], ls)
                groups[key][d["short_term_first"]] = _get_choice(d, spec.key)
            total = stable = 0
            for k, orders in groups.items():
                if True in orders and False in orders:
                    total += 1
                    if orders[True] == orders[False]:
                        stable += 1
            row.append(100 * stable / total if total else float("nan"))
        matrix.append(row)
    return np.array(matrix)


def _draw_heatmap(ax, matrix, *, row_labels, col_labels, vmin, vmax, cmap,
                   annotate_fn, blocks, zone_boundaries, title):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text, color, weight = annotate_fn(val)
            if text:
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=8, color=color, fontweight=weight)

    if _blocks_are_meaningful(blocks):
        for _, _, end in blocks[:-1]:
            ax.axhline(end + 0.5, color="white", linewidth=2.5, zorder=5)
            ax.axhline(end + 0.5, color="#374151", linewidth=0.6, zorder=6)
    _add_family_sidebar(ax, blocks, side="right")

    for b in zone_boundaries:
        ax.axvline(b, color="white", linewidth=1.8, zorder=4)
        ax.axvline(b, color="#6B7280", linewidth=0.4, zorder=5, alpha=0.5)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_title(title, fontsize=12, loc="center")
    return im


def plot_order_stability_heatmap(data, models, output_dir):
    horizons = HORIZON_MONTHS_ORDER
    h_display = [("None" if h is None else HORIZON_LABELS[h]) for h in horizons]
    matrix = _order_stability_matrix(data, models)

    def annotate(v):
        if v < 15:
            return f"{v:.0f}", "white", "bold"
        if v < 50:
            return f"{v:.0f}", "white", "normal"
        if v >= 95:
            return "", "black", "normal"
        return f"{v:.0f}", "black", "normal"

    blocks = _family_block_rows(models)
    zone_boundaries = [0.5, 2.5, 3.5, 6.5, 7.5]

    fig, ax = plt.subplots(figsize=(12, 0.42 * len(models) + 2.5))
    im = _draw_heatmap(
        ax, matrix,
        row_labels=[_target_yticklabel(m) for m in models],
        col_labels=h_display,
        vmin=0, vmax=100, cmap="RdYlGn",
        annotate_fn=annotate, blocks=blocks, zone_boundaries=zone_boundaries,
        title="Does the Same Choice Survive When ST/LT Are Swapped?\n"
              "(% of prompt pairs with identical choice under both presentation orders; blank cells are 95-100%.)",
    )
    _apply_target_row_styling(ax, models, len(h_display))

    zone_top_labels = [
        ("No horizon", 0),
        ("Before ST", 1.5),
        ("Exact ST", 3),
        ("Temporal reasoning\nzone (1-5y)", 5),
        ("Exact LT", 7),
        ("Beyond LT", 8.5),
    ]
    for lab, xp in zone_top_labels:
        color = "#B91C1C" if "reasoning" in lab else "#374151"
        ax.text(xp, -1.5, lab, ha="center", va="bottom", fontsize=8.5,
                color=color,
                fontweight=("bold" if "reasoning" in lab else "normal"),
                style=("italic" if "reasoning" in lab else "normal"))

    ax.set_xlabel("Time Horizon")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Order stability %")
    cbar.set_ticks([0, 25, 50, 75, 100])

    fig.tight_layout()
    _save(fig, output_dir, "03_order_stability_heatmap")


# ---------------------------------------------------------------------------
# Plot 4: Post-Training Recipe Effect (base / thinking / instruct within family+size)
# ---------------------------------------------------------------------------

def plot_instruct_vs_base(data, models, output_dir):
    # Group by (family, size) and accept any group with >=2 variants.
    groups: dict[tuple[str, float], dict[str, ModelSpec]] = defaultdict(dict)
    for spec in models:
        if spec.size_b is None or spec.variant == "api":
            continue
        groups[(spec.family, spec.size_b)][spec.variant] = spec

    valid = [(k, v) for k, v in groups.items() if len(v) >= 2]
    # Sort by size ascending.
    valid.sort(key=lambda kv: (FAMILY_ORDER.index(kv[0][0]) if kv[0][0] in FAMILY_ORDER else 99,
                                kv[0][1]))

    if not valid:
        print("  No recipe comparisons -- skipping instruct_vs_base.")
        return

    horizons = [h for h in HORIZON_MONTHS_ORDER if h is not None]
    h_labels = [HORIZON_LABELS[h] for h in horizons]

    variant_order = ["hybrid", "thinking", "instruct"]
    variant_title = {
        "hybrid":   "Hybrid (run in non-thinking mode)",
        "thinking": "Thinking-only specialist",
        "instruct": "Non-thinking-only specialist",
    }
    # Use a darker tint of the family color for titles so they stay legible
    # even when the line color is pale.
    def _dark(hex_color: str, factor: float = 0.55) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r, g, b = int(r * factor), int(g * factor), int(b * factor)
        return f"#{r:02X}{g:02X}{b:02X}"
    max_cols = max(len(v) for _, v in valid)
    n_rows = len(valid)

    fig, axes = plt.subplots(n_rows, max_cols,
                              figsize=(4.6 * max_cols, 3.6 * n_rows),
                              sharey=True, sharex=True, squeeze=False)

    for row_idx, ((family, size), vmap) in enumerate(valid):
        present_variants = [v for v in variant_order if v in vmap]
        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(present_variants):
                ax.axis("off")
                continue
            variant = present_variants[col_idx]
            spec = vmap[variant]

            st_first, lt_first = [], []
            for h in horizons:
                sv, _ = _lt_pct(data, spec.key, horizon=h, st_first=True)
                lv, _ = _lt_pct(data, spec.key, horizon=h, st_first=False)
                st_first.append(sv); lt_first.append(lv)

            x = np.arange(len(horizons))
            # Shade temporal reasoning zone.
            _shade_incoherence_zone(ax, cols=[3, 4, 5], alpha=0.12)

            gap = np.abs(np.array(lt_first) - np.array(st_first))
            ax.fill_between(x, st_first, lt_first, alpha=0.25,
                             color=spec.color, hatch="//", edgecolor=spec.color,
                             linewidth=0, zorder=2)
            ax.plot(x, st_first, "s--", color=spec.color, alpha=0.55,
                    markersize=5, linewidth=1.6, label="ST first", zorder=3)
            ax.plot(x, lt_first, "o-", color=spec.color, alpha=1.0,
                    markersize=5, linewidth=1.8, label="LT first", zorder=3)

            # Max gap callout (only in the temporal reasoning zone).
            between_slice = slice(3, 6)
            bmax = gap[between_slice].argmax()
            bval = gap[between_slice][bmax]
            if bval > 20:
                bx = 3 + bmax
                by = max(st_first[bx], lt_first[bx]) + 4
                ax.annotate(f"Δ={bval:.0f} at {h_labels[bx]}",
                            xy=(bx, by), ha="center", fontsize=7.5,
                            color="#B91C1C",
                            bbox=dict(facecolor="white", alpha=0.85,
                                      edgecolor="#B91C1C", boxstyle="round,pad=0.2"))

            ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1)
            ax.set_xticks(x); ax.set_xticklabels(h_labels, fontsize=8)
            ax.set_ylim(-4, 118)
            size_str = f"{size:g}B"
            title = f"{FAMILY_DISPLAY[family]} {size_str} - {variant_title[variant]}"
            is_target = (spec.key == TARGET_KEY)
            if is_target:
                title = f"★ {title} (target)"
            ax.set_title(title, fontsize=10.5, color=_dark(spec.color),
                         fontweight="bold")
            if is_target:
                # Draw a heavy border around target panel.
                for spine in ax.spines.values():
                    spine.set_edgecolor(spec.color)
                    spine.set_linewidth(2.4)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left", fontsize=8, handlelength=1.4)
            if col_idx == 0:
                ax.set_ylabel("% LT")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time Horizon")

    fig.suptitle("Post-Training Recipe Effect: Same Size, Different Training",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.text(0.5, -0.01,
             "Hatched band = ST-first vs LT-first gap (wider = stronger order bias). "
             "Pink shade = temporal reasoning zone (1-5y, neither anchor delivers).",
             ha="center", fontsize=8.5, color="#374151")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, output_dir, "04_instruct_vs_base")


# ---------------------------------------------------------------------------
# Plot 5: No-Horizon Default Preference (family grid with range bars)
# ---------------------------------------------------------------------------

def plot_no_horizon_order(data, models, output_dir):
    no_h = [d for d in data if d["thm"] is None]
    if not no_h:
        return

    families_present = [f for f in FAMILY_ORDER if _family_models(models, f)]
    n_fam = len(families_present)
    rows, cols = _grid(n_fam, max_cols=5)

    width_ratios = []
    for r in range(rows):
        row_fams = families_present[r * cols : (r + 1) * cols]
        while len(row_fams) < cols:
            row_fams.append(None)
        row_wr = [max(1, len(_family_models(models, f))) if f else 1 for f in row_fams]
        width_ratios.append(row_wr)

    # Use a uniform gridspec with max width ratios.
    combined_wr = [max(wr[c] for wr in width_ratios) for c in range(cols)]
    fig = plt.figure(figsize=(3.4 * sum(combined_wr) / max(combined_wr) + 2, 3.5 * rows + 1.2))
    gs = fig.add_gridspec(rows, cols, width_ratios=combined_wr, hspace=0.55, wspace=0.35)

    for idx, family in enumerate(families_present):
        r, c = divmod(idx, cols)
        ax = fig.add_subplot(gs[r, c])
        fam_models = _family_models(models, family)
        x = np.arange(len(fam_models))

        st_vals, lt_vals, overall = [], [], []
        for spec in fam_models:
            ov, _ = _lt_pct(no_h, spec.key)
            sv, _ = _lt_pct(no_h, spec.key, st_first=True)
            lv, _ = _lt_pct(no_h, spec.key, st_first=False)
            overall.append(ov); st_vals.append(sv); lt_vals.append(lv)

        for i, spec in enumerate(fam_models):
            low, high = min(st_vals[i], lt_vals[i]), max(st_vals[i], lt_vals[i])
            is_target = spec.key == TARGET_KEY
            # Red background if catastrophic gap.
            if abs(st_vals[i] - lt_vals[i]) > 40:
                ax.axvspan(i - 0.45, i + 0.45, color="#FEE2E2", alpha=0.6, zorder=0)
            # Target model highlight: yellow band (distinct from pink severe-bias
            # shading so the two don't visually conflate).
            if is_target:
                ax.axvspan(i - 0.45, i + 0.45, color="#FEF3C7", alpha=0.9, zorder=0)
            # Range bar.
            ax.plot([i, i], [low, high], color=spec.color,
                    linewidth=(8 if is_target else 6),
                    alpha=(0.85 if is_target else 0.45),
                    solid_capstyle="round", zorder=2)
            # ST / LT endpoints.
            ax.scatter(i, st_vals[i], marker="s", s=50, color=spec.color,
                        alpha=0.6, edgecolors="white", linewidth=0.8, zorder=3,
                        label="ST first" if i == 0 else None)
            ax.scatter(i, lt_vals[i], marker="o", s=50, color=spec.color,
                        alpha=1.0, edgecolors="white", linewidth=0.8, zorder=3,
                        label="LT first" if i == 0 else None)
            # Overall diamond.
            ax.scatter(i, overall[i], marker="D", s=85, color="#111827",
                        edgecolors="white", linewidth=1.2, zorder=4,
                        label="Overall" if i == 0 else None)
            # Overall value label.
            ax.annotate(f"{overall[i]:.0f}", xy=(i, overall[i] + 5),
                        ha="center", fontsize=7.5, color="#111827")

        ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([_target_yticklabel(m) for m in fam_models],
                           fontsize=7.5, rotation=35, ha="right")
        for lbl, spec in zip(ax.get_xticklabels(), fam_models):
            if spec.key == TARGET_KEY:
                lbl.set_fontweight("bold")
                lbl.set_color("#B45309")
        ax.set_title(FAMILY_DISPLAY[family], fontsize=11, fontweight="bold")
        ax.set_ylim(-4, 125)
        if c == 0:
            ax.set_ylabel("% Long-Term")

    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="#555555", linestyle="",
                   markersize=8, alpha=0.6, label="ST first"),
        plt.Line2D([0], [0], marker="o", color="#555555", linestyle="",
                   markersize=8, alpha=1.0, label="LT first"),
        plt.Line2D([0], [0], marker="D", color="#111827", linestyle="",
                   markersize=9, label="Overall"),
        Patch(facecolor="#FEE2E2", edgecolor="#FCA5A5", label="Severe order bias (|Δ|>40)"),
        Patch(facecolor="#FEF3C7", edgecolor="#FDE68A", label="Target model (Qwen3-4B-I)"),
    ]
    # Reserve the bottom band explicitly via subplots_adjust so tight_layout does
    # not recompute and collapse the gap. Subplots end at y=0.28 of the figure;
    # rotated tick labels extend below; legend sits in the bottom ~6% with a clean gap.
    fig.suptitle(f"What Does Each Model Pick When No Time Horizon Is Given? (n={len(no_h)} prompts)\n"
                 f"Bars span ST-first vs LT-first order; black diamond = overall %LT",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.22,
                        hspace=0.85, wspace=0.30)
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=True)
    _save(fig, output_dir, "05_no_horizon_order_bias")


# ---------------------------------------------------------------------------
# Plot 6: Context Sensitivity (heatmap + distribution strip)
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

CONTEXT_COLORS = {
    290464886:  "#111827",
    2129047351: "#DC2626",
    2015021528: "#F97316",
    974652745:  "#CA8A04",
    1171817341: "#16A34A",
    288782122:  "#0891B2",
    1476481225: "#2563EB",
    251547231:  "#7C3AED",
}


def plot_context_sensitivity(data, models, output_dir):
    contexts = sorted(set(d["context_id"] for d in data))
    if len(contexts) <= 1:
        print("  Only one context -- skipping context plot.")
        return

    ctx_labels = [CONTEXT_SHORT.get(c, str(c)) for c in contexts]

    matrix = np.full((len(models), len(contexts)), np.nan)
    for i, spec in enumerate(models):
        for j, ctx in enumerate(contexts):
            v, _ = _lt_pct(data, spec.key, context_id=ctx)
            matrix[i, j] = v

    blocks = _family_block_rows(models)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15.5, 0.42 * len(models) + 3),
        gridspec_kw={"width_ratios": [3.2, 1.1], "wspace": 0.28},
    )

    # Left: heatmap.
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=15, vmax=85, aspect="auto")
    for i in range(len(models)):
        for j in range(len(contexts)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val < 25 or val > 75 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7.5, color=text_color)

    if _blocks_are_meaningful(blocks):
        for _, _, end in blocks[:-1]:
            ax.axhline(end + 0.5, color="white", linewidth=2.5, zorder=5)
            ax.axhline(end + 0.5, color="#374151", linewidth=0.6, zorder=6)

    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(ctx_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_target_yticklabel(m) for m in models], fontsize=8.5)
    _apply_target_row_styling(ax, models, len(contexts))
    ax.set_title("%LT by Model x Scenario Framing", fontsize=12)
    _add_family_sidebar(ax, blocks, side="right")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label("% LT")

    # Right: per-model range (max - min) as horizontal bars.
    ranges = matrix.max(axis=1) - matrix.min(axis=1)
    y = np.arange(len(models))
    colors = [m.color for m in models]
    ax2.barh(y, ranges, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    if _blocks_are_meaningful(blocks):
        for _, _, end in blocks[:-1]:
            ax2.axhline(end + 0.5, color="#374151", linewidth=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax.invert_yaxis()
    ax2.set_xlabel("max - min %LT across contexts")
    ax2.set_title("Framing Spread\n(max - min %LT)", fontsize=11)
    ax2.set_xlim(0, max(50, ranges.max() * 1.1))

    for yi, r in enumerate(ranges):
        ax2.text(r + 0.8, yi, f"{r:.0f}", ha="left", va="center", fontsize=7.5)

    fig.suptitle("How Much Does Scenario Framing Shift a Model's %LT?",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, output_dir, "06_context_sensitivity")


# ---------------------------------------------------------------------------
# Plot 7: Reward Sensitivity (overlay slopes + sorted delta bar)
# ---------------------------------------------------------------------------

def plot_reward_sensitivity(data, models, output_dir):
    no_h = [d for d in data if d["thm"] is None]
    if not no_h:
        return

    rewards = sorted(set(d["long_term_reward"] for d in no_h))
    if len(rewards) <= 1:
        print("  Only one reward level -- skipping reward sensitivity plot.")
        return

    reward_labels = [f"${r/1000:.0f}K" for r in rewards]

    curves = {}
    for spec in models:
        vals = []
        for r in rewards:
            v, _ = _lt_pct(no_h, spec.key, reward=r)
            vals.append(v)
        curves[spec.key] = vals

    # Compute per-model max|delta| across reward levels.
    deltas = {k: max(v) - min(v) for k, v in curves.items()}

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13, 10),
        gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.26},
    )

    x = np.arange(len(rewards))
    target_spec = next((m for m in models if m.key == TARGET_KEY), None)
    for spec in models:
        if spec.key == TARGET_KEY:
            continue
        ax.plot(x, curves[spec.key], marker="o", color=spec.color,
                alpha=0.55, linewidth=1.1, markersize=3.5,
                linestyle=_variant_linestyle(spec.variant))
    all_vals = np.array([curves[m.key] for m in models])
    mean_line = all_vals.mean(axis=0)
    q25, q75 = np.percentile(all_vals, [25, 75], axis=0)
    ax.fill_between(x, q25, q75, color="#111827", alpha=0.12, zorder=5,
                     label="IQR across models")
    ax.plot(x, mean_line, color="#111827", linewidth=2.5, zorder=6, label="Mean")
    if target_spec is not None:
        ax.plot(x, curves[target_spec.key], marker="o", color=target_spec.color,
                alpha=1.0, linewidth=3.2, markersize=8,
                markeredgecolor="white", markeredgewidth=1.4,
                linestyle=_variant_linestyle(target_spec.variant),
                zorder=7, label=f"\u2605 {target_spec.short}")

    ax.set_xticks(x); ax.set_xticklabels(reward_labels, fontsize=11)
    ax.set_xlabel("Long-term reward (short-term = \\$20K fixed)")
    ax.set_ylabel("% Long-Term (no-horizon prompts)")
    ax.set_ylim(-4, 118)
    ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1)
    median_delta = np.median(list(deltas.values()))
    ax.set_title(
        f"%LT vs Long-Term Reward Size (median delta = {median_delta:.1f} pp)",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9)

    # Right: reward-delta bars, family-grouped and size-ordered.
    specs_sorted = list(models)
    y = np.arange(len(specs_sorted))
    colors = [m.color for m in specs_sorted]
    values = [deltas[m.key] for m in specs_sorted]
    edgecolors = ["#FF1493" if m.key == TARGET_KEY else "white" for m in specs_sorted]
    linewidths = [2.4 if m.key == TARGET_KEY else 0.5 for m in specs_sorted]
    ax2.barh(y, values, color=colors, alpha=0.9, edgecolor=edgecolors, linewidth=linewidths)
    ax2.set_yticks(y)
    ax2.set_yticklabels([_target_yticklabel(m) for m in specs_sorted], fontsize=8)
    for lbl, spec in zip(ax2.get_yticklabels(), specs_sorted):
        if spec.key == TARGET_KEY:
            lbl.set_fontweight("bold")
            lbl.set_color("#FF1493")
    _blocks_p7 = _family_block_rows(specs_sorted)
    if _blocks_are_meaningful(_blocks_p7):
        for _, _, end in _blocks_p7[:-1]:
            ax2.axhline(end + 0.5, color="#374151", linewidth=0.6)
    ax2.invert_yaxis()
    ax2.set_xlabel("max - min %LT across rewards")
    ax2.set_xlim(0, max(40, max(values) * 1.1))
    ax2.set_title("Reward Delta per Model\n(max - min %LT)", fontsize=11)
    for yi, v in enumerate(values):
        ax2.text(v + 0.6, yi, f"{v:.0f}", ha="left", va="center", fontsize=7.5)

    fig.suptitle("Do Models Respond to Long-Term Reward Size? ($100K vs $300K vs $500K)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, output_dir, "07_reward_sensitivity")


# ---------------------------------------------------------------------------
# Plot 8: Label Stability Heatmap (compact sanity plot)
# ---------------------------------------------------------------------------

def plot_label_stability_heatmap(data, models, output_dir):
    horizons = HORIZON_MONTHS_ORDER
    h_labels = [("None" if h is None else HORIZON_LABELS[h]) for h in horizons]

    matrix = []
    for spec in models:
        row = []
        for h in horizons:
            subset = [d for d in data if d["thm"] == h]
            groups = defaultdict(dict)
            for d in subset:
                ls = _get_label_style(d)
                key = (d["long_term_reward"], d["context_id"], d["short_term_first"])
                groups[key][ls] = _get_choice(d, spec.key)
            total = stable = 0
            for k, styles in groups.items():
                if "ab" in styles and "xy" in styles:
                    total += 1
                    if styles["ab"] == styles["xy"]:
                        stable += 1
            row.append(100 * stable / total if total else float("nan"))
        matrix.append(row)
    matrix = np.array(matrix)

    blocks = _family_block_rows(models)

    def annotate(v):
        if v < 55:
            return f"{v:.0f}", "white", "bold"
        if v < 75:
            return f"{v:.0f}", "black", "bold"
        return f"{v:.0f}", "black", "normal"

    # RdYlGn diverging keeps low-stability cells visible and matches plot 3.
    cmap = plt.cm.RdYlGn
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(models) + 2.2))
    im = _draw_heatmap(
        ax, matrix,
        row_labels=[_target_yticklabel(m) for m in models],
        col_labels=h_labels,
        vmin=40, vmax=100, cmap=cmap,
        annotate_fn=annotate, blocks=blocks, zone_boundaries=[0.5, 2.5, 3.5, 6.5, 7.5],
        title="Does the Same Choice Survive When Labels Change from a/b to x/y?\n"
              "(% of prompt pairs with identical choice across label wording; sanity check for label bias.)",
    )
    _apply_target_row_styling(ax, models, len(h_labels))
    ax.set_xlabel("Time Horizon")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Label stability %")
    fig.tight_layout()
    _save(fig, output_dir, "08_label_stability_heatmap")


# ---------------------------------------------------------------------------
# Plot 9: Claude family step-function with annotated zones
# ---------------------------------------------------------------------------

def plot_claude_step_function(data, models, output_dir):
    claudes = _family_models(models, "claude")
    if not claudes:
        print("  No Claude models -- skipping claude_step_function.")
        return

    horizons = [h for h in HORIZON_MONTHS_ORDER if h is not None]
    h_labels = [HORIZON_LABELS[h] for h in horizons]

    n = len(claudes)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5.2), sharey=True, squeeze=False)
    axes = axes[0]

    # Zone columns (within 9-length horizon axis): before (0-1), exactST (2),
    # between (3-5), exactLT (6), beyond (7-8).
    zone_shading = [
        ("#E5E7EB", 0, 2, "always-ST"),      # before + exactST
        ("#FEF3C7", 3, 6, "step region"),    # between + exactLT
        ("#FEE2E2", 7, 8, "order-bias\ntail"),
    ]

    for ax, spec in zip(axes, claudes):
        for color, lo, hi, label in zone_shading:
            ax.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=0.55, zorder=0)
            ax.text((lo + hi) / 2, 107, label, ha="center", va="bottom",
                     fontsize=8, color="#6B7280", style="italic")

        st_first, lt_first = [], []
        for h in horizons:
            sv, _ = _lt_pct(data, spec.key, horizon=h, st_first=True)
            lv, _ = _lt_pct(data, spec.key, horizon=h, st_first=False)
            st_first.append(sv); lt_first.append(lv)

        x = np.arange(len(horizons))
        ax.fill_between(x, st_first, lt_first, alpha=0.18, color=spec.color, zorder=2)
        ax.plot(x, st_first, "s--", color=spec.color, alpha=0.75,
                markersize=6, linewidth=2.2, markerfacecolor="white",
                label="ST first", zorder=3)
        ax.plot(x, lt_first, "o-", color=spec.color, alpha=1.0,
                markersize=6, linewidth=2.5, label="LT first", zorder=4)

        ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(h_labels, fontsize=9, rotation=35, ha="right")
        ax.set_xlabel("Time Horizon")
        ax.set_ylim(-6, 118)
        ax.set_title(spec.short, fontweight="bold", color=spec.color, fontsize=12)
        ax.legend(loc="center left", fontsize=9)

    axes[0].set_ylabel("% Choosing Long-Term")
    fig.suptitle("Claude Family: Flat-Zero %LT Until the 10y Threshold, Then Step to 100%",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, output_dir, "09_claude_step_function")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _grid(n: int, max_cols: int = 4) -> tuple[int, int]:
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    return rows, cols


# ---------------------------------------------------------------------------
# Plot 10: Rule heuristic match
#   For incoherent models we want to know: does the model just follow a
#   simple rule (e.g. "always pick first option", "always pick larger
#   reward")? Per model, compute the fraction of choices that match each
#   candidate rule. A near-100% bar identifies a hard-coded heuristic.
# ---------------------------------------------------------------------------


def _rule_predicts_long_term(d, rule: str) -> bool | None:
    """Return True/False for whether the rule predicts long-term, or None
    if the rule doesn't apply to this row."""
    if rule == "first":
        # "Always pick first listed option"
        # short_term_first=True → first is short_term → predicts SHORT
        return not d["short_term_first"]
    if rule == "last":
        return d["short_term_first"]
    if rule == "label_a":
        # "Always pick option labeled 'a)' (or the alphabetically-first label)"
        labels = d.get("labels") or []
        if not labels:
            return None
        # First-listed option is at labels[0]; map to short/long via short_term_first
        # If labels[0] == "a)" the model would pick the FIRST listed option
        first_is_alpha = labels[0] in ("a)", "x)")  # x) for xy variant
        # If first_is_alpha, "label_a" rule picks first listed → predicts (not short_term_first)
        if first_is_alpha:
            return not d["short_term_first"]
        return d["short_term_first"]
    if rule == "label_b":
        labels = d.get("labels") or []
        if not labels:
            return None
        first_is_alpha = labels[0] in ("a)", "x)")
        if first_is_alpha:
            return d["short_term_first"]  # "b)" is the second
        return not d["short_term_first"]
    if rule == "larger_reward":
        # "Always pick the option with the larger reward"
        # In this dataset long_term_reward >> short_term_reward typically
        if d["long_term_reward"] > d["short_term_reward"]:
            return True
        if d["long_term_reward"] < d["short_term_reward"]:
            return False
        return None
    if rule == "shorter_time":
        # "Always pick the option with the shorter delivery time"
        if d["short_term_time"] < d["long_term_time"]:
            return False  # picks short → not long_term
        if d["short_term_time"] > d["long_term_time"]:
            return True
        return None
    if rule == "rational":
        # "Pick whichever option is reachable given the time horizon"
        thm = d.get("thm")
        if thm is None or d["long_term_time"] is None:
            return None
        if thm < d["long_term_time"]:
            return False  # short
        return True       # long
    if rule == "closest_horizon":
        # "Pick the option whose delivery time is closest to the stated horizon"
        thm = d.get("thm")
        if thm is None or d["short_term_time"] is None or d["long_term_time"] is None:
            return None
        dst_st = abs(thm - d["short_term_time"])
        dst_lt = abs(thm - d["long_term_time"])
        if dst_st < dst_lt:
            return False
        if dst_lt < dst_st:
            return True
        return None
    return None


RULES = ["first", "last", "label_a", "label_b", "larger_reward", "shorter_time", "closest_horizon", "rational"]
RULE_LABELS = {
    "first":           "first listed",
    "last":            "last listed",
    "label_a":         "label 'a)/x)'",
    "label_b":         "label 'b)/y)'",
    "larger_reward":   "larger reward",
    "shorter_time":    "shorter delivery time",
    "closest_horizon": "closest to horizon",
    "rational":        "rational (horizon-aware)",
}


def _rule_match_pct(data, spec_key: str, rule: str,
                     zone_only: bool = False) -> tuple[float, int]:
    """Fraction of the model's choices that match a candidate rule.

    When zone_only=True, restrict to horizon-bearing prompts in the temporal
    reasoning zone (1-5y), which is the only regime where the "rational" rule
    (pick ST) and "closest to horizon" rule meaningfully predict ST rather
    than coinciding with the trivial "anchor-match" pattern.
    """
    n = 0
    matches = 0
    for d in _paired_responses(data, spec_key):
        if zone_only and d.get("thm") not in BETWEEN_ANCHOR:
            continue
        pred = _rule_predicts_long_term(d, rule)
        if pred is None:
            continue
        choice = _get_choice(d, spec_key)
        if choice is None:
            continue
        actual_lt = (choice == "long_term")
        if pred == actual_lt:
            matches += 1
        n += 1
    return (100.0 * matches / n if n else 0.0), n


def plot_rule_heuristic_match(data, models, output_dir):
    """Rule-match heatmap restricted to the temporal reasoning zone (1-5y).

    In the 1-5y zone, all four candidate rules diverge meaningfully:
    - "first/last listed": position-following (pure format artefact)
    - "label a/b", "larger reward", "shorter time": content-following
    - "closest to horizon" and "rational": horizon-aware (both predict ST in 1-5y)
    Outside this zone, "rational" degenerates to pattern-matching (ST at 6mo
    anchor, LT at 10y+), so we exclude those rows.
    """
    n_models = len(models)
    n_rules = len(RULES)

    mat = np.zeros((n_models, n_rules))
    for i, spec in enumerate(models):
        for j, rule in enumerate(RULES):
            pct, _ = _rule_match_pct(data, spec.key, rule, zone_only=True)
            mat[i, j] = pct

    fig, ax = plt.subplots(figsize=(max(9, n_rules * 1.3), max(8, n_models * 0.32)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    col_labels = [RULE_LABELS[r] for r in RULES]
    ax.set_xticks(range(n_rules))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([_target_yticklabel(m) for m in models], fontsize=8)
    for lbl, m in zip(ax.get_yticklabels(), models):
        if m.key == TARGET_KEY:
            lbl.set_fontweight("bold")
            lbl.set_color("#FF1493")

    # Highlight the two horizon-aware columns with a box; these are the
    # "coherent" rules. They coincide at this task's fixed delivery times
    # (ST=6mo, LT=10y) so their columns should be identical.
    horizon_cols = [RULES.index("closest_horizon"), RULES.index("rational")]
    for jc in horizon_cols:
        ax.add_patch(plt.Rectangle((jc - 0.5, -0.5), 1, n_models,
                                     fill=False, edgecolor="#0B486B",
                                     linewidth=2.0, zorder=5))
    # Vertical separator between heuristic and horizon-aware groups.
    ax.axvline(min(horizon_cols) - 0.5, color="#0B486B", linewidth=1.2, zorder=5)

    # Annotate values.
    for i in range(n_models):
        for j in range(n_rules):
            v = mat[i, j]
            color = "white" if v > 70 or v < 30 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="% choices in 1-5y matching rule", shrink=0.7)
    ax.set_xlabel("Candidate decision rule")
    ax.set_ylabel("Model")
    ax.set_title(
        "Which Rule Explains Each Model's Choices in the Temporal Reasoning Zone (1-5y)?\n"
        "Left 6 columns = surface heuristics (bad). Right 2 columns (boxed) = horizon-aware (good).\n"
        "For this task's ST=6mo / LT=10y, 'closest to horizon' and 'rational' predict the same choice in this zone.",
        fontsize=10.5, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir, "10_rule_heuristic_match")


# ---------------------------------------------------------------------------
# Plot 11: Context coherence
#   How much does the framing context affect each model's choices?
#   For each model, compute coherence (% rational on items that have a horizon)
#   and %LT (overall) per context_id. Visualize as:
#     left:  per-model spread (range of %LT across contexts) — bar, sorted
#     right: heatmap of %LT by (model, context) for the top-K most variable models
# ---------------------------------------------------------------------------


def plot_context_coherence(data, models, output_dir):
    contexts = sorted({d["context_id"] for d in data if d.get("context_id") is not None})
    if len(contexts) < 2:
        print("  Only one context — skipping context coherence plot.")
        return

    # Per-model % LT per context (no-horizon subset isolates context effect)
    no_h = [d for d in data if d["thm"] is None]
    if not no_h:
        no_h = data  # fall back to all data

    pct_by_model_ctx: dict[str, list[float]] = {}
    for spec in models:
        vals = []
        for c in contexts:
            v, _ = _lt_pct(no_h, spec.key, context_id=c)
            vals.append(v)
        pct_by_model_ctx[spec.key] = vals

    # Per-model spread (max - min across contexts)
    spreads = {k: max(v) - min(v) for k, v in pct_by_model_ctx.items()}

    # Coherence (rational match) per context for each model (paired only)
    coh_by_model_ctx: dict[str, list[float]] = {}
    for spec in models:
        paired = _paired_responses(data, spec.key)
        h_data = [d for d in paired if d["thm"] is not None and d["long_term_time"] is not None]
        vals = []
        for c in contexts:
            sub = [d for d in h_data if d["context_id"] == c]
            if not sub:
                vals.append(0.0)
                continue
            n_match = 0
            n = 0
            for d in sub:
                rational_long = d["thm"] >= d["long_term_time"]
                ch = _get_choice(d, spec.key)
                if ch is None:
                    continue
                actual = (ch == "long_term")
                if actual == rational_long:
                    n_match += 1
                n += 1
            vals.append(100.0 * n_match / n if n else 0.0)
        coh_by_model_ctx[spec.key] = vals

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, max(8, len(models) * 0.32)),
        gridspec_kw={"width_ratios": [0.8, 1.6], "wspace": 0.30},
    )

    # Left: spread bars, family-grouped and size-ordered (same as all other plots)
    specs_sorted = list(models)
    y = np.arange(len(specs_sorted))
    colors = [m.color for m in specs_sorted]
    edgecolors = ["#FF1493" if m.key == TARGET_KEY else "white" for m in specs_sorted]
    linewidths = [2.4 if m.key == TARGET_KEY else 0.5 for m in specs_sorted]
    values = [spreads[m.key] for m in specs_sorted]
    ax1.barh(y, values, color=colors, alpha=0.9, edgecolor=edgecolors, linewidth=linewidths)
    ax1.set_yticks(y)
    ax1.set_yticklabels([_target_yticklabel(m) for m in specs_sorted], fontsize=8)
    for lbl, m in zip(ax1.get_yticklabels(), specs_sorted):
        if m.key == TARGET_KEY:
            lbl.set_fontweight("bold")
            lbl.set_color("#FF1493")
    # Family dividers.
    _blocks_p11 = _family_block_rows(specs_sorted)
    if _blocks_are_meaningful(_blocks_p11):
        for _, _, end in _blocks_p11[:-1]:
            ax1.axhline(end + 0.5, color="#374151", linewidth=0.6)
    ax1.invert_yaxis()
    ax1.set_xlabel("max - min %LT across contexts (no-horizon subset)")
    ax1.set_title("Framing Spread (no-horizon)\n(max - min %LT across contexts)", fontsize=11)
    for yi, v in enumerate(values):
        ax1.text(v + 0.6, yi, f"{v:.0f}", ha="left", va="center", fontsize=7.5)

    # Right: coherence heatmap (model × context), same family/size order as left
    coh_mat = np.array([coh_by_model_ctx[m.key] for m in specs_sorted])

    im = ax2.imshow(coh_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax2.set_xticks(range(len(contexts)))
    ax2.set_xticklabels([f"ctx{i}" for i in range(len(contexts))], fontsize=8, rotation=30)
    ax2.set_yticks(range(len(specs_sorted)))
    ax2.set_yticklabels([_target_yticklabel(m) for m in specs_sorted], fontsize=8)
    for lbl, m in zip(ax2.get_yticklabels(), specs_sorted):
        if m.key == TARGET_KEY:
            lbl.set_fontweight("bold")
            lbl.set_color("#FF1493")
    if _blocks_are_meaningful(_blocks_p11):
        for _, _, end in _blocks_p11[:-1]:
            ax2.axhline(end + 0.5, color="white", linewidth=2.0, zorder=5)
            ax2.axhline(end + 0.5, color="#374151", linewidth=0.6, zorder=6)

    for i in range(coh_mat.shape[0]):
        for j in range(coh_mat.shape[1]):
            v = coh_mat[i, j]
            color = "white" if v > 70 or v < 30 else "black"
            ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax2, label="% rational (horizon-aware coherence)", shrink=0.7)
    ax2.set_xlabel("Context (framing)")
    ax2.set_title("Coherence per Framing Context\n(% rational on horizon-bearing prompts)", fontsize=11)

    fig.suptitle(
        "Does Framing Context Shift Temporal Reasoning? (left: no-horizon %LT spread, right: horizon-aware coherence)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, output_dir, "11_context_coherence")


# ---------------------------------------------------------------------------
# Plots 12-14: Qwen3-4B-Instruct-2507 deep dive (target model only)
# ---------------------------------------------------------------------------

def _target_spec(models):
    return next((m for m in models if m.key == TARGET_KEY), None)


def plot_target_horizon_context(data, models, output_dir):
    """Plot 12: horizon x context %LT heatmap for the target model."""
    target = _target_spec(models)
    if target is None:
        print("  Target model not found -- skipping plot 12.")
        return

    horizons = HORIZON_MONTHS_ORDER
    h_labels = [HORIZON_LABELS[h] for h in horizons]
    contexts = sorted(set(d["context_id"] for d in data))
    ctx_labels = [CONTEXT_SHORT.get(c, str(c)) for c in contexts]

    matrix = np.zeros((len(horizons), len(contexts)))
    for i, h in enumerate(horizons):
        for j, c in enumerate(contexts):
            v, _ = _lt_pct(data, target.key, horizon=h, context_id=c)
            matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(11.5, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if (v < 25 or v > 75) else "black"
            weight = "bold" if (v < 15 or v > 85) else "normal"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)

    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(ctx_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(horizons)))
    ax.set_yticklabels(h_labels, fontsize=10)
    ax.set_xlabel("Scenario framing (context)")
    ax.set_ylabel("Time horizon")

    tr_rows = [i for i, h in enumerate(horizons) if h in BETWEEN_ANCHOR]
    if tr_rows:
        ax.axhspan(min(tr_rows) - 0.5, max(tr_rows) + 0.5,
                   facecolor="none", edgecolor="#B91C1C",
                   linewidth=2.0, linestyle="--", zorder=6)
        # Label placed above the heatmap, centered, so it never overlaps data or colorbar.
        t = ax.text(
            (len(contexts) - 1) / 2, -1.2,
            "red-dashed box = temporal reasoning zone (1-5y) [coherence defined here]",
            ha="center", va="bottom", fontsize=9,
            color="#B91C1C", fontweight="bold",
        )
        t.set_clip_on(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("% Long-Term choice")

    fig.suptitle(
        f"{target.short}: %LT by Horizon x Scenario Framing\n"
        "(every cell = 12 prompts varying reward x label x order; coherence = 0% LT in red-boxed zone)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, output_dir, "12_target_horizon_context")


def plot_target_horizon_reward_order(data, models, output_dir):
    """Plot 13: ST-first / LT-first / order-bias delta across horizon x reward."""
    target = _target_spec(models)
    if target is None:
        print("  Target model not found -- skipping plot 13.")
        return

    horizons = HORIZON_MONTHS_ORDER
    h_labels = [HORIZON_LABELS[h] for h in horizons]
    rewards = sorted(set(d["long_term_reward"] for d in data))
    r_labels = [f"${r/1000:.0f}K" for r in rewards]

    def _lt_matrix(st_first):
        m = np.zeros((len(horizons), len(rewards)))
        for i, h in enumerate(horizons):
            for j, r in enumerate(rewards):
                v, _ = _lt_pct(data, target.key, horizon=h,
                               reward=r, st_first=st_first)
                m[i, j] = v
        return m

    mat_st = _lt_matrix(True)
    mat_lt = _lt_matrix(False)
    delta = mat_lt - mat_st  # positive = picks LT more when LT listed first

    fig, axes = plt.subplots(
        1, 3, figsize=(17, 5.8),
        gridspec_kw={"wspace": 0.55},
    )

    def _draw_pct(ax, m, title):
        im = ax.imshow(m, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                color = "white" if (v < 25 or v > 75) else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, color=color)
        ax.set_xticks(range(len(rewards)))
        ax.set_xticklabels(r_labels, fontsize=9)
        ax.set_yticks(range(len(horizons)))
        ax.set_yticklabels(h_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("LT reward")
        return im

    im_st = _draw_pct(axes[0], mat_st, "ST-first order (%LT)")
    im_lt = _draw_pct(axes[1], mat_lt, "LT-first order (%LT)")
    axes[0].set_ylabel("Time horizon")

    cbar_st = fig.colorbar(im_st, ax=axes[0], shrink=0.85, pad=0.04)
    cbar_st.set_label("% Long-Term")
    cbar_lt = fig.colorbar(im_lt, ax=axes[1], shrink=0.85, pad=0.04)
    cbar_lt.set_label("% Long-Term")

    # Delta panel: raw order bias in pp %LT (signed, symmetric).
    ax3 = axes[2]
    vmax = max(30, float(np.abs(delta).max()))
    im3 = ax3.imshow(delta, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta[i, j]
            if abs(v) < 3:
                continue
            color = "white" if abs(v) > vmax * 0.6 else "black"
            ax3.text(j, i, f"{v:+.0f}", ha="center", va="center",
                     fontsize=9, color=color)
    ax3.set_xticks(range(len(rewards)))
    ax3.set_xticklabels(r_labels, fontsize=9)
    ax3.set_yticks(range(len(horizons)))
    ax3.set_yticklabels(h_labels, fontsize=9)
    ax3.set_title("Order bias: LT-first - ST-first", fontsize=11)
    ax3.set_xlabel("LT reward")
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.85, pad=0.04)
    cbar3.set_label("pp %LT")

    fig.suptitle(
        f"{target.short}: Does Order Bias and Reward Sensitivity Vary by Horizon?",
        fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(top=0.88, bottom=0.1, left=0.06, right=0.97)
    _save(fig, output_dir, "13_target_horizon_reward_order")


def plot_target_variant_spread(data, models, output_dir):
    """Plot 14: per-horizon %LT stratified by each stimulus dimension."""
    target = _target_spec(models)
    if target is None:
        print("  Target model not found -- skipping plot 14.")
        return

    horizons = HORIZON_MONTHS_ORDER
    h_labels = [HORIZON_LABELS[h] for h in horizons]
    x = np.arange(len(horizons))

    rewards = sorted(set(d["long_term_reward"] for d in data))
    contexts = sorted(set(d["context_id"] for d in data))

    # Between-anchor zone indices.
    zone_idx = [i for i, h in enumerate(horizons) if h in BETWEEN_ANCHOR]

    def _per_horizon(spec_filter):
        return [_lt_pct(data, target.key, horizon=h, **spec_filter)[0]
                for h in horizons]

    overall = _per_horizon({})

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)

    def _zone_shade(ax):
        if zone_idx:
            ax.axvspan(min(zone_idx) - 0.4, max(zone_idx) + 0.4,
                       color="#FEE2E2", alpha=0.55, zorder=0)
        ax.axhline(50, color="#9CA3AF", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(h_labels, fontsize=9, rotation=25, ha="right")
        ax.set_ylim(-4, 105)
        ax.plot(x, overall, color="#111827", linewidth=2.8,
                marker="o", markersize=5, label="Pooled", zorder=5)

    # Panel (a): reward stratification.
    ax = axes[0, 0]
    _zone_shade(ax)
    reward_colors = ["#BFDBFE", "#60A5FA", "#1E40AF"]
    for r, col in zip(rewards, reward_colors):
        y = _per_horizon({"reward": r})
        ax.plot(x, y, color=col, linewidth=1.8, marker="s",
                markersize=4, label=f"${r/1000:.0f}K LT reward")
    ax.set_title("Stratified by LT reward size", fontsize=11)
    ax.set_ylabel("% Long-Term")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # Panel (b): label style stratification.
    ax = axes[0, 1]
    _zone_shade(ax)
    for ls, col in [("ab", "#F97316"), ("xy", "#7C3AED")]:
        y = _per_horizon({"label_style": ls})
        ax.plot(x, y, color=col, linewidth=1.8, marker="s",
                markersize=4, label=f"labels={ls}")
    ax.set_title("Stratified by label style (a/b vs x/y)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # Panel (c): order stratification.
    ax = axes[1, 0]
    _zone_shade(ax)
    for (st_first, lbl, col) in [
        (True,  "ST listed first", "#059669"),
        (False, "LT listed first", "#DC2626"),
    ]:
        y = _per_horizon({"st_first": st_first})
        ax.plot(x, y, color=col, linewidth=1.8, marker="s",
                markersize=4, label=lbl)
    ax.set_title("Stratified by presentation order", fontsize=11)
    ax.set_ylabel("% Long-Term")
    ax.set_xlabel("Time horizon")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # Panel (d): context stratification (8 lines + max-min spread annotations).
    ax = axes[1, 1]
    _zone_shade(ax)
    ctx_curves = []
    for c in contexts:
        y = _per_horizon({"context_id": c})
        ctx_curves.append(y)
        ax.plot(x, y, color=CONTEXT_COLORS.get(c, "#9CA3AF"),
                linewidth=1.2, alpha=0.75,
                label=CONTEXT_SHORT.get(c, str(c)))
    ctx_arr = np.array(ctx_curves)
    spread = ctx_arr.max(axis=0) - ctx_arr.min(axis=0)
    for xi, s in enumerate(spread):
        ax.text(xi, 2, f"{s:.0f}", ha="center", va="bottom",
                fontsize=7, color="#6B7280")
    ax.set_title("Stratified by scenario framing (labels = max-min spread)",
                 fontsize=11)
    ax.set_xlabel("Time horizon")
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)

    fig.suptitle(
        f"{target.short}: Where Does the %LT Variation Come From?\n"
        "(each panel holds one stimulus dimension; pooled %LT curve is identical across panels)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, output_dir, "14_target_variant_spread")


# ---------------------------------------------------------------------------
# Plot 15: coherence score per model (% ST in the 1-5y reasoning zone)
# ---------------------------------------------------------------------------

def plot_coherence_score(data, models, output_dir):
    """Single-number coherence score per model, sorted.

    Coherence = % of choices that pick the rational ST option across the three
    reasoning-zone horizons {1y, 2y, 5y}. This is the only regime where picking
    ST is strictly rational (only 6-month ST can deliver) and where coherence
    can be tested independent of pattern-matching at the anchor horizons.
    """
    zone_horizons = sorted(BETWEEN_ANCHOR)  # [12, 24, 60]

    rows = []
    for spec in models:
        n = lt = 0
        for d in _paired_responses(data, spec.key):
            if d.get("thm") not in BETWEEN_ANCHOR:
                continue
            choice = _get_choice(d, spec.key)
            if choice is None:
                continue
            n += 1
            if choice == "long_term":
                lt += 1
        coherence = 100.0 * (n - lt) / n if n else 0.0  # % ST
        rows.append((spec, coherence, n))

    # Sort by coherence ascending (worst at top so eye catches failures first).
    rows.sort(key=lambda r: r[1])
    labels = [_target_yticklabel(r[0]) for r in rows]
    coherences = [r[1] for r in rows]
    colors = [r[0].color for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(8, len(rows) * 0.32)))
    y = np.arange(len(rows))
    bars = ax.barh(y, coherences, color=colors, edgecolor="white",
                    linewidth=0.6, alpha=0.92)

    # Highlight target row with a heavy outline.
    for yi, (spec, _, _) in enumerate(rows):
        if spec.key == TARGET_KEY:
            bars[yi].set_edgecolor("#FF1493")
            bars[yi].set_linewidth(2.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    for lbl, (spec, _, _) in zip(ax.get_yticklabels(), rows):
        if spec.key == TARGET_KEY:
            lbl.set_fontweight("bold")
            lbl.set_color("#FF1493")

    # Reference thresholds.
    ax.axvline(50, color="#9CA3AF", linestyle=":", linewidth=1)
    ax.axvline(90, color="#16A34A", linestyle="--", linewidth=1)
    ax.text(90.5, len(rows) - 0.3, "≥90%: coherent",
            fontsize=8.5, color="#16A34A", fontweight="bold")

    # Annotate each bar with its value.
    for yi, c in enumerate(coherences):
        ax.text(c + 1, yi, f"{c:.0f}%", va="center", fontsize=8)

    ax.set_xlim(0, 108)
    ax.set_xlabel("Coherence % (fraction of 1-5y choices that pick the rational ST option)")
    ax.set_title(
        "Coherence Score per Model: %ST in the Temporal Reasoning Zone (1-5y)\n"
        "Rational target = pick ST (only 6mo option can deliver within 1-5y deadline). "
        "Sorted worst-to-best.",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir, "15_coherence_score")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python coherent_behavior_viz.py <output_dir>")
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
    print(f"Models ({len(models)}): {[m.short for m in models]}")
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
    plot_rule_heuristic_match(data, models, output_dir)
    plot_context_coherence(data, models, output_dir)
    plot_target_horizon_context(data, models, output_dir)
    plot_target_horizon_reward_order(data, models, output_dir)
    plot_target_variant_spread(data, models, output_dir)
    plot_coherence_score(data, models, output_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
