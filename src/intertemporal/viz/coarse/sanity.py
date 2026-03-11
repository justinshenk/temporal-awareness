"""Sanity check visualization for coarse activation patching.

Shows diagnostic information for the full-patch sanity check
(all layers + all positions patched simultaneously).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ....activation_patching import IntervenedChoiceMetrics
from ....activation_patching.coarse import CoarseActPatchResults
from ....common.contrastive_pair import ContrastivePair
from ....viz.palettes import BAR_COLORS
from ....viz.token_coloring import PairTokenColoring
from .helpers import finalize_plot, setup_grid


def plot_sanity_check(
    result: CoarseActPatchResults,
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
    pair: ContrastivePair | None = None,
) -> None:
    """Plot sanity check results.

    Creates a 2x2 figure with:
    - Panel 1: Greedy generation results text
    - Panel 2: Probability metrics bar chart
    - Panel 3: Per-position logprob difference
    - Panel 4: Reciprocal rank comparison

    Args:
        result: Coarse patching results containing sanity_result
        output_dir: Directory to save output
        coloring: Token coloring (unused, kept for API compatibility)
        pair: ContrastivePair for per-position logprob plot
    """
    sanity = result.sanity_result
    if sanity is None:
        print("[viz] No sanity result to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Sanity Check: Full Patching (All Layers + All Positions)",
        fontsize=14,
        fontweight="bold",
    )

    # Extract metrics
    d_metrics = (
        IntervenedChoiceMetrics.from_choice(sanity.denoising)
        if sanity.denoising
        else None
    )
    n_metrics = (
        IntervenedChoiceMetrics.from_choice(sanity.noising) if sanity.noising else None
    )

    # Panel 1: Greedy generation text
    _plot_greedy_results(axes[0, 0], sanity, d_metrics, n_metrics)

    # Panel 2: Probability metrics bar chart
    _plot_probability_bars(axes[0, 1], d_metrics, n_metrics)

    # Panel 3: Per-position logprob difference
    _plot_logprob_diff(axes[1, 0], pair)

    # Panel 4: Reciprocal rank bars
    _plot_reciprocal_rank_bars(axes[1, 1], d_metrics, n_metrics)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    finalize_plot(fig, output_dir / "sanity_check.png")


def _plot_greedy_results(
    ax: plt.Axes,
    sanity,
    d_metrics: IntervenedChoiceMetrics | None,
    n_metrics: IntervenedChoiceMetrics | None,
) -> None:
    """Plot greedy generation results as text."""
    ax.axis("off")

    lines = ["GREEDY GENERATION RESULTS", "=" * 50, ""]
    lines.append("Patching ALL layers + ALL positions at once:")
    lines.append("")

    if sanity.denoising:
        d = sanity.denoising
        orig_label, intv_label, orig_response = _extract_labels(d, "denoising")
        flip_marker = "FLIPPED" if orig_label != intv_label else "(no flip)"
        lines.extend(
            [
                "DENOISING (corrupt->clean):",
                f"  Original greedy: '{orig_label}'  {flip_marker}",
                f"  After patch:     '{intv_label}'",
                f"  Recovery: {d.recovery:.4f}",
            ]
        )
        if orig_response:
            lines.append(f"  Response: '{orig_response[:40]}...'")
        lines.append("")

    if sanity.noising:
        n = sanity.noising
        orig_label, intv_label, orig_response = _extract_labels(n, "noising")
        flip_marker = "FLIPPED" if orig_label != intv_label else "(no flip)"
        lines.extend(
            [
                "NOISING (clean->corrupt):",
                f"  Original greedy: '{orig_label}'  {flip_marker}",
                f"  After patch:     '{intv_label}'",
                f"  Recovery: {n.recovery:.4f}",
            ]
        )
        if orig_response:
            lines.append(f"  Response: '{orig_response[:40]}...'")

    # Combined sanity score
    lines.extend(["", "-" * 45, f"Combined Sanity Score: {sanity.score():.4f}"])
    if sanity.flip_count > 0:
        lines.append(f"Flips: {sanity.flip_count}/2")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )


def _extract_labels(choice, mode: str) -> tuple[str, str, str]:
    """Extract original and intervened labels from a choice."""
    orig_label = "?"
    intv_label = "?"
    orig_response = ""

    # Get original (baseline) choice
    orig = choice.baseline_corrupted if mode == "denoising" else choice.baseline_clean
    if orig:
        try:
            orig_label = orig.labels[orig.choice_idx] if orig.labels else "?"
            if orig.response_texts:
                orig_response = orig.response_texts[orig.choice_idx][:60]
        except Exception:
            pass

    # Get intervened choice
    if choice.intervened:
        try:
            intv_label = (
                choice.intervened.labels[choice.intervened.choice_idx]
                if choice.intervened.labels
                else "?"
            )
        except Exception:
            pass

    return orig_label, intv_label, orig_response


def _plot_probability_bars(
    ax: plt.Axes,
    d_metrics: IntervenedChoiceMetrics | None,
    n_metrics: IntervenedChoiceMetrics | None,
) -> None:
    """Plot probability metrics bar chart."""
    if not (d_metrics or n_metrics):
        ax.text(0.5, 0.5, "No sanity data", ha="center", va="center")
        ax.axis("off")
        return

    metrics_names = ["prob(short)", "prob(long)", "fork_div", "recovery"]
    x = np.arange(len(metrics_names))
    width = 0.35

    d_vals = (
        [
            d_metrics.prob_short,
            d_metrics.prob_long,
            d_metrics.fork_diversity,
            d_metrics.recovery,
        ]
        if d_metrics
        else [0, 0, 0, 0]
    )
    n_vals = (
        [
            n_metrics.prob_short,
            n_metrics.prob_long,
            n_metrics.fork_diversity,
            n_metrics.recovery,
        ]
        if n_metrics
        else [0, 0, 0, 0]
    )

    ax.bar(x - width / 2, d_vals, width, label="Denoising", color=BAR_COLORS["denoising"])
    ax.bar(x + width / 2, n_vals, width, label="Noising", color=BAR_COLORS["noising"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Sanity Check Metrics", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    setup_grid(ax)


def _plot_logprob_diff(ax: plt.Axes, pair: ContrastivePair | None) -> None:
    """Plot per-position logprob difference."""
    if pair is None:
        ax.text(0.5, 0.5, "No pair data for per-position plot", ha="center", va="center")
        ax.axis("off")
        return

    clean_logprobs = pair.clean_traj.logprobs
    corrupted_logprobs = pair.corrupted_traj.logprobs
    position_mapping = pair.position_mapping

    # Compute per-position difference
    positions = []
    logprob_diffs = []
    for src_pos in range(len(clean_logprobs)):
        dst_pos = position_mapping.get(src_pos, src_pos)
        if dst_pos is not None and dst_pos < len(corrupted_logprobs):
            positions.append(src_pos)
            diff = clean_logprobs[src_pos] - corrupted_logprobs[dst_pos]
            logprob_diffs.append(diff)

    if not positions:
        ax.text(0.5, 0.5, "No position data", ha="center", va="center")
        ax.axis("off")
        return

    ax.plot(positions, logprob_diffs, "b-", linewidth=1.5, alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Position", fontsize=11)
    ax.set_ylabel("logprob(short) - logprob(long)", fontsize=11)
    ax.set_title("Per-Position Logprob Diff (short - long traj)", fontsize=12, fontweight="bold")
    setup_grid(ax)

    # Summary stats
    mean_diff = np.mean(logprob_diffs)
    max_diff = np.max(np.abs(logprob_diffs))
    ax.text(
        0.02,
        0.98,
        f"mean={mean_diff:.4f}\nmax|diff|={max_diff:.4f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def _plot_reciprocal_rank_bars(
    ax: plt.Axes,
    d_metrics: IntervenedChoiceMetrics | None,
    n_metrics: IntervenedChoiceMetrics | None,
) -> None:
    """Plot reciprocal rank comparison bar chart."""
    if not (d_metrics or n_metrics):
        ax.text(0.5, 0.5, "No sanity data", ha="center", va="center")
        ax.axis("off")
        return

    labels = ["recip_rank(short)", "recip_rank(long)"]
    x = np.arange(len(labels))
    width = 0.35

    d_rr = [
        d_metrics.reciprocal_rank_short if d_metrics else 0,
        d_metrics.reciprocal_rank_long if d_metrics else 0,
    ]
    n_rr = [
        n_metrics.reciprocal_rank_short if n_metrics else 0,
        n_metrics.reciprocal_rank_long if n_metrics else 0,
    ]

    ax.bar(x - width / 2, d_rr, width, label="Denoising", color=BAR_COLORS["denoising"])
    ax.bar(x + width / 2, n_rr, width, label="Noising", color=BAR_COLORS["noising"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Reciprocal Rank (1/rank)", fontsize=11)
    ax.set_title("Sanity Check: Reciprocal Ranks", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    setup_grid(ax)
