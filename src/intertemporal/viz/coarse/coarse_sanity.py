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
from ....viz.viz_palettes import BAR_COLORS
from ....viz.token_coloring import PairTokenColoring
from .coarse_helpers import finalize_plot, setup_grid


def plot_sanity_check(
    result: CoarseActPatchResults,
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
    pair: ContrastivePair | None = None,
    component: str = "resid_post",
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
        component: Component being patched (for plot title)
    """
    sanity = result.sanity_result
    if sanity is None:
        print("[viz] No sanity result to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Sanity Check [{component}]: Full Patching (All Layers + All Positions)",
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
    """Plot greedy generation results as styled cards."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y_pos = 0.95

    # Title
    ax.text(
        0.5, y_pos, "Greedy Generation Results",
        transform=ax.transAxes, fontsize=14, fontweight="bold",
        ha="center", va="top"
    )
    y_pos -= 0.08

    # Denoising card
    if sanity.denoising:
        d = sanity.denoising
        orig_label, intv_label, _ = _extract_labels(d, "denoising")
        flipped = orig_label != intv_label
        y_pos = _draw_mode_card(
            ax, y_pos, "DENOISING", "corrupt → clean",
            orig_label, intv_label, d.recovery, "Recovery", flipped,
            card_color="#E8F5E9", border_color="#4CAF50"
        )

    # Noising card
    if sanity.noising:
        n = sanity.noising
        orig_label, intv_label, _ = _extract_labels(n, "noising")
        flipped = orig_label != intv_label
        y_pos = _draw_mode_card(
            ax, y_pos, "NOISING", "clean → corrupt",
            orig_label, intv_label, n.disruption, "Disruption", flipped,
            card_color="#FFEBEE", border_color="#F44336"
        )

    # Combined score box
    score = sanity.score()
    score_color = "#4CAF50" if score > 0.8 else "#FF9800" if score > 0.5 else "#F44336"
    ax.add_patch(plt.Rectangle(
        (0.1, y_pos - 0.12), 0.8, 0.1,
        transform=ax.transAxes, facecolor="#F5F5F5",
        edgecolor=score_color, linewidth=2, clip_on=False
    ))
    ax.text(
        0.5, y_pos - 0.07,
        f"Sanity Score: {score:.2f}  |  Flips: {sanity.flip_count}/2",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        ha="center", va="center", color=score_color
    )


def _draw_mode_card(
    ax: plt.Axes, y_top: float, mode_name: str, direction: str,
    orig_label: str, intv_label: str, metric_val: float, metric_name: str,
    flipped: bool, card_color: str, border_color: str
) -> float:
    """Draw a styled card for denoising/noising results. Returns new y position."""
    card_height = 0.22
    y_bottom = y_top - card_height

    # Card background
    ax.add_patch(plt.Rectangle(
        (0.05, y_bottom), 0.9, card_height - 0.02,
        transform=ax.transAxes, facecolor=card_color,
        edgecolor=border_color, linewidth=1.5, clip_on=False
    ))

    # Mode header
    ax.text(
        0.1, y_top - 0.04, f"{mode_name}",
        transform=ax.transAxes, fontsize=11, fontweight="bold", color=border_color
    )
    ax.text(
        0.35, y_top - 0.04, f"({direction})",
        transform=ax.transAxes, fontsize=10, color="#666666"
    )

    # Flip indicator
    if flipped:
        ax.text(
            0.85, y_top - 0.04, "FLIPPED",
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            color="white", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#4CAF50", edgecolor="none")
        )

    # Original → After
    ax.text(
        0.1, y_top - 0.10, "Original:",
        transform=ax.transAxes, fontsize=10, color="#555555"
    )
    ax.text(
        0.28, y_top - 0.10, f"'{orig_label}'",
        transform=ax.transAxes, fontsize=11, fontweight="bold", fontfamily="monospace"
    )
    ax.text(
        0.5, y_top - 0.10, "→",
        transform=ax.transAxes, fontsize=12, ha="center"
    )
    ax.text(
        0.55, y_top - 0.10, "After:",
        transform=ax.transAxes, fontsize=10, color="#555555"
    )
    ax.text(
        0.70, y_top - 0.10, f"'{intv_label}'",
        transform=ax.transAxes, fontsize=11, fontweight="bold", fontfamily="monospace"
    )

    # Metric value
    ax.text(
        0.1, y_top - 0.17, f"{metric_name}:",
        transform=ax.transAxes, fontsize=10, color="#555555"
    )
    metric_color = "#4CAF50" if metric_val > 0.8 else "#FF9800" if metric_val > 0.5 else "#F44336"
    ax.text(
        0.28, y_top - 0.17, f"{metric_val:.4f}",
        transform=ax.transAxes, fontsize=12, fontweight="bold", color=metric_color
    )

    return y_bottom - 0.02


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

    metrics_names = ["prob(short)", "prob(long)", "fork_div", "recov/disrupt"]
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
            n_metrics.disruption,
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
