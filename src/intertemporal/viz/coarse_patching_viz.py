"""Visualization for coarse activation patching results.

Creates line plots similar to logit_lens_viz.py style:
- White background
- Dual y-axes for different metric scales
- X-axis tick labels colored by token type (for position sweeps)
- Separate PNG files for denoising vs noising
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from ...activation_patching import IntervenedChoiceMetrics, ActPatchTargetResult
from ...viz.plot_helpers import finalize_plot as _finalize_plot
from ...viz.palettes import PALETTES, TOKEN_COLORS, BAR_COLORS
from ...activation_patching.coarse_activation_patching import (
    CoarseActPatchResults,
    CoarseActPatchAggregatedResults,
)
from ...viz.token_coloring import PairTokenColoring
from ...common.contrastive_pair import ContrastivePair

# Import from modular coarse viz package
from .coarse import (
    METRIC_COLORS,
    LINE_STYLES,
    LINE_WIDTHS,
    MARKERS,
    MARKER_SIZES,
    get_tick_spacing as _get_tick_spacing,
    get_tick_color as _get_tick_color,
    color_xaxis_ticks as _color_xaxis_ticks,
    save_with_colored_ticks as _save_with_colored_ticks,
    save_with_colored_ticks_multi as _save_with_colored_ticks_multi,
    add_token_type_legend as _add_token_type_legend,
    finalize_plot as _finalize_plot,
)


def visualize_coarse_patching(
    result: CoarseActPatchResults | CoarseActPatchAggregatedResults | None,
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
    pair: ContrastivePair | None = None,
) -> None:
    """Visualize coarse activation patching results.

    Creates (per step size):
    - coarse_layer_sweep_short_{step}.png: Layer sweep with short=clean (recovery)
    - coarse_layer_sweep_long_{step}.png: Layer sweep with long=clean (disruption)
    - coarse_position_sweep_short_{step}.png: Position sweep with short=clean (recovery)
    - coarse_position_sweep_long_{step}.png: Position sweep with long=clean (disruption)
    - denoising_vs_noising_{step}.png: Comparison scatter plots
    - sanity_check.png: Diagnostic metrics

    Args:
        result: CoarseActPatchResults or CoarseActPatchAggregatedResults
        output_dir: Directory to save plots
        coloring: Token coloring for position colors
        pair: ContrastivePair for sanity check visualization
    """
    if result is None:
        print("[viz] No coarse patching results to visualize")
        return

    # Handle aggregated results
    if isinstance(result, CoarseActPatchAggregatedResults):
        _visualize_aggregated_coarse(result, output_dir, coloring)
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer sweep visualization - separate files for short=clean and long=clean perspectives
    for step_size in result.layer_step_sizes:
        layer_data = result.get_layer_results_for_step(step_size)
        if layer_data:
            _plot_layer_sweep_single(layer_data, output_dir, step_size, "short")
            _plot_layer_sweep_single(layer_data, output_dir, step_size, "long")

    # Position sweep visualization - separate files for short=clean and long=clean perspectives
    for step_size in result.position_step_sizes:
        pos_data = result.get_position_results_for_step(step_size)
        if pos_data:
            _plot_position_sweep_single(pos_data, output_dir, step_size, "short", coloring)
            _plot_position_sweep_single(pos_data, output_dir, step_size, "long", coloring)

    # Denoising vs Noising comparison plots - one file per step size
    for step_size in result.layer_step_sizes:
        layer_data = result.get_layer_results_for_step(step_size)
        pos_data = result.get_position_results_for_step(step_size) if step_size in result.position_step_sizes else {}
        if layer_data or pos_data:
            _plot_denoising_vs_noising_comparison(layer_data, pos_data, output_dir, coloring, step_size)

    # Sanity check visualization
    if result.sanity_result:
        _plot_sanity_check(result, output_dir, coloring, pair)

    print(f"[viz] Coarse patching plots saved to {output_dir}")


def _plot_layer_sweep_single(
    layer_data: dict[int, ActPatchTargetResult],
    output_dir: Path,
    step_size: int,
    clean_traj: Literal["short", "long"],
) -> None:
    """Plot layer sweep with both denoising and noising, from one trajectory's perspective.

    Creates 2x6 subplots:
    - Top row: Denoising
    - Bottom row: Noising
    Columns: Core | Logprobs/Probs | Logits | Fork | Vocab | Trajectory
    """
    layers = sorted(layer_data.keys())
    if not layers:
        return

    # Create figure with 2x6 subplots - LARGE for readability
    fig, axes = plt.subplots(2, 6, figsize=(54, 16), facecolor="white")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")
    fig.suptitle(
        f"Coarse Layer Sweep, Clean = {clean_traj}, Steps = {step_size}",
        fontsize=24,
        fontweight="bold",
    )

    # Apply tick spacing
    tick_spacing = _get_tick_spacing(len(layers))
    tick_positions = layers[::tick_spacing]

    for row_idx, mode in enumerate(["denoising", "noising"]):
        # Extract metrics for each layer
        all_metrics = []
        for layer in layers:
            target_result = layer_data[layer]
            choice = (
                target_result.denoising if mode == "denoising" else target_result.noising
            )
            all_metrics.append(IntervenedChoiceMetrics.from_choice(choice))

        # Row label
        row_label = "Denoising" if mode == "denoising" else "Noising"

        # Add row subtitle on the left (outside the plots)
        fig.text(
            0.01, 0.75 - row_idx * 0.38, row_label,
            fontsize=20, fontweight="bold", rotation=90,
            va="center", ha="center"
        )

        # ─── Column 1: Core metrics ───
        ax1 = axes[row_idx, 0]
        recoveries = [m.recovery for m in all_metrics]
        logit_diffs = [m.logit_diff for m in all_metrics]
        norm_logit_diffs = [m.norm_logit_diff for m in all_metrics]
        rr_shorts = [m.reciprocal_rank_short for m in all_metrics]

        # Recovery/Disruption and recip_rank on primary axis (left)
        recovery_label = "recovery" if mode == "denoising" else "disruption"
        ax1.plot(layers, recoveries, linestyle=LINE_STYLES["recovery"],
                 color=METRIC_COLORS["recovery"], linewidth=LINE_WIDTHS["recovery"],
                 marker=MARKERS["recovery"], markersize=MARKER_SIZES["recovery"], label=recovery_label)
        ax1.plot(layers, rr_shorts,
                 linestyle=LINE_STYLES["rr_short"], color=METRIC_COLORS["rr_short"],
                 linewidth=LINE_WIDTHS["rr_short"], marker=MARKERS["rr_short"],
                 markersize=MARKER_SIZES["rr_short"], label="recip_rank(clean)")
        ylabel_left = "Recovery / RR" if mode == "denoising" else "Disruption / RR"
        ax1.set_ylabel(ylabel_left, fontsize=16, fontweight="bold")
        ax1.tick_params(axis="y", labelsize=13)
        ax1.set_ylim(-0.1, 1.1)

        # Logit diff on secondary axis (right)
        ax1b = ax1.twinx()
        ax1b.plot(layers, logit_diffs, linestyle=LINE_STYLES["logit_diff"],
                  color=METRIC_COLORS["logit_diff"], linewidth=LINE_WIDTHS["logit_diff"],
                  marker=MARKERS["logit_diff"], markersize=MARKER_SIZES["logit_diff"], label="logit_diff")
        ax1b.plot(layers, norm_logit_diffs, linestyle=LINE_STYLES["norm_logit_diff"],
                  color=METRIC_COLORS["norm_logit_diff"], linewidth=LINE_WIDTHS["norm_logit_diff"],
                  marker=MARKERS["norm_logit_diff"], markersize=MARKER_SIZES["norm_logit_diff"], label="norm_logit_diff")
        ax1b.set_ylabel("Logit Diff", fontsize=16, color=METRIC_COLORS["logit_diff"])
        ax1b.tick_params(axis="y", labelcolor=METRIC_COLORS["logit_diff"], labelsize=13)

        ax1.set_xlabel("Layer", fontsize=16)
        ax1.tick_params(axis="x", labelsize=13)
        ax1.set_title("Core", fontsize=18, fontweight="bold")
        ax1.grid(True, alpha=0.4, linewidth=1.0)
        ax1b.axhline(y=0, color="gray", linestyle="-", alpha=0.5, linewidth=1)
        ax1.set_xticks(tick_positions)
        if row_idx == 1:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines1b, labels1b = ax1b.get_legend_handles_labels()
            ax1.legend(lines1 + lines1b, labels1 + labels1b,
                       loc="upper left", bbox_to_anchor=(0, -0.10), fontsize=13, ncol=1, frameon=True, fancybox=True)

        # ─── Column 2: Probs/Logprobs ───
        ax2 = axes[row_idx, 1]
        logprob_shorts = [m.logprob_short for m in all_metrics]
        logprob_longs = [m.logprob_long for m in all_metrics]
        prob_shorts = [m.prob_short for m in all_metrics]
        prob_longs = [m.prob_long for m in all_metrics]

        # Prob on primary axis (left)
        ax2.plot(layers, prob_shorts, linestyle=LINE_STYLES["prob_short"],
                 color=METRIC_COLORS["prob_short"], linewidth=LINE_WIDTHS["prob_short"],
                 marker=MARKERS["prob_short"], markersize=MARKER_SIZES["prob_short"],
                 markerfacecolor="white", markeredgecolor=METRIC_COLORS["prob_short"],
                 markeredgewidth=2, label="prob(short)")
        ax2.plot(layers, prob_longs, linestyle=LINE_STYLES["prob_long"],
                 color=METRIC_COLORS["prob_long"], linewidth=LINE_WIDTHS["prob_long"],
                 marker=MARKERS["prob_long"], markersize=MARKER_SIZES["prob_long"],
                 markerfacecolor="white", markeredgecolor=METRIC_COLORS["prob_long"],
                 markeredgewidth=2, label="prob(long)")
        ax2.set_ylabel("Probability", fontsize=16, fontweight="bold")
        ax2.tick_params(axis="y", labelsize=13)
        ax2.set_ylim(-0.05, 1.05)

        # Logprob on secondary axis (right)
        ax2b = ax2.twinx()
        ax2b.plot(layers, logprob_shorts, linestyle=LINE_STYLES["logprob_short"],
                  color=METRIC_COLORS["logprob_short"], linewidth=LINE_WIDTHS["logprob_short"],
                  marker=MARKERS["logprob_short"], markersize=MARKER_SIZES["logprob_short"],
                  markerfacecolor=METRIC_COLORS["logprob_short"], label="logprob(short)")
        ax2b.plot(layers, logprob_longs, linestyle=LINE_STYLES["logprob_long"],
                  color=METRIC_COLORS["logprob_long"], linewidth=LINE_WIDTHS["logprob_long"],
                  marker=MARKERS["logprob_long"], markersize=MARKER_SIZES["logprob_long"],
                  markerfacecolor=METRIC_COLORS["logprob_long"], label="logprob(long)")
        ax2b.set_ylabel("Logprob", fontsize=16)
        ax2b.tick_params(axis="y", labelsize=13)

        ax2.set_xlabel("Layer", fontsize=16)
        ax2.tick_params(axis="x", labelsize=13)
        ax2.set_title("Probs/Logprobs", fontsize=18, fontweight="bold")
        ax2.grid(True, alpha=0.4, linewidth=1.0)
        ax2.set_xticks(tick_positions)
        if row_idx == 1:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines2b, labels2b = ax2b.get_legend_handles_labels()
            ax2.legend(lines2 + lines2b, labels2 + labels2b,
                       loc="upper left", bbox_to_anchor=(0, -0.10), fontsize=13, ncol=1, frameon=True, fancybox=True)

        # ─── Column 3: Logits ───
        ax3 = axes[row_idx, 2]
        logit_shorts = [m.logit_short for m in all_metrics]
        logit_longs = [m.logit_long for m in all_metrics]
        norm_logit_shorts = [m.norm_logit_short for m in all_metrics]
        norm_logit_longs = [m.norm_logit_long for m in all_metrics]

        # Check if logits have valid data (not all zeros)
        has_valid_logits = any(v != 0.0 for v in logit_shorts + logit_longs)

        if has_valid_logits:
            ax3.plot(layers, logit_shorts, linestyle=LINE_STYLES["logit_short"],
                     color=METRIC_COLORS["logit_short"], linewidth=LINE_WIDTHS["logit_short"],
                     marker=MARKERS["logit_short"], markersize=MARKER_SIZES["logit_short"],
                     markerfacecolor=METRIC_COLORS["logit_short"], label="logit(short)")
            ax3.plot(layers, logit_longs, linestyle=LINE_STYLES["logit_long"],
                     color=METRIC_COLORS["logit_long"], linewidth=LINE_WIDTHS["logit_long"],
                     marker=MARKERS["logit_long"], markersize=MARKER_SIZES["logit_long"],
                     markerfacecolor=METRIC_COLORS["logit_long"], label="logit(long)")

            ax3b = ax3.twinx()
            ax3b.plot(layers, norm_logit_shorts, linestyle=LINE_STYLES["norm_logit_short"],
                      color=METRIC_COLORS["norm_logit_short"], linewidth=LINE_WIDTHS["norm_logit_short"],
                      marker=MARKERS["norm_logit_short"], markersize=MARKER_SIZES["norm_logit_short"],
                      markerfacecolor="white", markeredgecolor=METRIC_COLORS["norm_logit_short"],
                      markeredgewidth=2, label="norm_logit(short)")
            ax3b.plot(layers, norm_logit_longs, linestyle=LINE_STYLES["norm_logit_long"],
                      color=METRIC_COLORS["norm_logit_long"], linewidth=LINE_WIDTHS["norm_logit_long"],
                      marker=MARKERS["norm_logit_long"], markersize=MARKER_SIZES["norm_logit_long"],
                      markerfacecolor="white", markeredgecolor=METRIC_COLORS["norm_logit_long"],
                      markeredgewidth=2, label="norm_logit(long)")
            ax3b.set_ylabel("Z-score", fontsize=16)
            ax3b.tick_params(axis="y", labelsize=13)

            ax3.set_xlabel("Layer", fontsize=16)
            ax3.set_ylabel("Raw Logit", fontsize=16)
            ax3.tick_params(axis="both", labelsize=13)
        else:
            ax3.text(0.5, 0.5, "No logit data", ha="center", va="center", transform=ax3.transAxes, fontsize=18)
            ax3.set_xlabel("Layer", fontsize=16)

        ax3.set_title("Logits", fontsize=18, fontweight="bold")
        ax3.grid(True, alpha=0.4, linewidth=1.0)
        ax3.set_xticks(tick_positions)
        if row_idx == 1 and has_valid_logits:
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines3b, labels3b = ax3b.get_legend_handles_labels()
            ax3.legend(lines3 + lines3b, labels3 + labels3b,
                       loc="upper left", bbox_to_anchor=(0, -0.10), fontsize=13, ncol=1, frameon=True, fancybox=True)

        # ─── Column 4: Fork metrics (dual y-axes: entropy vs diversity) ───
        ax4 = axes[row_idx, 3]
        fork_entropies = [m.fork_entropy for m in all_metrics]
        fork_divs = [m.fork_diversity for m in all_metrics]
        fork_simpsons = [m.fork_simpson for m in all_metrics]

        # Entropy on left axis
        ax4.plot(layers, fork_entropies, linestyle=LINE_STYLES["fork_entropy"],
                 color=METRIC_COLORS["fork_entropy"], linewidth=LINE_WIDTHS["fork_entropy"],
                 marker=MARKERS["fork_entropy"], markersize=MARKER_SIZES["fork_entropy"],
                 markerfacecolor=METRIC_COLORS["fork_entropy"], label="entropy")
        ax4.set_ylabel("Entropy (bits)", fontsize=16, color=METRIC_COLORS["fork_entropy"], fontweight="bold")
        ax4.tick_params(axis="y", labelcolor=METRIC_COLORS["fork_entropy"], labelsize=13)
        ax4.set_ylim(-0.05, 1.1)

        # Diversity (effective #) on right axis
        ax4b = ax4.twinx()
        ax4b.plot(layers, fork_divs, linestyle=LINE_STYLES["fork_diversity"],
                  color=METRIC_COLORS["fork_diversity"], linewidth=LINE_WIDTHS["fork_diversity"],
                  marker=MARKERS["fork_diversity"], markersize=MARKER_SIZES["fork_diversity"],
                  markerfacecolor=METRIC_COLORS["fork_diversity"], label="diversity")
        ax4b.plot(layers, fork_simpsons, linestyle=LINE_STYLES["fork_simpson"],
                  color=METRIC_COLORS["fork_simpson"], linewidth=LINE_WIDTHS["fork_simpson"],
                  marker=MARKERS["fork_simpson"], markersize=MARKER_SIZES["fork_simpson"],
                  markerfacecolor=METRIC_COLORS["fork_simpson"], label="simpson")
        ax4b.set_ylabel("Effective #", fontsize=16, color=METRIC_COLORS["fork_diversity"], fontweight="bold")
        ax4b.tick_params(axis="y", labelcolor=METRIC_COLORS["fork_diversity"], labelsize=13)
        ax4b.set_ylim(0.9, 2.1)

        ax4.set_xlabel("Layer", fontsize=16)
        ax4.tick_params(axis="x", labelsize=13)
        ax4.set_title("Fork", fontsize=18, fontweight="bold")
        ax4.grid(True, alpha=0.4, linewidth=1.0)
        ax4.set_xticks(tick_positions)
        if row_idx == 1:
            lines4, labels4 = ax4.get_legend_handles_labels()
            lines4b, labels4b = ax4b.get_legend_handles_labels()
            ax4.legend(lines4 + lines4b, labels4 + labels4b,
                       loc="upper left", bbox_to_anchor=(0, -0.10), fontsize=13, ncol=1, frameon=True, fancybox=True)

        # ─── Column 5: Vocab metrics (dual y-axes: entropy vs diversity) ───
        ax5 = axes[row_idx, 4]
        vocab_entropies = [m.vocab_entropy for m in all_metrics]
        vocab_divs = [m.vocab_diversity for m in all_metrics]
        vocab_simpsons = [m.vocab_simpson for m in all_metrics]

        # Entropy on left axis
        ax5.plot(layers, vocab_entropies, linestyle=LINE_STYLES["vocab_entropy"],
                 color=METRIC_COLORS["vocab_entropy"], linewidth=LINE_WIDTHS["vocab_entropy"],
                 marker=MARKERS["vocab_entropy"], markersize=MARKER_SIZES["vocab_entropy"],
                 markerfacecolor=METRIC_COLORS["vocab_entropy"], label="entropy")
        ax5.set_ylabel("Entropy (nats)", fontsize=16, color=METRIC_COLORS["vocab_entropy"], fontweight="bold")
        ax5.tick_params(axis="y", labelcolor=METRIC_COLORS["vocab_entropy"], labelsize=13)

        # Diversity (effective #) on right axis
        ax5b = ax5.twinx()
        ax5b.plot(layers, vocab_divs, linestyle=LINE_STYLES["vocab_diversity"],
                  color=METRIC_COLORS["vocab_diversity"], linewidth=LINE_WIDTHS["vocab_diversity"],
                  marker=MARKERS["vocab_diversity"], markersize=MARKER_SIZES["vocab_diversity"],
                  markerfacecolor=METRIC_COLORS["vocab_diversity"], label="diversity")
        ax5b.plot(layers, vocab_simpsons, linestyle=LINE_STYLES["vocab_simpson"],
                  color=METRIC_COLORS["vocab_simpson"], linewidth=LINE_WIDTHS["vocab_simpson"],
                  marker=MARKERS["vocab_simpson"], markersize=MARKER_SIZES["vocab_simpson"],
                  markerfacecolor=METRIC_COLORS["vocab_simpson"], label="simpson")
        ax5b.set_ylabel("Effective #", fontsize=16, color=METRIC_COLORS["vocab_diversity"], fontweight="bold")
        ax5b.tick_params(axis="y", labelcolor=METRIC_COLORS["vocab_diversity"], labelsize=13)

        ax5.set_xlabel("Layer", fontsize=16)
        ax5.tick_params(axis="x", labelsize=13)
        ax5.set_title("Vocab", fontsize=18, fontweight="bold")
        ax5.grid(True, alpha=0.4, linewidth=1.0)
        ax5.set_xticks(tick_positions)
        if row_idx == 1:
            lines5, labels5 = ax5.get_legend_handles_labels()
            lines5b, labels5b = ax5b.get_legend_handles_labels()
            ax5.legend(lines5 + lines5b, labels5 + labels5b,
                       loc="upper left", bbox_to_anchor=(0, -0.10), fontsize=13, ncol=1, frameon=True, fancybox=True)

        # ─── Column 6: Trajectory + TCB (inv_perplexity on left, TCB on right) ───
        ax6 = axes[row_idx, 5]
        inv_perps_short = [m.traj_inv_perplexity_short for m in all_metrics]
        inv_perps_long = [m.traj_inv_perplexity_long for m in all_metrics]
        vocab_tcbs = [m.vocab_tcb for m in all_metrics]

        # Inv perplexity on left axis
        ax6.plot(layers, inv_perps_short, linestyle=LINE_STYLES["inv_perplexity_short"],
                 color=METRIC_COLORS["inv_perplexity_short"], linewidth=LINE_WIDTHS["inv_perplexity_short"],
                 marker=MARKERS["inv_perplexity_short"], markersize=MARKER_SIZES["inv_perplexity_short"],
                 markerfacecolor=METRIC_COLORS["inv_perplexity_short"], label="inv_ppl(short)")
        ax6.plot(layers, inv_perps_long, linestyle=LINE_STYLES["inv_perplexity_long"],
                 color=METRIC_COLORS["inv_perplexity_long"], linewidth=LINE_WIDTHS["inv_perplexity_long"],
                 marker=MARKERS["inv_perplexity_long"], markersize=MARKER_SIZES["inv_perplexity_long"],
                 markerfacecolor=METRIC_COLORS["inv_perplexity_long"], label="inv_ppl(long)")
        ax6.set_ylabel("Inv Perplexity", fontsize=16)
        ax6.tick_params(axis="y", labelsize=13)
        # Auto-scale y-axis for small inv_perplexity values

        # TCB on right axis
        ax6b = ax6.twinx()
        ax6b.plot(layers, vocab_tcbs,
                  linestyle=":", color=METRIC_COLORS["vocab_tcb"],
                  linewidth=LINE_WIDTHS["vocab_tcb"], marker=MARKERS["vocab_tcb"],
                  markersize=MARKER_SIZES["vocab_tcb"],
                  markerfacecolor=METRIC_COLORS["vocab_tcb"], label="TCB")
        ax6b.set_ylabel("TCB", fontsize=16, color=METRIC_COLORS["vocab_tcb"], fontweight="bold")
        ax6b.tick_params(axis="y", labelcolor=METRIC_COLORS["vocab_tcb"], labelsize=13)

        ax6.set_xlabel("Layer", fontsize=16)
        ax6.tick_params(axis="x", labelsize=13)
        ax6.set_title("Trajectory + TCB", fontsize=18, fontweight="bold")
        ax6.grid(True, alpha=0.4, linewidth=1.0)
        ax6.set_xticks(tick_positions)
        if row_idx == 1:
            lines6, labels6 = ax6.get_legend_handles_labels()
            lines6b, labels6b = ax6b.get_legend_handles_labels()
            ax6.legend(lines6 + lines6b, labels6 + labels6b,
                       loc="upper left", bbox_to_anchor=(0, -0.10), fontsize=13, ncol=1, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0.03, 0.08, 1, 0.95])
    save_path = output_dir / f"coarse_layer_sweep_{clean_traj}_{step_size}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _plot_position_sweep_single(
    position_data: dict[int, ActPatchTargetResult],
    output_dir: Path,
    step_size: int,
    clean_traj: Literal["short", "long"],
    coloring: PairTokenColoring | None = None,
) -> None:
    """Plot position sweep with both denoising and noising, from one trajectory's perspective.

    Creates 2x6 subplots:
    - Top row: Denoising
    - Bottom row: Noising
    Columns: Core | Logprobs/Probs | Logits | Fork | Vocab | Trajectory

    X-axis tick labels are colored by token type.
    """
    positions = sorted(position_data.keys())
    if not positions:
        return

    # Get prompt/response boundary and choice divergent position
    prompt_boundary = None
    choice_div_pos = None
    if coloring:
        prompt_boundary = coloring.short_prompt_len
        for pos, info in coloring.short_colors.items():
            if info.is_choice_divergent:
                choice_div_pos = pos
                break

    # Create figure with 2x6 subplots - LARGE for readability
    fig, axes = plt.subplots(2, 6, figsize=(54, 16), facecolor="white")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")
    fig.suptitle(
        f"Coarse Position Sweep, Clean = {clean_traj}, Steps = {step_size}",
        fontsize=24,
        fontweight="bold",
    )

    all_axes = []  # Collect all axes for tick coloring

    # Apply tick spacing
    tick_spacing = _get_tick_spacing(len(positions))
    tick_positions = positions[::tick_spacing]

    def add_vlines(ax):
        """Add vertical lines for prompt boundary and choice divergent position."""
        if prompt_boundary and min(positions) < prompt_boundary < max(positions):
            ax.axvline(x=prompt_boundary, color="red", linestyle="--", alpha=0.7)
        if choice_div_pos and min(positions) <= choice_div_pos <= max(positions):
            ax.axvline(x=choice_div_pos, color=METRIC_COLORS["fork_diversity"], linestyle=":", linewidth=2, alpha=0.8)

    for row_idx, mode in enumerate(["denoising", "noising"]):
        # Extract metrics for each position
        all_metrics = []
        for pos in positions:
            target_result = position_data[pos]
            choice = (
                target_result.denoising if mode == "denoising" else target_result.noising
            )
            all_metrics.append(IntervenedChoiceMetrics.from_choice(choice))

        # Row label
        row_label = "Denoising" if mode == "denoising" else "Noising"

        # Add row subtitle on the left (outside the plots)
        fig.text(
            0.01, 0.75 - row_idx * 0.42, row_label,
            fontsize=14, fontweight="bold", rotation=90,
            va="center", ha="center"
        )

        # ─── Column 1: Core metrics ───
        ax1 = axes[row_idx, 0]
        all_axes.append(ax1)
        recoveries = [m.recovery for m in all_metrics]
        logit_diffs = [m.logit_diff for m in all_metrics]
        norm_logit_diffs = [m.norm_logit_diff for m in all_metrics]
        rr_shorts = [m.reciprocal_rank_short for m in all_metrics]

        # Recovery/Disruption and recip_rank on primary axis (left)
        recovery_label = "recovery" if mode == "denoising" else "disruption"
        ax1.plot(positions, recoveries, linestyle=LINE_STYLES["recovery"],
                 color=METRIC_COLORS["recovery"], linewidth=2.5,
                 marker=MARKERS["recovery"], markersize=5, label=recovery_label)
        ax1.plot(positions, rr_shorts,
                 linestyle=LINE_STYLES["rr_short"], color=METRIC_COLORS["rr_short"],
                 linewidth=2, marker=MARKERS["rr_short"], markersize=4,
                 label="recip_rank(clean)")
        ylabel_left = "Recovery / RR" if mode == "denoising" else "Disruption / RR"
        ax1.set_ylabel(ylabel_left, fontsize=10, fontweight="bold")
        ax1.set_ylim(-0.1, 1.1)

        # Logit diff on secondary axis (right)
        ax1b = ax1.twinx()
        ax1b.plot(positions, logit_diffs, linestyle=LINE_STYLES["logit_diff"],
                  color=METRIC_COLORS["logit_diff"], linewidth=2,
                  marker=MARKERS["logit_diff"], markersize=4, label="logit_diff")
        ax1b.plot(positions, norm_logit_diffs, linestyle=LINE_STYLES["norm_logit_diff"],
                  color=METRIC_COLORS["norm_logit_diff"], linewidth=2,
                  marker=MARKERS["norm_logit_diff"], markersize=4, label="norm_logit_diff")
        ax1b.set_ylabel("Logit Diff", fontsize=10, color=METRIC_COLORS["logit_diff"])
        ax1b.tick_params(axis="y", labelcolor=METRIC_COLORS["logit_diff"])

        ax1.set_xlabel("Position", fontsize=10)
        ax1.set_title("Core", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        add_vlines(ax1)
        if row_idx == 1:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines1b, labels1b = ax1b.get_legend_handles_labels()
            ax1.legend(lines1 + lines1b, labels1 + labels1b,
                       loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=7, ncol=1, frameon=False)

        # ─── Column 2: Probs/Logprobs ───
        ax2 = axes[row_idx, 1]
        all_axes.append(ax2)
        logprob_shorts = [m.logprob_short for m in all_metrics]
        logprob_longs = [m.logprob_long for m in all_metrics]
        prob_shorts = [m.prob_short for m in all_metrics]
        prob_longs = [m.prob_long for m in all_metrics]

        # Prob on primary axis (left)
        ax2.plot(positions, prob_shorts, linestyle=LINE_STYLES["prob_short"],
                 color=METRIC_COLORS["prob_short"], linewidth=2,
                 marker=MARKERS["prob_short"], markersize=4, label="prob(short)")
        ax2.plot(positions, prob_longs, linestyle=LINE_STYLES["prob_long"],
                 color=METRIC_COLORS["prob_long"], linewidth=2,
                 marker=MARKERS["prob_long"], markersize=4, label="prob(long)")
        ax2.set_ylabel("Probability", fontsize=10, fontweight="bold")
        ax2.set_ylim(-0.05, 1.05)

        # Logprob on secondary axis (right)
        ax2b = ax2.twinx()
        ax2b.plot(positions, logprob_shorts, linestyle=LINE_STYLES["logprob_short"],
                  color=METRIC_COLORS["logprob_short"], linewidth=1.5,
                  marker=MARKERS["logprob_short"], markersize=3, label="logprob(short)")
        ax2b.plot(positions, logprob_longs, linestyle=LINE_STYLES["logprob_long"],
                  color=METRIC_COLORS["logprob_long"], linewidth=1.5,
                  marker=MARKERS["logprob_long"], markersize=3, label="logprob(long)")
        ax2b.set_ylabel("Logprob", fontsize=10)

        ax2.set_xlabel("Position", fontsize=10)
        ax2.set_title("Probs/Logprobs", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        add_vlines(ax2)
        if row_idx == 1:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines2b, labels2b = ax2b.get_legend_handles_labels()
            ax2.legend(lines2 + lines2b, labels2 + labels2b,
                       loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=7, ncol=1, frameon=False)

        # ─── Column 3: Logits ───
        ax3 = axes[row_idx, 2]
        all_axes.append(ax3)
        logit_shorts = [m.logit_short for m in all_metrics]
        logit_longs = [m.logit_long for m in all_metrics]
        norm_logit_shorts = [m.norm_logit_short for m in all_metrics]
        norm_logit_longs = [m.norm_logit_long for m in all_metrics]

        # Check if logits have valid data
        has_valid_logits = any(v != 0.0 for v in logit_shorts + logit_longs)

        if has_valid_logits:
            ax3.plot(positions, logit_shorts, linestyle=LINE_STYLES["logit_short"],
                     color=METRIC_COLORS["logit_short"], linewidth=2,
                     marker=MARKERS["logit_short"], markersize=4, label="logit(short)")
            ax3.plot(positions, logit_longs, linestyle=LINE_STYLES["logit_long"],
                     color=METRIC_COLORS["logit_long"], linewidth=2,
                     marker=MARKERS["logit_long"], markersize=4, label="logit(long)")

            ax3b = ax3.twinx()
            ax3b.plot(positions, norm_logit_shorts, linestyle=LINE_STYLES["norm_logit_short"],
                      color=METRIC_COLORS["norm_logit_short"], linewidth=1.5,
                      marker=MARKERS["norm_logit_short"], markersize=3, label="norm_logit(short)")
            ax3b.plot(positions, norm_logit_longs, linestyle=LINE_STYLES["norm_logit_long"],
                      color=METRIC_COLORS["norm_logit_long"], linewidth=1.5,
                      marker=MARKERS["norm_logit_long"], markersize=3, label="norm_logit(long)")
            ax3b.set_ylabel("Z-score", fontsize=9)

            ax3.set_ylabel("Raw Logit", fontsize=10)
        else:
            ax3.text(0.5, 0.5, "No logit data", ha="center", va="center", transform=ax3.transAxes)

        ax3.set_xlabel("Position", fontsize=10)
        ax3.set_title("Logits", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        add_vlines(ax3)
        if row_idx == 1 and has_valid_logits:
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines3b, labels3b = ax3b.get_legend_handles_labels()
            ax3.legend(lines3 + lines3b, labels3 + labels3b,
                       loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=7, ncol=1, frameon=False)

        # ─── Column 4: Fork metrics (dual y-axes) ───
        ax4 = axes[row_idx, 3]
        all_axes.append(ax4)
        fork_entropies = [m.fork_entropy for m in all_metrics]
        fork_divs = [m.fork_diversity for m in all_metrics]
        fork_simpsons = [m.fork_simpson for m in all_metrics]

        # Entropy on left axis
        ax4.plot(positions, fork_entropies, linestyle=LINE_STYLES["fork_entropy"],
                 color=METRIC_COLORS["fork_entropy"], linewidth=2,
                 marker=MARKERS["fork_entropy"], markersize=4, label="entropy")
        ax4.set_ylabel("Entropy (bits)", fontsize=9, color=METRIC_COLORS["fork_entropy"])
        ax4.tick_params(axis="y", labelcolor=METRIC_COLORS["fork_entropy"])
        ax4.set_ylim(-0.05, 1.1)

        # Diversity on right axis
        ax4b = ax4.twinx()
        ax4b.plot(positions, fork_divs, linestyle=LINE_STYLES["fork_diversity"],
                  color=METRIC_COLORS["fork_diversity"], linewidth=2,
                  marker=MARKERS["fork_diversity"], markersize=5, label="diversity")
        ax4b.plot(positions, fork_simpsons, linestyle=LINE_STYLES["fork_simpson"],
                  color=METRIC_COLORS["fork_simpson"], linewidth=2,
                  marker=MARKERS["fork_simpson"], markersize=4, label="simpson")
        ax4b.set_ylabel("Effective # (diversity)", fontsize=9, color=METRIC_COLORS["fork_diversity"])
        ax4b.tick_params(axis="y", labelcolor=METRIC_COLORS["fork_diversity"])
        ax4b.set_ylim(0.9, 2.1)

        ax4.set_xlabel("Position", fontsize=10)
        ax4.set_title("Fork", fontsize=11, fontweight="bold")
        ax4.grid(True, alpha=0.3)
        add_vlines(ax4)
        if row_idx == 1:
            lines4, labels4 = ax4.get_legend_handles_labels()
            lines4b, labels4b = ax4b.get_legend_handles_labels()
            ax4.legend(lines4 + lines4b, labels4 + labels4b,
                       loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=7, ncol=1, frameon=False)

        # ─── Column 5: Vocab metrics (triple y-axes: entropy, diversity, TCB) ───
        # ─── Column 5: Vocab metrics (dual y-axes: entropy vs diversity) ───
        ax5 = axes[row_idx, 4]
        all_axes.append(ax5)
        vocab_entropies = [m.vocab_entropy for m in all_metrics]
        vocab_divs = [m.vocab_diversity for m in all_metrics]
        vocab_simpsons = [m.vocab_simpson for m in all_metrics]

        # Entropy on left axis
        ax5.plot(positions, vocab_entropies, linestyle=LINE_STYLES["vocab_entropy"],
                 color=METRIC_COLORS["vocab_entropy"], linewidth=2,
                 marker=MARKERS["vocab_entropy"], markersize=5, label="entropy")
        ax5.set_ylabel("Entropy", fontsize=9, color=METRIC_COLORS["vocab_entropy"])
        ax5.tick_params(axis="y", labelcolor=METRIC_COLORS["vocab_entropy"])

        # Diversity on right axis
        ax5b = ax5.twinx()
        ax5b.plot(positions, vocab_divs, linestyle=LINE_STYLES["vocab_diversity"],
                  color=METRIC_COLORS["vocab_diversity"], linewidth=1.5,
                  marker=MARKERS["vocab_diversity"], markersize=3, label="diversity")
        ax5b.plot(positions, vocab_simpsons, linestyle=LINE_STYLES["vocab_simpson"],
                  color=METRIC_COLORS["vocab_simpson"], linewidth=1.5,
                  marker=MARKERS["vocab_simpson"], markersize=3, label="simpson")
        ax5b.set_ylabel("Effective #", fontsize=9, color=METRIC_COLORS["vocab_diversity"])
        ax5b.tick_params(axis="y", labelcolor=METRIC_COLORS["vocab_diversity"])

        ax5.set_xlabel("Position", fontsize=10)
        ax5.set_title("Vocab", fontsize=11, fontweight="bold")
        ax5.grid(True, alpha=0.3)
        add_vlines(ax5)
        if row_idx == 1:
            lines5, labels5 = ax5.get_legend_handles_labels()
            lines5b, labels5b = ax5b.get_legend_handles_labels()
            ax5.legend(lines5 + lines5b, labels5 + labels5b,
                       loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=7, ncol=1, frameon=False)

        # ─── Column 6: Trajectory + TCB (inv_perplexity on left, TCB on right) ───
        ax6 = axes[row_idx, 5]
        all_axes.append(ax6)
        inv_perp_shorts = [m.traj_inv_perplexity_short for m in all_metrics]
        inv_perp_longs = [m.traj_inv_perplexity_long for m in all_metrics]
        vocab_tcbs = [m.vocab_tcb for m in all_metrics]

        # Inverse perplexity on left axis
        ax6.plot(positions, inv_perp_shorts, linestyle=LINE_STYLES["inv_perplexity_short"],
                 color=METRIC_COLORS["inv_perplexity_short"], linewidth=LINE_WIDTHS["inv_perplexity_short"],
                 marker=MARKERS["inv_perplexity_short"], markersize=MARKER_SIZES["inv_perplexity_short"],
                 label="inv_ppl(short)")
        ax6.plot(positions, inv_perp_longs, linestyle=LINE_STYLES["inv_perplexity_long"],
                 color=METRIC_COLORS["inv_perplexity_long"], linewidth=LINE_WIDTHS["inv_perplexity_long"],
                 marker=MARKERS["inv_perplexity_long"], markersize=MARKER_SIZES["inv_perplexity_long"],
                 label="inv_ppl(long)")
        ax6.set_ylabel("Inv Perplexity", fontsize=9)
        # Auto-scale y-axis for small inv_perplexity values

        # TCB on right axis
        ax6b = ax6.twinx()
        ax6b.plot(positions, vocab_tcbs,
                  linestyle=":", color=METRIC_COLORS["vocab_tcb"],
                  linewidth=1.5, marker=MARKERS["vocab_tcb"], markersize=4,
                  markerfacecolor=METRIC_COLORS["vocab_tcb"], label="TCB")
        ax6b.set_ylabel("TCB", fontsize=9, color=METRIC_COLORS["vocab_tcb"])
        ax6b.tick_params(axis="y", labelcolor=METRIC_COLORS["vocab_tcb"])

        ax6.set_xlabel("Position", fontsize=10)
        ax6.set_title("Trajectory + TCB", fontsize=11, fontweight="bold")
        ax6.grid(True, alpha=0.3)
        add_vlines(ax6)
        if row_idx == 1:
            lines6, labels6 = ax6.get_legend_handles_labels()
            lines6b, labels6b = ax6b.get_legend_handles_labels()
            ax6.legend(lines6 + lines6b, labels6 + labels6b,
                       loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=7, ncol=1, frameon=False)

    # Adjust layout for token type legend at bottom
    plt.tight_layout(rect=[0.03, 0.06, 1, 0.95])

    # Add token type legend at bottom
    _add_token_type_legend(fig)

    # Apply colored tick labels to all axes and save
    _save_with_colored_ticks_multi(
        fig,
        all_axes,
        tick_positions,
        coloring,
        output_dir / f"coarse_position_sweep_{clean_traj}_{step_size}.png",
    )


def _plot_denoising_vs_noising_comparison(
    layer_data: dict[int, ActPatchTargetResult],
    position_data: dict[int, ActPatchTargetResult],
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
    step_size: int = 8,
) -> None:
    """Plot denoising vs noising comparison scatter plots."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="white")
    fig.suptitle(
        f"Activation Patching: Denoising vs Noising Recovery (step size={step_size})",
        fontsize=20, fontweight="bold"
    )

    # Layer comparison
    ax1 = axes[0]
    ax1.set_facecolor("white")
    if layer_data:
        layers = sorted(layer_data.keys())
        d_recoveries = []
        n_recoveries = []
        for layer in layers:
            lr = layer_data[layer]
            d_recoveries.append(lr.denoising.recovery if lr.denoising else 0)
            n_recoveries.append(lr.noising.recovery if lr.noising else 0)

        # Scatter plot with viridis colormap for layers
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

        # Find extreme points to label (top 5 in each metric)
        d_sorted_idx = np.argsort(d_recoveries)[::-1][:5]  # Top 5 denoising
        n_sorted_idx = np.argsort(n_recoveries)[::-1][:5]  # Top 5 noising
        label_indices = set(d_sorted_idx) | set(n_sorted_idx)
        label_indices.add(0)  # First layer
        label_indices.add(len(layers) - 1)  # Last layer

        for i, (d, n, layer) in enumerate(zip(d_recoveries, n_recoveries, layers)):
            ax1.scatter(
                d, n, c=[colors[i]], s=200, edgecolors="black", linewidth=1.5,
                alpha=0.85, zorder=3
            )
            # Only label extreme points or first/last layer
            if i in label_indices:
                ax1.annotate(
                    f"L{layer}",
                    (d, n),
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    xytext=(0, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
                )

        ax1.plot([0, 1], [0, 1], "k--", alpha=0.7, linewidth=2.5, label="Equal effect (y=x)")
        ax1.set_xlabel("Recovery from Denoising Patch", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Recovery from Noising Patch", fontsize=14, fontweight="bold")
        ax1.set_title("Layer Sweep: Patch Effect Comparison", fontsize=16, fontweight="bold")
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.5, linewidth=1.2, color="#CCCCCC")
        ax1.axhline(y=0.5, color="#888888", linestyle=":", alpha=0.8, linewidth=2, label="Baseline (50%)")
        ax1.axvline(x=0.5, color="#888888", linestyle=":", alpha=0.8, linewidth=2)
        ax1.tick_params(axis="both", labelsize=12)
        ax1.legend(loc="lower right", fontsize=11, frameon=True, fancybox=True,
                   title="Reference Lines", title_fontsize=10)
    else:
        ax1.text(0.5, 0.5, "No layer results", ha="center", va="center", fontsize=14)
        ax1.axis("off")

    # Position comparison
    ax2 = axes[1]
    ax2.set_facecolor("white")
    if position_data:
        positions = sorted(position_data.keys())
        d_recoveries = []
        n_recoveries = []
        point_colors = []

        for pos in positions:
            pr = position_data[pos]
            d_recoveries.append(pr.denoising.recovery if pr.denoising else 0)
            n_recoveries.append(pr.noising.recovery if pr.noising else 0)
            point_colors.append(_get_tick_color(pos, coloring))

        # Find positions with highest recovery (top 3 in each metric to reduce clutter)
        d_arr = np.array(d_recoveries)
        n_arr = np.array(n_recoveries)
        d_sorted_idx = np.argsort(d_arr)[::-1][:3]  # Top 3 denoising
        n_sorted_idx = np.argsort(n_arr)[::-1][:3]  # Top 3 noising
        label_indices = set(d_sorted_idx) | set(n_sorted_idx)

        for i, (d, n, pos) in enumerate(zip(d_recoveries, n_recoveries, positions)):
            ax2.scatter(
                d,
                n,
                c=[point_colors[i]],
                s=200,
                edgecolors="black",
                linewidth=1.5,
                alpha=0.75,
                zorder=3,
            )
            # Only label positions with highest recovery
            if i in label_indices:
                ax2.annotate(
                    f"P{pos}",
                    (d, n),
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    xytext=(0, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
                )

        ax2.plot([0, 1], [0, 1], "k--", alpha=0.7, linewidth=2.5, label="Equal effect (y=x)")
        ax2.set_xlabel("Recovery from Denoising Patch", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Recovery from Noising Patch", fontsize=14, fontweight="bold")
        ax2.set_title("Position Sweep: Patch Effect Comparison", fontsize=16, fontweight="bold")
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.5, linewidth=1.2, color="#CCCCCC")
        ax2.axhline(y=0.5, color="#888888", linestyle=":", alpha=0.8, linewidth=2, label="Baseline (50%)")
        ax2.axvline(x=0.5, color="#888888", linestyle=":", alpha=0.8, linewidth=2)
        ax2.tick_params(axis="both", labelsize=12)
        ax2.legend(loc="lower right", fontsize=11, frameon=True, fancybox=True,
                   title="Reference Lines", title_fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No position results", ha="center", va="center", fontsize=14)
        ax2.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = output_dir / f"denoising_vs_noising_{step_size}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _plot_sanity_check(
    result: CoarseActPatchResults,
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
    pair: ContrastivePair | None = None,
) -> None:
    """Plot sanity check results ONLY.

    The sanity check patches ALL layers + ALL positions at once.
    This is a single data point that validates the patching works.

    Shows ONLY sanity check data:
    - Greedy generation results (before/after patching)
    - Probability metrics from sanity check
    - Recovery scores from sanity check
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

    # ─── Panel 1: Greedy Generation Results ───
    ax1 = axes[0, 0]
    ax1.axis("off")

    lines = ["GREEDY GENERATION RESULTS", "=" * 50, ""]
    lines.append("Patching ALL layers + ALL positions at once:")
    lines.append("")

    d_metrics = None
    n_metrics = None

    if sanity.denoising:
        d = sanity.denoising
        d_metrics = IntervenedChoiceMetrics.from_choice(d)

        # Original choice (before patching)
        orig_label = "?"
        orig_response = ""
        if d.original:
            try:
                orig_label = (
                    d.original.labels[d.original.choice_idx]
                    if d.original.labels
                    else "?"
                )
                if d.original.response_texts:
                    orig_response = d.original.response_texts[
                        d.original.choice_idx
                    ][:60]
            except Exception:
                pass

        # Intervened choice (after patching)
        intv_label = "?"
        if d.intervened:
            try:
                intv_label = (
                    d.intervened.labels[d.intervened.choice_idx]
                    if d.intervened.labels
                    else "?"
                )
            except Exception:
                pass

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
        n_metrics = IntervenedChoiceMetrics.from_choice(n)

        orig_label = "?"
        orig_response = ""
        intv_label = "?"
        if n.original:
            try:
                orig_label = (
                    n.original.labels[n.original.choice_idx]
                    if n.original.labels
                    else "?"
                )
                if n.original.response_texts:
                    orig_response = n.original.response_texts[
                        n.original.choice_idx
                    ][:60]
            except Exception:
                pass
        if n.intervened:
            try:
                intv_label = (
                    n.intervened.labels[n.intervened.choice_idx]
                    if n.intervened.labels
                    else "?"
                )
            except Exception:
                pass

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
    lines.append("")
    lines.append("-" * 45)
    lines.append(f"Combined Sanity Score: {sanity.score():.4f}")
    if sanity.flip_count > 0:
        lines.append(f"Flips: {sanity.flip_count}/2")

    ax1.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax1.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # ─── Panel 2: Probability Bar Chart (sanity only) ───
    ax2 = axes[0, 1]
    if sanity.denoising or sanity.noising:
        metrics_names = ["prob(short)", "prob(long)", "fork_div", "recovery"]
        x = np.arange(len(metrics_names))
        width = 0.35

        d_vals = [0, 0, 0, 0]
        n_vals = [0, 0, 0, 0]

        if d_metrics:
            d_vals = [
                d_metrics.prob_short,
                d_metrics.prob_long,
                d_metrics.fork_diversity,
                d_metrics.recovery,
            ]

        if n_metrics:
            n_vals = [
                n_metrics.prob_short,
                n_metrics.prob_long,
                n_metrics.fork_diversity,
                n_metrics.recovery,
            ]

        ax2.bar(x - width / 2, d_vals, width, label="Denoising", color=BAR_COLORS["denoising"])
        ax2.bar(x + width / 2, n_vals, width, label="Noising", color=BAR_COLORS["noising"])
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_names, fontsize=10)
        ax2.set_ylabel("Value", fontsize=11)
        ax2.set_title("Sanity Check Metrics", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "No sanity data", ha="center", va="center")
        ax2.axis("off")

    # ─── Panel 3: Per-position logprob difference (short vs long traj) ───
    ax3 = axes[1, 0]

    if pair is not None:
        # Get logprobs from both trajectories
        short_logprobs = pair.short_traj.logprobs
        long_logprobs = pair.long_traj.logprobs
        position_mapping = pair.position_mapping

        # Compute per-position difference using position mapping
        positions = []
        logprob_diffs = []

        for src_pos in range(len(short_logprobs)):
            dst_pos = position_mapping.get(src_pos, src_pos)
            if dst_pos is not None and dst_pos < len(long_logprobs):
                positions.append(src_pos)
                diff = short_logprobs[src_pos] - long_logprobs[dst_pos]
                logprob_diffs.append(diff)

        if positions:
            ax3.plot(positions, logprob_diffs, "b-", linewidth=1.5, alpha=0.8)
            ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax3.set_xlabel("Position", fontsize=11)
            ax3.set_ylabel("logprob(short) - logprob(long)", fontsize=11)
            ax3.set_title("Per-Position Logprob Diff (short - long traj)", fontsize=12, fontweight="bold")
            ax3.grid(True, alpha=0.3)

            # Add summary stats
            mean_diff = np.mean(logprob_diffs)
            max_diff = np.max(np.abs(logprob_diffs))
            ax3.text(
                0.02, 0.98,
                f"mean={mean_diff:.4f}\nmax|diff|={max_diff:.4f}",
                transform=ax3.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            ax3.text(0.5, 0.5, "No position data", ha="center", va="center")
            ax3.axis("off")
    else:
        ax3.text(0.5, 0.5, "No pair data for per-position plot", ha="center", va="center")
        ax3.axis("off")

    # ─── Panel 4: Reciprocal rank comparison (sanity only) ───
    ax4 = axes[1, 1]
    if d_metrics or n_metrics:
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

        ax4.bar(x - width / 2, d_rr, width, label="Denoising", color=BAR_COLORS["denoising"])
        ax4.bar(x + width / 2, n_rr, width, label="Noising", color=BAR_COLORS["noising"])
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, fontsize=10)
        ax4.set_ylabel("Reciprocal Rank (1/rank)", fontsize=11)
        ax4.set_title("Sanity Check: Reciprocal Ranks", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3, axis="y")
    else:
        ax4.text(0.5, 0.5, "No sanity data", ha="center", va="center")
        ax4.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _finalize_plot(fig, output_dir / "sanity_check.png")


def _visualize_aggregated_coarse(
    result: CoarseActPatchAggregatedResults,
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
) -> None:
    """Visualize aggregated coarse patching results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a plot for each step size combination
    all_step_sizes = set(result.layer_step_sizes) | set(result.position_step_sizes)

    for step_size in sorted(all_step_sizes):
        layer_scores = result.get_mean_layer_scores(step_size=step_size)
        pos_scores = result.get_mean_position_scores(step_size=step_size)

        if not layer_scores and not pos_scores:
            continue

        # Create combined figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Aggregated Coarse Patching (step={step_size}, {result.n_samples} samples)",
            fontsize=14,
            fontweight="bold",
        )

        # Layer sweep
        ax1 = axes[0]
        if layer_scores:
            layers = sorted(layer_scores.keys())
            recoveries = [layer_scores[l] for l in layers]

            ax1.plot(layers, recoveries, "b-", linewidth=2, marker="o", markersize=6)
            ax1.set_xlabel("Layer", fontsize=11)
            ax1.set_ylabel("Mean Recovery", fontsize=11)
            ax1.set_title("Layer Sweep", fontsize=11)
            ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            ax1.set_ylim(-0.1, 1.1)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No layer results", ha="center", va="center")
            ax1.axis("off")

        # Position sweep
        ax2 = axes[1]
        if pos_scores:
            positions = sorted(pos_scores.keys())
            recoveries = [pos_scores[p] for p in positions]

            ax2.plot(positions, recoveries, "b-", linewidth=2, marker="o", markersize=6)
            ax2.set_xlabel("Position", fontsize=11)
            ax2.set_ylabel("Mean Recovery", fontsize=11)
            ax2.set_title("Position Sweep", fontsize=11)
            ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)

            # Color x-axis tick labels
            _color_xaxis_ticks(ax2, positions, coloring)
        else:
            ax2.text(0.5, 0.5, "No position results", ha="center", va="center")
            ax2.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        _finalize_plot(fig, output_dir / f"coarse_patching_agg_{step_size}.png")

    print(f"[viz] Aggregated coarse patching plots saved to {output_dir}")
