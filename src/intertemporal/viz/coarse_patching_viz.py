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
from ...viz.palettes import LINE_COLORS, BAR_COLORS, TOKEN_COLORS
from ...activation_patching.coarse_activation_patching import (
    CoarseActPatchResults,
    CoarseActPatchAggregatedResults,
)
from ...viz.token_coloring import PairTokenColoring
from ...common.contrastive_pair import ContrastivePair


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


def _get_tick_spacing(n_points: int) -> int:
    """Return tick spacing to keep x-axis legible."""
    if n_points <= 15:
        return 1
    elif n_points <= 30:
        return 2
    elif n_points <= 60:
        return 5
    return 10


def _plot_layer_sweep_single(
    layer_data: dict[int, ActPatchTargetResult],
    output_dir: Path,
    step_size: int,
    clean_traj: Literal["short", "long"],
) -> None:
    """Plot layer sweep with both denoising and noising, from one trajectory's perspective.

    Creates 2x3 subplots:
    - Top row: Denoising
    - Bottom row: Noising
    Each row has: Logit diff & logprobs | Probabilities & recovery/disruption | Ranks & diversity

    Args:
        clean_traj: Which trajectory is considered "clean" for metric naming.
            - "short": denoising shows recovery, noising shows disruption
            - "long": denoising shows disruption, noising shows recovery
    """
    layers = sorted(layer_data.keys())
    if not layers:
        return

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Coarse Layer Sweep, Clean = {clean_traj}, Steps = {step_size}",
        fontsize=16,
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

        # Metric name: recovery for denoising, disruption for noising
        faith_metric = "recovery" if mode == "denoising" else "disruption"

        # Row subtitle (simple labels)
        subtitle = "Denoising" if mode == "denoising" else "Noising"

        # ─── Panel 1: Logit diff and logprobs ───
        ax1 = axes[row_idx, 0]
        logit_diffs = [m.logit_diff for m in all_metrics]
        logprob_shorts = [m.logprob_short for m in all_metrics]
        logprob_longs = [m.logprob_long for m in all_metrics]

        ax1.plot(layers, logit_diffs, "-", color=LINE_COLORS["logit_diff"],
                 linewidth=2.5, marker="o", markersize=5, label="logit_diff")
        ax1.plot(layers, logprob_shorts, "-", color=LINE_COLORS["logprob_short"],
                 linewidth=2, marker="s", markersize=4, label="logprob(short)")
        ax1.plot(layers, logprob_longs, "-", color=LINE_COLORS["logprob_long"],
                 linewidth=2, marker="^", markersize=4, label="logprob(long)")

        ax1.set_xlabel("Layer", fontsize=10)
        ax1.set_ylabel("Logit / Logprob", fontsize=10)
        ax1.set_title(f"{subtitle}\nLogit Diff & Logprobs", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax1.set_xticks(tick_positions)
        if row_idx == 1:
            ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.15), fontsize=8, ncol=3, frameon=False)

        # ─── Panel 2: Probabilities and faithfulness metric ───
        ax2 = axes[row_idx, 1]
        prob_shorts = [m.prob_short for m in all_metrics]
        prob_longs = [m.prob_long for m in all_metrics]
        recoveries = [m.recovery for m in all_metrics]
        # Compute faith_values based on metric type
        if faith_metric == "recovery":
            faith_values = recoveries
        else:  # disruption
            faith_values = [1 - r for r in recoveries]

        ax2.plot(layers, prob_shorts, "-", color=LINE_COLORS["prob_short"],
                 linewidth=2, marker="^", markersize=5, label="prob(short)")
        ax2.plot(layers, prob_longs, "-", color=LINE_COLORS["prob_long"],
                 linewidth=2, marker="v", markersize=5, label="prob(long)")
        ax2.plot(layers, faith_values, "-", color=LINE_COLORS["recovery"],
                 linewidth=2.5, marker="D", markersize=5, label=faith_metric)

        ax2.set_xlabel("Layer", fontsize=10)
        ax2.set_ylabel(f"Probability / {faith_metric.title()}", fontsize=10)
        ax2.set_title(f"{subtitle}\nProbs & {faith_metric.title()}", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xticks(tick_positions)
        if row_idx == 1:
            ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.15), fontsize=8, ncol=3, frameon=False)

        # ─── Panel 3: Ranks and diversity ───
        ax3 = axes[row_idx, 2]
        rr_shorts = [m.reciprocal_rank_short for m in all_metrics]
        rr_longs = [m.reciprocal_rank_long for m in all_metrics]
        fork_divs = [m.fork_diversity for m in all_metrics]
        vocab_entropies = [m.vocab_entropy for m in all_metrics]

        ax3.plot(layers, rr_shorts, "-", color=LINE_COLORS["rr_short"],
                 linewidth=2, marker="<", markersize=4, label="recip_rank(short)")
        ax3.plot(layers, rr_longs, "-", color=LINE_COLORS["rr_long"],
                 linewidth=2, marker=">", markersize=4, label="recip_rank(long)")
        ax3.plot(layers, fork_divs, "-", color=LINE_COLORS["fork_div"],
                 linewidth=2.5, marker="p", markersize=5, label="fork_div")

        # Entropy on secondary axis
        ax3b = ax3.twinx()
        ax3b.plot(layers, vocab_entropies, "-", color=LINE_COLORS["vocab_entropy"],
                  linewidth=2, marker="h", markersize=5, label="vocab_entropy")
        ax3b.set_ylabel("Entropy", fontsize=10, color=LINE_COLORS["vocab_entropy"])
        ax3b.tick_params(axis="y", labelcolor=LINE_COLORS["vocab_entropy"])
        # Prevent misleading y-axis when values are essentially constant
        ent_range = max(vocab_entropies) - min(vocab_entropies) if vocab_entropies else 0
        if ent_range < 1e-6:
            ent_mean = np.mean(vocab_entropies) if vocab_entropies else 0
            ax3b.set_ylim(ent_mean - 0.5, ent_mean + 0.5)

        ax3.set_xlabel("Layer", fontsize=10)
        ax3.set_ylabel("Rank / Diversity", fontsize=10)
        ax3.set_title(f"{subtitle}\nRanks & Diversity", fontsize=11, fontweight="bold")
        ax3.set_ylim(-0.05, 2.05)
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(tick_positions)
        if row_idx == 1:
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines3b, labels3b = ax3b.get_legend_handles_labels()
            ax3.legend(lines3 + lines3b, labels3 + labels3b,
                       loc="upper left", bbox_to_anchor=(0, -0.15), fontsize=8, ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    _finalize_plot(output_dir / f"coarse_layer_sweep_{clean_traj}_{step_size}.png")


def _plot_position_sweep_single(
    position_data: dict[int, ActPatchTargetResult],
    output_dir: Path,
    step_size: int,
    clean_traj: Literal["short", "long"],
    coloring: PairTokenColoring | None = None,
) -> None:
    """Plot position sweep with both denoising and noising, from one trajectory's perspective.

    Creates 2x3 subplots:
    - Top row: Denoising
    - Bottom row: Noising
    Each row has: Logit diff & logprobs | Probabilities & recovery/disruption | Ranks & diversity

    X-axis tick labels are colored by token type.

    Args:
        clean_traj: Which trajectory is considered "clean" for metric naming.
            - "short": denoising shows recovery, noising shows disruption
            - "long": denoising shows disruption, noising shows recovery
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

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"Coarse Position Sweep, Clean = {clean_traj}, Steps = {step_size}",
        fontsize=16,
        fontweight="bold",
    )

    all_axes = []  # Collect all axes for tick coloring

    # Apply tick spacing
    tick_spacing = _get_tick_spacing(len(positions))
    tick_positions = positions[::tick_spacing]

    for row_idx, mode in enumerate(["denoising", "noising"]):
        # Extract metrics for each position
        all_metrics = []
        for pos in positions:
            target_result = position_data[pos]
            choice = (
                target_result.denoising if mode == "denoising" else target_result.noising
            )
            all_metrics.append(IntervenedChoiceMetrics.from_choice(choice))

        # Metric name: recovery for denoising, disruption for noising
        faith_metric = "recovery" if mode == "denoising" else "disruption"

        # Row subtitle (simple labels)
        subtitle = "Denoising" if mode == "denoising" else "Noising"

        # ─── Panel 1: Logit diff and logprobs ───
        ax1 = axes[row_idx, 0]
        all_axes.append(ax1)
        logit_diffs = [m.logit_diff for m in all_metrics]
        logprob_shorts = [m.logprob_short for m in all_metrics]
        logprob_longs = [m.logprob_long for m in all_metrics]

        ax1.plot(positions, logit_diffs, "-", color=LINE_COLORS["logit_diff"],
                 linewidth=2.5, marker="o", markersize=5, label="logit_diff")
        ax1.plot(positions, logprob_shorts, "-", color=LINE_COLORS["logprob_short"],
                 linewidth=2, marker="s", markersize=4, label="logprob(short)")
        ax1.plot(positions, logprob_longs, "-", color=LINE_COLORS["logprob_long"],
                 linewidth=2, marker="^", markersize=4, label="logprob(long)")

        ax1.set_xlabel("Position", fontsize=10)
        ax1.set_ylabel("Logit / Logprob", fontsize=10)
        ax1.set_title(f"{subtitle}\nLogit Diff & Logprobs", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        if prompt_boundary and min(positions) < prompt_boundary < max(positions):
            ax1.axvline(x=prompt_boundary, color="red", linestyle="--", alpha=0.7)
        if choice_div_pos and min(positions) <= choice_div_pos <= max(positions):
            ax1.axvline(x=choice_div_pos, color=LINE_COLORS["fork_div"], linestyle=":", linewidth=2, alpha=0.8)
        if row_idx == 1:
            ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=8, ncol=3, frameon=False)

        # ─── Panel 2: Probabilities and faithfulness metric ───
        ax2 = axes[row_idx, 1]
        all_axes.append(ax2)
        prob_shorts = [m.prob_short for m in all_metrics]
        prob_longs = [m.prob_long for m in all_metrics]
        recoveries = [m.recovery for m in all_metrics]
        # Compute faith_values based on metric type
        if faith_metric == "recovery":
            faith_values = recoveries
        else:  # disruption
            faith_values = [1 - r for r in recoveries]

        ax2.plot(positions, prob_shorts, "-", color=LINE_COLORS["prob_short"],
                 linewidth=2, marker="^", markersize=5, label="prob(short)")
        ax2.plot(positions, prob_longs, "-", color=LINE_COLORS["prob_long"],
                 linewidth=2, marker="v", markersize=5, label="prob(long)")
        ax2.plot(positions, faith_values, "-", color=LINE_COLORS["recovery"],
                 linewidth=2.5, marker="D", markersize=5, label=faith_metric)

        ax2.set_xlabel("Position", fontsize=10)
        ax2.set_ylabel(f"Probability / {faith_metric.title()}", fontsize=10)
        ax2.set_title(f"{subtitle}\nProbs & {faith_metric.title()}", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        if prompt_boundary and min(positions) < prompt_boundary < max(positions):
            ax2.axvline(x=prompt_boundary, color="red", linestyle="--", alpha=0.7)
        if choice_div_pos and min(positions) <= choice_div_pos <= max(positions):
            ax2.axvline(x=choice_div_pos, color=LINE_COLORS["fork_div"], linestyle=":", linewidth=2, alpha=0.8)
        if row_idx == 1:
            ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=8, ncol=3, frameon=False)

        # ─── Panel 3: Ranks and diversity ───
        ax3 = axes[row_idx, 2]
        all_axes.append(ax3)
        rr_shorts = [m.reciprocal_rank_short for m in all_metrics]
        rr_longs = [m.reciprocal_rank_long for m in all_metrics]
        fork_divs = [m.fork_diversity for m in all_metrics]
        vocab_entropies = [m.vocab_entropy for m in all_metrics]

        ax3.plot(positions, rr_shorts, "-", color=LINE_COLORS["rr_short"],
                 linewidth=2, marker="<", markersize=4, label="recip_rank(short)")
        ax3.plot(positions, rr_longs, "-", color=LINE_COLORS["rr_long"],
                 linewidth=2, marker=">", markersize=4, label="recip_rank(long)")
        ax3.plot(positions, fork_divs, "-", color=LINE_COLORS["fork_div"],
                 linewidth=2, marker="p", markersize=5, label="fork_div")

        # Entropy on secondary axis
        ax3b = ax3.twinx()
        ax3b.plot(positions, vocab_entropies, "-", color=LINE_COLORS["vocab_entropy"],
                  linewidth=2, marker="h", markersize=5, label="vocab_entropy")
        ax3b.set_ylabel("Entropy", fontsize=10, color=LINE_COLORS["vocab_entropy"])
        ax3b.tick_params(axis="y", labelcolor=LINE_COLORS["vocab_entropy"])
        # Prevent misleading y-axis when values are essentially constant
        ent_range = max(vocab_entropies) - min(vocab_entropies) if vocab_entropies else 0
        if ent_range < 1e-6:
            ent_mean = np.mean(vocab_entropies) if vocab_entropies else 0
            ax3b.set_ylim(ent_mean - 0.5, ent_mean + 0.5)

        ax3.set_xlabel("Position", fontsize=10)
        ax3.set_ylabel("Rank / Diversity", fontsize=10)
        ax3.set_title(f"{subtitle}\nRanks & Diversity", fontsize=11, fontweight="bold")
        ax3.set_ylim(-0.05, 2.05)
        ax3.grid(True, alpha=0.3)
        if prompt_boundary and min(positions) < prompt_boundary < max(positions):
            ax3.axvline(x=prompt_boundary, color="red", linestyle="--", alpha=0.7)
        if choice_div_pos and min(positions) <= choice_div_pos <= max(positions):
            ax3.axvline(x=choice_div_pos, color=LINE_COLORS["fork_div"], linestyle=":", linewidth=2, alpha=0.8)
        if row_idx == 1:
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines3b, labels3b = ax3b.get_legend_handles_labels()
            ax3.legend(lines3 + lines3b, labels3 + labels3b,
                       loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=8, ncol=4, frameon=False)

    # Adjust layout for token type legend at bottom
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

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


def _get_tick_color(pos: int, coloring: PairTokenColoring | None) -> str:
    """Get the color for a tick label at given position."""
    if coloring is None or not coloring.short_colors:
        return TOKEN_COLORS["response_edge"]  # Default

    color_info = coloring.short_colors.get(pos)
    if color_info is None:
        # Try to find nearest position
        for offset in range(20):
            if pos + offset in coloring.short_colors:
                color_info = coloring.short_colors[pos + offset]
                break
            if pos - offset in coloring.short_colors:
                color_info = coloring.short_colors[pos - offset]
                break

    return color_info.edgecolor if color_info else TOKEN_COLORS["response_edge"]


def _color_xaxis_ticks(
    ax: plt.Axes,
    positions: list[int],
    coloring: PairTokenColoring | None,
) -> None:
    """Color x-axis tick labels by token type (simple version for non-position-sweep plots)."""
    colors = [_get_tick_color(pos, coloring) for pos in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in positions], fontsize=10, fontweight="bold")
    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(color)


def _save_with_colored_ticks(
    fig: plt.Figure,
    ax: plt.Axes,
    positions: list[int],
    coloring: PairTokenColoring | None,
    save_path: Path,
) -> None:
    """Save figure with colored x-axis tick labels.

    Forces a canvas draw before setting colors to ensure tick labels exist.
    """
    # Set ticks first
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in positions], fontsize=11, fontweight="bold")

    # FORCE RENDER to ensure tick labels are fully created
    fig.canvas.draw()

    # Compute and apply colors AFTER draw
    colors = [_get_tick_color(pos, coloring) for pos in positions]
    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(color)

    # Save with white background
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _save_with_colored_ticks_multi(
    fig: plt.Figure,
    axes: list[plt.Axes],
    positions: list[int],
    coloring: PairTokenColoring | None,
    save_path: Path,
) -> None:
    """Save figure with colored x-axis tick labels on multiple axes.

    Forces a canvas draw before setting colors to ensure tick labels exist.
    """
    # Set ticks on all axes first
    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(p) for p in positions], fontsize=10, fontweight="bold")

    # FORCE RENDER to ensure tick labels are fully created
    fig.canvas.draw()

    # Compute colors
    colors = [_get_tick_color(pos, coloring) for pos in positions]

    # Apply colors to all axes AFTER draw
    for ax in axes:
        for label, color in zip(ax.get_xticklabels(), colors):
            label.set_color(color)

    # Save with white background
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _add_token_type_legend(fig: plt.Figure) -> None:
    """Add a small legend for token type colors at the bottom."""
    from matplotlib.lines import Line2D

    # Use line markers instead of patches to show edgecolor more clearly
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=TOKEN_COLORS["prompt"],
            markeredgecolor=TOKEN_COLORS["prompt_edge"],
            markersize=10,
            markeredgewidth=2,
            label="Prompt",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=TOKEN_COLORS["response"],
            markeredgecolor=TOKEN_COLORS["response_edge"],
            markersize=10,
            markeredgewidth=2,
            label="Response",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=TOKEN_COLORS["choice_div"],
            markeredgecolor=TOKEN_COLORS["choice_div_edge"],
            markersize=10,
            markeredgewidth=2,
            label="Choice Div",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=TOKEN_COLORS["contrast_div"],
            markeredgecolor=TOKEN_COLORS["contrast_div_edge"],
            markersize=10,
            markeredgewidth=2,
            label="Contrast Div",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=9,
        title="Tick colors by token type",
        title_fontsize=9,
        ncol=4,
        bbox_to_anchor=(0.42, -0.01),
        frameon=True,
        fancybox=True,
        shadow=False,
    )


def _plot_denoising_vs_noising_comparison(
    layer_data: dict[int, ActPatchTargetResult],
    position_data: dict[int, ActPatchTargetResult],
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
    step_size: int = 8,
) -> None:
    """Plot denoising vs noising comparison scatter plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Denoising vs Noising Comparison (step={step_size})", fontsize=14, fontweight="bold")

    # Layer comparison
    ax1 = axes[0]
    if layer_data:
        layers = sorted(layer_data.keys())
        d_recoveries = []
        n_recoveries = []
        for layer in layers:
            lr = layer_data[layer]
            d_recoveries.append(lr.denoising.recovery if lr.denoising else 0)
            n_recoveries.append(lr.noising.recovery if lr.noising else 0)

        # Scatter plot with layer color gradient
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(layers)))
        for i, (d, n, layer) in enumerate(zip(d_recoveries, n_recoveries, layers)):
            ax1.scatter(
                d, n, c=[colors[i]], s=100, edgecolors="white", linewidth=1, zorder=3
            )
            ax1.annotate(
                str(layer),
                (d, n),
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )

        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
        ax1.set_xlabel("Denoising Recovery", fontsize=11)
        ax1.set_ylabel("Noising Recovery", fontsize=11)
        ax1.set_title("Layer Sweep: Denoising vs Noising", fontsize=11)
        ax1.set_xlim(-0.1, 1.1)
        ax1.set_ylim(-0.1, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax1.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    else:
        ax1.text(0.5, 0.5, "No layer results", ha="center", va="center")
        ax1.axis("off")

    # Position comparison
    ax2 = axes[1]
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

        for i, (d, n, pos) in enumerate(zip(d_recoveries, n_recoveries, positions)):
            ax2.scatter(
                d,
                n,
                c=[point_colors[i]],
                s=100,
                edgecolors="white",
                linewidth=1,
                zorder=3,
            )
            ax2.annotate(
                str(pos),
                (d, n),
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )

        ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
        ax2.set_xlabel("Denoising Recovery", fontsize=11)
        ax2.set_ylabel("Noising Recovery", fontsize=11)
        ax2.set_title("Position Sweep: Denoising vs Noising", fontsize=11)
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax2.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "No position results", ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _finalize_plot(output_dir / f"denoising_vs_noising_{step_size}.png")


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

    lines = ["GREEDY GENERATION RESULTS", "═" * 50, ""]
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

        flip_marker = "✓ FLIPPED" if orig_label != intv_label else "(no flip)"
        lines.extend(
            [
                "DENOISING (corrupt→clean):",
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

        flip_marker = "✓ FLIPPED" if orig_label != intv_label else "(no flip)"
        lines.extend(
            [
                "NOISING (clean→corrupt):",
                f"  Original greedy: '{orig_label}'  {flip_marker}",
                f"  After patch:     '{intv_label}'",
                f"  Recovery: {n.recovery:.4f}",
            ]
        )
        if orig_response:
            lines.append(f"  Response: '{orig_response[:40]}...'")

    # Combined sanity score
    lines.append("")
    lines.append("─" * 45)
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
    _finalize_plot(output_dir / "sanity_check.png")


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
        _finalize_plot(output_dir / f"coarse_patching_agg_{step_size}.png")

    print(f"[viz] Aggregated coarse patching plots saved to {output_dir}")
