"""Visualization for difference-in-means analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

from ....common.logging import log
from . import DiffMeansAggregatedResults, DiffMeansConfig, DiffMeansPairResult

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair


# Plot styling constants
PAIR_LINE_ALPHA = 0.15
PAIR_LINE_WIDTH = 0.8
MEAN_LINE_WIDTH = 2.0
MEAN_LINE_ALPHA = 1.0
MEAN_MARKER = "o"
MEAN_MARKER_SIZE = 5
GRID_ALPHA = 0.5
GRID_LINE_WIDTH = 0.5
MINOR_GRID_ALPHA = 0.25
MINOR_GRID_LINE_WIDTH = 0.3
DPI = 150
FILL_ALPHA = 0.2

# Colors
COSINE_COLOR = "#1E90FF"
ATTN_COLOR = "#FF6B6B"
MLP_COLOR = "#4ECDC4"
TOTAL_COLOR = "#9B59B6"
DIFF_NORM_COLOR = "#F39C12"
EFF_RANK_COLOR = "#2ECC71"
TOP1_COLOR = "#E74C3C"
LOGIT_DIR_COLOR = "#E91E63"
INITIAL_DIR_COLOR = "#00BCD4"

# Critical layer styling (values come from DiffMeansConfig)
CRITICAL_LAYER_COLOR = "#CCCCCC"
CRITICAL_LAYER_ALPHA = 0.3
ANNOTATION_COLOR = "#666666"

# Position colors for multi-position plots
POSITION_COLORS = [
    "#E91E63",  # Pink
    "#9C27B0",  # Purple
    "#673AB7",  # Deep Purple
    "#2196F3",  # Blue
    "#607D8B",  # Grey
]


def visualize_diffmeans(
    agg: DiffMeansAggregatedResults,
    output_dir: Path,
    config: DiffMeansConfig | None = None,
) -> None:
    """Generate all diffmeans visualizations.

    Args:
        agg: Aggregated diffmeans results
        output_dir: Directory to save plots
        config: DiffMeansConfig with visualization settings
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = DiffMeansConfig()

    if not agg.pair_results:
        log("[diffmeans_viz] No pair results to visualize")
        return

    n_plots = 0
    all_positions = sorted(agg.get_all_format_positions())

    if not all_positions:
        log("[diffmeans_viz] No format positions to visualize")
        return

    # Generate plots for each format_pos
    # Keys are either "time_horizon" (combined) or "time_horizon:0" (per rel_pos)
    for format_pos in all_positions:
        # Parse into base_name and subdirectory
        if ":" in format_pos:
            base_name, rel_pos = format_pos.rsplit(":", 1)
            pos_dir = output_dir / base_name / f"rel_pos_{rel_pos}"
        else:
            # Combined (no :N suffix)
            pos_dir = output_dir / format_pos / "combined"

        pos_dir.mkdir(parents=True, exist_ok=True)

        _plot_cosine_trajectory(agg, pos_dir / "cosine_trajectory.png", format_pos)
        n_plots += 1

        _plot_cosine_trajectory_zoomed(agg, pos_dir / "cosine_trajectory_zoomed.png", format_pos, config)
        n_plots += 1

        _plot_rotation_decomposition(agg, pos_dir / "rotation_decomposition.png", format_pos, config)
        n_plots += 1

        _plot_diff_norm_trajectory(agg, pos_dir / "diff_norm_trajectory.png", format_pos, config)
        n_plots += 1

        _plot_cosine_to_logit(agg, pos_dir / "cosine_to_logit.png", format_pos, config)
        n_plots += 1

        _plot_cosine_to_initial(agg, pos_dir / "cosine_to_initial.png", format_pos, config)
        n_plots += 1

        _plot_component_norm_decomposition(agg, pos_dir / "component_norm_decomposition.png", format_pos, config)
        n_plots += 1

    # SVD metrics (not position-specific)
    _plot_svd_metrics(agg, output_dir / "svd_metrics.png")
    n_plots += 1

    # Comparison plot across positions
    _plot_diff_norm_by_position(agg, output_dir / "diff_norm_by_position.png", config)
    n_plots += 1

    log(f"[diffmeans_viz] Generated {n_plots} plots in {output_dir}")


def visualize_diffmeans_pair(
    result: DiffMeansPairResult,
    output_dir: Path,
    runner: "BinaryChoiceRunner | None" = None,
    pair: "ContrastivePair | None" = None,
) -> None:
    """Generate diffmeans visualizations for a single pair.

    Args:
        result: Per-pair diffmeans results
        output_dir: Directory to save plots
        runner: Optional model runner for OV projection analysis
        pair: Optional contrastive pair for OV projection analysis
    """
    if not result.position_results:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots for each format_pos
    for format_pos, layer_results in result.position_results.items():
        if not layer_results:
            continue

        pos_dir = output_dir / format_pos
        pos_dir.mkdir(parents=True, exist_ok=True)

        # Plot cosine trajectory
        layers, cosines = result.get_cosine_trajectory(format_pos)
        if layers and cosines:
            _plot_single_cosine_trajectory(layers, cosines, pos_dir / "cosine_trajectory.png")

        # Plot rotation decomposition
        rotation_data = result.get_rotation_trajectory(format_pos)
        if "total" in rotation_data and rotation_data["total"][0]:
            _plot_single_rotation(rotation_data, pos_dir / "rotation_decomposition.png")

        # Plot diff norm trajectory
        pair_layers = [lr.layer for lr in layer_results]
        pair_norms = [float(lr.diff_norm) for lr in layer_results]
        if pair_layers and pair_norms:
            _plot_single_diff_norm(pair_layers, pair_norms, pos_dir / "diff_norm_trajectory.png")

        # Plot cosine to logit
        layers_logit, cosines_logit = result.get_cosine_to_logit_trajectory(format_pos)
        if layers_logit and cosines_logit:
            _plot_single_cosine_to_logit(
                layers_logit, cosines_logit, pos_dir / "cosine_to_logit.png"
            )

        # Plot cosine to initial
        layers_init, cosines_init = result.get_cosine_to_initial_trajectory(format_pos)
        if layers_init and cosines_init:
            _plot_single_cosine_to_initial(
                layers_init, cosines_init, pos_dir / "cosine_to_initial.png"
            )

    # OV projection analysis (requires TransformerLens backend)
    if runner is not None and pair is not None:
        visualize_ov_projection(runner, pair, output_dir)


def _setup_grid(ax: plt.Axes) -> None:
    """Set up major and minor grid lines."""
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=MINOR_GRID_ALPHA, linewidth=MINOR_GRID_LINE_WIDTH)


def _add_critical_layer_shading(
    ax: plt.Axes,
    critical_layer_ranges: list[tuple[int, int]],
) -> None:
    """Add vertical shading for critical layer ranges."""
    for start, end in critical_layer_ranges:
        ax.axvspan(start, end, color=CRITICAL_LAYER_COLOR, alpha=CRITICAL_LAYER_ALPHA, zorder=0)


def _add_reference_lines(
    ax: plt.Axes,
    layers: list[int],
    reference_layers: list[int],
) -> None:
    """Add reference lines at key layers."""
    for layer in reference_layers:
        if layer in layers or (layers and min(layers) <= layer <= max(layers)):
            ax.axvline(
                x=layer, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1
            )


def _add_layer_annotations(
    ax: plt.Axes,
    layers: list[int],
    annotation_layers: list[int],
    y_data: list[float] | None = None,
) -> None:
    """Add text annotations at key layers."""
    if not layers:
        return
    for layer in annotation_layers:
        if layer in layers or (min(layers) <= layer <= max(layers)):
            # Get y position from data if available, otherwise use top of plot
            if y_data and layer in layers:
                idx = layers.index(layer)
                y_pos = y_data[idx]
            else:
                y_pos = ax.get_ylim()[1] * 0.95
            ax.annotate(
                f"L{layer}",
                xy=(layer, y_pos),
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=7,
                color=ANNOTATION_COLOR,
                ha="center",
                va="bottom",
            )


# =============================================================================
# Single-pair plots
# =============================================================================


def _plot_single_cosine_trajectory(
    layers: list[int],
    cosines: list[float],
    output_path: Path,
) -> None:
    """Plot cosine trajectory for a single pair."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    ax.plot(
        layers,
        cosines,
        color=COSINE_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity (to next layer)")
    ax.set_title("Direction Stability Across Layers")
    ax.set_ylim(0, 1)  # Full 0-1 range
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_single_rotation(
    rotation_data: dict[str, tuple[list[int], list[float]]],
    output_path: Path,
) -> None:
    """Plot rotation decomposition for a single pair."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    for name, color in [("attn", ATTN_COLOR), ("mlp", MLP_COLOR), ("total", TOTAL_COLOR)]:
        if name not in rotation_data:
            continue
        pair_layers, pair_angles = rotation_data[name]
        if not pair_layers:
            continue
        label = {"attn": "Attention", "mlp": "MLP", "total": "Total"}[name]
        ax.plot(
            pair_layers,
            pair_angles,
            color=color,
            linewidth=MEAN_LINE_WIDTH,
            alpha=MEAN_LINE_ALPHA,
            marker=MEAN_MARKER,
            markersize=MEAN_MARKER_SIZE,
            label=label,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Rotation Angle (degrees)")
    ax.set_title("Direction Rotation by Component")
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)  # Start from 0
    ax.set_xlim(left=0)  # Start from layer 0
    _add_critical_layer_shading(ax, [])  # No critical layers for single-pair plots
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_single_diff_norm(
    layers: list[int],
    norms: list[float],
    output_path: Path,
) -> None:
    """Plot diff norm trajectory for a single pair."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    ax.plot(
        layers,
        norms,
        color=DIFF_NORM_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Difference Vector Norm")
    ax.set_title("Activation Difference Magnitude")
    _add_reference_lines(ax, layers)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_single_cosine_to_logit(
    layers: list[int],
    cosines: list[float],
    output_path: Path,
) -> None:
    """Plot cosine to logit direction for a single pair."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    ax.plot(
        layers,
        cosines,
        color=LOGIT_DIR_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity to Logit Direction")
    ax.set_title("Alignment with Answer Direction")
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color="#999999", linestyle="-", linewidth=0.5, alpha=0.5)
    _add_critical_layer_shading(ax, [])  # No critical layers for single-pair plots
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_single_cosine_to_initial(
    layers: list[int],
    cosines: list[float],
    output_path: Path,
) -> None:
    """Plot cosine to initial direction for a single pair."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    ax.plot(
        layers,
        cosines,
        color=INITIAL_DIR_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity to Initial Direction")
    ax.set_title("Cumulative Direction Drift from Layer 0")
    ax.set_ylim(0, 1)
    _add_critical_layer_shading(ax, [])  # No critical layers for single-pair plots
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# =============================================================================
# Aggregated plots
# =============================================================================


def _plot_cosine_trajectory(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
    format_pos: str,
) -> None:
    """Plot cosine similarity between consecutive layer directions.

    Fixed: Uses full 0-1 Y-axis range and shows only shaded confidence bands
    (no cluttered individual pair traces).
    """
    layers, means, stds = agg.get_mean_cosine_trajectory(format_pos)
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    # Plot mean with error band (no individual traces)
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(
        layers,
        means_arr - stds_arr,
        means_arr + stds_arr,
        color=COSINE_COLOR,
        alpha=FILL_ALPHA,
    )
    ax.plot(
        layers,
        means,
        color=COSINE_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity (to next layer)")
    ax.set_title("Direction Stability Across Layers")
    ax.set_ylim(0, 1)  # Full 0-1 range (not zoomed in)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_cosine_trajectory_zoomed(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
    format_pos: str,
    config: DiffMeansConfig,
) -> None:
    """Plot cosine trajectory with zoomed Y-axis to show subtle dips.

    This version makes mid-network instabilities visible.
    """
    layers, means, stds = agg.get_mean_cosine_trajectory(format_pos)
    if not layers:
        return

    zoom_min, zoom_max = config.cosine_zoom_range

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    _add_critical_layer_shading(ax, config.critical_layer_ranges)

    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(
        layers,
        np.maximum(means_arr - stds_arr, zoom_min),  # Clip to visible range
        np.minimum(means_arr + stds_arr, zoom_max),
        color=COSINE_COLOR,
        alpha=FILL_ALPHA,
    )
    ax.plot(
        layers,
        means,
        color=COSINE_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    # Add layer annotations
    _add_layer_annotations(ax, list(layers), config.annotation_layers, list(means))

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity (to next layer)")
    ax.set_title(f"Direction Stability (Zoomed: {zoom_min}-{zoom_max})")
    ax.set_ylim(zoom_min, zoom_max)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_rotation_decomposition(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
    format_pos: str,
    config: DiffMeansConfig,
) -> None:
    """Plot rotation angles decomposed into attention and MLP contributions.

    Fixed: Clean Y-axis ticks, shaded bands instead of individual traces,
    starts from layer 1 (not 0) to avoid rendering artifacts,
    adds critical layer shading.
    """
    rotation_data = agg.get_mean_rotation_trajectory(format_pos)
    if not rotation_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    # Add critical layer shading first (background)
    _add_critical_layer_shading(ax, config.critical_layer_ranges)

    # Plot each rotation type with shaded bands (no individual traces)
    for name, color in [("attn", ATTN_COLOR), ("mlp", MLP_COLOR), ("total", TOTAL_COLOR)]:
        if name not in rotation_data:
            continue
        layers, means, stds = rotation_data[name]
        if not layers:
            continue

        # Filter to start from layer 1 to avoid layer 0 artifacts
        filtered_data = [(l, m, s) for l, m, s in zip(layers, means, stds) if l >= 1]
        if not filtered_data:
            continue
        layers_f, means_f, stds_f = zip(*filtered_data)

        # Plot mean with error band
        means_arr = np.array(means_f)
        stds_arr = np.array(stds_f)
        ax.fill_between(
            layers_f,
            means_arr - stds_arr,
            means_arr + stds_arr,
            color=color,
            alpha=FILL_ALPHA,
        )
        label = {"attn": "Attention", "mlp": "MLP", "total": "Total"}[name]
        ax.plot(
            layers_f,
            means_f,
            color=color,
            linewidth=MEAN_LINE_WIDTH,
            alpha=MEAN_LINE_ALPHA,
            marker=MEAN_MARKER,
            markersize=MEAN_MARKER_SIZE,
            label=label,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Rotation Angle (degrees)")
    ax.set_title("Direction Rotation by Component")
    ax.legend(loc="upper right")

    # Clean Y-axis ticks at regular intervals
    ax.set_ylim(0, None)
    max_angle = ax.get_ylim()[1]
    if max_angle <= 50:
        ax.set_yticks(np.arange(0, max_angle + 1, 10))
    elif max_angle <= 100:
        ax.set_yticks(np.arange(0, max_angle + 1, 20))
    else:
        ax.set_yticks(np.arange(0, max_angle + 1, 30))

    ax.set_xlim(left=1)  # Start from layer 1 to avoid artifacts
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_diff_norm_trajectory(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
    format_pos: str,
    config: DiffMeansConfig,
) -> None:
    """Plot difference vector norms across layers.

    Enhanced: Adds reference lines at critical layers, includes log scale subplot.
    """
    layers, means, stds = agg.get_mean_diff_norm_trajectory(format_pos)
    if not layers:
        return

    # Create figure with two subplots: linear and log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

    means_arr = np.array(means)
    stds_arr = np.array(stds)

    # Linear scale plot
    _add_reference_lines(ax1, layers, config.reference_layers)
    ax1.fill_between(
        layers,
        means_arr - stds_arr,
        means_arr + stds_arr,
        color=DIFF_NORM_COLOR,
        alpha=FILL_ALPHA,
    )
    ax1.plot(
        layers,
        means,
        color=DIFF_NORM_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Difference Vector Norm")
    ax1.set_title("Activation Difference Magnitude (Linear Scale)")
    _setup_grid(ax1)

    # Log scale plot
    _add_reference_lines(ax2, layers, config.reference_layers)
    ax2.fill_between(
        layers,
        np.maximum(means_arr - stds_arr, 1e-6),  # Avoid log of negative
        means_arr + stds_arr,
        color=DIFF_NORM_COLOR,
        alpha=FILL_ALPHA,
    )
    ax2.plot(
        layers,
        means,
        color=DIFF_NORM_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Difference Vector Norm (log)")
    ax2.set_title("Activation Difference Magnitude (Log Scale)")
    ax2.set_yscale("log")
    _setup_grid(ax2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_svd_metrics(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
) -> None:
    """Plot SVD metrics (effective rank and top-1 ratio) across layers."""
    if not agg.svd_results:
        return

    layers_rank, eff_ranks = agg.get_svd_effective_rank_trajectory()
    layers_top1, top1_ratios = agg.get_svd_top1_ratio_trajectory()

    if not layers_rank and not layers_top1:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)

    # Effective rank
    if layers_rank and eff_ranks:
        ax1.plot(
            layers_rank,
            eff_ranks,
            color=EFF_RANK_COLOR,
            linewidth=MEAN_LINE_WIDTH,
            marker=MEAN_MARKER,
            markersize=MEAN_MARKER_SIZE,
        )
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Effective Rank")
        ax1.set_title("SVD Effective Rank (Dimensionality)")
        _setup_grid(ax1)

    # Top-1 ratio
    if layers_top1 and top1_ratios:
        ax2.plot(
            layers_top1,
            top1_ratios,
            color=TOP1_COLOR,
            linewidth=MEAN_LINE_WIDTH,
            marker=MEAN_MARKER,
            markersize=MEAN_MARKER_SIZE,
        )
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Top-1 Ratio")
        ax2.set_title("Top Singular Value Dominance (higher = more rank-1)")
        ax2.set_ylim(0, 1)
        _setup_grid(ax2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# =============================================================================
# New plots
# =============================================================================


def _plot_cosine_to_logit(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
    format_pos: str,
    config: DiffMeansConfig,
) -> None:
    """Plot cosine similarity to logit direction across layers.

    Shows when the causal direction aligns with the answer direction.
    Expect low in early layers, ramping up around L19-L24.
    """
    layers, means, stds = agg.get_mean_cosine_to_logit_trajectory(format_pos)
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    _add_critical_layer_shading(ax, config.critical_layer_ranges)

    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(
        layers,
        means_arr - stds_arr,
        means_arr + stds_arr,
        color=LOGIT_DIR_COLOR,
        alpha=FILL_ALPHA,
    )
    ax.plot(
        layers,
        means,
        color=LOGIT_DIR_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.axhline(y=0, color="#999999", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity to Logit Direction")
    ax.set_title("Alignment with Answer Direction (Logit Lens)")
    ax.set_ylim(-1, 1)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_cosine_to_initial(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
    format_pos: str,
    config: DiffMeansConfig,
) -> None:
    """Plot cosine similarity to initial direction (cumulative drift).

    Shows whether per-layer stability (high cos to next layer) accumulates
    into large total drift from the starting direction.
    """
    layers, means, stds = agg.get_mean_cosine_to_initial_trajectory(format_pos)
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    _add_critical_layer_shading(ax, config.critical_layer_ranges)

    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(
        layers,
        means_arr - stds_arr,
        means_arr + stds_arr,
        color=INITIAL_DIR_COLOR,
        alpha=FILL_ALPHA,
    )
    ax.plot(
        layers,
        means,
        color=INITIAL_DIR_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity to Initial Direction")
    ax.set_title("Cumulative Direction Drift from Layer 0")
    ax.set_ylim(0, 1)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_component_norm_decomposition(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
    format_pos: str,
    config: DiffMeansConfig,
) -> None:
    """Plot difference norm decomposed by component (attention vs MLP).

    Shows whether the magnitude growth comes from attention or MLP contributions.
    """
    component_data = agg.get_mean_component_norm_trajectory(format_pos)
    if not component_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    _add_critical_layer_shading(ax, config.critical_layer_ranges)

    for name, color in [("attn", ATTN_COLOR), ("mlp", MLP_COLOR)]:
        if name not in component_data:
            continue
        layers, means, stds = component_data[name]
        if not layers:
            continue

        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.fill_between(
            layers,
            means_arr - stds_arr,
            means_arr + stds_arr,
            color=color,
            alpha=FILL_ALPHA,
        )
        label = {"attn": "Attention Output", "mlp": "MLP Output"}[name]
        ax.plot(
            layers,
            means,
            color=color,
            linewidth=MEAN_LINE_WIDTH,
            alpha=MEAN_LINE_ALPHA,
            marker=MEAN_MARKER,
            markersize=MEAN_MARKER_SIZE,
            label=label,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Component Difference Norm")
    ax.set_title("Activation Difference by Component (Attention vs MLP)")
    ax.legend(loc="upper left")
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_diff_norm_by_position(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
    config: DiffMeansConfig,
) -> None:
    """Plot difference norm trajectories for different positions.

    Shows two side-by-side plots: one for prompt positions and one for response positions.

    Args:
        agg: Aggregated results
        output_path: Path to save the plot
        config: DiffMeansConfig with visualization settings
    """
    from ...common.semantic_positions import PROMPT_POSITIONS, RESPONSE_POSITIONS

    all_positions = agg.get_all_format_positions()
    if not all_positions:
        return

    # Split into prompt and response positions
    prompt_pos = [p for p in PROMPT_POSITIONS if p in all_positions]
    response_pos = [p for p in RESPONSE_POSITIONS if p in all_positions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=DPI)

    # Extended color palette for many positions
    colors = [
        "#E91E63", "#9C27B0", "#673AB7", "#3F51B5", "#2196F3",
        "#00BCD4", "#009688", "#4CAF50", "#8BC34A", "#CDDC39",
        "#FFC107", "#FF9800", "#FF5722", "#795548", "#607D8B",
    ]

    # Left plot: Prompt positions
    _add_critical_layer_shading(ax1, config.critical_layer_ranges)
    for idx, format_pos in enumerate(prompt_pos):
        layers, means, stds = agg.get_mean_diff_norm_trajectory(format_pos)
        if not layers:
            continue

        color = colors[idx % len(colors)]
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax1.fill_between(
            layers, means_arr - stds_arr, means_arr + stds_arr,
            color=color, alpha=0.15,
        )
        ax1.plot(
            layers, means, color=color,
            linewidth=MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA,
            marker=MEAN_MARKER, markersize=MEAN_MARKER_SIZE - 1,
            label=format_pos,
        )

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Difference Vector Norm")
    ax1.set_title("Prompt Positions")
    ax1.legend(loc="upper left", fontsize=8)
    _setup_grid(ax1)

    # Right plot: Response positions
    _add_critical_layer_shading(ax2, config.critical_layer_ranges)
    for idx, format_pos in enumerate(response_pos):
        layers, means, stds = agg.get_mean_diff_norm_trajectory(format_pos)
        if not layers:
            continue

        color = colors[idx % len(colors)]
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax2.fill_between(
            layers, means_arr - stds_arr, means_arr + stds_arr,
            color=color, alpha=0.15,
        )
        ax2.plot(
            layers, means, color=color,
            linewidth=MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA,
            marker=MEAN_MARKER, markersize=MEAN_MARKER_SIZE - 1,
            label=format_pos,
        )

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Difference Vector Norm")
    ax2.set_title("Response Positions")
    ax2.legend(loc="upper left", fontsize=8)
    _setup_grid(ax2)

    plt.suptitle("Activation Difference by Position", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# =============================================================================
# OV Projection Analysis (requires TransformerLens backend)
# =============================================================================


def visualize_ov_projection(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    output_dir: Path,
    layers: list[int] | None = None,
    top_n_heads: int = 10,
) -> int:
    """Visualize OV projection alignment with logit direction.

    For top heads (by attention to source positions), computes W_OV = W_V @ W_O
    and measures how much the head's output aligns with the logit direction.
    This shows which heads amplify vs suppress the task-relevant signal.

    Args:
        runner: Model runner with TransformerLens backend
        pair: Contrastive pair (for logit direction computation)
        output_dir: Output directory for plots
        layers: Layers to analyze (default: [19, 21, 24, 31, 34])
        top_n_heads: Number of top heads to show

    Returns:
        Number of plots generated
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if backend supports weight matrix access
    try:
        _ = runner._backend.get_W_OV(0, 0)
    except (NotImplementedError, AttributeError):
        log("[diffmeans_viz] OV projection requires TransformerLens backend")
        return 0

    if layers is None:
        layers = [19, 21, 24, 31, 34]

    # Compute logit direction
    logit_direction = _compute_logit_direction_tensor(runner, pair)
    if logit_direction is None:
        log("[diffmeans_viz] Could not compute logit direction for OV analysis")
        return 0

    # Get number of heads
    try:
        n_heads = runner._backend.get_n_heads()
    except (NotImplementedError, AttributeError):
        n_heads = 32  # Default for Qwen3-4B

    # Compute OV alignment for all heads in target layers
    head_data = []
    for layer in layers:
        if layer >= runner.n_layers:
            continue
        for head in range(n_heads):
            try:
                W_OV = runner._backend.get_W_OV(layer, head)  # [d_model, d_model]

                # Project logit direction through OV circuit
                # This tells us: if this head attends to a position containing
                # the logit direction, what does it output?
                ov_output = W_OV @ logit_direction
                ov_output = ov_output / (ov_output.norm() + 1e-10)

                # Compute alignment with logit direction
                alignment = float(torch.dot(ov_output, logit_direction).item())
                head_data.append((layer, head, alignment))
            except Exception as e:
                log(f"[diffmeans_viz] Error computing OV for L{layer}.H{head}: {e}")

    if not head_data:
        return 0

    # Sort by absolute alignment and take top N
    head_data.sort(key=lambda x: abs(x[2]), reverse=True)
    top_heads = head_data[:top_n_heads]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"L{l}.H{h}" for l, h, _ in top_heads]
    alignments = [a for _, _, a in top_heads]
    colors = ["#2ECC71" if a > 0 else "#E74C3C" for a in alignments]

    bars = ax.bar(range(len(alignments)), alignments, color=colors, alpha=0.8, edgecolor="black")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Head (Layer.Head)")
    ax.set_ylabel("OV Alignment with Logit Direction")
    ax.set_title(
        "OV Projection Analysis: Head Output Alignment with Answer Direction\n"
        "(Green = amplifies correct answer, Red = suppresses)"
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, alignments):
        height = bar.get_height()
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )

    plt.tight_layout()
    output_path = output_dir / "ov_projection.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    log(f"[diffmeans_viz] Saved OV projection: {output_path}")
    return 1


def _compute_logit_direction_tensor(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
) -> torch.Tensor | None:
    """Compute normalized logit direction as tensor."""
    W_U = runner.W_U
    if W_U is None:
        return None

    clean_div_pos = pair.clean_divergent_position
    corrupted_div_pos = pair.corrupted_divergent_position

    if clean_div_pos is None or corrupted_div_pos is None:
        clean_token = pair.clean_traj.token_ids[-1]
        corrupted_token = pair.corrupted_traj.token_ids[-1]
    else:
        clean_token = pair.clean_traj.token_ids[clean_div_pos]
        corrupted_token = pair.corrupted_traj.token_ids[corrupted_div_pos]

    if clean_token == corrupted_token:
        return None

    # Get direction from W_U
    if W_U.shape[0] > W_U.shape[1]:
        # [vocab_size, d_model]
        clean_vec = W_U[clean_token]
        corrupted_vec = W_U[corrupted_token]
    else:
        # [d_model, vocab_size]
        clean_vec = W_U[:, clean_token]
        corrupted_vec = W_U[:, corrupted_token]

    direction = clean_vec - corrupted_vec
    return direction / (direction.norm() + 1e-10)
