"""Visualization for difference-in-means analysis results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...common.logging import log
from ..experiments.diffmeans import DiffMeansAggregatedResults, DiffMeansPairResult


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

# Critical layers for reference lines (configurable)
CRITICAL_LAYER_RANGES = [(19, 24), (28, 34)]
CRITICAL_LAYER_COLOR = "#CCCCCC"
CRITICAL_LAYER_ALPHA = 0.3

# Key layers to annotate on plots
ANNOTATION_LAYERS = [19, 21, 24, 31, 34]
ANNOTATION_COLOR = "#666666"

# Position colors for multi-position plots
POSITION_COLORS = [
    "#E91E63",  # P86 - Pink
    "#9C27B0",  # P87 - Purple
    "#673AB7",  # P88 - Deep Purple
    "#2196F3",  # P145 - Blue
    "#607D8B",  # Other - Grey
]


def visualize_diffmeans(
    agg: DiffMeansAggregatedResults,
    output_dir: Path,
) -> None:
    """Generate all diffmeans visualizations.

    Args:
        agg: Aggregated diffmeans results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not agg.pair_results:
        log("[diffmeans_viz] No pair results to visualize")
        return

    n_plots = 0

    # Core plots (fixed versions)
    _plot_cosine_trajectory(agg, output_dir / "cosine_trajectory.png")
    n_plots += 1

    _plot_cosine_trajectory_zoomed(agg, output_dir / "cosine_trajectory_zoomed.png")
    n_plots += 1

    _plot_rotation_decomposition(agg, output_dir / "rotation_decomposition.png")
    n_plots += 1

    _plot_diff_norm_trajectory(agg, output_dir / "diff_norm_trajectory.png")
    n_plots += 1

    _plot_svd_metrics(agg, output_dir / "svd_metrics.png")
    n_plots += 1

    # New plots
    _plot_cosine_to_logit(agg, output_dir / "cosine_to_logit.png")
    n_plots += 1

    _plot_cosine_to_initial(agg, output_dir / "cosine_to_initial.png")
    n_plots += 1

    _plot_component_norm_decomposition(agg, output_dir / "component_norm_decomposition.png")
    n_plots += 1

    _plot_diff_norm_by_position(agg, output_dir / "diff_norm_by_position.png")
    n_plots += 1

    log(f"[diffmeans_viz] Generated {n_plots} plots in {output_dir}")


def visualize_diffmeans_pair(
    result: DiffMeansPairResult,
    output_dir: Path,
) -> None:
    """Generate diffmeans visualizations for a single pair.

    Args:
        result: Per-pair diffmeans results
        output_dir: Directory to save plots
    """
    if not result.layer_results:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot cosine trajectory
    layers, cosines = result.get_cosine_trajectory()
    if layers and cosines:
        _plot_single_cosine_trajectory(layers, cosines, output_dir / "cosine_trajectory.png")

    # Plot rotation decomposition
    rotation_data = result.get_rotation_trajectory()
    if "total" in rotation_data and rotation_data["total"][0]:
        _plot_single_rotation(rotation_data, output_dir / "rotation_decomposition.png")

    # Plot diff norm trajectory
    pair_layers = [lr.layer for lr in result.layer_results]
    pair_norms = [float(lr.diff_norm) for lr in result.layer_results]
    if pair_layers and pair_norms:
        _plot_single_diff_norm(pair_layers, pair_norms, output_dir / "diff_norm_trajectory.png")

    # Plot cosine to logit
    layers_logit, cosines_logit = result.get_cosine_to_logit_trajectory()
    if layers_logit and cosines_logit:
        _plot_single_cosine_to_logit(
            layers_logit, cosines_logit, output_dir / "cosine_to_logit.png"
        )

    # Plot cosine to initial
    layers_init, cosines_init = result.get_cosine_to_initial_trajectory()
    if layers_init and cosines_init:
        _plot_single_cosine_to_initial(
            layers_init, cosines_init, output_dir / "cosine_to_initial.png"
        )


def _setup_grid(ax: plt.Axes) -> None:
    """Set up major and minor grid lines."""
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=MINOR_GRID_ALPHA, linewidth=MINOR_GRID_LINE_WIDTH)


def _add_critical_layer_shading(ax: plt.Axes) -> None:
    """Add vertical shading for critical layer ranges."""
    for start, end in CRITICAL_LAYER_RANGES:
        ax.axvspan(start, end, color=CRITICAL_LAYER_COLOR, alpha=CRITICAL_LAYER_ALPHA, zorder=0)


def _add_reference_lines(ax: plt.Axes, layers: list[int]) -> None:
    """Add reference lines at key layers (L19, L24)."""
    for layer in [19, 24]:
        if layer in layers or (layers and min(layers) <= layer <= max(layers)):
            ax.axvline(
                x=layer, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1
            )


def _add_layer_annotations(ax: plt.Axes, layers: list[int], y_data: list[float] | None = None) -> None:
    """Add text annotations at key layers."""
    if not layers:
        return
    for layer in ANNOTATION_LAYERS:
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
    _add_critical_layer_shading(ax)
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
    _add_critical_layer_shading(ax)
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
    _add_critical_layer_shading(ax)
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
) -> None:
    """Plot cosine similarity between consecutive layer directions.

    Fixed: Uses full 0-1 Y-axis range and shows only shaded confidence bands
    (no cluttered individual pair traces).
    """
    layers, means, stds = agg.get_mean_cosine_trajectory()
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
) -> None:
    """Plot cosine trajectory with zoomed Y-axis (0.75-1.0) to show subtle dips.

    This version makes mid-network instabilities (L4, L21) visible.
    """
    layers, means, stds = agg.get_mean_cosine_trajectory()
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    _add_critical_layer_shading(ax)

    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(
        layers,
        np.maximum(means_arr - stds_arr, 0.75),  # Clip to visible range
        np.minimum(means_arr + stds_arr, 1.0),
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
    _add_layer_annotations(ax, list(layers), list(means))

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity (to next layer)")
    ax.set_title("Direction Stability (Zoomed: 0.75-1.0)")
    ax.set_ylim(0.75, 1.0)  # Zoomed range to see subtle dips
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_rotation_decomposition(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
) -> None:
    """Plot rotation angles decomposed into attention and MLP contributions.

    Fixed: Clean Y-axis ticks, shaded bands instead of individual traces,
    starts from layer 1 (not 0) to avoid rendering artifacts,
    adds critical layer shading.
    """
    rotation_data = agg.get_mean_rotation_trajectory()
    if not rotation_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    # Add critical layer shading first (background)
    _add_critical_layer_shading(ax)

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
) -> None:
    """Plot difference vector norms across layers.

    Enhanced: Adds reference lines at critical layers, includes log scale subplot.
    """
    layers, means, stds = agg.get_mean_diff_norm_trajectory()
    if not layers:
        return

    # Create figure with two subplots: linear and log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

    means_arr = np.array(means)
    stds_arr = np.array(stds)

    # Linear scale plot
    _add_reference_lines(ax1, layers)
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
    _add_reference_lines(ax2, layers)
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
) -> None:
    """Plot cosine similarity to logit direction across layers.

    Shows when the causal direction aligns with the answer direction.
    Expect low in early layers, ramping up around L19-L24.
    """
    layers, means, stds = agg.get_mean_cosine_to_logit_trajectory()
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    _add_critical_layer_shading(ax)

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
) -> None:
    """Plot cosine similarity to initial direction (cumulative drift).

    Shows whether per-layer stability (high cos to next layer) accumulates
    into large total drift from the starting direction.
    """
    layers, means, stds = agg.get_mean_cosine_to_initial_trajectory()
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    _add_critical_layer_shading(ax)

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
) -> None:
    """Plot difference norm decomposed by component (attention vs MLP).

    Shows whether the magnitude growth comes from attention or MLP contributions.
    """
    component_data = agg.get_mean_component_norm_trajectory()
    if not component_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    _add_critical_layer_shading(ax)

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
) -> None:
    """Plot difference norm trajectories for different positions.

    Shows whether the growing difference is localized to specific positions
    or spreading everywhere.
    """
    all_positions = sorted(agg.get_all_analyzed_positions())
    if not all_positions:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    _add_critical_layer_shading(ax)

    # Plot each position with different color
    for idx, pos in enumerate(all_positions[:5]):  # Limit to 5 positions
        layers, means, stds = agg.get_mean_position_diff_norm_trajectory(pos)
        if not layers:
            continue

        color = POSITION_COLORS[idx % len(POSITION_COLORS)]
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.fill_between(
            layers,
            means_arr - stds_arr,
            means_arr + stds_arr,
            color=color,
            alpha=0.1,
        )
        ax.plot(
            layers,
            means,
            color=color,
            linewidth=MEAN_LINE_WIDTH,
            alpha=MEAN_LINE_ALPHA,
            marker=MEAN_MARKER,
            markersize=MEAN_MARKER_SIZE - 1,
            label=f"P{pos}",
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Difference Vector Norm")
    ax.set_title("Activation Difference by Position")
    ax.legend(loc="upper left")
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
