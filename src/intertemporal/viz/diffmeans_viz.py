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

# Colors
COSINE_COLOR = "#1E90FF"
ATTN_COLOR = "#FF6B6B"
MLP_COLOR = "#4ECDC4"
TOTAL_COLOR = "#9B59B6"
DIFF_NORM_COLOR = "#F39C12"
EFF_RANK_COLOR = "#2ECC71"
TOP1_COLOR = "#E74C3C"


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

    # Generate plots
    _plot_cosine_trajectory(agg, output_dir / "cosine_trajectory.png")
    _plot_rotation_decomposition(agg, output_dir / "rotation_decomposition.png")
    _plot_diff_norm_trajectory(agg, output_dir / "diff_norm_trajectory.png")
    _plot_svd_metrics(agg, output_dir / "svd_metrics.png")

    log(f"[diffmeans_viz] Generated {4} plots in {output_dir}")


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
    ax.set_ylim(-0.1, 1.1)
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
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _setup_grid(ax: plt.Axes) -> None:
    """Set up major and minor grid lines."""
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=MINOR_GRID_ALPHA, linewidth=MINOR_GRID_LINE_WIDTH)


def _plot_cosine_trajectory(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
) -> None:
    """Plot cosine similarity between consecutive layer directions."""
    layers, means, stds = agg.get_mean_cosine_trajectory()
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    # Plot individual pairs
    for pr in agg.pair_results:
        pair_layers, pair_cosines = pr.get_cosine_trajectory()
        if pair_layers:
            ax.plot(
                pair_layers,
                pair_cosines,
                color=COSINE_COLOR,
                alpha=PAIR_LINE_ALPHA,
                linewidth=PAIR_LINE_WIDTH,
            )

    # Plot mean with error band
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(
        layers,
        means_arr - stds_arr,
        means_arr + stds_arr,
        color=COSINE_COLOR,
        alpha=0.2,
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
    ax.set_ylim(-0.1, 1.1)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_rotation_decomposition(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
) -> None:
    """Plot rotation angles decomposed into attention and MLP contributions."""
    rotation_data = agg.get_mean_rotation_trajectory()
    if not rotation_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    # Plot each rotation type
    for name, color in [("attn", ATTN_COLOR), ("mlp", MLP_COLOR), ("total", TOTAL_COLOR)]:
        if name not in rotation_data:
            continue
        layers, means, stds = rotation_data[name]
        if not layers:
            continue

        # Plot individual pairs
        for pr in agg.pair_results:
            pair_rot = pr.get_rotation_trajectory()
            if name in pair_rot:
                pair_layers, pair_angles = pair_rot[name]
                if pair_layers:
                    ax.plot(
                        pair_layers,
                        pair_angles,
                        color=color,
                        alpha=PAIR_LINE_ALPHA,
                        linewidth=PAIR_LINE_WIDTH,
                    )

        # Plot mean with error band
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.fill_between(
            layers,
            means_arr - stds_arr,
            means_arr + stds_arr,
            color=color,
            alpha=0.2,
        )
        label = {"attn": "Attention", "mlp": "MLP", "total": "Total"}[name]
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
    ax.set_ylabel("Rotation Angle (degrees)")
    ax.set_title("Direction Rotation by Component")
    ax.legend(loc="upper right")
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_diff_norm_trajectory(
    agg: DiffMeansAggregatedResults,
    output_path: Path,
) -> None:
    """Plot difference vector norms across layers."""
    if not agg.pair_results:
        return

    # Collect diff norms per layer
    layer_norms: dict[int, list[float]] = {}
    for pr in agg.pair_results:
        for lr in pr.layer_results:
            if lr.layer not in layer_norms:
                layer_norms[lr.layer] = []
            layer_norms[lr.layer].append(float(lr.diff_norm))

    if not layer_norms:
        return

    layers = sorted(layer_norms.keys())
    means = [float(np.mean(layer_norms[l])) for l in layers]
    stds = [float(np.std(layer_norms[l])) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    # Plot individual pairs
    for pr in agg.pair_results:
        pair_layers = [lr.layer for lr in pr.layer_results]
        pair_norms = [float(lr.diff_norm) for lr in pr.layer_results]
        ax.plot(
            pair_layers,
            pair_norms,
            color=DIFF_NORM_COLOR,
            alpha=PAIR_LINE_ALPHA,
            linewidth=PAIR_LINE_WIDTH,
        )

    # Plot mean with error band
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(
        layers,
        means_arr - stds_arr,
        means_arr + stds_arr,
        color=DIFF_NORM_COLOR,
        alpha=0.2,
    )
    ax.plot(
        layers,
        means,
        color=DIFF_NORM_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Difference Vector Norm")
    ax.set_title("Activation Difference Magnitude")
    _setup_grid(ax)

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
