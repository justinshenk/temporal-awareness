"""Visualization for geometric (PCA) analysis results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...common.logging import log
from ..experiments.geo import GeoAggregatedResults, GeoPairResult


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
SEPARATION_COLOR = "#E74C3C"
PC1_VAR_COLOR = "#3498DB"
PC2_VAR_COLOR = "#2ECC71"
PC3_VAR_COLOR = "#9B59B6"
ALIGNMENT_COLOR = "#F39C12"
CLEAN_COLOR = "#3498DB"
CORRUPTED_COLOR = "#E74C3C"


def _setup_grid(ax: plt.Axes) -> None:
    """Set up major and minor grid lines."""
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=MINOR_GRID_ALPHA, linewidth=MINOR_GRID_LINE_WIDTH)


def visualize_geo(
    agg: GeoAggregatedResults,
    output_dir: Path,
) -> None:
    """Generate all geo visualizations.

    Args:
        agg: Aggregated geo results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not agg.pair_results:
        log("[geo_viz] No pair results to visualize")
        return

    n_plots = 0

    # Generate consolidated plots (all positions in one figure)
    _plot_consolidated_separation(agg, output_dir / "separation_all_positions.png")
    n_plots += 1
    _plot_consolidated_variance(agg, output_dir / "variance_all_positions.png")
    n_plots += 1
    _plot_consolidated_alignment(agg, output_dir / "alignment_all_positions.png")
    n_plots += 1

    # Generate plots for each position
    for position in agg.positions_analyzed:
        pos_dir = output_dir / f"pos_{position}"
        pos_dir.mkdir(parents=True, exist_ok=True)

        # Separation trajectory
        _plot_separation_trajectory(agg, position, pos_dir / "separation_trajectory.png")
        n_plots += 1

        # Variance explained trajectory
        _plot_variance_trajectory(agg, position, pos_dir / "variance_trajectory.png")
        n_plots += 1

        # Alignment with logit diff
        _plot_alignment_trajectory(agg, position, pos_dir / "logit_alignment.png")
        n_plots += 1

        # 2D PCA scatter at key layers
        _plot_pca_scatter_layers(agg, position, pos_dir)
        n_plots += 1

    log(f"[geo_viz] Generated {n_plots} plots in {output_dir}")


def visualize_geo_pair(
    result: GeoPairResult,
    output_dir: Path,
) -> None:
    """Generate geo visualizations for a single pair.

    Args:
        result: Per-pair geo results
        output_dir: Directory to save plots
    """
    if not result.position_results:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pos_result in result.position_results:
        if not pos_result.layer_results:
            continue

        pos_dir = output_dir / f"pos_{pos_result.position}"
        pos_dir.mkdir(parents=True, exist_ok=True)

        # Separation trajectory
        layers, separations = pos_result.get_separation_trajectory()
        if layers:
            _plot_single_trajectory(
                layers, separations,
                "Separation Distance", "Clean/Corrupted Separation",
                pos_dir / "separation_trajectory.png",
                color=SEPARATION_COLOR,
            )

        # Variance trajectory
        layers, var_pc1 = pos_result.get_variance_trajectory(0)
        if layers:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
            ax.plot(layers, var_pc1, color=PC1_VAR_COLOR, linewidth=MEAN_LINE_WIDTH,
                    marker=MEAN_MARKER, markersize=MEAN_MARKER_SIZE, label="PC1")

            # Add PC2, PC3 if available
            _, var_pc2 = pos_result.get_variance_trajectory(1)
            _, var_pc3 = pos_result.get_variance_trajectory(2)
            if var_pc2:
                ax.plot(layers, var_pc2, color=PC2_VAR_COLOR, linewidth=MEAN_LINE_WIDTH,
                        marker=MEAN_MARKER, markersize=MEAN_MARKER_SIZE, label="PC2")
            if var_pc3:
                ax.plot(layers, var_pc3, color=PC3_VAR_COLOR, linewidth=MEAN_LINE_WIDTH,
                        marker=MEAN_MARKER, markersize=MEAN_MARKER_SIZE, label="PC3")

            ax.set_xlabel("Layer")
            ax.set_ylabel("Explained Variance Ratio")
            ax.set_title("PCA Variance Explained")
            ax.legend(loc="upper right")
            ax.set_ylim(0, 1)
            _setup_grid(ax)
            plt.tight_layout()
            plt.savefig(pos_dir / "variance_trajectory.png", dpi=DPI, bbox_inches="tight")
            plt.close()


def _plot_separation_trajectory(
    agg: GeoAggregatedResults,
    position: int,
    output_path: Path,
) -> None:
    """Plot separation distance trajectory across layers."""
    layers, means, stds = agg.get_mean_separation_trajectory(position)
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    # Plot individual pairs
    for pr in agg.pair_results:
        pos_result = pr.get_position_result(position)
        if pos_result:
            pair_layers, pair_seps = pos_result.get_separation_trajectory()
            if pair_layers:
                ax.plot(
                    pair_layers, pair_seps,
                    color=SEPARATION_COLOR,
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
        color=SEPARATION_COLOR,
        alpha=0.2,
    )
    ax.plot(
        layers, means,
        color=SEPARATION_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Separation Distance (PC space)")
    ax.set_title(f"Clean/Corrupted Separation at Position {position}")
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_variance_trajectory(
    agg: GeoAggregatedResults,
    position: int,
    output_path: Path,
) -> None:
    """Plot explained variance ratio trajectory."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    for pc_idx, (color, label) in enumerate([
        (PC1_VAR_COLOR, "PC1"),
        (PC2_VAR_COLOR, "PC2"),
        (PC3_VAR_COLOR, "PC3"),
    ]):
        layers, means, stds = agg.get_mean_variance_trajectory(position, pc_idx)
        if not layers:
            continue

        # Plot individual pairs (thin lines)
        for pr in agg.pair_results:
            pos_result = pr.get_position_result(position)
            if pos_result:
                pair_layers, pair_vars = pos_result.get_variance_trajectory(pc_idx)
                if pair_layers:
                    ax.plot(
                        pair_layers, pair_vars,
                        color=color,
                        alpha=PAIR_LINE_ALPHA,
                        linewidth=PAIR_LINE_WIDTH,
                    )

        # Plot mean
        ax.plot(
            layers, means,
            color=color,
            linewidth=MEAN_LINE_WIDTH,
            alpha=MEAN_LINE_ALPHA,
            marker=MEAN_MARKER,
            markersize=MEAN_MARKER_SIZE,
            label=label,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"PCA Variance Explained at Position {position}")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_alignment_trajectory(
    agg: GeoAggregatedResults,
    position: int,
    output_path: Path,
) -> None:
    """Plot PC1 alignment with logit diff direction."""
    if not agg.pair_results:
        return

    # Collect alignments per layer
    layer_alignments: dict[int, list[float]] = {}
    for pr in agg.pair_results:
        pos_result = pr.get_position_result(position)
        if pos_result:
            for lr in pos_result.layer_results:
                if lr.logit_diff_alignment is not None:
                    if lr.layer not in layer_alignments:
                        layer_alignments[lr.layer] = []
                    layer_alignments[lr.layer].append(abs(lr.logit_diff_alignment))

    if not layer_alignments:
        return

    layers = sorted(layer_alignments.keys())
    means = [float(np.mean(layer_alignments[l])) for l in layers]
    stds = [float(np.std(layer_alignments[l])) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    # Plot individual pairs
    for pr in agg.pair_results:
        pos_result = pr.get_position_result(position)
        if pos_result:
            pair_layers = []
            pair_aligns = []
            for lr in pos_result.layer_results:
                if lr.logit_diff_alignment is not None:
                    pair_layers.append(lr.layer)
                    pair_aligns.append(abs(lr.logit_diff_alignment))
            if pair_layers:
                ax.plot(
                    pair_layers, pair_aligns,
                    color=ALIGNMENT_COLOR,
                    alpha=PAIR_LINE_ALPHA,
                    linewidth=PAIR_LINE_WIDTH,
                )

    # Plot mean
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(
        layers,
        means_arr - stds_arr,
        means_arr + stds_arr,
        color=ALIGNMENT_COLOR,
        alpha=0.2,
    )
    ax.plot(
        layers, means,
        color=ALIGNMENT_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("|Cosine Similarity|")
    ax.set_title(f"PC1 Alignment with Logit Diff Direction at Position {position}")
    ax.set_ylim(0, 1)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_pca_scatter_layers(
    agg: GeoAggregatedResults,
    position: int,
    output_dir: Path,
) -> None:
    """Plot 2D PCA scatter at key layers."""
    if not agg.pair_results:
        return

    # Find layers with data
    layers_with_data = set()
    for pr in agg.pair_results:
        pos_result = pr.get_position_result(position)
        if pos_result:
            for lr in pos_result.layer_results:
                layers_with_data.add(lr.layer)

    if not layers_with_data:
        return

    # Select 4 representative layers
    all_layers = sorted(layers_with_data)
    n_layers = len(all_layers)
    if n_layers >= 4:
        # Early, mid-early, mid-late, late
        indices = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
        selected_layers = [all_layers[i] for i in indices]
    else:
        selected_layers = all_layers

    # Create subplot grid
    fig, axes = plt.subplots(1, len(selected_layers), figsize=(4 * len(selected_layers), 4), dpi=DPI)
    if len(selected_layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, selected_layers):
        clean_pcs = []
        corrupted_pcs = []

        for pr in agg.pair_results:
            pos_result = pr.get_position_result(position)
            if pos_result:
                for lr in pos_result.layer_results:
                    if lr.layer == layer and len(lr.clean_mean_pc) >= 2:
                        clean_pcs.append(lr.clean_mean_pc[:2])
                        corrupted_pcs.append(lr.corrupted_mean_pc[:2])

        if clean_pcs and corrupted_pcs:
            clean_arr = np.array(clean_pcs)
            corrupted_arr = np.array(corrupted_pcs)

            ax.scatter(
                clean_arr[:, 0], clean_arr[:, 1],
                c=CLEAN_COLOR, alpha=0.7, s=50, label="Clean"
            )
            ax.scatter(
                corrupted_arr[:, 0], corrupted_arr[:, 1],
                c=CORRUPTED_COLOR, alpha=0.7, s=50, label="Corrupted"
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Layer {layer}")
        ax.legend(loc="upper right", fontsize=8)
        _setup_grid(ax)

    plt.suptitle(f"PCA Projections at Position {position}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_scatter_layers.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_single_trajectory(
    layers: list[int],
    values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
    color: str = SEPARATION_COLOR,
) -> None:
    """Plot a single trajectory."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

    ax.plot(
        layers, values,
        color=color,
        linewidth=MEAN_LINE_WIDTH,
        alpha=MEAN_LINE_ALPHA,
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _get_position_colors(positions: list[int]) -> dict[int, str]:
    """Generate distinct colors for each position."""
    import matplotlib.cm as cm
    n_positions = len(positions)
    cmap = cm.get_cmap("tab20" if n_positions <= 20 else "viridis")
    colors = {}
    for i, pos in enumerate(sorted(positions)):
        colors[pos] = cmap(i / max(1, n_positions - 1))
    return colors


def _plot_consolidated_separation(
    agg: GeoAggregatedResults,
    output_path: Path,
) -> None:
    """Plot separation trajectory for all positions on one figure."""
    positions = agg.positions_analyzed
    if not positions:
        return

    colors = _get_position_colors(positions)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)

    for position in positions:
        layers, means, _ = agg.get_mean_separation_trajectory(position)
        if layers and means:
            ax.plot(
                layers, means,
                color=colors[position],
                linewidth=MEAN_LINE_WIDTH,
                alpha=0.8,
                marker=MEAN_MARKER,
                markersize=MEAN_MARKER_SIZE - 2,
                label=f"pos {position}",
            )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Separation Distance (PC space)")
    ax.set_title("Clean/Corrupted Separation - All Positions")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_consolidated_variance(
    agg: GeoAggregatedResults,
    output_path: Path,
) -> None:
    """Plot PC1 variance trajectory for all positions on one figure."""
    positions = agg.positions_analyzed
    if not positions:
        return

    colors = _get_position_colors(positions)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)

    for position in positions:
        layers, means, _ = agg.get_mean_variance_trajectory(position, pc_idx=0)
        if layers and means:
            ax.plot(
                layers, means,
                color=colors[position],
                linewidth=MEAN_LINE_WIDTH,
                alpha=0.8,
                marker=MEAN_MARKER,
                markersize=MEAN_MARKER_SIZE - 2,
                label=f"pos {position}",
            )

    ax.set_xlabel("Layer")
    ax.set_ylabel("PC1 Explained Variance Ratio")
    ax.set_title("PC1 Variance Explained - All Positions")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_consolidated_alignment(
    agg: GeoAggregatedResults,
    output_path: Path,
) -> None:
    """Plot PC1 alignment with logit diff for all positions on one figure."""
    positions = agg.positions_analyzed
    if not positions:
        return

    colors = _get_position_colors(positions)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
    has_data = False

    for position in positions:
        # Collect alignments per layer
        layer_alignments: dict[int, list[float]] = {}
        for pr in agg.pair_results:
            pos_result = pr.get_position_result(position)
            if pos_result:
                for lr in pos_result.layer_results:
                    if lr.logit_diff_alignment is not None:
                        if lr.layer not in layer_alignments:
                            layer_alignments[lr.layer] = []
                        layer_alignments[lr.layer].append(abs(lr.logit_diff_alignment))

        if layer_alignments:
            layers = sorted(layer_alignments.keys())
            means = [float(np.mean(layer_alignments[l])) for l in layers]
            ax.plot(
                layers, means,
                color=colors[position],
                linewidth=MEAN_LINE_WIDTH,
                alpha=0.8,
                marker=MEAN_MARKER,
                markersize=MEAN_MARKER_SIZE - 2,
                label=f"pos {position}",
            )
            has_data = True

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Layer")
    ax.set_ylabel("|Cosine Similarity|")
    ax.set_title("PC1 Alignment with Logit Diff - All Positions")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _setup_grid(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
