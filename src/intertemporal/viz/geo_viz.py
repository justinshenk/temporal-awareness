"""Visualization for geometric (PCA) analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from ...common.logging import log
from ..experiments.geo import GeoAggregatedResults, GeoPairResult

if TYPE_CHECKING:
    from ..common.contrastive_preferences import ContrastivePreferences


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

# Time horizon colormap: designed for clear transitions at 0, 1, 10, 50 years
# No horizon = gray, 0-1 years = red, 1-10 years = orange/yellow, 10-50+ = green/blue
HORIZON_NO_VALUE_COLOR = "#808080"  # Gray for samples without time horizon


def _create_horizon_colormap() -> tuple[mcolors.LinearSegmentedColormap, mcolors.BoundaryNorm]:
    """Create a colormap for time horizons with clear transitions.

    Returns:
        (colormap, norm) tuple for use with scatter plots
    """
    # Define color stops for key transitions
    colors = [
        (0.0, "#E74C3C"),    # 0 years - red
        (0.1, "#F39C12"),    # 1 year - orange
        (0.3, "#F1C40F"),    # 5 years - yellow
        (0.5, "#2ECC71"),    # 10 years - green
        (0.8, "#3498DB"),    # 30 years - blue
        (1.0, "#9B59B6"),    # 50+ years - purple
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "horizon",
        [(pos, col) for pos, col in colors],
    )

    # Define boundaries for discrete color stops
    boundaries = [0, 1, 5, 10, 20, 50, 100]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    return cmap, norm


def _get_pair_time_horizons(
    pref_pairs: list["ContrastivePreferences"] | None,
    pair_indices: list[int],
) -> list[float | None]:
    """Extract time horizons for pairs by index.

    Args:
        pref_pairs: List of ContrastivePreferences objects
        pair_indices: Indices of pairs to get horizons for

    Returns:
        List of time horizon values in years (None if no horizon)
    """
    from ...common.time_value import TimeValue

    if not pref_pairs:
        return [None] * len(pair_indices)

    horizons = []
    for idx in pair_indices:
        if idx < len(pref_pairs):
            # Use short_term sample's time_horizon (could be float, dict, or None)
            th = pref_pairs[idx].short_term.time_horizon
            if th is not None:
                if isinstance(th, (int, float)):
                    # Already in years
                    horizons.append(float(th))
                elif isinstance(th, dict):
                    # Convert dict to TimeValue
                    horizons.append(TimeValue.from_dict(th).to_years())
                else:
                    # Assume it's a TimeValue
                    horizons.append(th.to_years())
            else:
                horizons.append(None)
        else:
            horizons.append(None)
    return horizons


def _setup_grid(ax: plt.Axes) -> None:
    """Set up major and minor grid lines."""
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=MINOR_GRID_ALPHA, linewidth=MINOR_GRID_LINE_WIDTH)


def visualize_geo(
    agg: GeoAggregatedResults,
    output_dir: Path,
    pref_pairs: list["ContrastivePreferences"] | None = None,
) -> None:
    """Generate all geo visualizations.

    Creates consolidated plots with all positions on each figure:
    - separation.png: Clean/corrupted separation across layers
    - variance.png: PC1 variance explained across layers
    - alignment.png: PC1 alignment with logit diff across layers
    - pca_scatter.png: 2D PCA projections at key layers (grid of positions x layers)
    - pca_scatter_horizon.png: PCA colored by time horizon (if pref_pairs provided)

    Args:
        agg: Aggregated geo results
        output_dir: Directory to save plots
        pref_pairs: Optional preference pairs for time horizon coloring
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not agg.pair_results:
        log("[geo_viz] No pair results to visualize")
        return

    n_plots = 0

    # Generate consolidated plots (all positions in one figure)
    _plot_consolidated_separation(agg, output_dir / "separation.png")
    n_plots += 1
    _plot_consolidated_variance(agg, output_dir / "variance.png")
    n_plots += 1
    _plot_consolidated_alignment(agg, output_dir / "alignment.png")
    n_plots += 1
    _plot_consolidated_pca_scatter(agg, output_dir / "pca_scatter.png")
    n_plots += 1

    # Time horizon colored PCA scatter (if pref_pairs available)
    if pref_pairs:
        _plot_pca_scatter_by_horizon(agg, pref_pairs, output_dir / "pca_scatter_horizon.png")
        n_plots += 1

    log(f"[geo_viz] Generated {n_plots} plots in {output_dir}")


def visualize_geo_pair(
    result: GeoPairResult,
    output_dir: Path,
) -> None:
    """Generate geo visualizations for a single pair.

    Creates consolidated plots with all positions:
    - separation.png: All positions on one plot
    - variance.png: PC1 variance for all positions

    Args:
        result: Per-pair geo results
        output_dir: Directory to save plots
    """
    if not result.position_results:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = [pr.position for pr in result.position_results if pr.layer_results]
    if not positions:
        return

    colors = _get_position_colors(positions)

    # Separation plot - all positions
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
    has_separation_data = False
    for pos_result in result.position_results:
        if not pos_result.layer_results:
            continue
        layers, separations = pos_result.get_separation_trajectory()
        if layers:
            ax.plot(
                layers, separations,
                color=colors[pos_result.position],
                linewidth=MEAN_LINE_WIDTH,
                alpha=0.8,
                marker=MEAN_MARKER,
                markersize=MEAN_MARKER_SIZE - 2,
                label=f"pos {pos_result.position}",
            )
            has_separation_data = True

    if has_separation_data:
        ax.set_xlabel("Layer")
        ax.set_ylabel("Separation Distance (PC space)")
        ax.set_title("Clean/Corrupted Separation - All Positions")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        _setup_grid(ax)
        plt.tight_layout()
        plt.savefig(output_dir / "separation.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Variance plot - PC1 for all positions
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
    has_variance_data = False
    for pos_result in result.position_results:
        if not pos_result.layer_results:
            continue
        layers, var_pc1 = pos_result.get_variance_trajectory(0)
        if layers:
            ax.plot(
                layers, var_pc1,
                color=colors[pos_result.position],
                linewidth=MEAN_LINE_WIDTH,
                alpha=0.8,
                marker=MEAN_MARKER,
                markersize=MEAN_MARKER_SIZE - 2,
                label=f"pos {pos_result.position}",
            )
            has_variance_data = True

    if has_variance_data:
        ax.set_xlabel("Layer")
        ax.set_ylabel("PC1 Explained Variance Ratio")
        ax.set_title("PC1 Variance Explained - All Positions")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        _setup_grid(ax)
        plt.tight_layout()
        plt.savefig(output_dir / "variance.png", dpi=DPI, bbox_inches="tight")
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


def _plot_consolidated_pca_scatter(
    agg: GeoAggregatedResults,
    output_path: Path,
) -> None:
    """Plot 2D PCA scatter for all positions at key layers.

    Creates a grid: rows = positions, columns = representative layers.
    """
    positions = agg.positions_analyzed
    if not positions or not agg.pair_results:
        return

    # Find all layers with data
    layers_with_data: set[int] = set()
    for pr in agg.pair_results:
        for pos_result in pr.position_results:
            for lr in pos_result.layer_results:
                layers_with_data.add(lr.layer)

    if not layers_with_data:
        return

    # Select 4 representative layers
    all_layers = sorted(layers_with_data)
    n_layers = len(all_layers)
    if n_layers >= 4:
        indices = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
        selected_layers = [all_layers[i] for i in indices]
    else:
        selected_layers = all_layers

    n_positions = len(positions)
    n_cols = len(selected_layers)

    # Create subplot grid: positions x layers
    fig, axes = plt.subplots(
        n_positions, n_cols,
        figsize=(3 * n_cols, 3 * n_positions),
        dpi=DPI,
        squeeze=False,
    )

    for row_idx, position in enumerate(sorted(positions)):
        for col_idx, layer in enumerate(selected_layers):
            ax = axes[row_idx, col_idx]
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
                    c=CLEAN_COLOR, alpha=0.7, s=30, label="Clean"
                )
                ax.scatter(
                    corrupted_arr[:, 0], corrupted_arr[:, 1],
                    c=CORRUPTED_COLOR, alpha=0.7, s=30, label="Corrupted"
                )

            ax.set_xlabel("PC1", fontsize=8)
            ax.set_ylabel("PC2", fontsize=8)
            ax.tick_params(labelsize=7)

            # Row label (position)
            if col_idx == 0:
                ax.set_ylabel(f"Pos {position}\nPC2", fontsize=8)

            # Column label (layer)
            if row_idx == 0:
                ax.set_title(f"Layer {layer}", fontsize=9)

            # Legend only on first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=6)

            _setup_grid(ax)

    plt.suptitle("PCA Projections - All Positions", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_pca_scatter_by_horizon(
    agg: GeoAggregatedResults,
    pref_pairs: list["ContrastivePreferences"],
    output_path: Path,
) -> None:
    """Plot PCA scatter with points colored by time horizon.

    Creates a grid: rows = positions, columns = representative layers.
    Points are colored by time horizon in years:
    - Gray: No time horizon
    - Red: 0-1 years
    - Orange/Yellow: 1-10 years
    - Green/Blue: 10-50+ years
    """
    positions = agg.positions_analyzed
    if not positions or not agg.pair_results:
        return

    # Find all layers with data
    layers_with_data: set[int] = set()
    for pr in agg.pair_results:
        for pos_result in pr.position_results:
            for lr in pos_result.layer_results:
                layers_with_data.add(lr.layer)

    if not layers_with_data:
        return

    # Select 4 representative layers
    all_layers = sorted(layers_with_data)
    n_layers = len(all_layers)
    if n_layers >= 4:
        indices = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
        selected_layers = [all_layers[i] for i in indices]
    else:
        selected_layers = all_layers

    # Get time horizons for all pairs
    pair_indices = [pr.pair_idx for pr in agg.pair_results]
    horizons = _get_pair_time_horizons(pref_pairs, pair_indices)

    # Create colormap
    cmap, norm = _create_horizon_colormap()

    n_positions = len(positions)
    n_cols = len(selected_layers)

    # Create subplot grid: positions x layers
    fig, axes = plt.subplots(
        n_positions, n_cols,
        figsize=(3.5 * n_cols, 3 * n_positions),
        dpi=DPI,
        squeeze=False,
    )

    for row_idx, position in enumerate(sorted(positions)):
        for col_idx, layer in enumerate(selected_layers):
            ax = axes[row_idx, col_idx]

            # Collect data for this position/layer
            clean_pcs = []
            corrupted_pcs = []
            point_horizons = []
            no_horizon_clean = []
            no_horizon_corrupted = []

            for pair_idx, pr in enumerate(agg.pair_results):
                pos_result = pr.get_position_result(position)
                if pos_result:
                    for lr in pos_result.layer_results:
                        if lr.layer == layer and len(lr.clean_mean_pc) >= 2:
                            horizon = horizons[pair_idx]
                            if horizon is not None:
                                clean_pcs.append(lr.clean_mean_pc[:2])
                                corrupted_pcs.append(lr.corrupted_mean_pc[:2])
                                point_horizons.append(horizon)
                            else:
                                no_horizon_clean.append(lr.clean_mean_pc[:2])
                                no_horizon_corrupted.append(lr.corrupted_mean_pc[:2])

            # Plot points without horizon in gray
            if no_horizon_clean:
                nh_clean = np.array(no_horizon_clean)
                ax.scatter(
                    nh_clean[:, 0], nh_clean[:, 1],
                    c=HORIZON_NO_VALUE_COLOR, alpha=0.4, s=20,
                    marker="o", label="No horizon"
                )
            if no_horizon_corrupted:
                nh_corrupted = np.array(no_horizon_corrupted)
                ax.scatter(
                    nh_corrupted[:, 0], nh_corrupted[:, 1],
                    c=HORIZON_NO_VALUE_COLOR, alpha=0.4, s=20, marker="x"
                )

            # Plot points with horizon colored by value
            if clean_pcs and point_horizons:
                clean_arr = np.array(clean_pcs)
                corrupted_arr = np.array(corrupted_pcs)
                horizon_arr = np.array(point_horizons)

                sc = ax.scatter(
                    clean_arr[:, 0], clean_arr[:, 1],
                    c=horizon_arr, cmap=cmap, norm=norm,
                    alpha=0.8, s=35, marker="o", edgecolors="black", linewidths=0.3
                )
                ax.scatter(
                    corrupted_arr[:, 0], corrupted_arr[:, 1],
                    c=horizon_arr, cmap=cmap, norm=norm,
                    alpha=0.8, s=35, marker="x", linewidths=1.5
                )

            ax.set_xlabel("PC1", fontsize=8)
            ax.set_ylabel("PC2", fontsize=8)
            ax.tick_params(labelsize=7)

            # Row label (position)
            if col_idx == 0:
                ax.set_ylabel(f"Pos {position}\nPC2", fontsize=8)

            # Column label (layer)
            if row_idx == 0:
                ax.set_title(f"Layer {layer}", fontsize=9)

            _setup_grid(ax)

    # Add colorbar
    if clean_pcs and point_horizons:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Time Horizon (years)", fontsize=9)
        cbar.set_ticks([0.5, 3, 7.5, 15, 35, 75])
        cbar.set_ticklabels(["0-1", "1-5", "5-10", "10-20", "20-50", "50+"])

    # Legend for marker shapes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Clean (circle)'),
        Line2D([0], [0], marker='x', color='gray', markerfacecolor='w',
               markersize=8, label='Corrupted (x)'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=8,
               bbox_to_anchor=(0.91, 0.02))

    plt.suptitle("PCA Projections - Colored by Time Horizon", fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
