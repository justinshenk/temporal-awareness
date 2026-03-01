"""Enhanced heatmap visualizations for patching results.

Extends the base heatmap functionality with:
- Multi-metric comparison views
- Tested/untested region visualization
- Position range highlighting
- Activation vs Attribution comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..activation_patching import ActPatchAggregatedResult
from ..attribution_patching import AttributionSummary
from .layer_position_heatmaps import (
    HeatmapConfig,
    _compute_figsize,
    _compute_range,
    _draw_heatmap,
    _draw_section_markers,
    _finalize_plot,
    _set_title,
    _setup_position_axis,
)


@dataclass
class PatchingHeatmapConfig(HeatmapConfig):
    """Extended config for patching heatmaps.

    Attributes:
        show_untested_as_gray: If True, mask untested cells as NaN (gray)
        highlight_top_n: Number of top cells to highlight with markers
        highlight_color: Color for top cell markers
        show_flip_markers: If True, show markers where choice flipped
    """

    show_untested_as_gray: bool = True
    highlight_top_n: int = 0
    highlight_color: str = "black"
    show_flip_markers: bool = False


def plot_patching_heatmap(
    matrix: np.ndarray,
    layers: list[int],
    position_labels: list[str],
    *,
    tested_mask: np.ndarray | None = None,
    config: PatchingHeatmapConfig | None = None,
    section_markers: dict[str, int] | None = None,
    save_path: Path | None = None,
) -> None:
    """Enhanced heatmap for patching results.

    Features:
    - Gray cells for untested (layer, position) pairs
    - Highlights top N cells with markers
    - Section markers for prompt structure

    Args:
        matrix: 2D array [n_layers, n_positions]
        layers: Layer indices for y-axis
        position_labels: Labels for x-axis
        tested_mask: Boolean array, True = tested, False = untested (shown as gray)
        config: PatchingHeatmapConfig with styling options
        section_markers: Dict of {section_name: position} for vertical markers
        save_path: If provided, save to file
    """
    if config is None:
        config = PatchingHeatmapConfig()

    # Apply tested mask
    plot_matrix = matrix.copy()
    if tested_mask is not None and config.show_untested_as_gray:
        plot_matrix[~tested_mask] = np.nan

    n_layers, n_positions = plot_matrix.shape

    fig, ax = plt.subplots(
        figsize=_compute_figsize(config.figsize, n_layers, n_positions)
    )

    vmin, vmax = _compute_range(plot_matrix, config.vmin, config.vmax, config.cmap)
    im = _draw_heatmap(ax, plot_matrix, config.cmap, vmin, vmax)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(config.cbar_label, rotation=270, labelpad=20, fontsize=10)

    # Y-axis: layers
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=9)

    # X-axis: positions
    _setup_position_axis(ax, position_labels, config.max_labels)

    ax.set_xlabel("Token Position", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    _set_title(ax, config.title, config.subtitle)

    if section_markers:
        _draw_section_markers(ax, section_markers, n_positions)

    # Highlight top N cells
    if config.highlight_top_n > 0:
        _highlight_top_cells(ax, matrix, config.highlight_top_n, config.highlight_color)

    _finalize_plot(save_path)


def plot_multi_metric_heatmap(
    matrices: dict[str, np.ndarray],
    layers: list[int],
    position_labels: list[str],
    *,
    tested_mask: np.ndarray | None = None,
    layout: str = "horizontal",
    section_markers: dict[str, int] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot multiple metrics side by side.

    Args:
        matrices: Dict mapping metric name to 2D array
        layers: Layer indices
        position_labels: Position labels
        tested_mask: Boolean mask for tested cells
        layout: "horizontal", "vertical", or "grid"
        section_markers: Section markers
        save_path: Save path
    """
    n_metrics = len(matrices)
    if n_metrics == 0:
        return

    n_layers, n_positions = next(iter(matrices.values())).shape

    # Determine subplot layout
    if layout == "horizontal":
        n_rows, n_cols = 1, n_metrics
        figsize = (8 * n_metrics, max(6, n_layers * 0.4))
    elif layout == "vertical":
        n_rows, n_cols = n_metrics, 1
        figsize = (max(12, n_positions * 0.15), 5 * n_metrics)
    else:  # grid
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        figsize = (8 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (name, matrix) in enumerate(matrices.items()):
        ax = axes[idx]

        plot_matrix = matrix.copy()
        if tested_mask is not None:
            plot_matrix[~tested_mask] = np.nan

        vmin, vmax = _compute_range(plot_matrix, None, None, "RdYlGn")
        im = _draw_heatmap(ax, plot_matrix, "RdYlGn", vmin, vmax)

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Layer", fontsize=10)
        ax.set_xlabel("Position", fontsize=10)

        # Simplified axis labels
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
        ax.set_yticklabels(
            [f"L{layers[i]}" for i in range(0, n_layers, max(1, n_layers // 10))],
            fontsize=8,
        )

        step = max(1, n_positions // 15)
        ax.set_xticks(range(0, n_positions, step))
        ax.set_xticklabels(
            [
                position_labels[i] if i < len(position_labels) else str(i)
                for i in range(0, n_positions, step)
            ],
            fontsize=7,
            rotation=45,
            ha="right",
        )

        plt.colorbar(im, ax=ax, shrink=0.6)

        if section_markers:
            _draw_section_markers(ax, section_markers, n_positions, y_offset=-2)

    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    _finalize_plot(save_path)


def plot_activation_vs_attribution(
    activation_matrix: np.ndarray,
    attribution_matrix: np.ndarray,
    layers: list[int],
    position_labels: list[str],
    *,
    activation_label: str = "Recovery (Activation)",
    attribution_label: str = "Score (Attribution)",
    section_markers: dict[str, int] | None = None,
    save_path: Path | None = None,
) -> None:
    """Compare activation and attribution patching side by side.

    Args:
        activation_matrix: Recovery scores from activation patching
        attribution_matrix: Attribution scores from attribution patching
        layers: Layer indices
        position_labels: Position labels
        activation_label: Label for activation heatmap
        attribution_label: Label for attribution heatmap
        section_markers: Section markers
        save_path: Save path
    """
    plot_multi_metric_heatmap(
        matrices={
            activation_label: activation_matrix,
            attribution_label: attribution_matrix,
        },
        layers=layers,
        position_labels=position_labels,
        section_markers=section_markers,
        save_path=save_path,
    )


def visualize_activation_patching_result(
    result: ActPatchAggregatedResult,
    position_labels: list[str] | None = None,
    section_markers: dict[str, int] | None = None,
    save_path: Path | None = None,
) -> None:
    """Visualize activation patching results as heatmap.

    Args:
        result: Aggregated activation patching result
        position_labels: Optional position labels
        section_markers: Section markers
        save_path: Save path
    """
    # Build recovery matrix from results
    by_layer = result.get_recovery_by_layer()
    layers = sorted([l for l in by_layer.keys() if l is not None])

    if not layers:
        print("No per-layer results to visualize")
        return

    # Get positions from all pairs
    all_positions = set()
    for pair in result.by_sample.values():
        for target in pair.by_target.keys():
            if target.positions:
                all_positions.update(target.positions)

    if not all_positions:
        print("No per-position results to visualize")
        return

    positions = sorted(all_positions)
    n_layers = len(layers)
    n_positions = max(positions) + 1

    # Build recovery matrix
    matrix = np.full((n_layers, n_positions), np.nan)
    for pair in result.by_sample.values():
        for target, target_result in pair.by_target.items():
            if target.layers and len(target.layers) == 1:
                layer = target.layers[0]
                if layer in layers:
                    layer_idx = layers.index(layer)
                    if target.positions:
                        for pos in target.positions:
                            if 0 <= pos < n_positions:
                                if np.isnan(matrix[layer_idx, pos]):
                                    matrix[layer_idx, pos] = target_result.recovery
                                else:
                                    matrix[layer_idx, pos] = (
                                        matrix[layer_idx, pos] + target_result.recovery
                                    ) / 2

    if position_labels is None:
        position_labels = [f"p{i}" for i in range(n_positions)]

    config = PatchingHeatmapConfig(
        title="Activation Patching Recovery",
        cbar_label="Recovery",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )

    plot_patching_heatmap(
        matrix,
        layers,
        position_labels,
        config=config,
        section_markers=section_markers,
        save_path=save_path,
    )


def visualize_attribution_patching_result(
    result: AttributionSummary,
    position_labels: list[str] | None = None,
    section_markers: dict[str, int] | None = None,
    save_path: Path | None = None,
) -> None:
    """Visualize attribution patching results as heatmaps.

    Creates one heatmap per method in the result.

    Args:
        result: Aggregated attribution patching result
        position_labels: Optional position labels
        section_markers: Section markers
        save_path: Save path (will append method name)
    """
    for name, attr_result in result.results.items():
        layers = attr_result.layers
        n_positions = attr_result.n_positions

        if position_labels is None:
            pos_labels = [f"p{i}" for i in range(n_positions)]
        else:
            pos_labels = position_labels[:n_positions]

        # Use diverging colormap centered at 0 for attribution
        config = PatchingHeatmapConfig(
            title=f"Attribution: {name}",
            cbar_label="Attribution Score",
            cmap="RdBu_r",
        )

        # Determine save path for this method
        if save_path:
            method_path = (
                save_path.parent / f"{save_path.stem}_{name}{save_path.suffix}"
            )
        else:
            method_path = None

        plot_patching_heatmap(
            attr_result.scores,
            layers,
            pos_labels,
            config=config,
            section_markers=section_markers,
            save_path=method_path,
        )


def _highlight_top_cells(
    ax: plt.Axes,
    matrix: np.ndarray,
    n_top: int,
    color: str,
) -> None:
    """Add markers to top N cells by absolute value."""
    flat_indices = np.argsort(np.abs(matrix).ravel())[::-1][:n_top]

    for rank, idx in enumerate(flat_indices):
        row = int(idx // matrix.shape[1])
        col = int(idx % matrix.shape[1])
        val = matrix[row, col]

        if np.isnan(val):
            continue

        # Draw a circle marker
        circle = plt.Circle(
            (col, row), 0.3, fill=False, color=color, linewidth=2, alpha=0.8
        )
        ax.add_patch(circle)

        # Add rank number for top 3
        if rank < 3:
            ax.text(
                col,
                row - 0.45,
                str(rank + 1),
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
                fontweight="bold",
            )
