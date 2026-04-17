"""Overview plots: heatmaps showing the big picture."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ......activation_patching.coarse import CoarseActPatchResults, SweepStepResults
from .comp_constants import COMPONENTS
from .comp_utils import create_figure, save_plot

if TYPE_CHECKING:
    from ......common.position_mapping import SamplePositionMapping


def _get_position_label(pos: int, mapping: "SamplePositionMapping | None") -> str:
    """Semantic position label: format_pos:rel_pos with R_/L_ normalized."""
    if mapping:
        info = mapping.get_position(pos)
        if info and info.format_pos:
            fp = info.format_pos
            if fp.startswith("left_"):
                fp = "L_" + fp[5:]
            elif fp.startswith("right_"):
                fp = "R_" + fp[6:]
            if info.rel_pos is not None:
                return f"{fp}:{info.rel_pos}"
            return fp
    return f"P{pos}"


def plot_overview(
    layer_data: dict[str, SweepStepResults | None],
    pos_data: dict[str, SweepStepResults | None],
    results_by_component: dict[str, CoarseActPatchResults],
    output_dir: Path,
    layer_step_size: int = 1,
    pos_step_size: int = 10,
    position_mapping: "SamplePositionMapping | None" = None,
    agg_by_component: dict | None = None,
) -> None:
    """Generate all overview plots.

    layer_data and pos_data are now POPULATION MEANS (computed by comp_main.py
    from agg_by_component). The overview heatmaps therefore show the mean
    across all pairs, not a single arbitrary pair.
    """
    _plot_layer_heatmap(layer_data, output_dir, "denoising")
    _plot_layer_heatmap(layer_data, output_dir, "noising")
    _plot_layer_heatmap_colnorm(layer_data, output_dir, "denoising")
    _plot_layer_heatmap_colnorm(layer_data, output_dir, "noising")
    _plot_position_heatmap(pos_data, output_dir, "denoising", position_mapping)
    _plot_position_heatmap(pos_data, output_dir, "noising", position_mapping)
    _plot_layer_position_heatmap(layer_data, pos_data, output_dir, position_mapping)


def _build_layer_matrix(
    layer_data: dict[str, SweepStepResults | None],
    mode: Literal["denoising", "noising"],
) -> tuple[np.ndarray, list[int]]:
    """Build layer × component matrix."""
    all_layers = set()
    for comp, data in layer_data.items():
        if data:
            all_layers.update(data.keys())

    if not all_layers:
        return np.array([]), []

    layers = sorted(all_layers)
    n_layers = len(layers)
    n_components = len(COMPONENTS)

    matrix = np.full((n_layers, n_components), np.nan)
    for col_idx, comp in enumerate(COMPONENTS):
        data = layer_data.get(comp)
        if data:
            for row_idx, layer in enumerate(layers):
                if data.get(layer) is not None:
                    val = data[layer].recovery if mode == "denoising" else data[layer].disruption
                    if val is not None:
                        matrix[row_idx, col_idx] = val

    return matrix, layers


def _plot_layer_heatmap(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    mode: Literal["denoising", "noising"],
) -> None:
    """Plot component attribution heatmap for layers with text annotations."""
    matrix, layers = _build_layer_matrix(layer_data, mode)
    if len(layers) == 0:
        return

    n_layers = len(layers)
    n_components = len(COMPONENTS)

    fig, ax = create_figure(figsize=(8, max(10, n_layers * 0.3)))

    # Flip so layer 0 is at bottom
    matrix_flipped = matrix[::-1, :]
    layers_flipped = layers[::-1]

    im = ax.imshow(matrix_flipped, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recovery" if mode == "denoising" else "Disruption")

    # Text annotations for significant values
    if n_layers <= 40:
        for row_idx in range(n_layers):
            for col_idx in range(n_components):
                val = matrix_flipped[row_idx, col_idx]
                if not np.isnan(val) and val > 0.1:
                    text_color = "white" if val > 0.6 else "black"
                    ax.text(col_idx, row_idx, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color, fontweight="bold")

    ax.set_xticks(range(n_components))
    ax.set_xticklabels(COMPONENTS, rotation=45, ha="right")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{lyr}" for lyr in layers_flipped])

    ax.set_xlabel("Component", fontsize=12, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=12, fontweight="bold")

    title = "Denoising Recovery" if mode == "denoising" else "Noising Disruption"
    ax.set_title(f"Component Attribution Heatmap (Layers) - {title}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_plot(fig, output_dir, f"heatmap_layers_{mode}.png")


def _plot_layer_heatmap_colnorm(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    mode: Literal["denoising", "noising"],
) -> None:
    """Plot column-normalized heatmap to reveal fine structure in attn/mlp."""
    matrix, layers = _build_layer_matrix(layer_data, mode)
    if len(layers) == 0:
        return

    n_layers = len(layers)
    n_components = len(COMPONENTS)

    # Column-normalize
    matrix_normalized = np.copy(matrix)
    for col_idx in range(n_components):
        col = matrix[:, col_idx]
        col_valid = col[~np.isnan(col)]
        if len(col_valid) > 0:
            col_min, col_max = col_valid.min(), col_valid.max()
            if col_max > col_min:
                matrix_normalized[:, col_idx] = (col - col_min) / (col_max - col_min)
            else:
                matrix_normalized[:, col_idx] = 0.5

    fig, ax = create_figure(figsize=(8, max(10, n_layers * 0.3)))

    matrix_flipped = matrix_normalized[::-1, :]
    layers_flipped = layers[::-1]

    im = ax.imshow(matrix_flipped, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Column-Normalized Value")

    ax.set_xticks(range(n_components))
    ax.set_xticklabels(COMPONENTS, rotation=45, ha="right")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{lyr}" for lyr in layers_flipped])

    ax.set_xlabel("Component", fontsize=12, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=12, fontweight="bold")

    title = "Denoising Recovery" if mode == "denoising" else "Noising Disruption"
    ax.set_title(f"Column-Normalized Heatmap (Layers) - {title}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_plot(fig, output_dir, f"heatmap_layers_colnorm_{mode}.png")


def _plot_position_heatmap(
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    mode: Literal["denoising", "noising"],
    position_mapping: "SamplePositionMapping | None" = None,
) -> None:
    """Plot position heatmap with UNIFORM spacing and text annotations."""
    all_positions = set()
    for comp, data in pos_data.items():
        if data:
            all_positions.update(data.keys())

    if not all_positions:
        return

    # Use ALL positions for uniform spacing
    positions = sorted(all_positions)
    n_positions = len(positions)
    n_components = len(COMPONENTS)

    # Build matrix - uniform row for each position
    matrix = np.full((n_positions, n_components), np.nan)
    for col_idx, comp in enumerate(COMPONENTS):
        data = pos_data.get(comp)
        if data:
            for row_idx, pos in enumerate(positions):
                if data.get(pos) is not None:
                    val = data[pos].recovery if mode == "denoising" else data[pos].disruption
                    if val is not None:
                        matrix[row_idx, col_idx] = val

    # Flip so position 0 is at bottom
    matrix_flipped = matrix[::-1, :]
    positions_flipped = positions[::-1]

    fig, ax = create_figure(figsize=(8, max(10, n_positions * 0.12)))

    im = ax.imshow(matrix_flipped, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recovery" if mode == "denoising" else "Disruption")

    # Text annotations for significant values (same treatment as layer heatmaps)
    # Show annotations for datasets up to 100 positions, threshold at 0.1
    if n_positions <= 100:
        for row_idx in range(n_positions):
            for col_idx in range(n_components):
                val = matrix_flipped[row_idx, col_idx]
                if not np.isnan(val) and val > 0.1:
                    text_color = "white" if val > 0.6 else "black"
                    # Smaller font for large datasets
                    fontsize = 5 if n_positions > 50 else 6
                    ax.text(col_idx, row_idx, f"{val:.2f}", ha="center", va="center",
                            fontsize=fontsize, color=text_color, fontweight="bold")

    ax.set_xticks(range(n_components))
    ax.set_xticklabels(COMPONENTS, rotation=45, ha="right")

    # Show every Nth position label to avoid crowding
    if n_positions > 40:
        step = max(1, n_positions // 25)
        ax.set_yticks(range(0, n_positions, step))
        ax.set_yticklabels([_get_position_label(positions_flipped[i], position_mapping) for i in range(0, n_positions, step)])
    else:
        ax.set_yticks(range(n_positions))
        ax.set_yticklabels([_get_position_label(p, position_mapping) for p in positions_flipped])

    ax.set_xlabel("Component", fontsize=12, fontweight="bold")
    ax.set_ylabel("Position", fontsize=12, fontweight="bold")

    title = "Denoising Recovery" if mode == "denoising" else "Noising Disruption"
    ax.set_title(f"Component Attribution Heatmap (Positions) - {title}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_plot(fig, output_dir, f"heatmap_positions_{mode}.png")


def _plot_layer_position_heatmap(
    layer_data_all: dict[str, SweepStepResults | None],
    pos_data_all: dict[str, SweepStepResults | None],
    output_dir: Path,
    position_mapping: "SamplePositionMapping | None" = None,
) -> None:
    """Plot Layer × Position 2D localization heatmap using population-mean data.

    Uses resid_post layer and position sweep means for the outer product.
    """
    layer_data = layer_data_all.get("resid_post")
    pos_data = pos_data_all.get("resid_post")

    if not layer_data or not pos_data:
        return

    layers = sorted(layer_data.keys())
    positions = sorted(pos_data.keys())

    if not layers or not positions:
        return

    n_layers = len(layers)
    n_positions = len(positions)

    fig, axes = create_figure(1, 2, figsize=(16, max(8, n_layers * 0.3)))

    for ax_idx, mode in enumerate(["denoising", "noising"]):
        ax = axes[ax_idx]

        if mode == "denoising":
            layer_vals = np.array([layer_data[lyr].recovery or 0 for lyr in layers])
            pos_vals = np.array([pos_data[p].recovery or 0 for p in positions])
        else:
            layer_vals = np.array([layer_data[lyr].disruption or 0 for lyr in layers])
            pos_vals = np.array([pos_data[p].disruption or 0 for p in positions])

        # Normalize
        if layer_vals.max() > 0:
            layer_vals = layer_vals / layer_vals.max()
        if pos_vals.max() > 0:
            pos_vals = pos_vals / pos_vals.max()

        # Outer product as interaction proxy
        interaction = np.outer(layer_vals, pos_vals)
        interaction_flipped = interaction[::-1, :]
        layers_flipped = layers[::-1]

        im = ax.imshow(interaction_flipped, aspect="auto", cmap="hot", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Interaction Strength")

        # Position labels — show as many named positions as possible, small font
        ax.set_xticks(range(n_positions))
        ax.set_xticklabels(
            [_get_position_label(p, position_mapping) for p in positions],
            rotation=90,
            ha="center",
            fontsize=5,
        )

        # Layer labels
        if n_layers > 20:
            step = max(1, n_layers // 15)
            ax.set_yticks(range(0, n_layers, step))
            ax.set_yticklabels([f"L{layers_flipped[i]}" for i in range(0, n_layers, step)])
        else:
            ax.set_yticks(range(n_layers))
            ax.set_yticklabels([f"L{lyr}" for lyr in layers_flipped])

        ax.set_xlabel("Position", fontsize=12, fontweight="bold")
        ax.set_ylabel("Layer", fontsize=12, fontweight="bold")
        title = "Denoising Recovery" if mode == "denoising" else "Noising Disruption"
        ax.set_title(f"{title}", fontsize=12, fontweight="bold")

    fig.suptitle("Layer × Position Importance Map", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(fig, output_dir, "layer_position_heatmap.png")
