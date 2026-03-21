"""Redundancy analysis plots: noising vs denoising comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from .....activation_patching.coarse import SweepStepResults
from .constants import COMPONENTS, COMPONENT_COLORS
from .utils import adjust_labels, create_figure, get_sqrt_colors, save_plot, setup_grid


def plot_redundancy(
    layer_data: dict[str, SweepStepResults | None],
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Generate all redundancy analysis plots."""
    _plot_noise_vs_denoise(layer_data, output_dir, "layer")
    _plot_noise_vs_denoise(pos_data, output_dir, "position")
    _plot_redundancy_gap(layer_data, output_dir)
    _plot_redundancy_gap_sorted(layer_data, output_dir)
    _plot_difference_heatmap(layer_data, output_dir, "layer")
    _plot_difference_heatmap(pos_data, output_dir, "position")


def _plot_noise_vs_denoise(
    data: dict[str, SweepStepResults | None],
    output_dir: Path,
    sweep_type: Literal["layer", "position"],
) -> None:
    """Plot noising vs denoising scatter for each component with ALL points labeled."""
    # Get all indices
    all_indices = []
    for comp in COMPONENTS:
        comp_data = data.get(comp)
        if comp_data:
            all_indices.extend(comp_data.keys())

    if all_indices:
        max_idx, min_idx = max(all_indices), min(all_indices)
    else:
        max_idx, min_idx = 1, 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="white")
    axes = axes.flatten()

    for ax_idx, comp in enumerate(COMPONENTS):
        ax = axes[ax_idx]
        ax.set_facecolor("white")

        comp_data = data.get(comp)
        if not comp_data:
            ax.text(0.5, 0.5, f"No {comp} data", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        indices = sorted(comp_data.keys())
        recoveries, disruptions, labels, raw_indices = [], [], [], []

        for idx in indices:
            rec = comp_data[idx].recovery
            dis = comp_data[idx].disruption
            if rec is not None and dis is not None:
                recoveries.append(rec)
                disruptions.append(dis)
                labels.append(f"{'L' if sweep_type == 'layer' else 'P'}{idx}")
                raw_indices.append(idx)

        if not recoveries:
            ax.text(0.5, 0.5, f"No valid {comp} data", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        # Colors with sqrt mapping
        color_values = get_sqrt_colors(raw_indices)

        ax.scatter(recoveries, disruptions, c=color_values, cmap="viridis", vmin=0, vmax=1,
                   s=80, edgecolors="black", linewidth=0.5, alpha=0.8)

        # Label EVERY point
        texts = [ax.text(x, y, label, fontsize=6, alpha=0.8)
                 for x, y, label in zip(recoveries, disruptions, labels)]
        adjust_labels(texts, ax)

        # Reference elements
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x (equal effect)")
        # 0.5 reference lines create clear quadrants for "high effect" vs "low effect"
        # Higher alpha (0.6) for better visibility as requested
        ax.axhline(y=0.5, color="red", linestyle="-", alpha=0.6, linewidth=2)
        ax.axvline(x=0.5, color="red", linestyle="-", alpha=0.6, linewidth=2)

        # AND/OR labels
        ax.text(0.25, 0.75, "AND", fontsize=24, fontweight="bold", color="gray", alpha=0.15,
                ha="center", va="center", zorder=0)
        ax.text(0.75, 0.25, "OR", fontsize=24, fontweight="bold", color="gray", alpha=0.15,
                ha="center", va="center", zorder=0)

        ax.set_xlabel("Denoising Recovery", fontsize=10, fontweight="bold")
        ax.set_ylabel("Noising Disruption", fontsize=10, fontweight="bold")
        ax.set_title(f"{comp} ({sweep_type.title()})", fontsize=12, fontweight="bold",
                     color=COMPONENT_COLORS[comp])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        setup_grid(ax)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=min_idx, vmax=max_idx))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{'Layer' if sweep_type == 'layer' else 'Position'}", fontsize=10, fontweight="bold")

    plt.subplots_adjust(right=0.90)
    save_plot(fig, output_dir, f"noise_vs_denoise_per_component_{sweep_type}.png")


def _plot_redundancy_gap(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot redundancy gap per layer with trend lines."""
    all_layers = set()
    for comp, data in layer_data.items():
        if data:
            all_layers.update(data.keys())

    if not all_layers:
        return

    layers = sorted(all_layers)
    n_layers = len(layers)
    plot_components = ["attn_out", "mlp_out", "resid_post"]

    bar_width = 0.25
    x = np.arange(n_layers)

    fig, ax = create_figure(figsize=(max(12, n_layers * 0.4), 6))

    gaps_by_comp = {}
    for i, comp in enumerate(plot_components):
        data = layer_data.get(comp)
        gaps = []
        for layer in layers:
            if data and data.get(layer) is not None:
                rec = data[layer].recovery
                dis = data[layer].disruption
                if rec is not None and dis is not None:
                    gaps.append(dis - rec)
                else:
                    gaps.append(0)
            else:
                gaps.append(0)

        gaps_by_comp[comp] = gaps
        ax.bar(x + i * bar_width, gaps, bar_width, label=comp, color=COMPONENT_COLORS[comp], alpha=0.8)

        # Trend line
        if len(gaps) > 3:
            window = min(5, len(gaps) // 2)
            smoothed = np.convolve(gaps, np.ones(window) / window, mode="valid")
            smoothed_x = np.arange(window // 2, len(gaps) - window // 2)
            ax.plot(smoothed_x + i * bar_width, smoothed, color=COMPONENT_COLORS[comp],
                    linewidth=2, linestyle="--", alpha=0.7)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Redundancy Gap (Disruption - Recovery)", fontsize=12, fontweight="bold")
    ax.set_title("Redundancy Gap per Layer per Component", fontsize=14, fontweight="bold")

    # Tick labels - always rotate 45° and show every other label if cramped
    # Lower threshold to 15 to handle L18-L34 range (17 layers) mentioned in feedback
    ax.set_xticks(x + bar_width)
    if n_layers > 15:
        # Show every other label to avoid cramping
        tick_labels = [f"L{lyr}" if i % 2 == 0 else "" for i, lyr in enumerate(layers)]
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticklabels([f"L{lyr}" for lyr in layers], rotation=45, ha="right", fontsize=9)

    ax.legend(loc="best")
    setup_grid(ax)

    # Background labels
    ax.text(0.08, 0.75, "Necessity", transform=ax.transAxes, fontsize=36, fontweight="bold",
            color="gray", alpha=0.12, ha="left", va="center", zorder=0)
    ax.text(0.08, 0.25, "Sufficiency", transform=ax.transAxes, fontsize=36, fontweight="bold",
            color="gray", alpha=0.12, ha="left", va="center", zorder=0)

    plt.tight_layout()
    save_plot(fig, output_dir, "redundancy_gap.png")


def _plot_redundancy_gap_sorted(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot redundancy gap sorted by absolute magnitude."""
    all_layers = set()
    for comp, data in layer_data.items():
        if data:
            all_layers.update(data.keys())

    if not all_layers:
        return

    layers = sorted(all_layers)
    plot_components = ["attn_out", "mlp_out", "resid_post"]

    # Build gaps
    gaps_by_comp = {}
    for comp in plot_components:
        data = layer_data.get(comp)
        gaps = []
        for layer in layers:
            if data and data.get(layer) is not None:
                rec = data[layer].recovery
                dis = data[layer].disruption
                if rec is not None and dis is not None:
                    gaps.append(dis - rec)
                else:
                    gaps.append(0)
            else:
                gaps.append(0)
        gaps_by_comp[comp] = gaps

    # Sort by total absolute gap
    total_abs_gap = []
    for i, layer in enumerate(layers):
        total = sum(abs(gaps_by_comp[comp][i]) for comp in plot_components)
        total_abs_gap.append((layer, total, i))

    sorted_by_gap = sorted(total_abs_gap, key=lambda x: x[1], reverse=True)
    sorted_layers = [x[0] for x in sorted_by_gap]
    sorted_indices = [x[2] for x in sorted_by_gap]

    n_layers = len(sorted_layers)
    bar_width = 0.25
    x = np.arange(n_layers)

    fig, ax = create_figure(figsize=(max(12, n_layers * 0.4), 6))

    for i, comp in enumerate(plot_components):
        sorted_gaps = [gaps_by_comp[comp][idx] for idx in sorted_indices]
        ax.bar(x + i * bar_width, sorted_gaps, bar_width, label=comp, color=COMPONENT_COLORS[comp], alpha=0.8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Layer (sorted by |gap|)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Redundancy Gap (Disruption - Recovery)", fontsize=12, fontweight="bold")
    ax.set_title("Redundancy Gap - Sorted by Magnitude", fontsize=14, fontweight="bold")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"L{lyr}" for lyr in sorted_layers], rotation=45, ha="right")
    ax.legend(loc="best")
    setup_grid(ax)

    plt.tight_layout()
    save_plot(fig, output_dir, "redundancy_gap_sorted.png")


def _plot_difference_heatmap(
    data: dict[str, SweepStepResults | None],
    output_dir: Path,
    sweep_type: Literal["layer", "position"],
) -> None:
    """Plot difference heatmap (noising - denoising) showing redundancy per cell."""
    all_indices = set()
    for comp, comp_data in data.items():
        if comp_data:
            all_indices.update(comp_data.keys())

    if not all_indices:
        return

    indices = sorted(all_indices)
    n_indices = len(indices)
    n_components = len(COMPONENTS)

    # Build matrix
    matrix = np.full((n_indices, n_components), np.nan)
    for col_idx, comp in enumerate(COMPONENTS):
        comp_data = data.get(comp)
        if comp_data:
            for row_idx, idx in enumerate(indices):
                if comp_data.get(idx) is not None:
                    rec = comp_data[idx].recovery
                    dis = comp_data[idx].disruption
                    if rec is not None and dis is not None:
                        matrix[row_idx, col_idx] = dis - rec

    fig_height = max(10, n_indices * 0.3) if sweep_type == "layer" else max(10, n_indices * 0.15)
    fig, ax = create_figure(figsize=(8, fig_height))

    matrix_flipped = matrix[::-1, :]
    indices_flipped = indices[::-1]

    # Diverging colormap
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    im = ax.imshow(matrix_flipped, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Redundancy Gap (Disruption - Recovery)")

    # Text annotations
    if n_indices <= 40:
        for row_idx in range(n_indices):
            for col_idx in range(n_components):
                val = matrix_flipped[row_idx, col_idx]
                if not np.isnan(val) and abs(val) > 0.05:
                    text_color = "white" if abs(val) > vmax * 0.5 else "black"
                    ax.text(col_idx, row_idx, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color, fontweight="bold")

    ax.set_xticks(range(n_components))
    ax.set_xticklabels(COMPONENTS, rotation=45, ha="right")

    prefix = "L" if sweep_type == "layer" else "P"
    if n_indices > 30:
        step = max(1, n_indices // 20)
        ax.set_yticks(range(0, n_indices, step))
        ax.set_yticklabels([f"{prefix}{indices_flipped[i]}" for i in range(0, n_indices, step)])
    else:
        ax.set_yticks(range(n_indices))
        ax.set_yticklabels([f"{prefix}{idx}" for idx in indices_flipped])

    ax.set_xlabel("Component", fontsize=12, fontweight="bold")
    ax.set_ylabel(sweep_type.title(), fontsize=12, fontweight="bold")
    ax.set_title(f"Redundancy Gap Heatmap ({sweep_type.title()}s)\n+ve=Necessity, -ve=Sufficiency",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_plot(fig, output_dir, f"difference_heatmap_{sweep_type}.png")
