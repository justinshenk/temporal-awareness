"""Publication-quality visualizations for activation patching results.

Creates heatmaps with token labels, scatter plots, and line plots
matching reference images from mechanistic interpretability papers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def plot_layer_position_heatmap(
    layer_recovery: dict[int, float] | None = None,
    position_recovery: dict[int, float] | None = None,
    position_layer_matrix: np.ndarray | None = None,
    tokens: list[str] | None = None,
    title: str = "Activation Patching: Layer vs Position",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> plt.Figure:
    """Create publication-quality heatmap with token labels.

    Can work with either:
    1. Separate layer_recovery and position_recovery dicts
    2. A full position_layer_matrix [n_layers, n_positions]

    Args:
        layer_recovery: Dict mapping layer -> mean_recovery
        position_recovery: Dict mapping position -> mean_recovery
        position_layer_matrix: 2D array [n_layers, n_positions] of recovery values
        tokens: List of token strings for x-axis labels
        title: Plot title
        save_path: Path to save figure (if None, displays)
        figsize: Figure dimensions

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if position_layer_matrix is not None:
        matrix = position_layer_matrix
        n_layers, n_positions = matrix.shape
    elif layer_recovery and position_recovery:
        # Create 1D visualizations side by side
        return _plot_sweep_results(
            layer_recovery, position_recovery, tokens, title, save_path, figsize
        )
    else:
        ax.text(0.5, 0.5, "No data provided", ha="center", va="center", fontsize=14)
        _finalize_plot(save_path, fig)
        return fig

    # Diverging colormap centered at 0
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    # Token labels on x-axis with position numbers
    if tokens:
        token_labels = [f"{tok}_{i}" for i, tok in enumerate(tokens)]
    else:
        token_labels = [str(i) for i in range(n_positions)]

    ax.set_xticks(range(n_positions))
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)

    # Layer labels on y-axis
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="Recovery", shrink=0.8)

    ax.set_xlabel("Position (Token)", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    _finalize_plot(save_path, fig)
    return fig


def _plot_sweep_results(
    layer_recovery: dict[int, float],
    position_recovery: dict[int, float],
    tokens: list[str] | None,
    title: str,
    save_path: Path | str | None,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Plot layer and position sweep results side by side."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Layer sweep heatmap (1D)
    ax1 = axes[0]
    layers = sorted([l for l in layer_recovery.keys() if l is not None])
    recoveries = [layer_recovery[l] for l in layers]

    # Create 2D matrix for heatmap style
    matrix = np.array(recoveries).reshape(-1, 1)
    vmax = max(abs(min(recoveries)), abs(max(recoveries)), 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im1 = ax1.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels([f"L{l}" for l in layers], fontsize=9)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["Recovery"], fontsize=10)
    ax1.set_title("Layer Sweep", fontsize=12, fontweight="bold")

    # Add value annotations
    for i, val in enumerate(recoveries):
        ax1.text(
            0,
            i,
            f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(val) > vmax * 0.6 else "black",
        )

    plt.colorbar(im1, ax=ax1, label="Recovery", shrink=0.8)

    # Right: Position sweep heatmap (1D horizontal)
    ax2 = axes[1]
    positions = sorted(position_recovery.keys())
    pos_recoveries = [position_recovery[p] for p in positions]

    matrix2 = np.array(pos_recoveries).reshape(1, -1)
    vmax2 = max(abs(min(pos_recoveries)), abs(max(pos_recoveries)), 0.1)
    norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
    im2 = ax2.imshow(matrix2, cmap="RdBu_r", norm=norm2, aspect="auto")

    # Position labels
    if tokens and len(tokens) >= len(positions):
        pos_labels = [
            f"{tokens[p]}_{p}" if p < len(tokens) else str(p) for p in positions
        ]
    else:
        pos_labels = [str(p) for p in positions]

    ax2.set_xticks(range(len(positions)))
    ax2.set_xticklabels(pos_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks([0])
    ax2.set_yticklabels(["Recovery"], fontsize=10)
    ax2.set_title("Position Sweep", fontsize=12, fontweight="bold")

    plt.colorbar(
        im2, ax=ax2, label="Recovery", shrink=0.8, orientation="horizontal", pad=0.15
    )

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _finalize_plot(save_path, fig)
    return fig


def plot_layer_metrics_line(
    layer_recovery: dict[int, float],
    flip_rates: dict[int, float] | None = None,
    title: str = "Layer-by-Layer Patching Metrics",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Line plot showing multiple metrics across layers.

    Args:
        layer_recovery: Dict mapping layer -> recovery
        flip_rates: Optional dict mapping layer -> flip_rate
        title: Plot title
        save_path: Path to save figure
        figsize: Figure dimensions

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = sorted([l for l in layer_recovery.keys() if l is not None])
    recoveries = [layer_recovery[l] for l in layers]

    # Recovery line
    ax.plot(layers, recoveries, "b-o", label="Recovery", linewidth=2, markersize=6)

    # Flip rate line (if available)
    if flip_rates:
        rates = [flip_rates.get(l, 0) for l in layers]
        ax.plot(layers, rates, "r--s", label="Flip Rate", linewidth=2, markersize=6)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Mark best layer
    best_layer = max(layers, key=lambda l: layer_recovery[l])
    best_val = layer_recovery[best_layer]
    ax.annotate(
        f"Best: L{best_layer}\n({best_val:.3f})",
        xy=(best_layer, best_val),
        xytext=(best_layer + 1, best_val + 0.05),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=9,
        color="green",
    )

    plt.tight_layout()
    _finalize_plot(save_path, fig)
    return fig


def plot_attribution_vs_activation_scatter(
    attribution_values: list[float],
    activation_values: list[float],
    layers: list[int] | None = None,
    title: str = "Attribution vs Activation Patching",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Scatter plot comparing attribution vs activation patching effects.

    Args:
        attribution_values: Attribution patching effects
        activation_values: Activation patching effects
        layers: Layer indices for coloring
        title: Plot title
        save_path: Path to save figure
        figsize: Figure dimensions

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if layers:
        n_layers = max(layers) + 1
        colors = plt.cm.viridis(np.array(layers) / n_layers)
        scatter = ax.scatter(
            activation_values,
            attribution_values,
            c=colors,
            s=80,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, n_layers))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Layer")
    else:
        ax.scatter(
            activation_values,
            attribution_values,
            c="steelblue",
            s=80,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    # Diagonal line (perfect correlation)
    lims = [
        min(min(activation_values), min(attribution_values)) - 0.05,
        max(max(activation_values), max(attribution_values)) + 0.05,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x", linewidth=1.5)

    # Zero lines
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Activation Patching Effect", fontsize=11)
    ax.set_ylabel("Attribution Patching Effect", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Correlation coefficient
    if len(activation_values) > 1:
        corr = np.corrcoef(activation_values, attribution_values)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    _finalize_plot(save_path, fig)
    return fig


def plot_component_position_heatmap(
    data: dict[str, dict[int, float]],
    tokens: list[str] | None = None,
    title: str = "Component Attribution by Position",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Heatmap of components vs positions.

    Args:
        data: Dict mapping component_name -> {position: value}
        tokens: Token strings for x-axis labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure dimensions

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    components = list(data.keys())
    all_positions = sorted(
        set(p for comp_data in data.values() for p in comp_data.keys())
    )

    # Build matrix
    matrix = np.zeros((len(components), len(all_positions)))
    for i, comp in enumerate(components):
        for j, pos in enumerate(all_positions):
            matrix[i, j] = data[comp].get(pos, 0)

    # Plot
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    # Labels
    if tokens and len(tokens) >= len(all_positions):
        pos_labels = [
            f"{tokens[p]}_{p}" if p < len(tokens) else str(p) for p in all_positions
        ]
    else:
        pos_labels = [str(p) for p in all_positions]

    ax.set_xticks(range(len(all_positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components, fontsize=9)

    plt.colorbar(im, ax=ax, label="Attribution", shrink=0.8)

    ax.set_xlabel("Position", fontsize=11)
    ax.set_ylabel("Component", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    _finalize_plot(save_path, fig)
    return fig


def plot_sweep_summary(
    sweep_results: Any,  # SweepResults
    tokens: list[str] | None = None,
    title: str = "Activation Patching Sweep Summary",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (16, 10),
) -> plt.Figure:
    """Create comprehensive sweep summary with multiple panels.

    Args:
        sweep_results: SweepResults object from run_full_sweep
        tokens: Token strings for position labels
        title: Main title
        save_path: Path to save figure
        figsize: Figure dimensions

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid: 2 rows, 2 cols
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top left: Layer recovery bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    layers = sorted([l for l in sweep_results.layer_recovery.keys() if l is not None])
    recoveries = [sweep_results.layer_recovery[l] for l in layers]
    colors = [
        "forestgreen" if l in sweep_results.best_layers else "steelblue" for l in layers
    ]
    ax1.bar(layers, recoveries, color=colors, alpha=0.8)
    ax1.set_xlabel("Layer", fontsize=10)
    ax1.set_ylabel("Recovery", fontsize=10)
    ax1.set_title("Layer Sweep (green = best)", fontsize=11, fontweight="bold")
    ax1.axhline(y=0, color="black", linewidth=0.5)

    # Top right: Position recovery bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    positions = sorted(sweep_results.position_recovery.keys())
    pos_recoveries = [sweep_results.position_recovery[p] for p in positions]
    colors2 = [
        "forestgreen" if p in sweep_results.best_positions else "coral"
        for p in positions
    ]
    ax2.bar(range(len(positions)), pos_recoveries, color=colors2, alpha=0.8)

    if tokens and len(tokens) >= len(positions):
        pos_labels = [
            f"{tokens[p][:4]}_{p}" if p < len(tokens) else str(p) for p in positions
        ]
    else:
        pos_labels = [str(p) for p in positions]
    ax2.set_xticks(range(len(positions)))
    ax2.set_xticklabels(pos_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Position", fontsize=10)
    ax2.set_ylabel("Recovery", fontsize=10)
    ax2.set_title("Position Sweep (green = best)", fontsize=11, fontweight="bold")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # Bottom left: Layer line plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(layers, recoveries, "b-o", linewidth=2, markersize=6)
    for l in sweep_results.best_layers:
        if l in layers:
            idx = layers.index(l)
            ax3.scatter([l], [recoveries[idx]], c="red", s=100, zorder=5, marker="*")
    ax3.set_xlabel("Layer", fontsize=10)
    ax3.set_ylabel("Recovery", fontsize=10)
    ax3.set_title("Layer Recovery Trend", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Bottom right: Summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    best_layer = sweep_results.best_layers[0] if sweep_results.best_layers else "N/A"
    best_pos = sweep_results.best_positions[:3] if sweep_results.best_positions else []
    max_layer_rec = max(recoveries) if recoveries else 0
    max_pos_rec = max(pos_recoveries) if pos_recoveries else 0

    summary = (
        f"SWEEP SUMMARY\n"
        f"{'=' * 40}\n\n"
        f"Best Layers: {sweep_results.best_layers}\n"
        f"  Max recovery: {max_layer_rec:.4f} at L{best_layer}\n\n"
        f"Best Positions: {best_pos}\n"
        f"  Max recovery: {max_pos_rec:.4f}\n\n"
        f"Total layers tested: {len(layers)}\n"
        f"Total positions tested: {len(positions)}\n"
    )

    ax4.text(
        0.1,
        0.9,
        summary,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    _finalize_plot(save_path, fig)
    return fig


def _finalize_plot(save_path: Path | str | None, fig: plt.Figure = None) -> None:
    """Save or show plot."""
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
