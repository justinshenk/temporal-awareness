"""Visualization for coarse activation patching results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...viz.layer_position_heatmaps import (
    HeatmapConfig,
    _finalize_plot,
    plot_layer_sweep,
    plot_position_sweep,
)
from ..experiments.coarse_activation_patching import CoarseActPatchResults


def visualize_coarse_patching(
    result: CoarseActPatchResults | None,
    output_dir: Path,
    position_labels: list[str] | None = None,
) -> None:
    """Visualize coarse activation patching results.

    Creates:
    - Layer sweep bar chart showing recovery per layer
    - Position sweep heatmap showing recovery by position range
    - Combined summary plot

    Args:
        result: CoarseActPatchResults to visualize
        output_dir: Directory to save plots
        position_labels: Optional labels for positions
    """
    if result is None:
        print("[viz] No coarse patching results to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer sweep visualization
    if result.layer_results:
        _plot_layer_sweep(result, output_dir)

    # Position sweep visualization
    if result.position_results:
        _plot_position_sweep(result, output_dir, position_labels)

    # Combined summary
    if result.layer_results or result.position_results:
        _plot_combined_summary(result, output_dir)

    print(f"[viz] Coarse patching plots saved to {output_dir}")


def _plot_layer_sweep(
    result: CoarseActPatchResults,
    output_dir: Path,
) -> None:
    """Plot layer sweep as bar chart."""
    layers = sorted(result.layer_results.keys())
    recoveries = np.array([result.layer_results[l].score() for l in layers])

    config = HeatmapConfig(
        title="Coarse Patching: Layer Sweep",
        cbar_label="Recovery",
    )

    plot_layer_sweep(
        recoveries,
        layers,
        save_path=output_dir / "coarse_layer_sweep.png",
        config=config,
    )


def _plot_position_sweep(
    result: CoarseActPatchResults,
    output_dir: Path,
    position_labels: list[str] | None = None,
) -> None:
    """Plot position sweep as single-row heatmap."""
    positions = sorted(result.position_results.keys())
    recoveries = np.array([result.position_results[p].score() for p in positions])

    if position_labels is None:
        pos_labels = [f"p{p}" for p in positions]
    else:
        pos_labels = [
            position_labels[p] if p < len(position_labels) else f"p{p}"
            for p in positions
        ]

    config = HeatmapConfig(
        title="Coarse Patching: Position Sweep",
        cbar_label="Recovery",
        vmin=0.0,
        vmax=1.0,
    )

    plot_position_sweep(
        recoveries,
        pos_labels,
        save_path=output_dir / "coarse_position_sweep.png",
        config=config,
    )


def _plot_combined_summary(
    result: CoarseActPatchResults,
    output_dir: Path,
) -> None:
    """Plot combined summary with layer and position sweeps."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Sanity check annotation
    sanity_text = ""
    if result.sanity_result:
        sanity_text = f"Sanity check (all layers, all positions): {result.sanity_result.score():.3f}"

    # Layer sweep subplot
    ax1 = axes[0]
    if result.layer_results:
        layers = sorted(result.layer_results.keys())
        recoveries = [result.layer_results[l].score() for l in layers]

        x = np.arange(len(layers))
        bars = ax1.bar(x, recoveries, color="steelblue", alpha=0.8)

        # Color by value
        max_recovery = max(recoveries) if recoveries else 1.0
        for bar, recovery in zip(bars, recoveries):
            intensity = recovery / max_recovery if max_recovery > 0 else 0
            bar.set_color(plt.cm.RdYlGn(0.3 + 0.6 * intensity))

        ax1.set_xlabel("Layer", fontsize=11)
        ax1.set_ylabel("Recovery", fontsize=11)
        ax1.set_title("Layer Sweep", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
        ax1.set_ylim(bottom=0)

        # Highlight best layers
        best_layers = result.best_layers(n_top=3)
        for layer in best_layers:
            if layer in layers:
                idx = layers.index(layer)
                bars[idx].set_edgecolor("black")
                bars[idx].set_linewidth(2)

    # Position sweep subplot
    ax2 = axes[1]
    if result.position_results:
        positions = sorted(result.position_results.keys())
        recoveries = [result.position_results[p].score() for p in positions]

        x = np.arange(len(positions))
        bars = ax2.bar(x, recoveries, color="steelblue", alpha=0.8)

        max_recovery = max(recoveries) if recoveries else 1.0
        for bar, recovery in zip(bars, recoveries):
            intensity = recovery / max_recovery if max_recovery > 0 else 0
            bar.set_color(plt.cm.RdYlGn(0.3 + 0.6 * intensity))

        ax2.set_xlabel("Position Range Start", fontsize=11)
        ax2.set_ylabel("Recovery", fontsize=11)
        ax2.set_title("Position Sweep", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [f"{p}" for p in positions], fontsize=9, rotation=45, ha="right"
        )
        ax2.set_ylim(bottom=0)

        # Mark threshold line
        ax2.axhline(
            y=0.8, color="red", linestyle="--", alpha=0.5, label="80% threshold"
        )
        ax2.legend(fontsize=9)

    # Add sanity text
    if sanity_text:
        fig.text(
            0.5,
            0.98,
            sanity_text,
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _finalize_plot(output_dir / "coarse_patching_summary.png")
