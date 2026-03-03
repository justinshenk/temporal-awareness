"""Visualization for attribution patching results."""

from __future__ import annotations

from pathlib import Path

from ...attribution_patching import AttributionSummary
from ...viz.patching_heatmaps import (
    PatchingHeatmapConfig,
    plot_multi_metric_heatmap,
    plot_patching_heatmap,
)


def visualize_att_patching(
    result: AttributionSummary | None,
    output_dir: Path,
    position_labels: list[str] | None = None,
    section_markers: dict[str, int] | None = None,
) -> None:
    """Visualize attribution patching results.

    Creates heatmaps for each attribution method showing scores
    across layers and positions.

    Args:
        result: AttributionSummary to visualize
        output_dir: Directory to save plots
        position_labels: Optional labels for positions
        section_markers: Optional section markers for prompt structure
    """
    if result is None or not result.results:
        print("[viz] No attribution patching results to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot individual heatmaps for each method
    for name, attr_result in result.results.items():
        if attr_result.scores.size == 0:
            continue

        layers = attr_result.layers
        n_positions = attr_result.n_positions

        if position_labels is None:
            pos_labels = [f"p{i}" for i in range(n_positions)]
        else:
            pos_labels = position_labels[:n_positions]

        config = PatchingHeatmapConfig(
            title=f"Attribution Patching: {name}",
            subtitle=f"{attr_result.n_layers} layers, {n_positions} positions",
            cbar_label="Attribution Score",
            cmap="RdBu_r",
        )

        save_path = output_dir / f"att_patching_{name}.png"

        plot_patching_heatmap(
            attr_result.scores,
            layers,
            pos_labels,
            config=config,
            section_markers=section_markers,
            save_path=save_path,
        )

    # Plot multi-method comparison if multiple methods
    if len(result.results) > 1:
        matrices = {
            name: r.scores for name, r in result.results.items() if r.scores.size > 0
        }

        if matrices:
            # Use consistent dimensions
            first_result = next(iter(result.results.values()))
            layers = first_result.layers
            n_positions = first_result.n_positions

            if position_labels is None:
                pos_labels = [f"p{i}" for i in range(n_positions)]
            else:
                pos_labels = position_labels[:n_positions]

            plot_multi_metric_heatmap(
                matrices,
                layers,
                pos_labels,
                section_markers=section_markers,
                save_path=output_dir / "att_patching_comparison.png",
            )

    print(f"[viz] Attribution patching plots saved to {output_dir}")
