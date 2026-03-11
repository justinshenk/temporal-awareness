"""Visualization for coarse activation patching results.

Creates line plots for layer and position sweeps:
- White background with dual y-axes for different metric scales
- X-axis tick labels colored by token type (for position sweeps)
- Separate PNG files for denoising vs noising perspectives

This module delegates to submodules in the `coarse/` package:
- `coarse.sweep_plots`: Layer and position sweep visualizations
- `coarse.comparison`: Denoising vs noising scatter plots
- `coarse.sanity`: Sanity check diagnostic plots
- `coarse.aggregated`: Aggregated results across samples
"""

from __future__ import annotations

from pathlib import Path

from ...activation_patching.coarse import (
    CoarseActPatchAggregatedResults,
    CoarseActPatchResults,
)
from ...common.contrastive_pair import ContrastivePair
from ...viz.token_coloring import PairTokenColoring
from .coarse.aggregated import plot_aggregated
from .coarse.comparison import plot_comparison
from .coarse.sanity import plot_sanity_check
from .coarse.sweep_plots import plot_layer_sweep, plot_position_sweep


def visualize_coarse_patching(
    result: CoarseActPatchResults | CoarseActPatchAggregatedResults | None,
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
    pair: ContrastivePair | None = None,
) -> None:
    """Visualize coarse activation patching results.

    Creates (per step size):
    - coarse_layer_sweep_short_{step}.png: Layer sweep with short=clean
    - coarse_layer_sweep_long_{step}.png: Layer sweep with long=clean
    - coarse_position_sweep_short_{step}.png: Position sweep with short=clean
    - coarse_position_sweep_long_{step}.png: Position sweep with long=clean
    - denoising_vs_noising_{step}.png: Comparison scatter plots
    - sanity_check.png: Diagnostic metrics

    Args:
        result: CoarseActPatchResults (single pair) or
                CoarseActPatchAggregatedResults (multiple pairs)
        output_dir: Directory to save plots
        coloring: Token coloring for position tick colors
        pair: ContrastivePair for sanity check visualization
    """
    if result is None:
        print("[viz] No coarse patching results to visualize")
        return

    # Handle aggregated results
    if isinstance(result, CoarseActPatchAggregatedResults):
        plot_aggregated(result, output_dir, coloring)
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer sweep visualizations (both perspectives)
    for step_size in result.layer_step_sizes:
        layer_data = result.get_layer_results_for_step(step_size)
        if layer_data:
            plot_layer_sweep(layer_data, output_dir, step_size, "short")
            plot_layer_sweep(layer_data, output_dir, step_size, "long")

    # Position sweep visualizations (both perspectives)
    for step_size in result.position_step_sizes:
        pos_data = result.get_position_results_for_step(step_size)
        if pos_data:
            plot_position_sweep(pos_data, output_dir, step_size, "short", coloring)
            plot_position_sweep(pos_data, output_dir, step_size, "long", coloring)

    # Denoising vs Noising comparison plots
    for step_size in result.layer_step_sizes:
        layer_data = result.get_layer_results_for_step(step_size)
        pos_data = result.get_position_results_for_step(step_size)
        if layer_data or pos_data:
            plot_comparison(layer_data, pos_data, output_dir, coloring, step_size)

    # Sanity check visualization
    if result.sanity_result:
        plot_sanity_check(result, output_dir, coloring, pair)

    print(f"[viz] Coarse patching plots saved to {output_dir}")
