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
from ...common import profile
from ...common.contrastive_pair import ContrastivePair
from ...viz.token_coloring import PairTokenColoring
from .coarse.aggregated import plot_aggregated_structured, plot_all_aggregated_slices
from .coarse.comparison import plot_comparison
from .coarse.component_comparison import plot_all_component_comparisons
from .coarse.redundancy import plot_redundancy
from .coarse.sanity import plot_sanity_check
from .coarse.sweep_plots import plot_layer_sweep, plot_position_sweep


@profile
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
        plot_aggregated_structured(result, output_dir)
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    component = result.component

    # Layer sweep visualizations (both perspectives)
    for step_size in result.layer_step_sizes:
        layer_data = result.get_layer_results_for_step(step_size)
        if layer_data:
            plot_layer_sweep(layer_data, output_dir, step_size, "short", component)
            plot_layer_sweep(layer_data, output_dir, step_size, "long", component)

    # Position sweep visualizations (both perspectives)
    for step_size in result.position_step_sizes:
        pos_data = result.get_position_results_for_step(step_size)
        if pos_data:
            plot_position_sweep(pos_data, output_dir, step_size, "short", coloring, component)
            plot_position_sweep(pos_data, output_dir, step_size, "long", coloring, component)

    # Denoising vs Noising comparison and redundancy plots (for all step sizes)
    all_step_sizes = set(result.layer_step_sizes) | set(result.position_step_sizes)
    for step_size in sorted(all_step_sizes):
        layer_data = result.get_layer_results_for_step(step_size)
        pos_data = result.get_position_results_for_step(step_size)
        if layer_data or pos_data:
            plot_comparison(layer_data, pos_data, output_dir, coloring, step_size, component)
            plot_redundancy(layer_data, pos_data, output_dir, step_size, coloring, component)

    # Sanity check visualization
    if result.sanity_result:
        plot_sanity_check(result, output_dir, coloring, pair, component)

    print(f"[viz] Coarse patching plots saved to {output_dir}")


@profile
def visualize_component_comparison(
    results_by_component: dict[str, CoarseActPatchResults],
    output_dir: Path,
    step_size: int = 1,
) -> None:
    """Visualize multi-component comparison plots.

    Creates comparison plots across all available components:
    - Component attribution heatmaps (layers and positions)
    - Marginal contribution plots
    - Attention vs MLP scatter plots
    - Cumulative recovery plots
    - Redundancy gap analysis
    - And more...

    Args:
        results_by_component: Dict mapping component name to its results
        output_dir: Directory to save plots
        step_size: Step size to use for data extraction
    """
    if not results_by_component:
        print("[viz] No component results for comparison plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_all_component_comparisons(results_by_component, output_dir, step_size)


@profile
def visualize_all_aggregated(
    agg_by_component: dict[str, CoarseActPatchAggregatedResults],
    output_dir: Path,
) -> None:
    """Visualize all aggregated results with new folder structure.

    Creates:
        output_dir/
          all/
            sweep_resid_post/layer_sweep/denoising/...
            sweep_attn_out/...
            component_comparison/
          same_labels/
            sweep_resid_post/...
            component_comparison/
          ... (other analysis slices)

    Args:
        agg_by_component: Dict mapping component name to aggregated results
        output_dir: Base output directory (typically agg/)
    """
    if not agg_by_component:
        print("[viz] No aggregated results to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_all_aggregated_slices(agg_by_component, output_dir)
