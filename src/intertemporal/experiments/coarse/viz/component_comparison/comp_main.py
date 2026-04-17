"""Main entry point for component comparison visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ......activation_patching.coarse import CoarseActPatchResults
from ...coarse_results import ComponentComparisonResults
from .comp_constants import (
    COMPONENTS,
    SUBDIR_DECOMP,
    SUBDIR_OVERVIEW,
    SUBDIR_REDUNDANCY,
    SUBDIR_SANITY,
    SUBDIR_SYNTHESIS,
)
from .comp_decomposition import plot_decomposition
from .comp_overview import plot_overview
from .comp_redundancy import plot_redundancy
from .comp_sanity import plot_sanity_checks
from .comp_synthesis import plot_synthesis

if TYPE_CHECKING:
    from ......common.position_mapping import SamplePositionMapping


def plot_all_component_comparisons(
    results_by_component: dict[str, CoarseActPatchResults],
    output_dir: Path,
    step_size: int = 1,
    processed_results: ComponentComparisonResults | None = None,
    position_mapping: "SamplePositionMapping | None" = None,
    agg_by_component: dict | None = None,
) -> None:
    """Generate all multi-component comparison plots organized by category.

    Directory structure:
        01_sanity_checks/     - Validation plots (check these first)
        02_overview/          - Big picture heatmaps
        03_component_decomp/  - Attention vs MLP analysis
        04_redundancy/        - Noising vs denoising comparison
        05_circuit_synthesis/ - Information flow summary

    Args:
        results_by_component: Dict mapping component name to its results
        output_dir: Directory to save plots
        step_size: Step size to use for extracting layer data (position step size is auto-detected)
        processed_results: Pre-computed analysis results (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    sanity_dir = output_dir / SUBDIR_SANITY
    overview_dir = output_dir / SUBDIR_OVERVIEW
    decomp_dir = output_dir / SUBDIR_DECOMP
    redundancy_dir = output_dir / SUBDIR_REDUNDANCY
    synthesis_dir = output_dir / SUBDIR_SYNTHESIS

    for d in [sanity_dir, overview_dir, decomp_dir, redundancy_dir, synthesis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Extract POPULATION-MEAN layer and position data for each component.
    # When agg_by_component is available (aggregated across pairs), use
    # get_mean_layer_results / get_mean_position_results which compute
    # the mean recovery/disruption across all pairs. Only fall back to
    # single-pair data (results_by_component) when no aggregation exists.
    layer_data = {}
    pos_data = {}
    pos_step_size = step_size
    for comp in COMPONENTS:
        agg = agg_by_component.get(comp) if agg_by_component else None
        single = results_by_component.get(comp)
        if agg and agg.n_samples > 0:
            layer_data[comp] = agg.get_mean_layer_results(step_size)
            comp_pos_step = agg.position_step_sizes[0] if agg.position_step_sizes else step_size
            pos_data[comp] = agg.get_mean_position_results(comp_pos_step)
            if agg.position_step_sizes:
                pos_step_size = comp_pos_step
        elif single:
            layer_data[comp] = single.get_layer_results_for_step(step_size)
            comp_pos_step = single.position_step_sizes[0] if single.position_step_sizes else step_size
            pos_data[comp] = single.get_position_results_for_step(comp_pos_step)
            if single.position_step_sizes:
                pos_step_size = comp_pos_step

    if not layer_data:
        print("[viz] No component data available for comparison plots")
        return

    # Generate plots by category
    plot_sanity_checks(layer_data, sanity_dir)
    plot_overview(layer_data, pos_data, results_by_component, overview_dir, step_size, pos_step_size, position_mapping, agg_by_component)
    plot_decomposition(layer_data, pos_data, decomp_dir, processed_results, position_mapping, agg_by_component)
    plot_redundancy(layer_data, pos_data, redundancy_dir, processed_results, position_mapping, agg_by_component)
    if processed_results is not None:
        plot_synthesis(synthesis_dir, processed_results)

    # Generate README
    _generate_readme(output_dir)

    print(f"[viz] Component comparison plots saved to {output_dir}")


def _generate_readme(output_dir: Path) -> None:
    """Generate README.md documenting all plots with figure numbers."""
    readme_content = """# Component Comparison Plots

This directory contains activation patching analysis visualizations organized
into five categories. Each plot has a figure number for easy reference.

## Quick Start

1. **Check 01_sanity_checks/** first - if these fail, results are unreliable
2. **Read 05_circuit_synthesis/circuit_summary.txt** for the plain-text explanation
3. **View 05_circuit_synthesis/information_flow_diagram.png** for the visual summary
4. Drill into other folders for detailed analysis

---

## Directory Structure

### 01_sanity_checks/
*Validation plots - check these first. If they fail, everything else is suspect.*

| Fig | File | Description |
|-----|------|-------------|
| 1.1 | resid_delta_sanity.png | Compares resid_pre[L+1] vs resid_post[L] (should match) |
| 1.2 | resid_delta_difference.png | Difference plot with +/-0.02 tolerance band |

### 02_overview/
*The big picture: "Where does the computation happen?"*

| Fig | File | Description |
|-----|------|-------------|
| 2.1 | heatmap_layers_denoising.png | Layer × Component attribution (denoising recovery) |
| 2.2 | heatmap_layers_noising.png | Layer × Component attribution (noising disruption) |
| 2.3 | heatmap_layers_colnorm_denoising.png | Column-normalized heatmap (reveals fine structure) |
| 2.4 | heatmap_layers_colnorm_noising.png | Column-normalized heatmap (noising) |
| 2.5 | heatmap_positions_denoising.png | Position × Component attribution (denoising) |
| 2.6 | heatmap_positions_noising.png | Position × Component attribution (noising) |
| 2.7 | layer_position_heatmap.png | Layer × Position importance map (2D localization) |

### 03_component_decomp/
*Drilling in: "Is it attention or MLP? Which specific components?"*

| Fig | File | Description |
|-----|------|-------------|
| 3.1 | attn_vs_mlp_layer.png | Attention vs MLP scatter by layer |
| 3.2 | attn_vs_mlp_position.png | Attention vs MLP scatter by position |
| 3.3 | attn_vs_mlp_paired.png | Paired scatter with arrows (denoising→noising movement) |
| 3.4 | component_importance_ranked.png | Top components ranked with same-layer linking |
| 3.5 | cumulative_recovery.png | Cumulative recovery with dip annotations |
| 3.6 | marginal_contribution.png | Per-layer marginal contribution |
| 3.7 | position_component_interaction.png | Effect vs position for each component |
| 3.8 | position_interaction_zoomed.png | Zoomed panels for hub regions |

### 04_redundancy/
*Methodology comparison: "Can I trust the scores? What has backup pathways?"*

| Fig | File | Description |
|-----|------|-------------|
| 4.1 | noise_vs_denoise_per_component_layer.png | Noising vs denoising scatter (layers) |
| 4.2 | noise_vs_denoise_per_component_position.png | Noising vs denoising scatter (positions) |
| 4.3 | redundancy_gap.png | Redundancy gap (disruption - recovery) per layer |
| 4.4 | redundancy_gap_sorted.png | Redundancy gap sorted by magnitude |
| 4.5 | difference_heatmap_layer.png | Redundancy gap heatmap (layers) |
| 4.6 | difference_heatmap_position.png | Redundancy gap heatmap (positions) |

### 05_circuit_synthesis/
*The punchline: synthesized information flow.*

| Fig | File | Description |
|-----|------|-------------|
| 5.1 | information_flow_diagram.png | Circuit summary with redundancy and bottleneck annotations |
| — | circuit_summary.txt | Plain-text explanation of the circuit hypothesis |

---

## Interpretation Guide

### Scatter Plot Regions
- **AND region** (upper-right): High in both → necessary AND sufficient
- **OR region** (lower-right): High denoising, low noising → has backup pathways

### Redundancy Gap
- **Positive gap** (disruption > recovery): Component is necessary
- **Negative gap** (recovery > disruption): Component is sufficient but replaceable

### Scores in Circuit Diagram
- Format: noising_disruption / denoising_recovery
- Higher noising = more necessary (corrupting it breaks the model)
- Higher denoising = more sufficient (restoring it fixes the model)

### Visual Conventions
- ★ = Multi-function layer (appears in both attention and MLP top lists)
- Colored brackets = Same-layer components (e.g., L24_attn and L24_mlp)
- Red thick arrow = Critical bottleneck path
- Dashed arrow = Backup/bypass pathway

---

## Colormap Reference

| Plot Type | Colormap | Interpretation |
|-----------|----------|----------------|
| Recovery/Disruption heatmaps | RdYlGn | Green=high, Red=low |
| 2D localization map | Hot | White=high, Black=low |
| Redundancy gap heatmaps | RdBu_r | Red=necessity, Blue=sufficiency |
| Layer/position scatter | Viridis | Yellow=late, Purple=early |
"""
    (output_dir / "README.md").write_text(readme_content)
    print(f"Saved: {output_dir / 'README.md'}")
