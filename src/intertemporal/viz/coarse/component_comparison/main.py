"""Main entry point for component comparison visualizations."""

from __future__ import annotations

from pathlib import Path

from .....activation_patching.coarse import CoarseActPatchResults
from .constants import (
    COMPONENTS,
    SUBDIR_DECOMP,
    SUBDIR_OVERVIEW,
    SUBDIR_REDUNDANCY,
    SUBDIR_SANITY,
    SUBDIR_SYNTHESIS,
)
from .decomposition import plot_decomposition
from .overview import plot_overview
from .redundancy import plot_redundancy
from .sanity import plot_sanity_checks
from .synthesis import plot_synthesis


def plot_all_component_comparisons(
    results_by_component: dict[str, CoarseActPatchResults],
    output_dir: Path,
    step_size: int = 1,
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
        step_size: Step size to use for extracting data
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

    # Extract layer and position data for each component
    layer_data = {}
    pos_data = {}
    for comp in COMPONENTS:
        if comp in results_by_component:
            result = results_by_component[comp]
            layer_data[comp] = result.get_layer_results_for_step(step_size)
            pos_data[comp] = result.get_position_results_for_step(step_size)

    if not layer_data:
        print("[viz] No component data available for comparison plots")
        return

    # Generate plots by category
    plot_sanity_checks(layer_data, sanity_dir)
    plot_overview(layer_data, pos_data, results_by_component, overview_dir)
    plot_decomposition(layer_data, pos_data, decomp_dir)
    plot_redundancy(layer_data, pos_data, redundancy_dir)
    plot_synthesis(layer_data, pos_data, synthesis_dir)

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
