"""Visualization functions for fine-grained patching analysis.

Implements plots 17-26:
17. Head-level patching heatmap
18. Head-level ranked bar chart with cumulative line
19. Head scatter (denoising vs noising)
20. Head x position heatmap
21. Path patching head-to-MLP matrix
22. Path patching head-to-head matrix
23. Multi-site interaction heatmap
24. Neuron-level ranked bar chart
25. Cumulative neuron contribution curve
26. Layer-position fine heatmap
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from ....common.profiler import profile
from ....viz.plot_helpers import finalize_plot

from .fine_grained_results import (
    FineGrainedResults,
    HeadSweepResults,
    PositionPatchingResult,
    PathPatchingResult,
    MultiSiteResult,
    NeuronPatchingResult,
    LayerPositionResult,
    AttentionPatchingCorrelation,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....common.position_mapping import SamplePositionMapping


def _get_position_label(pos: int, mapping: "SamplePositionMapping | None") -> str:
    """Get semantic label for a position if available, else fall back to P{pos}."""
    if mapping:
        pos_info = mapping.get_position(pos)
        if pos_info and pos_info.format_pos:
            return pos_info.format_pos
    return f"P{pos}"


@profile
def visualize_fine_grained(
    results: FineGrainedResults | None,
    output_dir: Path,
    position_mapping: "SamplePositionMapping | None" = None,
) -> None:
    """Generate all fine-grained patching visualizations.

    Creates plots 17-26 as specified.

    Args:
        results: FineGrainedResults from run_fine_grained_analysis
        output_dir: Directory to save plots
        position_mapping: Optional mapping for semantic position labels
    """
    if results is None:
        print("[viz] No fine-grained results to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 17: Head-level heatmaps
    if results.head_sweep:
        _plot_head_heatmaps(results.head_sweep, output_dir)

    # Plot 18: Head-level ranked bar chart
    if results.head_sweep:
        _plot_head_ranked_bar(results.head_sweep, output_dir)

    # Plot 19: Head scatter
    if results.head_sweep:
        _plot_head_scatter(results.head_sweep, output_dir)

    # Plot 20: Head x position heatmap
    if results.position_results:
        _plot_head_position_heatmap(results.position_results, output_dir, position_mapping)

    # Plot 21: Path patching head-to-MLP
    if results.path_to_mlp:
        _plot_path_head_to_mlp(results.path_to_mlp, output_dir)

    # Plot 22: Path patching head-to-head
    if results.path_to_head:
        _plot_path_head_to_head(results.path_to_head, output_dir)

    # Plot 23: Multi-site interaction heatmap
    if results.multi_site:
        _plot_multi_site_interaction(results.multi_site, output_dir)

    # Plot 24: Neuron-level ranked bar chart
    if results.neuron_results:
        _plot_neuron_ranked_bar(results.neuron_results, results.neuron_target_layer, output_dir)

    # Plot 25: Cumulative neuron contribution curve
    if results.neuron_results:
        _plot_cumulative_neuron_curve(results, output_dir)

    # Plot 26: Layer-position fine heatmap
    if results.layer_position:
        _plot_layer_position_heatmaps(results.layer_position, output_dir, position_mapping)

    # Plot 27: Head redundancy gap chart
    if results.head_sweep:
        _plot_head_redundancy_gap(results.head_sweep, output_dir)

    # Plot 28: Attention-patching cross-reference table
    if results.attention_correlations:
        _plot_attention_patching_crossref(results.attention_correlations, output_dir)

    # Plot 29: Cross-layer path patching (L19/L21 → L24)
    if results.cross_layer_paths:
        _plot_cross_layer_paths(results.cross_layer_paths, output_dir)

    print(f"[viz] Fine-grained patching plots saved to {output_dir}")


def _plot_head_heatmaps(head_sweep: HeadSweepResults, output_dir: Path) -> None:
    """Plot 17: Head-level patching heatmap.

    Rows: layer index, Columns: head index
    Color: recovery/disruption score
    Two versions: denoising and noising
    """
    if head_sweep.denoising_matrix is None:
        head_sweep.build_matrices()

    if head_sweep.denoising_matrix is None:
        return

    layers = head_sweep.layers_analyzed

    # Denoising heatmap
    fig, ax = plt.subplots(figsize=(12, max(4, len(layers) * 0.3)))
    im = ax.imshow(
        head_sweep.denoising_matrix,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-1, vmax=1,
    )
    plt.colorbar(im, ax=ax, label="Denoising Recovery")

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_xlabel("Head Index", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("Head-Level Patching: Denoising Recovery", fontsize=12, fontweight="bold")

    plt.tight_layout()
    finalize_plot(output_dir / "17_head_heatmap_denoising.png")

    # Noising heatmap
    fig, ax = plt.subplots(figsize=(12, max(4, len(layers) * 0.3)))
    im = ax.imshow(
        head_sweep.noising_matrix,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-1, vmax=1,
    )
    plt.colorbar(im, ax=ax, label="Noising Disruption")

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_xlabel("Head Index", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("Head-Level Patching: Noising Disruption", fontsize=12, fontweight="bold")

    plt.tight_layout()
    finalize_plot(output_dir / "17_head_heatmap_noising.png")


def _plot_head_ranked_bar(head_sweep: HeadSweepResults, output_dir: Path) -> None:
    """Plot 18: Head-level ranked bar chart with cumulative line.

    X-axis: head labels (e.g., "L24.H7")
    Y-axis: denoising recovery
    Top 20 heads, paired bars for denoising/noising
    Cumulative percentage line on secondary axis
    """
    top_heads = head_sweep.get_top_heads(20, by="combined")
    if not top_heads:
        return

    labels = [h.label for h in top_heads]
    denoising = [h.denoising_recovery for h in top_heads]
    noising = [h.noising_disruption for h in top_heads]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Paired bars
    bars1 = ax1.bar(x - width/2, denoising, width, label="Denoising Recovery", color="steelblue", alpha=0.8)
    bars2 = ax1.bar(x + width/2, noising, width, label="Noising Disruption", color="indianred", alpha=0.8)

    ax1.set_xlabel("Head", fontsize=11)
    ax1.set_ylabel("Effect Score", fontsize=11)
    ax1.set_title("Top 20 Attention Heads: Denoising vs Noising", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.legend(loc="upper left")
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # Cumulative percentage on secondary axis
    ax2 = ax1.twinx()
    total = sum(abs(h.combined_score) for h in head_sweep.results)
    if total > 0:
        cumsum = np.cumsum([abs(h.combined_score) for h in top_heads])
        cumulative_pct = 100 * cumsum / total
        ax2.plot(x, cumulative_pct, "o-", color="darkgreen", linewidth=2, markersize=4, label="Cumulative %")
        ax2.set_ylabel("Cumulative % of Total Effect", fontsize=11, color="darkgreen")
        ax2.tick_params(axis="y", labelcolor="darkgreen")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right")

    plt.tight_layout()
    finalize_plot(output_dir / "18_head_ranked_bar.png")


def _plot_head_scatter(head_sweep: HeadSweepResults, output_dir: Path) -> None:
    """Plot 19: Head scatter (denoising vs noising).

    X-axis: denoising recovery
    Y-axis: noising disruption
    Each point = one head, labeled
    Diagonal for agreement
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    denoising = [h.denoising_recovery for h in head_sweep.results]
    noising = [h.noising_disruption for h in head_sweep.results]

    ax.scatter(denoising, noising, alpha=0.6, s=30, c="steelblue")

    # Label top heads
    top_heads = head_sweep.get_top_heads(10, by="combined")
    for h in top_heads:
        ax.annotate(
            h.label,
            (h.denoising_recovery, h.noising_disruption),
            fontsize=8,
            alpha=0.8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    # Diagonal line
    lims = [
        min(min(denoising), min(noising)) - 0.1,
        max(max(denoising), max(noising)) + 0.1,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="y=x (agreement)")

    ax.set_xlabel("Denoising Recovery", fontsize=11)
    ax.set_ylabel("Noising Disruption", fontsize=11)
    ax.set_title("Head-Level: Denoising vs Noising", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    finalize_plot(output_dir / "19_head_scatter.png")


def _plot_head_position_heatmap(
    position_results: list[PositionPatchingResult],
    output_dir: Path,
    position_mapping: "SamplePositionMapping | None" = None,
) -> None:
    """Plot 20: Head x position patching heatmap.

    For top heads: patch at each position independently
    Rows: position, Columns: heads
    Color: patching effect
    """
    if not position_results:
        return

    n_heads = len(position_results)
    positions = position_results[0].positions
    n_pos = len(positions)

    # Build matrix: [n_positions, n_heads]
    denoising_matrix = np.zeros((n_pos, n_heads))
    for hi, pr in enumerate(position_results):
        for pi, val in enumerate(pr.denoising_by_position):
            denoising_matrix[pi, hi] = val

    fig, ax = plt.subplots(figsize=(max(6, n_heads * 1.2), max(8, n_pos * 0.15)))
    im = ax.imshow(
        denoising_matrix,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Denoising Recovery")

    head_labels = [pr.label for pr in position_results]
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels(head_labels, rotation=45, ha="right", fontsize=9)

    # Show subset of position labels
    pos_step = max(1, n_pos // 20)
    ax.set_yticks(range(0, n_pos, pos_step))
    ax.set_yticklabels([_get_position_label(positions[i], position_mapping) for i in range(0, n_pos, pos_step)], fontsize=8)

    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Position", fontsize=11)
    ax.set_title("Head x Position Patching Effect (Denoising)", fontsize=12, fontweight="bold")

    # Add explanatory note about scale
    # Per-position effects are ~100x smaller than head-level effects because
    # this shows the per-position decomposition (effects sum across positions)
    note_text = (
        "Note: Per-position effects are smaller than head-level effects\n"
        "(~0.002 vs ~0.8) because each position contributes a fraction\n"
        "of the total head effect. Effects sum across all positions."
    )
    ax.text(
        0.02, 0.02, note_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    finalize_plot(output_dir / "20_head_position_heatmap.png")

    # Build noising matrix and create second heatmap
    noising_matrix = np.zeros((n_pos, n_heads))
    for hi, pr in enumerate(position_results):
        for pi, val in enumerate(pr.noising_by_position):
            noising_matrix[pi, hi] = val

    fig, ax = plt.subplots(figsize=(max(6, n_heads * 1.2), max(8, n_pos * 0.15)))
    im = ax.imshow(
        noising_matrix,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Noising Disruption")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels(head_labels, rotation=45, ha="right", fontsize=9)

    ax.set_yticks(range(0, n_pos, pos_step))
    ax.set_yticklabels([_get_position_label(positions[i], position_mapping) for i in range(0, n_pos, pos_step)], fontsize=8)

    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Position", fontsize=11)
    ax.set_title("Head x Position Patching Effect (Noising)", fontsize=12, fontweight="bold")

    # Add explanatory note about scale
    ax.text(
        0.02, 0.02, note_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    finalize_plot(output_dir / "20_head_position_heatmap_noising.png")


def _plot_path_head_to_mlp(
    path_results: list[PathPatchingResult],
    output_dir: Path,
) -> None:
    """Plot 21: Path patching head-to-MLP connectivity matrix.

    Rows: source heads (top 5)
    Columns: destination MLPs
    Color: path patching effect
    """
    if not path_results:
        return

    # Get unique source heads and dest MLPs
    source_heads = sorted(set((p.source_layer, p.source_head) for p in path_results))
    dest_mlps = sorted(set(p.dest_layer for p in path_results))

    if not source_heads or not dest_mlps:
        return

    # Build matrix
    matrix = np.zeros((len(source_heads), len(dest_mlps)))
    src_to_idx = {src: i for i, src in enumerate(source_heads)}
    dest_to_idx = {d: i for i, d in enumerate(dest_mlps)}

    for p in path_results:
        si = src_to_idx.get((p.source_layer, p.source_head))
        di = dest_to_idx.get(p.dest_layer)
        if si is not None and di is not None:
            matrix[si, di] = p.effect

    fig, ax = plt.subplots(figsize=(max(6, len(dest_mlps) * 1.5), max(4, len(source_heads) * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Path Effect")

    src_labels = [f"L{l}.H{h}" for l, h in source_heads]
    dest_labels = [f"L{l}.MLP" for l in dest_mlps]

    ax.set_yticks(range(len(source_heads)))
    ax.set_yticklabels(src_labels, fontsize=9)
    ax.set_xticks(range(len(dest_mlps)))
    ax.set_xticklabels(dest_labels, rotation=45, ha="right", fontsize=9)

    ax.set_xlabel("Destination MLP", fontsize=11)
    ax.set_ylabel("Source Head", fontsize=11)
    ax.set_title("Path Patching: Head to MLP Connectivity", fontsize=12, fontweight="bold")

    # Add value annotations
    for i in range(len(source_heads)):
        for j in range(len(dest_mlps)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    finalize_plot(output_dir / "21_path_head_to_mlp.png")


def _plot_path_head_to_head(
    path_results: list[PathPatchingResult],
    output_dir: Path,
) -> None:
    """Plot 22: Path patching head-to-head connectivity matrix.

    Rows: source heads at earlier layers
    Columns: destination heads at later layers
    Color: path patching effect
    """
    if not path_results:
        return

    # Get unique sources and destinations
    source_heads = sorted(set((p.source_layer, p.source_head) for p in path_results))
    dest_layers = sorted(set(p.dest_layer for p in path_results))

    if not source_heads or not dest_layers:
        return

    # Build matrix
    matrix = np.zeros((len(source_heads), len(dest_layers)))
    src_to_idx = {src: i for i, src in enumerate(source_heads)}
    dest_to_idx = {d: i for i, d in enumerate(dest_layers)}

    for p in path_results:
        si = src_to_idx.get((p.source_layer, p.source_head))
        di = dest_to_idx.get(p.dest_layer)
        if si is not None and di is not None:
            matrix[si, di] = p.effect

    fig, ax = plt.subplots(figsize=(max(6, len(dest_layers) * 1.5), max(4, len(source_heads) * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Path Effect")

    src_labels = [f"L{l}.H{h}" for l, h in source_heads]
    dest_labels = [f"L{l}.Attn" for l in dest_layers]

    ax.set_yticks(range(len(source_heads)))
    ax.set_yticklabels(src_labels, fontsize=9)
    ax.set_xticks(range(len(dest_layers)))
    ax.set_xticklabels(dest_labels, rotation=45, ha="right", fontsize=9)

    ax.set_xlabel("Destination Layer Attention", fontsize=11)
    ax.set_ylabel("Source Head", fontsize=11)
    ax.set_title("Path Patching: Head to Head Connectivity", fontsize=12, fontweight="bold")

    # Add value annotations
    for i in range(len(source_heads)):
        for j in range(len(dest_layers)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    finalize_plot(output_dir / "22_path_head_to_head.png")


def _plot_multi_site_interaction(
    multi_site: list[MultiSiteResult],
    output_dir: Path,
) -> None:
    """Plot 23: Multi-site patching interaction heatmap.

    Rows and columns: top 10 components
    Color: interaction effect = joint - individual_A - individual_B
    """
    if not multi_site:
        return

    # Get unique components
    all_comps = set()
    for ms in multi_site:
        all_comps.add(ms.component_a)
        all_comps.add(ms.component_b)
    components = sorted(all_comps)

    n = len(components)
    comp_to_idx = {c: i for i, c in enumerate(components)}

    # Build matrix (symmetric)
    matrix = np.zeros((n, n))
    for ms in multi_site:
        i = comp_to_idx[ms.component_a]
        j = comp_to_idx[ms.component_b]
        matrix[i, j] = ms.interaction
        matrix[j, i] = ms.interaction

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(8, n * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Interaction Effect")

    ax.set_xticks(range(n))
    ax.set_xticklabels(components, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(components, fontsize=9)

    ax.set_xlabel("Component B", fontsize=11)
    ax.set_ylabel("Component A", fontsize=11)
    ax.set_title("Multi-Site Interaction: Joint - Individual A - Individual B", fontsize=12, fontweight="bold")

    # Add value annotations
    for i in range(n):
        for j in range(n):
            if i != j:
                val = matrix[i, j]
                color = "white" if abs(val) > 0.15 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    finalize_plot(output_dir / "23_multi_site_interaction.png")


def _plot_neuron_ranked_bar(
    neuron_results: list[NeuronPatchingResult],
    target_layer: int,
    output_dir: Path,
) -> None:
    """Plot 24: Neuron-level patching at target layer - ranked.

    X-axis: neuron index (sorted by effect)
    Y-axis: drop in correct logit probability
    """
    if not neuron_results:
        return

    # Sort by absolute effect
    sorted_neurons = sorted(neuron_results, key=lambda x: abs(x.effect), reverse=True)[:50]

    labels = [f"N{n.neuron_idx}" for n in sorted_neurons]
    effects = [n.effect for n in sorted_neurons]
    colors = ["steelblue" if e > 0 else "indianred" for e in effects]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(labels))
    ax.bar(x, effects, color=colors, alpha=0.8)

    ax.set_xlabel("Neuron (sorted by effect)", fontsize=11)
    ax.set_ylabel("Differential Contribution to Logit", fontsize=11)
    ax.set_title(f"Top Neurons at L{target_layer}: Differential Contribution", fontsize=12, fontweight="bold")

    # Add note explaining this is differential contribution, not ablation
    ax.text(
        0.02, 0.98,
        "Effect = (clean_act − corrupt_act) × W_out_alignment\n"
        "This measures what changes between conditions,\n"
        "not what happens when neurons are ablated.",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_xticks(x[::5])
    ax.set_xticklabels(labels[::5], rotation=45, ha="right", fontsize=8)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Add legend explaining color meanings
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.8, label="Positive: promotes clean answer"),
        Patch(facecolor="indianred", alpha=0.8, label="Negative: promotes corrupted answer"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    finalize_plot(output_dir / "24_neuron_ranked_bar.png")


def _plot_cumulative_neuron_curve(
    results: FineGrainedResults,
    output_dir: Path,
) -> None:
    """Plot 25: Cumulative neuron contribution curve.

    X-axis: number of top neurons (1-50)
    Y-axis: fraction of layer's total patching effect recovered
    """
    if not results.neuron_results:
        return

    # Compute cumulative fractions
    ns = list(range(1, min(51, len(results.neuron_results) + 1)))
    fractions = [results.get_cumulative_neuron_effect(n) for n in ns]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ns, fractions, "o-", color="steelblue", linewidth=2, markersize=4)
    ax.fill_between(ns, fractions, alpha=0.2, color="steelblue")

    ax.set_xlabel("Number of Top Neurons", fontsize=11)
    ax.set_ylabel("Fraction of Total Effect", fontsize=11)
    ax.set_title(f"Cumulative Neuron Contribution at L{results.neuron_target_layer}", fontsize=12, fontweight="bold")

    ax.set_xlim(1, max(ns))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Add reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Add text annotations for 50%, 80%, 90% thresholds
    cumulative = np.array(fractions)
    for threshold, label in [(0.5, "50%"), (0.8, "80%"), (0.9, "90%")]:
        idx = int(np.searchsorted(cumulative, threshold))
        if idx < len(ns):
            ax.annotate(
                f"{label}: {ns[idx]} neurons",
                xy=(ns[idx], cumulative[idx]),
                xytext=(ns[idx] + 3, cumulative[idx] - 0.08),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.5, color="gray"),
            )

    plt.tight_layout()
    finalize_plot(output_dir / "25_cumulative_neuron_curve.png")


def _plot_layer_position_heatmaps(
    layer_position: dict[str, LayerPositionResult],
    output_dir: Path,
    position_mapping: "SamplePositionMapping | None" = None,
) -> None:
    """Plot 26: Layer-position fine patching heatmap.

    For attn_out and mlp_out separately:
    Rows: layer, Columns: position
    Color: patching effect
    True 2D localization (not outer product approximation)
    """
    for component, lp in layer_position.items():
        if lp.denoising_grid is None:
            continue

        layers = lp.layers
        positions = lp.positions
        n_layers = len(layers)
        n_pos = len(positions)

        # Denoising heatmap
        fig, ax = plt.subplots(figsize=(max(10, n_pos * 0.15), max(6, n_layers * 0.3)))
        im = ax.imshow(
            lp.denoising_grid,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Denoising Recovery")

        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)

        pos_step = max(1, n_pos // 15)
        ax.set_xticks(range(0, n_pos, pos_step))
        ax.set_xticklabels([_get_position_label(positions[i], position_mapping) for i in range(0, n_pos, pos_step)], fontsize=8, rotation=45, ha="right")

        ax.set_xlabel("Position", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_title(f"Layer x Position Fine Patching: {component} (Denoising)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        finalize_plot(output_dir / f"26_layer_position_{component}_denoising.png")

        # Noising heatmap
        if lp.noising_grid is not None:
            fig, ax = plt.subplots(figsize=(max(10, n_pos * 0.15), max(6, n_layers * 0.3)))
            im = ax.imshow(
                lp.noising_grid,
                aspect="auto",
                cmap="RdBu_r",
                interpolation="nearest",
            )
            plt.colorbar(im, ax=ax, label="Noising Disruption")

            ax.set_yticks(range(n_layers))
            ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)

            ax.set_xticks(range(0, n_pos, pos_step))
            ax.set_xticklabels([_get_position_label(positions[i], position_mapping) for i in range(0, n_pos, pos_step)], fontsize=8, rotation=45, ha="right")

            ax.set_xlabel("Position", fontsize=11)
            ax.set_ylabel("Layer", fontsize=11)
            ax.set_title(f"Layer x Position Fine Patching: {component} (Noising)", fontsize=12, fontweight="bold")

            plt.tight_layout()
            finalize_plot(output_dir / f"26_layer_position_{component}_noising.png")


def _plot_head_redundancy_gap(head_sweep: HeadSweepResults, output_dir: Path) -> None:
    """Plot 27: Head redundancy gap chart.

    Shows denoising - noising gap per head, sorted by gap magnitude.
    Small gap = bottleneck head (unique contribution)
    Large gap = redundant head (effect can be compensated by other paths)
    """
    top_heads = head_sweep.get_top_heads(20, by="combined")
    if not top_heads:
        return

    # Compute gaps: denoising - noising
    # Positive gap means denoising > noising (more recovery than disruption)
    gaps = [
        (h.label, h.denoising_recovery - h.noising_disruption)
        for h in top_heads
    ]
    # Sort by absolute gap magnitude (largest gaps first)
    gaps.sort(key=lambda x: abs(x[1]), reverse=True)

    labels = [g[0] for g in gaps]
    gap_values = [g[1] for g in gaps]
    colors = ["steelblue" if v >= 0 else "indianred" for v in gap_values]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(labels))
    bars = ax.bar(x, gap_values, color=colors, alpha=0.8)

    ax.set_xlabel("Head (sorted by gap magnitude)", fontsize=11)
    ax.set_ylabel("Denoising - Noising Gap", fontsize=11)
    ax.set_title("Head Redundancy Analysis: Denoising vs Noising Gap", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Add legend explaining interpretation
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.8, label="Positive: denoising > noising (unique info)"),
        Patch(facecolor="indianred", alpha=0.8, label="Negative: noising > denoising (redundant)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Annotate interpretation
    ax.text(
        0.02, 0.98,
        "Small gap = bottleneck (critical)\nLarge gap = redundant (compensatable)",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    finalize_plot(output_dir / "27_head_redundancy_gap.png")


def _plot_attention_patching_crossref(
    correlations: list,  # list[AttentionPatchingCorrelation]
    output_dir: Path,
) -> None:
    """Plot 28: Cross-reference table of patching scores vs attention patterns.

    Shows for each top head:
    - Patching score
    - Redundancy gap
    - Top-5 attended positions
    - Attention to source positions
    """
    if not correlations:
        return

    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, max(4, len(correlations) * 0.5 + 1)))
    ax.axis("off")

    # Build table data
    col_labels = [
        "Head",
        "Patching\nScore",
        "Denoising",
        "Noising",
        "Gap",
        "Attn→Src",
        "Top-5 Attended Positions",
    ]

    table_data = []
    for c in correlations:
        # Format top attended positions
        if c.top_attended_positions:
            top_pos_str = ", ".join(f"P{p}" for p in c.top_attended_positions[:5])
        else:
            top_pos_str = "N/A"

        row = [
            c.head_label,
            f"{c.patching_score:.3f}",
            f"{c.denoising_recovery:.3f}",
            f"{c.noising_disruption:.3f}",
            f"{c.redundancy_gap:+.3f}",
            f"{c.attn_to_source:.3f}",
            top_pos_str,
        ]
        table_data.append(row)

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color cells based on values
    for i, c in enumerate(correlations, start=1):
        # Gap column (index 4): green for positive (unique), red for negative (redundant)
        gap_cell = table[(i, 4)]
        if c.redundancy_gap > 0.1:
            gap_cell.set_facecolor("#C6EFCE")  # Light green
        elif c.redundancy_gap < -0.05:
            gap_cell.set_facecolor("#FFC7CE")  # Light red

        # Attn to source column (index 5): highlight high attention
        attn_cell = table[(i, 5)]
        if c.attn_to_source > 0.1:
            attn_cell.set_facecolor("#FFEB9C")  # Light yellow

    ax.set_title(
        "Attention-Patching Cross-Reference: Top Heads",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    finalize_plot(output_dir / "28_attention_patching_crossref.png")


def _plot_cross_layer_paths(
    cross_layer_results: list,  # list[PathPatchingResult]
    output_dir: Path,
) -> None:
    """Plot 29: Cross-layer path patching (multi-hop circuit).

    Shows information flow between layers:
    - L19 → L21 (early circuit)
    - L19/L21 → L24 (mid circuit)
    - L24 → L28-31 (late circuit)
    """
    if not cross_layer_results:
        return

    # Group paths by hop type
    l19_to_l21 = [r for r in cross_layer_results if r.source_layer == 19 and r.dest_layer == 21]
    early_to_l24 = [r for r in cross_layer_results if r.source_layer in (19, 21) and r.dest_layer == 24]
    l24_to_later = [r for r in cross_layer_results if r.source_layer == 24 and r.dest_layer > 24]

    # Count how many subplots we need
    groups = []
    if l19_to_l21:
        groups.append(("L19 → L21", l19_to_l21))
    if early_to_l24:
        groups.append(("L19/L21 → L24", early_to_l24))
    if l24_to_later:
        groups.append(("L24 → L28-31", l24_to_later))

    if not groups:
        return

    # Create subplots
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 6))
    if n_groups == 1:
        axes = [axes]

    for ax, (title, paths) in zip(axes, groups):
        source_heads = sorted(set((r.source_layer, r.source_head) for r in paths))
        dest_heads = sorted(set((r.dest_layer, r.dest_head) for r in paths))

        n_src = len(source_heads)
        n_dst = len(dest_heads)

        if n_src == 0 or n_dst == 0:
            ax.set_visible(False)
            continue

        matrix = np.zeros((n_src, n_dst))
        src_to_idx = {s: i for i, s in enumerate(source_heads)}
        dst_to_idx = {d: i for i, d in enumerate(dest_heads)}

        for r in paths:
            si = src_to_idx.get((r.source_layer, r.source_head))
            di = dst_to_idx.get((r.dest_layer, r.dest_head))
            if si is not None and di is not None:
                matrix[si, di] = r.effect

        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Path Effect")

        # Labels
        src_labels = [f"L{l}.H{h}" for l, h in source_heads]
        dst_labels = [f"L{l}.H{h}" for l, h in dest_heads]

        ax.set_yticks(range(n_src))
        ax.set_yticklabels(src_labels, fontsize=9)
        ax.set_xticks(range(n_dst))
        ax.set_xticklabels(dst_labels, rotation=45, ha="right", fontsize=9)

        ax.set_xlabel("Destination Head", fontsize=10)
        ax.set_ylabel("Source Head", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Add value annotations
        for i in range(n_src):
            for j in range(n_dst):
                val = matrix[i, j]
                if abs(val) > 0.01:
                    color = "white" if abs(val) > 0.15 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.suptitle("Cross-Layer Path Patching: Multi-Hop Circuit", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    finalize_plot(output_dir / "29_cross_layer_paths.png")
