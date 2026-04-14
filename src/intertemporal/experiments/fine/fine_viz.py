"""Visualization functions for fine-grained patching analysis.

Implements plots for:
- Path patching head-to-MLP matrix
- Path patching head-to-head matrix
- Multi-site interaction heatmap
- Cross-layer path patching

NOTE: Head attribution and position patching visualizations are now in attn_viz.py
NOTE: Neuron attribution visualizations are now in mlp_viz.py
NOTE: Layer-position visualizations are now in attn_viz.py and mlp_viz.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ....common.profiler import profile
from ....viz.plot_helpers import finalize_plot

from .fine_results import (
    FineResults,
    PathPatchingResult,
    MultiSiteResult,
)


def _symmetric_vrange(matrix: np.ndarray) -> tuple[float, float]:
    """Compute symmetric vmin/vmax for diverging colormaps (centers 0 at white)."""
    abs_max = np.abs(matrix).max()
    return -abs_max, abs_max


@profile
def visualize_fine(
    results: FineResults | None,
    output_dir: Path,
) -> None:
    """Generate fine-grained patching visualizations.

    Creates plots for path patching and multi-site interaction analysis.

    NOTE: Layer-position visualizations are now in attn_viz.py and mlp_viz.py.

    Args:
        results: FineResults from run_fine_analysis
        output_dir: Directory to save plots
    """
    if results is None:
        print("[viz] No fine-grained results to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Path patching head-to-MLP
    if results.path_to_mlp:
        _plot_path_head_to_mlp(results.path_to_mlp, output_dir)

    # Path patching head-to-head
    if results.path_to_head:
        _plot_path_head_to_head(results.path_to_head, output_dir)

    # Multi-site interaction heatmap
    if results.multi_site:
        _plot_multi_site_interaction(results.multi_site, output_dir)

    # Cross-layer path patching
    if results.cross_layer_paths:
        _plot_cross_layer_paths(results.cross_layer_paths, output_dir)

    print(f"[viz] Fine-grained patching plots saved to {output_dir}")


def _plot_path_head_to_mlp(
    path_results: list[PathPatchingResult],
    output_dir: Path,
) -> None:
    """Plot path patching head-to-MLP connectivity matrix.

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
    vmin, vmax = _symmetric_vrange(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.invert_yaxis()  # Lower layers at bottom
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
    finalize_plot(output_dir / "path_head_to_mlp.png")


def _plot_path_head_to_head(
    path_results: list[PathPatchingResult],
    output_dir: Path,
) -> None:
    """Plot path patching head-to-head connectivity matrix.

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
    vmin, vmax = _symmetric_vrange(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.invert_yaxis()  # Lower layers at bottom
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
    finalize_plot(output_dir / "path_head_to_head.png")


def _plot_multi_site_interaction(
    multi_site: list[MultiSiteResult],
    output_dir: Path,
) -> None:
    """Plot multi-site patching interaction heatmap.

    Rows and columns: top components
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
    vmin, vmax = _symmetric_vrange(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.invert_yaxis()  # Lower layers at bottom
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
    finalize_plot(output_dir / "multi_site_interaction.png")


def _plot_cross_layer_paths(
    cross_layer_results: list[PathPatchingResult],
    output_dir: Path,
) -> None:
    """Plot cross-layer path patching (multi-hop circuit).

    Shows information flow between layers:
    - L19 -> L21 (early circuit)
    - L19/L21 -> L24 (mid circuit)
    - L24 -> L28-31 (late circuit)
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
        groups.append(("L19 -> L21", l19_to_l21))
    if early_to_l24:
        groups.append(("L19/L21 -> L24", early_to_l24))
    if l24_to_later:
        groups.append(("L24 -> L28-31", l24_to_later))

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

        vmin, vmax = _symmetric_vrange(matrix)
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.invert_yaxis()  # Lower layers at bottom
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
    finalize_plot(output_dir / "cross_layer_paths.png")
