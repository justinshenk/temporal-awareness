"""Multi-component comparison visualizations.

Visualizations that compare patching effects across all four components:
resid_pre, attn_out, mlp_out, resid_post.

Requires data from all components to be available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from ....activation_patching.coarse import CoarseActPatchResults, SweepStepResults
from .helpers import setup_grid

# Component order for consistent visualization
COMPONENTS = ["resid_pre", "attn_out", "mlp_out", "resid_post"]
COMPONENT_COLORS = {
    "resid_pre": "#1f77b4",   # blue
    "attn_out": "#ff7f0e",    # orange
    "mlp_out": "#2ca02c",     # green
    "resid_post": "#d62728",  # red
}


def plot_all_component_comparisons(
    results_by_component: dict[str, CoarseActPatchResults],
    output_dir: Path,
    step_size: int = 1,
) -> None:
    """Generate all multi-component comparison plots.

    Args:
        results_by_component: Dict mapping component name to its results
        output_dir: Directory to save plots
        step_size: Step size to use for extracting data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # 1 & 2. Component Attribution Heatmaps (Layers)
    _plot_layer_heatmap(layer_data, output_dir, "denoising")
    _plot_layer_heatmap(layer_data, output_dir, "noising")

    # 3 & 4. Component Attribution Heatmaps (Positions)
    _plot_position_heatmap(pos_data, output_dir, "denoising")
    _plot_position_heatmap(pos_data, output_dir, "noising")

    # 5. Marginal Contribution Line Plot
    _plot_marginal_contribution(layer_data, output_dir)

    # 6 & 7. Attention vs MLP Scatter (per layer)
    _plot_attn_vs_mlp_scatter(layer_data, output_dir, "layer")

    # 8 & 9. Attention vs MLP Scatter (per position)
    _plot_attn_vs_mlp_scatter(pos_data, output_dir, "position")

    # 10. Cumulative Recovery Stacked Area Plot
    _plot_cumulative_recovery(layer_data, output_dir)

    # 11 & 12. Noising vs Denoising Scatter — Per Component
    _plot_noise_vs_denoise_per_component(layer_data, output_dir, "layer")
    _plot_noise_vs_denoise_per_component(pos_data, output_dir, "position")

    # 13. Redundancy Gap Bar Chart
    _plot_redundancy_gap(layer_data, output_dir)

    # 14. Component Importance Ranked Bar Chart
    _plot_component_importance_ranked(layer_data, output_dir)

    # 15. Position × Component Interaction Plot
    _plot_position_component_interaction(pos_data, output_dir)

    # 16. Delta Plot: resid_pre[L+1] vs resid_post[L]
    _plot_resid_delta_sanity(layer_data, output_dir)

    print(f"[viz] Component comparison plots saved to {output_dir}")


def _plot_layer_heatmap(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    mode: Literal["denoising", "noising"],
) -> None:
    """Plot component attribution heatmap for layers.

    Rows: layers, Columns: components, Cell color: recovery/disruption score.
    """
    # Get all layers from any component that has data
    all_layers = set()
    for comp, data in layer_data.items():
        if data:
            all_layers.update(data.keys())

    if not all_layers:
        return

    layers = sorted(all_layers)
    n_layers = len(layers)
    n_components = len(COMPONENTS)

    # Build matrix
    matrix = np.full((n_layers, n_components), np.nan)
    for col_idx, comp in enumerate(COMPONENTS):
        data = layer_data.get(comp)
        if data:
            for row_idx, layer in enumerate(layers):
                if data.get(layer) is not None:
                    val = data[layer].recovery if mode == "denoising" else data[layer].disruption
                    if val is not None:
                        matrix[row_idx, col_idx] = val

    # Plot
    fig, ax = plt.subplots(figsize=(8, max(10, n_layers * 0.3)), facecolor="white")
    ax.set_facecolor("white")

    # Flip matrix so layer 0 is at bottom
    matrix_flipped = matrix[::-1, :]
    layers_flipped = layers[::-1]

    im = ax.imshow(matrix_flipped, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recovery" if mode == "denoising" else "Disruption")

    ax.set_xticks(range(n_components))
    ax.set_xticklabels(COMPONENTS, rotation=45, ha="right")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{lyr}" for lyr in layers_flipped])

    ax.set_xlabel("Component", fontsize=12, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=12, fontweight="bold")

    title = "Denoising Recovery" if mode == "denoising" else "Noising Disruption"
    ax.set_title(f"Component Attribution Heatmap (Layers) - {title}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / f"heatmap_layers_{mode}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / f'heatmap_layers_{mode}.png'}")


def _plot_position_heatmap(
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    mode: Literal["denoising", "noising"],
) -> None:
    """Plot component attribution heatmap for positions."""
    # Get all positions from any component that has data
    all_positions = set()
    for comp, data in pos_data.items():
        if data:
            all_positions.update(data.keys())

    if not all_positions:
        return

    positions = sorted(all_positions)
    n_positions = len(positions)
    n_components = len(COMPONENTS)

    # Build matrix
    matrix = np.full((n_positions, n_components), np.nan)
    for col_idx, comp in enumerate(COMPONENTS):
        data = pos_data.get(comp)
        if data:
            for row_idx, pos in enumerate(positions):
                if data.get(pos) is not None:
                    val = data[pos].recovery if mode == "denoising" else data[pos].disruption
                    if val is not None:
                        matrix[row_idx, col_idx] = val

    # Plot - flip matrix so position 0 is at bottom
    matrix_flipped = matrix[::-1, :]
    positions_flipped = positions[::-1]

    fig, ax = plt.subplots(figsize=(8, max(10, n_positions * 0.15)), facecolor="white")
    ax.set_facecolor("white")

    im = ax.imshow(matrix_flipped, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recovery" if mode == "denoising" else "Disruption")

    ax.set_xticks(range(n_components))
    ax.set_xticklabels(COMPONENTS, rotation=45, ha="right")

    # Show subset of position labels to avoid crowding
    if n_positions > 30:
        step = max(1, n_positions // 20)
        ax.set_yticks(range(0, n_positions, step))
        ax.set_yticklabels([f"P{positions_flipped[i]}" for i in range(0, n_positions, step)])
    else:
        ax.set_yticks(range(n_positions))
        ax.set_yticklabels([f"P{p}" for p in positions_flipped])

    ax.set_xlabel("Component", fontsize=12, fontweight="bold")
    ax.set_ylabel("Position", fontsize=12, fontweight="bold")

    title = "Denoising Recovery" if mode == "denoising" else "Noising Disruption"
    ax.set_title(f"Component Attribution Heatmap (Positions) - {title}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / f"heatmap_positions_{mode}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / f'heatmap_positions_{mode}.png'}")


def _plot_marginal_contribution(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot marginal contribution: resid_post[L] - resid_pre[L] per layer."""
    resid_pre = layer_data.get("resid_pre")
    resid_post = layer_data.get("resid_post")

    if not resid_pre or not resid_post:
        return

    layers = sorted(set(resid_pre.keys()) & set(resid_post.keys()))
    if not layers:
        return

    denoise_marginal = []
    noise_marginal = []
    valid_layers = []

    for layer in layers:
        pre_rec = resid_pre[layer].recovery
        post_rec = resid_post[layer].recovery
        pre_dis = resid_pre[layer].disruption
        post_dis = resid_post[layer].disruption

        if all(v is not None for v in [pre_rec, post_rec, pre_dis, post_dis]):
            valid_layers.append(layer)
            denoise_marginal.append(post_rec - pre_rec)
            noise_marginal.append(post_dis - pre_dis)

    if not valid_layers:
        return

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(valid_layers, denoise_marginal, "o-", color="#2ca02c", linewidth=2,
            markersize=6, label="Denoising (recovery)", alpha=0.8)
    ax.plot(valid_layers, noise_marginal, "s-", color="#d62728", linewidth=2,
            markersize=6, label="Noising (disruption)", alpha=0.8)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("resid_post[L] - resid_pre[L]", fontsize=12, fontweight="bold")
    ax.set_title("Marginal Contribution per Layer", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    setup_grid(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "marginal_contribution.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / 'marginal_contribution.png'}")


def _plot_attn_vs_mlp_scatter(
    data: dict[str, SweepStepResults | None],
    output_dir: Path,
    sweep_type: Literal["layer", "position"],
) -> None:
    """Plot attention vs MLP scatter for each layer/position."""
    attn_data = data.get("attn_out")
    mlp_data = data.get("mlp_out")

    if not attn_data or not mlp_data:
        return

    indices = sorted(set(attn_data.keys()) & set(mlp_data.keys()))
    if not indices:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    for ax_idx, mode in enumerate(["denoising", "noising"]):
        ax = axes[ax_idx]
        ax.set_facecolor("white")

        attn_vals = []
        mlp_vals = []
        labels = []

        for idx in indices:
            if mode == "denoising":
                attn_v = attn_data[idx].recovery
                mlp_v = mlp_data[idx].recovery
            else:
                attn_v = attn_data[idx].disruption
                mlp_v = mlp_data[idx].disruption

            if attn_v is not None and mlp_v is not None:
                attn_vals.append(attn_v)
                mlp_vals.append(mlp_v)
                labels.append(f"{'L' if sweep_type == 'layer' else 'P'}{idx}")

        if not attn_vals:
            continue

        # Color by index
        colors = plt.cm.viridis(np.linspace(0, 1, len(attn_vals)))

        ax.scatter(attn_vals, mlp_vals, c=colors, s=100, edgecolors="black", linewidth=0.5, alpha=0.8)

        # Add labels for extreme points
        for i, (x, y, label) in enumerate(zip(attn_vals, mlp_vals, labels)):
            if abs(x) > 0.1 or abs(y) > 0.1:  # Label points with significant effect
                ax.annotate(label, (x, y), fontsize=8, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")

        # Diagonal line
        max_val = max(max(attn_vals), max(mlp_vals))
        min_val = min(min(attn_vals), min(mlp_vals))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Equal contribution")

        ax.set_xlabel("attn_out effect", fontsize=12, fontweight="bold")
        ax.set_ylabel("mlp_out effect", fontsize=12, fontweight="bold")
        title = "Denoising Recovery" if mode == "denoising" else "Noising Disruption"
        ax.set_title(f"Attention vs MLP ({sweep_type.title()}) - {title}", fontsize=12, fontweight="bold")
        setup_grid(ax)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / f"attn_vs_mlp_{sweep_type}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / f'attn_vs_mlp_{sweep_type}.png'}")


def _plot_cumulative_recovery(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot cumulative recovery stacked area for attn_out and mlp_out."""
    attn_data = layer_data.get("attn_out")
    mlp_data = layer_data.get("mlp_out")

    if not attn_data or not mlp_data:
        return

    layers = sorted(set(attn_data.keys()) & set(mlp_data.keys()))
    if not layers:
        return

    attn_recovery = []
    mlp_recovery = []

    for layer in layers:
        attn_r = attn_data[layer].recovery
        mlp_r = mlp_data[layer].recovery
        attn_recovery.append(attn_r if attn_r is not None else 0)
        mlp_recovery.append(mlp_r if mlp_r is not None else 0)

    # Cumulative sum
    attn_cumsum = np.cumsum(attn_recovery)
    mlp_cumsum = np.cumsum(mlp_recovery)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_facecolor("white")

    ax.fill_between(layers, 0, attn_cumsum, alpha=0.6, color=COMPONENT_COLORS["attn_out"], label="attn_out")
    ax.fill_between(layers, attn_cumsum, attn_cumsum + mlp_cumsum, alpha=0.6, color=COMPONENT_COLORS["mlp_out"], label="mlp_out")

    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Recovery", fontsize=12, fontweight="bold")
    ax.set_title("Cumulative Recovery Build-up Through Network", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    setup_grid(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "cumulative_recovery.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / 'cumulative_recovery.png'}")


def _plot_noise_vs_denoise_per_component(
    data: dict[str, SweepStepResults | None],
    output_dir: Path,
    sweep_type: Literal["layer", "position"],
) -> None:
    """Plot noising vs denoising scatter for each component."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor="white")
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
        recoveries = []
        disruptions = []
        labels = []

        for idx in indices:
            rec = comp_data[idx].recovery
            dis = comp_data[idx].disruption
            if rec is not None and dis is not None:
                recoveries.append(rec)
                disruptions.append(dis)
                labels.append(f"{'L' if sweep_type == 'layer' else 'P'}{idx}")

        if not recoveries:
            ax.text(0.5, 0.5, f"No valid {comp} data", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        colors = plt.cm.viridis(np.linspace(0, 1, len(recoveries)))
        ax.scatter(recoveries, disruptions, c=colors, s=80, edgecolors="black", linewidth=0.5, alpha=0.8)

        # Diagonal line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)

        # Add AND/OR labels
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

    plt.tight_layout()
    fig.savefig(output_dir / f"noise_vs_denoise_per_component_{sweep_type}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / f'noise_vs_denoise_per_component_{sweep_type}.png'}")


def _plot_redundancy_gap(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot redundancy gap (noising - denoising) per layer per component."""
    # Get all layers
    all_layers = set()
    for comp, data in layer_data.items():
        if data:
            all_layers.update(data.keys())

    if not all_layers:
        return

    layers = sorted(all_layers)
    n_layers = len(layers)

    # Components to plot (skip resid_pre for cleaner viz)
    plot_components = ["attn_out", "mlp_out", "resid_post"]

    bar_width = 0.25
    x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(max(12, n_layers * 0.4), 6), facecolor="white")
    ax.set_facecolor("white")

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

        ax.bar(x + i * bar_width, gaps, bar_width, label=comp, color=COMPONENT_COLORS[comp], alpha=0.8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Redundancy Gap (Noising - Denoising)", fontsize=12, fontweight="bold")
    ax.set_title("Redundancy Gap per Layer per Component", fontsize=14, fontweight="bold")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"L{lyr}" for lyr in layers], rotation=45, ha="right")
    ax.legend(loc="best")
    setup_grid(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "redundancy_gap.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / 'redundancy_gap.png'}")


def _plot_component_importance_ranked(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """Plot top N components ranked by denoising recovery."""
    all_components = []

    for comp in ["attn_out", "mlp_out"]:  # Focus on attention and MLP
        data = layer_data.get(comp)
        if not data:
            continue
        for layer, result in data.items():
            if result.recovery is not None:
                label = f"L{layer}_{comp.replace('_out', '')}"
                all_components.append((label, result.recovery, comp))

    if not all_components:
        return

    # Sort by recovery descending
    all_components.sort(key=lambda x: x[1], reverse=True)
    top_components = all_components[:top_n]

    labels = [c[0] for c in top_components]
    values = [c[1] for c in top_components]
    colors = [COMPONENT_COLORS[c[2]] for c in top_components]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)), facecolor="white")
    ax.set_facecolor("white")

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Denoising Recovery", fontsize=12, fontweight="bold")
    ax.set_title(f"Top {top_n} Components by Importance", fontsize=14, fontweight="bold")
    setup_grid(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "component_importance_ranked.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / 'component_importance_ranked.png'}")


def _plot_position_component_interaction(
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot patching effect vs position for each component."""
    # Get all positions
    all_positions = set()
    for comp, data in pos_data.items():
        if data:
            all_positions.update(data.keys())

    if not all_positions:
        return

    positions = sorted(all_positions)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    for ax_idx, mode in enumerate(["denoising", "noising"]):
        ax = axes[ax_idx]
        ax.set_facecolor("white")

        for comp in COMPONENTS:
            data = pos_data.get(comp)
            if not data:
                continue

            values = []
            valid_pos = []
            for pos in positions:
                if data.get(pos) is not None:
                    val = data[pos].recovery if mode == "denoising" else data[pos].disruption
                    if val is not None:
                        values.append(val)
                        valid_pos.append(pos)

            if valid_pos:
                ax.plot(valid_pos, values, "o-", color=COMPONENT_COLORS[comp],
                        linewidth=1.5, markersize=4, label=comp, alpha=0.7)

        ax.set_xlabel("Position", fontsize=12, fontweight="bold")
        ax.set_ylabel("Recovery" if mode == "denoising" else "Disruption", fontsize=12, fontweight="bold")
        title = "Denoising" if mode == "denoising" else "Noising"
        ax.set_title(f"Position × Component Interaction - {title}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        setup_grid(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "position_component_interaction.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / 'position_component_interaction.png'}")


def _plot_resid_delta_sanity(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Sanity check: resid_pre[L+1] vs resid_post[L] should match."""
    resid_pre = layer_data.get("resid_pre")
    resid_post = layer_data.get("resid_post")

    if not resid_pre or not resid_post:
        return

    pre_layers = sorted(resid_pre.keys())
    post_layers = sorted(resid_post.keys())

    if not pre_layers or not post_layers:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    for ax_idx, mode in enumerate(["denoising", "noising"]):
        ax = axes[ax_idx]
        ax.set_facecolor("white")

        # resid_pre[L+1] values
        pre_next_layers = []
        pre_next_values = []
        for layer in pre_layers[1:]:  # Skip first layer
            val = resid_pre[layer].recovery if mode == "denoising" else resid_pre[layer].disruption
            if val is not None:
                pre_next_layers.append(layer - 1)  # Plot at L position
                pre_next_values.append(val)

        # resid_post[L] values
        post_values = []
        post_plot_layers = []
        for layer in post_layers[:-1]:  # Skip last layer
            val = resid_post[layer].recovery if mode == "denoising" else resid_post[layer].disruption
            if val is not None:
                post_plot_layers.append(layer)
                post_values.append(val)

        if pre_next_values:
            ax.plot(pre_next_layers, pre_next_values, "o-", color="#1f77b4", linewidth=2,
                    markersize=6, label="resid_pre[L+1]", alpha=0.8)
        if post_values:
            ax.plot(post_plot_layers, post_values, "s--", color="#d62728", linewidth=2,
                    markersize=6, label="resid_post[L]", alpha=0.8)

        ax.set_xlabel("Layer Index (L)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Recovery" if mode == "denoising" else "Disruption", fontsize=12, fontweight="bold")
        title = "Denoising" if mode == "denoising" else "Noising"
        ax.set_title(f"Sanity Check: resid_pre[L+1] vs resid_post[L] - {title}", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        setup_grid(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "resid_delta_sanity.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / 'resid_delta_sanity.png'}")
