"""Component decomposition plots: attention vs MLP analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from .....activation_patching.coarse import SweepStepResults
from .constants import COMPONENTS, COMPONENT_COLORS
from .utils import adjust_labels, create_figure, get_sqrt_colors, save_plot, setup_grid


def plot_decomposition(
    layer_data: dict[str, SweepStepResults | None],
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Generate all component decomposition plots."""
    _plot_attn_vs_mlp_scatter(layer_data, output_dir, "layer")
    _plot_attn_vs_mlp_scatter(pos_data, output_dir, "position")
    _plot_attn_vs_mlp_paired(layer_data, output_dir)  # NEW: paired scatter with arrows
    _plot_component_importance(layer_data, output_dir)
    _plot_cumulative_recovery(layer_data, output_dir)
    _plot_marginal_contribution(layer_data, output_dir)
    _plot_position_interaction(pos_data, output_dir)
    _plot_position_interaction_zoomed(pos_data, output_dir)  # NEW: zoomed version


def _plot_attn_vs_mlp_scatter(
    data: dict[str, SweepStepResults | None],
    output_dir: Path,
    sweep_type: Literal["layer", "position"],
) -> None:
    """Plot attention vs MLP scatter with all points labeled."""
    attn_data = data.get("attn_out")
    mlp_data = data.get("mlp_out")

    if not attn_data or not mlp_data:
        return

    indices = sorted(set(attn_data.keys()) & set(mlp_data.keys()))
    if not indices:
        return

    # Collect data for both modes to determine shared limits
    all_vals = []
    data_by_mode = {}

    for mode in ["denoising", "noising"]:
        attn_vals, mlp_vals, labels = [], [], []
        for idx in indices:
            attn_v = attn_data[idx].recovery if mode == "denoising" else attn_data[idx].disruption
            mlp_v = mlp_data[idx].recovery if mode == "denoising" else mlp_data[idx].disruption
            if attn_v is not None and mlp_v is not None:
                attn_vals.append(attn_v)
                mlp_vals.append(mlp_v)
                labels.append(f"{'L' if sweep_type == 'layer' else 'P'}{idx}")
                all_vals.extend([attn_v, mlp_v])
        data_by_mode[mode] = (attn_vals, mlp_vals, labels)

    if not all_vals:
        return

    # Shared axis limits
    max_val = max(all_vals) * 1.1
    min_val = min(min(all_vals) - 0.05, -0.05)

    fig, axes = create_figure(1, 2, figsize=(16, 7))

    for ax_idx, mode in enumerate(["denoising", "noising"]):
        ax = axes[ax_idx]
        attn_vals, mlp_vals, labels = data_by_mode[mode]

        if not attn_vals:
            continue

        color_values = get_sqrt_colors(list(range(len(attn_vals))))
        ax.scatter(attn_vals, mlp_vals, c=color_values, cmap="viridis", vmin=0, vmax=1,
                   s=100, edgecolors="black", linewidth=0.5, alpha=0.8)

        # Label ALL points
        texts = [ax.text(x, y, label, fontsize=7, alpha=0.8)
                 for x, y, label in zip(attn_vals, mlp_vals, labels)]
        adjust_labels(texts, ax)

        # Reference elements
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.4)

        # Quadrant labels
        ax.text(0.85, 0.85, "Both", fontsize=14, fontweight="bold", color="gray", alpha=0.2,
                transform=ax.transAxes, ha="center", va="center")
        ax.text(0.15, 0.15, "Neither", fontsize=14, fontweight="bold", color="gray", alpha=0.2,
                transform=ax.transAxes, ha="center", va="center")
        ax.text(0.85, 0.15, "Attn", fontsize=14, fontweight="bold", color="gray", alpha=0.2,
                transform=ax.transAxes, ha="center", va="center")
        ax.text(0.15, 0.85, "MLP", fontsize=14, fontweight="bold", color="gray", alpha=0.2,
                transform=ax.transAxes, ha="center", va="center")

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel("attn_out effect", fontsize=12, fontweight="bold")
        ax.set_ylabel("mlp_out effect", fontsize=12, fontweight="bold")
        title = "Denoising Recovery" if mode == "denoising" else "Noising Disruption"
        ax.set_title(f"Attention vs MLP ({sweep_type.title()}) - {title}", fontsize=12, fontweight="bold")
        setup_grid(ax)

    plt.tight_layout()
    save_plot(fig, output_dir, f"attn_vs_mlp_{sweep_type}.png")


def _plot_attn_vs_mlp_paired(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot paired scatter with arrows showing denoising→noising movement per layer.

    Only draws arrows for layers that move more than a threshold distance to avoid
    cluttering the dense central cluster.
    """
    attn_data = layer_data.get("attn_out")
    mlp_data = layer_data.get("mlp_out")

    if not attn_data or not mlp_data:
        return

    layers = sorted(set(attn_data.keys()) & set(mlp_data.keys()))
    if not layers:
        return

    fig, ax = create_figure(figsize=(12, 10))

    all_vals = []
    paired_data = []

    for layer in layers:
        attn_den = attn_data[layer].recovery
        mlp_den = mlp_data[layer].recovery
        attn_noi = attn_data[layer].disruption
        mlp_noi = mlp_data[layer].disruption

        if all(v is not None for v in [attn_den, mlp_den, attn_noi, mlp_noi]):
            paired_data.append((layer, attn_den, mlp_den, attn_noi, mlp_noi))
            all_vals.extend([attn_den, mlp_den, attn_noi, mlp_noi])

    if not paired_data:
        return

    max_val = max(all_vals) * 1.1
    min_val = min(min(all_vals) - 0.05, -0.05)

    # Color by layer
    color_values = get_sqrt_colors([d[0] for d in paired_data])
    cmap = plt.cm.viridis

    # Calculate movement distances for threshold
    movements = []
    for layer, attn_den, mlp_den, attn_noi, mlp_noi in paired_data:
        dist = np.sqrt((attn_noi - attn_den) ** 2 + (mlp_noi - mlp_den) ** 2)
        movements.append(dist)

    # Only draw arrows for layers that move more than median distance
    # This prevents the central cluster from becoming illegible
    movement_threshold = np.median(movements) if movements else 0

    for i, (layer, attn_den, mlp_den, attn_noi, mlp_noi) in enumerate(paired_data):
        color = cmap(color_values[i])
        dist = movements[i]

        # Denoising point (circle)
        ax.scatter(attn_den, mlp_den, c=[color], s=80, marker="o", edgecolors="black", linewidth=0.5)

        # Noising point (square)
        ax.scatter(attn_noi, mlp_noi, c=[color], s=80, marker="s", edgecolors="black", linewidth=0.5)

        # Only draw arrow and label if movement exceeds threshold
        if dist > movement_threshold:
            # Arrow from denoising to noising
            ax.annotate("", xy=(attn_noi, mlp_noi), xytext=(attn_den, mlp_den),
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.6, lw=1.5))

            # Label at midpoint
            mid_x = (attn_den + attn_noi) / 2
            mid_y = (mlp_den + mlp_noi) / 2
            ax.text(mid_x, mid_y, f"L{layer}", fontsize=7, alpha=0.7, ha="center", va="center")

    # Reference elements
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.4)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("attn_out effect", fontsize=12, fontweight="bold")
    ax.set_ylabel("mlp_out effect", fontsize=12, fontweight="bold")
    ax.set_title("Paired Attn vs MLP: Denoising (○) → Noising (□)\n(Arrows shown only for layers with significant movement)",
                 fontsize=14, fontweight="bold")

    # Legend
    ax.scatter([], [], c="gray", s=80, marker="o", label="Denoising")
    ax.scatter([], [], c="gray", s=80, marker="s", label="Noising")
    ax.legend(loc="upper left")

    setup_grid(ax)
    plt.tight_layout()
    save_plot(fig, output_dir, "attn_vs_mlp_paired.png")


def _plot_component_importance(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """Plot top N components with denoising and noising scores side-by-side.

    Same-layer components (e.g., L24_attn and L24_mlp both in top N) are visually
    linked with colored brackets and shared background highlighting.
    """
    all_components = []

    for comp in ["attn_out", "mlp_out"]:
        data = layer_data.get(comp)
        if not data:
            continue
        for layer, result in data.items():
            if result.recovery is not None and result.disruption is not None:
                all_components.append({
                    "label": f"L{layer}_{comp.replace('_out', '')}",
                    "layer": layer,
                    "recovery": result.recovery,
                    "disruption": result.disruption,
                    "comp": comp,
                })

    if not all_components:
        return

    # Sort by recovery
    all_components.sort(key=lambda x: x["recovery"], reverse=True)
    top_components = all_components[:top_n]

    labels = [c["label"] for c in top_components]
    recoveries = [c["recovery"] for c in top_components]
    disruptions = [c["disruption"] for c in top_components]
    colors = [COMPONENT_COLORS[c["comp"]] for c in top_components]
    layers_used = [c["layer"] for c in top_components]

    # Find layers that appear multiple times
    layer_counts = {}
    for layer in layers_used:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    multi_layers = {layer for layer, count in layer_counts.items() if count > 1}

    # Assign unique colors to multi-layer groups for visual linking
    multi_layer_colors = {}
    link_colors = ["#FFD700", "#FF6B6B", "#4ECDC4", "#9B59B6", "#3498DB"]  # Gold, coral, teal, purple, blue
    for i, layer in enumerate(sorted(multi_layers)):
        multi_layer_colors[layer] = link_colors[i % len(link_colors)]

    fig, ax = create_figure(figsize=(12, max(6, top_n * 0.5)))

    y_pos = np.arange(len(labels))
    bar_height = 0.35

    # Background highlighting for same-layer components
    for i, layer in enumerate(layers_used):
        if layer in multi_layers:
            ax.axhspan(y_pos[i] - 0.45, y_pos[i] + 0.45, alpha=0.15,
                       color=multi_layer_colors[layer], zorder=0)

    # Bars
    ax.barh(y_pos - bar_height / 2, recoveries, bar_height, color=colors, alpha=0.8,
            edgecolor="black", label="Denoising Recovery")
    ax.barh(y_pos + bar_height / 2, disruptions, bar_height, color=colors, alpha=0.4,
            edgecolor="black", label="Noising Disruption", hatch="//")

    # Mark same-layer components with colored bracket and label
    for i, layer in enumerate(layers_used):
        if layer in multi_layers:
            bracket_color = multi_layer_colors[layer]
            # Draw bracket on left side
            ax.plot([-0.02, -0.02], [y_pos[i] - 0.3, y_pos[i] + 0.3],
                    color=bracket_color, linewidth=4, transform=ax.get_yaxis_transform(),
                    clip_on=False, solid_capstyle="butt")
            # Small text label
            ax.text(-0.04, y_pos[i], f"L{layer}", fontsize=7, fontweight="bold",
                    color=bracket_color, ha="right", va="center",
                    transform=ax.get_yaxis_transform())

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Effect Score", fontsize=12, fontweight="bold")
    ax.set_title(f"Top {top_n} Components by Importance\n(Colored brackets link same-layer attn+mlp pairs)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    setup_grid(ax)

    # Cumulative percentage on secondary axis
    ax2 = ax.twiny()
    total_recovery = sum(c["recovery"] for c in all_components)
    if total_recovery > 0:
        cumsum = np.cumsum(recoveries)
        cum_pct = (cumsum / total_recovery) * 100
        ax2.plot(cum_pct, y_pos, "ko-", markersize=4, linewidth=1.5, alpha=0.7)
        ax2.set_xlabel("Cumulative % of Total Recovery", fontsize=10)
        ax2.set_xlim(0, 100)

        idx_80 = np.searchsorted(cum_pct, 80)
        if idx_80 < len(y_pos):
            ax2.axvline(x=80, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_plot(fig, output_dir, "component_importance_ranked.png")


def _plot_cumulative_recovery(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot cumulative recovery stacked area with dip annotations."""
    attn_data = layer_data.get("attn_out")
    mlp_data = layer_data.get("mlp_out")

    if not attn_data or not mlp_data:
        return

    layers = sorted(set(attn_data.keys()) & set(mlp_data.keys()))
    if not layers:
        return

    attn_recovery = [attn_data[layer].recovery or 0 for layer in layers]
    mlp_recovery = [mlp_data[layer].recovery or 0 for layer in layers]

    attn_cumsum = np.cumsum(attn_recovery)
    mlp_cumsum = np.cumsum(mlp_recovery)
    total_cumsum = attn_cumsum + mlp_cumsum

    fig, ax = create_figure(figsize=(12, 6))

    ax.fill_between(layers, 0, attn_cumsum, alpha=0.6, color=COMPONENT_COLORS["attn_out"], label="attn_out")
    ax.fill_between(layers, attn_cumsum, total_cumsum, alpha=0.6, color=COMPONENT_COLORS["mlp_out"], label="mlp_out")

    # Reference line at y=1.0
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax.annotate("Full Recovery (1.0)", xy=(layers[0], 1.0), xytext=(layers[0] + 2, 1.05),
                fontsize=10, fontweight="bold", color="black", alpha=0.7)

    # Detect and annotate dips in cumulative attention
    # Two types of dips:
    # 1. Explicitly negative attention (attn_val < 0) - counterproductive
    # 2. Relative dips - where attention contribution drops significantly vs recent average
    if len(attn_recovery) > 3:
        # Calculate rolling average for comparison
        window = 3
        for i in range(window, len(attn_recovery)):
            layer = layers[i]
            attn_val = attn_recovery[i]
            recent_avg = np.mean(attn_recovery[max(0, i-window):i])

            # Type 1: Explicitly negative
            if attn_val < -0.02:
                ax.annotate(f"L{layer} attn\ncounterproductive",
                            xy=(layer, attn_cumsum[i]), xytext=(layer + 1, attn_cumsum[i] - 0.15),
                            fontsize=8, fontweight="bold", color="red", alpha=0.8,
                            arrowprops=dict(arrowstyle="->", color="red", alpha=0.5))
            # Type 2: Relative dip (current << recent average)
            elif recent_avg > 0.05 and attn_val < recent_avg * 0.3:
                ax.annotate(f"L{layer} attn\ndip",
                            xy=(layer, attn_cumsum[i]), xytext=(layer + 1, attn_cumsum[i] + 0.1),
                            fontsize=7, color="orange", alpha=0.8,
                            arrowprops=dict(arrowstyle="->", color="orange", alpha=0.4))

    # Mark key layers (top contributors)
    total_per_layer = [a + m for a, m in zip(attn_recovery, mlp_recovery)]
    key_layers = sorted(zip(layers, total_per_layer), key=lambda x: x[1], reverse=True)[:5]
    for layer, val in key_layers:
        ax.axvline(x=layer, color="gray", linestyle=":", alpha=0.5, linewidth=1.5)
        idx = layers.index(layer)
        ax.annotate(f"L{layer}", xy=(layer, total_cumsum[idx]), xytext=(layer + 0.5, total_cumsum[idx] + 0.1),
                    fontsize=8, fontweight="bold", color="gray")

    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Recovery", fontsize=12, fontweight="bold")
    ax.set_title("Cumulative Recovery Build-up Through Network", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    setup_grid(ax)

    plt.tight_layout()
    save_plot(fig, output_dir, "cumulative_recovery.png")


def _plot_marginal_contribution(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot marginal contribution with secondary y-axis for absolute values."""
    resid_pre = layer_data.get("resid_pre")
    resid_post = layer_data.get("resid_post")

    if not resid_pre or not resid_post:
        return

    layers = sorted(set(resid_pre.keys()) & set(resid_post.keys()))
    if not layers:
        return

    denoise_marginal = []
    noise_marginal = []
    resid_post_denoise = []
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
            resid_post_denoise.append(post_rec)

    if not valid_layers:
        return

    fig, ax = create_figure(figsize=(12, 6))

    ax.plot(valid_layers, denoise_marginal, "o-", color="#2ca02c", linewidth=2,
            markersize=6, label="Denoising (recovery)", alpha=0.8)
    ax.plot(valid_layers, noise_marginal, "s-", color="#d62728", linewidth=2,
            markersize=6, label="Noising (disruption)", alpha=0.8)

    # Mark key layers
    denoise_sorted = sorted(zip(valid_layers, denoise_marginal), key=lambda x: abs(x[1]), reverse=True)
    for layer, val in denoise_sorted[:3]:
        ax.axvline(x=layer, color="#2ca02c", linestyle=":", alpha=0.4, linewidth=1.5)
        ax.annotate(f"L{layer}", (layer, val), fontsize=8, color="#2ca02c",
                    xytext=(5, 5), textcoords="offset points", fontweight="bold")

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Marginal: resid_post[L] - resid_pre[L]", fontsize=12, fontweight="bold")
    ax.set_title("Marginal Contribution per Layer", fontsize=14, fontweight="bold")

    # Secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(valid_layers, resid_post_denoise, "^--", color="#9467bd", linewidth=1.5,
             markersize=4, label="Absolute resid_post (denoising)", alpha=0.6)
    ax2.set_ylabel("Absolute resid_post Recovery", fontsize=10, color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    setup_grid(ax)
    plt.tight_layout()
    save_plot(fig, output_dir, "marginal_contribution.png")


def _detect_hub_regions(
    pos_data: dict[str, SweepStepResults | None],
    positions: list[int],
    threshold: float,
) -> list[tuple[int, int]]:
    """Detect contiguous regions where average effect exceeds threshold."""
    avg_effects = []
    for pos in positions:
        effects = []
        for comp in COMPONENTS:
            data = pos_data.get(comp)
            if data and data.get(pos) is not None:
                rec = data[pos].recovery
                dis = data[pos].disruption
                if rec is not None:
                    effects.append(rec)
                if dis is not None:
                    effects.append(dis)
        avg_effects.append(np.mean(effects) if effects else 0)

    regions = []
    in_region = False
    start = None

    for i, (pos, val) in enumerate(zip(positions, avg_effects)):
        if val >= threshold and not in_region:
            in_region = True
            start = pos
        elif val < threshold and in_region:
            in_region = False
            regions.append((start, positions[i - 1]))

    if in_region:
        regions.append((start, positions[-1]))

    return regions


def _plot_position_interaction(
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot position × component interaction with hub shading and shared y-scale."""
    all_positions = set()
    for comp, data in pos_data.items():
        if data:
            all_positions.update(data.keys())

    if not all_positions:
        return

    positions = sorted(all_positions)

    # Collect all values for shared limits
    all_values = []
    data_by_mode = {}

    for mode in ["denoising", "noising"]:
        mode_data = {}
        for comp in COMPONENTS:
            data = pos_data.get(comp)
            if not data:
                continue
            values, valid_pos = [], []
            for pos in positions:
                if data.get(pos) is not None:
                    val = data[pos].recovery if mode == "denoising" else data[pos].disruption
                    if val is not None:
                        values.append(val)
                        valid_pos.append(pos)
                        all_values.append(val)
            if valid_pos:
                mode_data[comp] = (valid_pos, values)
        data_by_mode[mode] = mode_data

    if not all_values:
        return

    y_min = min(all_values) - 0.05
    y_max = max(all_values) + 0.05

    hub_threshold = np.percentile(all_values, 85)
    hub_regions = _detect_hub_regions(pos_data, positions, hub_threshold)

    fig, axes = create_figure(1, 2, figsize=(16, 6))

    for ax_idx, mode in enumerate(["denoising", "noising"]):
        ax = axes[ax_idx]

        # Hub shading
        for start, end in hub_regions:
            ax.axvspan(start, end, alpha=0.15, color="yellow", zorder=0)

        mode_data = data_by_mode[mode]
        for comp in COMPONENTS:
            if comp in mode_data:
                valid_pos, values = mode_data[comp]
                ax.plot(valid_pos, values, "o-", color=COMPONENT_COLORS[comp],
                        linewidth=1.5, markersize=4, label=comp, alpha=0.7)

        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Position", fontsize=12, fontweight="bold")
        ax.set_ylabel("Recovery" if mode == "denoising" else "Disruption", fontsize=12, fontweight="bold")
        title = "Denoising" if mode == "denoising" else "Noising"
        ax.set_title(f"Position × Component Interaction - {title}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        setup_grid(ax)

    plt.tight_layout()
    save_plot(fig, output_dir, "position_component_interaction.png")


def _plot_position_interaction_zoomed(
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Plot position interaction with zoomed panels for hub regions."""
    all_positions = set()
    for comp, data in pos_data.items():
        if data:
            all_positions.update(data.keys())

    if not all_positions:
        return

    positions = sorted(all_positions)

    # Collect all values
    all_values = []
    data_by_comp = {}

    for comp in COMPONENTS:
        data = pos_data.get(comp)
        if not data:
            continue
        values_den, values_noi, valid_pos = [], [], []
        for pos in positions:
            if data.get(pos) is not None:
                rec = data[pos].recovery
                dis = data[pos].disruption
                if rec is not None and dis is not None:
                    values_den.append(rec)
                    values_noi.append(dis)
                    valid_pos.append(pos)
                    all_values.extend([rec, dis])
        if valid_pos:
            data_by_comp[comp] = (valid_pos, values_den, values_noi)

    if not all_values:
        return

    # Detect hub regions
    hub_threshold = np.percentile(all_values, 80)
    hub_regions = _detect_hub_regions(pos_data, positions, hub_threshold)

    if len(hub_regions) < 2:
        return  # Need at least 2 regions for zoomed view

    # Take first and last hub regions
    region1 = hub_regions[0]
    region2 = hub_regions[-1]

    y_min = min(all_values) - 0.05
    y_max = max(all_values) + 0.05

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")

    for row_idx, region in enumerate([region1, region2]):
        start, end = region
        padding = max(3, (end - start) // 2)
        x_min = max(min(positions), start - padding)
        x_max = min(max(positions), end + padding)

        for col_idx, mode in enumerate(["denoising", "noising"]):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("white")

            # Highlight region
            ax.axvspan(start, end, alpha=0.2, color="yellow", zorder=0)

            for comp in COMPONENTS:
                if comp not in data_by_comp:
                    continue
                valid_pos, values_den, values_noi = data_by_comp[comp]
                values = values_den if mode == "denoising" else values_noi

                # Filter to region
                mask = [(p >= x_min and p <= x_max) for p in valid_pos]
                region_pos = [p for p, m in zip(valid_pos, mask) if m]
                region_vals = [v for v, m in zip(values, mask) if m]

                if region_pos:
                    ax.plot(region_pos, region_vals, "o-", color=COMPONENT_COLORS[comp],
                            linewidth=2, markersize=6, label=comp, alpha=0.8)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Position", fontsize=10, fontweight="bold")
            ax.set_ylabel("Recovery" if mode == "denoising" else "Disruption", fontsize=10)

            region_label = f"P{start}-P{end}"
            mode_label = "Denoising" if mode == "denoising" else "Noising"
            ax.set_title(f"{region_label} ({mode_label})", fontsize=11, fontweight="bold")

            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="best", fontsize=8)
            setup_grid(ax)

    fig.suptitle("Position Interaction - Zoomed Hub Regions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, output_dir, "position_interaction_zoomed.png")
