"""Component decomposition plots: attention vs MLP analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from ......activation_patching.coarse import SweepStepResults
from ...coarse_results import ComponentComparisonResults
from .comp_constants import COMPONENTS, COMPONENT_COLORS
from .comp_utils import adjust_labels, create_figure, get_sqrt_colors, save_plot, setup_grid


def _per_pair_layer_values(
    agg,
    metric: str = "recovery",
) -> dict[int, list[float]]:
    """Extract per-pair values by layer from a CoarseActPatchAggregatedResults.

    Args:
        agg: CoarseActPatchAggregatedResults with by_sample
        metric: "recovery" or "disruption"

    Returns:
        {layer: [value_pair0, value_pair1, ...]}
    """
    out: dict[int, list[float]] = {}
    if not hasattr(agg, "by_sample") or not agg.by_sample:
        return out
    for sample in agg.by_sample.values():
        steps = list(sample.layer_results.keys())
        if not steps:
            continue
        step = min(int(s) for s in steps)
        sw = sample.get_layer_results_for_step(step)
        if not sw:
            continue
        for layer in sw.keys():
            tr = sw.get(layer)
            if tr is None:
                continue
            val = getattr(tr, metric, None)
            if val is not None:
                out.setdefault(int(layer), []).append(float(val))
    return out


def plot_decomposition(
    layer_data: dict[str, SweepStepResults | None],
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    processed_results: ComponentComparisonResults | None = None,
    position_mapping=None,
    agg_by_component: dict | None = None,
) -> None:
    """Generate all component decomposition plots."""
    _plot_attn_vs_mlp_scatter(layer_data, output_dir, "layer", agg_by_component=agg_by_component)
    _plot_attn_vs_mlp_scatter(pos_data, output_dir, "position", agg_by_component=agg_by_component)
    _plot_attn_vs_mlp_paired(layer_data, output_dir)
    _plot_component_importance(layer_data, output_dir, agg_by_component=agg_by_component)
    _plot_cumulative_recovery(layer_data, output_dir, agg_by_component=agg_by_component)
    _plot_marginal_contribution(layer_data, output_dir, agg_by_component=agg_by_component)
    if agg_by_component:
        _plot_marginal_contribution_var(agg_by_component, output_dir)
    _plot_layer_interaction(layer_data, output_dir, agg_by_component=agg_by_component)
    _plot_position_interaction(pos_data, output_dir, position_mapping, agg_by_component=agg_by_component)
    _plot_position_interaction_zoomed(pos_data, output_dir)  # NEW: zoomed version


def _plot_attn_vs_mlp_scatter(
    data: dict[str, SweepStepResults | None],
    output_dir: Path,
    sweep_type: Literal["layer", "position"],
    agg_by_component: dict | None = None,
) -> None:
    """Plot attention vs MLP scatter with ±std crosshairs when multi-pair data available."""
    attn_data = data.get("attn_out")
    mlp_data = data.get("mlp_out")

    if not attn_data or not mlp_data:
        return

    indices = sorted(set(attn_data.keys()) & set(mlp_data.keys()))
    if not indices:
        return

    # Per-pair std for error crosshairs
    _ppv = _per_pair_layer_values if sweep_type == "layer" else _per_pair_position_values
    attn_std: dict[str, dict[int, float]] = {}
    mlp_std: dict[str, dict[int, float]] = {}
    if agg_by_component:
        for metric in ["recovery", "disruption"]:
            attn_agg = agg_by_component.get("attn_out")
            mlp_agg = agg_by_component.get("mlp_out")
            if attn_agg:
                attn_std[metric] = {l: float(np.std(vs)) for l, vs in _ppv(attn_agg, metric).items() if len(vs) > 1}
            if mlp_agg:
                mlp_std[metric] = {l: float(np.std(vs)) for l, vs in _ppv(mlp_agg, metric).items() if len(vs) > 1}

    all_vals = []
    data_by_mode = {}

    for mode in ["denoising", "noising"]:
        metric = "recovery" if mode == "denoising" else "disruption"
        attn_vals, mlp_vals, labels, a_errs, m_errs = [], [], [], [], []
        for idx in indices:
            attn_v = attn_data[idx].recovery if mode == "denoising" else attn_data[idx].disruption
            mlp_v = mlp_data[idx].recovery if mode == "denoising" else mlp_data[idx].disruption
            if attn_v is not None and mlp_v is not None:
                attn_vals.append(attn_v)
                mlp_vals.append(mlp_v)
                labels.append(f"{'L' if sweep_type == 'layer' else 'P'}{idx}")
                all_vals.extend([attn_v, mlp_v])
                a_errs.append(attn_std.get(metric, {}).get(int(idx), 0.0))
                m_errs.append(mlp_std.get(metric, {}).get(int(idx), 0.0))
        data_by_mode[mode] = (attn_vals, mlp_vals, labels, a_errs, m_errs)

    if not all_vals:
        return

    # Shared axis limits
    max_val = max(all_vals) * 1.1
    min_val = min(min(all_vals) - 0.05, -0.05)

    fig, axes = create_figure(1, 2, figsize=(16, 7))

    for ax_idx, mode in enumerate(["denoising", "noising"]):
        ax = axes[ax_idx]
        attn_vals, mlp_vals, labels, a_errs, m_errs = data_by_mode[mode]

        if not attn_vals:
            continue

        color_values = get_sqrt_colors(list(range(len(attn_vals))))

        # Error crosshairs (±std) before scatter so dots render on top
        if any(e > 0 for e in a_errs) or any(e > 0 for e in m_errs):
            ax.errorbar(attn_vals, mlp_vals, xerr=a_errs, yerr=m_errs,
                        fmt='none', ecolor='gray', elinewidth=0.8, alpha=0.4, capsize=2)

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
        dist = movements[i]
        if dist <= movement_threshold:
            continue  # Skip layers without significant movement
        color = cmap(color_values[i])

        ax.scatter(attn_den, mlp_den, c=[color], s=80, marker="o", edgecolors="black", linewidth=0.5)
        ax.scatter(attn_noi, mlp_noi, c=[color], s=80, marker="s", edgecolors="black", linewidth=0.5)
        ax.annotate("", xy=(attn_noi, mlp_noi), xytext=(attn_den, mlp_den),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.6, lw=1.5))
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
    ax.set_title("Paired Attn vs MLP Significant: Denoising (○) → Noising (□)",
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
    top_n: int = 20,
    agg_by_component: dict | None = None,
) -> None:
    """Top N components with denoising/noising bars and ±std error bars."""
    # Build per-component per-layer std from aggregated data
    rec_std: dict[str, dict[int, float]] = {}
    dis_std: dict[str, dict[int, float]] = {}
    if agg_by_component:
        for comp in ["attn_out", "mlp_out"]:
            agg = agg_by_component.get(comp)
            if agg is None:
                continue
            rv = _per_pair_layer_values(agg, "recovery")
            dv = _per_pair_layer_values(agg, "disruption")
            rec_std[comp] = {l: float(np.std(vs)) for l, vs in rv.items() if len(vs) > 1}
            dis_std[comp] = {l: float(np.std(vs)) for l, vs in dv.items() if len(vs) > 1}

    all_components = []
    for comp in ["attn_out", "mlp_out"]:
        data = layer_data.get(comp)
        if not data:
            continue
        for layer, result in data.items():
            if result.recovery is not None and result.disruption is not None:
                all_components.append({
                    "label": f"L{layer}_{comp.replace('_out', '')}",
                    "layer": int(layer),
                    "recovery": result.recovery,
                    "disruption": result.disruption,
                    "rec_std": rec_std.get(comp, {}).get(int(layer), 0.0),
                    "dis_std": dis_std.get(comp, {}).get(int(layer), 0.0),
                    "comp": comp,
                })

    if not all_components:
        return

    all_components.sort(key=lambda x: x["recovery"] + x["disruption"], reverse=True)
    top_components = all_components[:top_n]

    labels = [c["label"] for c in top_components]
    recoveries = [c["recovery"] for c in top_components]
    disruptions = [c["disruption"] for c in top_components]
    rec_errs = [c["rec_std"] for c in top_components]
    dis_errs = [c["dis_std"] for c in top_components]
    colors = [COMPONENT_COLORS[c["comp"]] for c in top_components]

    fig, ax = create_figure(figsize=(12, max(6, top_n * 0.5)))
    y_pos = np.arange(len(labels))
    bar_height = 0.35

    ax.barh(y_pos - bar_height / 2, recoveries, bar_height, xerr=rec_errs,
            color=colors, alpha=0.8, edgecolor="black", capsize=2, ecolor="gray",
            label="Denoising Recovery")
    ax.barh(y_pos + bar_height / 2, disruptions, bar_height, xerr=dis_errs,
            color=colors, alpha=0.4, edgecolor="black", capsize=2, ecolor="gray",
            label="Noising Disruption", hatch="//")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Effect Score", fontsize=12, fontweight="bold")
    ax.set_title(f"Top {top_n} Components by Importance (mean ± std)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    setup_grid(ax)
    plt.tight_layout()
    save_plot(fig, output_dir, "component_importance_ranked.png")


def _plot_cumulative_recovery(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    agg_by_component: dict | None = None,
) -> None:
    """Plot cumulative recovery stacked area with dip annotations."""
    attn_data = layer_data.get("attn_out")
    mlp_data = layer_data.get("mlp_out")
    resid_post_data = layer_data.get("resid_post")

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

    # resid_post recovery (absolute, not cumulative)
    if resid_post_data:
        rp_layers = [l for l in layers if l in resid_post_data and resid_post_data[l].recovery is not None]
        rp_vals = [resid_post_data[l].recovery for l in rp_layers]
        if rp_layers:
            ax.fill_between(
                rp_layers, 0, rp_vals,
                color=COMPONENT_COLORS.get("resid_post", "#d62728"),
                alpha=0.25, label="resid_post",
            )
            ax.plot(rp_layers, rp_vals, "-",
                    color=COMPONENT_COLORS.get("resid_post", "#d62728"),
                    linewidth=2)
            # ±std band for resid_post across pairs
            if agg_by_component and agg_by_component.get("resid_post"):
                rp_pair = _per_pair_layer_values(agg_by_component["resid_post"], "recovery")
                rp_stds = [float(np.std(rp_pair.get(int(l), [0]))) for l in rp_layers]
                lo = [v - s for v, s in zip(rp_vals, rp_stds)]
                hi = [v + s for v, s in zip(rp_vals, rp_stds)]
                ax.fill_between(rp_layers, lo, hi,
                                color=COMPONENT_COLORS.get("resid_post", "#d62728"), alpha=0.10)

    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Recovery", fontsize=12, fontweight="bold")
    ax.set_title("Cumulative Recovery Build-up Through Network", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    setup_grid(ax)

    plt.tight_layout()
    save_plot(fig, output_dir, "cumulative_recovery.png")


def _plot_marginal_contribution(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    agg_by_component: dict | None = None,
) -> None:
    """Plot marginal contribution with ±std bands and secondary y-axis."""
    resid_pre = layer_data.get("resid_pre")
    resid_mid = layer_data.get("resid_mid")
    resid_post = layer_data.get("resid_post")

    if not resid_pre or not resid_post:
        return

    layers = sorted(set(resid_pre.keys()) & set(resid_post.keys()))
    if not layers:
        return

    denoise_marginal = []
    noise_marginal = []
    resid_pre_denoise = []
    resid_mid_denoise = []
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
            resid_pre_denoise.append(pre_rec)
            resid_post_denoise.append(post_rec)
            # Get resid_mid if available
            if resid_mid and layer in resid_mid and resid_mid[layer].recovery is not None:
                resid_mid_denoise.append(resid_mid[layer].recovery)
            else:
                resid_mid_denoise.append(None)

    if not valid_layers:
        return

    fig, ax = create_figure(figsize=(12, 6))

    ax.plot(valid_layers, denoise_marginal, "o-", color="#2ca02c", linewidth=2,
            markersize=6, label="Δ Sufficiency", alpha=0.8)
    ax.plot(valid_layers, noise_marginal, "s-", color="#d62728", linewidth=2,
            markersize=6, label="Δ Necessity", alpha=0.8)

    # ±std bands from per-pair data
    if agg_by_component:
        pre_agg = agg_by_component.get("resid_pre")
        post_agg = agg_by_component.get("resid_post")
        if pre_agg and post_agg:
            pre_rec = _per_pair_layer_values(pre_agg, "recovery")
            post_rec = _per_pair_layer_values(post_agg, "recovery")
            pre_dis = _per_pair_layer_values(pre_agg, "disruption")
            post_dis = _per_pair_layer_values(post_agg, "disruption")
            suff_stds, nec_stds = [], []
            for L in valid_layers:
                n = min(len(pre_rec.get(int(L), [])), len(post_rec.get(int(L), [])))
                if n > 1:
                    sg = np.array(post_rec[int(L)][:n]) - np.array(pre_rec[int(L)][:n])
                    suff_stds.append(float(sg.std()))
                else:
                    suff_stds.append(0.0)
                n2 = min(len(pre_dis.get(int(L), [])), len(post_dis.get(int(L), [])))
                if n2 > 1:
                    ng = np.array(post_dis[int(L)][:n2]) - np.array(pre_dis[int(L)][:n2])
                    nec_stds.append(float(ng.std()))
                else:
                    nec_stds.append(0.0)
            dm = np.array(denoise_marginal); nm = np.array(noise_marginal)
            ss = np.array(suff_stds); ns = np.array(nec_stds)
            ax.fill_between(valid_layers, dm - ss, dm + ss, color="#2ca02c", alpha=0.15)
            ax.fill_between(valid_layers, nm - ns, nm + ns, color="#d62728", alpha=0.15)

    # Mark key layers (vertical guides only)
    denoise_sorted = sorted(zip(valid_layers, denoise_marginal), key=lambda x: abs(x[1]), reverse=True)
    for layer, _val in denoise_sorted[:3]:
        ax.axvline(x=layer, color="#2ca02c", linestyle=":", alpha=0.4, linewidth=1.5)

    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Marginal: resid_post[L] - resid_pre[L]", fontsize=12, fontweight="bold")
    ax.set_title("Marginal Contribution per Layer", fontsize=14, fontweight="bold")

    # Secondary y-axis: only resid_post recovery, faded
    ax2 = ax.twinx()
    ax2.plot(valid_layers, resid_post_denoise, "^--", color=COMPONENT_COLORS["resid_post"],
             linewidth=1.5, markersize=5, label="resid_post", alpha=0.35)
    ax2.set_ylabel("Recovery", fontsize=10)

    # Combined legend - place outside plot to avoid overlap
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
              bbox_to_anchor=(0, -0.12), fontsize=9, ncol=5, frameon=False)

    # Ensure x-axis only shows integer layer values
    ax.set_xticks(valid_layers)
    ax.set_xticklabels([str(l) for l in valid_layers])
    setup_grid(ax)
    plt.tight_layout()
    save_plot(fig, output_dir, "marginal_contribution.png")


def _plot_marginal_contribution_var(
    agg_by_component: dict,
    output_dir: Path,
) -> None:
    """Variance variant of marginal_contribution: per-pair scatter + mean ± std bands."""
    pre_agg = agg_by_component.get("resid_pre")
    post_agg = agg_by_component.get("resid_post")
    if pre_agg is None or post_agg is None:
        return
    if not getattr(pre_agg, "by_sample", None) or not getattr(post_agg, "by_sample", None):
        return

    # Collect per-pair (layer -> recovery) for resid_pre and resid_post
    def _per_pair_layer_recovery(agg) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        for sample in agg.by_sample.values():
            steps = list(sample.layer_results.keys())
            if not steps:
                continue
            step = min(int(s) for s in steps)
            sw = sample.get_layer_results_for_step(step)
            if not sw:
                continue
            for layer in sw.keys():
                tr = sw.get(layer)
                if tr is None or tr.recovery is None:
                    continue
                out.setdefault(int(layer), []).append(float(tr.recovery))
        return out

    def _per_pair_layer_disruption(agg) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        for sample in agg.by_sample.values():
            steps = list(sample.layer_results.keys())
            if not steps:
                continue
            step = min(int(s) for s in steps)
            sw = sample.get_layer_results_for_step(step)
            if not sw:
                continue
            for layer in sw.keys():
                tr = sw.get(layer)
                if tr is None or tr.disruption is None:
                    continue
                out.setdefault(int(layer), []).append(float(tr.disruption))
        return out

    pre_rec = _per_pair_layer_recovery(pre_agg)
    post_rec = _per_pair_layer_recovery(post_agg)
    pre_dis = _per_pair_layer_disruption(pre_agg)
    post_dis = _per_pair_layer_disruption(post_agg)

    layers = sorted(set(pre_rec.keys()) & set(post_rec.keys()) & set(pre_dis.keys()) & set(post_dis.keys()))
    if not layers:
        return

    # Compute per-pair marginals (post - pre) for each layer
    suff_means, suff_stds, nec_means, nec_stds = [], [], [], []
    for L in layers:
        n = min(len(pre_rec[L]), len(post_rec[L]))
        suff = np.array(post_rec[L][:n]) - np.array(pre_rec[L][:n])
        n2 = min(len(pre_dis[L]), len(post_dis[L]))
        nec = np.array(post_dis[L][:n2]) - np.array(pre_dis[L][:n2])
        suff_means.append(float(suff.mean()) if len(suff) else 0.0)
        suff_stds.append(float(suff.std()) if len(suff) else 0.0)
        nec_means.append(float(nec.mean()) if len(nec) else 0.0)
        nec_stds.append(float(nec.std()) if len(nec) else 0.0)

    suff_means = np.array(suff_means)
    suff_stds = np.array(suff_stds)
    nec_means = np.array(nec_means)
    nec_stds = np.array(nec_stds)

    fig, ax = create_figure(figsize=(12, 6))
    ax.plot(layers, suff_means, "o-", color="#2ca02c", linewidth=2, markersize=6, label="Δ Sufficiency", alpha=0.9)
    ax.fill_between(layers, suff_means - suff_stds, suff_means + suff_stds, color="#2ca02c", alpha=0.2)
    ax.plot(layers, nec_means, "s-", color="#d62728", linewidth=2, markersize=6, label="Δ Necessity", alpha=0.9)
    ax.fill_between(layers, nec_means - nec_stds, nec_means + nec_stds, color="#d62728", alpha=0.2)

    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Marginal: resid_post[L] - resid_pre[L]", fontsize=12, fontweight="bold")
    ax.set_title("Marginal Contribution per Layer (mean ± std across pairs)", fontsize=14, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(loc="upper left", fontsize=9)
    setup_grid(ax)
    plt.tight_layout()
    save_plot(fig, output_dir, "marginal_contribution_var.png")


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


def _plot_layer_interaction(
    layer_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    agg_by_component: dict | None = None,
) -> None:
    """Layer × component interaction with ±std bands when multi-pair data available."""
    all_layers = set()
    for comp, data in layer_data.items():
        if data:
            all_layers.update(data.keys())
    if not all_layers:
        return
    layers = sorted(all_layers)

    # Pre-compute per-pair std for each component
    comp_std: dict[str, dict[str, dict[int, float]]] = {}  # comp -> metric -> layer -> std
    if agg_by_component:
        for comp in ["attn_out", "mlp_out", "resid_post"]:
            agg = agg_by_component.get(comp)
            if agg is None:
                continue
            comp_std[comp] = {
                "recovery": {l: float(np.std(vs)) for l, vs in _per_pair_layer_values(agg, "recovery").items() if len(vs) > 1},
                "disruption": {l: float(np.std(vs)) for l, vs in _per_pair_layer_values(agg, "disruption").items() if len(vs) > 1},
            }

    all_values = []
    data_by_mode = {}
    plot_components = ["attn_out", "mlp_out", "resid_post"]
    for mode in ["denoising", "noising"]:
        metric = "recovery" if mode == "denoising" else "disruption"
        mode_data = {}
        for comp in plot_components:
            data = layer_data.get(comp)
            if not data:
                continue
            values, valid_layers, stds = [], [], []
            for layer in layers:
                if data.get(layer) is not None:
                    val = getattr(data[layer], metric, None)
                    if val is not None:
                        values.append(val)
                        valid_layers.append(layer)
                        all_values.append(val)
                        stds.append(comp_std.get(comp, {}).get(metric, {}).get(int(layer), 0.0))
            if valid_layers:
                mode_data[comp] = (valid_layers, values, stds)
        data_by_mode[mode] = mode_data

    if not all_values:
        return
    y_min = min(all_values) - 0.05
    y_max = max(all_values) + 0.05

    fig, axes = create_figure(1, 2, figsize=(16, 6))
    for ax_idx, mode in enumerate(["denoising", "noising"]):
        ax = axes[ax_idx]
        mode_data = data_by_mode[mode]
        for comp in plot_components:
            if comp in mode_data:
                valid_layers, values, stds = mode_data[comp]
                color = COMPONENT_COLORS[comp]
                ax.plot(valid_layers, values, "o-", color=color,
                        linewidth=1.5, markersize=4, label=comp, alpha=0.8)
                if any(s > 0 for s in stds):
                    lo = [v - s for v, s in zip(values, stds)]
                    hi = [v + s for v, s in zip(values, stds)]
                    ax.fill_between(valid_layers, lo, hi, color=color, alpha=0.15)

        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
        ax.set_ylabel("Recovery" if mode == "denoising" else "Disruption", fontsize=12, fontweight="bold")
        title = "Denoising" if mode == "denoising" else "Noising"
        ax.set_title(f"Layer × Component Interaction - {title}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.set_xticks(layers)
        ax.set_xticklabels([str(l) for l in layers])
        setup_grid(ax)

    plt.tight_layout()
    save_plot(fig, output_dir, "layer_component_interaction.png")


def _per_pair_position_values(
    agg,
    metric: str = "recovery",
) -> dict[int, list[float]]:
    """Like _per_pair_layer_values but for position sweeps."""
    out: dict[int, list[float]] = {}
    if not hasattr(agg, "by_sample") or not agg.by_sample:
        return out
    for sample in agg.by_sample.values():
        steps = list(sample.position_results.keys())
        if not steps:
            continue
        step = min(int(s) for s in steps)
        sw = sample.get_position_results_for_step(step)
        if not sw:
            continue
        for pos in sw.keys():
            tr = sw.get(pos)
            if tr is None:
                continue
            val = getattr(tr, metric, None)
            if val is not None:
                out.setdefault(int(pos), []).append(float(val))
    return out


def _plot_position_interaction(
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
    position_mapping=None,
    agg_by_component: dict | None = None,
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

    plot_components = ["attn_out", "mlp_out", "resid_post"]

    # Build groups: format_pos label -> ordered list of abs positions
    def _group_label(pos: int) -> str:
        if position_mapping is None:
            return f"P{pos}"
        info = position_mapping.get_position(pos)
        if not info or not info.format_pos:
            return f"P{pos}"
        fp = info.format_pos
        if fp.startswith("left_"):
            fp = "L_" + fp[5:]
        elif fp.startswith("right_"):
            fp = "R_" + fp[6:]
        return fp

    group_order: list[str] = []
    group_positions: dict[str, list[int]] = {}
    for pos in positions:
        lbl = _group_label(pos)
        if lbl not in group_positions:
            group_positions[lbl] = []
            group_order.append(lbl)
        group_positions[lbl].append(pos)

    # Pre-compute per-pair position std for each component
    pos_pair_std: dict[str, dict[str, dict[int, float]]] = {}
    if agg_by_component:
        for comp in plot_components:
            agg = agg_by_component.get(comp)
            if agg is None:
                continue
            pos_pair_std[comp] = {
                "recovery": {p: float(np.std(vs)) for p, vs in _per_pair_position_values(agg, "recovery").items() if len(vs) > 1},
                "disruption": {p: float(np.std(vs)) for p, vs in _per_pair_position_values(agg, "disruption").items() if len(vs) > 1},
            }

    for mode in ["denoising", "noising"]:
        metric = "recovery" if mode == "denoising" else "disruption"
        mode_data = {}
        for comp in plot_components:
            data = pos_data.get(comp)
            if not data:
                continue
            values, valid_groups, stds = [], [], []
            for gi, lbl in enumerate(group_order):
                vs = []
                pos_stds = []
                for p in group_positions[lbl]:
                    if data.get(p) is not None:
                        v = getattr(data[p], metric, None)
                        if v is not None:
                            vs.append(v)
                            pos_stds.append(pos_pair_std.get(comp, {}).get(metric, {}).get(int(p), 0.0))
                if vs:
                    mean_v = float(np.mean(vs))
                    values.append(mean_v)
                    valid_groups.append(gi)
                    all_values.append(mean_v)
                    stds.append(float(np.mean(pos_stds)) if pos_stds else 0.0)
            if valid_groups:
                mode_data[comp] = (valid_groups, values, stds)
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

        # Hub shading removed per user request
        # for start, end in hub_regions:
        #     ax.axvspan(start, end, alpha=0.15, color="yellow", zorder=0)

        mode_data = data_by_mode[mode]
        draw_order = [
            ("resid_post", {"marker": "o", "linestyle": "-", "linewidth": 2.5, "ms": 6, "alpha": 0.7}),
            ("mlp_out",    {"marker": "s", "linestyle": "-", "linewidth": 1.8, "ms": 5, "alpha": 0.85}),
            ("attn_out",   {"marker": "x", "linestyle": "--", "linewidth": 1.8, "ms": 7, "alpha": 1.0}),
        ]
        for comp, style in draw_order:
            if comp in mode_data:
                valid_groups, values, stds = mode_data[comp]
                color = COMPONENT_COLORS[comp]
                ax.plot(
                    valid_groups, values,
                    color=color,
                    marker=style["marker"], linestyle=style["linestyle"],
                    linewidth=style["linewidth"], markersize=style["ms"], alpha=style["alpha"],
                    label=comp,
                )
                if any(s > 0 for s in stds):
                    lo = [v - s for v, s in zip(values, stds)]
                    hi = [v + s for v, s in zip(values, stds)]
                    ax.fill_between(valid_groups, lo, hi, color=color, alpha=0.12)

        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("format_pos", fontsize=12, fontweight="bold")
        ax.set_ylabel("Recovery" if mode == "denoising" else "Disruption", fontsize=12, fontweight="bold")
        title = "Denoising" if mode == "denoising" else "Noising"
        ax.set_title(f"Position × Component Interaction - {title}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)

        # X-tick per format_pos group
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(group_order, rotation=60, ha='right', fontsize=7)

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
