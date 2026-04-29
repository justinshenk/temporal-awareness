"""Attention visualization entry points.

Per-pair plots live in ``attn_pair_viz`` and consume the label-aligned
``DstGroupAttention`` storage. This module provides:

* ``visualize_attn_pair`` — orchestrates per-pair plots and add-ons
  (head attribution, head redundancy, layer-position).
* ``visualize_attn_analysis`` / ``visualize_all_attn_slices`` — aggregated
  per-head metric plots across pairs and slices.
* ``visualize_head_attribution`` / ``visualize_head_redundancy`` /
  ``_plot_layer_position_heatmaps`` — add-on plots for the analysis side
  artifacts produced when those config flags are enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from ....common.logging import log
from ....viz.plot_helpers import add_pair_label, finalize_plot, save_figure
from . import AttnAggregatedResults, AttnPairResult
from .attn_analysis_results import DstGroupAttention
from .attn_head_attribution import HeadAttributionResults, HeadSweepResults
from .attn_pair_viz import visualize_attn_pair as _visualize_attn_pair_impl
from ..fine.fine_results import LayerPositionResult

if TYPE_CHECKING:
    from ...common.sample_position_mapping import SamplePositionMapping


# ─── Style ──────────────────────────────────────────────────────────────────
DPI = 150
LAYER_COLORS = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FF9800', '#795548']
BAR_ALPHA = 0.8
GRID_ALPHA = 0.3


# ─── Per-pair entry point ───────────────────────────────────────────────────


def visualize_attn_pair(
    result: AttnPairResult,
    output_dir: Path,
    pair_idx: int | None = None,
) -> None:
    """Generate per-pair attention visualizations and add-ons."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _visualize_attn_pair_impl(result, output_dir, pair_idx=pair_idx)

    if result.head_attribution is not None:
        visualize_head_attribution(result, output_dir)
    if result.head_redundancy is not None:
        visualize_head_redundancy(result.head_redundancy, output_dir)
    if result.layer_position is not None:
        _plot_layer_position_heatmaps(result.layer_position, output_dir)
    log(f"[attn_viz] Generated per-pair plots in {output_dir}")


# ─── Aggregated per-head metric plots ───────────────────────────────────────


def _aggregate_dst_groups_from_disk(
    pairs_dir: Path,
    pair_indices: list[int],
) -> "dict[str, DstGroupAttention]":
    """Stream-load per-pair attn_results.json and average dst_group_attention.

    Returns a dict of aggregated DstGroupAttention keyed by dst_label.
    Pairs that have no JSON or no dst_group_attention are silently skipped.
    """
    import json
    from .attn_analysis_results import DstGroupAttention

    # Pass 1: collect canonical_labels and dst groups from all pairs
    all_dst_labels: set[str] = set()
    all_canonical: list[str] = []
    seen_canonical: set[str] = set()

    pair_data: list[dict] = []  # list of raw dst_group_attention dicts
    for idx in pair_indices:
        path = pairs_dir / f"pair_{idx}" / "attn" / "attn_results.json"
        if not path.exists():
            pair_data.append({})
            continue
        raw = json.loads(path.read_text())
        dga_raw = raw.get("dst_group_attention", {})
        pair_data.append(dga_raw)
        for dst_label, grp in dga_raw.items():
            all_dst_labels.add(dst_label)
            for lbl in grp.get("canonical_labels", []):
                if lbl not in seen_canonical:
                    all_canonical.append(lbl)
                    seen_canonical.add(lbl)

    if not all_dst_labels:
        return {}

    label_idx = {lbl: i for i, lbl in enumerate(all_canonical)}
    n_labels = len(all_canonical)

    # Pass 2: accumulate per (dst, layer, head, label) sums + counts
    result: dict[str, DstGroupAttention] = {}
    for dst_label in sorted(all_dst_labels):
        clean_sum: dict[int, np.ndarray] = {}
        corr_sum: dict[int, np.ndarray] = {}
        clean_cnt: dict[int, np.ndarray] = {}
        corr_cnt: dict[int, np.ndarray] = {}
        dst_pos_indices: set[int] = set()

        for dga_raw in pair_data:
            grp = dga_raw.get(dst_label)
            if not grp:
                continue
            pair_labels = grp.get("canonical_labels", [])
            pair_label_map = {lbl: i for i, lbl in enumerate(pair_labels)}
            for di in grp.get("dst_position_indices", []):
                if di < len(pair_labels):
                    target = pair_labels[di]
                    if target in label_idx:
                        dst_pos_indices.add(label_idx[target])

            for side_key, sum_d, cnt_d in [
                ("clean", clean_sum, clean_cnt),
                ("corrupted", corr_sum, corr_cnt),
            ]:
                side = grp.get(side_key, {})
                for layer_str, mat in side.items():
                    layer = int(layer_str)
                    arr = np.array(mat, dtype=np.float64)
                    n_heads = arr.shape[0]
                    if layer not in sum_d:
                        sum_d[layer] = np.zeros((n_heads, n_labels), dtype=np.float64)
                        cnt_d[layer] = np.zeros((n_heads, n_labels), dtype=np.float64)
                    # Map pair labels to global labels
                    for pi, plbl in enumerate(pair_labels):
                        gi = label_idx.get(plbl)
                        if gi is not None and pi < arr.shape[1]:
                            sum_d[layer][:n_heads, gi] += arr[:n_heads, pi]
                            cnt_d[layer][:n_heads, gi] += 1.0

        clean_mean: dict[int, list[list[float]]] = {}
        corr_mean: dict[int, list[list[float]]] = {}
        for layer in clean_sum:
            safe = np.where(clean_cnt[layer] > 0, clean_cnt[layer], 1.0)
            clean_mean[layer] = (clean_sum[layer] / safe).tolist()
        for layer in corr_sum:
            safe = np.where(corr_cnt[layer] > 0, corr_cnt[layer], 1.0)
            corr_mean[layer] = (corr_sum[layer] / safe).tolist()

        result[dst_label] = DstGroupAttention(
            dst_label=dst_label,
            canonical_labels=all_canonical,
            dst_position_indices=sorted(dst_pos_indices),
            clean=clean_mean,
            corrupted=corr_mean,
        )

    return result


def visualize_attn_analysis(
    agg: AttnAggregatedResults,
    output_dir: Path,
    pairs_dir: Path | None = None,
) -> None:
    """Aggregated head-metric plots + aggregated per-dst attention plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not agg.pair_results:
        log("[attn_viz] No pair results to visualize")
        return

    has_head_data = any(
        lr.n_heads > 0 for pr in agg.pair_results for lr in pr.layer_results
    )
    if not has_head_data:
        return

    _plot_source_attention_by_layer(agg, output_dir / "attn_source_by_layer.png")
    _plot_top_attending_heads(agg, output_dir / "attn_top_heads.png")
    _plot_dynamic_heads(agg, output_dir / "attn_dynamic_heads.png")
    _plot_attention_to_source_summary(agg, output_dir / "attn_to_source_summary.png")
    _plot_head_importance_vs_shift(agg, output_dir / "head_importance_vs_shift.png")
    _plot_cross_layer_consistency(agg, output_dir / "cross_layer_consistency.png")

    # Aggregated per-dst attention plots (stream-loaded from per-pair JSONs)
    if pairs_dir is not None:
        pair_indices = [pr.pair_idx for pr in agg.pair_results]
        agg_dst = _aggregate_dst_groups_from_disk(pairs_dir, pair_indices)
        if agg_dst:
            _visualize_attn_pair_impl(
                AttnPairResult(pair_idx=-1, dst_group_attention=agg_dst),
                output_dir,
                pair_idx=None,
            )
            log(f"[attn_viz] Generated aggregated per-dst plots in {output_dir}")

    log(f"[attn_viz] Generated aggregated plots in {output_dir}")


def visualize_all_attn_slices(
    agg: AttnAggregatedResults,
    output_dir: Path,
    pref_pairs: list | None = None,
    pairs_dir: Path | None = None,
) -> None:
    """Run aggregated viz for every analysis slice that matches at least one pair."""
    from ..coarse.viz.aggregated.analysis_slices import ANALYSIS_SLICES
    from ...viz.slice_config import CORE_SLICES, GENERATE_ALL_SLICES

    output_dir = Path(output_dir)

    n_samples = agg.n_pairs
    if not GENERATE_ALL_SLICES or n_samples <= 2:
        slices_to_generate = [s for s in ANALYSIS_SLICES if s.name in CORE_SLICES]
    else:
        slices_to_generate = ANALYSIS_SLICES

    for analysis_slice in slices_to_generate:
        slice_name = analysis_slice.name
        slice_dir = output_dir / slice_name

        if slice_name == "all" or pref_pairs is None:
            filtered_agg = agg
        else:
            indices = [
                i for i, pref in enumerate(pref_pairs)
                if analysis_slice.req.passes(pref)
            ]
            if not indices:
                continue
            filtered_agg = agg.filter_by_indices(indices)

        visualize_attn_analysis(filtered_agg, slice_dir, pairs_dir=pairs_dir)

    log(f"[attn_viz] All attention slices saved to {output_dir}")


def _plot_source_attention_by_layer(agg: AttnAggregatedResults, output_path: Path) -> None:
    layers = agg.layers_analyzed
    if not layers:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_attn = []
    std_attn = []
    for layer in layers:
        attns = [lr.mean_attn_to_source for pr in agg.pair_results
                 if (lr := pr.get_layer_result(layer)) is not None]
        if attns:
            mean_attn.append(float(np.mean(attns)))
            std_attn.append(float(np.std(attns)) if len(attns) > 1 else 0.0)
        else:
            mean_attn.append(0.0)
            std_attn.append(0.0)

    x = np.arange(len(layers))
    colors = LAYER_COLORS[:len(layers)] if len(layers) <= len(LAYER_COLORS) else \
        plt.cm.viridis(np.linspace(0, 1, len(layers)))
    ax.bar(x, mean_attn, color=colors, alpha=BAR_ALPHA,
           yerr=std_attn if agg.n_pairs > 1 else None, capsize=3, ecolor='gray')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Attention to Source Positions', fontsize=12)
    ax.set_title(
        f'Attention to Source Positions ({agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""})',
        fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.grid(axis='y', alpha=GRID_ALPHA)
    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_top_attending_heads(agg: AttnAggregatedResults, output_path: Path) -> None:
    layers = agg.layers_analyzed
    if not layers:
        return
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 5), squeeze=False)
    for i, layer in enumerate(layers):
        ax = axes[0, i]
        consistent = agg.get_consistent_source_heads(layer, min_attn=0.02, min_pairs=1)
        if consistent:
            heads = [h[0] for h in consistent[:10]]
            attns = [h[1] for h in consistent[:10]]
            y = np.arange(len(heads))
            ax.barh(y, attns, color=LAYER_COLORS[i % len(LAYER_COLORS)], alpha=BAR_ALPHA)
            ax.set_yticks(y)
            ax.set_yticklabels([f'H{h}' for h in heads])
            ax.set_xlabel('Mean Attention to Source')
            ax.set_title(f'Layer {layer}')
            ax.grid(axis='x', alpha=GRID_ALPHA)
        else:
            ax.text(0.5, 0.5, 'No heads above\nthreshold',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Layer {layer}')
    fig.suptitle('Top Source-Attending Heads by Layer', fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_dynamic_heads(agg: AttnAggregatedResults, output_path: Path) -> None:
    layers = agg.layers_analyzed
    if not layers:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    layer_labels = [f'L{l}' for l in layers]
    dynamic_counts = [len(agg.get_dynamic_heads_across_pairs(l, min_pairs=1)) for l in layers]
    x = np.arange(len(layers))
    colors = ['#FF6B6B' if c > 0 else '#CCCCCC' for c in dynamic_counts]
    bars = ax.bar(x, dynamic_counts, color=colors, alpha=BAR_ALPHA)
    for bar, count in zip(bars, dynamic_counts):
        if count > 0:
            ax.annotate(f'{count}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Number of Dynamic Heads', fontsize=12)
    ax.set_title('Heads with Dynamic Attention Patterns', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.grid(axis='y', alpha=GRID_ALPHA)
    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_attention_to_source_summary(agg: AttnAggregatedResults, output_path: Path) -> None:
    if not agg.layers_analyzed:
        return
    layers = agg.layers_analyzed[:6] if len(agg.layers_analyzed) > 6 else agg.layers_analyzed
    head_data: dict[tuple[int, int], dict[str, list[float]]] = {}
    for pr in agg.pair_results:
        for lr in pr.layer_results:
            if lr.layer not in layers:
                continue
            for hi in lr.head_results:
                key = (lr.layer, hi.head_idx)
                if key not in head_data:
                    head_data[key] = {'clean': [], 'diff': []}
                head_data[key]['clean'].append(hi.attn_to_source)
                head_data[key]['diff'].append(hi.attn_pattern_diff)
    if not head_data:
        return
    sorted_heads = sorted(head_data.items(), key=lambda x: float(np.mean(x[1]['clean'])), reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [f'L{l}.H{h}' for (l, h), _ in sorted_heads]
    clean_means = [float(np.mean(d['clean'])) for _, d in sorted_heads]
    clean_stds = [float(np.std(d['clean'])) for _, d in sorted_heads]
    x = np.arange(len(labels))
    colors = [LAYER_COLORS[layers.index(l) % len(LAYER_COLORS)] for (l, _), _ in sorted_heads]
    ax.bar(x, clean_means, 0.6, yerr=clean_stds if agg.n_pairs > 1 else None,
           color=colors, alpha=BAR_ALPHA, capsize=2, ecolor='gray')
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Attention to Source Positions', fontsize=12)
    ax.set_title(f'Top Attention Heads — {agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""}',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=GRID_ALPHA)
    patches = [mpatches.Patch(color=LAYER_COLORS[i % len(LAYER_COLORS)], label=f'L{l}')
               for i, l in enumerate(layers)]
    ax.legend(handles=patches, loc='upper right')
    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_head_importance_vs_shift(agg: AttnAggregatedResults, output_path: Path) -> None:
    layers = agg.layers_analyzed
    if not layers:
        return
    head_data: list[tuple[int, int, float, float]] = []
    for layer in layers:
        head_attns: dict[int, list[float]] = {}
        head_diffs: dict[int, list[float]] = {}
        for pr in agg.pair_results:
            lr = pr.get_layer_result(layer)
            if lr is None:
                continue
            for hi in lr.head_results:
                head_attns.setdefault(hi.head_idx, []).append(hi.attn_to_source)
                head_diffs.setdefault(hi.head_idx, []).append(hi.attn_pattern_diff)
        for h, attns in head_attns.items():
            head_data.append((layer, h, float(np.mean(attns)), float(np.mean(head_diffs[h]))))
    if not head_data:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, layer in enumerate(layers):
        layer_data = [(h, a, d) for (l, h, a, d) in head_data if l == layer]
        if not layer_data:
            continue
        heads, attns, diffs = zip(*layer_data)
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        ax.scatter(attns, diffs, c=color, alpha=0.7, s=50, label=f'L{layer}')
    ax.set_xlabel('Mean Attention to Source Positions', fontsize=12)
    ax.set_ylabel('Mean Attention Pattern Difference', fontsize=12)
    ax.set_title(f'Head Importance vs Attention Shift — {agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""}',
                 fontsize=13)
    ax.legend(loc='upper right')
    ax.grid(alpha=GRID_ALPHA)
    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_cross_layer_consistency(agg: AttnAggregatedResults, output_path: Path) -> None:
    layers = agg.layers_analyzed
    if not layers:
        return
    head_scores: dict[tuple[int, int], list[float]] = {}
    for layer in layers:
        for pr in agg.pair_results:
            lr = pr.get_layer_result(layer)
            if lr is None:
                continue
            for hi in lr.head_results:
                if hi.is_dynamic:
                    head_scores.setdefault((layer, hi.head_idx), []).append(hi.attn_pattern_diff)
    if not head_scores:
        return
    avg = {k: float(np.mean(v)) for k, v in head_scores.items()}
    sorted_heads = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:15]
    if not sorted_heads:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f'L{l}.H{h}' for (l, h), _ in sorted_heads]
    scores = [s for _, s in sorted_heads]
    colors = [LAYER_COLORS[layers.index(l) % len(LAYER_COLORS)] for (l, _), _ in sorted_heads]
    x = np.arange(len(labels))
    ax.bar(x, scores, color=colors, alpha=BAR_ALPHA)
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Attention Pattern Difference', fontsize=12)
    ax.set_title(f'Top Dynamic Heads — {agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""}',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=GRID_ALPHA)
    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


# ─── Head attribution / redundancy ──────────────────────────────────────────


def visualize_head_attribution(result: AttnPairResult, output_dir: Path) -> int:
    """Disruption | Recovery side-by-side heatmap + top heads bar chart."""
    if result.head_attribution is None:
        return 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0
    if result.head_attribution.attribution_matrix is not None:
        _plot_head_attribution_heatmap(
            result.head_attribution,
            output_dir / "head_attribution_heatmap.png",
            pair_idx=result.pair_idx,
            redundancy=result.head_redundancy,
        )
        n_plots += 1
    if result.head_attribution.results:
        _plot_head_attribution_bar(
            result.head_attribution,
            output_dir / "head_attribution_bar.png",
            pair_idx=result.pair_idx,
        )
        n_plots += 1
    return n_plots


def _plot_head_attribution_heatmap(
    head_attr: HeadAttributionResults,
    output_path: Path,
    pair_idx: int | None = None,
    redundancy: HeadSweepResults | None = None,
) -> None:
    if head_attr.attribution_matrix is None:
        return
    recovery = head_attr.attribution_matrix
    layers = head_attr.layers_analyzed
    n_heads = head_attr.n_heads

    disruption = None
    if redundancy is not None and redundancy.results:
        layer_to_idx = {l: i for i, l in enumerate(layers)}
        d = np.zeros_like(recovery)
        seen = False
        for r in redundancy.results:
            if r.layer in layer_to_idx and 0 <= r.head < n_heads:
                d[layer_to_idx[r.layer], r.head] = r.noising_disruption
                seen = True
        disruption = d if seen else None

    panels = []
    if disruption is not None:
        panels.append(("Disruption (noising)", disruption))
    panels.append(("Recovery (denoising)", recovery))

    n_cols = len(panels)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(max(12, n_heads * 0.4) * n_cols / 2, max(4, len(layers) * 0.5)),
        squeeze=False,
    )
    for ci, (name, mat) in enumerate(panels):
        ax = axes[0, ci]
        vmax = max(abs(mat.min()), abs(mat.max()), 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, label="Attribution Score", shrink=0.8)
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels(range(n_heads), fontsize=7)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers], fontsize=9)
        ax.set_xlabel("Head Index")
        if ci == 0:
            ax.set_ylabel("Layer")
        ax.set_title(name, fontsize=11)
    pair_str = f" — pair {pair_idx}" if pair_idx is not None else ""
    fig.suptitle(f"Head Attribution{pair_str}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    add_pair_label(fig, pair_idx)
    save_figure(None, output_path, dpi=DPI)


def _plot_head_attribution_bar(
    head_attr: HeadAttributionResults,
    output_path: Path,
    n_top: int = 20,
    pair_idx: int | None = None,
) -> None:
    top_heads = head_attr.get_top_heads(n_top)
    if not top_heads:
        return
    labels = [h.label for h in top_heads]
    scores = [h.attribution_score for h in top_heads]
    abs_scores = [h.abs_score for h in top_heads]
    total_abs = sum(abs_scores)
    cumulative = np.cumsum(abs_scores) / total_abs * 100 if total_abs > 0 else np.zeros(len(abs_scores))

    fig, ax1 = plt.subplots(figsize=(14, 6))
    colors = ['#4CAF50' if s >= 0 else '#F44336' for s in scores]
    ax1.bar(range(len(labels)), scores, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax1.set_xlabel("Head", fontsize=12)
    ax1.set_ylabel("Attribution Score", fontsize=12)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(range(len(labels)), cumulative, "o-", color="#2196F3",
             linewidth=2, markersize=4, label="Cumulative %")
    ax2.set_ylabel("Cumulative % of Total Effect", fontsize=12, color="#2196F3")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", labelcolor="#2196F3")
    ax2.legend(loc="upper right")
    ax1.set_title(f"Top {len(labels)} Attention Heads by Attribution",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    add_pair_label(fig, pair_idx)
    save_figure(None, output_path, dpi=DPI)


def visualize_head_redundancy(redundancy: HeadSweepResults, output_dir: Path) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not redundancy.results:
        return 0
    _plot_head_redundancy_gap(redundancy, output_dir / "head_redundancy_gap.png")
    return 1


def _plot_head_redundancy_gap(
    redundancy: HeadSweepResults,
    output_path: Path,
    n_top: int = 20,
) -> None:
    sorted_heads = redundancy.get_sorted_by_gap(descending=True)[:n_top]
    if not sorted_heads:
        return
    labels = [h.label for h in sorted_heads]
    gaps = [h.gap for h in sorted_heads]
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#5B9BD5' if g >= 0 else '#C65B5B' for g in gaps]
    ax.bar(range(len(labels)), gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel("Head (sorted by gap magnitude)", fontsize=12)
    ax.set_ylabel("Denoising - Noising Gap", fontsize=12)
    ax.set_title("Head Redundancy: Denoising vs Noising Gap",
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=GRID_ALPHA)
    legend_elements = [
        mpatches.Patch(facecolor='#5B9BD5', alpha=0.8, label='Positive (unique info)'),
        mpatches.Patch(facecolor='#C65B5B', alpha=0.8, label='Negative (redundant)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


# ─── Layer-position fine patching ───────────────────────────────────────────


def _plot_layer_position_heatmaps(
    layer_position: LayerPositionResult,
    output_dir: Path,
) -> None:
    """Side-by-side Denoising | Noising layer × position heatmaps."""
    if layer_position.denoising_grid is None and layer_position.noising_grid is None:
        return

    component = layer_position.component
    layers = layer_position.layers
    positions = layer_position.positions
    n_layers = len(layers)
    n_pos = len(positions)

    panels = []
    if layer_position.denoising_grid is not None:
        panels.append(("Denoising", layer_position.denoising_grid))
    if layer_position.noising_grid is not None:
        panels.append(("Noising", layer_position.noising_grid))

    pos_step = max(1, n_pos // 20)
    tick_idx = list(range(0, n_pos, pos_step))
    tick_labels = [f"P{positions[i]}" for i in tick_idx]

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(max(10, n_pos * 0.15) * len(panels), max(6, n_layers * 0.3)),
        squeeze=False,
    )
    for ci, (name, grid) in enumerate(panels):
        ax = axes[0, ci]
        vmax = max(abs(grid.min()), abs(grid.max()), 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, label=f"{name} effect", shrink=0.8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=90)
        ax.set_xlabel("Position")
        if ci == 0:
            ax.set_ylabel("Layer")
        ax.set_title(name, fontsize=11)
    fig.suptitle(f"Layer × Position Patching: {component}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    finalize_plot(output_dir / f"layer_position_{component}.png")
