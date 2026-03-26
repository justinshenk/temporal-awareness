"""Visualization for attention pattern analysis.

Provides comprehensive visualizations including:
1. Attention probability heatmaps per head (clean vs corrupted)
2. Attention difference heatmaps (clean - corrupted)
3. Attention to source positions summary bar charts
4. Attention flow diagrams for top heads
5. Head importance vs attention shift scatter plots
6. Cross-layer attention consistency (cosine similarity)
7. OV projection analysis (if weight matrices available)
8. QK analysis heatmaps (if weight matrices available)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from ...common.logging import log
from ..experiments.attn_analysis import AttnAggregatedResults, AttnPairResult

if TYPE_CHECKING:
    from ...patching.binary_choice_runner import BinaryChoiceRunner
    from ..common.sample_position_mapping import SamplePositionMapping


# Plot styling
DPI = 150
LAYER_COLORS = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FF9800', '#795548']
BAR_ALPHA = 0.8
GRID_ALPHA = 0.3

# Default position ranges for cropping (configurable via results data)
DEFAULT_SRC_RANGE = (80, 95)  # P80-P95
DEFAULT_DST_RANGE = (135, 150)  # P135-P150
KEY_LAYERS = [19, 21, 24]

# Colors for clean vs corrupted comparisons
CLEAN_COLOR = '#2196F3'  # Blue
CORRUPTED_COLOR = '#FF9800'  # Orange
DIFF_CMAP = 'RdBu_r'  # Red = more attention in clean, Blue = more in corrupted


def visualize_attn_analysis(
    agg: AttnAggregatedResults,
    output_dir: Path,
) -> None:
    """Generate all attention analysis visualizations.

    Args:
        agg: Aggregated attention analysis results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not agg.pair_results:
        log("[attn_viz] No pair results to visualize")
        return

    n_plots = 0

    # Check if we have per-head data
    has_head_data = any(
        lr.n_heads > 0
        for pr in agg.pair_results
        for lr in pr.layer_results
    )

    if has_head_data:
        # Per-head attention analysis plots
        _plot_source_attention_by_layer(agg, output_dir / "attn_source_by_layer.png")
        n_plots += 1

        _plot_top_attending_heads(agg, output_dir / "attn_top_heads.png")
        n_plots += 1

        _plot_dynamic_heads(agg, output_dir / "attn_dynamic_heads.png")
        n_plots += 1

        # Plot 3: Attention to source positions summary bar chart
        _plot_attention_to_source_summary(agg, output_dir / "attn_to_source_summary.png")
        n_plots += 1

        # Plot 5: Head importance vs attention shift scatter
        _plot_head_importance_vs_shift(agg, output_dir / "head_importance_vs_shift.png")
        n_plots += 1

        # Plot 6: Cross-layer attention consistency
        _plot_cross_layer_consistency(agg, output_dir / "cross_layer_consistency.png")
        n_plots += 1
    else:
        # Layer-level summary when per-head data not available
        _plot_layer_summary(agg, output_dir / "attn_layer_summary.png")
        n_plots += 1

    log(f"[attn_viz] Generated {n_plots} plots in {output_dir}")


def _plot_layer_summary(agg: AttnAggregatedResults, output_path: Path) -> None:
    """Plot summary when only layer-level data is available."""
    layers = agg.layers_analyzed
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Count layers with data vs without
    layers_with_data = []
    layers_without_data = []

    for layer in layers:
        has_data = False
        for pr in agg.pair_results:
            lr = pr.get_layer_result(layer)
            if lr and lr.n_heads > 0:
                has_data = True
                break
        if has_data:
            layers_with_data.append(layer)
        else:
            layers_without_data.append(layer)

    # Create info text
    info_text = f"""Attention Analysis Summary

Layers analyzed: {layers}
Pairs analyzed: {agg.n_pairs}

Note: Per-head attention patterns not available.
This typically happens when using HuggingFace backend
which doesn't expose internal attention weights.

For detailed per-head analysis, use TransformerLens backend
with a supported model.

Available data:
- Layer-level attention output differences
- Source position tracking
"""

    ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Attention Analysis - Layer Summary', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_source_attention_by_layer(agg: AttnAggregatedResults, output_path: Path) -> None:
    """Plot mean attention to source positions by layer."""
    layers = agg.layers_analyzed
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute mean attention to source for each layer
    mean_attn = []
    std_attn = []
    for layer in layers:
        attns = []
        for pr in agg.pair_results:
            lr = pr.get_layer_result(layer)
            if lr:
                attns.append(lr.mean_attn_to_source)
        if attns:
            mean_attn.append(np.mean(attns))
            std_attn.append(np.std(attns) if len(attns) > 1 else 0)
        else:
            mean_attn.append(0)
            std_attn.append(0)

    x = np.arange(len(layers))
    colors = LAYER_COLORS[:len(layers)] if len(layers) <= len(LAYER_COLORS) else plt.cm.viridis(np.linspace(0, 1, len(layers)))

    bars = ax.bar(x, mean_attn, color=colors, alpha=BAR_ALPHA,
                  yerr=std_attn if agg.n_pairs > 1 else None,
                  capsize=3, ecolor='gray')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Attention to Source Positions', fontsize=12)
    ax.set_title(f'Attention to Source (Horizon) Positions\n({agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.grid(axis='y', alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_top_attending_heads(agg: AttnAggregatedResults, output_path: Path) -> None:
    """Plot heads that consistently attend to source positions."""
    layers = agg.layers_analyzed
    if not layers:
        return

    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 5), squeeze=False)

    for i, layer in enumerate(layers):
        ax = axes[0, i]

        # Get consistent heads for this layer
        consistent = agg.get_consistent_source_heads(layer, min_attn=0.05, min_pairs=1)

        if consistent:
            heads = [h[0] for h in consistent[:10]]  # Top 10
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

    fig.suptitle('Top Source-Attending Heads by Layer', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_dynamic_heads(agg: AttnAggregatedResults, output_path: Path) -> None:
    """Plot heads with dynamic attention patterns (clean vs corrupted)."""
    layers = agg.layers_analyzed
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect dynamic head counts per layer
    layer_labels = []
    dynamic_counts = []

    for layer in layers:
        dynamic = agg.get_dynamic_heads_across_pairs(layer, min_pairs=1)
        layer_labels.append(f'L{layer}')
        dynamic_counts.append(len(dynamic))

    x = np.arange(len(layers))
    colors = ['#FF6B6B' if c > 0 else '#CCCCCC' for c in dynamic_counts]

    bars = ax.bar(x, dynamic_counts, color=colors, alpha=BAR_ALPHA)

    # Add count labels
    for bar, count in zip(bars, dynamic_counts):
        if count > 0:
            ax.annotate(f'{count}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Number of Dynamic Heads', fontsize=12)
    ax.set_title('Heads with Dynamic Attention Patterns\n(Pattern changes between clean/corrupted)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.grid(axis='y', alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_attention_to_source_summary(agg: AttnAggregatedResults, output_path: Path) -> None:
    """Plot 3: Summary bar chart of attention from dest to source positions.

    X-axis: head index (all heads at key layers)
    Y-axis: total attention weight from destination positions to source positions
    Two bars per head: clean vs corrupted (when available)
    """
    layers = agg.layers_analyzed or KEY_LAYERS
    layers = [l for l in layers if l in KEY_LAYERS or len(layers) <= 3]

    if not layers:
        layers = agg.layers_analyzed[:3] if agg.layers_analyzed else []
    if not layers:
        return

    # Collect per-head attention data across pairs
    head_data = {}  # (layer, head) -> {'clean': [], 'diff': []}

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

    # Sort heads by mean attention
    sorted_heads = sorted(
        head_data.items(),
        key=lambda x: np.mean(x[1]['clean']),
        reverse=True
    )[:20]  # Top 20 heads

    fig, ax = plt.subplots(figsize=(14, 6))

    labels = [f'L{l}.H{h}' for (l, h), _ in sorted_heads]
    clean_means = [np.mean(d['clean']) for _, d in sorted_heads]
    clean_stds = [np.std(d['clean']) for _, d in sorted_heads]

    x = np.arange(len(labels))
    width = 0.6

    # Color by layer
    colors = []
    for (l, _), _ in sorted_heads:
        idx = layers.index(l) if l in layers else 0
        colors.append(LAYER_COLORS[idx % len(LAYER_COLORS)])

    bars = ax.bar(x, clean_means, width, yerr=clean_stds if agg.n_pairs > 1 else None,
                  color=colors, alpha=BAR_ALPHA, capsize=2, ecolor='gray')

    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Attention to Source Positions', fontsize=12)
    ax.set_title(f'Top Attention Heads: Dest -> Source\nLayers {layers} ({agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=GRID_ALPHA)

    # Add legend for layers
    patches = [mpatches.Patch(color=LAYER_COLORS[i % len(LAYER_COLORS)], label=f'L{l}')
               for i, l in enumerate(layers)]
    ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_head_importance_vs_shift(agg: AttnAggregatedResults, output_path: Path) -> None:
    """Plot 5: Scatter of head-level metrics (attention to source vs pattern change).

    X-axis: attention to source (proxy for importance)
    Y-axis: attention pattern difference (L2 norm of clean - corrupted)
    Each point = one head, colored by layer
    """
    layers = agg.layers_analyzed
    if not layers:
        return

    # Collect per-head data
    head_data = []  # [(layer, head, mean_attn, mean_diff)]

    for layer in layers:
        head_attns: dict[int, list[float]] = {}
        head_diffs: dict[int, list[float]] = {}

        for pr in agg.pair_results:
            lr = pr.get_layer_result(layer)
            if not lr:
                continue
            for hi in lr.head_results:
                if hi.head_idx not in head_attns:
                    head_attns[hi.head_idx] = []
                    head_diffs[hi.head_idx] = []
                head_attns[hi.head_idx].append(hi.attn_to_source)
                head_diffs[hi.head_idx].append(hi.attn_pattern_diff)

        for head_idx in head_attns:
            mean_attn = np.mean(head_attns[head_idx])
            mean_diff = np.mean(head_diffs[head_idx])
            head_data.append((layer, head_idx, mean_attn, mean_diff))

    if not head_data:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each layer with different color
    for i, layer in enumerate(layers):
        layer_data = [(h, a, d) for (l, h, a, d) in head_data if l == layer]
        if not layer_data:
            continue

        heads, attns, diffs = zip(*layer_data)
        color = LAYER_COLORS[i % len(LAYER_COLORS)]

        ax.scatter(attns, diffs, c=color, alpha=0.7, s=50, label=f'L{layer}')

        # Label top heads
        top_indices = np.argsort(diffs)[-3:]  # Top 3 by diff
        for idx in top_indices:
            if diffs[idx] > 0.05:  # Only label significant ones
                ax.annotate(f'H{heads[idx]}', (attns[idx], diffs[idx]),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=8, alpha=0.8)

    ax.set_xlabel('Mean Attention to Source Positions', fontsize=12)
    ax.set_ylabel('Mean Attention Pattern Difference\n(Clean vs Corrupted)', fontsize=12)
    ax.set_title(f'Head Importance vs Attention Shift\n({agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""})', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_cross_layer_consistency(agg: AttnAggregatedResults, output_path: Path) -> None:
    """Plot 6: Cross-layer attention consistency via pattern difference.

    For each important head, show the pattern difference (as proxy for consistency).
    Grouped bar chart: one bar per head.
    """
    layers = agg.layers_analyzed
    if not layers:
        return

    # Get top dynamic heads across all layers
    all_heads = []
    for layer in layers:
        for pr in agg.pair_results:
            lr = pr.get_layer_result(layer)
            if not lr:
                continue
            for hi in lr.head_results:
                if hi.is_dynamic:
                    all_heads.append((layer, hi.head_idx, hi.attn_pattern_diff))

    if not all_heads:
        # Fallback: use top attending heads
        for layer in layers:
            consistent = agg.get_consistent_source_heads(layer, min_attn=0.05, min_pairs=1)
            for head_idx, mean_attn, n_pairs in consistent[:3]:
                all_heads.append((layer, head_idx, mean_attn))

    if not all_heads:
        return

    # Group by (layer, head) and average
    head_scores = {}
    for layer, head, score in all_heads:
        key = (layer, head)
        if key not in head_scores:
            head_scores[key] = []
        head_scores[key].append(score)

    avg_scores = {k: np.mean(v) for k, v in head_scores.items()}
    sorted_heads = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:15]

    if not sorted_heads:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f'L{l}.H{h}' for (l, h), _ in sorted_heads]
    scores = [s for _, s in sorted_heads]

    # Color by layer
    colors = []
    for (l, _), _ in sorted_heads:
        idx = layers.index(l) if l in layers else 0
        colors.append(LAYER_COLORS[idx % len(LAYER_COLORS)])

    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color=colors, alpha=BAR_ALPHA)

    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Attention Pattern Difference\n(Clean vs Corrupted)', fontsize=12)
    ax.set_title(f'Attention Consistency: Top Dynamic Heads\n({agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


# =============================================================================
# Per-pair visualizations
# =============================================================================


def visualize_attn_pair(
    result: AttnPairResult,
    output_dir: Path,
    runner: "BinaryChoiceRunner | None" = None,
    mapping: "SamplePositionMapping | None" = None,
) -> None:
    """Generate attention analysis visualizations for a single pair.

    Args:
        result: Attention analysis results for one pair
        output_dir: Directory to save plots
        runner: Optional model runner for QK analysis (plot 8)
        mapping: Optional SamplePositionMapping for token identity annotation

    Note: OV projection analysis (plot 7) has been moved to diffmeans_viz.py
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result.layer_results:
        return

    n_plots = 0

    # Check if we have per-head data
    has_head_data = any(lr.n_heads > 0 for lr in result.layer_results)

    if has_head_data:
        # Per-head attention plots
        _plot_pair_head_attention(result, output_dir / "attn_heads.png")
        n_plots += 1

        # Attention pattern heatmap if available
        if result.attention_patterns:
            # Plot 1: Attention probability heatmaps per head (clean only)
            _plot_pair_attention_heatmaps(result, output_dir / "attn_heatmaps.png")
            n_plots += 1

            # Plot 1b: Side-by-side clean vs corrupted heatmaps (if corrupted available)
            if result.corrupted_attention_patterns:
                _plot_pair_attention_sidebyside(result, output_dir / "attn_sidebyside.png")
                n_plots += 1

                # Plot 2: Attention difference heatmaps
                _plot_pair_attention_diff(result, output_dir / "attn_diff.png")
                n_plots += 1

            # Plot 4: Attention flow diagram (for top heads)
            _plot_pair_attention_flow(result, output_dir / "attn_flow.png", mapping=mapping)
            n_plots += 1

        # Top attended positions visualization
        _plot_pair_top_attended(result, output_dir / "attn_top_attended.png")
        n_plots += 1

        # Source attention bars (attention to time_horizon positions)
        if result.source_positions and result.corrupted_attention_patterns:
            _plot_source_attention_bars(result, output_dir / "attn_source_bars.png")
            n_plots += 1

        # Cosine consistency visualization
        if any(hi.attn_pattern_cosine != 0.0 for lr in result.layer_results for hi in lr.head_results):
            _plot_cosine_consistency(result, output_dir / "attn_consistency.png")
            n_plots += 1
    else:
        # Layer-level summary for this pair
        _plot_pair_layer_summary(result, output_dir / "attn_pair_summary.png")
        n_plots += 1

    # Plot 7: OV projection analysis (requires TransformerLens backend)
    if runner is not None:
        try:
            n_plots += visualize_attn_ov_projection(runner, result, output_dir)
        except Exception as e:
            log(f"[attn_viz] OV projection skipped: {e}")

    # Plot 8: QK analysis (requires TransformerLens backend)
    if runner is not None:
        try:
            n_plots += visualize_qk_analysis(runner, result, output_dir)
        except Exception as e:
            log(f"[attn_viz] QK analysis skipped: {e}")

    if n_plots > 0:
        log(f"[attn_viz] Generated {n_plots} pair plots in {output_dir}")


def _plot_pair_layer_summary(result: AttnPairResult, output_path: Path) -> None:
    """Plot layer-level summary for a single pair."""
    layers = [lr.layer for lr in result.layer_results]
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create info text with source positions
    source_pos_str = f"{result.source_positions[:5]}..." if len(result.source_positions) > 5 else str(result.source_positions)
    info_text = f"""Attention Analysis - Pair {result.pair_idx}

Destination position: {result.dest_position}
Source positions: {source_pos_str}
Layers analyzed: {layers}

Note: Per-head attention patterns not available.
Using HuggingFace backend which doesn't expose
internal attention weights.

For detailed per-head analysis, use TransformerLens
backend with a supported model.
"""

    ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Attention Analysis - Pair {result.pair_idx}', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_pair_head_attention(result: AttnPairResult, output_path: Path) -> None:
    """Plot per-head attention to source for a single pair."""
    layers = [lr.layer for lr in result.layer_results]
    if not layers:
        return

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 5), squeeze=False)

    for i, lr in enumerate(result.layer_results):
        ax = axes[0, i]

        if lr.head_results:
            # Sort heads by attention to source
            sorted_heads = sorted(lr.head_results, key=lambda h: h.attn_to_source, reverse=True)
            top_heads = sorted_heads[:10]  # Top 10

            heads = [h.head_idx for h in top_heads]
            attns = [h.attn_to_source for h in top_heads]

            y = np.arange(len(heads))
            color = LAYER_COLORS[i % len(LAYER_COLORS)]
            ax.barh(y, attns, color=color, alpha=BAR_ALPHA)
            ax.set_yticks(y)
            ax.set_yticklabels([f'H{h}' for h in heads])
            ax.set_xlabel('Attention to Source')
            ax.set_title(f'Layer {lr.layer}')
            ax.grid(axis='x', alpha=GRID_ALPHA)
        else:
            ax.text(0.5, 0.5, 'No head data',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Layer {lr.layer}')

    fig.suptitle(f'Pair {result.pair_idx}: Head Attention to Source Positions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_pair_attention_heatmaps(result: AttnPairResult, output_path: Path) -> None:
    """Plot 1: Attention probability heatmaps per head (clean).

    One heatmap per head at key layers.
    X-axis: source position (key)
    Y-axis: head index
    Color: attention weight
    """
    if not result.attention_patterns:
        return

    layers = sorted(result.attention_patterns.keys())
    n_layers = len(layers)
    if n_layers == 0:
        return

    # Determine crop ranges based on source/dest positions
    src_start, src_end = DEFAULT_SRC_RANGE
    if result.source_positions:
        src_start = max(0, min(result.source_positions) - 5)
        src_end = min(200, max(result.source_positions) + 5)

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)

    for i, layer in enumerate(layers):
        ax = axes[0, i]
        patterns = result.attention_patterns[layer]

        if not patterns:
            ax.text(0.5, 0.5, 'No pattern data',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Layer {layer}')
            continue

        # patterns is [n_heads, seq_len] - attention from dest to all positions
        pattern_array = np.array(patterns)
        n_heads, seq_len = pattern_array.shape

        # Crop to interesting range
        crop_start = max(0, src_start)
        crop_end = min(seq_len, src_end)
        if crop_end <= crop_start:
            crop_start, crop_end = 0, seq_len

        cropped = pattern_array[:, crop_start:crop_end]

        # Show heatmap
        im = ax.imshow(cropped, aspect='auto', cmap='viridis',
                       extent=[crop_start, crop_end, n_heads - 0.5, -0.5])
        ax.set_xlabel('Position (Key)')
        ax.set_ylabel('Head')
        ax.set_title(f'Layer {layer}')

        # Mark source positions
        for sp in result.source_positions:
            if crop_start <= sp < crop_end:
                ax.axvline(x=sp, color='red', alpha=0.5, linewidth=1, linestyle='--')

        plt.colorbar(im, ax=ax, label='Attention', shrink=0.8)

    fig.suptitle(f'Pair {result.pair_idx}: Attention Patterns (red lines = source positions)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_pair_attention_flow(
    result: AttnPairResult,
    output_path: Path,
    mapping: "SamplePositionMapping | None" = None,
) -> None:
    """Plot 4: Attention flow diagram for top heads.

    Lines from query positions to key positions, thickness = attention weight.
    For the top N heads by attention to source.
    Optionally annotates key positions with token identities.
    """
    if not result.attention_patterns:
        return

    # Find top 5 heads by attention to source
    top_heads = []
    for lr in result.layer_results:
        for hi in lr.head_results:
            top_heads.append((lr.layer, hi.head_idx, hi.attn_to_source, hi.top_attended_positions, hi.top_attended_weights))

    top_heads = sorted(top_heads, key=lambda x: x[2], reverse=True)[:5]

    if not top_heads:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create a flow diagram showing attention from dest to source
    dest_pos = result.dest_position
    y_positions = {
        'dest': 0.8,
        'src': 0.2
    }

    # Plot destination position
    ax.scatter([dest_pos], [y_positions['dest']], s=200, c='#E91E63',
               marker='s', zorder=5, label='Dest Position')

    # Plot source positions
    src_colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(result.source_positions)))
    ax.scatter(result.source_positions, [y_positions['src']] * len(result.source_positions),
               s=100, c=src_colors, marker='o', zorder=5, label='Source Positions')

    # Annotate key positions with token identities if mapping available
    if mapping is not None:
        _annotate_key_tokens(ax, result, mapping, y_positions)

    # Draw attention lines for top heads and collect legend handles
    colors = LAYER_COLORS[:len(top_heads)]
    legend_handles = []
    for idx, (layer, head, attn_to_src, top_pos, top_weights) in enumerate(top_heads):
        color = colors[idx % len(colors)]
        label = f'L{layer}.H{head} ({attn_to_src:.3f})'

        # Draw lines to top attended positions
        if top_pos and top_weights:
            for pos, weight in zip(top_pos[:3], top_weights[:3]):  # Top 3
                if pos in result.source_positions or weight > 0.05:
                    ax.annotate('',
                                xy=(pos, y_positions['src']),
                                xytext=(dest_pos, y_positions['dest']),
                                arrowprops=dict(arrowstyle='->', color=color,
                                               lw=weight * 10 + 0.5, alpha=0.6))

        # Create legend handle for this head
        legend_handles.append(mpatches.Patch(color=color, label=label))

    ax.set_xlim(min(result.source_positions) - 10 if result.source_positions else 0,
                dest_pos + 10)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: Attention Flow (Top 5 Heads)\nDest P{dest_pos} -> Source Positions',
                 fontsize=14)
    ax.set_yticks([y_positions['src'], y_positions['dest']])
    ax.set_yticklabels(['Source\n(Key)', 'Dest\n(Query)'])
    ax.grid(axis='x', alpha=GRID_ALPHA)

    # Add legend outside the plot area
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _annotate_key_tokens(
    ax: plt.Axes,
    result: AttnPairResult,
    mapping: "SamplePositionMapping",
    y_positions: dict[str, float],
) -> None:
    """Annotate key positions with token identities on the attention flow diagram.

    Args:
        ax: Matplotlib axes
        result: Attention analysis results
        mapping: SamplePositionMapping with token info
        y_positions: Y positions for source and dest
    """
    # Annotate destination position
    dest_info = mapping.get_position(result.dest_position)
    if dest_info:
        token_repr = repr(dest_info.decoded_token)[:20]  # Truncate long tokens
        ax.annotate(
            f'P{result.dest_position}: {token_repr}',
            xy=(result.dest_position, y_positions['dest']),
            xytext=(0, 15), textcoords='offset points',
            fontsize=8, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFECB3', alpha=0.8)
        )

    # Annotate key source positions
    key_positions = result.source_positions[:5]

    for pos in key_positions:
        pos_info = mapping.get_position(pos)
        if pos_info:
            token_repr = repr(pos_info.decoded_token)[:15]
            format_label = f" ({pos_info.format_pos})" if pos_info.format_pos else ""
            ax.annotate(
                f'P{pos}: {token_repr}{format_label}',
                xy=(pos, y_positions['src']),
                xytext=(0, -20), textcoords='offset points',
                fontsize=7, ha='center', va='top', rotation=45,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD', alpha=0.8)
            )


def _plot_pair_top_attended(result: AttnPairResult, output_path: Path) -> None:
    """Plot top attended positions for each head with clean vs corrupted comparison."""
    layers = [lr.layer for lr in result.layer_results]
    if not layers:
        return

    has_corrupted = bool(result.corrupted_attention_patterns)

    # Collect top attended position data
    head_data = []
    for lr in result.layer_results:
        for hi in lr.head_results:
            if hi.top_attended_positions:
                hd = {
                    'layer': lr.layer,
                    'head': hi.head_idx,
                    'attn_to_source': hi.attn_to_source,
                    'top_pos': hi.top_attended_positions[:5],
                    'top_weights': hi.top_attended_weights[:5],
                    'corrupted_weights': [],
                }
                # Extract corrupted attention weights for the same positions
                if has_corrupted and lr.layer in result.corrupted_attention_patterns:
                    corr_pattern = result.corrupted_attention_patterns[lr.layer]
                    if hi.head_idx < len(corr_pattern):
                        corr_head_attn = corr_pattern[hi.head_idx]
                        for pos in hd['top_pos']:
                            if pos < len(corr_head_attn):
                                hd['corrupted_weights'].append(corr_head_attn[pos])
                            else:
                                hd['corrupted_weights'].append(0.0)
                head_data.append(hd)

    if not head_data:
        return

    # Sort by attention to source and take top 10
    head_data = sorted(head_data, key=lambda x: x['attn_to_source'], reverse=True)[:10]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), squeeze=False)
    bar_height = 0.35

    for idx, hd in enumerate(head_data):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        positions = hd['top_pos']
        clean_weights = hd['top_weights']
        corrupted_weights = hd['corrupted_weights']

        y = np.arange(len(positions))

        # Draw paired bars: clean (blue) and corrupted (orange)
        if corrupted_weights and len(corrupted_weights) == len(clean_weights):
            # Clean bars (top of each pair)
            clean_colors = [('#E91E63' if pos in result.source_positions else CLEAN_COLOR)
                           for pos in positions]
            ax.barh(y + bar_height/2, clean_weights, bar_height,
                   color=clean_colors, alpha=0.8, label='Clean')
            # Corrupted bars (bottom of each pair)
            corr_colors = [('#FF6B6B' if pos in result.source_positions else CORRUPTED_COLOR)
                          for pos in positions]
            ax.barh(y - bar_height/2, corrupted_weights, bar_height,
                   color=corr_colors, alpha=0.8, label='Corrupted')
        else:
            # Single bars (no corrupted data)
            colors = [('#E91E63' if pos in result.source_positions else CLEAN_COLOR)
                     for pos in positions]
            ax.barh(y, clean_weights, color=colors, alpha=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels([f'P{p}' for p in positions])
        ax.set_xlabel('Attention')
        ax.set_title(f'L{hd["layer"]}.H{hd["head"]}\n(src={hd["attn_to_source"]:.3f})',
                    fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=GRID_ALPHA)

    # Fill empty subplots
    for idx in range(len(head_data), 10):
        row = idx // 5
        col = idx % 5
        axes[row, col].axis('off')

    # Add legend for clean vs corrupted
    legend_handles = [
        mpatches.Patch(color=CLEAN_COLOR, label='Clean', alpha=0.8),
        mpatches.Patch(color=CORRUPTED_COLOR, label='Corrupted', alpha=0.8),
        mpatches.Patch(color='#E91E63', label='Source (Clean)', alpha=0.8),
    ]
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(f'Pair {result.pair_idx}: Top Attended Positions',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_pair_attention_sidebyside(result: AttnPairResult, output_path: Path) -> None:
    """Plot 1b: Side-by-side heatmaps for clean vs corrupted attention.

    For each layer, show clean and corrupted attention patterns side by side.
    """
    if not result.attention_patterns or not result.corrupted_attention_patterns:
        return

    # Only plot layers that have both clean and corrupted patterns
    layers = sorted(set(result.attention_patterns.keys()) &
                   set(result.corrupted_attention_patterns.keys()))
    if not layers:
        return

    # Determine crop ranges
    src_start, src_end = DEFAULT_SRC_RANGE
    if result.source_positions:
        src_start = max(0, min(result.source_positions) - 5)
        src_end = min(200, max(result.source_positions) + 5)

    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 4 * n_layers), squeeze=False)

    for i, layer in enumerate(layers):
        clean_patterns = np.array(result.attention_patterns[layer])
        corrupted_patterns = np.array(result.corrupted_attention_patterns[layer])

        n_heads = clean_patterns.shape[0]
        seq_len_clean = clean_patterns.shape[1]
        seq_len_corr = corrupted_patterns.shape[1]

        # Crop to interesting range
        crop_start = max(0, src_start)
        crop_end_clean = min(seq_len_clean, src_end)
        crop_end_corr = min(seq_len_corr, src_end)

        clean_cropped = clean_patterns[:, crop_start:crop_end_clean]
        corr_cropped = corrupted_patterns[:, crop_start:crop_end_corr]

        # Compute shared vmin/vmax for consistent coloring
        vmax = max(clean_cropped.max(), corr_cropped.max())

        # Clean heatmap
        ax_clean = axes[i, 0]
        im_clean = ax_clean.imshow(clean_cropped, aspect='auto', cmap='viridis',
                                    extent=[crop_start, crop_end_clean, n_heads - 0.5, -0.5],
                                    vmin=0, vmax=vmax)
        ax_clean.set_xlabel('Position (Key)')
        ax_clean.set_ylabel('Head')
        ax_clean.set_title(f'Layer {layer} - Clean')
        for sp in result.source_positions:
            if crop_start <= sp < crop_end_clean:
                ax_clean.axvline(x=sp, color='red', alpha=0.5, linewidth=1, linestyle='--')
        plt.colorbar(im_clean, ax=ax_clean, label='Attention', shrink=0.8)

        # Corrupted heatmap
        ax_corr = axes[i, 1]
        im_corr = ax_corr.imshow(corr_cropped, aspect='auto', cmap='viridis',
                                  extent=[crop_start, crop_end_corr, n_heads - 0.5, -0.5],
                                  vmin=0, vmax=vmax)
        ax_corr.set_xlabel('Position (Key)')
        ax_corr.set_ylabel('Head')
        ax_corr.set_title(f'Layer {layer} - Corrupted')
        for sp in result.source_positions:
            if crop_start <= sp < crop_end_corr:
                ax_corr.axvline(x=sp, color='red', alpha=0.5, linewidth=1, linestyle='--')
        plt.colorbar(im_corr, ax=ax_corr, label='Attention', shrink=0.8)

    fig.suptitle(f'Pair {result.pair_idx}: Clean vs Corrupted Attention\n(red lines = source positions)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_pair_attention_diff(result: AttnPairResult, output_path: Path) -> None:
    """Plot 2: Attention difference heatmaps (clean - corrupted).

    Diverging colormap: Red = more attention in clean, Blue = more in corrupted.
    """
    if not result.attention_patterns or not result.corrupted_attention_patterns:
        return

    # Only plot layers that have both clean and corrupted patterns
    layers = sorted(set(result.attention_patterns.keys()) &
                   set(result.corrupted_attention_patterns.keys()))
    if not layers:
        return

    # Determine crop ranges
    src_start, src_end = DEFAULT_SRC_RANGE
    if result.source_positions:
        src_start = max(0, min(result.source_positions) - 5)
        src_end = min(200, max(result.source_positions) + 5)

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)

    for i, layer in enumerate(layers):
        ax = axes[0, i]
        clean_patterns = np.array(result.attention_patterns[layer])
        corrupted_patterns = np.array(result.corrupted_attention_patterns[layer])

        n_heads = clean_patterns.shape[0]
        seq_len = min(clean_patterns.shape[1], corrupted_patterns.shape[1])

        # Compute difference (truncate to same length)
        diff = clean_patterns[:, :seq_len] - corrupted_patterns[:, :seq_len]

        # Crop to interesting range
        crop_start = max(0, src_start)
        crop_end = min(seq_len, src_end)
        if crop_end <= crop_start:
            crop_start, crop_end = 0, seq_len

        diff_cropped = diff[:, crop_start:crop_end]

        # Use diverging colormap centered at 0
        vabs = max(abs(diff_cropped.min()), abs(diff_cropped.max()), 0.01)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

        im = ax.imshow(diff_cropped, aspect='auto', cmap=DIFF_CMAP, norm=norm,
                       extent=[crop_start, crop_end, n_heads - 0.5, -0.5])
        ax.set_xlabel('Position (Key)')
        ax.set_ylabel('Head')
        ax.set_title(f'Layer {layer}')

        # Mark source positions
        for sp in result.source_positions:
            if crop_start <= sp < crop_end:
                ax.axvline(x=sp, color='black', alpha=0.5, linewidth=1, linestyle='--')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Clean - Corrupted')

    fig.suptitle(f'Pair {result.pair_idx}: Attention Difference\n(Red = more in clean, Blue = more in corrupted)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


# =============================================================================
# Source Attention Visualizations
# =============================================================================


def _plot_source_attention_bars(result: AttnPairResult, output_path: Path) -> None:
    """Plot attention to source positions (time_horizon tokens).

    Shows paired clean/corrupted bars per head, directly answering:
    "Which heads read the time-horizon tokens?"
    """
    if not result.source_positions:
        return

    if not result.attention_patterns or not result.corrupted_attention_patterns:
        return

    layers = sorted(set(result.attention_patterns.keys()) &
                   set(result.corrupted_attention_patterns.keys()))
    if not layers:
        return

    # Collect attention to source positions for each head
    # Use source_positions for corrupted and source_positions_clean for clean
    source_pos_clean = result.source_positions_clean if result.source_positions_clean else []
    source_pos_corr = result.source_positions

    head_data = []
    for layer in layers:
        clean_patterns = result.attention_patterns[layer]
        corr_patterns = result.corrupted_attention_patterns[layer]

        n_heads = len(clean_patterns)
        for head_idx in range(n_heads):
            clean_attn = clean_patterns[head_idx]
            corr_attn = corr_patterns[head_idx]

            # Sum attention to source positions (using frame-appropriate positions)
            clean_sum = sum(
                clean_attn[p] for p in source_pos_clean
                if p < len(clean_attn)
            )
            corr_sum = sum(
                corr_attn[p] for p in source_pos_corr
                if p < len(corr_attn)
            )

            head_data.append({
                'layer': layer,
                'head': head_idx,
                'clean': clean_sum,
                'corrupted': corr_sum,
                'diff': clean_sum - corr_sum,
            })

    if not head_data:
        return

    # Sort by clean attention and take top 20
    head_data = sorted(head_data, key=lambda x: x['clean'], reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(14, 6))

    labels = [f"L{d['layer']}.H{d['head']}" for d in head_data]
    clean_vals = [d['clean'] for d in head_data]
    corr_vals = [d['corrupted'] for d in head_data]

    x = np.arange(len(labels))
    width = 0.35

    bars_clean = ax.bar(x - width/2, clean_vals, width, label='Clean',
                        color=CLEAN_COLOR, alpha=0.8)
    bars_corr = ax.bar(x + width/2, corr_vals, width, label='Corrupted',
                       color=CORRUPTED_COLOR, alpha=0.8)

    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Attention to Source Positions', fontsize=12)

    # Use semantic position names if available
    if result.source_position_names:
        pos_label = ', '.join(result.source_position_names)
    else:
        pos_label = f"P{min(result.source_positions)}-P{max(result.source_positions)}"

    ax.set_title(f'Pair {result.pair_idx}: Attention to Source ({pos_label})',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_cosine_consistency(result: AttnPairResult, output_path: Path) -> None:
    """Plot cosine similarity of attention patterns between clean/corrupted.

    Shows which heads have consistent vs changing attention patterns.
    """
    layers = [lr.layer for lr in result.layer_results]
    if not layers:
        return

    # Collect cosine similarity for each head
    head_data = []
    for lr in result.layer_results:
        for hi in lr.head_results:
            if hi.attn_pattern_cosine != 0.0:  # Only if computed
                head_data.append({
                    'layer': lr.layer,
                    'head': hi.head_idx,
                    'cosine': hi.attn_pattern_cosine,
                    'attn_to_source': hi.attn_to_source,
                })

    if not head_data:
        return

    # Sort by cosine similarity (lowest = most different patterns)
    head_data = sorted(head_data, key=lambda x: x['cosine'])

    # Take heads from both ends: most different and most similar
    n_show = min(15, len(head_data))
    most_different = head_data[:n_show // 2]
    most_similar = head_data[-(n_show - n_show // 2):]
    selected = most_different + most_similar

    fig, ax = plt.subplots(figsize=(14, 6))

    labels = [f"L{d['layer']}.H{d['head']}" for d in selected]
    cosines = [d['cosine'] for d in selected]
    colors = ['#E74C3C' if c < 0.9 else '#2ECC71' for c in cosines]

    x = np.arange(len(labels))
    bars = ax.bar(x, cosines, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Cosine Similarity (Clean vs Corrupted)', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: Attention Pattern Consistency\n'
                 '(Red = patterns differ, Green = patterns similar)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Identical')
    ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='Threshold (0.9)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


# =============================================================================
# Plot 8: QK Analysis
# =============================================================================
# Note: Plot 7 (OV Projection Analysis) has been moved to diffmeans_viz.py

def visualize_qk_analysis(
    runner: "BinaryChoiceRunner",
    result: AttnPairResult,
    output_dir: Path,
    top_heads: list[tuple[int, int]] | None = None,
    query_positions: list[int] | None = None,
    key_positions: list[int] | None = None,
) -> int:
    """Plot 8: QK analysis - what determines where heads look.

    For top heads: computes W_QK = W_Q @ W_K^T and visualizes how query/key
    projections at specific positions affect attention patterns.

    Args:
        runner: Model runner with backend
        result: Attention analysis results
        output_dir: Output directory
        top_heads: Heads to analyze
        query_positions: Query positions (destination)
        key_positions: Key positions (source)

    Returns:
        Number of plots generated
    """
    try:
        # Check if backend supports weight matrix access
        _ = runner._backend.get_W_QK(0, 0)
    except NotImplementedError:
        log("[attn_viz] QK analysis requires TransformerLens backend")
        return 0

    if top_heads is None:
        all_heads = []
        for lr in result.layer_results:
            for hr in lr.head_results:
                all_heads.append((lr.layer, hr.head_idx, hr.attn_to_source))
        all_heads.sort(key=lambda x: x[2], reverse=True)
        top_heads = [(layer, head) for layer, head, _ in all_heads[:6]]

    if not top_heads:
        return 0

    if query_positions is None:
        query_positions = [result.dest_position] if result.dest_position else [144, 145]
    if key_positions is None:
        key_positions = result.source_positions[:10] if result.source_positions else list(range(86, 96))

    # Get activations at query and key positions
    # For now, we'll visualize the W_QK structure itself
    n_heads = len(top_heads)
    n_cols = min(3, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_heads == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (layer, head) in enumerate(top_heads):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        try:
            W_QK = runner._backend.get_W_QK(layer, head)  # [d_model, d_model]

            # Cast to float32 for SVD compatibility (some backends use bfloat16)
            W_QK = W_QK.float()

            # Compute SVD to understand the low-rank structure
            U, S, Vh = torch.linalg.svd(W_QK)

            # Plot top singular values
            top_s = S[:20].detach().cpu().float().numpy()
            ax.bar(range(len(top_s)), top_s, color='steelblue', alpha=0.7)
            ax.set_xlabel('Singular Value Index')
            ax.set_ylabel('Singular Value')
            ax.set_title(f'L{layer}.H{head} QK Spectrum')
            ax.grid(True, alpha=0.3)

            # Annotate with effective rank
            S_cpu = S.detach().cpu().float()
            total_var = float(S_cpu.sum())
            cumsum = torch.cumsum(S_cpu, 0)
            eff_rank = int((cumsum < 0.9 * total_var).sum()) + 1
            ax.text(0.95, 0.95, f'Eff. Rank (90%): {eff_rank}',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', transform=ax.transAxes,
                   ha='center', va='center')
            ax.set_title(f'L{layer}.H{head}')

    # Hide empty subplots
    for idx in range(n_heads, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle('Plot 8: QK Matrix Analysis\n(Low rank = position-based attention, High rank = content-based)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'attn_qk_analysis.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    return 1


# =============================================================================
# Plot 7: OV Projection Analysis (for attention heads)
# =============================================================================


def visualize_attn_ov_projection(
    runner: "BinaryChoiceRunner",
    result: AttnPairResult,
    output_dir: Path,
    top_n_heads: int = 10,
) -> int:
    """Visualize OV projection alignment with logit direction for top attention heads.

    Uses the top heads by attention to source from AttnPairResult.
    Computes W_OV = W_V @ W_O and measures how much the head's output
    aligns with the logit direction. Shows which heads amplify vs suppress
    the task-relevant signal.

    Args:
        runner: Model runner with TransformerLens backend
        result: Attention analysis results (used to identify top heads)
        output_dir: Output directory for plots
        top_n_heads: Number of top heads to show

    Returns:
        Number of plots generated
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if backend supports weight matrix access
    try:
        _ = runner._backend.get_W_OV(0, 0)
    except (NotImplementedError, AttributeError):
        log("[attn_viz] OV projection requires TransformerLens backend")
        return 0

    # Compute logit direction from runner
    logit_direction = _compute_logit_direction_for_attn(runner, result)
    if logit_direction is None:
        log("[attn_viz] Could not compute logit direction for OV analysis")
        return 0

    # Get top heads by attention to source from result
    all_heads = []
    for lr in result.layer_results:
        for hi in lr.head_results:
            all_heads.append((lr.layer, hi.head_idx, hi.attn_to_source))
    all_heads.sort(key=lambda x: x[2], reverse=True)
    top_heads = all_heads[:top_n_heads * 2]  # Get more to filter

    # Compute OV alignment for top heads
    head_data = []
    for layer, head, attn in top_heads:
        try:
            W_OV = runner._backend.get_W_OV(layer, head)  # [d_model, d_model]

            # Project logit direction through OV circuit
            ov_output = W_OV @ logit_direction
            ov_output = ov_output / (ov_output.norm() + 1e-10)

            # Compute alignment with logit direction
            alignment = float(torch.dot(ov_output, logit_direction).item())
            head_data.append((layer, head, alignment, attn))
        except Exception as e:
            log(f"[attn_viz] Error computing OV for L{layer}.H{head}: {e}")

    if not head_data:
        return 0

    # Sort by absolute alignment and take top N
    head_data.sort(key=lambda x: abs(x[2]), reverse=True)
    selected_heads = head_data[:top_n_heads]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"L{l}.H{h}\n(attn={a:.2f})" for l, h, _, a in selected_heads]
    alignments = [align for _, _, align, _ in selected_heads]
    colors = ["#2ECC71" if a > 0 else "#E74C3C" for a in alignments]

    bars = ax.bar(range(len(alignments)), alignments, color=colors, alpha=0.8, edgecolor="black")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Head (Layer.Head, attn=attention to source)")
    ax.set_ylabel("OV Alignment with Logit Direction")
    ax.set_title(
        f"Pair {result.pair_idx}: OV Projection Analysis\n"
        "(Green = amplifies correct answer, Red = suppresses)"
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, alignments):
        height = bar.get_height()
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )

    plt.tight_layout()
    output_path = output_dir / "attn_ov_projection.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    log(f"[attn_viz] Saved OV projection: {output_path}")
    return 1


# =============================================================================
# Intermediate Attention Analysis
# =============================================================================


def visualize_intermediate_attention(
    runner: "BinaryChoiceRunner",
    token_ids: list[int],
    output_dir: Path,
    query_positions: list[int] | None = None,
    source_positions: list[int] | None = None,
    layers: list[int] | None = None,
    pair_idx: int = 0,
) -> int:
    """Analyze attention from intermediate positions (e.g., P108/P111/P112) in early layers.

    Tests information flow from source positions (e.g., P86 time horizon) to
    intermediate positions in early layers (L10-L17).

    Args:
        runner: Model runner with access to attention weights
        token_ids: Token IDs of the sequence to analyze
        output_dir: Output directory for plots
        query_positions: Query (destination) positions to analyze (default: [108, 111, 112])
        source_positions: Source positions to track (default: [86, 87, 88])
        layers: Layers to analyze (default: [10, 11, 12, 13, 14, 15, 16, 17])
        pair_idx: Pair index for title

    Returns:
        Number of plots generated
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if query_positions is None:
        query_positions = [108, 111, 112]
    if source_positions is None:
        source_positions = [86, 87, 88]
    if layers is None:
        layers = [10, 11, 12, 13, 14, 15, 16, 17]

    # Filter to valid positions
    seq_len = len(token_ids)
    query_positions = [p for p in query_positions if p < seq_len]
    source_positions = [p for p in source_positions if p < seq_len]

    if not query_positions or not source_positions:
        log("[attn_viz] Intermediate attention: positions out of range")
        return 0

    # Build hooks for attention patterns
    hooks = set()
    for layer in layers:
        hooks.add(f"blocks.{layer}.attn.hook_pattern")
        hooks.add(f"blocks.{layer}.attn.hook_attn")

    names_filter = lambda name: name in hooks

    # Run model
    input_ids = torch.tensor([token_ids], device=runner.device)
    with torch.no_grad():
        _, cache = runner._backend.run_with_cache(input_ids, names_filter=names_filter)

    # Collect attention to source for each (layer, query_pos) combination
    results_data = []

    for layer in layers:
        # Get attention patterns
        attn = None
        for key in [f"blocks.{layer}.attn.hook_pattern", f"blocks.{layer}.attn.hook_attn"]:
            if key in cache:
                attn = cache[key][0]  # [n_heads, seq_q, seq_k]
                break

        if attn is None:
            continue

        n_heads = attn.shape[0]

        for q_pos in query_positions:
            if q_pos >= attn.shape[1]:
                continue

            for head_idx in range(n_heads):
                head_attn = attn[head_idx, q_pos, :]  # [seq_k]

                # Sum attention to source positions
                attn_to_source = sum(
                    float(head_attn[p]) for p in source_positions if p < len(head_attn)
                )

                results_data.append({
                    'layer': layer,
                    'head': head_idx,
                    'query_pos': q_pos,
                    'attn_to_source': attn_to_source,
                })

    if not results_data:
        log("[attn_viz] Intermediate attention: no data collected")
        return 0

    # Create heatmap: rows = heads (grouped by layer), cols = query positions
    # Actually, let's create a cleaner visualization: for each query position,
    # show top heads across all layers

    n_query = len(query_positions)
    fig, axes = plt.subplots(1, n_query, figsize=(6 * n_query, 8), squeeze=False)

    for q_idx, q_pos in enumerate(query_positions):
        ax = axes[0, q_idx]

        # Filter to this query position
        q_data = [d for d in results_data if d['query_pos'] == q_pos]

        # Sort by attention to source
        q_data = sorted(q_data, key=lambda x: x['attn_to_source'], reverse=True)[:15]

        if not q_data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'P{q_pos}')
            continue

        labels = [f"L{d['layer']}.H{d['head']}" for d in q_data]
        attns = [d['attn_to_source'] for d in q_data]

        # Color by layer
        layer_to_color = {l: LAYER_COLORS[i % len(LAYER_COLORS)] for i, l in enumerate(sorted(layers))}
        colors = [layer_to_color[d['layer']] for d in q_data]

        y = np.arange(len(labels))
        ax.barh(y, attns, color=colors, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Attention to Source')
        ax.set_title(f'Query P{q_pos}')
        ax.set_xlim(0, max(attns) * 1.1 if attns else 1)
        ax.grid(axis='x', alpha=GRID_ALPHA)
        ax.invert_yaxis()

    src_str = ', '.join(f'P{p}' for p in source_positions)
    fig.suptitle(
        f'Pair {pair_idx}: Intermediate Attention (L{min(layers)}-L{max(layers)})\n'
        f'Attention to Source ({src_str})',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    output_path = output_dir / 'attn_intermediate.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    log(f"[attn_viz] Saved intermediate attention: {output_path}")
    return 1


def _compute_logit_direction_for_attn(
    runner: "BinaryChoiceRunner",
    result: AttnPairResult,
) -> torch.Tensor | None:
    """Compute normalized logit direction for OV analysis.

    Uses the runner's W_U matrix to compute the direction between
    clean and corrupted logits based on the pair's divergent tokens.
    """
    W_U = runner.W_U
    if W_U is None:
        return None

    # We need token IDs - use the result's stored info if available
    # For now, use a heuristic based on common divergent patterns
    # This should be enhanced to pass actual pair info
    try:
        # Try to get from runner's last cached pair
        if hasattr(runner, '_last_pair') and runner._last_pair is not None:
            pair = runner._last_pair
            clean_div_pos = pair.clean_divergent_position
            corrupted_div_pos = pair.corrupted_divergent_position

            if clean_div_pos is None or corrupted_div_pos is None:
                clean_token = pair.clean_traj.token_ids[-1]
                corrupted_token = pair.corrupted_traj.token_ids[-1]
            else:
                clean_token = pair.clean_traj.token_ids[clean_div_pos]
                corrupted_token = pair.corrupted_traj.token_ids[corrupted_div_pos]

            if clean_token == corrupted_token:
                return None

            if W_U.shape[0] > W_U.shape[1]:
                clean_vec = W_U[clean_token]
                corrupted_vec = W_U[corrupted_token]
            else:
                clean_vec = W_U[:, clean_token]
                corrupted_vec = W_U[:, corrupted_token]

            direction = clean_vec - corrupted_vec
            return direction / torch.norm(direction)
    except Exception:
        pass

    return None
