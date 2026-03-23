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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from ...common.logging import log
from ..experiments.attn_analysis import AttnAggregatedResults, AttnPairResult


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
    ax.set_title(f'Attention to Source (Horizon) Positions\n({agg.n_pairs} pairs)', fontsize=14)
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
    ax.set_title(f'Top Attention Heads: Dest -> Source\nLayers {layers} ({agg.n_pairs} pairs)', fontsize=14)
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
    ax.set_title(f'Head Importance vs Attention Shift\n({agg.n_pairs} pairs)', fontsize=14)
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
    ax.set_title(f'Attention Consistency: Top Dynamic Heads\n({agg.n_pairs} pairs)', fontsize=14)
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
) -> None:
    """Generate attention analysis visualizations for a single pair.

    Args:
        result: Attention analysis results for one pair
        output_dir: Directory to save plots
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
            _plot_pair_attention_flow(result, output_dir / "attn_flow.png")
            n_plots += 1

        # Top attended positions visualization
        _plot_pair_top_attended(result, output_dir / "attn_top_attended.png")
        n_plots += 1
    else:
        # Layer-level summary for this pair
        _plot_pair_layer_summary(result, output_dir / "attn_pair_summary.png")
        n_plots += 1

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


def _plot_pair_attention_flow(result: AttnPairResult, output_path: Path) -> None:
    """Plot 4: Attention flow diagram for top heads.

    Lines from query positions to key positions, thickness = attention weight.
    For the top N heads by attention to source.
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

    fig, ax = plt.subplots(figsize=(10, 8))

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

    # Draw attention lines for top heads
    colors = LAYER_COLORS[:len(top_heads)]
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

        # Add head label
        ax.text(dest_pos + (idx - 2) * 3, y_positions['dest'] + 0.05, label,
               fontsize=8, ha='center', color=color)

    ax.set_xlim(min(result.source_positions) - 10 if result.source_positions else 0,
                dest_pos + 10)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: Attention Flow (Top 5 Heads)\nDest P{dest_pos} -> Source Positions',
                 fontsize=14)
    ax.set_yticks([y_positions['src'], y_positions['dest']])
    ax.set_yticklabels(['Source\n(Key)', 'Dest\n(Query)'])
    ax.grid(axis='x', alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def _plot_pair_top_attended(result: AttnPairResult, output_path: Path) -> None:
    """Plot top attended positions for each head."""
    layers = [lr.layer for lr in result.layer_results]
    if not layers:
        return

    # Collect top attended position data
    head_data = []
    for lr in result.layer_results:
        for hi in lr.head_results:
            if hi.top_attended_positions:
                head_data.append({
                    'layer': lr.layer,
                    'head': hi.head_idx,
                    'attn_to_source': hi.attn_to_source,
                    'top_pos': hi.top_attended_positions[:5],
                    'top_weights': hi.top_attended_weights[:5],
                })

    if not head_data:
        return

    # Sort by attention to source and take top 10
    head_data = sorted(head_data, key=lambda x: x['attn_to_source'], reverse=True)[:10]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), squeeze=False)

    for idx, hd in enumerate(head_data):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        positions = hd['top_pos']
        weights = hd['top_weights']

        # Color bars by whether position is in source range
        colors = []
        for pos in positions:
            if pos in result.source_positions:
                colors.append('#E91E63')  # Highlight source positions
            else:
                colors.append('#2196F3')

        ax.barh(range(len(positions)), weights, color=colors, alpha=0.8)
        ax.set_yticks(range(len(positions)))
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

    # Add legend
    fig.suptitle(f'Pair {result.pair_idx}: Top Attended Positions (Pink = Source Position)',
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


# Legacy function kept for backwards compatibility
def _plot_pair_attention_patterns(result: AttnPairResult, output_path: Path) -> None:
    """Plot attention pattern heatmaps for a single pair (legacy function)."""
    # Redirect to new heatmap function
    _plot_pair_attention_heatmaps(result, output_path)
