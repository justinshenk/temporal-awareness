"""Visualization for MLP layer contribution analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ....common.logging import log
from ....viz.plot_helpers import finalize_plot, save_figure
from ..fine.fine_results import LayerPositionResult
from . import MLPAggregatedResults, MLPPairResult, NeuronInfo

if TYPE_CHECKING:
    from ...common.sample_position_mapping import SamplePositionMapping


# Plot styling
DPI = 150
POSITIVE_COLOR = "#4ECDC4"  # Teal for positive contributions
NEGATIVE_COLOR = "#FF6B6B"  # Red for negative contributions
CLEAN_COLOR = "#2ecc71"  # Green for clean activations
CORRUPTED_COLOR = "#9b59b6"  # Purple for corrupted activations
BAR_ALPHA = 0.8
GRID_ALPHA = 0.3

def _find_best_target_layer(agg: MLPAggregatedResults, format_pos: str) -> int | None:
    """Find the best layer for detailed neuron analysis.

    Returns the layer with the most neuron data across pairs.
    Prefers later layers (more likely to show task-relevant patterns).
    """
    layers = agg.layers_analyzed
    if not layers:
        return None

    # Score each layer by amount of neuron data
    layer_scores: dict[int, int] = {}
    for layer in layers:
        score = 0
        for pr in agg.pair_results:
            lr = pr.get_layer_result(format_pos, layer)
            if lr and lr.top_neurons:
                score += len(lr.top_neurons)
        layer_scores[layer] = score

    # Find layers with data
    layers_with_data = [l for l, s in layer_scores.items() if s > 0]
    if not layers_with_data:
        return None

    # Prefer the highest (latest) layer with data
    return max(layers_with_data)


def _get_first_format_pos(agg: MLPAggregatedResults) -> str | None:
    """Get the first available format_pos from results."""
    positions = agg.get_all_positions()
    if positions:
        return sorted(positions)[0]
    return None


def _get_first_format_pos_pair(result: MLPPairResult) -> str | None:
    """Get the first available format_pos from a single pair result."""
    positions = result.get_all_format_positions()
    if positions:
        return sorted(positions)[0]
    return None


def _find_best_target_layer_pair(result: MLPPairResult, format_pos: str) -> int | None:
    """Find the best layer for detailed neuron analysis in a single pair.

    Returns the layer with the most neuron data, preferring later layers.
    """
    if format_pos not in result.position_results:
        return None

    layer_results = result.position_results[format_pos]
    if not layer_results:
        return None

    # Score each layer by amount of neuron data
    layer_scores: dict[int, int] = {}
    for lr in layer_results:
        if lr.top_neurons:
            layer_scores[lr.layer] = len(lr.top_neurons)

    if not layer_scores:
        return None

    # Prefer the highest (latest) layer with data
    return max(layer_scores.keys())


def visualize_mlp_analysis(
    agg: MLPAggregatedResults,
    output_dir: Path,
    format_pos: str | None = None,
) -> None:
    """Generate all MLP analysis visualizations.

    Args:
        agg: Aggregated MLP analysis results
        output_dir: Directory to save plots
        format_pos: Position to visualize (default: "response_choice" or first available)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not agg.pair_results:
        log("[mlp_viz] No pair results to visualize")
        return

    if format_pos is None:
        format_pos = _get_first_format_pos(agg)

    if format_pos is None:
        log("[mlp_viz] No positions available to visualize")
        return

    n_plots = 0

    # Layer contribution bar chart
    _plot_layer_contributions(agg, output_dir / "mlp_layer_contributions.png", format_pos)
    n_plots += 1

    # Layer contribution heatmap across pairs
    if agg.n_pairs > 1:
        _plot_contribution_heatmap(agg, output_dir / "mlp_contribution_heatmap.png", format_pos)
        n_plots += 1

    # Per-pair layer contributions
    _plot_per_pair_contributions(agg, output_dir / "mlp_per_pair_contributions.png", format_pos)
    n_plots += 1

    # Generate neuron-level plots for ALL layers with data
    for layer in agg.layers_analyzed:
        if not _has_neuron_data(agg, layer, format_pos):
            continue

        # Differential logit contribution bar chart (top neurons by task relevance)
        _plot_differential_logit_contribution(
            agg, output_dir / f"mlp_differential_logit_contrib_L{layer}.png", format_pos,
            layer=layer
        )
        n_plots += 1

        # Cumulative neuron contribution curve
        _plot_cumulative_neuron_contribution(
            agg, output_dir / f"mlp_cumulative_contribution_L{layer}.png", format_pos,
            layer=layer
        )
        n_plots += 1

        # Clean vs corrupted neuron activation scatter
        _plot_neuron_clean_vs_corrupted(
            agg, output_dir / f"mlp_clean_vs_corrupted_L{layer}.png", format_pos,
            layer=layer
        )
        n_plots += 1

        # 10. Neuron activation difference heatmap across layers
        _plot_neuron_activation_heatmap_across_layers(
            agg, output_dir / f"neuron_activation_heatmap_layers_L{layer}.png", format_pos,
            target_layer=layer
        )
        n_plots += 1

        # 11. Neuron consistency across pairs
        if agg.n_pairs > 1:
            _plot_neuron_consistency_across_pairs(
                agg, output_dir / f"neuron_consistency_across_pairs_L{layer}.png", format_pos,
                layer=layer
            )
            n_plots += 1

        # 15. Cross-layer neuron patterns
        if len(agg.layers_analyzed) > 1:
            _plot_cross_layer_neuron_patterns(
                agg, output_dir / f"cross_layer_neuron_patterns_L{layer}.png", format_pos,
                target_layer=layer
            )
            n_plots += 1

    log(f"[mlp_viz] Generated {n_plots} plots in {output_dir} (position: {format_pos})")


def visualize_all_mlp_slices(
    agg: MLPAggregatedResults,
    output_dir: Path,
    pref_pairs: list | None = None,
) -> None:
    """Visualize MLP analysis for all analysis slices.

    Args:
        agg: Aggregated MLP analysis results
        output_dir: Output directory
        pref_pairs: List of ContrastivePreferences for slice filtering
    """
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

        # Filter data for this slice
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

        visualize_mlp_analysis(filtered_agg, slice_dir)

    log(f"[mlp_viz] All MLP slices saved to {output_dir}")


def _plot_layer_contributions(
    agg: MLPAggregatedResults, output_path: Path, format_pos: str
) -> None:
    """Plot mean layer contributions as grouped bar chart showing clean vs corrupted.

    Shows three bars per layer:
    - Clean: contribution from clean-run activations
    - Corrupted: contribution from corrupted-run activations
    - Difference: clean - corrupted (the task-relevant signal)
    """
    layers = agg.layers_analyzed
    if not layers:
        return

    # Compute mean contributions per layer (clean, corrupted, difference)
    mean_clean_contribs = []
    mean_corrupt_contribs = []
    mean_diff_contribs = []

    for layer in layers:
        clean_contribs = []
        corrupt_contribs = []
        diff_contribs = []

        for pr in agg.pair_results:
            lr = pr.get_layer_result(format_pos, layer)
            if lr:
                # Sum neuron-level contributions
                # clean_contrib = sum(activation * w_out_alignment)
                layer_clean = sum(
                    n.clean_activation * n.w_out_logit_alignment
                    for n in lr.top_neurons
                )
                layer_corrupt = sum(
                    n.corrupted_activation * n.w_out_logit_alignment
                    for n in lr.top_neurons
                )
                clean_contribs.append(layer_clean)
                corrupt_contribs.append(layer_corrupt)
                diff_contribs.append(lr.total_logit_contribution)

        mean_clean_contribs.append(np.mean(clean_contribs) if clean_contribs else 0)
        mean_corrupt_contribs.append(np.mean(corrupt_contribs) if corrupt_contribs else 0)
        mean_diff_contribs.append(np.mean(diff_contribs) if diff_contribs else 0)

    # Create figure with grouped bars
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(layers))
    width = 0.25

    bars_clean = ax.bar(x - width, mean_clean_contribs, width,
                        label='Clean Run', color=CLEAN_COLOR, alpha=BAR_ALPHA)
    bars_corrupt = ax.bar(x, mean_corrupt_contribs, width,
                          label='Corrupted Run', color=CORRUPTED_COLOR, alpha=BAR_ALPHA)
    bars_diff = ax.bar(x + width, mean_diff_contribs, width,
                       label='Difference (Clean - Corrupted)', color='steelblue', alpha=BAR_ALPHA)

    # Add value labels on difference bars only (to avoid clutter)
    for bar, val in zip(bars_diff, mean_diff_contribs):
        height = bar.get_height()
        sign = "+" if val >= 0 else ""
        ax.annotate(f'{sign}{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Logit Contribution', fontsize=12)
    ax.set_title(f'MLP Layer Contributions: Clean vs Corrupted\n({agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""}, position: {format_pos})',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)
    ax.legend(loc='best')

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_differential_logit_contribution(
    agg: MLPAggregatedResults, output_path: Path, format_pos: str,
    layer: int, top_n: int = 30
) -> None:
    """Plot differential logit contribution bar chart for neurons.

    Shows: (clean_activation - corrupted_activation) * W_out_alignment
    sorted by absolute value. This is the single most important metric
    for identifying which neurons actually drive the task.
    """
    # Collect neuron contributions across pairs
    neuron_contribs: dict[int, list[float]] = {}

    for pr in agg.pair_results:
        lr = pr.get_layer_result(format_pos, layer)
        if not lr or not lr.top_neurons:
            continue
        for ni in lr.top_neurons:
            if ni.neuron_idx not in neuron_contribs:
                neuron_contribs[ni.neuron_idx] = []
            neuron_contribs[ni.neuron_idx].append(ni.logit_contribution)

    if not neuron_contribs:
        return

    # Compute mean contribution per neuron and sort by absolute value
    neuron_means = [
        (idx, np.mean(contribs), np.std(contribs) if len(contribs) > 1 else 0)
        for idx, contribs in neuron_contribs.items()
    ]
    neuron_means.sort(key=lambda x: abs(x[1]), reverse=True)
    neuron_means = neuron_means[:top_n]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(neuron_means) * 0.25)))

    indices = [n[0] for n in neuron_means]
    means = [n[1] for n in neuron_means]
    stds = [n[2] for n in neuron_means]

    y = np.arange(len(indices))
    colors = [POSITIVE_COLOR if m >= 0 else NEGATIVE_COLOR for m in means]

    bars = ax.barh(y, means, xerr=stds if agg.n_pairs > 1 else None,
                   color=colors, alpha=BAR_ALPHA, capsize=3, ecolor='gray')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, means)):
        width = bar.get_width()
        sign = "+" if val >= 0 else ""
        x_pos = width + (0.01 * max(abs(m) for m in means)) if val >= 0 else width - (0.01 * max(abs(m) for m in means))
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{sign}{val:.3f}', va='center', ha='left' if val >= 0 else 'right',
                fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([f'N{idx}' for idx in indices])
    ax.set_xlabel('Differential Logit Contribution\n(clean_act - corrupt_act) * W_out_alignment', fontsize=11)
    ax.set_ylabel(f'Neuron (L{layer})', fontsize=12)
    ax.set_title(f'Top {len(neuron_means)} Neurons by Differential Logit Contribution\n'
                 f'L{layer} ({agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""}, position: {format_pos})', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=GRID_ALPHA)

    # Add legend
    legend_elements = [
        Patch(facecolor=POSITIVE_COLOR, alpha=BAR_ALPHA, label='Supports clean prediction'),
        Patch(facecolor=NEGATIVE_COLOR, alpha=BAR_ALPHA, label='Opposes clean prediction')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_neuron_clean_vs_corrupted(
    agg: MLPAggregatedResults, output_path: Path, format_pos: str,
    layer: int, top_n: int = 20
) -> None:
    """Plot clean vs corrupted activation for top neurons.

    Scatter plot showing which neurons are more active on clean vs corrupted.
    """
    # Collect neuron activations
    neuron_data: dict[int, tuple[list[float], list[float]]] = {}
    for pr in agg.pair_results:
        lr = pr.get_layer_result(format_pos, layer)
        if not lr or not lr.top_neurons:
            continue
        for ni in lr.top_neurons:
            if ni.neuron_idx not in neuron_data:
                neuron_data[ni.neuron_idx] = ([], [])
            neuron_data[ni.neuron_idx][0].append(ni.clean_activation)
            neuron_data[ni.neuron_idx][1].append(ni.corrupted_activation)

    if not neuron_data:
        return

    # Get top neurons by activation difference
    neuron_stats = []
    for idx, (clean_acts, corrupt_acts) in neuron_data.items():
        mean_clean = np.mean(clean_acts)
        mean_corrupt = np.mean(corrupt_acts)
        diff = abs(mean_clean - mean_corrupt)
        neuron_stats.append((idx, mean_clean, mean_corrupt, diff))

    neuron_stats.sort(key=lambda x: x[3], reverse=True)
    neuron_stats = neuron_stats[:top_n]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    clean_vals = [n[1] for n in neuron_stats]
    corrupt_vals = [n[2] for n in neuron_stats]
    indices = [n[0] for n in neuron_stats]

    # Scatter plot
    scatter = ax.scatter(corrupt_vals, clean_vals, c=range(len(neuron_stats)),
                        cmap='viridis', s=100, alpha=0.7)

    # Add diagonal line (y=x)
    lims = [min(min(clean_vals), min(corrupt_vals)) - 0.1,
            max(max(clean_vals), max(corrupt_vals)) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x (no difference)')

    # Label points
    for i, (idx, clean, corrupt, _) in enumerate(neuron_stats):
        ax.annotate(f'N{idx}', (corrupt, clean), fontsize=8,
                   xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('Corrupted Activation', fontsize=12)
    ax.set_ylabel('Clean Activation', fontsize=12)
    ax.set_title(f'Neuron Activations: Clean vs Corrupted (L{layer})\n'
                 f'Top {len(neuron_stats)} neurons by |difference| (position: {format_pos})', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(alpha=GRID_ALPHA)

    # Color bar for ranking
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Rank by |clean - corrupted|')

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_cumulative_neuron_contribution(
    agg: MLPAggregatedResults, output_path: Path, format_pos: str,
    layer: int
) -> None:
    """Plot cumulative neuron contribution curve.

    Shows fraction of total effect recovered by top-N neurons.
    Tells you if it's a 5-neuron or 500-neuron story.
    """
    # Collect neuron contributions
    neuron_contribs: dict[int, list[float]] = {}
    for pr in agg.pair_results:
        lr = pr.get_layer_result(format_pos, layer)
        if not lr or not lr.top_neurons:
            continue
        for ni in lr.top_neurons:
            if ni.neuron_idx not in neuron_contribs:
                neuron_contribs[ni.neuron_idx] = []
            neuron_contribs[ni.neuron_idx].append(abs(ni.logit_contribution))

    if not neuron_contribs:
        return

    # Sort by mean absolute contribution
    neuron_means = [(idx, np.mean(contribs)) for idx, contribs in neuron_contribs.items()]
    neuron_means.sort(key=lambda x: x[1], reverse=True)

    # Compute cumulative sum
    total = sum(m[1] for m in neuron_means)
    if total == 0:
        return

    cumsum = np.cumsum([m[1] for m in neuron_means])
    fractions = cumsum / total * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(1, len(fractions) + 1)
    ax.plot(x, fractions, 'b-', linewidth=2, marker='o', markersize=3)
    ax.fill_between(x, fractions, alpha=0.3)

    # Add reference lines
    for thresh in [50, 80, 90, 95]:
        ax.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5)
        # Find first neuron to cross threshold
        idx = np.searchsorted(fractions, thresh)
        if idx < len(fractions):
            ax.axvline(x=idx+1, color='red', linestyle=':', alpha=0.5)
            ax.annotate(f'{thresh}% @ N={idx+1}', xy=(idx+1, thresh),
                       xytext=(idx+5, thresh-5), fontsize=9,
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

    ax.set_xlabel('Top N Neurons', fontsize=12)
    ax.set_ylabel('Cumulative Contribution (%)', fontsize=12)
    ax.set_title(f'Cumulative Neuron Contribution (L{layer})\n'
                 f'{agg.n_pairs} pair{"s" if agg.n_pairs != 1 else ""} (position: {format_pos})', fontsize=14)
    ax.set_xlim(1, min(len(fractions), 100))
    ax.set_ylim(0, 105)
    ax.grid(alpha=GRID_ALPHA)

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_contribution_heatmap(
    agg: MLPAggregatedResults, output_path: Path, format_pos: str
) -> None:
    """Plot heatmap of layer contributions across pairs."""
    layers = agg.layers_analyzed
    if not layers or agg.n_pairs < 2:
        return

    # Build contribution matrix
    matrix = []
    for pr in agg.pair_results:
        row = []
        for layer in layers:
            lr = pr.get_layer_result(format_pos, layer)
            row.append(lr.total_logit_contribution if lr else 0)
        matrix.append(row)

    matrix = np.array(matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.8), max(4, agg.n_pairs * 0.4)))

    # Use diverging colormap centered at 0
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.invert_yaxis()  # Pair 0 at bottom

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Logit Contribution', fontsize=10)

    # Labels
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_yticks(np.arange(agg.n_pairs))
    ax.set_yticklabels([f'Pair {i}' for i in range(agg.n_pairs)])
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Pair', fontsize=12)
    ax.set_title(f'MLP Layer Contributions Across Pairs (position: {format_pos})', fontsize=14)

    # Add text annotations
    for i in range(agg.n_pairs):
        for j in range(len(layers)):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=8)

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_per_pair_contributions(
    agg: MLPAggregatedResults, output_path: Path, format_pos: str
) -> None:
    """Plot layer contributions for each pair as grouped bars."""
    layers = agg.layers_analyzed
    if not layers:
        return

    n_pairs = min(agg.n_pairs, 10)  # Limit to 10 pairs for readability

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(layers))
    width = 0.8 / n_pairs

    colors = plt.cm.tab10(np.linspace(0, 1, n_pairs))

    for i, pr in enumerate(agg.pair_results[:n_pairs]):
        contribs = []
        for layer in layers:
            lr = pr.get_layer_result(format_pos, layer)
            contribs.append(lr.total_logit_contribution if lr else 0)

        offset = (i - n_pairs / 2 + 0.5) * width
        ax.bar(x + offset, contribs, width, label=f'Pair {i}',
               color=colors[i], alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Logit Contribution', fontsize=12)
    ax.set_title(f'MLP Layer Contributions by Pair (position: {format_pos})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _has_neuron_data(agg: MLPAggregatedResults, layer: int, format_pos: str) -> bool:
    """Check if aggregated results have per-neuron data for a layer."""
    for pr in agg.pair_results:
        lr = pr.get_layer_result(format_pos, layer)
        if lr and lr.top_neurons:
            return True
    return False


def _get_top_neurons_for_layer(
    agg: MLPAggregatedResults, layer: int, format_pos: str, n: int = 20
) -> list[tuple[int, float, float, float]]:
    """Get top neurons by mean activation difference across pairs.

    Returns:
        List of (neuron_idx, mean_act_diff, mean_clean_act, mean_corrupted_act)
    """
    neuron_data: dict[int, list[NeuronInfo]] = {}

    for pr in agg.pair_results:
        lr = pr.get_layer_result(format_pos, layer)
        if not lr or not lr.top_neurons:
            continue
        for ni in lr.top_neurons:
            if ni.neuron_idx not in neuron_data:
                neuron_data[ni.neuron_idx] = []
            neuron_data[ni.neuron_idx].append(ni)

    # Compute mean statistics
    results = []
    for neuron_idx, infos in neuron_data.items():
        mean_act_diff = np.mean([ni.activation_diff for ni in infos])
        mean_clean_act = np.mean([ni.clean_activation for ni in infos])
        mean_corrupted_act = np.mean([ni.corrupted_activation for ni in infos])
        results.append((neuron_idx, mean_act_diff, mean_clean_act, mean_corrupted_act))

    # Sort by absolute activation difference
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results[:n]


def _plot_neuron_activation_heatmap_across_layers(
    agg: MLPAggregatedResults, output_path: Path, format_pos: str,
    target_layer: int
) -> None:
    """Plot 10: Heatmap of top neuron activation differences across layers.

    Rows: top 20 neurons by activation difference at target_layer
    Columns: layers analyzed
    Color: activation difference at each layer
    """
    layers = agg.layers_analyzed
    if not layers:
        return

    # Get top neurons at target layer
    top_neurons = _get_top_neurons_for_layer(agg, target_layer, format_pos, n=20)
    if not top_neurons:
        return

    neuron_indices = [n[0] for n in top_neurons]

    # Build matrix: for each neuron, get its activation diff at each layer
    matrix = np.zeros((len(neuron_indices), len(layers)))

    for j, layer in enumerate(layers):
        layer_top = _get_top_neurons_for_layer(agg, layer, format_pos, n=100)
        layer_map = {n[0]: n[1] for n in layer_top}  # neuron_idx -> act_diff

        for i, neuron_idx in enumerate(neuron_indices):
            matrix[i, j] = layer_map.get(neuron_idx, 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 1.2), max(6, len(neuron_indices) * 0.3)))

    vmax = max(abs(matrix.min()), abs(matrix.max())) if matrix.size > 0 else 1
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.invert_yaxis()  # Neuron 0 at bottom

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation Difference (clean - corrupted)', fontsize=10)

    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_yticks(np.arange(len(neuron_indices)))
    ax.set_yticklabels([f'N{n}' for n in neuron_indices])
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(f'Neuron (top 20 from L{target_layer})', fontsize=12)
    ax.set_title(f'Neuron Activation Differences Across Layers\n(Top 20 neurons from L{target_layer}, position: {format_pos})', fontsize=14)

    # Add text annotations for non-zero values
    for i in range(len(neuron_indices)):
        for j in range(len(layers)):
            val = matrix[i, j]
            if abs(val) > 0.01:
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=7)

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def visualize_mlp_pair(
    result: MLPPairResult,
    output_dir: Path,
    format_pos: str | None = None,
    position_mapping: "SamplePositionMapping | None" = None,
    pair_idx: int | None = None,
) -> None:
    """Generate MLP analysis visualizations for a single pair.

    Args:
        result: MLP analysis results for one pair
        output_dir: Directory to save plots
        format_pos: Position to visualize (default: "response_choice" or first available)
        position_mapping: Optional mapping for semantic position labels in layer_position plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result.position_results:
        return

    if format_pos is None:
        format_pos = _get_first_format_pos_pair(result)

    if format_pos is None or format_pos not in result.position_results:
        return

    layer_results = result.position_results[format_pos]
    if not layer_results:
        return

    n_plots = 0

    # Layer contribution bar chart for this pair
    _plot_pair_layer_contributions(result, format_pos, output_dir / "mlp_layer_contributions.png")
    n_plots += 1

    # Generate neuron-level plots for ALL layers with data
    for lr in layer_results:
        layer = lr.layer
        has_neuron_data = lr.top_neurons is not None and len(lr.top_neurons) > 0

        if has_neuron_data:
            # 9. Neuron activation difference ranked bar chart
            _plot_neuron_activation_diff_ranked(
                result, format_pos, layer, output_dir / f"neuron_activation_diff_L{layer}.png"
            )
            n_plots += 1

            # 12. Neuron activation distribution (clean vs corrupted)
            _plot_neuron_activation_distribution(
                result, format_pos, layer, output_dir / f"neuron_activation_dist_L{layer}.png"
            )
            n_plots += 1

            # 13. Neuron output direction alignment
            _plot_neuron_output_alignment(
                result, format_pos, layer, output_dir / f"neuron_output_alignment_L{layer}.png"
            )
            n_plots += 1

            # 14. Neuron contribution decomposition
            _plot_neuron_contribution_decomposition(
                result, format_pos, layer, output_dir / f"neuron_contrib_decomp_L{layer}.png"
            )
            n_plots += 1

            # 16. Two-dimensional neuron activation scatter
            _plot_neuron_activation_scatter_2d(
                result, format_pos, layer, output_dir / f"neuron_activation_scatter_L{layer}.png"
            )
            n_plots += 1

    # Layer x position patching heatmap
    if result.layer_position is not None:
        _plot_layer_position_heatmaps(result.layer_position, output_dir, position_mapping)
        n_plots += 1

    if n_plots > 0:
        log(f"[mlp_viz] Generated {n_plots} pair plots in {output_dir} (position: {format_pos})")


def _plot_pair_layer_contributions(
    result: MLPPairResult, format_pos: str, output_path: Path
) -> None:
    """Plot layer contributions for a single pair."""
    if format_pos not in result.position_results:
        return

    layer_results = result.position_results[format_pos]
    layers = [lr.layer for lr in layer_results]
    if not layers:
        return

    # Get contributions
    contribs = [lr.total_logit_contribution for lr in layer_results]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(layers))
    colors = [POSITIVE_COLOR if c >= 0 else NEGATIVE_COLOR for c in contribs]

    bars = ax.bar(x, contribs, color=colors, alpha=BAR_ALPHA)

    # Add value labels on bars
    for bar, val in zip(bars, contribs):
        height = bar.get_height()
        sign = "+" if val >= 0 else ""
        ax.annotate(f'{sign}{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Logit Contribution', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: MLP Layer Contributions\n(Position: {format_pos})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)

    # Add legend
    legend_elements = [
        Patch(facecolor=POSITIVE_COLOR, alpha=BAR_ALPHA, label='Supports clean prediction'),
        Patch(facecolor=NEGATIVE_COLOR, alpha=BAR_ALPHA, label='Opposes clean prediction')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_neuron_activation_diff_ranked(
    result: MLPPairResult, format_pos: str, layer: int, output_path: Path, n_top: int = 50
) -> None:
    """Plot 9: Neuron activation difference ranked bar chart.

    X-axis: neuron index, sorted by |clean_activation - corrupted_activation|
    Y-axis: activation difference (signed)
    """
    lr = result.get_layer_result(format_pos, layer)
    if not lr or not lr.top_neurons:
        return

    # Get top neurons sorted by absolute activation difference
    neurons = sorted(lr.top_neurons, key=lambda n: abs(n.activation_diff), reverse=True)[:n_top]

    if not neurons:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(neurons))
    act_diffs = [n.activation_diff for n in neurons]
    neuron_ids = [n.neuron_idx for n in neurons]

    colors = [POSITIVE_COLOR if d >= 0 else NEGATIVE_COLOR for d in act_diffs]
    ax.bar(x, act_diffs, color=colors, alpha=BAR_ALPHA)

    ax.set_xlabel('Neuron Index (sorted by |activation diff|)', fontsize=12)
    ax.set_ylabel('Activation Difference (clean - corrupted)', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: L{layer} Neuron Activation Differences\n(Top {len(neurons)} neurons, position: {format_pos})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in neuron_ids], rotation=90, fontsize=7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)

    legend_elements = [
        Patch(facecolor=POSITIVE_COLOR, alpha=BAR_ALPHA, label='Higher in clean'),
        Patch(facecolor=NEGATIVE_COLOR, alpha=BAR_ALPHA, label='Higher in corrupted')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_neuron_activation_distribution(
    result: MLPPairResult, format_pos: str, layer: int, output_path: Path, n_top: int = 10
) -> None:
    """Plot 12: Neuron activation distribution (clean vs corrupted).

    Each panel shows one neuron with clean vs corrupted activation values.
    """
    lr = result.get_layer_result(format_pos, layer)
    if not lr or not lr.top_neurons:
        return

    # Get top neurons by logit contribution
    neurons = sorted(lr.top_neurons, key=lambda n: abs(n.logit_contribution), reverse=True)[:n_top]

    if not neurons:
        return

    n_neurons = len(neurons)
    n_cols = min(5, n_neurons)
    n_rows = (n_neurons + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_neurons == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, neuron in enumerate(neurons):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Plot clean and corrupted activations as vertical lines
        clean_act = neuron.clean_activation
        corrupted_act = neuron.corrupted_activation

        ax.axvline(x=clean_act, color=CLEAN_COLOR, linewidth=3, label='Clean')
        ax.axvline(x=corrupted_act, color=CORRUPTED_COLOR, linewidth=3, label='Corrupted')

        # Add difference annotation
        diff = clean_act - corrupted_act
        mid = (clean_act + corrupted_act) / 2

        # Draw arrow between the two
        ax.annotate('', xy=(clean_act, 0.5), xytext=(corrupted_act, 0.5),
                   arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
        ax.text(mid, 0.6, f'{diff:+.2f}', ha='center', fontsize=9, color='gray')

        ax.set_title(f'N{neuron.neuron_idx}', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        if idx == 0:
            ax.legend(loc='upper right', fontsize=7)

    # Hide empty subplots
    for idx in range(n_neurons, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'Pair {result.pair_idx}: L{layer} Top Neuron Activations\n(by patching effect, position: {format_pos})', fontsize=14)
    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_neuron_output_alignment(
    result: MLPPairResult, format_pos: str, layer: int, output_path: Path, n_top: int = 30
) -> None:
    """Plot 13: Neuron output direction alignment.

    Bar chart: X = neuron, Y = cosine similarity (W_out[neuron] @ logit_direction)
    """
    lr = result.get_layer_result(format_pos, layer)
    if not lr or not lr.top_neurons:
        return

    # Get neurons with alignment data
    neurons = [n for n in lr.top_neurons if n.w_out_logit_alignment != 0]
    neurons = sorted(neurons, key=lambda n: abs(n.w_out_logit_alignment), reverse=True)[:n_top]

    if not neurons:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(neurons))
    alignments = [n.w_out_logit_alignment for n in neurons]
    neuron_ids = [n.neuron_idx for n in neurons]

    colors = [POSITIVE_COLOR if a >= 0 else NEGATIVE_COLOR for a in alignments]
    ax.bar(x, alignments, color=colors, alpha=BAR_ALPHA)

    ax.set_xlabel('Neuron Index', fontsize=12)
    ax.set_ylabel('W_out alignment with logit direction', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: L{layer} Neuron Output Direction Alignment\n(W_out[neuron] @ logit_direction, position: {format_pos})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in neuron_ids], rotation=90, fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)

    legend_elements = [
        Patch(facecolor=POSITIVE_COLOR, alpha=BAR_ALPHA, label='Aligned with clean'),
        Patch(facecolor=NEGATIVE_COLOR, alpha=BAR_ALPHA, label='Aligned with corrupted')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_neuron_contribution_decomposition(
    result: MLPPairResult, format_pos: str, layer: int, output_path: Path, n_top: int = 30
) -> None:
    """Plot 14: Neuron contribution decomposition (clean vs corrupted).

    X-axis: neuron index (top 30)
    Stacked bar: contribution in clean run vs corrupted run
    Contribution = activation * W_out_logit_alignment
    """
    lr = result.get_layer_result(format_pos, layer)
    if not lr or not lr.top_neurons:
        return

    # Get top neurons by contribution magnitude
    neurons = sorted(lr.top_neurons, key=lambda n: abs(n.logit_contribution), reverse=True)[:n_top]

    if not neurons:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(neurons))
    neuron_ids = [n.neuron_idx for n in neurons]

    # Compute contributions: activation * w_out_alignment
    clean_contribs = [n.clean_activation * n.w_out_logit_alignment for n in neurons]
    corrupted_contribs = [n.corrupted_activation * n.w_out_logit_alignment for n in neurons]

    width = 0.35
    ax.bar(x - width/2, clean_contribs, width, label='Clean contribution',
           color=CLEAN_COLOR, alpha=BAR_ALPHA)
    ax.bar(x + width/2, corrupted_contribs, width, label='Corrupted contribution',
           color=CORRUPTED_COLOR, alpha=BAR_ALPHA)

    ax.set_xlabel('Neuron Index', fontsize=12)
    ax.set_ylabel('Contribution (activation * W_out alignment)', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: L{layer} Neuron Contribution Decomposition\n(Position: {format_pos})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in neuron_ids], rotation=90, fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=GRID_ALPHA)
    ax.legend(loc='best')

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_neuron_activation_scatter_2d(
    result: MLPPairResult, format_pos: str, layer: int, output_path: Path
) -> None:
    """Plot 16: Two-dimensional neuron activation scatter.

    X-axis: activation of top neuron
    Y-axis: activation of second neuron
    Two points: clean and corrupted
    """
    lr = result.get_layer_result(format_pos, layer)
    if not lr or len(lr.top_neurons) < 2:
        return

    # Get top 2 neurons by contribution
    neurons = sorted(lr.top_neurons, key=lambda n: abs(n.logit_contribution), reverse=True)[:2]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract activations
    n1, n2 = neurons[0], neurons[1]

    clean_x, clean_y = n1.clean_activation, n2.clean_activation
    corr_x, corr_y = n1.corrupted_activation, n2.corrupted_activation

    # Plot points
    ax.scatter([clean_x], [clean_y], s=200, c=CLEAN_COLOR, marker='o',
              label='Clean', edgecolors='black', linewidths=2, zorder=3)
    ax.scatter([corr_x], [corr_y], s=200, c=CORRUPTED_COLOR, marker='s',
              label='Corrupted', edgecolors='black', linewidths=2, zorder=3)

    # Draw arrow from corrupted to clean
    ax.annotate('', xy=(clean_x, clean_y), xytext=(corr_x, corr_y),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))

    # Add coordinate labels
    ax.annotate(f'({clean_x:.2f}, {clean_y:.2f})', xy=(clean_x, clean_y),
               xytext=(10, 10), textcoords='offset points', fontsize=9)
    ax.annotate(f'({corr_x:.2f}, {corr_y:.2f})', xy=(corr_x, corr_y),
               xytext=(10, -15), textcoords='offset points', fontsize=9)

    ax.set_xlabel(f'Neuron {n1.neuron_idx} activation', fontsize=12)
    ax.set_ylabel(f'Neuron {n2.neuron_idx} activation', fontsize=12)
    ax.set_title(f'Pair {result.pair_idx}: L{layer} Top 2 Neuron Activation Space\n(Position: {format_pos})', fontsize=14)
    ax.legend(loc='best')
    ax.grid(alpha=GRID_ALPHA)

    # Make plot square
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_neuron_consistency_across_pairs(
    agg: MLPAggregatedResults, output_path: Path, format_pos: str,
    layer: int, n_top: int = 30
) -> None:
    """Plot 11: Neuron consistency across pairs.

    Shows which neurons are consistently important (appear in top-N) across
    multiple contrastive pairs. Higher bars indicate more consistent importance.

    X-axis: neuron index
    Y-axis: number of pairs where this neuron appears in top-N
    Color: mean activation difference across pairs
    """
    if agg.n_pairs < 2:
        return

    layers = agg.layers_analyzed
    if not layers:
        return

    # Count how often each neuron appears in top neurons across pairs
    neuron_counts: dict[int, int] = {}
    neuron_mean_diffs: dict[int, list[float]] = {}

    for pr in agg.pair_results:
        lr = pr.get_layer_result(format_pos, layer)
        if not lr or not lr.top_neurons:
            continue
        for ni in lr.top_neurons:
            if ni.neuron_idx not in neuron_counts:
                neuron_counts[ni.neuron_idx] = 0
                neuron_mean_diffs[ni.neuron_idx] = []
            neuron_counts[ni.neuron_idx] += 1
            neuron_mean_diffs[ni.neuron_idx].append(ni.activation_diff)

    if not neuron_counts:
        return

    # Sort by count (consistency) and take top N
    sorted_neurons = sorted(neuron_counts.items(), key=lambda x: x[1], reverse=True)[:n_top]
    neuron_indices = [n[0] for n in sorted_neurons]
    counts = [n[1] for n in sorted_neurons]
    mean_diffs = [np.mean(neuron_mean_diffs[n]) for n in neuron_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(neuron_indices))

    # Color by mean activation difference
    norm = plt.Normalize(vmin=min(mean_diffs), vmax=max(mean_diffs))
    cmap = plt.cm.RdBu_r
    colors = [cmap(norm(d)) for d in mean_diffs]

    bars = ax.bar(x, counts, color=colors, alpha=BAR_ALPHA, edgecolor='black', linewidth=0.5)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Mean Activation Diff (clean - corrupted)', fontsize=10)

    ax.set_xlabel('Neuron Index', fontsize=12)
    ax.set_ylabel(f'Pairs where neuron is in top-{len(agg.pair_results[0].position_results.get(format_pos, [[]])[0].top_neurons) if agg.pair_results and agg.pair_results[0].position_results.get(format_pos) else "N"}', fontsize=12)
    ax.set_title(f'Neuron Consistency Across {agg.n_pairs} Pairs (L{layer})\nHigher = More Consistent Importance (position: {format_pos})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in neuron_indices], rotation=90, fontsize=8)
    ax.set_ylim(0, agg.n_pairs + 0.5)
    ax.axhline(y=agg.n_pairs / 2, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
    ax.grid(axis='y', alpha=GRID_ALPHA)
    ax.legend(loc='upper right')

    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _plot_cross_layer_neuron_patterns(
    agg: MLPAggregatedResults, output_path: Path, format_pos: str,
    target_layer: int, n_top: int = 10
) -> None:
    """Plot 15: Cross-layer neuron patterns.

    Shows correlation/patterns between top neuron activations across different
    layers. Each panel compares activation differences between two layers.

    This helps identify if certain neurons are co-activated across layers,
    suggesting possible information flow pathways.
    """
    layers = agg.layers_analyzed
    if len(layers) < 2:
        return

    # Get top neurons per layer across all pairs
    layer_neuron_data: dict[int, dict[int, list[float]]] = {}  # layer -> neuron -> [act_diffs]

    for layer in layers:
        layer_neuron_data[layer] = {}
        for pr in agg.pair_results:
            lr = pr.get_layer_result(format_pos, layer)
            if not lr or not lr.top_neurons:
                continue
            for ni in lr.top_neurons[:n_top]:
                if ni.neuron_idx not in layer_neuron_data[layer]:
                    layer_neuron_data[layer][ni.neuron_idx] = []
                layer_neuron_data[layer][ni.neuron_idx].append(ni.activation_diff)

    # Create correlation matrix between layers based on mean activation patterns
    n_layers = len(layers)
    corr_matrix = np.zeros((n_layers, n_layers))

    for i, layer_i in enumerate(layers):
        for j, layer_j in enumerate(layers):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue

            # Get mean activation per pair for each layer
            vals_i = []
            vals_j = []
            for pr in agg.pair_results:
                lr_i = pr.get_layer_result(format_pos, layer_i)
                lr_j = pr.get_layer_result(format_pos, layer_j)
                if lr_i and lr_j:
                    vals_i.append(lr_i.total_logit_contribution)
                    vals_j.append(lr_j.total_logit_contribution)

            # Require minimum 5 pairs for meaningful correlation
            if len(vals_i) >= 5:
                corr = np.corrcoef(vals_i, vals_j)[0, 1]
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0
            else:
                corr_matrix[i, j] = np.nan  # Mark insufficient data

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Layer contribution correlation matrix
    im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax1.invert_yaxis()  # Layer 0 at bottom
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Correlation', fontsize=10)

    ax1.set_xticks(np.arange(n_layers))
    ax1.set_yticks(np.arange(n_layers))
    ax1.set_xticklabels([f'L{l}' for l in layers])
    ax1.set_yticklabels([f'L{l}' for l in layers])
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Layer', fontsize=12)
    ax1.set_title('Layer Contribution Correlations\nAcross Pairs', fontsize=12)

    # Add text annotations (note: y-coords inverted by invert_yaxis)
    for i in range(n_layers):
        for j in range(n_layers):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    # Panel 2: Top neuron activation pattern across layers
    # For the target layer, show how its top neurons' importance varies across layers
    target_neurons = _get_top_neurons_for_layer(agg, target_layer, format_pos, n=n_top)
    if target_neurons:
        neuron_indices = [n[0] for n in target_neurons]

        # Build matrix: for each top neuron, see if it's also important at other layers
        pattern_matrix = np.zeros((len(neuron_indices), n_layers))

        for j, layer in enumerate(layers):
            layer_top = _get_top_neurons_for_layer(agg, layer, format_pos, n=100)
            layer_map = {n[0]: n[1] for n in layer_top}  # neuron_idx -> mean_act_diff

            for i, neuron_idx in enumerate(neuron_indices):
                pattern_matrix[i, j] = layer_map.get(neuron_idx, 0)

        vmax = max(abs(pattern_matrix.min()), abs(pattern_matrix.max())) if pattern_matrix.size > 0 else 1
        im2 = ax2.imshow(pattern_matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax2.invert_yaxis()  # Neuron 0 at bottom
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Mean Activation Diff', fontsize=10)

        ax2.set_xticks(np.arange(n_layers))
        ax2.set_yticks(np.arange(len(neuron_indices)))
        ax2.set_xticklabels([f'L{l}' for l in layers])
        ax2.set_yticklabels([f'N{n}' for n in neuron_indices])
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel(f'Neuron (top {n_top} from L{target_layer})', fontsize=12)
        ax2.set_title(f'Top Neuron Patterns Across Layers\n(Neurons selected from L{target_layer})', fontsize=12)

    fig.suptitle(f'Cross-Layer MLP Neuron Patterns (position: {format_pos})', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(None, output_path, dpi=DPI)


def _get_position_label(pos: int, mapping: "SamplePositionMapping | None") -> str:
    """Get semantic label for a position if available, else fall back to P{pos}."""
    if mapping:
        pos_info = mapping.get_position(pos)
        if pos_info and pos_info.format_pos:
            return pos_info.format_pos
    return f"P{pos}"


def _plot_layer_position_heatmaps(
    lp: LayerPositionResult,
    output_dir: Path,
    position_mapping: "SamplePositionMapping | None" = None,
) -> None:
    """Plot layer-position fine patching heatmap for mlp_out.

    Rows: layer, Columns: position
    Color: patching effect
    True 2D localization (not outer product approximation)
    """
    if lp.denoising_grid is None:
        return

    component = lp.component
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
    ax.invert_yaxis()  # Layer 0 at bottom
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
    finalize_plot(output_dir / f"layer_position_{component}_denoising.png")

    # Noising heatmap
    if lp.noising_grid is not None:
        fig, ax = plt.subplots(figsize=(max(10, n_pos * 0.15), max(6, n_layers * 0.3)))
        im = ax.imshow(
            lp.noising_grid,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
        )
        ax.invert_yaxis()  # Layer 0 at bottom
        plt.colorbar(im, ax=ax, label="Noising Disruption")

        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)

        ax.set_xticks(range(0, n_pos, pos_step))
        ax.set_xticklabels([_get_position_label(positions[i], position_mapping) for i in range(0, n_pos, pos_step)], fontsize=8, rotation=45, ha="right")

        ax.set_xlabel("Position", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_title(f"Layer x Position Fine Patching: {component} (Noising)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        finalize_plot(output_dir / f"layer_position_{component}_noising.png")
