"""Sweep plot visualization for coarse activation patching.

Provides unified plotting for layer and position sweeps with
2x6 subplot grids (denoising on top, noising on bottom).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt

from ....activation_patching import IntervenedChoiceMetrics
from ....activation_patching.act_patch_metrics import DEFAULT_AGGREGATION, PLOT_AGGREGATION_METHODS
from ....activation_patching.coarse import SweepStepResults
from ....common.choice.grouped_binary_choice import ForkAggregation
from ....viz.token_coloring import PairTokenColoring
from .columns import column_core, column_fork, column_logits, column_probs, column_trajectory, column_vocab
from .coarse_helpers import (
    add_boundary_legend,
    add_token_type_legend,
    add_xaxis_boundary_markers,
    color_xaxis_ticks,
    get_tick_spacing,
)


@dataclass
class ExtractionMode:
    """Specifies how to extract metrics from multilabel results."""

    # None = use default aggregation, int = specific fork
    fork_idx: int | None = None
    # None = use default, else use specific method
    method: ForkAggregation | None = None
    # Suffix for output filename
    suffix: str = ""
    # Label pair for title display (e.g., ("a)", "b)"))
    label_pair: tuple[str, str] | None = None

    @classmethod
    def default(cls) -> ExtractionMode:
        """Default extraction (aggregated with default method)."""
        return cls()

    @classmethod
    def for_method(cls, method: ForkAggregation) -> ExtractionMode:
        """Extract using specific aggregation method."""
        return cls(method=method, suffix=f"_{method.value}")

    @classmethod
    def for_fork(cls, fork_idx: int, label_pair: tuple[str, str] | None = None) -> ExtractionMode:
        """Extract for specific fork (label pair)."""
        return cls(fork_idx=fork_idx, suffix=f"_fork{fork_idx}", label_pair=label_pair)

    def get_title_suffix(self) -> str:
        """Get suffix for plot titles."""
        if self.method:
            return f" [{self.method.value}]"
        elif self.fork_idx is not None:
            if self.label_pair:
                return f" [{self.label_pair[0]}/{self.label_pair[1]}]"
            return f" [Fork {self.fork_idx}]"
        return ""


# ============================================================================
#  Data Extraction
# ============================================================================


def _extract_metrics_for_mode(
    sweep_data: SweepStepResults,
    x_values: Sequence[int],
    mode: Literal["denoising", "noising"],
    clean_traj: Literal["short", "long"],
    extraction: ExtractionMode | None = None,
) -> list[IntervenedChoiceMetrics]:
    """Extract metrics for each x value in a sweep.

    Uses cached metrics if available (from pop_heavy), otherwise computes from choice.

    Args:
        sweep_data: Results mapping x values to ActPatchTargetResult
        x_values: Sorted list of x values (layers or positions)
        mode: "denoising" or "noising"
        clean_traj: Which trajectory is considered "clean" ("short" or "long")
        extraction: Optional extraction mode for multilabel (by_method or by_fork)

    Returns:
        List of IntervenedChoiceMetrics, one per x value
    """
    extraction = extraction or ExtractionMode.default()
    metrics = []
    for x in x_values:
        target_result = sweep_data[x]
        # Switch perspective if viewing from long-term's point of view
        if clean_traj == "long":
            target_result = target_result.switch()

        # Extract metrics based on mode
        if extraction.fork_idx is not None:
            # Per-fork extraction
            if mode == "denoising":
                metrics.append(target_result.get_denoising_metrics_per_fork(extraction.fork_idx))
            else:
                metrics.append(target_result.get_noising_metrics_per_fork(extraction.fork_idx))
        elif extraction.method is not None:
            # By-method extraction
            if mode == "denoising":
                metrics.append(target_result.get_denoising_metrics_by_method(extraction.method))
            else:
                metrics.append(target_result.get_noising_metrics_by_method(extraction.method))
        else:
            # Default extraction (uses cached if available)
            if mode == "denoising":
                metrics.append(target_result.get_denoising_metrics())
            else:
                metrics.append(target_result.get_noising_metrics())
    return metrics


# ============================================================================
#  Row Plotting
# ============================================================================


def _plot_sweep_row(
    axes_row: Sequence[plt.Axes],
    x_values: Sequence[int],
    metrics: list[IntervenedChoiceMetrics],
    mode: Literal["denoising", "noising"],
    tick_positions: Sequence[int],
    xlabel: str,
    coloring: PairTokenColoring | None = None,
) -> list[plt.Axes | None]:
    """Plot a single row of sweep metrics (6 columns).

    Args:
        axes_row: 6 axes for this row
        x_values: X-axis values (layers or positions)
        metrics: Metrics for each x value
        mode: "denoising" or "noising"
        tick_positions: X-axis tick positions
        xlabel: X-axis label (e.g., "Layer" or "Position")
        coloring: Optional token coloring for position tick colors

    Returns:
        List of 6 secondary axes (one per column) for y-axis synchronization
    """
    # All columns return their secondary axis for synchronization
    secondary_axes: list[plt.Axes | None] = []

    # Column 0: Core metrics
    secondary_axes.append(column_core.plot(axes_row[0], x_values, metrics, mode, tick_positions, xlabel))

    # Column 1: Probs/Logprobs
    secondary_axes.append(column_probs.plot(axes_row[1], x_values, metrics, tick_positions, xlabel))

    # Column 2: Logits
    secondary_axes.append(column_logits.plot(axes_row[2], x_values, metrics, tick_positions, xlabel))

    # Column 3: Fork metrics
    secondary_axes.append(column_fork.plot(axes_row[3], x_values, metrics, tick_positions, xlabel))

    # Column 4: Vocab metrics
    secondary_axes.append(column_vocab.plot(axes_row[4], x_values, metrics, tick_positions, xlabel))

    # Column 5: Trajectory metrics
    secondary_axes.append(column_trajectory.plot(axes_row[5], x_values, metrics, tick_positions, xlabel))

    # Color x-axis ticks if coloring is provided (for position sweeps)
    # Use tick_positions (already subsampled) not x_values to avoid overwriting
    if coloring:
        for ax in axes_row:
            color_xaxis_ticks(ax, list(tick_positions), coloring)

    return secondary_axes


# ============================================================================
#  Y-Axis Synchronization
# ============================================================================


def _synchronize_y_axes(
    axes: Sequence[Sequence[plt.Axes]],
    secondary_axes_by_row: list[list[plt.Axes | None]],
) -> None:
    """Synchronize y-axes across rows for all columns.

    Both primary (left) and secondary (right) y-axes are synchronized so that
    the same column in different rows shares the same scale.

    Args:
        axes: 2D array of primary axes (rows x cols)
        secondary_axes_by_row: List of [6 secondary axes] for each row
    """
    n_cols = len(axes[0])

    # Synchronize each column's PRIMARY axes between rows
    for col in range(n_cols):
        y_mins = [axes[row][col].get_ylim()[0] for row in range(2)]
        y_maxs = [axes[row][col].get_ylim()[1] for row in range(2)]
        shared_ylim = (min(y_mins), max(y_maxs))
        for row in range(2):
            axes[row][col].set_ylim(shared_ylim)

    # Synchronize each column's SECONDARY axes between rows
    for col in range(n_cols):
        col_sec_axes = [secondary_axes_by_row[row][col] for row in range(2)]
        # Skip if any are None (e.g., logits with no data)
        if None in col_sec_axes:
            continue
        y_mins = [ax.get_ylim()[0] for ax in col_sec_axes]
        y_maxs = [ax.get_ylim()[1] for ax in col_sec_axes]
        shared_ylim = (min(y_mins), max(y_maxs))
        for ax in col_sec_axes:
            ax.set_ylim(shared_ylim)

    # Additionally sync Fork (col 3) and Vocab (col 4) PRIMARY axes together
    fork_vocab_ymins = [axes[r][c].get_ylim()[0] for r in range(2) for c in [3, 4]]
    fork_vocab_ymaxs = [axes[r][c].get_ylim()[1] for r in range(2) for c in [3, 4]]
    fork_vocab_ylim = (min(fork_vocab_ymins), max(fork_vocab_ymaxs))
    for row in range(2):
        axes[row][3].set_ylim(fork_vocab_ylim)
        axes[row][4].set_ylim(fork_vocab_ylim)

    # Additionally sync Fork (col 3) and Vocab (col 4) SECONDARY axes together
    fork_vocab_sec = []
    for row in range(2):
        if secondary_axes_by_row[row][3] is not None:
            fork_vocab_sec.append(secondary_axes_by_row[row][3])
        if secondary_axes_by_row[row][4] is not None:
            fork_vocab_sec.append(secondary_axes_by_row[row][4])
    if fork_vocab_sec:
        sec_ymins = [ax.get_ylim()[0] for ax in fork_vocab_sec]
        sec_ymaxs = [ax.get_ylim()[1] for ax in fork_vocab_sec]
        sec_ylim = (min(sec_ymins), max(sec_ymaxs))
        for ax in fork_vocab_sec:
            ax.set_ylim(sec_ylim)


# ============================================================================
#  Main Plot Functions
# ============================================================================


def plot_layer_sweep(
    layer_data: SweepStepResults,
    output_dir: Path,
    step_size: int,
    clean_traj: Literal["short", "long"],
    component: str = "resid_post",
    extraction: ExtractionMode | None = None,
) -> None:
    """Plot layer sweep with 2x6 subplots (denoising/noising x 6 metric columns).

    Args:
        layer_data: Results mapping layer -> ActPatchTargetResult
        output_dir: Directory to save output
        step_size: Step size used in the sweep
        clean_traj: Which trajectory is "clean" ("short" or "long")
        component: Component being patched (for plot title)
        extraction: Optional extraction mode for multilabel results
    """
    extraction = extraction or ExtractionMode.default()
    layers = sorted(layer_data.keys())
    if not layers:
        return

    # Create figure
    fig, axes = plt.subplots(2, 6, figsize=(52, 18), facecolor="white")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")

    # Title with baseline info and extraction mode
    baseline_info = _get_baseline_info(layer_data, layers[0])
    title_suffix = extraction.get_title_suffix()
    fig.suptitle(
        f"Coarse Layer Sweep [{component}], Clean = {clean_traj}, Steps = {step_size}{title_suffix}\n{baseline_info}",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    tick_positions = layers[:: get_tick_spacing(len(layers))]

    # Adjust spacing: more horizontal padding, room for legends between rows
    fig.subplots_adjust(left=0.045, right=0.96, top=0.91, bottom=0.08, wspace=0.40, hspace=0.48)

    # Plot each row and collect secondary axes for synchronization
    secondary_axes_by_row: list[list[plt.Axes | None]] = []
    for row_idx, mode in enumerate(["denoising", "noising"]):
        metrics = _extract_metrics_for_mode(layer_data, layers, mode, clean_traj, extraction)
        row_secondary = _plot_sweep_row(
            axes[row_idx], layers, metrics, mode, tick_positions, "Layer"
        )
        secondary_axes_by_row.append(row_secondary)

        # Row label - positioned based on subplot locations
        row_label = "Denoising" if mode == "denoising" else "Noising"
        fig.text(
            0.012,
            0.72 - row_idx * 0.44,
            row_label,
            fontsize=15,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    _synchronize_y_axes(axes, secondary_axes_by_row)

    # Save with extraction suffix
    save_path = output_dir / f"coarse_layer_sweep_{clean_traj}_{step_size}{extraction.suffix}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_position_sweep(
    pos_data: SweepStepResults,
    output_dir: Path,
    step_size: int,
    clean_traj: Literal["short", "long"],
    coloring: PairTokenColoring | None = None,
    component: str = "resid_post",
    extraction: ExtractionMode | None = None,
) -> None:
    """Plot position sweep with 2x6 subplots (denoising/noising x 6 metric columns).

    Args:
        pos_data: Results mapping position -> ActPatchTargetResult
        output_dir: Directory to save output
        step_size: Step size used in the sweep
        clean_traj: Which trajectory is "clean" ("short" or "long")
        coloring: Optional token coloring for position tick colors
        component: Component being patched (for plot title)
        extraction: Optional extraction mode for multilabel results
    """
    extraction = extraction or ExtractionMode.default()
    positions = sorted(pos_data.keys())
    if not positions:
        return

    # Create figure (extra height for bottom legends)
    fig, axes = plt.subplots(2, 6, figsize=(52, 20), facecolor="white")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")

    # Title with extraction mode
    title_suffix = extraction.get_title_suffix()
    fig.suptitle(
        f"Coarse Position Sweep [{component}], Clean = {clean_traj}, Steps = {step_size}{title_suffix}",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    tick_positions = positions[:: get_tick_spacing(len(positions))]

    # Get boundary markers from BOTH trajectories to ensure consistent x-axis
    prompt_boundary = None
    choice_div_pos = None
    all_boundaries: list[int] = []
    if coloring:
        traj_type = "clean" if clean_traj == "short" else "corrupted"
        section_markers = coloring.get_section_markers(traj_type)
        prompt_boundary = section_markers.get("prompt_boundary")
        choice_div_pos = section_markers.get("choice_div_pos")

        for traj in ["clean", "corrupted"]:
            markers = coloring.get_section_markers(traj)
            if markers.get("prompt_boundary") is not None:
                all_boundaries.append(markers["prompt_boundary"])
            if markers.get("choice_div_pos") is not None:
                all_boundaries.append(markers["choice_div_pos"])

    # Calculate x-axis range that includes data AND all boundary markers
    x_min = min(positions)
    x_max = max(positions)
    if all_boundaries:
        x_min = min(x_min, min(all_boundaries))
        x_max = max(x_max, max(all_boundaries))

    # Adjust spacing: extra bottom margin for Token Type and Boundary legends
    fig.subplots_adjust(left=0.045, right=0.96, top=0.91, bottom=0.14, wspace=0.40, hspace=0.48)

    # Plot each row and collect secondary axes for synchronization
    secondary_axes_by_row: list[list[plt.Axes | None]] = []
    for row_idx, mode in enumerate(["denoising", "noising"]):
        metrics = _extract_metrics_for_mode(pos_data, positions, mode, clean_traj, extraction)
        row_secondary = _plot_sweep_row(
            axes[row_idx], positions, metrics, mode, tick_positions, "Position", coloring
        )
        secondary_axes_by_row.append(row_secondary)

        for ax in axes[row_idx]:
            ax.set_xlim(x_min - 1, x_max + 1)
            add_xaxis_boundary_markers(ax, prompt_boundary, choice_div_pos)

        row_label = "Denoising" if mode == "denoising" else "Noising"
        fig.text(
            0.012,
            0.72 - row_idx * 0.44,
            row_label,
            fontsize=15,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    _synchronize_y_axes(axes, secondary_axes_by_row)

    add_token_type_legend(fig)
    add_boundary_legend(fig)

    # Save with extraction suffix
    save_path = output_dir / f"coarse_position_sweep_{clean_traj}_{step_size}{extraction.suffix}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _get_baseline_info(sweep_data: SweepStepResults, first_key: int) -> str:
    """Extract baseline info string from first result."""
    first_result = sweep_data[first_key]
    if first_result.denoising and first_result.denoising.baseline_corrupted:
        orig = first_result.denoising.baseline_corrupted
        lp_diff = orig.divergent_logprobs[0] - orig.divergent_logprobs[1]
        return f"Baseline logprob diff: {lp_diff:.2f}"
    return ""


def get_n_labels_from_sweep(sweep_data: SweepStepResults) -> int:
    """Get number of labels from sweep data."""
    if not sweep_data:
        return 1
    first_result = next(iter(sweep_data.values()))
    return first_result.n_labels


def get_multilabel_extraction_modes(
    n_labels: int,
    label_pairs: tuple[tuple[str, str], ...] | None = None,
) -> tuple[list[ExtractionMode], list[ExtractionMode]]:
    """Get extraction modes for multilabel visualization.

    Args:
        n_labels: Number of label pairs
        label_pairs: Optional tuple of (label_a, label_b) pairs for title display

    Returns:
        Tuple of (by_method_modes, by_fork_modes)
    """
    by_method = [ExtractionMode.for_method(m) for m in PLOT_AGGREGATION_METHODS]
    by_fork = []
    for i in range(n_labels):
        lp = label_pairs[i] if label_pairs and i < len(label_pairs) else None
        by_fork.append(ExtractionMode.for_fork(i, label_pair=lp))
    return by_method, by_fork
