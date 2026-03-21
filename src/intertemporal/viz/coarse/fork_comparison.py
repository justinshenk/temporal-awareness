"""Fork comparison visualization for multilabel experiments.

Creates a compact plot comparing key metrics across all forks (label pairs),
showing how different labels affect the model's behavior under patching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ....activation_patching.coarse import SweepStepResults
from .sweep_plots import ExtractionMode


def _extract_fork_metrics(
    sweep_data: SweepStepResults,
    x_values: Sequence[int],
    mode: Literal["denoising", "noising"],
    clean_traj: Literal["short", "long"],
    n_forks: int,
) -> list[dict[str, list[float]]]:
    """Extract key metrics for each fork across all x values.

    Returns:
        List of dicts (one per fork), each mapping metric name to values
    """
    from ....activation_patching import IntervenedChoiceMetrics

    fork_data = []
    for fork_idx in range(n_forks):
        extraction = ExtractionMode.for_fork(fork_idx)
        metrics_list: dict[str, list[float]] = {
            "recovery": [],
            "logit_diff": [],
            "prob_short": [],
            "reciprocal_rank": [],
        }

        for x in x_values:
            target_result = sweep_data[x]
            if clean_traj == "long":
                target_result = target_result.switch()

            if mode == "denoising":
                m = target_result.get_denoising_metrics_per_fork(fork_idx)
            else:
                m = target_result.get_noising_metrics_per_fork(fork_idx)

            metrics_list["recovery"].append(m.recovery if m.recovery is not None else 0.0)
            metrics_list["logit_diff"].append(m.logit_diff if m.logit_diff is not None else 0.0)
            metrics_list["prob_short"].append(m.prob_short if m.prob_short is not None else 0.0)
            metrics_list["reciprocal_rank"].append(
                m.reciprocal_rank_short if m.reciprocal_rank_short is not None else 0.0
            )

        fork_data.append(metrics_list)

    return fork_data


def plot_fork_comparison(
    sweep_data: SweepStepResults,
    output_dir: Path,
    step_size: int,
    clean_traj: Literal["short", "long"],
    label_pairs: tuple[tuple[str, str], ...] | None = None,
    sweep_type: Literal["layer", "position"] = "layer",
    component: str = "resid_post",
) -> None:
    """Create fork comparison plot showing metrics across all forks.

    Creates a 2x4 grid showing key metrics (recovery, logit_diff, prob_short, reciprocal_rank)
    for both denoising and noising modes, with one line per fork.

    Args:
        sweep_data: Results mapping x value to ActPatchTargetResult
        output_dir: Directory to save output
        step_size: Step size for filename
        clean_traj: Which trajectory is "clean"
        label_pairs: Optional tuple of (label_a, label_b) pairs for legend
        sweep_type: "layer" or "position"
        component: Component being patched
    """
    # Get n_forks from first result
    first_result = next(iter(sweep_data.values()))
    n_forks = first_result.n_labels
    if n_forks <= 1:
        return  # No comparison needed for single label

    x_values = sorted(sweep_data.keys())

    # Color palette for forks
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_forks, 10)))

    # Create 2x4 figure (denoising/noising x 4 metrics)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="white")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")
            ax.grid(True, alpha=0.3)

    metric_names = ["recovery", "logit_diff", "prob_short", "reciprocal_rank"]
    metric_labels = ["Recovery", "Logit Diff", "Prob(short)", "Reciprocal Rank"]

    for row_idx, mode in enumerate(["denoising", "noising"]):
        fork_data = _extract_fork_metrics(
            sweep_data, x_values, mode, clean_traj, n_forks
        )

        for col_idx, (metric_name, metric_label) in enumerate(
            zip(metric_names, metric_labels)
        ):
            ax = axes[row_idx, col_idx]

            for fork_idx in range(n_forks):
                values = fork_data[fork_idx][metric_name]
                label = None
                if label_pairs and fork_idx < len(label_pairs):
                    label = f"{label_pairs[fork_idx][0]}/{label_pairs[fork_idx][1]}"
                else:
                    label = f"Fork {fork_idx}"

                ax.plot(
                    x_values,
                    values,
                    color=colors[fork_idx % len(colors)],
                    linewidth=2.0,
                    label=label,
                    marker="o",
                    markersize=4,
                    alpha=0.8,
                )

            ax.set_xlabel(sweep_type.capitalize())
            ax.set_ylabel(metric_label)

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=8)

            mode_label = "Denoising" if mode == "denoising" else "Noising"
            ax.set_title(f"{mode_label}: {metric_label}")

    fig.suptitle(
        f"Fork Comparison [{component}], Clean = {clean_traj}, "
        f"Sweep = {sweep_type}, Steps = {step_size}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save to by_fork directory
    by_fork_dir = output_dir / "by_fork"
    by_fork_dir.mkdir(parents=True, exist_ok=True)
    save_path = by_fork_dir / f"fork_comparison_{clean_traj}_{sweep_type}_{step_size}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")
