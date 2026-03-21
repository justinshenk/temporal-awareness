"""Data extraction for aggregated visualization.

Extracts metrics across pairs and step sizes for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .....activation_patching.coarse import CoarseActPatchAggregatedResults
from .....activation_patching.act_patch_metrics import LabelPerspective
from .style import COLUMN_METRICS


@dataclass
class MetricSeries:
    """Single metric aggregated across pairs.

    Stores per-pair series aligned to a common x-axis grid.
    """

    metric_name: str
    x_values: list[int] = field(default_factory=list)
    per_pair_series: list[list[float | None]] = field(default_factory=list)
    mean_series: list[float] = field(default_factory=list)

    def compute_mean(self) -> None:
        """Compute mean_series from per_pair_series."""
        n_points = len(self.x_values)
        self.mean_series = []
        for x_idx in range(n_points):
            values = [
                series[x_idx]
                for series in self.per_pair_series
                if series[x_idx] is not None
            ]
            self.mean_series.append(sum(values) / len(values) if values else 0.0)


@dataclass
class ColumnData:
    """All metrics for one column (e.g., vocab)."""

    column_name: str
    metrics: list[MetricSeries] = field(default_factory=list)


def extract_all_columns(
    agg_results: CoarseActPatchAggregatedResults,
    sweep_type: Literal["layer", "position"],
    clean_traj: Literal["short", "long"],
    mode: Literal["denoising", "noising"],
    label_perspective: LabelPerspective = "clean",
) -> dict[str, ColumnData]:
    """Extract metric data for ALL columns at once.

    This is more efficient than calling extract_column_data repeatedly
    because it only iterates through samples once.

    Args:
        agg_results: Aggregated coarse patching results
        sweep_type: "layer" or "position"
        clean_traj: "short" or "long" - which is treated as clean
        mode: "denoising" or "noising"
        label_perspective: Which label system to use for metrics

    Returns:
        Dict mapping column_name to ColumnData
    """
    step_sizes = (
        agg_results.layer_step_sizes
        if sweep_type == "layer"
        else agg_results.position_step_sizes
    )

    # Single pass: collect x-values AND cache sweep_results with their metrics
    all_x_values: set[int] = set()
    cached_metrics: list[dict[int, object]] = []  # List of {x: metrics} for each sample+step

    for sample_id, result in agg_results.by_sample.items():
        for step_size in step_sizes:
            sweep_results = (
                result.get_layer_results_for_step(step_size)
                if sweep_type == "layer"
                else result.get_position_results_for_step(step_size)
            )
            if not sweep_results:
                continue

            all_x_values.update(sweep_results.keys())

            # Pre-compute metrics for each x value
            metrics_for_sweep: dict[int, object] = {}
            for x, target_result in sweep_results.items():
                if clean_traj == "short":
                    if mode == "denoising":
                        metrics = target_result.get_denoising_metrics(label_perspective)
                    else:
                        metrics = target_result.get_noising_metrics(label_perspective)
                else:
                    switched = target_result.switch()
                    if mode == "denoising":
                        metrics = switched.get_denoising_metrics(label_perspective)
                    else:
                        metrics = switched.get_noising_metrics(label_perspective)
                metrics_for_sweep[x] = metrics
            cached_metrics.append(metrics_for_sweep)

    x_values = sorted(all_x_values)
    if not x_values:
        return {col: ColumnData(column_name=col) for col in COLUMN_METRICS}

    # Initialize all columns
    all_columns: dict[str, ColumnData] = {}
    for column_name, metric_names in COLUMN_METRICS.items():
        metric_series_map = {
            name: MetricSeries(metric_name=name, x_values=x_values)
            for name in metric_names
        }
        all_columns[column_name] = ColumnData(column_name=column_name)
        all_columns[column_name]._metric_map = metric_series_map  # Temporary storage

    # Extract all metrics using cached data
    for metrics_for_sweep in cached_metrics:
        for column_name, metric_names in COLUMN_METRICS.items():
            for metric_name in metric_names:
                series: list[float | None] = []
                for x in x_values:
                    metrics = metrics_for_sweep.get(x)
                    if metrics is None:
                        series.append(None)
                    else:
                        value = getattr(metrics, metric_name, None)
                        series.append(value)
                all_columns[column_name]._metric_map[metric_name].per_pair_series.append(series)

    # Compute means and finalize
    for column_name in all_columns:
        metric_map = all_columns[column_name]._metric_map
        for metric_series in metric_map.values():
            metric_series.compute_mean()
        all_columns[column_name].metrics = list(metric_map.values())
        del all_columns[column_name]._metric_map

    return all_columns


def extract_column_data(
    agg_results: CoarseActPatchAggregatedResults,
    column_name: str,
    sweep_type: Literal["layer", "position"],
    clean_traj: Literal["short", "long"],
    mode: Literal["denoising", "noising"],
    label_perspective: LabelPerspective = "clean",
) -> ColumnData:
    """Extract metric data for a column, aggregated across pairs and step sizes.

    Args:
        agg_results: Aggregated coarse patching results
        column_name: Column name (core, probs, logits, fork, vocab, trajectory)
        sweep_type: "layer" or "position"
        clean_traj: "short" or "long" - which is treated as clean
        mode: "denoising" or "noising"
        label_perspective: Which label system to use for metrics:
            - "clean": Use clean labels (default)
            - "corrupted": Use corrupted labels
            - "combined": Aggregate across both label systems

    Returns:
        ColumnData with metrics aligned to common x-axis
    """
    # Get metric names for this column
    metric_names = list(COLUMN_METRICS.get(column_name, []))

    step_sizes = (
        agg_results.layer_step_sizes
        if sweep_type == "layer"
        else agg_results.position_step_sizes
    )

    # Single pass: collect x-values AND cache sweep_results
    all_x_values: set[int] = set()
    cached_sweeps: list[dict] = []  # List of (sweep_results) for each sample+step combo

    for sample_id, result in agg_results.by_sample.items():
        for step_size in step_sizes:
            sweep_results = (
                result.get_layer_results_for_step(step_size)
                if sweep_type == "layer"
                else result.get_position_results_for_step(step_size)
            )
            if sweep_results:
                all_x_values.update(sweep_results.keys())
                cached_sweeps.append(sweep_results)

    x_values = sorted(all_x_values)
    if not x_values:
        return ColumnData(column_name=column_name)

    # Initialize metric series
    metric_series_map: dict[str, MetricSeries] = {
        name: MetricSeries(metric_name=name, x_values=x_values)
        for name in metric_names
    }

    # Extract data using cached sweep_results (no duplicate lookups)
    for sweep_results in cached_sweeps:
        # Extract all metrics for this sweep in one pass
        for metric_name in metric_names:
            series: list[float | None] = []
            for x in x_values:
                target_result = sweep_results.get(x)
                if target_result is None:
                    series.append(None)
                    continue

                # Get metrics for the appropriate perspective
                if clean_traj == "short":
                    if mode == "denoising":
                        metrics = target_result.get_denoising_metrics(label_perspective)
                    else:
                        metrics = target_result.get_noising_metrics(label_perspective)
                else:
                    switched = target_result.switch()
                    if mode == "denoising":
                        metrics = switched.get_denoising_metrics(label_perspective)
                    else:
                        metrics = switched.get_noising_metrics(label_perspective)

                value = getattr(metrics, metric_name, None)
                series.append(value)

            metric_series_map[metric_name].per_pair_series.append(series)

    # Compute means
    for metric_series in metric_series_map.values():
        metric_series.compute_mean()

    return ColumnData(
        column_name=column_name,
        metrics=list(metric_series_map.values()),
    )
