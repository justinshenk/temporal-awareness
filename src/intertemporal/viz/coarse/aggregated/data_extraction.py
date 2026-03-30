"""Data extraction for aggregated visualization.

Extracts metrics across pairs and step sizes for plotting.
Supports different aggregation methods and per-fork extraction for multilabel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from .....activation_patching.coarse import CoarseActPatchAggregatedResults
from .....activation_patching.act_patch_metrics import (
    IntervenedChoiceMetrics,
    LabelPerspective,
)
from .....activation_patching.act_patch_results import ActPatchTargetResult
from .....common.choice.grouped_binary_choice import ForkAggregation
from .agg_style import COLUMN_METRICS


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


# Type for metric extraction function
MetricsExtractor = Callable[[ActPatchTargetResult, str], IntervenedChoiceMetrics]


def _make_standard_extractor(
    mode: Literal["denoising", "noising"],
    clean_traj: Literal["short", "long"],
    label_perspective: LabelPerspective = "clean",
) -> MetricsExtractor:
    """Create a standard metrics extractor (using default aggregation)."""

    def extractor(target_result: ActPatchTargetResult, _mode: str) -> IntervenedChoiceMetrics:
        if clean_traj == "long":
            target_result = target_result.switch()
        if mode == "denoising":
            return target_result.get_denoising_metrics(label_perspective)
        return target_result.get_noising_metrics(label_perspective)

    return extractor


def _make_method_extractor(
    mode: Literal["denoising", "noising"],
    clean_traj: Literal["short", "long"],
    method: ForkAggregation,
) -> MetricsExtractor:
    """Create a metrics extractor for a specific aggregation method."""

    def extractor(target_result: ActPatchTargetResult, _mode: str) -> IntervenedChoiceMetrics:
        if clean_traj == "long":
            target_result = target_result.switch()
        if mode == "denoising":
            return target_result.get_denoising_metrics_by_method(method)
        return target_result.get_noising_metrics_by_method(method)

    return extractor


def _make_fork_extractor(
    mode: Literal["denoising", "noising"],
    clean_traj: Literal["short", "long"],
    fork_idx: int,
) -> MetricsExtractor:
    """Create a metrics extractor for a specific fork."""

    def extractor(target_result: ActPatchTargetResult, _mode: str) -> IntervenedChoiceMetrics:
        if clean_traj == "long":
            target_result = target_result.switch()
        if mode == "denoising":
            return target_result.get_denoising_metrics_per_fork(fork_idx)
        return target_result.get_noising_metrics_per_fork(fork_idx)

    return extractor


def _extract_with_extractor(
    agg_results: CoarseActPatchAggregatedResults,
    sweep_type: Literal["layer", "position"],
    mode: Literal["denoising", "noising"],
    extractor: MetricsExtractor,
) -> dict[str, ColumnData]:
    """Extract all columns using a given metrics extractor function."""
    step_sizes = (
        agg_results.layer_step_sizes
        if sweep_type == "layer"
        else agg_results.position_step_sizes
    )

    # Single pass: collect x-values AND cache metrics
    all_x_values: set[int] = set()
    cached_metrics: list[dict[int, IntervenedChoiceMetrics]] = []

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

            metrics_for_sweep: dict[int, IntervenedChoiceMetrics] = {}
            for x, target_result in sweep_results.items():
                metrics_for_sweep[x] = extractor(target_result, mode)
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
        all_columns[column_name]._metric_map = metric_series_map

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


def extract_all_columns(
    agg_results: CoarseActPatchAggregatedResults,
    sweep_type: Literal["layer", "position"],
    clean_traj: Literal["short", "long"],
    mode: Literal["denoising", "noising"],
    label_perspective: LabelPerspective = "clean",
) -> dict[str, ColumnData]:
    """Extract metric data for ALL columns at once (standard extraction).

    For multilabel, uses the default aggregation method (MEAN_NORMALIZED).
    All metrics are internally consistent.

    Args:
        agg_results: Aggregated coarse patching results
        sweep_type: "layer" or "position"
        clean_traj: "short" or "long" - which is treated as clean
        mode: "denoising" or "noising"
        label_perspective: "clean", "corrupted", or "combined"

    Returns:
        Dict mapping column_name to ColumnData
    """
    extractor = _make_standard_extractor(mode, clean_traj, label_perspective)
    return _extract_with_extractor(agg_results, sweep_type, mode, extractor)


def extract_all_columns_by_method(
    agg_results: CoarseActPatchAggregatedResults,
    sweep_type: Literal["layer", "position"],
    clean_traj: Literal["short", "long"],
    mode: Literal["denoising", "noising"],
    method: ForkAggregation,
) -> dict[str, ColumnData]:
    """Extract metric data using a specific aggregation method.

    For multilabel, uses the specified aggregation method.
    All metrics are internally consistent for that method.

    Args:
        agg_results: Aggregated coarse patching results
        sweep_type: "layer" or "position"
        clean_traj: "short" or "long" - which is treated as clean
        mode: "denoising" or "noising"
        method: ForkAggregation method to use

    Returns:
        Dict mapping column_name to ColumnData
    """
    extractor = _make_method_extractor(mode, clean_traj, method)
    return _extract_with_extractor(agg_results, sweep_type, mode, extractor)


def extract_all_columns_per_fork(
    agg_results: CoarseActPatchAggregatedResults,
    sweep_type: Literal["layer", "position"],
    clean_traj: Literal["short", "long"],
    mode: Literal["denoising", "noising"],
    fork_idx: int,
) -> dict[str, ColumnData]:
    """Extract metric data for a specific fork (label pair).

    For multilabel, extracts metrics from only the specified fork.
    All metrics are internally consistent for that fork.

    Args:
        agg_results: Aggregated coarse patching results
        sweep_type: "layer" or "position"
        clean_traj: "short" or "long" - which is treated as clean
        mode: "denoising" or "noising"
        fork_idx: Index of the fork to extract

    Returns:
        Dict mapping column_name to ColumnData
    """
    extractor = _make_fork_extractor(mode, clean_traj, fork_idx)
    return _extract_with_extractor(agg_results, sweep_type, mode, extractor)
