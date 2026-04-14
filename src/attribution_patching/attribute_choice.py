"""Attribution patching for binary choices."""

from __future__ import annotations

import numpy as np

from ..common.device_utils import clear_gpu_memory
from ..common.patching_types import PatchingMode
from ..common.profiler import profile
from .attribution_key import AttributionKey
from .attribution_metric import AttributionMetric
from .attribution_settings import AttributionSettings
from .attribution_results import (
    AttributionPatchingResult,
    AttributionSummary,
    AttrPatchTargetResult,
    AttrPatchPairResult,
)
from .attribution_runner import run_attribution

from ..binary_choice import BinaryChoiceRunner
from ..common.contrastive_pair import ContrastivePair


def _convert_to_results(
    raw_results: dict[AttributionKey, np.ndarray],
    layers: list[int],
) -> dict[str, AttributionPatchingResult]:
    """Convert typed results to serializable dict with string keys."""
    results = {}
    for key, scores in raw_results.items():
        # Use str(key) for dict key (human-readable format)
        str_key = str(key)
        results[str_key] = AttributionPatchingResult(
            scores=scores,
            layers=layers,
            component=key.component,
            method=key.method,
        )
    return results


@profile
def attribute_for_choice(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    settings: AttributionSettings | None = None,
    mode: PatchingMode = "denoising",
) -> AttributionSummary:
    """Compute attribution scores for a contrastive pair.

    Args:
        runner: Model runner
        pair: Contrastive pair with trajectories
        settings: Attribution settings
        mode: "denoising" (long->short) or "noising" (short->long)

    Returns:
        AttributionSummary with results for all methods
    """
    if settings is None:
        settings = AttributionSettings.all()

    metric = AttributionMetric.from_contrastive_pair(runner, pair, mode)
    layers = list(range(runner.n_layers))

    raw_results = run_attribution(
        runner=runner,
        pair=pair,
        metric=metric,
        mode=mode,
        methods=settings.methods,
        ig_steps=settings.ig_steps,
        quadratures=settings.quadrature,
    )

    results = _convert_to_results(raw_results, layers)

    clear_gpu_memory()

    return AttributionSummary(results=results, n_pairs=1, mode=mode)


@profile
def attribute_pair(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    settings: AttributionSettings | None = None,
    modes: tuple[str, ...] = ("denoising", "noising"),
) -> AttrPatchPairResult:
    """Run attribution for a contrastive pair in both directions.

    Args:
        runner: Model runner
        pair: Contrastive pair with trajectories
        settings: Attribution settings
        modes: Which modes to run ("denoising", "noising", or both)

    Returns:
        AttrPatchPairResult with denoising and/or noising results
    """
    result = AttrPatchTargetResult(
        denoising=attribute_for_choice(runner, pair, settings, "denoising")
        if "denoising" in modes
        else None,
        noising=attribute_for_choice(runner, pair, settings, "noising")
        if "noising" in modes
        else None,
    )
    return AttrPatchPairResult(sample_id=pair.sample_id, result=result)
