"""Attribution patching for binary choices."""

from __future__ import annotations

from typing import Literal

from .attribution_metric import AttributionMetric
from .attribution_settings import AttributionSettings
from .attribution_results import (
    AttributionPatchingResult,
    AttributionSummary,
    AttrPatchTargetResult,
    AttrPatchPairResult,
)
from .attribution_runner import run_all_attribution_methods

from ..binary_choice import BinaryChoiceRunner
from ..common.contrastive_pair import ContrastivePair


def _parse_result_key(key: str) -> tuple[Literal["standard", "eap", "eap_ig"], str]:
    """Parse raw result key into method and component.

    Handles both standard keys (e.g., 'resid', 'eap_attn') and
    grad_at-suffixed keys (e.g., 'resid_clean', 'eap_attn_corrupted').
    """
    # Strip grad_at suffix if present
    base_key = key
    for suffix in ["_clean", "_corrupted"]:
        if key.endswith(suffix):
            base_key = key[:-len(suffix)]
            break

    if base_key in ["resid", "attn", "mlp"]:
        method: Literal["standard", "eap", "eap_ig"] = "standard"
        component = {"resid": "resid_post", "attn": "attn_out", "mlp": "mlp_out"}[base_key]
    elif base_key.startswith("eap_ig_"):
        method = "eap_ig"
        component = base_key.replace("eap_ig_", "") + "_out"
    elif base_key.startswith("eap_"):
        method = "eap"
        component = base_key.replace("eap_", "") + "_out"
    else:
        method = "standard"
        component = "resid_post"
    return method, component


def _build_results(
    raw_results: dict[str, np.ndarray],
    layers: list[int],
) -> dict[str, AttributionPatchingResult]:
    """Convert raw attribution results to result objects."""
    results = {}
    for key, scores in raw_results.items():
        method, component = _parse_result_key(key)
        results[key] = AttributionPatchingResult(
            scores=scores, layers=layers, component=component, method=method
        )
    return results


def attribute_for_choice(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    settings: AttributionSettings | None = None,
    mode: Literal["denoising", "noising"] = "denoising",
) -> AttributionSummary:
    """Compute attribution scores for a contrastive pair.

    Uses generate_trajectory_with_cache_and_grad for gradient computation.

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

    raw_results = run_all_attribution_methods(
        runner=runner,
        pair=pair,
        metric=metric,
        mode=mode,
        methods=settings.methods,
        ig_steps=settings.ig_steps,
        grad_at=settings.grad_at,
    )

    results = _build_results(raw_results, layers)
    return AttributionSummary(results=results, n_pairs=1, mode=mode)


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
