"""Attribution patching for binary choices.

Main entry points:
- attribute_for_choice(): Full attribution from ContrastivePair
- attribute_simple(): Quick attribution from raw texts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .attribution_metric import AttributionMetric
from .attribution_target import AttributionTarget
from .attribution_results import (
    AttributionPatchingResult,
    AggregatedAttributionResult,
)
from .attribution_algorithms import run_all_attribution_methods

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner
    from ..common.contrastive_pair import ContrastivePair


def _parse_result_key(key: str) -> tuple[Literal["standard", "eap", "eap_ig"], str]:
    """Parse raw result key into method and component."""
    if key in ["resid", "attn", "mlp"]:
        method: Literal["standard", "eap", "eap_ig"] = "standard"
        component = {"resid": "resid_post", "attn": "attn_out", "mlp": "mlp_out"}[key]
    elif key.startswith("eap_ig_"):
        method = "eap_ig"
        component = key.replace("eap_ig_", "") + "_out"
    elif key.startswith("eap_"):
        method = "eap"
        component = key.replace("eap_", "") + "_out"
    else:
        method = "standard"
        component = "resid_post"
    return method, component


def _build_results(
    raw_results: dict[str, np.ndarray],
    all_layers: list[int],
    requested_layers: list[int],
) -> dict[str, AttributionPatchingResult]:
    """Convert raw attribution results to result objects."""
    filter_layers = requested_layers != all_layers
    results = {}

    for key, scores in raw_results.items():
        if filter_layers:
            layer_indices = [all_layers.index(l) for l in requested_layers if l in all_layers]
            scores = scores[layer_indices, :]
            layers = requested_layers
        else:
            layers = all_layers

        method, component = _parse_result_key(key)
        results[key] = AttributionPatchingResult(
            scores=scores,
            layers=layers,
            component=component,
            method=method,
        )

    return results


def attribute_for_choice(
    runner: "BinaryChoiceRunner",
    contrastive_pair: "ContrastivePair",
    target: AttributionTarget | None = None,
) -> AggregatedAttributionResult:
    """Compute attribution scores for a binary choice.

    This is the main entry point for attribution patching, mirroring
    patch_activation_for_choice() in activation_patching.

    Attribution = (clean - corrupted) * gradient
    Approximates causal effects without actual interventions.

    Args:
        runner: BinaryChoiceRunner with model loaded
        contrastive_pair: Pair with short/long trajectories
        target: Layers/positions/methods to use (default: all)

    Returns:
        AggregatedAttributionResult with scores for each method
    """
    if target is None:
        target = AttributionTarget.all()

    metric = AttributionMetric.from_contrastive_pair(runner, contrastive_pair)
    clean_text = contrastive_pair.long_text
    corrupted_text = contrastive_pair.short_text

    # Position mapping in ContrastivePair maps SHORT -> LONG
    # But core functions expect CLEAN (long) -> CORRUPTED (short)
    # So we need to invert: for each (short, long) pair, store (long, short)
    pm = contrastive_pair.position_mapping
    raw_mapping = dict(pm.mapping) if hasattr(pm, "mapping") else dict(pm)
    pos_mapping = {long_pos: short_pos for short_pos, long_pos in raw_mapping.items()}

    all_layers = list(range(runner.n_layers))
    requested_layers = target.resolve_layers(all_layers)

    raw_results = run_all_attribution_methods(
        runner=runner,
        clean_text=clean_text,
        corrupted_text=corrupted_text,
        metric=metric,
        pos_mapping=pos_mapping,
        ig_steps=target.ig_steps,
        methods=target.methods,
    )

    results = _build_results(raw_results, all_layers, requested_layers)
    return AggregatedAttributionResult(results=results, n_pairs=1)


def attribute_simple(
    runner: "BinaryChoiceRunner",
    clean_text: str,
    corrupted_text: str,
    target_token_ids: tuple[int, int],
    pos_mapping: dict[int, int] | None = None,
    target: AttributionTarget | None = None,
) -> AggregatedAttributionResult:
    """Simple attribution from raw texts.

    Useful for quick experiments without ContrastivePair setup.

    Args:
        runner: BinaryChoiceRunner
        clean_text: Clean input (target behavior)
        corrupted_text: Corrupted input (baseline behavior)
        target_token_ids: (chosen_id, alternative_id) for metric
        pos_mapping: Position mapping (default: identity)
        target: Attribution target specification

    Returns:
        AggregatedAttributionResult
    """
    if target is None:
        target = AttributionTarget.all()

    if pos_mapping is None:
        pos_mapping = {}

    metric = AttributionMetric(target_token_ids=target_token_ids)

    all_layers = list(range(runner.n_layers))
    requested_layers = target.resolve_layers(all_layers)

    raw_results = run_all_attribution_methods(
        runner=runner,
        clean_text=clean_text,
        corrupted_text=corrupted_text,
        metric=metric,
        pos_mapping=pos_mapping,
        ig_steps=target.ig_steps,
        methods=target.methods,
    )

    results = _build_results(raw_results, all_layers, requested_layers)
    return AggregatedAttributionResult(results=results, n_pairs=1)
