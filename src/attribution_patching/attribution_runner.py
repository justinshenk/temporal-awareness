"""Attribution runner - computes attribution scores for all configured methods."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..common.contrastive_pair import ContrastivePair
from ..common.device_utils import clear_gpu_memory
from ..common.hook_utils import attribution_filter
from ..common.profiler import P
from ..common.patching_types import PatchingMode

from .attribution_key import (
    AttributionKey,
    STANDARD_COMPONENTS,
    EAP_COMPONENTS,
)
from .attribution_metric import AttributionMetric
from .trajectory_helpers import get_all_caches

COMPONENT_TO_EAP_KEY = {
    "resid_pre": "resid_pre",
    "resid_mid": "resid_mid",
    "resid_post": "resid",
    "attn_out": "attn",
    "mlp_out": "mlp",
}
from .attribution_eap import compute_eap_from_caches
from .eap_ig import compute_eap_ig_from_caches
from .embedding_alignment import PaddingStrategy
from .attribution_quadrature import QuadratureMethod
from .standard_attribution import compute_attribution_from_caches, _get_grad_at_for_mode

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner


def _compute_logit_diff(logits: torch.Tensor, token_ids: tuple[int, int], position: int) -> float:
    """Compute logit difference at position.

    If position is out of bounds, uses the last position.
    """
    seq_len = logits.shape[1] if logits.ndim == 3 else logits.shape[0]
    if position < 0:
        position = seq_len + position
    # Clamp position to valid range
    if position < 0 or position >= seq_len:
        position = seq_len - 1
    if logits.ndim == 3:
        pos_logits = logits[0, position, :]
    else:
        pos_logits = logits[position, :]
    return float(pos_logits[token_ids[0]] - pos_logits[token_ids[1]])


def run_attribution(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    metric: AttributionMetric,
    mode: PatchingMode,
    methods: list[str] | None = None,
    ig_steps: int = 10,
    padding_strategy: PaddingStrategy = PaddingStrategy.ZERO,
    quadratures: list[QuadratureMethod] | None = None,
) -> dict[AttributionKey, np.ndarray]:
    """Run attribution methods using shared caches.

    Computes caches ONCE and reuses for all methods.

    Args:
        runner: Model runner
        pair: Contrastive pair
        metric: Attribution metric (token_ids and divergent_position)
        mode: "denoising" or "noising"
        methods: Methods to run (default: ["eap_ig"])
        ig_steps: Integration steps for EAP-IG
        padding_strategy: Padding for EAP-IG
        quadratures: Quadrature methods

    Returns:
        Dict mapping AttributionKey to scores [n_layers, seq_len]
    """
    if methods is None:
        methods = ["eap_ig"]
    if quadratures is None:
        quadratures = [QuadratureMethod.CHEBYSHEV]

    grad_at = _get_grad_at_for_mode(mode)

    # Get caches ONCE for all methods - both with gradients enabled
    with P("attribution_caches"):
        clean_logits, corr_logits, clean_cache, corr_cache = get_all_caches(
            runner, pair, mode, attribution_filter
        )

    # Determine which cache has gradients for this mode
    if grad_at == "corrupted":
        grad_logits = corr_logits
        grad_cache = corr_cache
    else:
        grad_logits = clean_logits
        grad_cache = clean_cache

    # Compute logit diffs from caches (no extra forward pass!)
    # Note: get_all_caches swaps trajectories based on mode:
    # - denoising: "clean" uses pair.corrupted_traj, "corrupted" uses pair.clean_traj
    # - noising: "clean" uses pair.clean_traj, "corrupted" uses pair.corrupted_traj
    # So we must use the matching divergent positions
    if mode == "denoising":
        clean_div_pos = pair.corrupted_divergent_position or -1
        corr_div_pos = pair.clean_divergent_position or -1
    else:
        clean_div_pos = pair.clean_divergent_position or -1
        corr_div_pos = pair.corrupted_divergent_position or -1

    clean_diff = _compute_logit_diff(clean_logits, metric.target_token_ids, clean_div_pos)
    corr_diff = _compute_logit_diff(corr_logits, metric.target_token_ids, corr_div_pos)

    # Update metric with computed logit diffs
    metric = replace(
        metric,
        clean_logit_diff=clean_diff,
        corrupted_logit_diff=corr_diff,
    )

    results: dict[AttributionKey, np.ndarray] = {}

    # Standard: one result per component
    if "standard" in methods:
        with P("standard_attribution"):
            for comp in STANDARD_COMPONENTS:
                key = AttributionKey.standard(comp)
                results[key] = compute_attribution_from_caches(
                    runner, pair, metric, mode, comp,
                    grad_logits, clean_cache, corr_cache, grad_cache
                )

    # EAP: all components
    if "eap" in methods:
        with P("eap"):
            for quad in quadratures:
                eap_raw = compute_eap_from_caches(
                    runner, pair, metric, mode,
                    grad_logits, clean_cache, corr_cache, grad_cache
                )
                for comp in EAP_COMPONENTS:
                    eap_key = COMPONENT_TO_EAP_KEY[comp]
                    key = AttributionKey.eap(comp, quad.value)
                    results[key] = eap_raw[eap_key]

    # EAP-IG: uses base caches for difference, but does own integration
    if "eap_ig" in methods:
        with P("eap_ig"):
            for quad in quadratures:
                eap_ig_raw = compute_eap_ig_from_caches(
                    runner, pair, metric, mode,
                    clean_cache, corr_cache,
                    ig_steps, padding_strategy, quadrature=quad,
                )
                for comp in EAP_COMPONENTS:
                    eap_key = COMPONENT_TO_EAP_KEY[comp]
                    key = AttributionKey.eap_ig(comp, quad.value)
                    results[key] = eap_ig_raw[eap_key]

    # Clean up
    del clean_cache, corr_cache, grad_cache, grad_logits, clean_logits, corr_logits
    clear_gpu_memory()

    return results


def find_top_attributions(
    scores: np.ndarray,
    layers: list[int],
    n_top: int = 5,
) -> list[tuple[int, int, float]]:
    """Find top N attribution scores by absolute value."""
    flat_indices = np.argsort(np.abs(scores).ravel())[::-1][:n_top]
    return [
        (
            layers[idx // scores.shape[1]],
            idx % scores.shape[1],
            float(scores.ravel()[idx]),
        )
        for idx in flat_indices
    ]
