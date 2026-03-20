"""Attribution runner - computes attribution scores for all configured methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..common.contrastive_pair import ContrastivePair
from ..common.profiler import P
from ..common.patching_types import PatchingMode

from .attribution_key import (
    AttributionKey,
    STANDARD_COMPONENTS,
    EAP_COMPONENTS,
)

# Map component names to their short form in EAP/EAP-IG results
COMPONENT_TO_EAP_KEY = {
    "resid_post": "resid",
    "attn_out": "attn",
    "mlp_out": "mlp",
}
from .attribution_eap import compute_eap
from .eap_ig import compute_eap_ig
from .embedding_alignment import PaddingStrategy
from .attribution_quadrature import QuadratureMethod
from .standard_attribution import compute_attribution

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner
    from .attribution_metric import AttributionMetric


def run_attribution(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    metric: "AttributionMetric",
    mode: PatchingMode,
    methods: list[str] | None = None,
    ig_steps: int = 10,
    padding_strategy: PaddingStrategy = PaddingStrategy.ZERO,
    quadratures: list[QuadratureMethod] | None = None,
) -> dict[AttributionKey, np.ndarray]:
    """Run attribution methods and return results keyed by AttributionKey.

    Args:
        runner: Model runner
        pair: Contrastive pair
        metric: Attribution metric
        mode: "denoising" or "noising"
        methods: Methods to run (default: ["eap_ig"])
        ig_steps: Integration steps for EAP-IG
        padding_strategy: Padding for EAP-IG
        quadratures: Quadrature methods for EAP/EAP-IG

    Returns:
        Dict mapping AttributionKey to scores [n_layers, seq_len]
    """
    if methods is None:
        methods = ["eap_ig"]
    if quadratures is None:
        quadratures = [QuadratureMethod.CHEBYSHEV]

    results: dict[AttributionKey, np.ndarray] = {}

    # Standard: one result per component, no quadrature
    if "standard" in methods:
        with P("standard_attribution"):
            for comp in STANDARD_COMPONENTS:
                key = AttributionKey.standard(comp)
                results[key] = compute_attribution(runner, pair, metric, mode, comp)

    # EAP: per quadrature, all components
    if "eap" in methods:
        with P("eap"):
            for quad in quadratures:
                eap_raw = compute_eap(runner, pair, metric, mode)
                for comp in EAP_COMPONENTS:
                    eap_key = COMPONENT_TO_EAP_KEY[comp]
                    key = AttributionKey.eap(comp, quad.value)
                    results[key] = eap_raw[eap_key]

    # EAP-IG: per quadrature, all components
    if "eap_ig" in methods:
        with P("eap_ig"):
            for quad in quadratures:
                eap_ig_raw = compute_eap_ig(
                    runner, pair, metric, mode,
                    ig_steps, padding_strategy, quadrature=quad,
                )
                for comp in EAP_COMPONENTS:
                    eap_key = COMPONENT_TO_EAP_KEY[comp]
                    key = AttributionKey.eap_ig(comp, quad.value)
                    results[key] = eap_ig_raw[eap_key]

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
