"""Combined attribution runner and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..common.contrastive_pair import ContrastivePair
from ..common.profiler import P
from ..common.patching_types import GradTarget, PatchingMode

from .eap import compute_eap
from .eap_ig import compute_eap_ig
from .embedding_alignment import PaddingStrategy
from .quadrature import QuadratureMethod
from .standard_attribution import compute_attribution

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner
    from .attribution_metric import AttributionMetric


def _run_methods_for_grad_point(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    metric: "AttributionMetric",
    mode: PatchingMode,
    methods: list[str],
    ig_steps: int,
    padding_strategy: PaddingStrategy,
    grad_at: GradTarget,
    quadrature: QuadratureMethod,
) -> dict[str, np.ndarray]:
    """Run attribution methods for a single gradient point."""
    results = {}

    if "standard" in methods:
        with P("standard_attribution"):
            for comp in ["resid_post", "attn_out", "mlp_out"]:
                key = comp.replace("_post", "").replace("_out", "")
                results[key] = compute_attribution(
                    runner, pair, metric, mode, comp, grad_at=grad_at
                )

    if "eap" in methods:
        with P("eap"):
            eap = compute_eap(runner, pair, metric, mode, grad_at=grad_at)
            results["eap_attn"] = eap["attn"]
            results["eap_mlp"] = eap["mlp"]

    if "eap_ig" in methods:
        with P("eap_ig"):
            eap_ig = compute_eap_ig(
                runner, pair, metric, mode, ig_steps, padding_strategy,
                grad_at=grad_at, quadrature=quadrature
            )
            results["eap_ig_attn"] = eap_ig["attn"]
            results["eap_ig_mlp"] = eap_ig["mlp"]
            # Note: aligned_len and pos_maps are metadata, not score arrays
            # They are not included in results to avoid type mismatches

    return results


def run_all_attribution_methods(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    metric: "AttributionMetric",
    mode: PatchingMode,
    methods: list[str] | None = None,
    ig_steps: int = 10,
    padding_strategy: PaddingStrategy = PaddingStrategy.ZERO,
    grad_at: list[GradTarget] | None = None,
    quadrature: list[QuadratureMethod] | None = None,
) -> dict[str, np.ndarray]:
    """Run specified attribution methods and return results.

    Args:
        runner: Model runner
        pair: Contrastive pair with trajectories
        metric: Attribution metric
        mode: "denoising" or "noising"
        methods: Methods to run (default: ["standard", "eap"])
            - "standard": Standard attribution (clean-corrupted)*grad
            - "eap": Edge Attribution Patching
            - "eap_ig": EAP with Integrated Gradients
        ig_steps: Integration steps for EAP-IG
        padding_strategy: How to pad segments for EAP-IG
        grad_at: Where to compute gradients (list of "clean" and/or "corrupted")
        quadrature: Quadrature methods for EAP-IG (list)

    Returns:
        Dict with keys like 'resid', 'attn', 'mlp', 'eap_attn', 'eap_ig_attn', etc.
        Keys are suffixed with "_clean" or "_corrupted" for grad_at variants.
    """
    if methods is None:
        methods = ["standard", "eap"]
    if grad_at is None:
        grad_at = ["clean", "corrupted"]
    if quadrature is None:
        quadrature = [QuadratureMethod.MIDPOINT]

    results = {}
    for point in grad_at:
        for quad in quadrature:
            point_results = _run_methods_for_grad_point(
                runner, pair, metric, mode, methods, ig_steps,
                padding_strategy, point, quad
            )
            # Build suffix based on variants
            suffix_parts = []
            if len(grad_at) > 1:
                suffix_parts.append(point)
            if len(quadrature) > 1:
                suffix_parts.append(quad.value)
            suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
            for key, val in point_results.items():
                results[f"{key}{suffix}"] = val

    return results


def find_top_attributions(
    scores: np.ndarray,
    layers: list[int],
    n_top: int = 5,
) -> list[tuple[int, int, float]]:
    """Find top N attribution scores by absolute value.

    Args:
        scores: Attribution scores [n_layers, seq_len]
        layers: Layer indices corresponding to rows
        n_top: Number of top scores to return

    Returns:
        List of (layer, position, score) tuples sorted by |score|
    """
    flat_indices = np.argsort(np.abs(scores).ravel())[::-1][:n_top]
    return [
        (layers[idx // scores.shape[1]], idx % scores.shape[1], float(scores.ravel()[idx]))
        for idx in flat_indices
    ]
