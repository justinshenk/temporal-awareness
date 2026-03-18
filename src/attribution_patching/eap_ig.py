"""EAP with Integrated Gradients (EAP-IG).

Uses embedding-level interpolation for mathematically correct Integrated Gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..common.contrastive_pair import ContrastivePair
from ..common.hook_utils import attribution_filter, hook_name
from ..common.profiler import P, profile
from ..common.patching_types import GradTarget, PatchingMode
from ..inference.interventions import interpolate_embeddings

from .trajectory_helpers import get_cache
from .embedding_alignment import PaddingStrategy, align_embeddings
from .quadrature import QuadratureMethod, get_quadrature

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner
    from .attribution_metric import AttributionMetric


def _get_activation_at_position(act: torch.Tensor, pos: int) -> torch.Tensor:
    """Extract activation at a specific position."""
    return act[0, pos, :] if act.ndim == 3 else act[pos, :]


def _compute_edge_attribution(
    clean_cache: dict,
    corrupted_cache: dict,
    grad: torch.Tensor,
    layer: int,
    component: str,
    aligned_idx: int,
    clean_orig: int,
    corr_orig: int,
) -> float:
    """Compute attribution for a single edge at one position."""
    cache_name = hook_name(layer, component)
    clean_act = clean_cache.get(cache_name)
    corr_act = corrupted_cache.get(cache_name)

    if clean_act is None or corr_act is None:
        return 0.0

    c = _get_activation_at_position(clean_act, clean_orig)
    r = _get_activation_at_position(corr_act, corr_orig)
    g = grad[0, aligned_idx, :] if grad.ndim == 3 else grad[aligned_idx, :]

    return torch.sum((c - r) * g).detach().cpu().item()


@profile
def compute_eap_ig(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    metric: "AttributionMetric",
    mode: PatchingMode,
    n_steps: int = 10,
    padding_strategy: PaddingStrategy = PaddingStrategy.ZERO,
    grad_at: GradTarget = "corrupted",
    quadrature: QuadratureMethod = QuadratureMethod.MIDPOINT,
) -> dict[str, np.ndarray]:
    """Edge Attribution Patching with Integrated Gradients.

    True Integrated Gradients: interpolates embeddings, not activations.
    Uses anchor-based alignment to handle different-length sequences.

    Formula: IG = (clean - corrupted) * integral(gradient at interpolated points)

    Note: The grad_at parameter is accepted for API consistency but does not
    affect EAP-IG computation, which integrates gradients along the full path.

    Args:
        runner: Model runner
        pair: Contrastive pair with trajectories
        metric: Attribution metric
        mode: "denoising" or "noising"
        n_steps: Integration steps (higher = more accurate but slower)
        padding_strategy: How to pad segments between anchors
        grad_at: Accepted for API consistency (ignored for EAP-IG)
        quadrature: Quadrature method for numerical integration

    Returns:
        Dict with 'attn' and 'mlp' attribution arrays [n_layers, aligned_len]
    """
    del grad_at  # Unused - EAP-IG integrates along full path
    n_layers = runner.n_layers

    # Determine clean/corrupted based on mode
    clean_traj = pair.corrupted_traj if mode == "denoising" else pair.clean_traj
    corrupted_traj = pair.clean_traj if mode == "denoising" else pair.corrupted_traj

    with P("eap_ig_embeddings"):
        clean_embeds = runner.get_embeddings(clean_traj.token_ids)
        corrupted_embeds = runner.get_embeddings(corrupted_traj.token_ids)
        aligned = align_embeddings(
            clean_embeds, corrupted_embeds, pair.position_mapping,
            padding_strategy=padding_strategy,
        )

    with P("eap_ig_base_activations"):
        _, clean_cache = get_cache(runner, pair, "clean", mode, attribution_filter, with_grad=False)
        with torch.no_grad():
            corrupted_base = runner.compute_trajectory_with_cache(
                corrupted_traj.token_ids, attribution_filter
            )
        corrupted_cache = corrupted_base.internals

    # Initialize accumulators
    aligned_len = aligned.aligned_len
    attn_scores = np.zeros((n_layers, aligned_len))
    mlp_scores = np.zeros((n_layers, aligned_len))

    clean_np = aligned.clean_embeds[0].detach().cpu().numpy()
    corrupted_np = aligned.corrupted_embeds[0].detach().cpu().numpy()

    # Get quadrature nodes and weights
    quad = get_quadrature(n_steps, quadrature, a=0.0, b=1.0)

    with P("eap_ig_integration"):
        for step_idx in range(n_steps):
            alpha = quad.nodes[step_idx]
            weight = quad.weights[step_idx]

            embed_intervention = interpolate_embeddings(
                source_values=corrupted_np, target_values=clean_np, alpha=alpha,
            )

            interp_traj = runner.compute_trajectory_with_intervention_and_cache(
                [0] * aligned_len, [embed_intervention], names_filter=attribution_filter,
            )
            metric_val = metric.compute_raw(interp_traj.full_logits.unsqueeze(0))

            # Collect activations for all components
            components = ["attn_out", "mlp_out"]
            all_acts = []
            all_info = []  # (component, layer) tuples

            for component in components:
                for layer in range(n_layers):
                    act = interp_traj.internals.get(hook_name(layer, component))
                    if act is not None and act.requires_grad:
                        all_acts.append(act)
                        all_info.append((component, layer))

            if not all_acts:
                continue

            grad_list = torch.autograd.grad(
                metric_val, all_acts, retain_graph=True, allow_unused=True
            )

            # Organize gradients by component
            component_grads: dict[str, dict[int, torch.Tensor]] = {"attn_out": {}, "mlp_out": {}}
            for (component, layer), grad in zip(all_info, grad_list):
                if grad is not None:
                    component_grads[component][layer] = grad

            # Accumulate attribution scores with quadrature weights
            for layer in range(n_layers):
                for aligned_idx in range(aligned_len):
                    clean_orig = aligned.clean_pos_map[aligned_idx]
                    corr_orig = aligned.corrupted_pos_map[aligned_idx]
                    if clean_orig is None or corr_orig is None:
                        continue

                    # Attention attribution using attn_out gradient
                    attn_grad = component_grads["attn_out"].get(layer)
                    if attn_grad is not None:
                        attn_scores[layer, aligned_idx] += weight * _compute_edge_attribution(
                            clean_cache, corrupted_cache, attn_grad, layer, "attn_out",
                            aligned_idx, clean_orig, corr_orig,
                        )

                    # MLP attribution using mlp_out gradient
                    mlp_grad = component_grads["mlp_out"].get(layer)
                    if mlp_grad is not None:
                        mlp_scores[layer, aligned_idx] += weight * _compute_edge_attribution(
                            clean_cache, corrupted_cache, mlp_grad, layer, "mlp_out",
                            aligned_idx, clean_orig, corr_orig,
                        )

    return {
        "attn": attn_scores,
        "mlp": mlp_scores,
        "aligned_len": aligned_len,
        "clean_pos_map": aligned.clean_pos_map,
        "corrupted_pos_map": aligned.corrupted_pos_map,
    }
