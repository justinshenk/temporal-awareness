"""EAP with Integrated Gradients (EAP-IG).

Uses embedding-level interpolation for mathematically correct Integrated Gradients.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..common.contrastive_pair import ContrastivePair
from ..common.device_utils import clear_gpu_memory
from ..common.hook_utils import attribution_filter, hook_name
from ..common.profiler import P
from ..common.patching_types import PatchingMode
from ..inference.interventions import interpolate_embeddings

from .embedding_alignment import PaddingStrategy, align_embeddings
from .attribution_quadrature import QuadratureMethod, get_quadrature

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner
    from .attribution_metric import AttributionMetric

logger = logging.getLogger(__name__)


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

    # Bounds check to prevent IndexError
    clean_seq_len = clean_act.shape[1] if clean_act.ndim == 3 else clean_act.shape[0]
    corr_seq_len = corr_act.shape[1] if corr_act.ndim == 3 else corr_act.shape[0]
    if clean_orig >= clean_seq_len or corr_orig >= corr_seq_len:
        return 0.0

    c = _get_activation_at_position(clean_act, clean_orig)
    r = _get_activation_at_position(corr_act, corr_orig)
    g = grad[0, aligned_idx, :] if grad.ndim == 3 else grad[aligned_idx, :]

    return torch.sum((c - r) * g).detach().cpu().item()


def compute_eap_ig_from_caches(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    metric: "AttributionMetric",
    mode: PatchingMode,
    clean_cache: dict,
    corr_cache: dict,
    n_steps: int = 10,
    padding_strategy: PaddingStrategy = PaddingStrategy.ZERO,
    quadrature: QuadratureMethod = QuadratureMethod.MIDPOINT,
) -> dict[str, np.ndarray]:
    """EAP-IG using pre-computed base caches.

    Args:
        runner: Model runner
        pair: Contrastive pair
        metric: Attribution metric
        mode: "denoising" or "noising"
        clean_cache: Cached activations from clean trajectory (for difference computation)
        corr_cache: Cached activations from corrupted trajectory (for difference computation)
        n_steps: Integration steps
        padding_strategy: How to pad segments
        quadrature: Quadrature method

    Returns:
        Dict with 'resid', 'attn', 'mlp' attribution arrays [n_layers, aligned_len]
    """
    n_layers = runner.n_layers

    logger.debug(f"EAP-IG: mode={mode}, n_steps={n_steps}, quadrature={quadrature}")

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
        # Free original embeddings after alignment (aligned holds its own copies)
        del clean_embeds, corrupted_embeds

    # Initialize accumulators
    aligned_len = aligned.aligned_len
    logger.debug(f"  aligned_len={aligned_len}")

    # Adjust metric for aligned sequence if needed
    adjusted_metric = metric
    if metric.divergent_position >= aligned_len and metric.divergent_position != -1:
        logger.warning(
            f"EAP-IG: Position {metric.divergent_position} out of bounds for aligned_len={aligned_len}. "
            "Using last position."
        )
        adjusted_metric = replace(metric, divergent_position=-1)

    resid_pre_scores = np.zeros((n_layers, aligned_len))
    resid_mid_scores = np.zeros((n_layers, aligned_len))
    resid_scores = np.zeros((n_layers, aligned_len))
    attn_scores = np.zeros((n_layers, aligned_len))
    mlp_scores = np.zeros((n_layers, aligned_len))

    clean_np = aligned.clean_embeds[0].detach().cpu().numpy()

    # Free GPU tensors immediately after converting to numpy (position maps still needed)
    aligned.clean_embeds = None
    aligned.corrupted_embeds = None
    clear_gpu_memory()

    # Get quadrature nodes and weights
    quad = get_quadrature(n_steps, quadrature, a=0.0, b=1.0)

    with P("eap_ig_integration"):
        print(f"[eap-ig] Starting integration: {n_steps} steps, {n_layers} layers, {aligned_len} positions", flush=True)
        for step_idx in range(n_steps):
            print(f"[eap-ig]   Step {step_idx+1}/{n_steps}...", flush=True)
            alpha = quad.nodes[step_idx]
            weight = quad.weights[step_idx]

            embed_intervention = interpolate_embeddings(
                target_values=clean_np, alpha=alpha,
            )

            interp_traj = runner.compute_trajectory_with_intervention_and_cache(
                [0] * aligned_len, [embed_intervention], names_filter=attribution_filter,
                with_grad=True,
            )
            metric_val = adjusted_metric.compute_raw(interp_traj.full_logits.unsqueeze(0))

            # Collect activations for all components
            components = ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"]
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
                metric_val, all_acts, retain_graph=False, allow_unused=True
            )

            # Organize gradients by component
            component_grads: dict[str, dict[int, torch.Tensor]] = {
                "resid_pre": {}, "resid_mid": {}, "resid_post": {}, "attn_out": {}, "mlp_out": {}
            }
            for (component, layer), grad in zip(all_info, grad_list):
                if grad is not None:
                    component_grads[component][layer] = grad.detach()

            # Accumulate attribution scores with quadrature weights
            for layer in range(n_layers):
                for aligned_idx in range(aligned_len):
                    clean_orig = aligned.clean_pos_map[aligned_idx]
                    corr_orig = aligned.corrupted_pos_map[aligned_idx]
                    if clean_orig is None or corr_orig is None:
                        continue

                    # Residual pre (input to layer) attribution
                    resid_pre_grad = component_grads["resid_pre"].get(layer)
                    if resid_pre_grad is not None:
                        resid_pre_scores[layer, aligned_idx] += weight * _compute_edge_attribution(
                            clean_cache, corr_cache, resid_pre_grad, layer, "resid_pre",
                            aligned_idx, clean_orig, corr_orig,
                        )

                    # Residual mid (after attention, before MLP) attribution
                    resid_mid_grad = component_grads["resid_mid"].get(layer)
                    if resid_mid_grad is not None:
                        resid_mid_scores[layer, aligned_idx] += weight * _compute_edge_attribution(
                            clean_cache, corr_cache, resid_mid_grad, layer, "resid_mid",
                            aligned_idx, clean_orig, corr_orig,
                        )

                    # Residual post (output of layer) attribution
                    resid_grad = component_grads["resid_post"].get(layer)
                    if resid_grad is not None:
                        resid_scores[layer, aligned_idx] += weight * _compute_edge_attribution(
                            clean_cache, corr_cache, resid_grad, layer, "resid_post",
                            aligned_idx, clean_orig, corr_orig,
                        )

                    # Attention attribution
                    attn_grad = component_grads["attn_out"].get(layer)
                    if attn_grad is not None:
                        attn_scores[layer, aligned_idx] += weight * _compute_edge_attribution(
                            clean_cache, corr_cache, attn_grad, layer, "attn_out",
                            aligned_idx, clean_orig, corr_orig,
                        )

                    # MLP attribution
                    mlp_grad = component_grads["mlp_out"].get(layer)
                    if mlp_grad is not None:
                        mlp_scores[layer, aligned_idx] += weight * _compute_edge_attribution(
                            clean_cache, corr_cache, mlp_grad, layer, "mlp_out",
                            aligned_idx, clean_orig, corr_orig,
                        )

            # Clean up ALL intermediate objects each step to prevent memory accumulation
            del embed_intervention, interp_traj, component_grads, metric_val, all_acts, all_info, grad_list
            clear_gpu_memory(aggressive=True)

    # Save values before cleanup
    clean_pos_map = aligned.clean_pos_map
    corrupted_pos_map = aligned.corrupted_pos_map

    del aligned
    clear_gpu_memory()

    return {
        "resid_pre": resid_pre_scores,
        "resid_mid": resid_mid_scores,
        "resid": resid_scores,
        "attn": attn_scores,
        "mlp": mlp_scores,
        "aligned_len": aligned_len,
        "clean_pos_map": clean_pos_map,
        "corrupted_pos_map": corrupted_pos_map,
    }
