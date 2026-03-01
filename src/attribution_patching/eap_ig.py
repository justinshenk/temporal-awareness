"""EAP with Integrated Gradients (EAP-IG).

Uses embedding-level interpolation for mathematically correct Integrated Gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from ..common.profiler import P
from ..common.hook_utils import hook_name, hook_filter_for_component, attribution_filter
from ..common.contrastive_pair import ContrastivePair
from ..inference.interventions import interpolate_embeddings

from .trajectory_helpers import get_cache
from .embedding_alignment import PaddingStrategy, align_embeddings

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


def compute_eap_ig(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    metric: "AttributionMetric",
    mode: Literal["denoising", "noising"],
    n_steps: int = 10,
    padding_strategy: PaddingStrategy = PaddingStrategy.ZERO,
    grad_at: Literal["clean", "corrupted"] = "corrupted",
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

    Returns:
        Dict with 'attn' and 'mlp' attribution arrays [n_layers, aligned_len]
    """
    del grad_at  # Unused - EAP-IG integrates along full path
    n_layers = runner.n_layers

    # Determine clean/corrupted based on mode
    clean_traj = pair.long_traj if mode == "denoising" else pair.short_traj
    corrupted_traj = pair.short_traj if mode == "denoising" else pair.long_traj

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
            corrupted_base = runner.generate_trajectory_with_cache(
                corrupted_traj.token_ids, attribution_filter
            )
        corrupted_cache = corrupted_base.internals

    # Initialize accumulators
    aligned_len = aligned.aligned_len
    attn_scores = np.zeros((n_layers, aligned_len))
    mlp_scores = np.zeros((n_layers, aligned_len))

    resid_filter = hook_filter_for_component("resid_post")
    clean_np = aligned.clean_embeds[0].detach().cpu().numpy()
    corrupted_np = aligned.corrupted_embeds[0].detach().cpu().numpy()

    with P("eap_ig_integration"):
        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps
            embed_intervention = interpolate_embeddings(
                source_values=corrupted_np, target_values=clean_np, alpha=alpha,
            )

            interp_traj = runner.generate_trajectory_with_intervention_and_cache(
                [0] * aligned_len, [embed_intervention], names_filter=resid_filter,
            )
            metric_val = metric.compute_raw(interp_traj.full_logits.unsqueeze(0))

            # Collect resid activations for gradient
            resid_acts, resid_layers = [], []
            for layer in range(n_layers):
                act = interp_traj.internals.get(hook_name(layer, "resid_post"))
                if act is not None and act.requires_grad:
                    resid_acts.append(act)
                    resid_layers.append(layer)

            if not resid_acts:
                continue

            grad_list = torch.autograd.grad(
                metric_val, resid_acts, retain_graph=True, allow_unused=True
            )

            # Accumulate attribution scores
            for layer, grad in zip(resid_layers, grad_list):
                if grad is None:
                    continue

                for aligned_idx in range(aligned_len):
                    clean_orig = aligned.clean_pos_map[aligned_idx]
                    corr_orig = aligned.corrupted_pos_map[aligned_idx]
                    if clean_orig is None or corr_orig is None:
                        continue

                    attn_scores[layer, aligned_idx] += _compute_edge_attribution(
                        clean_cache, corrupted_cache, grad, layer, "attn_out",
                        aligned_idx, clean_orig, corr_orig,
                    ) / n_steps

                    mlp_scores[layer, aligned_idx] += _compute_edge_attribution(
                        clean_cache, corrupted_cache, grad, layer, "mlp_out",
                        aligned_idx, clean_orig, corr_orig,
                    ) / n_steps

    return {
        "attn": attn_scores,
        "mlp": mlp_scores,
        "aligned_len": aligned_len,
        "clean_pos_map": aligned.clean_pos_map,
        "corrupted_pos_map": aligned.corrupted_pos_map,
    }
