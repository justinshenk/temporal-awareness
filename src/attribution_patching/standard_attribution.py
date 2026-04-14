"""Standard attribution patching: (clean - corrupted) * grad."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..common.contrastive_pair import ContrastivePair
from ..common.hook_utils import hook_names_for_layers
from ..common.profiler import P
from ..common.token_positions import build_position_arrays
from ..common.patching_types import GradTarget, PatchingMode

from .trajectory_helpers import get_seq_len
from .attribution_vectorized import compute_attribution_vectorized

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner
    from .attribution_metric import AttributionMetric

logger = logging.getLogger(__name__)


def _get_grad_at_for_mode(mode: PatchingMode) -> GradTarget:
    """Determine gradient computation point from mode.

    - noising: grad@clean (gradients at clean/source state)
    - denoising: grad@corrupted (gradients at corrupted/source state)
    """
    return "clean" if mode == "noising" else "corrupted"


def _compute_gradients(
    metric: "AttributionMetric",
    grad_logits: torch.Tensor,
    grad_cache: dict,
    grad_at: GradTarget,
) -> dict[str, torch.Tensor]:
    """Compute gradients of metric w.r.t. cached activations."""
    seq_len = grad_logits.shape[0] if grad_logits.ndim == 2 else grad_logits.shape[1]
    logger.debug(
        f"Computing gradients: grad_at={grad_at}, seq_len={seq_len}, metric_pos={metric.divergent_position}"
    )

    # Adjust position if out of bounds for this trajectory
    adjusted_metric = metric
    if metric.divergent_position >= seq_len:
        logger.warning(
            f"Position {metric.divergent_position} out of bounds for seq_len={seq_len} "
            f"(grad_at={grad_at}). Using last position."
        )
        adjusted_metric = replace(metric, divergent_position=-1)

    metric_val = adjusted_metric.compute_raw(grad_logits.unsqueeze(0))

    acts_with_grad = [
        (name, act) for name, act in grad_cache.items() if act.requires_grad
    ]
    if not acts_with_grad:
        return {}

    names, acts = zip(*acts_with_grad)
    grad_list = torch.autograd.grad(
        metric_val, acts, retain_graph=True, allow_unused=True
    )
    return {
        name: grad.detach() for name, grad in zip(names, grad_list) if grad is not None
    }


def compute_attribution_from_caches(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    metric: "AttributionMetric",
    mode: PatchingMode,
    component: str,
    grad_logits: torch.Tensor,
    clean_cache: dict,
    corr_cache: dict,
    grad_cache: dict,
) -> np.ndarray:
    """Standard attribution using pre-computed caches.

    Args:
        runner: Model runner
        pair: Contrastive pair
        metric: Attribution metric
        mode: "denoising" or "noising"
        component: Component to analyze
        grad_logits: Logits from the gradient trajectory
        clean_cache: Cached activations from clean trajectory
        corr_cache: Cached activations from corrupted trajectory
        grad_cache: Cache with requires_grad=True (reference to clean or corr)

    Returns:
        Attribution scores [n_layers, seq_len]
    """
    n_layers = runner.n_layers
    grad_at = _get_grad_at_for_mode(mode)

    pos_mapping = (
        pair.position_mapping.inv()
        if mode == "denoising"
        else dict(pair.position_mapping.mapping)
    )

    logger.debug(
        f"Standard attribution: mode={mode}, component={component}, grad_at={grad_at}"
    )

    with P("attr_grads"):
        grads = _compute_gradients(metric, grad_logits, grad_cache, grad_at)

    with P("attr_scores"):
        hook_names = hook_names_for_layers(range(n_layers), component)
        first_hook = hook_names[0]
        clean_len = get_seq_len(clean_cache, first_hook)
        corr_len = (
            get_seq_len(corr_cache, first_hook)
            if first_hook in corr_cache
            else clean_len
        )

        clean_pos, corr_pos, valid = build_position_arrays(
            pos_mapping, clean_len, corr_len
        )
        results = np.zeros((n_layers, clean_len))

        for layer in range(n_layers):
            name = hook_names[layer]
            clean_act = clean_cache.get(name)
            corr_act = corr_cache.get(name)
            grad = grads.get(name)

            if clean_act is None or corr_act is None or grad is None:
                continue

            results[layer] = compute_attribution_vectorized(
                clean_act, corr_act, grad, clean_pos, corr_pos, valid
            )

    return results
