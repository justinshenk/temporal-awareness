"""Standard attribution patching: (clean - corrupted) * grad."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from ..common.profiler import P
from ..common.hook_utils import hook_names_for_layers, hook_filter_for_component
from ..common.token_positions import build_position_arrays
from ..common.contrastive_pair import ContrastivePair

from .trajectory_helpers import get_caches_for_attribution, get_seq_len
from .vectorized import compute_attribution_vectorized

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner
    from .attribution_metric import AttributionMetric


def _compute_gradients(
    metric: "AttributionMetric",
    grad_logits: torch.Tensor,
    grad_cache: dict,
) -> dict[str, torch.Tensor]:
    """Compute gradients of metric w.r.t. cached activations."""
    metric_val = metric.compute_raw(grad_logits.unsqueeze(0))

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
        name: grad.detach()
        for name, grad in zip(names, grad_list)
        if grad is not None
    }


def compute_attribution(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    metric: "AttributionMetric",
    mode: Literal["denoising", "noising"],
    component: str = "resid_post",
    grad_at: Literal["clean", "corrupted"] = "corrupted",
) -> np.ndarray:
    """Standard attribution patching: (clean - corrupted) * grad.

    Args:
        runner: Model runner
        pair: Contrastive pair with trajectories
        metric: Attribution metric
        mode: "denoising" or "noising"
        component: Component to analyze
        grad_at: Where to compute gradients ("clean" or "corrupted")

    Returns:
        Attribution scores [n_layers, seq_len]
    """
    n_layers = runner.n_layers
    hook_filter = hook_filter_for_component(component)

    pos_mapping = pair.position_mapping.inv() if mode == "denoising" else dict(pair.position_mapping.mapping)

    with P("attr_caches"):
        grad_logits, clean_cache, corr_cache, grad_cache = get_caches_for_attribution(
            runner, pair, mode, hook_filter, grad_at
        )

    with P("attr_grads"):
        grads = _compute_gradients(metric, grad_logits, grad_cache)

    with P("attr_scores"):
        hook_names = hook_names_for_layers(range(n_layers), component)
        first_hook = hook_names[0]
        clean_len = get_seq_len(clean_cache, first_hook)
        corr_len = get_seq_len(corr_cache, first_hook) if first_hook in corr_cache else clean_len

        clean_pos, corr_pos, valid = build_position_arrays(pos_mapping, clean_len, corr_len)
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
