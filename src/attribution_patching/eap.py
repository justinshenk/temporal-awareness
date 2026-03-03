"""Edge Attribution Patching (EAP): attribute to edges between components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from ..common.profiler import P
from ..common.hook_utils import hook_name, attribution_filter
from ..common.token_positions import build_position_arrays
from ..common.contrastive_pair import ContrastivePair

from .trajectory_helpers import get_caches_for_attribution, get_seq_len
from .vectorized import compute_attribution_vectorized

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner
    from .attribution_metric import AttributionMetric


def _compute_resid_gradients(
    metric: "AttributionMetric",
    grad_logits: torch.Tensor,
    grad_cache: dict,
    n_layers: int,
) -> dict[int, torch.Tensor]:
    """Compute gradients w.r.t. residual stream at each layer."""
    metric_val = metric.compute_raw(grad_logits.unsqueeze(0))

    resid_acts = []
    resid_layers = []
    for layer in range(n_layers):
        name = hook_name(layer, "resid_post")
        act = grad_cache.get(name)
        if act is not None and act.requires_grad:
            resid_acts.append(act)
            resid_layers.append(layer)

    if not resid_acts:
        return {}

    grad_list = torch.autograd.grad(
        metric_val, resid_acts, retain_graph=True, allow_unused=True
    )
    return {
        layer: grad.detach()
        for layer, grad in zip(resid_layers, grad_list)
        if grad is not None
    }


def compute_eap(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    metric: "AttributionMetric",
    mode: Literal["denoising", "noising"],
    grad_at: Literal["clean", "corrupted"] = "corrupted",
) -> dict[str, np.ndarray]:
    """Edge Attribution Patching: attribute to edges between components.

    Computes attribution for:
    - attn_out -> resid (attention contribution to residual)
    - mlp_out -> resid (MLP contribution to residual)

    Args:
        runner: Model runner
        pair: Contrastive pair with trajectories
        metric: Attribution metric
        mode: "denoising" or "noising"
        grad_at: Where to compute gradients ("clean" or "corrupted")

    Returns:
        Dict with 'attn' and 'mlp' attribution arrays [n_layers, seq_len]
    """
    n_layers = runner.n_layers
    pos_mapping = pair.position_mapping.inv() if mode == "denoising" else dict(pair.position_mapping.mapping)

    with P("eap_caches"):
        grad_logits, clean_cache, corr_cache, grad_cache = get_caches_for_attribution(
            runner, pair, mode, attribution_filter, grad_at
        )

    with P("eap_grads"):
        resid_grads = _compute_resid_gradients(metric, grad_logits, grad_cache, n_layers)

    with P("eap_scores"):
        first_hook = hook_name(0, "resid_post")
        clean_len = get_seq_len(clean_cache, first_hook)
        corr_len = get_seq_len(corr_cache, first_hook)

        clean_pos, corr_pos, valid = build_position_arrays(pos_mapping, clean_len, corr_len)

        attn_scores = np.zeros((n_layers, clean_len))
        mlp_scores = np.zeros((n_layers, clean_len))

        for layer in range(n_layers):
            grad = resid_grads.get(layer)
            if grad is None:
                continue

            # Attention edge
            attn_name = hook_name(layer, "attn_out")
            clean_attn = clean_cache.get(attn_name)
            corr_attn = corr_cache.get(attn_name)
            if clean_attn is not None and corr_attn is not None:
                attn_scores[layer] = compute_attribution_vectorized(
                    clean_attn, corr_attn, grad, clean_pos, corr_pos, valid
                )

            # MLP edge
            mlp_name = hook_name(layer, "mlp_out")
            clean_mlp = clean_cache.get(mlp_name)
            corr_mlp = corr_cache.get(mlp_name)
            if clean_mlp is not None and corr_mlp is not None:
                mlp_scores[layer] = compute_attribution_vectorized(
                    clean_mlp, corr_mlp, grad, clean_pos, corr_pos, valid
                )

    return {"attn": attn_scores, "mlp": mlp_scores}
