"""Core attribution patching algorithms.

Implements:
- Standard attribution: (clean - corrupted) * gradient
- EAP (Edge Attribution Patching): attributes to component edges
- EAP-IG: EAP with Integrated Gradients for accuracy

IMPORTANT DESIGN PRINCIPLES:
1. NEVER use any backend API directly - ALWAYS use ModelRunner API
2. All functionality must work identically across all backends
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..common.profiler import P
from ..inference.interventions import patch

if TYPE_CHECKING:
    from ..inference import ModelRunner
    from .attribution_metric import AttributionMetric


# =============================================================================
# Hook Filters
# =============================================================================


def _attribution_filter(n: str) -> bool:
    """Filter for capturing resid, attn, and mlp activations."""
    return "hook_resid_post" in n or "hook_attn_out" in n or "hook_mlp_out" in n


def _component_filter(component: str):
    """Create filter for a specific component."""
    return lambda n: f"hook_{component}" in n


# =============================================================================
# Position Mapping Utilities
# =============================================================================


def build_position_arrays(
    pos_mapping: dict[int, int], clean_len: int, corr_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build position mapping arrays for vectorized indexing.

    Args:
        pos_mapping: Maps clean positions to corrupted positions
        clean_len: Length of clean sequence
        corr_len: Length of corrupted sequence

    Returns:
        Tuple of:
        - clean_pos: Array of clean position indices [0, 1, ..., clean_len-1]
        - corr_pos: Mapped corrupted positions for each clean position
        - valid: Boolean mask for valid mappings (in bounds)
    """
    clean_pos = np.arange(clean_len)
    corr_pos = np.array([pos_mapping.get(p, p) for p in range(clean_len)])
    # Mask positions that are out of bounds
    valid = corr_pos < corr_len
    return clean_pos, corr_pos, valid


# =============================================================================
# Vectorized Attribution Computation
# =============================================================================


def compute_attribution_vectorized(
    clean_act: torch.Tensor,
    corr_act: torch.Tensor,
    grad: torch.Tensor,
    clean_pos: np.ndarray,
    corr_pos: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Compute attribution scores using vectorized operations.

    Attribution = (clean - corrupted) * gradient

    Args:
        clean_act: Clean activations [batch, seq, hidden]
        corr_act: Corrupted activations [batch, seq, hidden]
        grad: Gradient of metric w.r.t. corrupted activations
        clean_pos: Clean position indices
        corr_pos: Corrupted position indices (mapped from clean)
        valid: Boolean mask for valid positions

    Returns:
        Attribution scores [clean_len]
    """
    clean_len = len(clean_pos)
    scores = np.zeros(clean_len)

    # Get valid positions
    valid_clean = clean_pos[valid]
    valid_corr = corr_pos[valid]

    if len(valid_clean) == 0:
        return scores

    # Vectorized: diff = clean[valid_clean] - corr[valid_corr]
    clean_acts = clean_act[0, valid_clean, :]  # (n_valid, d_model)
    corr_acts = corr_act[0, valid_corr, :]  # (n_valid, d_model)
    diff = clean_acts - corr_acts

    # Get gradients at corrupted positions
    if grad.ndim == 3:
        grads = grad[0, valid_corr, :]
    else:
        grads = grad[valid_corr, :]

    # Dot product along feature dimension
    attr = torch.sum(diff * grads, dim=1).detach().cpu().numpy()
    scores[valid] = attr

    return scores


# =============================================================================
# Standard Attribution
# =============================================================================


def compute_attribution(
    runner: "ModelRunner",
    clean_text: str,
    corrupted_text: str,
    metric: "AttributionMetric",
    pos_mapping: dict[int, int],
    component: str = "resid_post",
) -> np.ndarray:
    """Standard attribution patching: (clean - corrupted) * grad.

    Args:
        runner: Model runner
        clean_text: Clean input (target behavior)
        corrupted_text: Corrupted input (baseline behavior)
        metric: Attribution metric
        pos_mapping: Clean to corrupted position mapping
        component: Component to analyze

    Returns:
        Attribution scores [n_layers, seq_len]
    """
    n_layers = runner.n_layers
    hook_filter = _component_filter(component)

    with P("attr_clean_cache"):
        with torch.no_grad():
            _, clean_cache = runner.run_with_cache(clean_text, names_filter=hook_filter)

    with P("attr_corr_cache"):
        corr_logits, corr_cache = runner.run_with_cache_and_grad(
            corrupted_text, names_filter=hook_filter
        )

    with P("attr_grads"):
        metric_val = metric.compute_raw(corr_logits)
        # Compute all gradients at once for efficiency
        acts_with_grad = [
            (name, act) for name, act in corr_cache.items() if act.requires_grad
        ]
        if acts_with_grad:
            names, acts = zip(*acts_with_grad)
            grad_list = torch.autograd.grad(
                metric_val, acts, retain_graph=True, allow_unused=True
            )
            grads = {
                name: grad.detach()
                for name, grad in zip(names, grad_list)
                if grad is not None
            }
        else:
            grads = {}

    with P("attr_scores"):
        hook_names = [f"blocks.{l}.hook_{component}" for l in range(n_layers)]
        clean_len = clean_cache[hook_names[0]].shape[1]
        corr_len = (
            corr_cache[hook_names[0]].shape[1]
            if hook_names[0] in corr_cache
            else clean_len
        )

        clean_pos, corr_pos, valid = build_position_arrays(
            pos_mapping, clean_len, corr_len
        )
        results = np.zeros((n_layers, clean_len))

        for l in range(n_layers):
            name = hook_names[l]
            try:
                clean_act = clean_cache[name]
            except KeyError:
                continue
            corr_act = corr_cache.get(name)
            grad = grads.get(name)

            if corr_act is None or grad is None:
                continue

            results[l] = compute_attribution_vectorized(
                clean_act, corr_act, grad, clean_pos, corr_pos, valid
            )

    return results


# =============================================================================
# Edge Attribution Patching (EAP)
# =============================================================================


def compute_eap(
    runner: "ModelRunner",
    clean_text: str,
    corrupted_text: str,
    metric: "AttributionMetric",
    pos_mapping: dict[int, int],
) -> dict[str, np.ndarray]:
    """Edge Attribution Patching: attribute to edges between components.

    Computes attribution for:
    - attn_out -> resid (attention contribution to residual)
    - mlp_out -> resid (MLP contribution to residual)

    Args:
        runner: Model runner
        clean_text: Clean input
        corrupted_text: Corrupted input
        metric: Attribution metric
        pos_mapping: Position mapping

    Returns:
        Dict with 'attn' and 'mlp' attribution arrays [n_layers, seq_len]
    """
    n_layers = runner.n_layers

    with P("eap_clean_cache"):
        with torch.no_grad():
            _, clean_cache = runner.run_with_cache(
                clean_text, names_filter=_attribution_filter
            )

    with P("eap_corr_cache"):
        corr_logits, corr_cache = runner.run_with_cache_and_grad(
            corrupted_text, names_filter=_attribution_filter
        )

    with P("eap_grads"):
        metric_val = metric.compute_raw(corr_logits)
        # Collect residual activations for gradient computation
        resid_acts = []
        resid_layers = []
        for l in range(n_layers):
            name = f"blocks.{l}.hook_resid_post"
            act = corr_cache.get(name)
            if act is not None and act.requires_grad:
                resid_acts.append(act)
                resid_layers.append(l)

        resid_grads = {}
        if resid_acts:
            grad_list = torch.autograd.grad(
                metric_val, resid_acts, retain_graph=True, allow_unused=True
            )
            for l, grad in zip(resid_layers, grad_list):
                if grad is not None:
                    resid_grads[l] = grad.detach()

    with P("eap_scores"):
        first_hook = "blocks.0.hook_resid_post"
        clean_len = clean_cache[first_hook].shape[1]
        corr_len = corr_cache[first_hook].shape[1]
        clean_pos, corr_pos, valid = build_position_arrays(
            pos_mapping, clean_len, corr_len
        )

        attn_scores = np.zeros((n_layers, clean_len))
        mlp_scores = np.zeros((n_layers, clean_len))

        for l in range(n_layers):
            grad = resid_grads.get(l)
            if grad is None:
                continue

            # Attention edge
            attn_name = f"blocks.{l}.hook_attn_out"
            clean_attn = clean_cache.get(attn_name)
            corr_attn = corr_cache.get(attn_name)
            if clean_attn is not None and corr_attn is not None:
                attn_scores[l] = compute_attribution_vectorized(
                    clean_attn, corr_attn, grad, clean_pos, corr_pos, valid
                )

            # MLP edge
            mlp_name = f"blocks.{l}.hook_mlp_out"
            clean_mlp = clean_cache.get(mlp_name)
            corr_mlp = corr_cache.get(mlp_name)
            if clean_mlp is not None and corr_mlp is not None:
                mlp_scores[l] = compute_attribution_vectorized(
                    clean_mlp, corr_mlp, grad, clean_pos, corr_pos, valid
                )

    return {"attn": attn_scores, "mlp": mlp_scores}


# =============================================================================
# EAP with Integrated Gradients (EAP-IG)
# =============================================================================


def compute_eap_ig(
    runner: "ModelRunner",
    clean_text: str,
    corrupted_text: str,
    metric: "AttributionMetric",
    pos_mapping: dict[int, int],
    n_steps: int = 10,
) -> dict[str, np.ndarray]:
    """Edge Attribution Patching with Integrated Gradients.

    Integrates gradients along path from corrupted to clean:
        IG = (clean - corrupted) * integral(gradient at interpolated points)

    More accurate than standard EAP but slower.

    Args:
        runner: Model runner
        clean_text: Clean input
        corrupted_text: Corrupted input
        metric: Attribution metric
        pos_mapping: Position mapping
        n_steps: Integration steps (higher = more accurate but slower)

    Returns:
        Dict with 'attn' and 'mlp' attribution arrays
    """
    n_layers = runner.n_layers

    with P("eap_ig_cache"):
        with torch.no_grad():
            _, clean_cache = runner.run_with_cache(
                clean_text, names_filter=_attribution_filter
            )
            _, corr_cache_base = runner.run_with_cache(
                corrupted_text, names_filter=_attribution_filter
            )

    first_hook = "blocks.0.hook_resid_post"
    clean_len = clean_cache[first_hook].shape[1]
    corr_len = corr_cache_base[first_hook].shape[1]
    clean_pos, corr_pos, valid = build_position_arrays(
        pos_mapping, clean_len, corr_len
    )

    attn_grads_sum = np.zeros((n_layers, clean_len))
    mlp_grads_sum = np.zeros((n_layers, clean_len))

    with P("eap_ig_integration"):
        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps

            # Build position-aware interpolation interventions
            interventions = []
            for layer in range(n_layers):
                for comp in ["resid_post", "attn_out", "mlp_out"]:
                    hook_name = f"blocks.{layer}.hook_{comp}"
                    if hook_name not in clean_cache or hook_name not in corr_cache_base:
                        continue

                    clean_act = clean_cache[hook_name]
                    corr_act = corr_cache_base[hook_name]

                    # Convert to numpy
                    if isinstance(clean_act, torch.Tensor):
                        clean_np = clean_act[0].detach().cpu().numpy()
                    else:
                        clean_np = np.array(clean_act[0])

                    if isinstance(corr_act, torch.Tensor):
                        corr_np = corr_act[0].detach().cpu().numpy()
                    else:
                        corr_np = np.array(corr_act[0])

                    # Create interpolation interventions for valid positions
                    for i in range(len(clean_pos)):
                        if not valid[i]:
                            continue
                        cp = clean_pos[i]
                        rp = corr_pos[i]

                        if cp >= clean_np.shape[0] or rp >= corr_np.shape[0]:
                            continue

                        # Interpolated value: corr + alpha * (clean - corr)
                        interp_value = corr_np[rp] + alpha * (
                            clean_np[cp] - corr_np[rp]
                        )

                        interventions.append(
                            patch(
                                layer=layer,
                                values=interp_value,
                                positions=int(rp),
                                component=comp,
                            )
                        )

            # Run with interventions and get gradients
            logits, cache = runner.run_with_intervention_and_cache(
                corrupted_text,
                interventions,
                names_filter=lambda n: "hook_resid_post" in n,
            )

            metric_val = metric.compute_raw(logits)

            # Collect activations for gradient computation
            resid_acts = []
            resid_layers = []
            for layer in range(n_layers):
                resid_name = f"blocks.{layer}.hook_resid_post"
                act = cache.get(resid_name)
                if act is not None and act.requires_grad:
                    resid_acts.append(act)
                    resid_layers.append(layer)

            if not resid_acts:
                continue

            # Compute gradients
            grad_list = torch.autograd.grad(
                metric_val, resid_acts, retain_graph=True, allow_unused=True
            )

            # Accumulate attribution scores
            for layer, grad in zip(resid_layers, grad_list):
                if grad is None:
                    continue

                # Attention edge
                attn_name = f"blocks.{layer}.hook_attn_out"
                clean_attn = clean_cache.get(attn_name)
                corr_attn = corr_cache_base.get(attn_name)
                if clean_attn is not None and corr_attn is not None:
                    attr = compute_attribution_vectorized(
                        clean_attn, corr_attn, grad, clean_pos, corr_pos, valid
                    )
                    attn_grads_sum[layer] += attr / n_steps

                # MLP edge
                mlp_name = f"blocks.{layer}.hook_mlp_out"
                clean_mlp = clean_cache.get(mlp_name)
                corr_mlp = corr_cache_base.get(mlp_name)
                if clean_mlp is not None and corr_mlp is not None:
                    attr = compute_attribution_vectorized(
                        clean_mlp, corr_mlp, grad, clean_pos, corr_pos, valid
                    )
                    mlp_grads_sum[layer] += attr / n_steps

    return {"attn": attn_grads_sum, "mlp": mlp_grads_sum}


# =============================================================================
# Combined Attribution Runner
# =============================================================================


def run_all_attribution_methods(
    runner: "ModelRunner",
    clean_text: str,
    corrupted_text: str,
    metric: "AttributionMetric",
    pos_mapping: dict[int, int],
    ig_steps: int = 10,
    methods: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Run specified attribution methods and return results.

    Args:
        runner: Model runner
        clean_text: Clean input
        corrupted_text: Corrupted input
        metric: Attribution metric
        pos_mapping: Position mapping
        ig_steps: Integration steps for EAP-IG
        methods: Methods to run (default: all)

    Returns:
        Dict with keys like 'resid', 'attn', 'mlp', 'eap_attn', etc.
    """
    if methods is None:
        methods = ["standard", "eap", "eap_ig"]

    results = {}

    if "standard" in methods:
        with P("standard_attribution"):
            for comp in ["resid_post", "attn_out", "mlp_out"]:
                key = comp.replace("_post", "").replace("_out", "")
                results[key] = compute_attribution(
                    runner, clean_text, corrupted_text, metric, pos_mapping, comp
                )

    if "eap" in methods:
        with P("eap"):
            eap = compute_eap(
                runner, clean_text, corrupted_text, metric, pos_mapping
            )
            results["eap_attn"] = eap["attn"]
            results["eap_mlp"] = eap["mlp"]

    if "eap_ig" in methods:
        with P("eap_ig"):
            eap_ig = compute_eap_ig(
                runner, clean_text, corrupted_text, metric, pos_mapping, ig_steps
            )
            results["eap_ig_attn"] = eap_ig["attn"]
            results["eap_ig_mlp"] = eap_ig["mlp"]

    return results


# =============================================================================
# Utilities
# =============================================================================


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
    results = []
    for idx in flat_indices:
        layer_idx = int(idx // scores.shape[1])
        pos = int(idx % scores.shape[1])
        results.append((layers[layer_idx], pos, float(scores[layer_idx, pos])))
    return results
