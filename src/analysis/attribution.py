"""Attribution patching methods: standard, EAP, and EAP-IG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ..profiler import P

if TYPE_CHECKING:
    from ..models import ModelRunner
    from .patching import PatchingMetric


@dataclass
class AttributionResult:
    """Result from attribution patching."""

    scores: np.ndarray  # (n_layers, seq_len) or (n_layers, n_heads, seq_len)
    method: str
    component: str
    n_steps: int = 1  # For IG methods


def _build_pos_arrays(
    pos_mapping: dict[int, int], clean_len: int, corr_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build position mapping arrays for vectorized indexing."""
    clean_pos = np.arange(clean_len)
    corr_pos = np.array([pos_mapping.get(p, p) for p in range(clean_len)])
    # Mask positions that are out of bounds
    valid = corr_pos < corr_len
    return clean_pos, corr_pos, valid


def _compute_attribution_vectorized(
    clean_act: torch.Tensor,
    corr_act: torch.Tensor,
    grad: torch.Tensor,
    clean_pos: np.ndarray,
    corr_pos: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Compute attribution scores using vectorized operations."""
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


def compute_attribution(
    runner: "ModelRunner",
    clean_text: str,
    corrupted_text: str,
    metric: "PatchingMetric",
    pos_mapping: dict[int, int],
    component: str = "resid_post",
) -> np.ndarray:
    """Standard attribution patching: (clean - corrupted) · grad.

    Args:
        runner: Model runner
        clean_text: Clean input
        corrupted_text: Corrupted input
        metric: Patching metric
        pos_mapping: Clean to corrupted position mapping
        component: Component to analyze

    Returns:
        Attribution scores (n_layers, seq_len)
    """
    n_layers = runner.n_layers
    hook_filter = lambda n: f"hook_{component}" in n

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
        acts_with_grad = [(name, act) for name, act in corr_cache.items() if act.requires_grad]
        if acts_with_grad:
            names, acts = zip(*acts_with_grad)
            grad_list = torch.autograd.grad(
                metric_val, acts, retain_graph=True, allow_unused=True
            )
            grads = {
                name: grad.detach() for name, grad in zip(names, grad_list) if grad is not None
            }
        else:
            grads = {}

    with P("attr_scores"):
        hook_names = [f"blocks.{l}.hook_{component}" for l in range(n_layers)]
        clean_len = clean_cache[hook_names[0]].shape[1]
        corr_len = corr_cache[hook_names[0]].shape[1] if hook_names[0] in corr_cache else clean_len

        clean_pos, corr_pos, valid = _build_pos_arrays(pos_mapping, clean_len, corr_len)
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

            results[l] = _compute_attribution_vectorized(
                clean_act, corr_act, grad, clean_pos, corr_pos, valid
            )

    return results


def compute_eap(
    runner: "ModelRunner",
    clean_text: str,
    corrupted_text: str,
    metric: "PatchingMetric",
    pos_mapping: dict[int, int],
) -> dict[str, np.ndarray]:
    """Edge Attribution Patching: attribute to edges between components.

    Computes attribution for:
    - attn_out -> resid (attention contribution to residual)
    - mlp_out -> resid (MLP contribution to residual)

    Returns:
        Dict with 'attn' and 'mlp' attribution arrays (n_layers, seq_len)
    """
    n_layers = runner.n_layers

    def all_filter(n):
        return "hook_resid_post" in n or "hook_attn_out" in n or "hook_mlp_out" in n

    with P("eap_clean_cache"):
        with torch.no_grad():
            _, clean_cache = runner.run_with_cache(clean_text, names_filter=all_filter)

    with P("eap_corr_cache"):
        corr_logits, corr_cache = runner.run_with_cache_and_grad(
            corrupted_text, names_filter=all_filter
        )

    with P("eap_grads"):
        metric_val = metric.compute_raw(corr_logits)
        # Compute all gradients at once for efficiency
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
        clean_len = clean_cache[f"blocks.0.hook_resid_post"].shape[1]
        corr_len = corr_cache[f"blocks.0.hook_resid_post"].shape[1]
        clean_pos, corr_pos, valid = _build_pos_arrays(pos_mapping, clean_len, corr_len)

        attn_scores = np.zeros((n_layers, clean_len))
        mlp_scores = np.zeros((n_layers, clean_len))

        for l in range(n_layers):
            grad = resid_grads.get(l)
            if grad is None:
                continue

            # Attention edge
            attn_name = f"blocks.{l}.hook_attn_out"
            try:
                clean_attn = clean_cache[attn_name]
                corr_attn = corr_cache.get(attn_name)
                if corr_attn is not None:
                    attn_scores[l] = _compute_attribution_vectorized(
                        clean_attn, corr_attn, grad, clean_pos, corr_pos, valid
                    )
            except KeyError:
                pass

            # MLP edge
            mlp_name = f"blocks.{l}.hook_mlp_out"
            try:
                clean_mlp = clean_cache[mlp_name]
                corr_mlp = corr_cache.get(mlp_name)
                if corr_mlp is not None:
                    mlp_scores[l] = _compute_attribution_vectorized(
                        clean_mlp, corr_mlp, grad, clean_pos, corr_pos, valid
                    )
            except KeyError:
                pass

    return {"attn": attn_scores, "mlp": mlp_scores}


def compute_eap_ig(
    runner: "ModelRunner",
    clean_text: str,
    corrupted_text: str,
    metric: "PatchingMetric",
    pos_mapping: dict[int, int],
    n_steps: int = 10,
) -> dict[str, np.ndarray]:
    """Edge Attribution Patching with Integrated Gradients.

    Integrates gradients along path from corrupted to clean:
        IG = (clean - corrupted) · ∫₀¹ ∂metric/∂act(corrupted + α(clean - corrupted)) dα

    Args:
        runner: Model runner
        clean_text: Clean input
        corrupted_text: Corrupted input
        metric: Patching metric
        pos_mapping: Position mapping
        n_steps: Integration steps (higher = more accurate but slower)

    Returns:
        Dict with 'attn' and 'mlp' attribution arrays
    """
    n_layers = runner.n_layers

    def all_filter(n):
        return "hook_resid_post" in n or "hook_attn_out" in n or "hook_mlp_out" in n

    with P("eap_ig_cache"):
        with torch.no_grad():
            _, clean_cache = runner.run_with_cache(clean_text, names_filter=all_filter)
            _, corr_cache_base = runner.run_with_cache(corrupted_text, names_filter=all_filter)

    clean_len = clean_cache[f"blocks.0.hook_resid_post"].shape[1]
    corr_len = corr_cache_base[f"blocks.0.hook_resid_post"].shape[1]
    clean_pos, corr_pos, valid = _build_pos_arrays(pos_mapping, clean_len, corr_len)

    attn_grads_sum = np.zeros((n_layers, clean_len))
    mlp_grads_sum = np.zeros((n_layers, clean_len))

    formatted = runner._apply_chat_template(corrupted_text)
    input_ids = runner.tokenize(formatted)

    with P("eap_ig_integration"):
        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps

            # Create interpolated activations
            def make_interp_hook(layer, component, alpha=alpha):
                clean_name = f"blocks.{layer}.hook_{component}"
                try:
                    clean_act = clean_cache[clean_name]
                    corr_act = corr_cache_base[clean_name]
                except KeyError:
                    return None

                def hook(act, hook=None):
                    n_pos = min(act.shape[1], clean_act.shape[1])
                    valid_mask = corr_pos[:n_pos] < corr_act.shape[1]
                    for i in range(n_pos):
                        if valid_mask[i]:
                            cp = clean_pos[i]
                            rp = corr_pos[i]
                            interp = corr_act[0, rp, :] + alpha * (
                                clean_act[0, cp, :] - corr_act[0, rp, :]
                            )
                            act[:, rp, :] = interp.detach()
                    return act

                return hook

            hooks = []
            for l in range(n_layers):
                for comp in ["resid_post", "attn_out", "mlp_out"]:
                    hook_fn = make_interp_hook(l, comp)
                    if hook_fn is not None:
                        hooks.append((f"blocks.{l}.hook_{comp}", hook_fn))

            cache = {}

            def capture_hook(name):
                def hook(act, hook=None):
                    cache[name] = act
                    return act

                return hook

            all_hooks = hooks.copy()
            for l in range(n_layers):
                all_hooks.append(
                    (f"blocks.{l}.hook_resid_post", capture_hook(f"blocks.{l}.hook_resid_post"))
                )

            logits = runner.model.run_with_hooks(input_ids, fwd_hooks=all_hooks)
            metric_val = metric.compute_raw(logits)

            # Collect all activations for batch gradient computation
            resid_acts = []
            resid_layers = []
            for l in range(n_layers):
                resid_name = f"blocks.{l}.hook_resid_post"
                act = cache.get(resid_name)
                if act is not None and act.requires_grad:
                    resid_acts.append(act)
                    resid_layers.append(l)

            if not resid_acts:
                continue

            # Compute all gradients at once
            grad_list = torch.autograd.grad(
                metric_val, resid_acts, retain_graph=True, allow_unused=True
            )

            # Accumulate attribution scores
            for l, grad in zip(resid_layers, grad_list):
                if grad is None:
                    continue

                # Attention edge
                attn_name = f"blocks.{l}.hook_attn_out"
                try:
                    clean_attn = clean_cache[attn_name]
                    corr_attn = corr_cache_base[attn_name]
                    attr = _compute_attribution_vectorized(
                        clean_attn, corr_attn, grad, clean_pos, corr_pos, valid
                    )
                    attn_grads_sum[l] += attr / n_steps
                except KeyError:
                    pass

                # MLP edge
                mlp_name = f"blocks.{l}.hook_mlp_out"
                try:
                    clean_mlp = clean_cache[mlp_name]
                    corr_mlp = corr_cache_base[mlp_name]
                    attr = _compute_attribution_vectorized(
                        clean_mlp, corr_mlp, grad, clean_pos, corr_pos, valid
                    )
                    mlp_grads_sum[l] += attr / n_steps
                except KeyError:
                    pass

    return {"attn": attn_grads_sum, "mlp": mlp_grads_sum}


def aggregate_attribution_results(
    results_list: list[dict[str, np.ndarray]],
    n_layers: int,
) -> dict[str, np.ndarray]:
    """Aggregate multiple attribution results with padding.

    Args:
        results_list: List of dicts, each mapping method name to (n_layers, seq_len) array
        n_layers: Number of layers in model

    Returns:
        Aggregated results dict with mean across all pairs
    """
    if not results_list:
        return {}

    keys = results_list[0].keys()
    aggregated = {}

    for key in keys:
        arrays = [r[key] for r in results_list if key in r]
        if not arrays:
            aggregated[key] = np.zeros((n_layers, 1))
            continue

        max_len = max(a.shape[1] for a in arrays)
        padded = []
        for a in arrays:
            if a.shape[1] < max_len:
                p = np.zeros((a.shape[0], max_len))
                p[:, : a.shape[1]] = a
                padded.append(p)
            else:
                padded.append(a)
        aggregated[key] = np.mean(padded, axis=0)

    return aggregated


def find_top_attributions(
    scores: np.ndarray,
    layers: list[int],
    n_top: int = 5,
) -> list[tuple[int, int, float]]:
    """Find top N attribution scores by absolute value.

    Returns:
        List of (layer, position, score) tuples sorted by abs(score)
    """
    flat_indices = np.argsort(np.abs(scores).ravel())[::-1][:n_top]
    results = []
    for idx in flat_indices:
        layer_idx = int(idx // scores.shape[1])
        pos = int(idx % scores.shape[1])
        results.append((layers[layer_idx], pos, float(scores[layer_idx, pos])))
    return results


def run_all_attribution_methods(
    runner: "ModelRunner",
    clean_text: str,
    corrupted_text: str,
    metric: "PatchingMetric",
    pos_mapping: dict[int, int],
    ig_steps: int = 10,
) -> dict[str, np.ndarray]:
    """Run all attribution methods and return results.

    Returns dict with keys:
    - 'resid': Standard attribution on residual stream
    - 'attn': Standard attribution on attention output
    - 'mlp': Standard attribution on MLP output
    - 'eap_attn': EAP for attention edges
    - 'eap_mlp': EAP for MLP edges
    - 'eap_ig_attn': EAP-IG for attention edges
    - 'eap_ig_mlp': EAP-IG for MLP edges
    """
    results = {}

    with P("standard_attribution"):
        for comp in ["resid_post", "attn_out", "mlp_out"]:
            key = comp.replace("_post", "").replace("_out", "")
            results[key] = compute_attribution(
                runner, clean_text, corrupted_text, metric, pos_mapping, comp
            )

    with P("eap"):
        eap = compute_eap(runner, clean_text, corrupted_text, metric, pos_mapping)
        results["eap_attn"] = eap["attn"]
        results["eap_mlp"] = eap["mlp"]

    with P("eap_ig"):
        eap_ig = compute_eap_ig(
            runner, clean_text, corrupted_text, metric, pos_mapping, ig_steps
        )
        results["eap_ig_attn"] = eap_ig["attn"]
        results["eap_ig_mlp"] = eap_ig["mlp"]

    return results
