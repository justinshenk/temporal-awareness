"""Inference utilities for activation extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .model_runner import ModelRunner


def get_all_activations(
    runner: "ModelRunner",
    tokens: list[int],
    components: list[str] | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Get activations at all layers for specified components.

    Args:
        runner: Model runner
        tokens: Input token IDs
        components: Components to extract (default: resid_post, resid_pre, attn_out, mlp_out)

    Returns:
        Dict mapping component -> layer -> activations [seq_len, d_model]
    """
    if components is None:
        components = ["resid_post", "resid_pre", "attn_out", "mlp_out"]

    result = {comp: {} for comp in components}

    input_ids = torch.tensor([tokens], device=runner.device)

    with torch.no_grad():
        _, cache = runner._backend.run_with_cache(input_ids, names_filter=None)

    for key, value in cache.items():
        if not key.startswith("blocks."):
            continue

        parts = key.split(".")
        if len(parts) < 3:
            continue

        try:
            layer = int(parts[1])
        except ValueError:
            continue

        hook_part = parts[2]
        if hook_part.startswith("hook_"):
            comp = hook_part[5:]
            if comp in components:
                result[comp][layer] = value[0].cpu().numpy()

    return result


def get_resid_post_activations(
    runner: "ModelRunner",
    tokens: list[int],
) -> dict[int, np.ndarray]:
    """Get resid_post activations at all layers.

    Args:
        runner: Model runner
        tokens: Input token IDs

    Returns:
        Dict mapping layer -> activations [seq_len, d_model]
    """
    return get_all_activations(runner, tokens, ["resid_post"])["resid_post"]


def get_logit_diff_direction(
    runner: "ModelRunner",
    token_a: int,
    token_b: int,
) -> np.ndarray | None:
    """Get the logit difference direction between two tokens.

    Args:
        runner: Model runner
        token_a: First token ID
        token_b: Second token ID

    Returns:
        Normalized direction vector (token_a - token_b) or None if unavailable
    """
    try:
        W_U = runner.W_U
        if W_U is None:
            return None

        emb_a = W_U[:, token_a].cpu().numpy()
        emb_b = W_U[:, token_b].cpu().numpy()

        direction = emb_a - emb_b
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            return direction / norm
        return None
    except (AttributeError, IndexError):
        return None


def get_mlp_neuron_activations(
    runner: "ModelRunner",
    tokens: list[int],
    layers: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Get per-neuron MLP activations (after activation function) for all positions.

    Uses mlp.hook_post which captures neuron activations [batch, seq, d_mlp].

    Args:
        runner: Model runner
        tokens: Input token IDs
        layers: Layers to extract (default: all layers)

    Returns:
        Dict mapping layer -> activations [seq_len, d_mlp]
    """
    if layers is None:
        layers = list(range(runner.n_layers))

    hooks = {f"blocks.{layer}.mlp.hook_post" for layer in layers}
    names_filter = lambda name: name in hooks

    input_ids = torch.tensor([tokens], device=runner.device)
    with torch.no_grad():
        _, cache = runner._backend.run_with_cache(input_ids, names_filter=names_filter)

    result = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.mlp.hook_post"
        if hook_name in cache:
            # [batch, seq, d_mlp] -> [seq, d_mlp]
            result[layer] = cache[hook_name][0].cpu().numpy()

    return result


def get_mlp_out_activations(
    runner: "ModelRunner",
    tokens: list[int],
    layers: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Get MLP output activations for all positions.

    Uses hook_mlp_out which captures [batch, seq, d_model].

    Args:
        runner: Model runner
        tokens: Input token IDs
        layers: Layers to extract (default: all layers)

    Returns:
        Dict mapping layer -> activations [seq_len, d_model]
    """
    if layers is None:
        layers = list(range(runner.n_layers))

    hooks = {f"blocks.{layer}.hook_mlp_out" for layer in layers}
    names_filter = lambda name: name in hooks

    input_ids = torch.tensor([tokens], device=runner.device)
    with torch.no_grad():
        _, cache = runner._backend.run_with_cache(input_ids, names_filter=names_filter)

    result = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_mlp_out"
        if hook_name in cache:
            # [batch, seq, d_model] -> [seq, d_model]
            result[layer] = cache[hook_name][0].cpu().numpy()

    return result


def get_mlp_w_out(runner: "ModelRunner", layer: int) -> np.ndarray | None:
    """Get W_out matrix [d_mlp, d_model] for a layer.

    For TransformerLens models: blocks[layer].mlp.W_out
    For HuggingFace models: model.layers[layer].mlp.down_proj.weight.T

    Args:
        runner: Model runner
        layer: Layer index

    Returns:
        W_out matrix [d_mlp, d_model] or None if not available
    """
    # Try TransformerLens-style first
    try:
        if hasattr(runner._model, "blocks"):
            return runner._model.blocks[layer].mlp.W_out.detach().cpu().numpy()
    except (AttributeError, IndexError):
        pass

    # Try HuggingFace-style (Qwen, Llama, etc.)
    try:
        if hasattr(runner._model, "model") and hasattr(runner._model.model, "layers"):
            mlp = runner._model.model.layers[layer].mlp
            if hasattr(mlp, "down_proj"):
                # down_proj.weight is [d_model, d_mlp], we need [d_mlp, d_model]
                return mlp.down_proj.weight.T.detach().cpu().numpy()
    except (AttributeError, IndexError):
        pass

    return None
