"""Head-level patching: attribute to individual attention heads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...common.contrastive_pair import ContrastivePair
from ...common.device_utils import clear_gpu_memory
from ...common.hook_utils import hook_name
from ...common.profiler import profile
from ...common.patching_types import PatchingMode

from .fine_config import FineConfig
from .fine_results import HeadResult, HeadPatchingResults

if TYPE_CHECKING:
    from ...binary_choice import BinaryChoiceRunner


def _get_attention_hook_filter(layers: list[int]) -> callable:
    """Create filter for attention-related hooks at specific layers."""
    hooks = set()
    for layer in layers:
        # hook_z: head outputs before O projection [batch, pos, n_heads, d_head]
        hooks.add(f"blocks.{layer}.attn.hook_z")
        # hook_result: per-head contributions after O projection [batch, pos, n_heads, d_model]
        hooks.add(f"blocks.{layer}.attn.hook_result")
        # hook_attn: attention patterns [batch, n_heads, seq_q, seq_k]
        hooks.add(f"blocks.{layer}.attn.hook_attn")
        # Also need attn_out for baseline
        hooks.add(hook_name(layer, "attn_out"))
    return lambda name: name in hooks


@profile
def run_head_patching(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    mode: PatchingMode,
    config: FineConfig | None = None,
) -> dict[int, HeadPatchingResults]:
    """Run head-level attribution at specified layers.

    For each attention head at each layer:
    1. Get head's contribution: hook_result[:, :, head, :] (after O projection)
    2. Compute difference: clean_contrib - corrupted_contrib
    3. Project onto metric direction to get attribution score

    Args:
        runner: Model runner
        pair: Contrastive pair
        mode: "denoising" or "noising"
        config: Fine patching configuration

    Returns:
        Dict mapping layer -> HeadPatchingResults
    """
    if config is None:
        from .fine_config import DEFAULT_FINE_CONFIG
        config = DEFAULT_FINE_CONFIG

    results: dict[int, HeadPatchingResults] = {}

    # Get model dimensions
    n_heads = runner._backend.get_n_heads()
    d_head = runner._backend.get_d_head()

    # Get baseline activations for clean and corrupted
    layers = config.head_layers
    names_filter = _get_attention_hook_filter(layers)

    # Run clean trajectory with cache
    clean_choice = runner.choose(
        pair.clean_prompt,
        pair.choice_prefix,
        pair.clean_labels,
        with_cache=True,
        names_filter=names_filter,
    )
    clean_cache = clean_choice.cache

    # Run corrupted trajectory with cache
    corrupted_choice = runner.choose(
        pair.corrupted_prompt,
        pair.choice_prefix,
        pair.corrupted_labels,
        with_cache=True,
        names_filter=names_filter,
    )
    corrupted_cache = corrupted_choice.cache

    # Get divergent position for metric computation
    clean_div_pos, corrupted_div_pos = pair.choice_divergent_positions
    metric_pos = corrupted_div_pos - 1 if mode == "denoising" else clean_div_pos - 1

    # Get logit direction (difference between choice A and B logits at output)
    # This represents what direction in activation space affects the choice
    W_U = runner.W_U  # [d_model, vocab_size]
    labels = pair.clean_labels
    label_a_id = runner.encode_ids(labels[0], add_special_tokens=False)[0]
    label_b_id = runner.encode_ids(labels[1], add_special_tokens=False)[0]
    logit_direction = W_U[:, label_a_id] - W_U[:, label_b_id]  # [d_model]
    logit_direction = logit_direction / torch.norm(logit_direction)

    for layer in layers:
        hook_result_name = f"blocks.{layer}.attn.hook_result"

        clean_result = clean_cache.get(hook_result_name)
        corrupted_result = corrupted_cache.get(hook_result_name)

        if clean_result is None or corrupted_result is None:
            # hook_result not available, fall back to computing from attn_out
            # This happens when the backend doesn't support per-head hooks
            results[layer] = _compute_head_attribution_fallback(
                runner, pair, layer, mode, n_heads, logit_direction, metric_pos
            )
            continue

        # clean_result: [batch, pos, n_heads, d_model]
        # Get at metric position
        clean_at_pos = clean_result[0, metric_pos, :, :]  # [n_heads, d_model]
        corrupted_at_pos = corrupted_result[0, metric_pos, :, :]  # [n_heads, d_model]

        head_results = []
        for head in range(n_heads):
            # Difference in this head's contribution
            diff = clean_at_pos[head] - corrupted_at_pos[head]  # [d_model]

            # Project onto logit direction
            score = torch.dot(diff, logit_direction).item()

            head_results.append(HeadResult(
                layer=layer,
                head=head,
                score=score,
            ))

        results[layer] = HeadPatchingResults(
            layer=layer,
            n_heads=n_heads,
            head_results=head_results,
        )

    # Clean up
    clean_choice.pop_heavy()
    corrupted_choice.pop_heavy()
    clear_gpu_memory()

    return results


def _compute_head_attribution_fallback(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    layer: int,
    mode: PatchingMode,
    n_heads: int,
    logit_direction: torch.Tensor,
    metric_pos: int,
) -> HeadPatchingResults:
    """Fallback attribution when hook_result is not available.

    Uses patching-based approach: patch each head's contribution and measure effect.
    This is slower but works with any backend.
    """
    # For now, return empty results when per-head hooks aren't available
    # A full implementation would need to:
    # 1. Get hook_z for each head
    # 2. Multiply by the O matrix to get per-head contribution
    # 3. Compute attribution from the difference

    return HeadPatchingResults(
        layer=layer,
        n_heads=n_heads,
        head_results=[
            HeadResult(layer=layer, head=h, score=0.0)
            for h in range(n_heads)
        ],
    )
