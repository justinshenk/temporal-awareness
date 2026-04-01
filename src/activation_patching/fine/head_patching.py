"""Head-level patching: attribute to individual attention heads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...common.contrastive_pair import ContrastivePair
from ...common.device_utils import clear_gpu_memory
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
    1. Get hook_z (head outputs before O projection) [batch, pos, n_heads, d_head]
    2. Compute per-head contribution via: z @ W_O (einsum over d_head dimension)
    3. Compute difference: clean_contrib - corrupted_contrib
    4. Project onto logit direction to get attribution score

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

    with torch.no_grad():
        for layer in layers:
            hook_z_name = f"blocks.{layer}.attn.hook_z"

            clean_z = clean_cache.get(hook_z_name)
            corrupted_z = corrupted_cache.get(hook_z_name)

            if clean_z is None or corrupted_z is None:
                raise RuntimeError(
                    f"hook_z not available for layer {layer}. "
                    "Check that your backend supports attention hooks."
                )

            # Get W_O: [n_heads, d_head, d_model]
            W_O = runner._backend.get_W_O(layer)

            # clean_z: [batch, pos, n_heads, d_head]
            # Clamp metric_pos to be within bounds of the actual sequence length
            clean_seq_len = clean_z.shape[1]
            corrupted_seq_len = corrupted_z.shape[1]
            clean_metric_pos = min(metric_pos, clean_seq_len - 1)
            corrupted_metric_pos = min(metric_pos, corrupted_seq_len - 1)

            # Get z at metric position: [n_heads, d_head]
            # Clone to avoid inference mode issues
            clean_z_at_pos = clean_z[0, clean_metric_pos, :, :].clone()
            corrupted_z_at_pos = corrupted_z[0, corrupted_metric_pos, :, :].clone()

            # Compute per-head contributions: z @ W_O[head] for each head
            # [n_heads, d_head] @ [n_heads, d_head, d_model] -> [n_heads, d_model]
            clean_at_pos = torch.einsum("hd,hdm->hm", clean_z_at_pos, W_O)
            corrupted_at_pos = torch.einsum("hd,hdm->hm", corrupted_z_at_pos, W_O)

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
