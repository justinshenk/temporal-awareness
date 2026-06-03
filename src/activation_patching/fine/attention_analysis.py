"""Attention pattern analysis: extract and analyze attention without patching.

This is the simpler alternative to full path patching. It extracts attention
patterns from important heads and computes what information flows along edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...common.contrastive_pair import ContrastivePair
from ...common.device_utils import clear_gpu_memory
from ...common.profiler import profile

from .fine_config import FineConfig
from .fine_results import AttentionPatternResult, HeadResult

if TYPE_CHECKING:
    from ...binary_choice import BinaryChoiceRunner


def _get_attention_pattern_filter(layers: list[int]) -> callable:
    """Create filter for attention pattern hooks at specific layers."""
    hooks = set()
    for layer in layers:
        # hook_attn: attention patterns after softmax [batch, n_heads, seq_q, seq_k]
        hooks.add(f"blocks.{layer}.attn.hook_attn")
        # hook_v: value vectors [batch, pos, n_heads, d_head]
        hooks.add(f"blocks.{layer}.attn.hook_v")
    return lambda name: name in hooks


@profile
def analyze_attention_patterns(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    important_heads: list[HeadResult],
    config: FineConfig | None = None,
) -> list[AttentionPatternResult]:
    """Analyze attention patterns for important heads.

    For each important head:
    1. Extract attention pattern from clean run
    2. Check if head attends from destination positions to source positions
    3. Compute attention-weighted value difference

    This provides evidence of information flow without causal patching.

    Args:
        runner: Model runner
        pair: Contrastive pair
        important_heads: List of heads to analyze (from head patching results)
        config: Fine patching configuration

    Returns:
        List of AttentionPatternResult for each analyzed head
    """
    if config is None:
        from .fine_config import DEFAULT_FINE_CONFIG
        config = DEFAULT_FINE_CONFIG

    if not important_heads:
        return []

    # Get unique layers from important heads
    layers = sorted(set(h.layer for h in important_heads))
    names_filter = _get_attention_pattern_filter(layers)

    # Run clean trajectory with attention hooks
    clean_choice = runner.choose(
        pair.clean_prompt,
        pair.choice_prefix,
        pair.clean_labels,
        with_cache=True,
        names_filter=names_filter,
    )
    clean_cache = clean_choice.cache

    # Run corrupted trajectory for value comparison
    corrupted_choice = runner.choose(
        pair.corrupted_prompt,
        pair.choice_prefix,
        pair.corrupted_labels,
        with_cache=True,
        names_filter=names_filter,
    )
    corrupted_cache = corrupted_choice.cache

    results = []

    for head_info in important_heads:
        layer = head_info.layer
        head = head_info.head

        attn_hook_name = f"blocks.{layer}.attn.hook_attn"
        v_hook_name = f"blocks.{layer}.attn.hook_v"

        clean_attn = clean_cache.get(attn_hook_name)
        clean_v = clean_cache.get(v_hook_name)
        corrupted_v = corrupted_cache.get(v_hook_name)

        if clean_attn is None:
            results.append(AttentionPatternResult(
                layer=layer,
                head=head,
                mean_attention_to_source=0.0,
                info_flow_norm=0.0,
            ))
            continue

        # clean_attn: [batch, n_heads, seq_q, seq_k]
        attn_pattern = clean_attn[0, head, :, :]  # [seq_q, seq_k]

        # Extract attention from destination positions to source positions
        src_positions = config.source_positions
        dest_positions = config.destination_positions

        # Filter to valid positions
        seq_len = attn_pattern.shape[0]
        valid_src = [p for p in src_positions if p < seq_len]
        valid_dest = [p for p in dest_positions if p < seq_len]

        if not valid_src or not valid_dest:
            results.append(AttentionPatternResult(
                layer=layer,
                head=head,
                mean_attention_to_source=0.0,
                info_flow_norm=0.0,
            ))
            continue

        # Extract submatrix: attention from destination to source
        src_to_dest = attn_pattern[np.ix_(valid_dest, valid_src)]  # [n_dest, n_src]
        mean_attn_to_src = src_to_dest.mean().item()

        # Compute attention-weighted value difference
        info_flow_norm = 0.0
        if clean_v is not None and corrupted_v is not None:
            # clean_v, corrupted_v: [batch, pos, n_heads, d_head]
            v_diff = clean_v[0, :, head, :] - corrupted_v[0, :, head, :]  # [pos, d_head]

            # Weight value differences by attention from destination positions
            # For each destination position, compute attention-weighted sum of value diffs
            attn_weighted_diff = torch.zeros(v_diff.shape[1], device=v_diff.device)
            for dest_idx, dest_pos in enumerate(valid_dest):
                for src_idx, src_pos in enumerate(valid_src):
                    weight = attn_pattern[dest_pos, src_pos]
                    attn_weighted_diff += weight * v_diff[src_pos, :]

            info_flow_norm = torch.norm(attn_weighted_diff).item()

        results.append(AttentionPatternResult(
            layer=layer,
            head=head,
            src_to_dest_attention=src_to_dest.detach().cpu().numpy(),
            mean_attention_to_source=mean_attn_to_src,
            attention_weighted_value_diff=attn_weighted_diff.detach().cpu().numpy() if clean_v is not None else None,
            info_flow_norm=info_flow_norm,
        ))

    # Clean up
    clean_choice.pop_heavy()
    corrupted_choice.pop_heavy()
    clear_gpu_memory()

    return results


@profile
def get_attention_summary(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    layers: list[int],
    config: FineConfig | None = None,
) -> dict[int, np.ndarray]:
    """Get attention pattern summary for visualization.

    Returns attention from destination to source positions for each layer.

    Args:
        runner: Model runner
        pair: Contrastive pair
        layers: Layers to analyze
        config: Fine patching configuration

    Returns:
        Dict mapping layer -> attention matrix [n_heads, n_dest, n_src]
    """
    if config is None:
        from .fine_config import DEFAULT_FINE_CONFIG
        config = DEFAULT_FINE_CONFIG

    names_filter = _get_attention_pattern_filter(layers)

    # Run clean trajectory
    clean_choice = runner.choose(
        pair.clean_prompt,
        pair.choice_prefix,
        pair.clean_labels,
        with_cache=True,
        names_filter=names_filter,
    )
    clean_cache = clean_choice.cache

    n_heads = runner._backend.get_n_heads()
    src_positions = config.source_positions
    dest_positions = config.destination_positions

    results = {}

    for layer in layers:
        attn_hook_name = f"blocks.{layer}.attn.hook_attn"
        clean_attn = clean_cache.get(attn_hook_name)

        if clean_attn is None:
            continue

        # clean_attn: [batch, n_heads, seq_q, seq_k]
        seq_len = clean_attn.shape[2]
        valid_src = [p for p in src_positions if p < seq_len]
        valid_dest = [p for p in dest_positions if p < seq_len]

        if not valid_src or not valid_dest:
            continue

        # Extract for all heads: [n_heads, n_dest, n_src]
        attn_submatrix = clean_attn[0, :, :, :][:, valid_dest, :][:, :, valid_src]
        results[layer] = attn_submatrix.detach().cpu().numpy()

    clean_choice.pop_heavy()
    clear_gpu_memory()

    return results
