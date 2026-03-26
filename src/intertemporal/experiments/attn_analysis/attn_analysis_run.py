"""Main analysis function for attention pattern analysis.

Uses semantic position names from SamplePositionMapping instead of absolute positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....common.logging import log
from .attn_analysis_config import AttnAnalysisConfig

__all__ = [
    "run_attn_analysis",
    "resolve_positions",
]
from .attn_analysis_results import (
    AttnLayerResult,
    AttnPairResult,
    HeadAttnInfo,
)

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair
    from ...common.sample_position_mapping import SamplePositionMapping


def run_attn_analysis(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    mapping: "SamplePositionMapping",
    pair_idx: int = 0,
    config: AttnAnalysisConfig | None = None,
) -> AttnPairResult:
    """Run attention pattern analysis for a single pair.

    Analyzes attention patterns from destination positions (response) to
    source positions (time horizon tokens) using semantic position names.

    Args:
        runner: Model runner with access to attention weights
        pair: Contrastive pair
        mapping: SamplePositionMapping for resolving semantic position names
        pair_idx: Pair index for tracking
        config: Analysis configuration (uses defaults if None)

    Returns:
        AttnPairResult with per-layer, per-head analysis
    """
    if config is None:
        config = AttnAnalysisConfig()

    # Resolve semantic positions to absolute positions
    # NOTE: Positions are in the CORRUPTED frame (mapping built from long_term sample)
    source_positions_corrupted = resolve_positions(mapping, config.source_positions)
    dest_positions_corrupted = resolve_positions(mapping, config.dest_positions)

    if not source_positions_corrupted:
        log(f"[attn] Pair {pair_idx}: No source positions found for {config.source_positions}")
        return _empty_result(pair_idx, config)

    if not dest_positions_corrupted:
        log(f"[attn] Pair {pair_idx}: No dest positions found for {config.dest_positions}")
        return _empty_result(pair_idx, config)

    # Build reverse position mapping (corrupted -> clean)
    reverse_mapping: dict[int, int] = {}
    for clean_pos, corr_pos in pair.position_mapping.items():
        reverse_mapping[corr_pos] = clean_pos

    # Map positions to clean frame
    # For positions that don't exist in clean (e.g., time_horizon), they won't be mapped
    source_positions_clean = [reverse_mapping.get(p, -1) for p in source_positions_corrupted]
    source_positions_clean = [p for p in source_positions_clean if p >= 0]

    dest_positions_clean = [reverse_mapping.get(p, p) for p in dest_positions_corrupted]

    # Use first dest position as primary query position
    dest_position_clean = dest_positions_clean[0] if dest_positions_clean else 0
    dest_position_corrupted = dest_positions_corrupted[0]

    # Log position mappings for debugging
    log(f"[attn] Pair {pair_idx}: Source positions - corrupted: {source_positions_corrupted}, clean: {source_positions_clean}")
    log(f"[attn] Pair {pair_idx}: Dest positions - corrupted: {dest_position_corrupted}, clean: {dest_position_clean}")

    # Get logit direction for OV analysis
    logit_direction = _compute_logit_direction(runner, pair)

    # Get attention patterns for clean and corrupted (using frame-appropriate positions)
    clean_attn, clean_head_outs = _get_attention_data(
        runner, pair.clean_traj.token_ids, config.layers, dest_position_clean
    )

    corrupted_attn, corrupted_head_outs = _get_attention_data(
        runner, pair.corrupted_traj.token_ids, config.layers, dest_position_corrupted
    )

    # Analyze each layer
    layer_results = []
    attention_patterns = {}
    corrupted_attention_patterns = {}

    for layer in config.layers:
        if layer not in clean_attn:
            continue

        clean_a = clean_attn[layer]  # [n_heads, seq_len]
        corrupted_a = corrupted_attn.get(layer)
        clean_outs = clean_head_outs.get(layer, {})

        n_heads = clean_a.shape[0]
        head_results = []

        for head_idx in range(n_heads):
            clean_head_attn = clean_a[head_idx]  # [seq_len]

            # Compute attention to source positions
            # Use corrupted attention for primary metric (since time_horizon is in corrupted)
            # Also compute clean attention for comparison
            if corrupted_a is not None:
                corr_head_attn = corrupted_a[head_idx]
                valid_src_corr = [p for p in source_positions_corrupted if p < len(corr_head_attn)]
                attn_to_source = float(corr_head_attn[valid_src_corr].sum()) if valid_src_corr else 0.0
            else:
                attn_to_source = 0.0

            # Also compute clean attention to source (for comparison)
            valid_src_clean = [p for p in source_positions_clean if p < len(clean_head_attn)]
            attn_to_source_clean = float(clean_head_attn[valid_src_clean].sum()) if valid_src_clean else 0.0

            # Self-attention to dest
            attn_to_dest = float(clean_head_attn[dest_position_clean]) if dest_position_clean < len(clean_head_attn) else 0.0

            # Entropy of attention distribution (use clean for consistency)
            attn_np = clean_head_attn.cpu().numpy().copy()
            # Only include values above threshold to avoid log(0)
            mask = attn_np > 1e-10
            if mask.sum() > 0:
                attn_masked = attn_np[mask]
                attn_masked = attn_masked / attn_masked.sum()
                attn_entropy = float(-np.sum(attn_masked * np.log(attn_masked)))
            else:
                attn_entropy = 0.0

            # Top attended positions (from clean attention)
            top_k = min(5, len(clean_head_attn))
            top_indices = np.argsort(attn_np)[::-1][:top_k]
            top_positions = [int(i) for i in top_indices]
            top_weights = [float(attn_np[i]) for i in top_indices]

            # Logit contribution from head output
            logit_contribution = 0.0
            output_norm = 0.0
            if head_idx in clean_outs and logit_direction is not None:
                head_out = clean_outs[head_idx]
                output_norm = float(torch.norm(head_out))
                if head_out.shape[0] == logit_direction.shape[0]:
                    logit_contribution = float(torch.dot(head_out, logit_direction))

            # Compare clean vs corrupted attention
            attn_pattern_diff = 0.0
            attn_pattern_diff_l1 = 0.0
            attn_pattern_cosine = 0.0
            is_dynamic = False

            if corrupted_a is not None:
                corrupted_head_attn = corrupted_a[head_idx]
                min_len = min(len(clean_head_attn), len(corrupted_head_attn))
                clean_vec = clean_head_attn[:min_len]
                corr_vec = corrupted_head_attn[:min_len]
                diff = clean_vec - corr_vec

                attn_pattern_diff = float(torch.norm(diff))
                attn_pattern_diff_l1 = float(torch.abs(diff).sum())
                is_dynamic = attn_pattern_diff > config.dynamic_threshold

                dot = torch.dot(clean_vec, corr_vec)
                norm_product = clean_vec.norm() * corr_vec.norm() + 1e-10
                attn_pattern_cosine = float(dot / norm_product)

            head_results.append(HeadAttnInfo(
                head_idx=head_idx,
                attn_to_source=attn_to_source,
                attn_to_dest=attn_to_dest,
                attn_entropy=attn_entropy,
                top_attended_positions=top_positions,
                top_attended_weights=top_weights,
                logit_contribution=logit_contribution,
                output_norm=output_norm,
                attn_pattern_diff=attn_pattern_diff,
                attn_pattern_diff_l1=attn_pattern_diff_l1,
                attn_pattern_cosine=attn_pattern_cosine,
                is_dynamic=is_dynamic,
            ))

        # Layer-level aggregates
        total_attn = sum(h.attn_to_source for h in head_results)
        mean_attn = total_attn / n_heads if n_heads > 0 else 0.0
        n_source_attending = sum(1 for h in head_results if h.attn_to_source > 0.1)

        layer_results.append(AttnLayerResult(
            layer=layer,
            n_heads=n_heads,
            head_results=head_results,
            total_attn_to_source=total_attn,
            mean_attn_to_source=mean_attn,
            n_source_attending_heads=n_source_attending,
        ))

        # Store patterns if configured
        if config.store_patterns:
            attention_patterns[layer] = clean_a.cpu().tolist()
            if corrupted_a is not None:
                corrupted_attention_patterns[layer] = corrupted_a.cpu().tolist()

    return AttnPairResult(
        pair_idx=pair_idx,
        dest_position=dest_position_corrupted,
        source_positions=source_positions_corrupted,
        source_positions_clean=source_positions_clean,
        source_position_names=config.source_positions,
        dest_position_names=config.dest_positions,
        layer_results=layer_results,
        attention_patterns=attention_patterns if config.store_patterns else {},
        corrupted_attention_patterns=corrupted_attention_patterns if config.store_patterns else {},
    )


def resolve_positions(
    mapping: "SamplePositionMapping",
    position_names: list[str],
) -> list[int]:
    """Resolve semantic position names to absolute positions.

    Args:
        mapping: SamplePositionMapping with named_positions
        position_names: List of semantic position names (e.g., ["time_horizon"])

    Returns:
        List of absolute positions
    """
    positions = []
    for name in position_names:
        abs_positions = mapping.named_positions.get(name, [])
        positions.extend(abs_positions)
    return sorted(set(positions))


def _empty_result(pair_idx: int, config: AttnAnalysisConfig) -> AttnPairResult:
    """Create an empty result when positions can't be resolved."""
    return AttnPairResult(
        pair_idx=pair_idx,
        dest_position=0,
        source_positions=[],
        source_position_names=config.source_positions,
        dest_position_names=config.dest_positions,
        layer_results=[],
        attention_patterns={},
        corrupted_attention_patterns={},
    )


def _compute_logit_direction(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
) -> torch.Tensor | None:
    """Compute normalized logit direction between clean and corrupted tokens."""
    W_U = runner.W_U
    if W_U is None:
        return None

    clean_div_pos = pair.clean_divergent_position
    corrupted_div_pos = pair.corrupted_divergent_position

    if clean_div_pos is None or corrupted_div_pos is None:
        clean_token = pair.clean_traj.token_ids[-1]
        corrupted_token = pair.corrupted_traj.token_ids[-1]
    else:
        clean_token = pair.clean_traj.token_ids[clean_div_pos]
        corrupted_token = pair.corrupted_traj.token_ids[corrupted_div_pos]

    if clean_token == corrupted_token:
        return None

    if W_U.shape[0] > W_U.shape[1]:
        clean_vec = W_U[clean_token]
        corrupted_vec = W_U[corrupted_token]
    else:
        clean_vec = W_U[:, clean_token]
        corrupted_vec = W_U[:, corrupted_token]

    direction = clean_vec - corrupted_vec
    return direction / torch.norm(direction)


def _get_attention_data(
    runner: "BinaryChoiceRunner",
    token_ids: list[int],
    layers: list[int],
    dest_position: int,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[int, torch.Tensor]]]:
    """Get attention patterns and head outputs.

    Returns:
        (attn_patterns, head_outputs) where:
        - attn_patterns: {layer: attention[n_heads, seq_len]} from dest_position
        - head_outputs: {layer: {head_idx: output[d_head]}} at dest_position
    """
    # Build hook filter for attention patterns
    hooks = set()
    for layer in layers:
        hooks.add(f"blocks.{layer}.attn.hook_pattern")
        hooks.add(f"blocks.{layer}.attn.hook_attn")
        hooks.add(f"blocks.{layer}.attn.hook_z")
        hooks.add(f"blocks.{layer}.attn.hook_result")

    names_filter = lambda name: name in hooks

    # Run model
    input_ids = torch.tensor([token_ids], device=runner.device)
    with torch.no_grad():
        _, cache = runner._backend.run_with_cache(input_ids, names_filter=names_filter)

    attn_patterns = {}
    head_outputs = {}

    for layer in layers:
        # Get attention patterns
        for attn_key in [f"blocks.{layer}.attn.hook_pattern", f"blocks.{layer}.attn.hook_attn"]:
            if attn_key in cache:
                attn = cache[attn_key][0]  # [n_heads, seq_q, seq_k]
                dest_pos = min(dest_position, attn.shape[1] - 1)
                attn_patterns[layer] = attn[:, dest_pos, :]
                break

        # Get head outputs
        for result_key in [f"blocks.{layer}.attn.hook_z", f"blocks.{layer}.attn.hook_result"]:
            if result_key in cache:
                result = cache[result_key][0]  # [pos, n_heads, d_head]
                dest_pos = min(dest_position, result.shape[0] - 1)
                n_heads = result.shape[1]
                head_outputs[layer] = {
                    head_idx: result[dest_pos, head_idx, :]
                    for head_idx in range(n_heads)
                }
                break

    return attn_patterns, head_outputs
