"""Main analysis function for attention pattern analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....common.logging import log

from .attn_analysis_results import (
    AttnLayerResult,
    AttnPairResult,
    HeadAttnInfo,
)

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair


def run_attn_analysis(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    pair_idx: int = 0,
    layers: list[int] | None = None,
    dest_position: int | None = None,
    source_positions: list[int] | None = None,
    store_patterns: bool = False,
    dynamic_threshold: float = 0.1,
) -> AttnPairResult:
    """Run attention pattern analysis for a single pair.

    Computes:
    1. Per-head attention from dest to source positions
    2. Per-head output direction (logit contribution)
    3. Clean vs corrupted attention comparison (dynamic routing detection)
    4. Attention entropy and top-attended positions

    Args:
        runner: Model runner with access to attention weights
        pair: Contrastive pair
        pair_idx: Pair index for tracking
        layers: Layers to analyze (default: [19, 21, 24])
        dest_position: Destination position (None = last token / P_dest)
        source_positions: Source positions to track (None = auto-detect from pair)
        store_patterns: Whether to store full attention patterns (heavy)
        dynamic_threshold: Threshold for detecting dynamic attention changes

    Returns:
        AttnPairResult with per-layer, per-head analysis
    """
    if layers is None:
        layers = [19, 21, 24]

    # Determine positions
    if dest_position is None:
        dest_position = min(len(pair.clean_traj.token_ids), len(pair.corrupted_traj.token_ids)) - 1

    if source_positions is None:
        # Auto-detect source positions from pair's position mapping
        # Default: use a range around the divergence point
        source_positions = _detect_source_positions(pair)

    # Get logit direction
    logit_direction = _compute_logit_direction(runner, pair)

    # Get attention patterns and head outputs
    clean_attn, clean_head_outs, clean_attn_outs = _get_attention_data(
        runner, pair.clean_traj.token_ids, layers, dest_position
    )
    corrupted_attn, corrupted_head_outs, corrupted_attn_outs = _get_attention_data(
        runner, pair.corrupted_traj.token_ids, layers,
        pair.position_mapping.get(dest_position, dest_position)
    )

    layer_results = []
    attention_patterns = {}
    corrupted_attention_patterns = {}

    for layer in layers:
        # Check if we have detailed attention patterns or just attn_out
        has_patterns = layer in clean_attn
        has_attn_out = layer in clean_attn_outs

        if not has_patterns and not has_attn_out:
            log(f"[attn] Layer {layer}: no data available")
            continue

        # If we only have attn_out (not per-head patterns), create a summary result
        if not has_patterns:
            clean_out = clean_attn_outs[layer]
            corrupted_out = corrupted_attn_outs.get(layer)
            attn_diff = 0.0
            if corrupted_out is not None:
                attn_diff = float(torch.norm(clean_out - corrupted_out))

            # Create layer result without per-head info
            layer_results.append(AttnLayerResult(
                layer=layer,
                n_heads=0,
                head_results=[],
                total_attn_to_source=0.0,
                mean_attn_to_source=0.0,
                n_source_attending_heads=0,
            ))
            continue

        clean_a = clean_attn[layer]  # [n_heads, seq_len]
        corrupted_a = corrupted_attn.get(layer)
        clean_outs = clean_head_outs.get(layer, {})

        n_heads = clean_a.shape[0]
        head_results = []

        for head_idx in range(n_heads):
            head_attn = clean_a[head_idx]  # [seq_len]

            # Compute attention to source positions
            valid_src = [p for p in source_positions if p < len(head_attn)]
            attn_to_source = float(head_attn[valid_src].sum()) if valid_src else 0.0

            # Self-attention to dest
            attn_to_dest = float(head_attn[dest_position]) if dest_position < len(head_attn) else 0.0

            # Entropy of attention distribution
            attn_np = head_attn.cpu().numpy()
            attn_np = np.clip(attn_np, 1e-10, 1.0)  # Avoid log(0)
            attn_entropy = float(-np.sum(attn_np * np.log(attn_np)))

            # Top attended positions
            top_k = min(5, len(head_attn))
            top_indices = np.argsort(attn_np)[::-1][:top_k]
            top_positions = [int(i) for i in top_indices]
            top_weights = [float(attn_np[i]) for i in top_indices]

            # Logit contribution from head output
            logit_contribution = 0.0
            output_norm = 0.0
            if head_idx in clean_outs and logit_direction is not None:
                head_out = clean_outs[head_idx]  # [d_model]
                output_norm = float(torch.norm(head_out))
                logit_contribution = float(torch.dot(head_out, logit_direction))

            # Compare clean vs corrupted attention
            attn_pattern_diff = 0.0
            is_dynamic = False
            if corrupted_a is not None:
                corrupted_head_attn = corrupted_a[head_idx]
                # Compute L2 distance (truncate to shorter length)
                min_len = min(len(head_attn), len(corrupted_head_attn))
                diff = head_attn[:min_len] - corrupted_head_attn[:min_len]
                attn_pattern_diff = float(torch.norm(diff))
                is_dynamic = attn_pattern_diff > dynamic_threshold

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

        # Optionally store full patterns
        if store_patterns:
            attention_patterns[layer] = clean_a.cpu().tolist()
            if corrupted_a is not None:
                corrupted_attention_patterns[layer] = corrupted_a.cpu().tolist()

    return AttnPairResult(
        pair_idx=pair_idx,
        dest_position=dest_position,
        source_positions=source_positions,
        layer_results=layer_results,
        attention_patterns=attention_patterns if store_patterns else {},
        corrupted_attention_patterns=corrupted_attention_patterns if store_patterns else {},
    )


def _detect_source_positions(pair: "ContrastivePair") -> list[int]:
    """Detect source positions to track from pair's position mapping.

    Uses the inverse mapping to find which clean positions map to
    interesting corrupted positions (typically around the divergence).
    """
    # Default: positions 70-90 cover typical horizon token region
    # This can be refined based on pair-specific analysis
    seq_len = len(pair.clean_traj.token_ids)
    start = max(0, seq_len - 80)
    end = max(start + 20, seq_len - 10)
    return list(range(start, min(end, seq_len)))


def _compute_logit_direction(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
) -> torch.Tensor | None:
    """Compute normalized logit direction."""
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

    Tries TransformerLens-style hooks first, falls back to standard attn_out.

    Returns:
        (attn_patterns, head_outputs) where:
        - attn_patterns: {layer: attention[n_heads, seq_len]} from dest_position
        - head_outputs: {layer: {head_idx: output[d_model]}} at dest_position
    """
    # Build hook filter - try multiple TransformerLens hook name variants
    # Different TL versions/models use different names:
    # - hook_attn: older TL versions
    # - hook_pattern: Qwen3 and newer models (attention probabilities after softmax)
    # - hook_z: per-head outputs before final projection
    hooks = set()
    for layer in layers:
        hooks.add(f"blocks.{layer}.attn.hook_attn")  # Older TL attention patterns
        hooks.add(f"blocks.{layer}.attn.hook_pattern")  # Qwen3/newer TL attention patterns
        hooks.add(f"blocks.{layer}.attn.hook_result")  # Older TL per-head outputs
        hooks.add(f"blocks.{layer}.attn.hook_z")  # Qwen3/newer TL per-head outputs
        hooks.add(f"blocks.{layer}.hook_attn_out")  # Standard attn_out hook

    names_filter = lambda name: name in hooks

    # Run model
    input_ids = torch.tensor([token_ids], device=runner.device)
    with torch.no_grad():
        _, cache = runner._backend.run_with_cache(input_ids, names_filter=names_filter)

    attn_patterns = {}
    head_outputs = {}
    attn_outputs = {}  # For standard backend fallback

    for layer in layers:
        # Try TransformerLens attention patterns: [batch, n_heads, seq_q, seq_k]
        # Check both hook_attn (older) and hook_pattern (Qwen3/newer)
        for attn_key in [f"blocks.{layer}.attn.hook_pattern", f"blocks.{layer}.attn.hook_attn"]:
            if attn_key in cache:
                attn = cache[attn_key][0]  # [n_heads, seq_q, seq_k]
                dest_pos = min(dest_position, attn.shape[1] - 1)
                attn_patterns[layer] = attn[:, dest_pos, :]  # [n_heads, seq_k]
                break

        # Try TransformerLens head outputs: [batch, pos, n_heads, d_head]
        # Check both hook_result (older) and hook_z (Qwen3/newer)
        for result_key in [f"blocks.{layer}.attn.hook_z", f"blocks.{layer}.attn.hook_result"]:
            if result_key in cache:
                result = cache[result_key][0]  # [pos, n_heads, d_head]
                dest_pos = min(dest_position, result.shape[0] - 1)
                n_heads = result.shape[1]

                head_outputs[layer] = {}
                for head_idx in range(n_heads):
                    head_out = result[dest_pos, head_idx, :]  # [d_head]
                    head_outputs[layer][head_idx] = head_out
                break

        # Fallback: standard attn_out hook [batch, seq, d_model]
        attn_out_key = f"blocks.{layer}.hook_attn_out"
        if attn_out_key in cache and layer not in attn_patterns:
            attn_out = cache[attn_out_key][0]  # [seq, d_model]
            dest_pos = min(dest_position, attn_out.shape[0] - 1)
            attn_outputs[layer] = attn_out[dest_pos, :]  # [d_model]

    return attn_patterns, head_outputs, attn_outputs
