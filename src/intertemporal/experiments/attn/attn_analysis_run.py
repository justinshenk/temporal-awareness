"""Main analysis function for attention pattern analysis.

Uses semantic position names from SamplePositionMapping instead of absolute positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....common.logging import log
from .attn_analysis_config import AttnAnalysisConfig
from .attn_analysis_results import (
    AttnLayerResult,
    AttnPairResult,
    DstGroupAttention,
    HeadAttnInfo,
)


def canonical_format_pos(format_pos: str) -> str:
    """Normalize a format_pos string to its canonical form (R_/L_ prefix)."""
    if not format_pos:
        return format_pos
    if format_pos.startswith("left_"):
        return "L_" + format_pos[5:]
    if format_pos.startswith("right_"):
        return "R_" + format_pos[6:]
    return format_pos


def build_named_position_index(
    mapping: "SamplePositionMapping | None",
) -> "tuple[list[tuple[int, str, str]], dict[str, list[int]]]":
    """Build the index of named positions in a frame.

    Returns:
        (entries, group_to_positions) where
        - entries: ordered list of (abs_pos, group_label, full_label)
          covering every absolute position with a non-empty format_pos.
          full_label = "format_pos:rel_pos" (canonical R_/L_).
        - group_to_positions: dict from group_label (no rel_pos) to the
          ordered list of abs positions in that group.
    """
    if mapping is None:
        return [], {}
    entries: list[tuple[int, str, str]] = []
    groups: dict[str, list[int]] = {}
    seen: set[int] = set()
    seq_len = 0
    if mapping.positions:
        seq_len = max(p.abs_pos for p in mapping.positions) + 1
    for pos in range(seq_len):
        info = mapping.get_position(pos)
        if info is None or not info.format_pos:
            continue
        if pos in seen:
            continue
        seen.add(pos)
        group = canonical_format_pos(info.format_pos)
        rel = info.rel_pos if info.rel_pos is not None else 0
        full = f"{group}:{rel}"
        entries.append((pos, group, full))
        groups.setdefault(group, []).append(pos)
    entries.sort(key=lambda e: e[0])
    return entries, groups

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair
    from ...common.sample_position_mapping import SamplePositionMapping


def run_attn_analysis(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    mapping: "SamplePositionMapping",
    clean_mapping: "SamplePositionMapping",
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

    # `mapping` is the long_term (corrupted-frame) mapping; `clean_mapping` is short_term.
    corrupted_mapping = mapping

    # Build a per-frame ordered index of every named position (= every position
    # whose format_pos is set in the mapping) and the format_pos→positions
    # group lookup. We use ALL named positions as both candidate sources
    # (columns of the per-dst attention matrices) AND candidate destinations
    # (one DstGroupAttention entry per format_pos group present in either frame).
    clean_entries, clean_groups = build_named_position_index(clean_mapping)
    corr_entries, corr_groups = build_named_position_index(corrupted_mapping)

    # Canonical column labels = union of full labels in both frames, ordered
    # by their natural appearance (clean order first, then any extras in
    # corrupted that didn't appear in clean).
    canonical_labels: list[str] = []
    seen_labels: set[str] = set()
    for _pos, _grp, full in clean_entries:
        if full not in seen_labels:
            canonical_labels.append(full)
            seen_labels.add(full)
    for _pos, _grp, full in corr_entries:
        if full not in seen_labels:
            canonical_labels.append(full)
            seen_labels.add(full)

    # Per-frame: full_label -> abs_pos in that frame.
    clean_label_to_pos: dict[str, int] = {full: pos for pos, _g, full in clean_entries}
    corr_label_to_pos: dict[str, int] = {full: pos for pos, _g, full in corr_entries}

    # Destination groups: union of group_labels in both frames.
    dst_groups: list[str] = []
    seen_groups: set[str] = set()
    for grp in list(clean_groups.keys()) + list(corr_groups.keys()):
        if grp not in seen_groups:
            dst_groups.append(grp)
            seen_groups.add(grp)

    log(f"[attn] Pair {pair_idx}: {len(canonical_labels)} canonical labels, "
        f"{len(dst_groups)} dst groups")

    # Per-dst dst-position lists per frame
    dst_positions_clean: dict[str, list[int]] = {g: clean_groups.get(g, []) for g in dst_groups}
    dst_positions_corr: dict[str, list[int]] = {g: corr_groups.get(g, []) for g in dst_groups}

    # We need attention patterns for EVERY named destination position in both
    # frames. Collect the union of all dst abs positions per frame.
    all_clean_dst_positions = sorted({p for ps in dst_positions_clean.values() for p in ps})
    all_corr_dst_positions = sorted({p for ps in dst_positions_corr.values() for p in ps})

    # Get logit direction for OV analysis
    logit_direction = _compute_logit_direction(runner, pair)

    # Use ALL layers for attention pattern analysis (cheap to collect).
    # config.layers is reserved for head_attribution / position_patching only.
    all_layers = list(range(runner.n_layers))

    # Pull attention patterns from each frame at every needed dst position.
    clean_attn_by_pos, clean_head_outs = _get_attention_at_positions(
        runner, pair.clean_traj.token_ids, all_layers, all_clean_dst_positions
    )
    corr_attn_by_pos, corrupted_head_outs = _get_attention_at_positions(
        runner, pair.corrupted_traj.token_ids, all_layers, all_corr_dst_positions
    )

    # Pick a representative dst for the per-layer/per-head metrics: use the
    # last named position in each frame (typically the response_choice).
    # Layer/head metrics like "attn_to_source" become an aggregate signal
    # across that dst's full named-position context.
    metric_dst_clean = all_clean_dst_positions[-1] if all_clean_dst_positions else 0
    metric_dst_corr = all_corr_dst_positions[-1] if all_corr_dst_positions else 0

    layer_results = []
    for layer in all_layers:
        clean_layer_pos_attn = clean_attn_by_pos.get(layer, {})
        corr_layer_pos_attn = corr_attn_by_pos.get(layer, {})

        clean_a = clean_layer_pos_attn.get(metric_dst_clean)
        corrupted_a = corr_layer_pos_attn.get(metric_dst_corr)
        clean_outs = clean_head_outs.get(layer, {})

        if clean_a is None:
            continue

        n_heads = clean_a.shape[0]
        head_results = []

        # All named positions in each frame (for "attn_to_source" metric)
        all_named_corr = sorted({pos for pos, _g, _f in corr_entries})
        all_named_clean = sorted({pos for pos, _g, _f in clean_entries})

        for head_idx in range(n_heads):
            clean_head_attn = clean_a[head_idx]  # [seq_len]

            # "attn_to_source" = sum of attention from the metric dst to ALL
            # named positions in that frame (corrupted has all sources of interest).
            if corrupted_a is not None:
                corr_head_attn = corrupted_a[head_idx]
                valid_src_corr = [p for p in all_named_corr if p < len(corr_head_attn)]
                attn_to_source = float(corr_head_attn[valid_src_corr].sum()) if valid_src_corr else 0.0
            else:
                attn_to_source = 0.0

            # Self-attention to dest (clean frame metric dst)
            attn_to_dest = float(clean_head_attn[metric_dst_clean]) if metric_dst_clean < len(clean_head_attn) else 0.0

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

            # Use clean_mapping for label lookup since top_positions are clean-frame indices.
            top_labels = []
            for pos in top_positions:
                pos_info = clean_mapping.get_position(pos)
                if pos_info and pos_info.format_pos:
                    label = canonical_format_pos(pos_info.format_pos)
                    if pos_info.rel_pos is not None and pos_info.rel_pos >= 0:
                        label = f"{label}:{pos_info.rel_pos}"
                    top_labels.append(label)
                else:
                    top_labels.append(f"P{pos}")

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
                # Use float32 for accurate computation (model may output float16)
                clean_vec = clean_head_attn[:min_len].float()
                corr_vec = corrupted_head_attn[:min_len].float()
                diff = clean_vec - corr_vec

                attn_pattern_diff = float(torch.norm(diff))
                attn_pattern_diff_l1 = float(torch.abs(diff).sum())
                is_dynamic = attn_pattern_diff > config.dynamic_threshold

                dot = torch.dot(clean_vec, corr_vec)
                norm_product = clean_vec.norm() * corr_vec.norm() + 1e-10
                # Clamp to valid range (numerical precision can cause > 1.0)
                attn_pattern_cosine = float(torch.clamp(dot / norm_product, -1.0, 1.0))

            head_results.append(HeadAttnInfo(
                head_idx=head_idx,
                attn_to_source=attn_to_source,
                attn_to_dest=attn_to_dest,
                attn_entropy=attn_entropy,
                top_attended_positions=top_positions,
                top_attended_weights=top_weights,
                top_attended_labels=top_labels,
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

    # Build label-aligned per-dst attention.
    # For each destination format_pos group, mean attention over its rel_pos
    # in each frame. Then column-align both frames to the union of canonical
    # labels (positions absent in one frame are zero on that side).
    dst_group_attention: dict[str, DstGroupAttention] = {}
    if config.store_patterns:
        for grp in dst_groups:
            grp_dst_clean = dst_positions_clean.get(grp, [])
            grp_dst_corr = dst_positions_corr.get(grp, [])
            if not grp_dst_clean and not grp_dst_corr:
                continue

            clean_layer_map: dict[int, list[list[float]]] = {}
            corr_layer_map: dict[int, list[list[float]]] = {}

            for layer in all_layers:
                cl_pos_attn = clean_attn_by_pos.get(layer, {})
                co_pos_attn = corr_attn_by_pos.get(layer, {})

                # Clean side: mean across this dst group's clean positions
                clean_aligned: np.ndarray | None = None
                if grp_dst_clean and cl_pos_attn:
                    stacks = [cl_pos_attn[p] for p in grp_dst_clean if p in cl_pos_attn]
                    if stacks:
                        mean_clean = torch.stack(stacks, dim=0).mean(dim=0).cpu().numpy()
                        # Reindex to canonical_labels
                        n_heads_l = mean_clean.shape[0]
                        clean_aligned = np.zeros((n_heads_l, len(canonical_labels)), dtype=np.float32)
                        for ci, full in enumerate(canonical_labels):
                            ap = clean_label_to_pos.get(full)
                            if ap is not None and ap < mean_clean.shape[1]:
                                clean_aligned[:, ci] = mean_clean[:, ap]

                # Corrupted side
                corr_aligned: np.ndarray | None = None
                if grp_dst_corr and co_pos_attn:
                    stacks = [co_pos_attn[p] for p in grp_dst_corr if p in co_pos_attn]
                    if stacks:
                        mean_corr = torch.stack(stacks, dim=0).mean(dim=0).cpu().numpy()
                        n_heads_l = mean_corr.shape[0]
                        corr_aligned = np.zeros((n_heads_l, len(canonical_labels)), dtype=np.float32)
                        for ci, full in enumerate(canonical_labels):
                            ap = corr_label_to_pos.get(full)
                            if ap is not None and ap < mean_corr.shape[1]:
                                corr_aligned[:, ci] = mean_corr[:, ap]

                if clean_aligned is not None:
                    clean_layer_map[layer] = clean_aligned.tolist()
                if corr_aligned is not None:
                    corr_layer_map[layer] = corr_aligned.tolist()

            # canonical positions belonging to this dst group (column indices)
            dst_indices = [
                ci for ci, full in enumerate(canonical_labels)
                if full.split(":", 1)[0] == grp
            ]

            dst_group_attention[grp] = DstGroupAttention(
                dst_label=grp,
                canonical_labels=canonical_labels,
                dst_position_indices=dst_indices,
                clean=clean_layer_map,
                corrupted=corr_layer_map,
            )

    return AttnPairResult(
        pair_idx=pair_idx,
        layer_results=layer_results,
        dst_group_attention=dst_group_attention,
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


def _get_attention_at_positions(
    runner: "BinaryChoiceRunner",
    token_ids: list[int],
    layers: list[int],
    dst_positions: list[int],
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    """Get attention patterns from each requested dst position, plus head outputs.

    Returns:
        (attn_by_layer_pos, head_outputs_by_layer) where
        - attn_by_layer_pos[layer][dst_pos] = [n_heads, seq_len] attention from dst_pos
        - head_outputs_by_layer[layer][head_idx] = [d_head] output at last_dst_pos
          (used by logit_contribution; doesn't need to be per-position)
    """
    if not dst_positions:
        return {}, {}
    last_dst = dst_positions[-1]

    # Build hook filter for attention patterns
    hooks = set()
    for layer in layers:
        hooks.add(f"blocks.{layer}.attn.hook_pattern")
        hooks.add(f"blocks.{layer}.attn.hook_attn")
        hooks.add(f"blocks.{layer}.attn.hook_z")
        hooks.add(f"blocks.{layer}.attn.hook_result")

    names_filter = lambda name: name in hooks

    input_ids = torch.tensor([token_ids], device=runner.device)
    with torch.no_grad():
        _, cache = runner._backend.run_with_cache(input_ids, names_filter=names_filter)

    attn_by_layer_pos: dict[int, dict[int, torch.Tensor]] = {}
    head_outputs: dict[int, dict[int, torch.Tensor]] = {}

    for layer in layers:
        for attn_key in [f"blocks.{layer}.attn.hook_pattern", f"blocks.{layer}.attn.hook_attn"]:
            if attn_key in cache:
                attn = cache[attn_key][0]  # [n_heads, seq_q, seq_k]
                per_pos: dict[int, torch.Tensor] = {}
                for dp in dst_positions:
                    dpc = min(dp, attn.shape[1] - 1)
                    if dpc < 0:
                        continue
                    per_pos[dp] = attn[:, dpc, :].clone()
                attn_by_layer_pos[layer] = per_pos
                break

        for result_key in [f"blocks.{layer}.attn.hook_z", f"blocks.{layer}.attn.hook_result"]:
            if result_key in cache:
                result = cache[result_key][0]  # [pos, n_heads, d_head]
                last_pos = min(last_dst, result.shape[0] - 1)
                if last_pos >= 0:
                    n_heads = result.shape[1]
                    head_outputs[layer] = {
                        head_idx: result[last_pos, head_idx, :].clone()
                        for head_idx in range(n_heads)
                    }
                break

    return attn_by_layer_pos, head_outputs


