"""Embedding alignment utilities for EAP-IG with different-length sequences.

Aligns embeddings between sequences using anchor texts, with configurable
padding strategies for segments between anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..common.token_positions import PositionMapping


class PaddingStrategy(Enum):
    """Strategy for padding segments between anchors."""

    ZERO = "zero"  # Pad with zeros
    REPEAT_LAST = "repeat_last"  # Repeat last token of segment
    REPEAT_FIRST = "repeat_first"  # Repeat first token of segment
    INTERPOLATE = "interpolate"  # Linear interpolation to fill gap
    MEAN = "mean"  # Use mean of segment


@dataclass
class AlignedSegment:
    """A segment of embeddings aligned between two sequences."""

    src_start: int
    src_end: int
    dst_start: int
    dst_end: int
    anchor_text: str | None = None  # Text marker at start of segment


@dataclass
class AlignedEmbeddings:
    """Aligned embeddings ready for interpolation."""

    clean_embeds: torch.Tensor  # [1, aligned_len, hidden]
    corrupted_embeds: torch.Tensor  # [1, aligned_len, hidden]
    aligned_len: int
    segments: list[AlignedSegment]
    # Maps aligned position -> original position in each sequence
    clean_pos_map: list[int | None]  # aligned_pos -> clean_pos (None if padding)
    corrupted_pos_map: list[int | None]  # aligned_pos -> corrupted_pos


def get_segments_from_anchors(
    pos_mapping: "PositionMapping",
) -> list[AlignedSegment]:
    """Extract segments from position mapping anchors.

    Each segment spans from one anchor to the next (or sequence boundaries).
    """
    anchors = pos_mapping.anchors
    anchor_texts = pos_mapping.anchor_texts
    src_len = pos_mapping.src_len
    dst_len = pos_mapping.dst_len

    if not anchors:
        # No anchors - single segment for whole sequence
        return [
            AlignedSegment(
                src_start=0,
                src_end=src_len,
                dst_start=0,
                dst_end=dst_len,
                anchor_text=None,
            )
        ]

    segments = []

    # First segment: start to first anchor
    first_src, first_dst = anchors[0]
    if first_src > 0 or first_dst > 0:
        segments.append(
            AlignedSegment(
                src_start=0,
                src_end=first_src,
                dst_start=0,
                dst_end=first_dst,
                anchor_text=None,
            )
        )

    # Middle segments: between consecutive anchors
    for i in range(len(anchors)):
        src_pos, dst_pos = anchors[i]
        anchor_text = anchor_texts[i] if i < len(anchor_texts) else None

        if i + 1 < len(anchors):
            next_src, next_dst = anchors[i + 1]
        else:
            next_src, next_dst = src_len, dst_len

        segments.append(
            AlignedSegment(
                src_start=src_pos,
                src_end=next_src,
                dst_start=dst_pos,
                dst_end=next_dst,
                anchor_text=anchor_text,
            )
        )

    return segments


def pad_segment(
    embeds: torch.Tensor,
    current_len: int,
    target_len: int,
    strategy: PaddingStrategy,
) -> torch.Tensor:
    """Pad embeddings to target length using specified strategy.

    Args:
        embeds: Embeddings [current_len, hidden]
        current_len: Current sequence length
        target_len: Target sequence length
        strategy: Padding strategy to use

    Returns:
        Padded embeddings [target_len, hidden]
    """
    if current_len >= target_len:
        return embeds[:target_len]

    hidden_dim = embeds.shape[-1]
    pad_len = target_len - current_len

    if strategy == PaddingStrategy.ZERO:
        padding = torch.zeros(pad_len, hidden_dim, dtype=embeds.dtype, device=embeds.device)

    elif strategy == PaddingStrategy.REPEAT_LAST:
        last_embed = embeds[-1:] if current_len > 0 else torch.zeros(1, hidden_dim, dtype=embeds.dtype, device=embeds.device)
        padding = last_embed.expand(pad_len, -1)

    elif strategy == PaddingStrategy.REPEAT_FIRST:
        first_embed = embeds[:1] if current_len > 0 else torch.zeros(1, hidden_dim, dtype=embeds.dtype, device=embeds.device)
        padding = first_embed.expand(pad_len, -1)

    elif strategy == PaddingStrategy.INTERPOLATE:
        if current_len < 2:
            padding = embeds[:1].expand(pad_len, -1) if current_len > 0 else torch.zeros(pad_len, hidden_dim, dtype=embeds.dtype, device=embeds.device)
        else:
            # Linearly interpolate between last two tokens
            start = embeds[-2]
            end = embeds[-1]
            alphas = torch.linspace(0, 1, pad_len + 2, device=embeds.device)[1:-1]
            padding = start.unsqueeze(0) * (1 - alphas.unsqueeze(1)) + end.unsqueeze(0) * alphas.unsqueeze(1)

    elif strategy == PaddingStrategy.MEAN:
        if current_len > 0:
            mean_embed = embeds.mean(dim=0, keepdim=True)
            padding = mean_embed.expand(pad_len, -1)
        else:
            padding = torch.zeros(pad_len, hidden_dim, dtype=embeds.dtype, device=embeds.device)

    else:
        raise ValueError(f"Unknown padding strategy: {strategy}")

    return torch.cat([embeds, padding], dim=0)


def align_embeddings(
    clean_embeds: torch.Tensor,
    corrupted_embeds: torch.Tensor,
    pos_mapping: "PositionMapping",
    padding_strategy: PaddingStrategy = PaddingStrategy.ZERO,
) -> AlignedEmbeddings:
    """Align embeddings from two sequences using anchor-based segmentation.

    Each segment between anchors is padded to the max length of that segment
    across both sequences. This ensures aligned positions for interpolation.

    Args:
        clean_embeds: Clean sequence embeddings [1, clean_len, hidden] or [clean_len, hidden]
        corrupted_embeds: Corrupted sequence embeddings [1, corr_len, hidden] or [corr_len, hidden]
        pos_mapping: Position mapping with anchors
        padding_strategy: How to pad shorter segments

    Returns:
        AlignedEmbeddings with same-length tensors ready for interpolation
    """
    # Handle batch dimension
    if clean_embeds.ndim == 3:
        clean_embeds = clean_embeds[0]
    if corrupted_embeds.ndim == 3:
        corrupted_embeds = corrupted_embeds[0]

    segments = get_segments_from_anchors(pos_mapping)

    aligned_clean_parts = []
    aligned_corrupted_parts = []
    clean_pos_map = []
    corrupted_pos_map = []

    for seg in segments:
        # Extract segment from each sequence
        clean_seg = clean_embeds[seg.dst_start : seg.dst_end]  # dst = clean (long)
        corrupted_seg = corrupted_embeds[seg.src_start : seg.src_end]  # src = corrupted (short)

        clean_seg_len = seg.dst_end - seg.dst_start
        corrupted_seg_len = seg.src_end - seg.src_start
        max_seg_len = max(clean_seg_len, corrupted_seg_len)

        if max_seg_len == 0:
            continue

        # Pad segments to same length
        padded_clean = pad_segment(clean_seg, clean_seg_len, max_seg_len, padding_strategy)
        padded_corrupted = pad_segment(corrupted_seg, corrupted_seg_len, max_seg_len, padding_strategy)

        aligned_clean_parts.append(padded_clean)
        aligned_corrupted_parts.append(padded_corrupted)

        # Build position maps
        for i in range(max_seg_len):
            clean_orig_pos = seg.dst_start + i if i < clean_seg_len else None
            corrupted_orig_pos = seg.src_start + i if i < corrupted_seg_len else None
            clean_pos_map.append(clean_orig_pos)
            corrupted_pos_map.append(corrupted_orig_pos)

    # Concatenate all segments
    if aligned_clean_parts:
        aligned_clean = torch.cat(aligned_clean_parts, dim=0).unsqueeze(0)
        aligned_corrupted = torch.cat(aligned_corrupted_parts, dim=0).unsqueeze(0)
        aligned_len = aligned_clean.shape[1]
    else:
        hidden_dim = clean_embeds.shape[-1]
        aligned_clean = torch.zeros(1, 0, hidden_dim, dtype=clean_embeds.dtype, device=clean_embeds.device)
        aligned_corrupted = torch.zeros(1, 0, hidden_dim, dtype=corrupted_embeds.dtype, device=corrupted_embeds.device)
        aligned_len = 0

    return AlignedEmbeddings(
        clean_embeds=aligned_clean,
        corrupted_embeds=aligned_corrupted,
        aligned_len=aligned_len,
        segments=segments,
        clean_pos_map=clean_pos_map,
        corrupted_pos_map=corrupted_pos_map,
    )
