"""Shared utilities for activation and attribution patching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..common.token_positions import find_label_positions
from ..profiler import P
from .markers import find_section_markers, get_token_labels, SECTION_COLORS

if TYPE_CHECKING:
    from ..models import ModelRunner
    from ..data import PreferenceItem

# Re-export for convenience
__all__ = [
    "PatchingMetric",
    "build_position_mapping",
    "create_metric",
    "find_section_markers",
    "get_token_labels",
    "SECTION_COLORS",
]


@dataclass
class PatchingMetric:
    """Metric for activation/attribution patching.

    Stores token IDs and positions for computing logit differences.
    """

    clean_short_id: int
    clean_long_id: int
    corr_short_id: int
    corr_long_id: int
    clean_pos: int
    corr_pos: int
    clean_val: float
    corr_val: float
    diff: float

    def __call__(self, logits: torch.Tensor) -> float:
        """Compute normalized recovery (0=corrupted, 1=clean)."""
        raw = (
            logits[0, self.corr_pos, self.corr_short_id]
            - logits[0, self.corr_pos, self.corr_long_id]
        ).item()
        return (raw - self.corr_val) / self.diff if abs(self.diff) > 1e-8 else 0.0

    def compute_raw(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute raw logit difference as a tensor (for gradients)."""
        return (
            logits[0, self.corr_pos, self.corr_short_id]
            - logits[0, self.corr_pos, self.corr_long_id]
        )


def build_position_mapping(
    runner: "ModelRunner",
    clean_text: str,
    corrupted_text: str,
    clean_labels: list[str],
    corrupted_labels: list[str],
) -> tuple[dict[int, int], int, int]:
    """Build mapping from clean positions to corrupted positions.

    Uses semantic matching via sample labels, then interpolation for unmatched.

    Args:
        runner: ModelRunner instance
        clean_text: Clean prompt text
        corrupted_text: Corrupted prompt text
        clean_labels: Labels from clean sample [short_term_label, long_term_label]
        corrupted_labels: Labels from corrupted sample

    Returns:
        (mapping, clean_len, corrupted_len)
    """
    with P("pos_map_tokenize"):
        clean_formatted = runner._apply_chat_template(clean_text)
        corr_formatted = runner._apply_chat_template(corrupted_text)
        clean_tokens = runner.tokenize(clean_formatted)[0]
        corr_tokens = runner.tokenize(corr_formatted)[0]
        clean_len = len(clean_tokens)
        corr_len = len(corr_tokens)

        clean_token_strs = [runner.tokenizer.decode([t]) for t in clean_tokens]
        corr_token_strs = [runner.tokenizer.decode([t]) for t in corr_tokens]

    with P("pos_map_anchors"):
        clean_positions = find_label_positions(clean_token_strs, clean_labels)
        corr_positions = find_label_positions(corr_token_strs, corrupted_labels)

    # Build anchor points from matching labels
    anchors = []
    for clean_label, corr_label in zip(clean_labels, corrupted_labels):
        if clean_label in clean_positions and corr_label in corr_positions:
            anchors.append((clean_positions[clean_label], corr_positions[corr_label]))

    anchors.sort(key=lambda x: x[0])
    anchors = [(0, 0)] + anchors + [(clean_len - 1, corr_len - 1)]

    # Interpolate positions between anchors
    mapping = {}
    for i in range(len(anchors) - 1):
        c_start, r_start = anchors[i]
        c_end, r_end = anchors[i + 1]
        c_range = c_end - c_start
        r_range = r_end - r_start

        if c_range == 0:
            continue

        for c_pos in range(c_start, c_end + 1):
            t = (c_pos - c_start) / c_range if c_range > 0 else 0
            r_pos = int(r_start + t * r_range)
            mapping[c_pos] = max(0, min(r_pos, corr_len - 1))

    return mapping, clean_len, corr_len


def create_metric(
    runner: "ModelRunner",
    clean_sample: "PreferenceItem",
    corrupted_sample: "PreferenceItem",
    clean_text: str,
    corrupted_text: str,
) -> PatchingMetric:
    """Create metric for patching experiments.

    Args:
        runner: ModelRunner instance
        clean_sample: Sample that chose short_term
        corrupted_sample: Sample that chose long_term
        clean_text: Full text for clean sample
        corrupted_text: Full text for corrupted sample

    Returns:
        PatchingMetric with baseline values computed
    """
    clean_short_id, clean_long_id = runner.get_divergent_token_ids(
        clean_sample.short_term_label, clean_sample.long_term_label
    )
    corr_short_id, corr_long_id = runner.get_divergent_token_ids(
        corrupted_sample.short_term_label, corrupted_sample.long_term_label
    )

    def find_choice_pos(text, label):
        formatted = runner._apply_chat_template(text)
        tokens = runner.tokenize(formatted)[0].tolist()
        label_ids = runner.tokenizer.encode(" " + label, add_special_tokens=False)
        if not label_ids:
            return len(tokens) - 1
        for i, t in enumerate(tokens):
            if t == label_ids[0]:
                return max(0, i - 1)
        return len(tokens) - 1

    clean_label = (
        clean_sample.short_term_label
        if clean_sample.choice == "short_term"
        else clean_sample.long_term_label
    )
    corr_label = (
        corrupted_sample.short_term_label
        if corrupted_sample.choice == "short_term"
        else corrupted_sample.long_term_label
    )

    clean_pos = find_choice_pos(clean_text, clean_label)
    corr_pos = find_choice_pos(corrupted_text, corr_label)

    with torch.no_grad():
        clean_logits, _ = runner.run_with_cache(clean_text)
        corr_logits, _ = runner.run_with_cache(corrupted_text)

    clean_val = (
        clean_logits[0, clean_pos, clean_short_id]
        - clean_logits[0, clean_pos, clean_long_id]
    ).item()
    corr_val = (
        corr_logits[0, corr_pos, corr_short_id]
        - corr_logits[0, corr_pos, corr_long_id]
    ).item()

    return PatchingMetric(
        clean_short_id=clean_short_id,
        clean_long_id=clean_long_id,
        corr_short_id=corr_short_id,
        corr_long_id=corr_long_id,
        clean_pos=clean_pos,
        corr_pos=corr_pos,
        clean_val=clean_val,
        corr_val=corr_val,
        diff=clean_val - corr_val,
    )
