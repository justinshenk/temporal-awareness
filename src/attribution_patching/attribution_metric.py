"""Attribution metric for computing gradients of choice difference.

The metric computes a scalar value from logits that can be differentiated
to get attribution scores. For binary choices, this is typically the
logit difference between the two options at the divergent position.

IMPORTANT: The metric must be computed at the correct position:
- For denoising (clean base): compute at clean's divergent position
- For noising (corrupted base): compute at corrupted's divergent position
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from ..binary_choice import BinaryChoiceRunner
from ..common.base_schema import BaseSchema
from ..common.contrastive_pair import ContrastivePair
from ..common.patching_types import PatchingMode


@dataclass
class AttributionMetric(BaseSchema):
    """Metric for computing attribution scores.

    Computes a scalar metric from logits that measures the model's
    preference between two choices. The gradient of this metric
    w.r.t. activations gives attribution scores.

    Attributes:
        target_token_ids: (chosen_id, alternative_id) token IDs
        divergent_position: Position at which to compute the metric
            (where A/B tokens diverge in the base trajectory)
        clean_logit_diff: Logit difference on clean (target) input
        corrupted_logit_diff: Logit difference on corrupted (baseline) input
    """

    target_token_ids: tuple[int, int]
    divergent_position: int = -1  # -1 means last position (legacy behavior)
    clean_logit_diff: float = 0.0
    corrupted_logit_diff: float = 0.0

    @property
    def diff(self) -> float:
        """Difference between clean and corrupted metric values.

        This represents the total effect we're trying to attribute.
        """
        return self.clean_logit_diff - self.corrupted_logit_diff

    def compute_raw(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute scalar metric from logits (must be differentiable).

        Returns the logit difference: logit[chosen] - logit[alternative]
        at the divergent position.

        Args:
            logits: Model output logits [batch=1, seq_len, vocab]

        Returns:
            Scalar tensor for gradient computation

        Raises:
            ValueError: If batch size != 1 or position out of bounds
        """
        if logits.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {logits.shape[0]}")

        seq_len = logits.shape[1]
        position = self.divergent_position
        if position < 0:
            position = seq_len + position

        if position < 0 or position >= seq_len:
            raise ValueError(f"Position {position} out of bounds for seq_len={seq_len}")

        pos_logits = logits[0, position, :]
        chosen_id, alt_id = self.target_token_ids
        return pos_logits[chosen_id] - pos_logits[alt_id]

    def compute_at_position(
        self, logits: torch.Tensor, position: int
    ) -> torch.Tensor:
        """Compute metric at a specific position.

        Args:
            logits: Model output logits [batch=1, seq_len, vocab]
            position: Position to compute metric at

        Returns:
            Scalar tensor for gradient computation

        Raises:
            ValueError: If batch size != 1 or position out of bounds
        """
        if logits.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {logits.shape[0]}")

        seq_len = logits.shape[1]
        if position < 0:
            position = seq_len + position
        if position < 0 or position >= seq_len:
            raise ValueError(f"Position {position} out of bounds for seq_len={seq_len}")

        pos_logits = logits[0, position, :]
        chosen_id, alt_id = self.target_token_ids
        return pos_logits[chosen_id] - pos_logits[alt_id]

    @classmethod
    def from_contrastive_pair(
        cls,
        runner: BinaryChoiceRunner,
        contrastive_pair: ContrastivePair,
        mode: PatchingMode = "denoising",
    ) -> "AttributionMetric":
        """Create metric from a contrastive pair.

        The metric measures logit difference between corrupted and clean choices.
        The position at which to compute depends on the mode:

        - denoising: clean is base, compute at clean's divergent position
        - noising: corrupted is base, compute at corrupted's divergent position

        For both modes:
        - positive logit_diff favors corrupted choice
        - negative logit_diff favors clean choice

        Args:
            runner: BinaryChoiceRunner with tokenizer
            contrastive_pair: Pair with clean/corrupted trajectories
            mode: "denoising" (clean as base) or "noising" (corrupted as base)

        Returns:
            AttributionMetric configured for this pair and mode
        """
        # Get first token IDs for the choice labels
        # clean_labels = (short_term_label, long_term_label) for clean trajectory
        # corrupted_labels = (short_term_label, long_term_label) for corrupted trajectory
        # Clean trajectory chose short_term (index 0), corrupted chose long_term (index 1)
        if contrastive_pair.clean_labels:
            clean_label = contrastive_pair.clean_labels[0]  # The chosen label in clean
        else:
            clean_label = ""
        if contrastive_pair.corrupted_labels:
            corrupted_label = contrastive_pair.corrupted_labels[1]  # The chosen label in corrupted
        else:
            corrupted_label = ""

        clean_ids = runner.encode_ids(clean_label, add_special_tokens=False)
        corrupted_ids = runner.encode_ids(corrupted_label, add_special_tokens=False)

        if not clean_ids or not corrupted_ids:
            raise ValueError(
                f"Could not tokenize labels: clean='{clean_label}', corrupted='{corrupted_label}'"
            )

        if len(clean_ids) > 1 or len(corrupted_ids) > 1:
            warnings.warn(
                f"Labels tokenize to multiple tokens (clean={len(clean_ids)}, corrupted={len(corrupted_ids)}). "
                "Only the first token of each label will be used for attribution."
            )

        clean_id = clean_ids[0]
        corrupted_id = corrupted_ids[0]

        # Determine divergent position based on mode
        # The metric is computed at the divergent position of the BASE text
        # (the one we run with gradients)
        if mode == "denoising":
            # Base is clean, compute at clean's divergent position
            divergent_position = contrastive_pair.clean_divergent_position
            if divergent_position is None:
                divergent_position = -1  # fallback to last position
        else:
            # Base is corrupted, compute at corrupted's divergent position
            divergent_position = contrastive_pair.corrupted_divergent_position
            if divergent_position is None:
                divergent_position = -1

        # Compute logit differences
        # Measure: corrupted_logit - clean_logit
        # Positive = prefers corrupted, Negative = prefers clean
        clean_diff = _compute_logit_diff_at_position(
            runner,
            contrastive_pair.clean_text,
            corrupted_id,
            clean_id,
            divergent_position if mode == "denoising" else -1,
        )
        corrupted_diff = _compute_logit_diff_at_position(
            runner,
            contrastive_pair.corrupted_text,
            corrupted_id,
            clean_id,
            divergent_position if mode == "noising" else -1,
        )

        return cls(
            target_token_ids=(corrupted_id, clean_id),
            divergent_position=divergent_position,
            clean_logit_diff=clean_diff,
            corrupted_logit_diff=corrupted_diff,
        )


def _compute_logit_diff_at_position(
    runner: BinaryChoiceRunner,
    text: str,
    chosen_id: int,
    alt_id: int,
    position: int = -1,
) -> float:
    """Compute logit difference at a specific position.

    Args:
        runner: BinaryChoiceRunner
        text: Input text
        chosen_id: Token ID for chosen option
        alt_id: Token ID for alternative option
        position: Position to compute at (-1 for last)

    Returns:
        Logit difference (chosen - alternative)
    """
    with torch.no_grad():
        logits = runner.forward(text)
        if position < 0:
            position = logits.shape[1] + position
        pos_logits = logits[0, position, :]
        diff = pos_logits[chosen_id] - pos_logits[alt_id]
        return diff.item()
