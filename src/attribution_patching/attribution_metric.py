"""Attribution metric for computing gradients of choice difference.

The metric computes a scalar value from logits that can be differentiated
to get attribution scores. For binary choices, this is typically the
logit difference between the two options at the divergent position.

IMPORTANT: The metric must be computed at the correct position:
- For denoising (short base): compute at short's divergent position
- For noising (long base): compute at long's divergent position
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..common.base_schema import BaseSchema
from ..binary_choice import BinaryChoiceRunner
from ..common.contrastive_pair import ContrastivePair


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
            logits: Model output logits [batch, seq_len, vocab]

        Returns:
            Scalar tensor for gradient computation
        """
        position = self.divergent_position
        if position < 0:
            # Handle negative indexing (e.g., -1 for last position)
            position = logits.shape[1] + position

        pos_logits = logits[0, position, :]
        chosen_id, alt_id = self.target_token_ids
        return pos_logits[chosen_id] - pos_logits[alt_id]

    def compute_at_position(
        self, logits: torch.Tensor, position: int
    ) -> torch.Tensor:
        """Compute metric at a specific position.

        Args:
            logits: Model output logits [batch, seq_len, vocab]
            position: Position to compute metric at

        Returns:
            Scalar tensor for gradient computation
        """
        pos_logits = logits[0, position, :]
        chosen_id, alt_id = self.target_token_ids
        return pos_logits[chosen_id] - pos_logits[alt_id]

    @classmethod
    def from_contrastive_pair(
        cls,
        runner: BinaryChoiceRunner,
        contrastive_pair: ContrastivePair,
        mode: Literal["denoising", "noising"] = "denoising",
    ) -> "AttributionMetric":
        """Create metric from a contrastive pair.

        The metric measures logit difference between long and short choices.
        The position at which to compute depends on the mode:

        - denoising: short is base, compute at short's divergent position
        - noising: long is base, compute at long's divergent position

        For both modes:
        - positive logit_diff favors long-term choice
        - negative logit_diff favors short-term choice

        Args:
            runner: BinaryChoiceRunner with tokenizer
            contrastive_pair: Pair with short/long trajectories
            mode: "denoising" (short as base) or "noising" (long as base)

        Returns:
            AttributionMetric configured for this pair and mode
        """
        # Get first token IDs for the choice labels
        short_label = contrastive_pair.short_label or ""
        long_label = contrastive_pair.long_label or ""

        short_ids = runner.encode_ids(short_label, add_special_tokens=False)
        long_ids = runner.encode_ids(long_label, add_special_tokens=False)

        if not short_ids or not long_ids:
            raise ValueError(
                f"Could not tokenize labels: short='{short_label}', long='{long_label}'"
            )

        short_id = short_ids[0]
        long_id = long_ids[0]

        # Determine divergent position based on mode
        # The metric is computed at the divergent position of the BASE text
        # (the one we run with gradients)
        if mode == "denoising":
            # Base is short, compute at short's divergent position
            divergent_position = contrastive_pair.short_divergent_position
            if divergent_position is None:
                divergent_position = -1  # fallback to last position
        else:
            # Base is long, compute at long's divergent position
            divergent_position = contrastive_pair.long_divergent_position
            if divergent_position is None:
                divergent_position = -1

        # Compute logit differences
        # Always measure: long_logit - short_logit
        # This makes positive values = prefers long, negative = prefers short
        clean_diff = _compute_logit_diff_at_position(
            runner,
            contrastive_pair.long_text,
            long_id,
            short_id,
            divergent_position if mode == "noising" else -1,
        )
        corrupted_diff = _compute_logit_diff_at_position(
            runner,
            contrastive_pair.short_text,
            long_id,
            short_id,
            divergent_position if mode == "denoising" else -1,
        )

        return cls(
            target_token_ids=(long_id, short_id),
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
