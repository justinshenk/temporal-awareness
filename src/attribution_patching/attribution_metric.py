"""Attribution metric for computing gradients of choice difference.

The metric computes a scalar value from logits that can be differentiated
to get attribution scores. For binary choices, this is typically the
logit difference between the two options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..common.base_schema import BaseSchema

if TYPE_CHECKING:
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
        clean_logit_diff: Logit difference on clean (target) input
        corrupted_logit_diff: Logit difference on corrupted (baseline) input
    """

    target_token_ids: tuple[int, int]
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

        Args:
            logits: Model output logits [batch, seq_len, vocab]

        Returns:
            Scalar tensor for gradient computation
        """
        # Get logits at last position
        last_logits = logits[0, -1, :]
        chosen_id, alt_id = self.target_token_ids
        return last_logits[chosen_id] - last_logits[alt_id]

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
        runner: "BinaryChoiceRunner",
        contrastive_pair: "ContrastivePair",
    ) -> "AttributionMetric":
        """Create metric from a contrastive pair.

        The metric measures logit difference between short and long choices.
        For attribution to work correctly:
        - clean = long (target behavior we want)
        - corrupted = short (baseline behavior)

        Args:
            runner: BinaryChoiceRunner with tokenizer
            contrastive_pair: Pair with short/long trajectories

        Returns:
            AttributionMetric configured for this pair
        """
        tokenizer = runner.tokenizer

        # Get first token IDs for the choice labels
        short_label = contrastive_pair.short_label or ""
        long_label = contrastive_pair.long_label or ""

        short_ids = tokenizer.encode(short_label, add_special_tokens=False)
        long_ids = tokenizer.encode(long_label, add_special_tokens=False)

        if not short_ids or not long_ids:
            raise ValueError(
                f"Could not tokenize labels: short='{short_label}', long='{long_label}'"
            )

        short_id = short_ids[0]
        long_id = long_ids[0]

        # Compute logit differences
        # For clean (long text), we want long_logit - short_logit
        # For corrupted (short text), we measure the same difference
        clean_diff = _compute_logit_diff(
            runner, contrastive_pair.long_text, long_id, short_id
        )
        corrupted_diff = _compute_logit_diff(
            runner, contrastive_pair.short_text, long_id, short_id
        )

        return cls(
            target_token_ids=(long_id, short_id),
            clean_logit_diff=clean_diff,
            corrupted_logit_diff=corrupted_diff,
        )


def _compute_logit_diff(
    runner: "BinaryChoiceRunner",
    text: str,
    chosen_id: int,
    alt_id: int,
) -> float:
    """Compute logit difference for given text.

    Args:
        runner: BinaryChoiceRunner
        text: Input text
        chosen_id: Token ID for chosen option
        alt_id: Token ID for alternative option

    Returns:
        Logit difference (chosen - alternative)
    """
    with torch.no_grad():
        logits = runner.forward(text)
        last_logits = logits[0, -1, :]
        diff = last_logits[chosen_id] - last_logits[alt_id]
        return diff.item()
