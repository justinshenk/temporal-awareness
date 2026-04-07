"""Attribution metric for computing gradients of choice difference.

The metric computes a scalar value from logits that can be differentiated
to get attribution scores. For binary choices, this is typically the
logit difference between the two options at the divergent position.
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
        clean_logit_diff: Logit difference on clean input (filled by run_attribution)
        corrupted_logit_diff: Logit difference on corrupted input (filled by run_attribution)
    """

    target_token_ids: tuple[int, int]
    divergent_position: int = -1  # -1 means last position
    clean_logit_diff: float = 0.0
    corrupted_logit_diff: float = 0.0

    @property
    def diff(self) -> float:
        """Difference between clean and corrupted metric values."""
        return self.clean_logit_diff - self.corrupted_logit_diff

    def compute_raw(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute scalar metric from logits (must be differentiable).

        Returns the logit difference at the divergent position.

        Args:
            logits: Model output logits [batch=1, seq_len, vocab]

        Returns:
            Scalar tensor for gradient computation
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

    @classmethod
    def from_contrastive_pair(
        cls,
        runner: BinaryChoiceRunner,
        contrastive_pair: ContrastivePair,
        mode: PatchingMode = "denoising",
    ) -> "AttributionMetric":
        """Create metric from a contrastive pair.

        Extracts token IDs and divergent position only. Logit diffs are
        computed later by run_attribution from the forward pass caches.

        Args:
            runner: BinaryChoiceRunner with tokenizer
            contrastive_pair: Pair with clean/corrupted trajectories
            mode: "denoising" or "noising"

        Returns:
            AttributionMetric with token IDs and position (logit diffs = 0)
        """
        # Get first token IDs for the choice labels
        if contrastive_pair.clean_labels:
            clean_label = contrastive_pair.clean_labels[0]
        else:
            clean_label = ""
        if contrastive_pair.corrupted_labels:
            corrupted_label = contrastive_pair.corrupted_labels[1]
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
                "Only the first token will be used."
            )

        clean_id = clean_ids[0]
        corrupted_id = corrupted_ids[0]

        # Divergent position depends on mode
        if mode == "denoising":
            divergent_position = contrastive_pair.clean_divergent_position
        else:
            divergent_position = contrastive_pair.corrupted_divergent_position
        if divergent_position is None:
            divergent_position = -1

        # Logit diffs are computed by run_attribution from caches
        return cls(
            target_token_ids=(corrupted_id, clean_id),
            divergent_position=divergent_position,
            clean_logit_diff=0.0,
            corrupted_logit_diff=0.0,
        )
