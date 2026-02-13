"""Generated trajectory from model inference.

Provides the GeneratedTrajectory class (extends TokenTrajectory with internals capture)
and utility functions for creating trajectories from forward pass outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..common.token_tree import TokenTrajectory


@dataclass
class GeneratedTrajectory(TokenTrajectory):
    """TokenTrajectory generated from model inference, optionally with captured internals.

    Used by ModelRunner when running inference. The internals dict can store
    captured activations from the forward pass.

    Prefer using GeneratedTrajectory.from_inference() which handles the
    logprob computation from logits automatically.
    """

    internals: dict = field(default_factory=dict)

    def has_internals(self) -> bool:
        return bool(self.internals)

    @classmethod
    def from_inference(
        cls,
        token_ids: list[int],
        logits: torch.Tensor,
        device: str = "cpu",
        internals: dict | None = None,
    ) -> GeneratedTrajectory:
        """Build a GeneratedTrajectory from inference outputs.

        Takes the FULL token_ids sequence (n_sequence length). Computes logprobs
        from the logits tensor. The first token has logprob=0 (probability 1).

        Args:
            token_ids: Full sequence of token IDs [n_sequence]
            logits: Full logits tensor [n_sequence, vocab_size]
            device: Device to use for tensor operations
            internals: Optional dict of captured internals from forward pass

        Returns:
            GeneratedTrajectory with n_sequence length arrays
        """
        n_sequence = len(token_ids)
        n_pred = n_sequence - 1

        # First token: probability 1, logprob = 0, logit = 0
        all_logprobs = [0.0]
        all_logits = [0.0]

        if n_pred > 0:
            # Prediction logits: logits[i] predicts token_ids[i+1]
            pred_full_logits = logits[:-1]  # [n_pred, vocab_size]
            target_ids = torch.tensor(token_ids[1:], device=device)  # [n_pred]
            indices = torch.arange(n_pred, device=device)

            # Gather scalar logits for each target token
            gathered_logits = pred_full_logits[indices, target_ids]  # [n_pred]

            # Numerically stable log-softmax, then gather scalar logprobs
            pred_logprobs = torch.log_softmax(
                pred_full_logits, dim=-1
            )  # [n_pred, vocab_size]
            gathered_logprobs = pred_logprobs[indices, target_ids]  # [n_pred]

            all_logprobs.extend(gathered_logprobs.tolist())
            all_logits.extend(gathered_logits.tolist())

        # Build full_logits: first position gets zeros, rest from input logits
        first_logits = torch.zeros(
            1, logits.shape[-1], device=logits.device, dtype=logits.dtype
        )
        full_logits_out = (
            torch.cat([first_logits, logits[:-1]], dim=0)
            if n_pred > 0
            else first_logits
        )

        return cls(
            token_ids=token_ids,
            logprobs=all_logprobs,
            logits=all_logits,
            full_logits=full_logits_out,
            internals=internals or {},
        )

    @classmethod
    def from_token_trajectory(
        cls,
        trajectory: TokenTrajectory,
        internals: dict | None = None,
    ) -> GeneratedTrajectory:
        """Create from existing TokenTrajectory plus optional internals cache."""
        return cls(
            token_ids=trajectory.token_ids,
            logprobs=trajectory.logprobs,
            logits=trajectory.logits,
            full_logits=trajectory.full_logits,
            nodes_idx=trajectory.nodes_idx,
            analysis=trajectory.analysis,
            internals=internals or {},
        )


def calculate_trajectory_from_logits(
    token_ids: list[int],
    logits: torch.Tensor,
    device: str = "cpu",
) -> GeneratedTrajectory:
    """Build a GeneratedTrajectory from a forward pass.

    DEPRECATED: Use GeneratedTrajectory.from_inference() instead.
    """
    return GeneratedTrajectory.from_inference(token_ids, logits, device)


def calculate_trajectories_for_batch(
    token_ids_batch: list[list[int]],
    logits_batch: torch.Tensor,
    device: str = "cpu",
) -> list[GeneratedTrajectory]:
    """Build trajectories for a batch, trimming padding per sequence.

    Args:
        token_ids_batch: List of token ID sequences (variable length, each n_sequence)
        logits_batch: Padded logits tensor [batch, max_seq_len, vocab_size]
        device: Device to use for tensor operations

    Returns:
        List of GeneratedTrajectory, one per batch item (each with n_sequence length)
    """
    trajectories = []
    for i, token_ids in enumerate(token_ids_batch):
        n_sequence = len(token_ids)
        logits = logits_batch[i, :n_sequence]
        traj = GeneratedTrajectory.from_inference(token_ids, logits, device)
        trajectories.append(traj)
    return trajectories
