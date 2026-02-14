"""Branch-level metrics for vocabulary distributions.

"Branch" = a probability distribution over alternatives at a single decision point.
For LLMs, this is the next-token distribution at a single position.

Input: a proper distribution p = [p₁, …, pₙ] with Σpᵢ = 1
       (or logits that can be converted to one)

These metrics wrap the core entropy_diversity functions, converting probs
to logprobs internally for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .num_types import Num, Nums
from .entropy_diversity import (
    probs_to_logprobs,
    q_diversity,
    q_concentration,
    renyi_entropy,
    shannon_entropy,
)


# ── Generalized branch metrics (order q) — most general ──────────────────────


def q_branch_diversity(probs: Nums, q: float) -> Num:
    """Effective number of alternatives at this branch (Hill number D_q).

    This is the central branch metric: how many "real" choices exist?

    Args:
        probs: Distribution over alternatives (will be converted to logprobs)
        q: Order parameter
            q=0: count all non-zero options (richness)
            q=1: Shannon diversity exp(H)
            q=2: Simpson diversity 1/Σpᵢ²
            q→∞: 1/max(p) (dominated by most likely)

    Returns:
        Effective number in [1, n]. Higher = more choices available.

    Examples:
        [0.5, 0.5] → 2.0 (two equally likely options)
        [0.9, 0.1] → ~1.5 (one dominant option)
        [1.0, 0.0] → 1.0 (no real choice)
    """
    logprobs = probs_to_logprobs(probs)
    return q_diversity(logprobs, q)


def q_branch_entropy(probs: Nums, q: float) -> Num:
    """Rényi entropy at this branch (H_q).

    The "uncertainty" interpretation of diversity.
    For q=1, this is Shannon entropy H = -Σ pᵢ log pᵢ.

    Range: [0, log n]. Lower = more certain.
    """
    logprobs = probs_to_logprobs(probs)
    return renyi_entropy(logprobs, q)


def q_branch_concentration(probs: Nums, q: float) -> Num:
    """How concentrated is this branch? (1/D_q).

    Range: [1/n, 1]. Higher = more concentrated on few options.
    """
    logprobs = probs_to_logprobs(probs)
    return q_concentration(logprobs, q)


# ── Logits-based utilities (for raw model outputs) ───────────────────────────


def vocab_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the full vocabulary distribution from logits.

    Convenience wrapper: applies log_softmax then calls shannon_entropy.

    Args:
        logits: Raw model output logits (before softmax)

    Returns:
        Entropy in nats. Range: [0, log |V|].
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return shannon_entropy(log_probs)
