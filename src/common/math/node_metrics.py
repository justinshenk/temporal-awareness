"""Node-level metrics for vocabulary distributions.

"Node" = a probability distribution over alternatives at a single decision point.
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


# ── Generalized node metrics (order q) — most general ──────────────────────


def q_node_diversity(probs: Nums, q: float) -> Num:
    """Effective number of alternatives at this node (Hill number D_q).

    This is the central node metric: how many "real" choices exist?

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


def q_node_entropy(probs: Nums, q: float) -> Num:
    """Rényi entropy at this node (H_q).

    The "uncertainty" interpretation of diversity.
    For q=1, this is Shannon entropy H = -Σ pᵢ log pᵢ.

    Range: [0, log n]. Lower = more certain.
    """
    logprobs = probs_to_logprobs(probs)
    return renyi_entropy(logprobs, q)


def q_node_concentration(probs: Nums, q: float) -> Num:
    """How concentrated is this node? (1/D_q).

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


# ── Token Constraint Bound (TCB) ─────────────────────────────────────────────


def compute_tcb(
    hidden_state: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor | None = None,
    epsilon: float = 1.0,
) -> float:
    """Token Constraint Bound: maximum perturbation radius before token flip.

    The TCB measures how stable the current token prediction is. A larger TCB
    means the hidden state can be perturbed more before the predicted token
    changes.

    Formula:
        δ_TCB = ε / sqrt(Σᵢ pᵢ² ||wᵢ - μ_w||²)

    where:
        - pᵢ = softmax probability of token i
        - wᵢ = row i of W_U (unembedding weight for token i)
        - μ_w = Σᵢ pᵢ wᵢ (probability-weighted mean of unembedding vectors)
        - ε = perturbation scale (default 1.0)

    Args:
        hidden_state: Final hidden state tensor [d_model]
        output_weight: Unembedding weight matrix [d_model, vocab_size] (W_U)
        output_bias: Optional unembedding bias [vocab_size] (b_U)
        epsilon: Perturbation scale factor

    Returns:
        TCB value. Higher = more stable prediction.
    """
    h = hidden_state.detach().float()
    W = output_weight.detach().float()  # [d_model, vocab_size]

    # Compute logits: logits = W^T h + b
    logits = W.T @ h  # [vocab_size]
    if output_bias is not None:
        logits = logits + output_bias.detach().float()

    # Compute softmax probabilities
    probs = F.softmax(logits, dim=0)  # [vocab_size]

    # Compute probability-weighted mean of unembedding vectors
    # W is [d_model, vocab_size], probs is [vocab_size]
    mu = W @ probs  # [d_model]

    # Compute squared distance from mean for each token
    # diff[i] = W[:, i] - mu
    diff = W - mu.unsqueeze(1)  # [d_model, vocab_size]
    sq_dist = (diff ** 2).sum(dim=0)  # [vocab_size]

    # Jacobian norm squared: Σᵢ pᵢ² ||wᵢ - μ||²
    jac_norm_sq = (probs ** 2 * sq_dist).sum().item()

    # TCB = ε / sqrt(jac_norm_sq)
    return epsilon / (jac_norm_sq ** 0.5 + 1e-12)


