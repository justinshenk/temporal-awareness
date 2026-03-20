"""Builder functions for analysis objects.

Provides functions to build analysis objects from tree components.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..math import (
    compute_tcb,
    log_odds,
    logprob_to_prob,
    probability_ratio,
    q_diversity,
    q_fork_concentration,
    q_fork_diversity,
    q_fork_entropy,
    vocab_entropy_from_logits,
)
from .metrics import (
    ForkMetrics,
    ForkAnalysis,
    NodeMetrics,
    NodeAnalysis,
)

if TYPE_CHECKING:
    from ..binary_fork import BinaryFork
    from ..branching_node import BranchingNode
    from ..token_tree import TokenTree


def build_fork_analysis(fork_idx: int, fork: "BinaryFork") -> ForkAnalysis:
    """Build analysis for a binary fork.

    Args:
        fork_idx: Index of the fork in the tree
        fork: The BinaryFork to analyze

    Returns:
        ForkAnalysis containing fork metrics
    """
    lp_a, lp_b = fork.next_token_logprobs
    p_a, p_b = logprob_to_prob(lp_a), logprob_to_prob(lp_b)

    # Compute raw logits and normalized logits if vocab_logits available
    logits = None
    normalized_logits = None
    fork_vocab_logits = getattr(fork, "vocab_logits", None)
    if fork_vocab_logits is not None:
        vocab_logits = torch.tensor(fork_vocab_logits)
        token_a, token_b = fork.next_token_ids
        logit_a = vocab_logits[token_a].item()
        logit_b = vocab_logits[token_b].item()
        logits = (logit_a, logit_b)

        # Z-score normalization
        mean = vocab_logits.mean().item()
        std = vocab_logits.std().item()
        if std > 1e-8:
            norm_a = (logit_a - mean) / std
            norm_b = (logit_b - mean) / std
            normalized_logits = (norm_a, norm_b)

    return ForkAnalysis(
        fork_idx=fork_idx,
        metrics=ForkMetrics(
            next_token_logprobs=(lp_a, lp_b),
            fork_entropy=q_fork_entropy(p_a, p_b, q=1.0),
            fork_diversity=q_fork_diversity(p_a, p_b, q=1.0),
            fork_simpson=q_fork_diversity(p_a, p_b, q=2.0),
            fork_concentration=q_fork_concentration(p_a, p_b, q=1.0),
            probability_ratio=probability_ratio(p_a, p_b),
            log_odds=log_odds(p_a, p_b),
            logit_diff=lp_a - lp_b,
            reciprocal_rank_a=1.0 if lp_a >= lp_b else 0.5,
            logits=logits,
            normalized_logits=normalized_logits,
        ),
    )


def build_node_analysis(
    node_idx: int,
    node: "BranchingNode",
    tree: "TokenTree",
    W_U: torch.Tensor | None = None,
    b_U: torch.Tensor | None = None,
) -> NodeAnalysis:
    """Build analysis for a branching node.

    Args:
        node_idx: Index of the node in the tree
        node: The BranchingNode to analyze
        tree: The parent TokenTree (for logits lookup)
        W_U: Optional unembedding matrix [d_model, vocab_size] for TCB computation
        b_U: Optional unembedding bias [vocab_size] for TCB computation

    Returns:
        NodeAnalysis containing node metrics
    """
    next_token_logprobs = [float(lp) for lp in node.next_token_logprobs]

    # Use vocab_logits stored on node if available, fallback to tree lookup
    logits = None
    if node.vocab_logits is not None:
        logits = torch.tensor(node.vocab_logits)
    else:
        pos = node.branching_token_position
        logits = tree.get_logits_at_node(node_idx, pos)

    if logits is not None:
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        v_entropy = vocab_entropy_from_logits(logits).item()
        v_simpson = q_diversity(log_probs, q=2.0).item()
    else:
        v_entropy = 0.0
        v_simpson = 1.0

    # Compute TCB if we have the unembedding matrix and hidden state
    v_tcb = 0.0
    if W_U is not None:
        # Try to get hidden state from trajectory internals
        pos = node.branching_token_position
        hidden_state = None

        # Look for resid_post activation in trajectory internals
        if tree.trajs and node.traj_idx:
            traj = tree.trajs[node.traj_idx[0]]
            if hasattr(traj, "internals") and traj.internals:
                # Look for the last layer's residual stream activation
                for key in traj.internals:
                    if "resid_post" in key or "resid_pre" in key:
                        act = traj.internals[key]
                        if act.ndim == 3:  # [batch, seq, d_model]
                            hidden_state = act[0, pos, :]
                        elif act.ndim == 2:  # [seq, d_model]
                            hidden_state = act[pos, :]
                        break

        if hidden_state is not None:
            v_tcb = compute_tcb(hidden_state, W_U, b_U)

    return NodeAnalysis(
        node_idx=node_idx,
        metrics=NodeMetrics(
            next_token_logprobs=next_token_logprobs,
            vocab_entropy=v_entropy,
            vocab_diversity=math.exp(v_entropy),
            vocab_simpson=v_simpson,
            vocab_tcb=v_tcb,
        ),
    )
