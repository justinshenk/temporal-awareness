"""Token tree data structures.

Provides types for representing token sequences with logprobs, including
branching nodes and forks where outputs diverge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_schema import BaseSchema


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TokenTrajectory(BaseSchema):
    """A sequence of tokens with associated logprobs and logits.

    All arrays have length n_sequence (full sequence length).
    The first token has logprob=0 (probability 1, since it's given).
    Use pred_* properties to access the prediction-relevant portion [1:].
    """

    token_ids: list[int]  # [n_sequence] full sequence token IDs
    logprobs: list[float]  # [n_sequence] per-token log-probabilities (first is 0.0)
    logits: list[float]  # [n_sequence] per-token raw logits (first is 0.0)

    full_logits: nn.Tensor | None = None  # [n_sequence, vocab_size] full logits matrix

    nodes_idx: tuple[int, ...] | None = None
    analysis: Any | None = None

    def has_internals(self) -> bool:
        """Return True if this trajectory has captured internals."""
        return False

    # ── Sequence-length accessors (n_sequence) ───────────────────────────

    @property
    def n_sequence(self) -> int:
        """Full sequence length."""
        return len(self.token_ids)

    @property
    def sequence_length(self) -> int:
        """Alias for n_sequence."""
        return self.n_sequence

    @property
    def length(self) -> int:
        return self.n_sequence

    # ── Prediction accessors (n_pred = n_sequence - 1) ────────────────────

    @property
    def n_pred(self) -> int:
        """Number of predictions (n_sequence - 1)."""
        return max(0, self.n_sequence - 1)

    @property
    def predictions_length(self) -> int:
        """Alias for n_pred."""
        return self.n_pred

    @property
    def pred_token_ids(self) -> list[int]:
        """Token IDs being predicted [1:] (n_pred length)."""
        return self.token_ids[1:]

    @property
    def pred_logprobs(self) -> list[float]:
        """Logprobs for predictions [1:] (n_pred length)."""
        return self.logprobs[1:]

    @property
    def pred_logits(self) -> list[float]:
        """Logits for predictions [1:] (n_pred length)."""
        return self.logits[1:]

    @property
    def pred_full_logits(self) -> nn.Tensor | None:
        """Full logits matrix for predictions [1:] (n_pred, vocab_size)."""
        if self.full_logits is None:
            return None
        return self.full_logits[1:]

    @property
    def next_token_logprob_sequence(self) -> list[float]:
        """Alias for pred_logprobs (prediction logprobs)."""
        return self.pred_logprobs

    @property
    def branching_points(self) -> list[int]:
        """Token-position indices where this trajectory passes through a
        branching node.  Empty when the trajectory has no tree context."""
        if self.nodes_idx is None:
            return []
        # Requires access to the parent tree's node list — but we store the
        # positions directly via _BranchingNodePositions convention: the
        # TokenTree populates nodes_idx, and callers look up positions
        # through TokenTree.nodes[i].branching_token_position.
        # As a lightweight shortcut we stash positions at build time; see
        # _attach_nodes_to_trajectories.
        return list(getattr(self, "_branching_positions", []))

    def pop_full_logits(self) -> nn.Tensor | None:
        """Detach and return the full logits tensor, clearing it from self."""
        seq = self.full_logits
        self.full_logits = None
        return seq

    def to_dict(self) -> dict:
        full_logits = self.pop_full_logits()
        d = super().to_dict()
        self.full_logits = full_logits
        return d

    def get_conditional_prob(
        self, start_token_ids_pos: int, end_token_ids_pos: int
    ) -> float | None:
        if (
            start_token_ids_pos < 0
            or end_token_ids_pos > self.length
            or start_token_ids_pos >= end_token_ids_pos
        ):
            return None
        log_prob_sum = sum(self.logprobs[start_token_ids_pos:end_token_ids_pos])
        return math.exp(log_prob_sum)


@dataclass
class BranchingNode(BaseSchema):
    """A node where trajectories diverge, choosing different next tokens.

    Attributes:
        next_token_ids: Token IDs chosen by each branch at this divergence point
        next_token_logprobs: Log-probabilities for each branch's chosen token
        branching_token_position: Token position in the sequence where divergence occurs
        vocab_logits: Full logits over vocabulary at this position (from first branch)
        forks_idx: Indices into the parent tree's forks list
    """

    next_token_ids: tuple[int, ...]
    next_token_logprobs: tuple[float, ...]
    branching_token_position: int
    vocab_logits: list[float] | None = None
    forks_idx: tuple[int, ...] | None = None
    analysis: Any | None = None


@dataclass
class BinaryFork(BaseSchema):
    """A pairwise comparison between two branches at a divergence point.

    Attributes:
        next_token_ids: The two token IDs being compared (branch_a, branch_b)
        next_token_logprobs: Log-probabilities for each token
    """

    next_token_ids: tuple[int, int]
    next_token_logprobs: tuple[float, float]
    analysis: Any | None = None


@dataclass
class TokenTree:
    """A tree of token trajectories with branching points."""

    trajs: tuple[TokenTrajectory, ...]
    nodes: tuple[BranchingNode, ...] | None = None
    forks: tuple[BinaryFork, ...] | None = None

    @classmethod
    def from_trajectories(cls, trajs: tuple[TokenTrajectory, ...]) -> TokenTree:
        return parse_tree_from_trajs(trajs)

    def get_logits_at_node(self, node_idx: int, pos: int) -> torch.Tensor | None:
        """Retrieve logits at *pos* from the first trajectory passing through
        the node at *node_idx*."""
        for traj in self.trajs:
            if (
                traj.nodes_idx
                and node_idx in traj.nodes_idx
                and traj.full_logits is not None
            ):
                return traj.full_logits[pos]
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Internal Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class _Branch:
    """One arm of a divergence: a group of trajectories that share the same
    logits (and therefore the same next-token distribution) at a position."""

    logits: torch.Tensor
    traj_indices: list[int]
    token_id: int
    token_logprob: float


@dataclass
class _TreeAccumulator:
    """Single mutable store threaded through the recursive build so that
    every helper stays side-effect-free in its *logic*."""

    trajs: list[TokenTrajectory]
    nodes: list[BranchingNode] = field(default_factory=list)
    forks: list[BinaryFork] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Public Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def parse_tree_from_trajs(
    trajs: tuple[TokenTrajectory, ...],
) -> TokenTree:
    """Build a TokenTree by recursively finding divergence points
    across trajectories that share a common prefix."""
    acc = _TreeAccumulator(trajs=list(trajs))

    _build_subtree(acc, traj_indices=list(range(len(trajs))), depth=0)
    _attach_branching_positions(acc)

    return TokenTree(
        trajs=tuple(acc.trajs),
        nodes=tuple(acc.nodes),
        forks=tuple(acc.forks),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Recursive Builder
# ═══════════════════════════════════════════════════════════════════════════════


def _build_subtree(
    acc: _TreeAccumulator,
    traj_indices: list[int],
    depth: int,
) -> None:
    """Find the next divergence among *traj_indices*, register it, recurse.

    1. Scan forward to find the first position where token IDs differ.
    2. Group trajectories into branches by token ID (not logits).
    3. Register resulting forks, node, and node-membership on trajectories.
    4. Recurse into each branch independently.
    """
    if len(traj_indices) <= 1:
        return

    divergence_pos = _find_token_divergence_position(acc.trajs, traj_indices, depth)
    if divergence_pos is None:
        return

    branches = _group_by_token_id(acc.trajs, traj_indices, divergence_pos)
    _register_divergence(acc, branches, traj_indices, divergence_pos)

    for branch in branches:
        _build_subtree(acc, branch.traj_indices, depth=divergence_pos + 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Scanning
# ═══════════════════════════════════════════════════════════════════════════════


def _find_token_divergence_position(
    trajs: list[TokenTrajectory],
    traj_indices: list[int],
    start_depth: int,
) -> int | None:
    """Return the first position ≥ *start_depth* where at least two
    trajectories have different token IDs, or None if they never diverge."""
    horizon = min(trajs[i].length for i in traj_indices)

    for pos in range(start_depth, horizon):
        if not _all_tokens_match(trajs, traj_indices, pos):
            return pos
    return None


def _all_tokens_match(
    trajs: list[TokenTrajectory],
    traj_indices: list[int],
    pos: int,
) -> bool:
    """True when every trajectory has the same token ID at *pos*."""
    ref_token = trajs[traj_indices[0]].token_ids[pos]
    return all(trajs[i].token_ids[pos] == ref_token for i in traj_indices[1:])


def _find_divergence_position(
    trajs: list[TokenTrajectory],
    traj_indices: list[int],
    start_depth: int,
) -> int | None:
    """Return the first position ≥ *start_depth* where at least two
    trajectories have non-identical logits, or None if they never diverge."""
    horizon = min(trajs[i].length for i in traj_indices)

    for pos in range(start_depth, horizon):
        if not _all_logits_match(trajs, traj_indices, pos):
            return pos
    return None


def _all_logits_match(
    trajs: list[TokenTrajectory],
    traj_indices: list[int],
    pos: int,
) -> bool:
    """True when every trajectory has identical logits at *pos*."""
    ref = trajs[traj_indices[0]].full_logits[pos]
    return all(torch.equal(ref, trajs[i].full_logits[pos]) for i in traj_indices[1:])


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Grouping
# ═══════════════════════════════════════════════════════════════════════════════


def _group_by_logits(
    trajs: list[TokenTrajectory],
    traj_indices: list[int],
    pos: int,
) -> list[_Branch]:
    """Partition trajectories into branches that share the same logits at *pos*."""
    branches: list[_Branch] = []

    for idx in traj_indices:
        logits = trajs[idx].full_logits[pos]
        logprob = trajs[idx].logprobs[pos]

        existing = _find_matching_branch(branches, logits)
        if existing is not None:
            existing.traj_indices.append(idx)
        else:
            branches.append(
                _Branch(
                    logits=logits,
                    traj_indices=[idx],
                    token_id=_resolve_token_id(logits, logprob),
                    token_logprob=logprob,
                )
            )
    return branches


def _group_by_token_id(
    trajs: list[TokenTrajectory],
    traj_indices: list[int],
    pos: int,
) -> list[_Branch]:
    """Partition trajectories into branches that have the same token ID at *pos*.

    Unlike _group_by_logits, this groups by the actual chosen token, not the
    probability distribution. This is appropriate for forced-continuation
    trajectories where we want to identify divergence in token sequences.
    """
    branches: list[_Branch] = []
    token_to_branch: dict[int, _Branch] = {}

    for idx in traj_indices:
        token_id = trajs[idx].token_ids[pos]
        logits = trajs[idx].full_logits[pos]
        logprob = trajs[idx].logprobs[pos]

        if token_id in token_to_branch:
            token_to_branch[token_id].traj_indices.append(idx)
        else:
            branch = _Branch(
                logits=logits,
                traj_indices=[idx],
                token_id=token_id,
                token_logprob=logprob,
            )
            branches.append(branch)
            token_to_branch[token_id] = branch

    return branches


def _find_matching_branch(
    branches: list[_Branch], logits: torch.Tensor
) -> _Branch | None:
    """Return the first branch whose logits match *logits*, or None."""
    return next(
        (b for b in branches if torch.equal(b.logits, logits)),
        None,
    )


def _resolve_token_id(logits: torch.Tensor, target_logprob: float) -> int:
    """Identify the chosen token by finding the closest log-softmax match."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return (log_probs - target_logprob).abs().argmin().item()


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Registration
# ═══════════════════════════════════════════════════════════════════════════════


def _register_divergence(
    acc: _TreeAccumulator,
    branches: list[_Branch],
    traj_indices: list[int],
    pos: int,
) -> None:
    """Create forks + a branching node for *branches* and wire them into
    the accumulator and the affected trajectories."""
    fork_indices = _create_forks(acc, branches)
    node = _create_node(branches, pos, fork_indices)

    node_idx = len(acc.nodes)
    acc.nodes.append(node)

    _tag_trajectories(acc.trajs, traj_indices, node_idx)


def _create_forks(
    acc: _TreeAccumulator,
    branches: list[_Branch],
) -> tuple[int, ...]:
    """Create a BinaryFork for every pair of branches; return their indices."""
    indices: list[int] = []
    for i, a in enumerate(branches):
        for b in branches[i + 1 :]:
            indices.append(len(acc.forks))
            acc.forks.append(
                BinaryFork(
                    next_token_ids=(a.token_id, b.token_id),
                    next_token_logprobs=(a.token_logprob, b.token_logprob),
                )
            )
    return tuple(indices)


def _create_node(
    branches: list[_Branch],
    pos: int,
    fork_indices: tuple[int, ...],
) -> BranchingNode:
    # Extract vocab logits from first branch (all branches have same logits at divergence)
    vocab_logits = branches[0].logits.tolist() if branches else None

    return BranchingNode(
        next_token_ids=tuple(b.token_id for b in branches),
        next_token_logprobs=tuple(b.token_logprob for b in branches),
        branching_token_position=pos,
        vocab_logits=vocab_logits,
        forks_idx=fork_indices or None,
    )


def _tag_trajectories(
    trajs: list[TokenTrajectory],
    traj_indices: list[int],
    node_idx: int,
) -> None:
    """Record *node_idx* on every trajectory that passes through it."""
    for idx in traj_indices:
        existing = trajs[idx].nodes_idx or ()
        trajs[idx].nodes_idx = existing + (node_idx,)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Post-Processing
# ═══════════════════════════════════════════════════════════════════════════════


def _attach_branching_positions(acc: _TreeAccumulator) -> None:
    """Populate each trajectory's _branching_positions cache so that the
    `branching_points` property can return positions without a tree lookup."""
    for traj in acc.trajs:
        if traj.nodes_idx is None:
            traj._branching_positions = []
        else:
            traj._branching_positions = [
                acc.nodes[ni].branching_token_position for ni in traj.nodes_idx
            ]
