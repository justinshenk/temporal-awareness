"""Token tree data structures.

Provides TokenTree for representing branching token sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .token_trajectory import TokenTrajectory
from .branching_node import BranchingNode
from .binary_fork import BinaryFork


@dataclass
class TokenTree:
    """A tree of token trajectories with branching points."""

    trajs: tuple[TokenTrajectory, ...]
    nodes: tuple[BranchingNode, ...] | None = None
    forks: tuple[BinaryFork, ...] | None = None
    fork_arms: tuple[tuple[int, int], ...] | None = (
        None  # Which group pairs create forks
    )
    trunk_length: int | None = None  # Length of shared trunk (prompt tokens)
    analysis: Any | None = None  # Set by analyze_token_tree

    @classmethod
    def from_trajectories(
        cls,
        trajs: Sequence[TokenTrajectory],
        groups_per_traj: Sequence[Sequence[int]] | None = None,
        fork_arms: Sequence[tuple[int, int]] | None = None,
        trunk: Sequence[int] | None = None,
    ) -> TokenTree:
        """Build a TokenTree from trajectories with group assignments.

        Args:
            trajs: Sequence of trajectories.
            groups_per_traj: For each trajectory, which groups it belongs to.
                If None, all trajectories are in group 0 with no forks.
            fork_arms: Which group pairs should create forks when they diverge.
                If None, no forks are created (forks are opt-in).

        Returns:
            TokenTree with nodes at divergence points and forks between
            trajectories from specified group pairs.
        """
        return parse_tree_from_trajs(trajs, groups_per_traj, fork_arms, trunk)

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

    def add_trajectory(
        self, traj: TokenTrajectory, group_idx: Sequence[int]
    ) -> TokenTree:
        """Add a trajectory and return a new tree.

        Args:
            traj: New trajectory to add
            group_idx: Which groups this trajectory belongs to

        Returns:
            New TokenTree with the trajectory added, nodes/forks recalculated
        """
        return add_trajectory_to_tree(self, traj, group_idx)

    def add_fork_between_groups(self, fork_arm: tuple[int, int]) -> TokenTree:
        """Add a fork relationship between two groups and return a new tree.

        Args:
            fork_arm: Tuple of (group_a, group_b) that should create forks

        Returns:
            New TokenTree with the fork relationship added
        """
        return add_fork_between_groups(self, fork_arm)

    @property
    def groups(self) -> tuple[int, ...]:
        """All unique group indices in this tree, sorted."""
        all_groups: set[int] = set()
        for traj in self.trajs:
            if traj.group_idx:
                all_groups.update(traj.group_idx)
        return tuple(sorted(all_groups))

    @property
    def n_groups(self) -> int:
        """Number of unique groups in this tree."""
        return len(self.groups)

    def pop_heavy(self) -> None:
        """Pop full_logits from all trajectories.

        Returns:
            Dict mapping trajectory index to its full_logits tensor.
            Only includes trajectories that had full_logits.
        """
        for traj in self.trajs:
            traj.pop_heavy()

    @classmethod
    def from_dict(cls, d: dict) -> "TokenTree":
        """Construct a TokenTree from a dictionary representation.

        Handles nested deserialization of trajectories, nodes, and forks.
        """
        trajs = []
        for traj_dict in d.get("trajs", []):
            if isinstance(traj_dict, TokenTrajectory):
                trajs.append(traj_dict)
            else:
                trajs.append(TokenTrajectory.from_dict(traj_dict))

        nodes = None
        if "nodes" in d and d["nodes"]:
            nodes = tuple(
                BranchingNode.from_dict(n) if isinstance(n, dict) else n
                for n in d["nodes"]
            )

        forks = None
        if "forks" in d and d["forks"]:
            forks = tuple(
                BinaryFork.from_dict(f) if isinstance(f, dict) else f
                for f in d["forks"]
            )

        fork_arms = None
        if "fork_arms" in d and d["fork_arms"]:
            fork_arms = tuple(tuple(arm) for arm in d["fork_arms"])

        return cls(
            trajs=tuple(trajs),
            nodes=nodes,
            forks=forks,
            fork_arms=fork_arms,
            trunk_length=d.get("trunk_length"),
            analysis=d.get("analysis"),
        )


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
    traj_to_groups: list[tuple[int, ...]] = field(
        default_factory=list
    )  # traj_idx -> groups
    fork_arms: list[tuple[int, int]] = field(
        default_factory=list
    )  # Which group pairs create forks


@dataclass
class _BranchWithGroups:
    """Branch info with group membership for cross-group fork creation."""

    token_id: int
    token_logprob: float
    groups: set[int]  # Which groups have trajectories in this branch


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Public Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def parse_tree_from_trajs(
    trajs: Sequence[TokenTrajectory],
    groups_per_traj: Sequence[Sequence[int]] | None = None,
    fork_arms: Sequence[tuple[int, int]] | None = None,
    trunk: Sequence[int] | None = None,
) -> TokenTree:
    """Build a TokenTree from trajectories with group assignments.

    Args:
        trajs: Sequence of trajectories.
        groups_per_traj: For each trajectory, which groups it belongs to.
            If None, no groups are assigned and no forks are created.
        fork_arms: Which group pairs should create forks when they diverge.
            Each tuple (g_i, g_j) means: create forks when trajectories from
            group g_i diverge from trajectories in group g_j.
            If None, no forks are created (forks are opt-in).

    Returns:
        TokenTree with nodes at divergence points and forks between
        trajectories from specified group pairs.
    """
    if not trajs:
        return TokenTree(trajs=(), nodes=(), forks=(), fork_arms=())

    trajs_list = list(trajs)

    # Handle groups_per_traj
    if groups_per_traj is None:
        # No groups, no forks
        traj_to_groups: list[tuple[int, ...]] = []
        resolved_fork_arms: list[tuple[int, int]] = []
    else:
        if len(groups_per_traj) != len(trajs_list):
            raise ValueError(
                f"groups_per_traj length ({len(groups_per_traj)}) must match "
                f"trajs length ({len(trajs_list)})"
            )
        traj_to_groups = [tuple(g) for g in groups_per_traj]

        # Determine fork_arms (no forks if not specified)
        resolved_fork_arms = list(fork_arms) if fork_arms else []

    # Set group_idx on each trajectory
    for traj, groups in zip(trajs_list, traj_to_groups):
        traj.group_idx = groups

    acc = _TreeAccumulator(
        trajs=trajs_list,
        traj_to_groups=traj_to_groups,
        fork_arms=resolved_fork_arms,
    )

    # Build nodes only (no forks during this phase)
    _build_subtree_nodes_only(acc, list(range(len(trajs_list))), depth=0)

    # Create forks for specified fork_arms
    _create_forks_for_arms(acc)

    _attach_branching_positions(acc)

    return TokenTree(
        trajs=tuple(acc.trajs),
        nodes=tuple(acc.nodes),
        forks=tuple(acc.forks),
        fork_arms=tuple(resolved_fork_arms),
        trunk_length=len(trunk) if trunk else None,
    )


def add_trajectory_to_tree(
    tree: TokenTree,
    traj: TokenTrajectory,
    group_idx: Sequence[int],
) -> TokenTree:
    """Add a trajectory to an existing tree.

    Args:
        tree: Existing TokenTree
        traj: New trajectory to add
        group_idx: Which groups this trajectory belongs to

    Returns:
        New TokenTree with the trajectory added, nodes/forks recalculated
    """
    # Collect existing trajectories and their groups
    existing_trajs = list(tree.trajs)
    groups_per_traj: list[tuple[int, ...]] = [t.group_idx or () for t in existing_trajs]

    # Find existing groups
    existing_groups: set[int] = set()
    for groups in groups_per_traj:
        existing_groups.update(groups)

    # Add new trajectory
    new_groups = tuple(group_idx)
    existing_trajs.append(traj)
    groups_per_traj.append(new_groups)

    # Start with existing fork_arms
    fork_arms = list(tree.fork_arms) if tree.fork_arms else []

    # For any new groups, add fork_arms with existing groups
    for g_new in new_groups:
        if g_new not in existing_groups:
            # This is a new group - add fork_arms with all existing groups
            for g_existing in sorted(existing_groups):
                arm = (g_existing, g_new) if g_existing < g_new else (g_new, g_existing)
                if arm not in fork_arms:
                    fork_arms.append(arm)

    return parse_tree_from_trajs(existing_trajs, groups_per_traj, fork_arms, trunk=None)


def add_fork_between_groups(
    tree: TokenTree,
    fork_arm: tuple[int, int],
) -> TokenTree:
    """Add a fork relationship between two groups.

    Args:
        tree: Existing TokenTree
        fork_arm: Tuple of (group_a, group_b) that should create forks

    Returns:
        New TokenTree with the fork relationship added
    """
    # Collect existing data
    trajs = list(tree.trajs)
    groups_per_traj = [t.group_idx or () for t in trajs]

    # Add new fork_arm
    existing_arms = list(tree.fork_arms) if tree.fork_arms else []
    if fork_arm not in existing_arms:
        existing_arms.append(fork_arm)

    return parse_tree_from_trajs(trajs, groups_per_traj, existing_arms, trunk=None)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Recursive Builder
# ═══════════════════════════════════════════════════════════════════════════════


def _build_subtree_nodes_only(
    acc: _TreeAccumulator,
    traj_indices: list[int],
    depth: int,
) -> None:
    """Like _build_subtree but does NOT create forks during traversal.

    Used by parse_tree_from_trajs to separate node creation from
    fork creation, allowing forks to be created only between specified groups.
    """
    if len(traj_indices) <= 1:
        return

    divergence_pos = _find_token_divergence_position(acc.trajs, traj_indices, depth)
    if divergence_pos is None:
        return

    branches = _group_by_token_id(acc.trajs, traj_indices, divergence_pos)
    _register_divergence_node_only(acc, branches, traj_indices, divergence_pos)

    for branch in branches:
        _build_subtree_nodes_only(acc, branch.traj_indices, depth=divergence_pos + 1)


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Grouping
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Tree Parsing — Registration
# ═══════════════════════════════════════════════════════════════════════════════


def _register_divergence_node_only(
    acc: _TreeAccumulator,
    branches: list[_Branch],
    traj_indices: list[int],
    pos: int,
) -> None:
    """Create a branching node WITHOUT forks.

    Used by _build_subtree_nodes_only for grouped trajectory parsing.
    Forks are created later in a post-pass by _create_forks_for_arms.
    """
    # Create node with no forks (forks_idx=None)
    node = _create_node(acc.trajs, branches, traj_indices, pos, fork_indices=[])

    node_idx = len(acc.nodes)
    acc.nodes.append(node)

    _tag_trajectories(acc.trajs, traj_indices, node_idx)


def _create_node(
    trajs: list[TokenTrajectory],
    branches: list[_Branch],
    traj_indices: list[int],
    pos: int,
    fork_indices: list[int],
) -> BranchingNode:
    """Create a BranchingNode with logits from all trajectories."""
    # Collect vocab logits from each trajectory at this position
    vocab_logits: list[list[float]] = []
    for idx in traj_indices:
        traj = trajs[idx]
        if traj.full_logits is not None and pos < len(traj.full_logits):
            vocab_logits.append(traj.full_logits[pos].tolist())

    return BranchingNode(
        next_token_ids=tuple(b.token_id for b in branches),
        next_token_logprobs=tuple(b.token_logprob for b in branches),
        branching_token_position=pos,
        traj_idx=list(traj_indices),
        vocab_logits=vocab_logits if vocab_logits else None,
        forks_idx=fork_indices if fork_indices else None,
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


def _create_forks_for_arms(acc: _TreeAccumulator) -> None:
    """Create forks for all specified fork_arms.

    Iterates over all nodes and creates forks between branches that contain
    trajectories from groups specified in fork_arms.
    """
    for node_idx, node in enumerate(acc.nodes):
        # Build branch info with group membership
        branches_with_groups = _get_branches_with_groups(acc, node)

        if len(branches_with_groups) < 2:
            continue

        # Create forks only for specified fork_arms
        fork_indices = _create_forks_for_node(acc, branches_with_groups)

        # Update node's forks_idx
        if fork_indices:
            acc.nodes[node_idx] = BranchingNode(
                next_token_ids=node.next_token_ids,
                next_token_logprobs=node.next_token_logprobs,
                branching_token_position=node.branching_token_position,
                traj_idx=node.traj_idx,
                vocab_logits=node.vocab_logits,
                forks_idx=fork_indices,
                analysis=node.analysis,
            )


def _get_branches_with_groups(
    acc: _TreeAccumulator,
    node: BranchingNode,
) -> list[_BranchWithGroups]:
    """Extract branch info with group membership from a node.

    For each unique token_id at the node's position, determine which groups
    have trajectories that chose that token.
    """
    pos = node.branching_token_position
    token_to_info: dict[int, _BranchWithGroups] = {}

    for traj_idx, traj in enumerate(acc.trajs):
        # Check if this trajectory passes through this node
        if traj.nodes_idx is None:
            continue

        # Check if this trajectory has a token at this position
        if pos >= len(traj.token_ids):
            continue

        token_id = traj.token_ids[pos]

        # Get this trajectory's groups
        groups = acc.traj_to_groups[traj_idx] if acc.traj_to_groups else (0,)

        if token_id not in token_to_info:
            # Get logprob for this token at this position
            logprob = traj.logprobs[pos] if pos < len(traj.logprobs) else 0.0
            token_to_info[token_id] = _BranchWithGroups(
                token_id=token_id,
                token_logprob=logprob,
                groups=set(groups),
            )
        else:
            token_to_info[token_id].groups.update(groups)

    return list(token_to_info.values())


def _create_forks_for_node(
    acc: _TreeAccumulator,
    branches: list[_BranchWithGroups],
) -> list[int]:
    """Create forks between branches for specified fork_arms.

    Returns indices of created forks in acc.forks.
    """
    fork_indices: list[int] = []

    # Iterate over fork_arms
    for g_i, g_j in acc.fork_arms:
        # Find branches that contain g_i and branches that contain g_j
        branches_with_gi = [b for b in branches if g_i in b.groups]
        branches_with_gj = [b for b in branches if g_j in b.groups]

        # Create forks between each pair of branches from the two groups
        for b_i in branches_with_gi:
            for b_j in branches_with_gj:
                # Skip if same branch (same token_id)
                if b_i.token_id == b_j.token_id:
                    continue

                # Check if this fork already exists
                fork_exists = any(
                    (
                        f.next_token_ids == (b_i.token_id, b_j.token_id)
                        or f.next_token_ids == (b_j.token_id, b_i.token_id)
                    )
                    for f in acc.forks
                )
                if fork_exists:
                    continue

                # Create fork with g_i's branch first (deterministic ordering)
                fork_idx = len(acc.forks)
                acc.forks.append(
                    BinaryFork(
                        next_token_ids=(b_i.token_id, b_j.token_id),
                        next_token_logprobs=(b_i.token_logprob, b_j.token_logprob),
                        group_idx=(g_i, g_j),
                    )
                )
                fork_indices.append(fork_idx)

    return fork_indices


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
