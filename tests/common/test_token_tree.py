"""Tests for token_tree.py.

Run all tests:
    uv run pytest tests/common/test_token_tree.py -v

Run specific test class:
    uv run pytest tests/common/test_token_tree.py::TestParseTree -v
    uv run pytest tests/common/test_token_tree.py::TestTrajectoryProperties -v
"""

import torch

from src.common.token_tree import (
    BinaryFork,
    BranchingNode,
    TokenTrajectory,
    TokenTree,
    add_trajectory_to_tree,
    parse_tree_from_trajs,
)


def make_trajectory(
    token_ids: list[int],
    logprobs: list[float] | None = None,
    logits: list[float] | None = None,
    full_logits: torch.Tensor | None = None,
) -> TokenTrajectory:
    """Create a test trajectory with sensible defaults."""
    n = len(token_ids)
    if logprobs is None:
        logprobs = [0.0] * n
    if logits is None:
        logits = [0.0] * n
    if full_logits is None:
        vocab_size = 100
        full_logits = torch.zeros(n, vocab_size)
        for i, tid in enumerate(token_ids):
            if tid < vocab_size:
                full_logits[i, tid] = 10.0

    return TokenTrajectory(
        token_ids=token_ids,
        logprobs=logprobs,
        logits=logits,
        full_logits=full_logits,
    )


class TestParseTree:
    """Tests for parse_tree_from_trajs."""

    def test_two_groups_single_traj(self):
        """Group 0: [1,2,3,4], Group 1: [1,2,5,6] -> 1 node at pos 2, 1 fork."""
        traj_a = make_trajectory([1, 2, 3, 4])
        traj_b = make_trajectory([1, 2, 5, 6])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )

        assert len(tree.trajs) == 2
        assert len(tree.nodes) == 1
        assert len(tree.forks) == 1
        assert tree.nodes[0].branching_token_position == 2
        assert set(tree.forks[0].next_token_ids) == {3, 5}

    def test_two_groups_multi_traj(self):
        """Multiple trajs per group -> only cross-group forks."""
        traj_a1 = make_trajectory([1, 2, 3, 4])
        traj_a2 = make_trajectory([1, 2, 3, 7])
        traj_b1 = make_trajectory([1, 2, 5, 6])
        traj_b2 = make_trajectory([1, 2, 5, 8])

        tree = parse_tree_from_trajs(
            [traj_a1, traj_a2, traj_b1, traj_b2],
            groups_per_traj=[[0], [0], [1], [1]],
            fork_arms=[(0, 1)],
        )

        assert len(tree.trajs) == 4
        assert len(tree.nodes) >= 1

        # Only cross-group forks
        cross_group_forks = [f for f in tree.forks if set(f.next_token_ids) == {3, 5}]
        assert len(cross_group_forks) == 1

        # No within-group forks
        assert not any(set(f.next_token_ids) == {4, 7} for f in tree.forks)
        assert not any(set(f.next_token_ids) == {6, 8} for f in tree.forks)

    def test_three_groups(self):
        """3 groups -> 3 pairwise forks."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])
        traj_c = make_trajectory([1, 2, 7])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b, traj_c],
            groups_per_traj=[[0], [1], [2]],
            fork_arms=[(0, 1), (0, 2), (1, 2)],
        )

        assert len(tree.trajs) == 3
        assert len(tree.nodes) == 1
        assert len(tree.forks) == 3

        fork_pairs = [set(f.next_token_ids) for f in tree.forks]
        assert {3, 5} in fork_pairs
        assert {3, 7} in fork_pairs
        assert {5, 7} in fork_pairs

    def test_three_groups_ordering(self):
        """Fork ordering is deterministic: (g0,g1), (g0,g2), (g1,g2)."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])
        traj_c = make_trajectory([1, 2, 7])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b, traj_c],
            groups_per_traj=[[0], [1], [2]],
            fork_arms=[(0, 1), (0, 2), (1, 2)],
        )

        assert tree.forks[0].next_token_ids == (3, 5)
        assert tree.forks[1].next_token_ids == (3, 7)
        assert tree.forks[2].next_token_ids == (5, 7)

    def test_no_divergence(self):
        """Identical trajectories -> no nodes, no forks."""
        traj_a = make_trajectory([1, 2, 3, 4])
        traj_b = make_trajectory([1, 2, 3, 4])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
        )

        assert len(tree.trajs) == 2
        assert len(tree.nodes) == 0
        assert len(tree.forks) == 0

    def test_single_trajectory(self):
        """Single trajectory -> no nodes, no forks."""
        traj = make_trajectory([1, 2, 3])

        tree = parse_tree_from_trajs([traj], groups_per_traj=[[0]])

        assert len(tree.trajs) == 1
        assert len(tree.nodes) == 0
        assert len(tree.forks) == 0

    def test_empty_trajs(self):
        """Empty input -> empty tree."""
        tree = parse_tree_from_trajs([])

        assert len(tree.trajs) == 0
        assert len(tree.nodes) == 0
        assert len(tree.forks) == 0

    def test_single_group_no_forks(self):
        """Single group with divergence -> nodes but no forks."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [0]],
        )

        assert len(tree.trajs) == 2
        assert len(tree.nodes) == 1
        assert len(tree.forks) == 0

    def test_from_trajectories_class_method(self):
        """TokenTree.from_trajectories works."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = TokenTree.from_trajectories(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )

        assert len(tree.trajs) == 2
        assert len(tree.nodes) == 1
        assert len(tree.forks) == 1

    def test_node_has_fork_indices(self):
        """Nodes reference their associated forks."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )

        node = tree.nodes[0]
        assert node.forks_idx is not None
        assert node.forks_idx == [0]

    def test_nested_divergences(self):
        """Nested divergences create forks at each level."""
        traj_a = make_trajectory([1, 2, 3, 10])
        traj_b = make_trajectory([1, 2, 3, 20])
        traj_c = make_trajectory([1, 2, 5, 30])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b, traj_c],
            groups_per_traj=[[0], [0], [1]],
            fork_arms=[(0, 1)],
        )

        assert len(tree.trajs) == 3
        assert len(tree.nodes) == 2
        assert len(tree.forks) == 3

        fork_pairs = [set(f.next_token_ids) for f in tree.forks]
        assert {3, 5} in fork_pairs
        assert {10, 30} in fork_pairs
        assert {20, 30} in fork_pairs
        assert {10, 20} not in fork_pairs

    def test_explicit_fork_arms(self):
        """Custom fork_arms controls which groups create forks."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])
        traj_c = make_trajectory([1, 2, 7])

        # Only create forks between groups 0 and 2
        tree = parse_tree_from_trajs(
            [traj_a, traj_b, traj_c],
            groups_per_traj=[[0], [1], [2]],
            fork_arms=[(0, 2)],
        )

        assert len(tree.forks) == 1
        assert set(tree.forks[0].next_token_ids) == {3, 7}
        assert tree.forks[0].group_idx == (0, 2)

    def test_multi_group_trajectory(self):
        """Trajectory can belong to multiple groups."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        # traj_a in both groups 0 and 1
        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0, 1], [2]],
        )

        assert tree.trajs[0].group_idx == (0, 1)
        assert tree.trajs[1].group_idx == (2,)


class TestTrajectoryProperties:
    """Tests for TokenTrajectory properties."""

    def test_group_idx_populated(self):
        """Trajectories have group_idx set after parsing."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])
        traj_c = make_trajectory([1, 2, 7])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b, traj_c],
            groups_per_traj=[[0], [1], [1]],
        )

        assert tree.trajs[0].group_idx == (0,)
        assert tree.trajs[1].group_idx == (1,)
        assert tree.trajs[2].group_idx == (1,)

    def test_branching_points_populated(self):
        """Trajectories have branching_points populated."""
        traj_a = make_trajectory([1, 2, 3, 4])
        traj_b = make_trajectory([1, 2, 5, 6])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
        )

        assert tree.trajs[0].branching_points == [2]
        assert tree.trajs[1].branching_points == [2]

    def test_nodes_idx_populated(self):
        """Trajectories have nodes_idx populated."""
        traj_a = make_trajectory([1, 2, 3, 4])
        traj_b = make_trajectory([1, 2, 5, 6])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
        )

        assert tree.trajs[0].nodes_idx == (0,)
        assert tree.trajs[1].nodes_idx == (0,)


class TestAddTrajectory:
    """Tests for add_trajectory_to_tree."""

    def test_add_to_existing_group(self):
        """Adding to existing group doesn't create new forks."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )
        assert len(tree.forks) == 1

        traj_c = make_trajectory([1, 2, 3])
        new_tree = add_trajectory_to_tree(tree, traj_c, group_idx=[0])

        assert len(new_tree.trajs) == 3
        assert sum(1 for t in new_tree.trajs if 0 in t.group_idx) == 2
        assert len(new_tree.forks) == 1

    def test_add_to_new_group(self):
        """Adding to new group creates new forks."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )

        traj_c = make_trajectory([1, 2, 7])
        new_tree = add_trajectory_to_tree(tree, traj_c, group_idx=[2])

        assert len(new_tree.trajs) == 3
        assert new_tree.trajs[2].group_idx == (2,)
        assert len(new_tree.forks) == 3

    def test_add_to_empty_tree(self):
        """Adding to empty tree works."""
        empty_tree = TokenTree(trajs=(), nodes=(), forks=())

        traj = make_trajectory([1, 2, 3])
        new_tree = add_trajectory_to_tree(empty_tree, traj, group_idx=[0])

        assert len(new_tree.trajs) == 1
        assert new_tree.trajs[0].group_idx == (0,)

    def test_group_idx_set(self):
        """New trajectory has group_idx set."""
        traj_a = make_trajectory([1, 2, 3])
        tree = parse_tree_from_trajs([traj_a], groups_per_traj=[[0]])

        traj_b = make_trajectory([1, 2, 5])
        new_tree = add_trajectory_to_tree(tree, traj_b, group_idx=[1])

        assert new_tree.trajs[1].group_idx == (1,)

    def test_method_on_token_tree(self):
        """TokenTree.add_trajectory method works."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )

        traj_c = make_trajectory([1, 2, 7])
        new_tree = tree.add_trajectory(traj_c, group_idx=[2])

        assert len(new_tree.trajs) == 3
        assert new_tree.trajs[2].group_idx == (2,)


class TestBranchingNode:
    """Tests for BranchingNode fields."""

    def test_traj_idx_populated(self):
        """BranchingNode.traj_idx contains trajectory indices."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
        )

        node = tree.nodes[0]
        assert node.traj_idx is not None
        assert set(node.traj_idx) == {0, 1}

    def test_vocab_logits_populated(self):
        """BranchingNode.vocab_logits contains logits."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
        )

        node = tree.nodes[0]
        assert node.vocab_logits is not None
        assert len(node.vocab_logits) == 2
        assert len(node.vocab_logits[0]) == 100

    def test_forks_idx_is_list(self):
        """BranchingNode.forks_idx is a list."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )

        node = tree.nodes[0]
        assert node.forks_idx is not None
        assert isinstance(node.forks_idx, list)
        assert node.forks_idx == [0]


class TestTreeGroups:
    """Tests for TokenTree.groups property."""

    def test_groups_property(self):
        """TokenTree.groups returns all unique groups."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])
        traj_c = make_trajectory([1, 2, 7])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b, traj_c],
            groups_per_traj=[[0], [1], [2]],
        )

        assert tree.groups == (0, 1, 2)
        assert tree.n_groups == 3

    def test_groups_empty_when_no_groups(self):
        """TokenTree.groups is empty when no groups assigned."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs([traj_a, traj_b])

        assert tree.groups == ()
        assert tree.n_groups == 0

    def test_groups_multi_group_trajectory(self):
        """Groups includes all groups from multi-group trajectories."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0, 1], [2]],
        )

        assert tree.groups == (0, 1, 2)
        assert tree.n_groups == 3


class TestBinaryFork:
    """Tests for BinaryFork fields."""

    def test_group_idx_set(self):
        """BinaryFork.group_idx is set."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )

        fork = tree.forks[0]
        assert fork.group_idx == (0, 1)

    def test_group_idx_three_groups(self):
        """Three groups have correct fork group_idx pairs."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])
        traj_c = make_trajectory([1, 2, 7])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b, traj_c],
            groups_per_traj=[[0], [1], [2]],
            fork_arms=[(0, 1), (0, 2), (1, 2)],
        )

        group_pairs = [f.group_idx for f in tree.forks]
        assert (0, 1) in group_pairs
        assert (0, 2) in group_pairs
        assert (1, 2) in group_pairs

    def test_group_idx_ordering(self):
        """Fork group_idx has lower group first."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])

        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )

        fork = tree.forks[0]
        assert fork.group_idx[0] < fork.group_idx[1]


class TestAddTrajectoryToTree:
    """Tests for add_trajectory_to_tree function."""

    # uv run pytest tests/common/test_token_tree.py::TestAddTrajectoryToTree::test_add_to_existing_group -v
    def test_add_to_existing_group(self):
        """Adding to existing group doesn't create new cross-group forks."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])
        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )

        # Add another trajectory to group 0
        traj_c = make_trajectory([1, 2, 7])
        new_tree = add_trajectory_to_tree(tree, traj_c, [0])

        assert len(new_tree.trajs) == 3
        assert new_tree.trajs[2].group_idx == (0,)
        # Still 2 groups
        assert new_tree.groups == (0, 1)

    # uv run pytest tests/common/test_token_tree.py::TestAddTrajectoryToTree::test_add_to_new_group -v
    def test_add_to_new_group(self):
        """Adding to new group creates new cross-group forks."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])
        tree = parse_tree_from_trajs(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )

        # Add trajectory to new group 2
        traj_c = make_trajectory([1, 2, 7])
        new_tree = add_trajectory_to_tree(tree, traj_c, [2])

        assert len(new_tree.trajs) == 3
        assert new_tree.trajs[2].group_idx == (2,)
        assert new_tree.groups == (0, 1, 2)

        # Should have new forks for (0,2) and (1,2) pairs
        group_pairs = [f.group_idx for f in new_tree.forks]
        assert (0, 2) in group_pairs
        assert (1, 2) in group_pairs

    # uv run pytest tests/common/test_token_tree.py::TestAddTrajectoryToTree::test_add_to_empty_tree -v
    def test_add_to_empty_tree(self):
        """Adding to empty tree works."""
        traj = make_trajectory([1, 2, 3])
        empty_tree = TokenTree(trajs=())

        new_tree = add_trajectory_to_tree(empty_tree, traj, [0])

        assert len(new_tree.trajs) == 1
        assert new_tree.trajs[0].group_idx == (0,)

    # uv run pytest tests/common/test_token_tree.py::TestAddTrajectoryToTree::test_group_idx_set_on_new_traj -v
    def test_group_idx_set_on_new_traj(self):
        """New trajectory has group_idx set."""
        traj_a = make_trajectory([1, 2, 3])
        tree = parse_tree_from_trajs([traj_a], groups_per_traj=[[0]])

        traj_b = make_trajectory([1, 2, 5])
        new_tree = add_trajectory_to_tree(tree, traj_b, [1])

        assert new_tree.trajs[1].group_idx == (1,)

    # uv run pytest tests/common/test_token_tree.py::TestAddTrajectoryToTree::test_method_on_token_tree -v
    def test_method_on_token_tree(self):
        """TokenTree.add_trajectory method works same as function."""
        traj_a = make_trajectory([1, 2, 3])
        traj_b = make_trajectory([1, 2, 5])
        tree = parse_tree_from_trajs([traj_a], groups_per_traj=[[0]])

        new_tree = tree.add_trajectory(traj_b, [1])

        assert len(new_tree.trajs) == 2
        assert new_tree.trajs[1].group_idx == (1,)
