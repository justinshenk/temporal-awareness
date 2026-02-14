"""Tests for tree_as_structures_system module."""

import math

import pytest
import torch

from src.common.analysis.tree_as_structures_system import (
    Structure,
    System,
    SystemCore,
    SystemOrientation,
    Normativity,
    StructureSystemAnalysis,
    build_tree_as_structures_system,
    calculate_normativity,
    _traj_probability,
    _normalize_probs,
)
from src.common.token_tree import (
    TokenTrajectory,
    parse_tree_from_trajs,
)


def make_traj(
    token_ids: tuple[int, ...],
    logprobs: tuple[float, ...] | None = None,
) -> TokenTrajectory:
    """Helper to create a TokenTrajectory."""
    if logprobs is None:
        logprobs = tuple([0.0] * len(token_ids))
    full_logits = []
    for t_id in token_ids:
        logit_vec = torch.zeros(100)
        if t_id < 100:
            logit_vec[t_id] = 10.0
        full_logits.append(logit_vec)
    return TokenTrajectory(
        token_ids=list(token_ids),
        logprobs=list(logprobs),
        logits=[0.0] * len(token_ids),
        full_logits=full_logits,
    )


class TestHelpers:
    """Tests for helper functions."""

    def test_traj_probability(self):
        traj = make_traj((1, 2, 3), logprobs=(-0.5, -0.5, -0.5))
        assert _traj_probability(traj) == pytest.approx(math.exp(-1.5))

    def test_normalize_probs(self):
        probs = _normalize_probs([0.2, 0.3, 0.5])
        assert sum(probs) == pytest.approx(1.0)

    def test_normalize_probs_scales(self):
        probs = _normalize_probs([1.0, 2.0, 3.0])
        assert probs == pytest.approx([1/6, 2/6, 3/6])


class TestStructure:
    """Tests for Structure class."""

    def test_from_trajectories_basic(self):
        trajs = [make_traj((1, 2, 3)), make_traj((1, 2, 4)), make_traj((1, 5, 6))]
        trajs[0].group_idx = (0,)
        trajs[1].group_idx = (0,)
        trajs[2].group_idx = (1,)
        struct = Structure.from_trajectories(0, trajs)
        assert struct.compliances == (1.0, 1.0, 0.0)

    def test_from_trajectories_multi_group(self):
        trajs = [make_traj((1, 2, 3)), make_traj((1, 2, 4)), make_traj((1, 5, 6))]
        trajs[0].group_idx = (0, 1)
        trajs[1].group_idx = (0,)
        trajs[2].group_idx = (1,)
        assert Structure.from_trajectories(0, trajs).compliances == (1.0, 1.0, 0.0)
        assert Structure.from_trajectories(1, trajs).compliances == (1.0, 0.0, 1.0)


class TestSystem:
    """Tests for System class."""

    def test_from_fork_basic(self):
        trajs = [make_traj((1, 2)), make_traj((1, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]], fork_arms=[(0, 1)])
        assert len(tree.forks) == 1
        system = System.from_fork(0, tree.forks[0])
        assert system.structure_idx == (0, 1)


class TestCalculateNormativity:
    """Tests for calculate_normativity function."""

    def test_root_normativity_equal_probs(self):
        trajs = [make_traj((1, 2)), make_traj((1, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]], fork_arms=[(0, 1)])

        structures = {i: Structure.from_trajectories(i, tree.trajs) for i in [0, 1]}
        systems = [System.from_fork(0, tree.forks[0])]

        norm = calculate_normativity(0, None, systems, structures, tree.trajs, [0, 1])

        assert norm.node_idx == 0
        assert norm.cores[0].struct_cores == pytest.approx((0.5, 0.5))

    def test_core_weighted_by_probability(self):
        # Traj 0: prob=1, Traj 1: prob=e^(-1)
        trajs = [
            make_traj((1, 2), logprobs=(0.0, 0.0)),
            make_traj((1, 3), logprobs=(-0.5, -0.5)),
        ]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]], fork_arms=[(0, 1)])

        structures = {i: Structure.from_trajectories(i, tree.trajs) for i in [0, 1]}
        systems = [System.from_fork(0, tree.forks[0])]

        norm = calculate_normativity(0, None, systems, structures, tree.trajs, [0, 1])

        p0 = 1.0 / (1.0 + math.exp(-1))
        p1 = math.exp(-1) / (1.0 + math.exp(-1))
        assert norm.cores[0].struct_cores == pytest.approx((p0, p1))

    def test_branch_normativity_subset(self):
        trajs = [make_traj((1, 2)), make_traj((1, 3)), make_traj((1, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1], [1]], fork_arms=[(0, 1)])

        structures = {i: Structure.from_trajectories(i, tree.trajs) for i in [0, 1]}
        systems = [System.from_fork(0, tree.forks[0])]

        # Root: (1/3, 2/3)
        root = calculate_normativity(0, None, systems, structures, tree.trajs, [0, 1, 2])
        assert root.cores[0].struct_cores == pytest.approx((1/3, 2/3))

        # Branch with only group 1: (0, 1)
        branch = calculate_normativity(1, 1, systems, structures, tree.trajs, [1, 2])
        assert branch.cores[0].struct_cores == (0.0, 1.0)

    def test_core_entropy_and_diversity(self):
        trajs = [make_traj((1, 2)), make_traj((1, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]], fork_arms=[(0, 1)])

        structures = {i: Structure.from_trajectories(i, tree.trajs) for i in [0, 1]}
        systems = [System.from_fork(0, tree.forks[0])]

        norm = calculate_normativity(0, None, systems, structures, tree.trajs, [0, 1])
        # Core = (0.5, 0.5) -> diversity = 2
        assert norm.cores[0].diversity == pytest.approx(2.0)

    def test_orientation_and_deviance_computed(self):
        trajs = [make_traj((1, 2)), make_traj((1, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]], fork_arms=[(0, 1)])

        structures = {i: Structure.from_trajectories(i, tree.trajs) for i in [0, 1]}
        systems = [System.from_fork(0, tree.forks[0])]

        norm = calculate_normativity(0, None, systems, structures, tree.trajs, [0, 1])
        orient = norm.orientations[0]

        assert all(o is not None for o in orient.traj_orientations)
        assert orient.expected_deviance >= 0

    def test_excluded_trajectories_have_none(self):
        trajs = [make_traj((1, 2)), make_traj((1, 3)), make_traj((1, 4))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1], [1]], fork_arms=[(0, 1)])

        structures = {i: Structure.from_trajectories(i, tree.trajs) for i in [0, 1]}
        systems = [System.from_fork(0, tree.forks[0])]

        # Only traj 0
        norm = calculate_normativity(1, 1, systems, structures, tree.trajs, [0])
        orient = norm.orientations[0]

        assert orient.traj_orientations[0] is not None
        assert orient.traj_orientations[1] is None
        assert orient.traj_orientations[2] is None


class TestBuildTreeAsStructuresSystem:
    """Tests for build_tree_as_structures_system function."""

    def test_empty_tree(self):
        trajs = [make_traj((1, 2, 3))]
        tree = parse_tree_from_trajs(trajs)
        analysis = build_tree_as_structures_system(tree)
        assert analysis.n_structures == 0

    def test_tree_without_forks(self):
        trajs = [make_traj((1, 2, 3)), make_traj((1, 2, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]])
        analysis = build_tree_as_structures_system(tree)
        assert analysis.n_systems == 0

    def test_simple_binary_fork(self):
        trajs = [make_traj((1, 2)), make_traj((1, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]], fork_arms=[(0, 1)])
        analysis = build_tree_as_structures_system(tree)

        assert analysis.n_structures == 2
        assert analysis.n_systems == 1
        assert analysis.get_normativity(0) is not None

    def test_different_cores_at_different_nodes(self):
        trajs = [make_traj((1, 2, 10)), make_traj((1, 3, 20)), make_traj((1, 3, 21))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1], [1]], fork_arms=[(0, 1)])
        analysis = build_tree_as_structures_system(tree)

        root = analysis.get_normativity(0)
        assert root.cores[0].struct_cores == pytest.approx((1/3, 2/3))

    def test_structure_lookup(self):
        trajs = [make_traj((1, 2)), make_traj((1, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]], fork_arms=[(0, 1)])
        analysis = build_tree_as_structures_system(tree)

        assert analysis.get_structure(0).compliances == (1.0, 0.0)
        assert analysis.get_structure(1).compliances == (0.0, 1.0)

    def test_system_lookup(self):
        trajs = [make_traj((1, 2)), make_traj((1, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]], fork_arms=[(0, 1)])
        analysis = build_tree_as_structures_system(tree)

        assert analysis.get_system(0).structure_idx == (0, 1)

    def test_get_nonexistent(self):
        trajs = [make_traj((1, 2)), make_traj((1, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]], fork_arms=[(0, 1)])
        analysis = build_tree_as_structures_system(tree)

        assert analysis.get_structure(99) is None
        assert analysis.get_system(99) is None
        assert analysis.get_normativity(99) is None

    def test_three_groups(self):
        """Test with 3 groups - verifies generalization beyond binary."""
        trajs = [make_traj((1, 2)), make_traj((1, 3)), make_traj((1, 4))]
        tree = parse_tree_from_trajs(
            trajs,
            groups_per_traj=[[0], [1], [2]],
            fork_arms=[(0, 1), (0, 2), (1, 2)],
        )
        analysis = build_tree_as_structures_system(tree)

        assert analysis.n_structures == 3
        assert analysis.n_systems == 3

        # Each fork creates a system comparing 2 groups
        root = analysis.get_normativity(0)
        assert root is not None
        assert len(root.cores) == 3  # One core per system

        # Each core should have 2 elements (comparing 2 structures)
        for core in root.cores:
            assert len(core.struct_cores) == 2

    def test_system_with_three_structures(self):
        """Test a single system comparing 3 structures."""
        trajs = [make_traj((1, 2)), make_traj((1, 3)), make_traj((1, 4))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1], [2]])

        # Manually create a 3-way system
        structures = {i: Structure.from_trajectories(i, tree.trajs) for i in [0, 1, 2]}
        systems = [System.from_groups(0, [0, 1, 2])]

        norm = calculate_normativity(0, None, systems, structures, tree.trajs, [0, 1, 2])

        # Core should have 3 elements
        assert len(norm.cores[0].struct_cores) == 3
        # With equal probs, core should be (1/3, 1/3, 1/3)
        assert norm.cores[0].struct_cores == pytest.approx((1/3, 1/3, 1/3))

    def test_roundtrip_to_from_dict(self):
        """Test that to_dict -> from_dict roundtrip preserves root-level data.

        Note: to_dict rounds to 3 decimal places, so we use abs=0.001 tolerance.
        """
        trajs = [make_traj((1, 2)), make_traj((1, 3))]
        tree = parse_tree_from_trajs(trajs, groups_per_traj=[[0], [1]], fork_arms=[(0, 1)])
        original = build_tree_as_structures_system(tree)

        d = original.to_dict()
        restored = StructureSystemAnalysis.from_dict(d)

        # Simplified format only preserves root
        orig_root = original.normativities[0]
        rest_root = restored.normativities[0]

        # Values are rounded to 3 decimal places in to_dict
        assert rest_root.traj_probs == pytest.approx(orig_root.traj_probs, abs=0.001)
        assert rest_root.cores[0].struct_cores == pytest.approx(
            orig_root.cores[0].struct_cores, abs=0.001
        )
        assert rest_root.cores[0].entropy == pytest.approx(
            orig_root.cores[0].entropy, abs=0.001
        )
        assert rest_root.orientations[0].expected_deviance == pytest.approx(
            orig_root.orientations[0].expected_deviance, abs=0.001
        )

    def test_to_dict_multiple_systems(self):
        """Test to_dict with multiple systems (3 groups, 3 forks)."""
        trajs = [make_traj((1, 2)), make_traj((1, 3)), make_traj((1, 4))]
        tree = parse_tree_from_trajs(
            trajs,
            groups_per_traj=[[0], [1], [2]],
            fork_arms=[(0, 1), (0, 2), (1, 2)],
        )
        analysis = build_tree_as_structures_system(tree)
        d = analysis.to_dict()

        # Should have "systems" dict with 3 entries
        root = d["root"]
        assert "systems" in root
        assert len(root["systems"]) == 3

        # Keys should be group pairs
        assert "[0, 1]" in root["systems"]
        assert "[0, 2]" in root["systems"]
        assert "[1, 2]" in root["systems"]

        # Each system should have core, core_entropy, deviance stats, generalizations
        for key, sys_data in root["systems"].items():
            assert "core" in sys_data
            assert "core_entropy" in sys_data
            assert "core_diversity" in sys_data
            assert "min_deviance" in sys_data
            assert "max_deviance" in sys_data
            assert "std_deviance" in sys_data
            assert "median_deviance" in sys_data
            assert "generalizations" in sys_data
            gen = sys_data["generalizations"]
            assert "traj_excess_deviance" in gen
            assert "traj_deficit_deviance" in gen
            assert "traj_excess_compliance" in gen
            assert "traj_deficit_compliance" in gen
            assert len(sys_data["core"]) == 2  # Each system compares 2 groups
