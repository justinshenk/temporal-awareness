"""Token tree analysis entry point and per-component analyzers.

Provides analyze_token_tree and helper functions for analyzing
trajectories, forks, and nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .analysis_builders import build_fork_analysis, build_node_analysis
from .metrics import TrajectoryAnalysis
from .tree_as_structures_system import (
    StructureSystemAnalysis,
    build_tree_as_structures_system,
)

if TYPE_CHECKING:
    from ..token_tree import TokenTree


def analyze_token_tree(
    tree: "TokenTree",
    W_U: torch.Tensor | None = None,
    b_U: torch.Tensor | None = None,
) -> StructureSystemAnalysis | None:
    """Populate the ``analysis`` field on tree, trajectories, nodes, and forks.

    Mutates *tree* in place, setting tree.analysis to the StructureSystemAnalysis.

    Args:
        tree: TokenTree to analyze
        W_U: Optional unembedding matrix [d_model, vocab_size] for TCB computation
        b_U: Optional unembedding bias [vocab_size] for TCB computation

    Returns:
        StructureSystemAnalysis if tree has forks and groups, else None.
        Contains per-node cores/orientations based on trajectory probabilities.
    """
    # First pass: basic analysis
    _analyze_trajectories_basic(tree)
    _analyze_forks(tree)
    _analyze_nodes_basic(tree, W_U, b_U)

    # Second pass: structure-aware analysis (needs all forks analyzed first)
    if tree.forks and tree.groups:
        tree.analysis = build_tree_as_structures_system(tree)
    return tree.analysis


def _analyze_trajectories_basic(tree: "TokenTree") -> None:
    """First pass: compute basic trajectory metrics.

    Computes metrics for full trajectory, and if trunk_length is set,
    also computes trunk_only and continuation_only metrics.
    """
    for i, traj in enumerate(tree.trajs):
        traj.analysis = TrajectoryAnalysis.from_trajectory(
            traj_idx=i, traj=traj, trunk_length=tree.trunk_length
        )


def _analyze_forks(tree: "TokenTree") -> None:
    """Analyze all forks in the tree."""
    if not tree.forks:
        return
    for i, fork in enumerate(tree.forks):
        fork.analysis = build_fork_analysis(i, fork)


def _analyze_nodes_basic(
    tree: "TokenTree",
    W_U: torch.Tensor | None = None,
    b_U: torch.Tensor | None = None,
) -> None:
    """First pass: compute basic node metrics."""
    if not tree.nodes:
        return
    for i, node in enumerate(tree.nodes):
        node.analysis = build_node_analysis(i, node, tree, W_U, b_U)
