"""Simple binary choice data classes.

A SimpleBinaryChoice wraps a TokenTree of exactly two trajectories and
derives the decision (which response the model prefers) from the first
divergence point in the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


from ..analysis.trajectory_branch_node_analysis import (
    StructureSystemAnalysis,
    analyze_token_tree,
)
from ..token_tree import TokenTrajectory, TokenTree
from .binary_choice import BinaryChoice, LabeledBinaryChoice


# ═══════════════════════════════════════════════════════════════════════════════
#  Core
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SimpleBinaryChoice(BinaryChoice):
    """Concrete binary choice implementation using a TokenTree.

    Derives the choice from comparing logprobs at the first divergence point.
    """

    tree: TokenTree  # exactly 2 trajectories

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_trajectories(
        cls,
        traj_a: TokenTrajectory,
        traj_b: TokenTrajectory,
        trunk: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> SimpleBinaryChoice:
        """Build a SimpleBinaryChoice (or subclass) from two trajectories.

        Each trajectory represents a different label/choice, so they are
        placed in separate groups for cross-group fork creation.
        """
        tree = TokenTree.from_trajectories(
            [traj_a, traj_b],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
            trunk=trunk,
        )
        analyze_token_tree(tree)  # Sets tree.analysis
        return cls(tree=tree, **kwargs)

    # ── Decision ─────────────────────────────────────────────────────────

    @property
    def choice_idx(self) -> int:
        """0 if model prefers A, 1 if B, -1 if tied."""
        lp_a, lp_b = self._divergent_logprobs
        if lp_a > lp_b:
            return 0
        if lp_b > lp_a:
            return 1
        return -1

    @property
    def alternative_idx(self) -> int:
        """1 if model prefers A, 0 if B, -1 if tied."""
        idx = self.choice_idx
        if idx == -1:
            return -1
        return 1 - idx

    @property
    def choice_logprob(self) -> float | None:
        """Logprob of the *chosen* token at the divergent position."""
        idx = self.choice_idx
        if idx == -1:
            return None
        return self._divergent_logprobs[idx]

    @property
    def alternative_logprob(self) -> float | None:
        """Logprob of the *rejected* token at the divergent position."""
        idx = self.choice_idx
        if idx == -1:
            return None
        return self._divergent_logprobs[1 - idx]

    # ── Trajectory access ────────────────────────────────────────────────

    @property
    def chosen_traj(self) -> TokenTrajectory | None:
        idx = self.choice_idx
        if idx == -1:
            return None
        return self.tree.trajs[idx]

    @property
    def alternative_traj(self) -> TokenTrajectory | None:
        idx = self.choice_idx
        if idx == -1:
            return None
        return self.tree.trajs[1 - idx]

    @property
    def divergent_position(self) -> int | None:
        """Token position where the two trajectories first diverge."""
        if not self.tree.nodes:
            return None
        return self.tree.nodes[0].branching_token_position

    # ── Internal ─────────────────────────────────────────────────────────

    @property
    def _divergent_logprobs(self) -> tuple[float, float]:
        """(logprob_a, logprob_b) at the first divergent position."""
        if not self.tree.forks:
            return (0.0, 0.0)
        lp = self.tree.forks[0].next_token_logprobs
        return (float(lp[0]), float(lp[1]))

    def pop_heavy(self):
        self.tree.pop_heavy()

    def load_internals_from_disk(self, paths: dict | None) -> None:
        """Load internals from disk into trajectories.

        Args:
            paths: Dict with 'activations' key pointing to .pt file
        """
        if paths is None:
            return
        activations_path = paths.get("activations")
        if activations_path is None:
            return
        # Load into chosen trajectory if it supports internals
        traj = self.chosen_traj
        if traj is not None and traj.can_have_internals():
            traj.load_internals_from_disk(activations_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  With labels
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LabeledSimpleBinaryChoice(SimpleBinaryChoice, LabeledBinaryChoice):
    """SimpleBinaryChoice with semantic labels for each option.

    Inherits chosen_label/alternative_label from LabeledBinaryChoice.
    """

    labels: tuple[str, str] | None = None  # e.g. ("a)", "b)")
    response_texts: tuple[str, str] | None = (
        None  # e.g. ("I choose: a)", "I choose: b)")
    )

    def without_labels(self) -> SimpleBinaryChoice:
        """Strip labels, returning a plain SimpleBinaryChoice."""
        return SimpleBinaryChoice(tree=self.tree)
