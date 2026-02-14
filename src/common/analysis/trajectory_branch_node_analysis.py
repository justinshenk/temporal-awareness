"""Token tree analysis utilities.

Provides analysis data classes and functions for examining token tree
behavior at different granularities (trajectory, node, fork).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..base_schema import BaseSchema
from ..token_tree import (
    BinaryFork,
    BranchingNode,
    TokenTrajectory,
    TokenTree,
)
from ..math import (
    empirical_cross_entropy,
    inv_perplexity,
    log_odds,
    logprob_to_prob,
    perplexity,
    probability_ratio,
    q_fork_concentration,
    q_fork_diversity,
    q_fork_entropy,
    total_logprob,
    vocab_entropy_from_logits,
    worst_token_logprob,
    worst_token_position,
)
from .tree_as_structures_system import (
    StructureSystemAnalysis,
    build_tree_as_structures_system,
)


# ─── Base Class for Analysis with Prob/Odds Expansion ───────────────────────


@dataclass
class DistributionalAnalysis(BaseSchema):
    """Base class for analysis dataclasses that adds prob/odds fields to to_dict().

    Automatically converts (via _to_dict_hook, called by _canon):
    - *_logprob (float) -> *_prob via exp(logprob)
    - *_logprobs (list) -> *_probs via [exp(lp) for lp in logprobs]
    - log_odds -> odds via exp(log_odds)
    - *_log_odds -> *_odds via exp(log_odds)
    """

    def _to_dict_hook(self, d: dict) -> dict:
        """Hook called by _canon to add probability/odds fields.

        This is called automatically during serialization, ensuring nested
        analysis objects also get their prob/odds fields expanded.
        """
        return self._expand_logprob_fields(d)

    def _expand_logprob_fields(self, d: dict) -> dict:
        """Add probability fields for logprob fields."""
        additions = {}

        def _exp_safe(lp: float) -> float:
            """Safely compute exp(logprob), rounded to 4 decimal places."""
            if isinstance(lp, str):
                # Handle "Inf", "-Inf", "NaN" strings from _canon
                if lp == "Inf":
                    return float("inf")
                if lp == "-Inf" or lp == "NaN":
                    return 0.0
                return 0.0
            if math.isfinite(lp):
                return round(math.exp(lp), 4)
            return 0.0 if lp == float("-inf") else float("inf")

        for key, value in d.items():
            if value is None:
                continue

            # Handle exact "logprob" field -> "prob"
            if key == "logprob" and isinstance(value, (int, float, str)):
                additions["prob"] = _exp_safe(value)

            # Handle scalar *_logprob fields -> *_prob
            elif key.endswith("_logprob") and isinstance(value, (int, float, str)):
                prob_key = key.replace("_logprob", "_prob")
                additions[prob_key] = _exp_safe(value)

            # Handle *_logprobs sequences -> *_probs (e.g., next_token_logprobs)
            elif key.endswith("_logprobs") and isinstance(value, (list, tuple)):
                prob_key = key.replace("_logprobs", "_probs")
                additions[prob_key] = [_exp_safe(lp) for lp in value]

            # Handle sequence logprob fields (e.g., logprob_trajectory -> prob_trajectory)
            elif "logprob" in key and isinstance(value, list):
                prob_key = key.replace("logprob", "prob")
                additions[prob_key] = [_exp_safe(lp) for lp in value]

            # Handle log_odds -> odds
            elif key == "log_odds" and isinstance(value, (int, float, str)):
                if isinstance(value, str):
                    additions["odds"] = float("inf") if value == "Inf" else 0.0
                elif math.isfinite(value):
                    additions["odds"] = round(math.exp(value), 4)
                else:
                    additions["odds"] = float("inf") if value > 0 else 0.0

            # Handle *_log_odds -> *_odds
            elif key.endswith("_log_odds") and isinstance(value, (int, float, str)):
                odds_key = key.replace("_log_odds", "_odds")
                if isinstance(value, str):
                    additions[odds_key] = float("inf") if value == "Inf" else 0.0
                elif math.isfinite(value):
                    additions[odds_key] = round(math.exp(value), 4)
                else:
                    additions[odds_key] = float("inf") if value > 0 else 0.0

        d.update(additions)
        return d


# ─── Metrics Data Classes ────────────────────────────────────────────────────


@dataclass
class ForkMetrics(DistributionalAnalysis):
    """Metrics for a binary fork (two competing tokens)."""

    next_token_logprobs: tuple[float, float]  # (lp_A, lp_B) at the fork

    fork_entropy: float  # H of (p_A, p_B) normalised      — lower  = more decisive
    fork_diversity: float  # D₁ = e^H             ∈ [1, 2]   — lower  = more decisive
    fork_simpson: float  # D₂ = 1/(p_A²+p_B²)  ∈ [1, 2]   — lower  = more decisive
    fork_concentration: (
        float  # 1/D₁ = e^{-H}        ∈ [0.5, 1] — higher = more decisive
    )

    probability_ratio: float  # p_A / p_B at divergent pos       — >1 means A wins
    log_odds: float  # log(p_A / p_B)                   — >0 means A wins


@dataclass
class NodeMetrics(DistributionalAnalysis):
    """Metrics at a branching node's vocab distribution."""

    next_token_logprobs: list[float]  # logprobs of candidate tokens at this node
    vocab_entropy: float  # H of full vocab dist at divergent pos — lower = more decisive
    vocab_diversity: float  # D₁ = e^H — effective vocab size at decision point


@dataclass
class TrajectoryMetrics(DistributionalAnalysis):
    """Metrics computed over a trajectory's logprob sequence."""

    logprob_trajectory: list[float]  # the raw per-token logprobs

    empirical_cross_entropy: float  # H = −mean(logprobs)         — lower  = better
    inv_perplexity: float  # e^{-H} = geomean(probs)  ∈ (0, 1] — higher = better
    perplexity: float  # e^{H}  = 1/inv_ppl       ∈ [1, ∞) — lower  = better
    total_logprob: float  # Σ logprobs — length-dependent      — higher = better

    worst_token_logprob: float  # min(logprobs)                 — higher = better
    worst_token_position: int  # argmin(logprobs) — where the hardest token is

    @classmethod
    def from_logprobs(cls, logprobs: list[float]) -> TrajectoryMetrics:
        return cls(
            logprob_trajectory=logprobs,
            empirical_cross_entropy=empirical_cross_entropy(logprobs),
            inv_perplexity=inv_perplexity(logprobs),
            perplexity=perplexity(logprobs),
            total_logprob=total_logprob(logprobs),
            worst_token_logprob=worst_token_logprob(logprobs),
            worst_token_position=worst_token_position(logprobs),
        )


# ─── Analysis Data Classes ───────────────────────────────────────────────────


@dataclass
class ForkAnalysis(BaseSchema):
    """Analysis for a binary fork."""

    fork_idx: int
    metrics: ForkMetrics


@dataclass
class NodeAnalysis(BaseSchema):
    """Analysis at a branching node."""

    node_idx: int
    metrics: NodeMetrics


@dataclass
class TrajectoryAnalysis(BaseSchema):
    """Analysis for a trajectory."""

    traj_idx: int
    metrics: TrajectoryMetrics

    @classmethod
    def from_trajectory(
        cls,
        traj_idx: int,
        traj: TokenTrajectory,
        start: int = 0,
        end: int | None = None,
    ) -> TrajectoryAnalysis:
        logprobs = traj.logprobs[start : (end if end is not None else traj.length)]
        return cls.from_logprobs(traj_idx, logprobs)

    @classmethod
    def from_logprobs(cls, traj_idx: int, logprobs: list[float]) -> TrajectoryAnalysis:
        return cls(
            traj_idx=traj_idx,
            metrics=TrajectoryMetrics.from_logprobs(logprobs),
        )


# ─── Public Entry Point ─────────────────────────────────────────────────────


def analyze_token_tree(tree: TokenTree) -> StructureSystemAnalysis | None:
    """Populate the ``analysis`` field on tree, trajectories, nodes, and forks.

    Mutates *tree* in place, setting tree.analysis to the StructureSystemAnalysis.

    Returns:
        StructureSystemAnalysis if tree has forks and groups, else None.
        Contains per-node cores/orientations based on trajectory probabilities.
    """
    # First pass: basic analysis
    _analyze_trajectories_basic(tree)
    _analyze_forks(tree)
    _analyze_nodes_basic(tree)

    # Second pass: structure-aware analysis (needs all forks analyzed first)
    if tree.forks and tree.groups:
        tree.analysis = build_tree_as_structures_system(tree)
    return tree.analysis


# ─── Per-Component Analyzers ─────────────────────────────────────────────────


def _analyze_trajectories_basic(tree: TokenTree) -> None:
    """First pass: compute basic trajectory metrics."""
    for i, traj in enumerate(tree.trajs):
        traj.analysis = TrajectoryAnalysis.from_trajectory(traj_idx=i, traj=traj)


def _analyze_forks(tree: TokenTree) -> None:
    if not tree.forks:
        return
    for i, fork in enumerate(tree.forks):
        fork.analysis = _build_fork_analysis(i, fork)


def _analyze_nodes_basic(tree: TokenTree) -> None:
    """First pass: compute basic node metrics."""
    if not tree.nodes:
        return
    for i, node in enumerate(tree.nodes):
        node.analysis = _build_node_analysis(i, node, tree)


# ─── Builders ────────────────────────────────────────────────────────────────


def _build_fork_analysis(fork_idx: int, fork: BinaryFork) -> ForkAnalysis:
    lp_a, lp_b = fork.next_token_logprobs
    p_a, p_b = logprob_to_prob(lp_a), logprob_to_prob(lp_b)

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
        ),
    )


def _build_node_analysis(
    node_idx: int, node: BranchingNode, tree: TokenTree
) -> NodeAnalysis:
    next_token_logprobs = [float(lp) for lp in node.next_token_logprobs]

    # Use vocab_logits stored on node if available, fallback to tree lookup
    if node.vocab_logits is not None:
        logits = torch.tensor(node.vocab_logits)
        v_entropy = vocab_entropy_from_logits(logits).item()
    else:
        pos = node.branching_token_position
        logits = tree.get_logits_at_node(node_idx, pos)
        v_entropy = (
            vocab_entropy_from_logits(logits).item() if logits is not None else 0.0
        )

    return NodeAnalysis(
        node_idx=node_idx,
        metrics=NodeMetrics(
            next_token_logprobs=next_token_logprobs,
            vocab_entropy=v_entropy,
            vocab_diversity=math.exp(v_entropy),
        ),
    )
