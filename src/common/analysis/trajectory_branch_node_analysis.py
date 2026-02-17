"""Token tree analysis utilities.

Provides analysis data classes and functions for examining token tree
behavior at different granularities (trajectory, node, fork).

## Position Conventions

All position fields in analysis objects report ABSOLUTE positions within
the full trajectory. This means:

- `worst_token_position`: Index in `traj.token_ids` where worst token occurs
- `worst_rank_position`: Index in `traj.token_ids` where worst-ranked token occurs
- `top_p_normalized.worst_token_position`: Same convention

When using `start` or `rank_start` parameters, positions are still reported
as absolute. For example:

    traj.length = 100
    metrics = TrajectoryMetrics.from_trajectory(traj, start=20)
    # Metrics computed over positions 20-99 (80 tokens)
    # worst_token_position might be 45 (absolute), not 25 (relative)
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
    token_ranks_from_logits,
    top_p_normalized_logprobs,
    total_logprob,
    vocab_entropy_from_logits,
    worst_rank_position,
    worst_token_logprob,
    worst_token_position,
    worst_token_rank,
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
    logit_diff: float  # logit_A - logit_B = lp_A - lp_B — >0 means A wins
    reciprocal_rank_a: (
        float  # 1/rank_A in binary comparison — 1.0 if A wins, 0.5 if B wins
    )


@dataclass
class NodeMetrics(DistributionalAnalysis):
    """Metrics at a branching node's vocab distribution."""

    next_token_logprobs: list[float]  # logprobs of candidate tokens at this node
    vocab_entropy: (
        float  # H of full vocab dist at divergent pos — lower = more decisive
    )
    vocab_diversity: float  # D₁ = e^H — effective vocab size at decision point


@dataclass
class TopPNormalizedMetrics(DistributionalAnalysis):
    """Metrics computed with top-p normalized probabilities."""

    p: int  # number of top tokens considered
    total_logprob: float  # Σ normalized logprobs
    worst_token_logprob: float  # min(normalized logprobs)
    worst_token_position: int  # argmin(normalized logprobs) - ABSOLUTE position

    @classmethod
    def from_logits(
        cls,
        token_ids: list[int],
        full_logits,
        p: int = 100,
    ) -> TopPNormalizedMetrics | None:
        """Build normalized metrics from logits.

        Args:
            token_ids: Token IDs to compute metrics for
            full_logits: Full vocabulary logits for each position
            p: Number of top tokens to consider

        Returns:
            TopPNormalizedMetrics if any tokens fall within top-p, else None.
            The worst_token_position is relative to the input slice (0-indexed).
            Caller is responsible for adjusting to absolute position.
        """
        norm_logprobs = top_p_normalized_logprobs(token_ids, full_logits, p)

        # Filter out -inf values and track their original positions
        finite_positions = [
            i for i, lp in enumerate(norm_logprobs) if math.isfinite(lp)
        ]
        finite_logprobs = [norm_logprobs[i] for i in finite_positions]

        if not finite_logprobs:
            return None

        # Find worst position among finite values only
        worst_idx_in_finite = worst_token_position(finite_logprobs)
        actual_position = finite_positions[worst_idx_in_finite]

        return cls(
            p=p,
            total_logprob=total_logprob(finite_logprobs),
            worst_token_logprob=worst_token_logprob(finite_logprobs),
            worst_token_position=actual_position,
        )


@dataclass
class TrajectoryMetrics(DistributionalAnalysis):
    """Metrics computed over a trajectory's logprob sequence.

    Note: The raw logprobs are NOT stored here since they already exist
    on the trajectory itself (traj.logprobs). This avoids duplication.
    """

    empirical_cross_entropy: float  # H = −mean(logprobs)         — lower  = better
    inv_perplexity: float  # e^{-H} = geomean(probs)  ∈ (0, 1] — higher = better
    perplexity: float  # e^{H}  = 1/inv_ppl       ∈ [1, ∞) — lower  = better
    total_logprob: float  # Σ logprobs — length-dependent      — higher = better

    worst_token_logprob: float  # min(logprobs)                 — higher = better
    worst_token_position: int  # argmin(logprobs) — where the hardest token is

    # Rank-based metrics (only when full_logits available)
    worst_token_rank: int | None = None  # rank of worst token (1=greedy)
    worst_rank_position: int | None = None  # position of worst-ranked token

    # Top-p normalized metrics (only when full_logits available)
    top_p_normalized: TopPNormalizedMetrics | None = None

    @classmethod
    def from_logprobs(cls, logprobs: list[float]) -> TrajectoryMetrics:
        return cls(
            empirical_cross_entropy=empirical_cross_entropy(logprobs),
            inv_perplexity=inv_perplexity(logprobs),
            perplexity=perplexity(logprobs),
            total_logprob=total_logprob(logprobs),
            worst_token_logprob=worst_token_logprob(logprobs),
            worst_token_position=worst_token_position(logprobs),
        )

    @classmethod
    def from_trajectory(
        cls,
        traj: TokenTrajectory,
        start: int = 0,
        end: int | None = None,
        rank_start: int | None = None,
        top_p: int = 100,
    ) -> TrajectoryMetrics:
        """Build metrics from a trajectory, using full_logits if available.

        Args:
            traj: The trajectory to analyze
            start: Start position for all metrics (0 = full trajectory)
            end: End position (None = end of trajectory)
            rank_start: Start position for rank-based metrics (defaults to start).
                        Only set if you need rank metrics over a different range.
            top_p: Number of top tokens for normalized metrics

        All returned position fields are ABSOLUTE (relative to full trajectory).
        """
        end_pos = end if end is not None else traj.length
        rank_start_pos = rank_start if rank_start is not None else start

        # Stage 1: Basic metrics from logprobs
        logprobs = traj.logprobs[start:end_pos]
        metrics = cls.from_logprobs(logprobs)
        # Convert worst_token_position to absolute
        metrics.worst_token_position += start

        # Stage 2: Rank-based metrics (if available)
        if traj.full_logits is not None:
            _add_rank_metrics(metrics, traj, rank_start_pos, end_pos, top_p)

        return metrics


def _add_rank_metrics(
    metrics: TrajectoryMetrics,
    traj: TokenTrajectory,
    start: int,
    end: int,
    top_p: int,
) -> None:
    """Add rank-based metrics to existing TrajectoryMetrics.

    Modifies metrics in place, adding worst_token_rank, worst_rank_position,
    and top_p_normalized fields. All positions are reported as ABSOLUTE
    (relative to full trajectory).

    Args:
        metrics: TrajectoryMetrics to augment
        traj: Source trajectory with full_logits
        start: Start position for rank metrics
        end: End position for rank metrics
        top_p: Number of top tokens for normalized metrics
    """
    if start >= traj.full_logits.shape[0]:
        return  # No logits available for this range

    token_ids = traj.token_ids[start:end]
    logits = traj.full_logits[start:end]

    # Basic rank metrics
    ranks = token_ranks_from_logits(token_ids, logits)
    metrics.worst_token_rank = worst_token_rank(ranks)
    metrics.worst_rank_position = worst_rank_position(ranks) + start  # Absolute

    # Top-p normalized metrics
    top_p_metrics = TopPNormalizedMetrics.from_logits(token_ids, logits, top_p)
    if top_p_metrics is not None:
        top_p_metrics.worst_token_position += start  # Convert to absolute
        metrics.top_p_normalized = top_p_metrics


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
    """Analysis for a trajectory with trunk/continuation breakdown.

    Attributes:
        traj_idx: Index of this trajectory in the tree.
        trunk_last_idx: Last index of trunk (= trunk_length - 1), or None if no trunk info.
        full_traj: Metrics over the full trajectory [0, end).
        trunk_only: Metrics over [0, trunk_length) if trunk_length > 0, else None.
        continuation_only: Metrics over [trunk_length, end) if applicable, else None.
    """

    traj_idx: int
    trunk_last_idx: int | None
    full_traj: TrajectoryMetrics
    trunk_only: TrajectoryMetrics | None
    continuation_only: TrajectoryMetrics | None

    @classmethod
    def from_trajectory(
        cls,
        traj_idx: int,
        traj: TokenTrajectory,
        trunk_length: int | None = None,
        top_p: int = 100,
    ) -> TrajectoryAnalysis:
        """Build analysis from a trajectory with optional trunk breakdown.

        Args:
            traj_idx: Index of this trajectory.
            traj: The trajectory to analyze.
            trunk_length: Length of trunk (prompt tokens). If None, no trunk info.
            top_p: Number of top tokens for normalized metrics.

        Returns:
            TrajectoryAnalysis with full_traj always computed, and trunk_only/
            continuation_only computed when trunk_length is provided and valid.
        """
        # Always compute full trajectory metrics
        full_traj = TrajectoryMetrics.from_trajectory(traj, start=0, top_p=top_p)

        trunk_only = None
        continuation_only = None
        trunk_last_idx = None

        if trunk_length is not None and trunk_length > 0:
            trunk_last_idx = trunk_length - 1

            # Trunk metrics: [0, trunk_length)
            trunk_only = TrajectoryMetrics.from_trajectory(
                traj, start=0, end=trunk_length, top_p=top_p
            )

            # Continuation metrics: [trunk_length, end) if there's continuation
            if trunk_length < traj.length:
                continuation_only = TrajectoryMetrics.from_trajectory(
                    traj, start=trunk_length, top_p=top_p
                )

        return cls(
            traj_idx=traj_idx,
            trunk_last_idx=trunk_last_idx,
            full_traj=full_traj,
            trunk_only=trunk_only,
            continuation_only=continuation_only,
        )

    @classmethod
    def from_logprobs(cls, traj_idx: int, logprobs: list[float]) -> TrajectoryAnalysis:
        """Build analysis from raw logprobs (no trunk info)."""
        return cls(
            traj_idx=traj_idx,
            trunk_last_idx=None,
            full_traj=TrajectoryMetrics.from_logprobs(logprobs),
            trunk_only=None,
            continuation_only=None,
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
    """First pass: compute basic trajectory metrics.

    Computes metrics for full trajectory, and if trunk_length is set,
    also computes trunk_only and continuation_only metrics.
    """
    for i, traj in enumerate(tree.trajs):
        traj.analysis = TrajectoryAnalysis.from_trajectory(
            traj_idx=i, traj=traj, trunk_length=tree.trunk_length
        )


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
            logit_diff=lp_a - lp_b,
            reciprocal_rank_a=1.0 if lp_a >= lp_b else 0.5,
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
