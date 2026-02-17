"""Grouped binary choice data classes.

A GroupedBinaryChoice aggregates multiple forks (label pairs) to determine
the preferred choice. Each fork corresponds to a different label style
(e.g., 'a)' vs 'b)', '[i]' vs '[ii]', '<x>' vs '<y>').

Example:
    labels_a = ["a)", "[i]", "<x>"]  # short-term labels
    labels_b = ["b)", "[ii]", "<y>"]  # long-term labels

    Fork 0: 'a)' vs 'b)' -> choice at divergent position
    Fork 1: '[i]' vs '[ii]' -> choice at divergent position
    Fork 2: '<x>' vs '<y>' -> choice at divergent position

    Final choice = aggregate across forks
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import math

from ..token_tree import BinaryFork, TokenTree
from .binary_choice import LabeledBinaryChoice
from .simple_binary_choice import SimpleBinaryChoice


# ═══════════════════════════════════════════════════════════════════════════════
#  Aggregation Methods
# ═══════════════════════════════════════════════════════════════════════════════


class ForkAggregation(Enum):
    """Methods for aggregating choices across multiple forks.

    MEAN_LOGPROB: Average the divergent logprobs, then compare.
    SUM_LOGPROB: Sum the divergent logprobs, then compare.
    MIN_LOGPROB: Use minimum divergent logprob per side, then compare.
    MAX_LOGPROB: Use maximum divergent logprob per side, then compare.

    MEAN_NORMALIZED: Normalize each fork's logprobs to [0,1], average, compare.
    VOTE: Each fork votes (majority wins).
    WEIGHTED_VOTE: Each fork votes weighted by logprob difference.
    """

    MEAN_LOGPROB = "mean_logprob"
    SUM_LOGPROB = "sum_logprob"
    MIN_LOGPROB = "min_logprob"
    MAX_LOGPROB = "max_logprob"
    MEAN_NORMALIZED = "mean_normalized"
    VOTE = "vote"
    WEIGHTED_VOTE = "weighted_vote"


# ═══════════════════════════════════════════════════════════════════════════════
#  Core
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GroupedBinaryChoice(SimpleBinaryChoice):
    """Binary choice aggregated from multiple forks (label pairs).

    Inherits tree: TokenTree from SimpleBinaryChoice. The tree contains
    2*N trajectories (N label pairs), with forks representing divergences
    between label pairs.

    Attributes:
        tree: TokenTree with all trajectories and forks.
        label_pairs: Tuple of (label_a, label_b) pairs for each fork.
        aggregation: Method for combining fork results.
    """

    label_pairs: tuple[tuple[str, str], ...] | None = None
    aggregation: ForkAggregation = ForkAggregation.MEAN_LOGPROB

    # ── Access to forks ─────────────────────────────────────────────────

    @property
    def forks(self) -> tuple[BinaryFork, ...]:
        """Access BinaryFork objects from underlying tree."""
        return self.tree.forks or ()

    # ── Aggregated Decision ──────────────────────────────────────────────

    def choice_idx_by_method(self, method: ForkAggregation) -> int:
        """Compute choice index using a specific aggregation method.

        Returns:
            0 if model prefers A, 1 if B, -1 if tied.
        """
        if not self.forks:
            return -1

        if method == ForkAggregation.VOTE:
            return self._vote_choice_idx()
        elif method == ForkAggregation.WEIGHTED_VOTE:
            return self._weighted_vote_choice_idx()
        else:
            # Logprob-based methods
            lp_a, lp_b = self._aggregated_logprobs_by_method(method)
            if lp_a > lp_b:
                return 0
            if lp_b > lp_a:
                return 1
            return -1

    @property
    def choice_idx(self) -> int:
        """0 if model prefers A, 1 if B, -1 if tied."""
        return self.choice_idx_by_method(self.aggregation)

    @property
    def alternative_idx(self) -> int:
        """1 if model prefers A, 0 if B, -1 if tied."""
        idx = self.choice_idx
        if idx == -1:
            return -1
        return 1 - idx

    def choice_logprob_by_method(self, method: ForkAggregation) -> float | None:
        """Aggregated logprob of the chosen option using a specific method."""
        idx = self.choice_idx_by_method(method)
        if idx == -1:
            return None
        return self._aggregated_logprobs_by_method(method)[idx]

    @property
    def choice_logprob(self) -> float | None:
        """Aggregated logprob of the chosen option."""
        return self.choice_logprob_by_method(self.aggregation)

    def alternative_logprob_by_method(self, method: ForkAggregation) -> float | None:
        """Aggregated logprob of the rejected option using a specific method."""
        idx = self.choice_idx_by_method(method)
        if idx == -1:
            return None
        return self._aggregated_logprobs_by_method(method)[1 - idx]

    @property
    def alternative_logprob(self) -> float | None:
        """Aggregated logprob of the rejected option."""
        return self.alternative_logprob_by_method(self.aggregation)

    # ── Fork Analysis ────────────────────────────────────────────────────

    @property
    def n_forks(self) -> int:
        """Number of forks (label pairs)."""
        return len(self.forks)

    def fork_choices(self) -> list[int]:
        """Choice index for each fork: 0, 1, or -1."""
        choices = []
        for fork in self.forks:
            lp_a, lp_b = fork.next_token_logprobs
            if lp_a > lp_b:
                choices.append(0)
            elif lp_b > lp_a:
                choices.append(1)
            else:
                choices.append(-1)
        return choices

    def fork_agreement(self) -> float:
        """Fraction of forks that agree with the overall choice."""
        overall = self.choice_idx
        if overall == -1 or not self.forks:
            return 0.0
        choices = self.fork_choices()
        agreeing = sum(1 for c in choices if c == overall)
        return agreeing / len(choices)

    def disagreeing_forks(self) -> list[int]:
        """Indices of forks that disagree with the overall choice."""
        overall = self.choice_idx
        if overall == -1:
            return []
        return [i for i, c in enumerate(self.fork_choices()) if c != overall]

    def fork_logprob_diffs(self) -> list[float]:
        """Logprob difference (A - B) for each fork."""
        return [
            fork.next_token_logprobs[0] - fork.next_token_logprobs[1]
            for fork in self.forks
        ]

    def divergent_positions(self) -> list[int | None]:
        """Divergent position for each fork.

        Note: With single-tree structure, all forks share the same divergent
        position (first divergence in tree). Returns list for API compatibility.
        """
        if not self.tree.nodes:
            return [None] * self.n_forks
        pos = self.tree.nodes[0].branching_token_position
        return [pos] * self.n_forks

    # ── Internal Aggregation ─────────────────────────────────────────────

    def _get_fork_logprobs(self, fork: BinaryFork) -> tuple[float, float]:
        """Get (logprob_a, logprob_b) from a BinaryFork."""
        return (float(fork.next_token_logprobs[0]), float(fork.next_token_logprobs[1]))

    def _aggregated_logprobs_by_method(
        self, method: ForkAggregation
    ) -> tuple[float, float]:
        """(aggregated_logprob_a, aggregated_logprob_b) using specified method."""
        if not self.forks:
            return (0.0, 0.0)

        # Collect per-fork logprobs
        lps_a = [self._get_fork_logprobs(f)[0] for f in self.forks]
        lps_b = [self._get_fork_logprobs(f)[1] for f in self.forks]

        if method == ForkAggregation.MEAN_LOGPROB:
            return (sum(lps_a) / len(lps_a), sum(lps_b) / len(lps_b))
        elif method == ForkAggregation.SUM_LOGPROB:
            return (sum(lps_a), sum(lps_b))
        elif method == ForkAggregation.MIN_LOGPROB:
            return (min(lps_a), min(lps_b))
        elif method == ForkAggregation.MAX_LOGPROB:
            return (max(lps_a), max(lps_b))
        elif method == ForkAggregation.MEAN_NORMALIZED:
            return self._mean_normalized_logprobs()
        else:
            # Vote methods don't use logprobs directly
            return (sum(lps_a) / len(lps_a), sum(lps_b) / len(lps_b))

    def _mean_normalized_logprobs(self) -> tuple[float, float]:
        """Normalize each fork's logprobs to probabilities via softmax, then average."""
        if not self.forks:
            return (0.0, 0.0)

        prob_a_sum = 0.0
        prob_b_sum = 0.0

        for fork in self.forks:
            lp_a, lp_b = self._get_fork_logprobs(fork)
            # Softmax: convert logprobs to normalized probabilities
            # Use log-sum-exp trick for numerical stability
            max_lp = max(lp_a, lp_b)
            exp_a = math.exp(lp_a - max_lp)
            exp_b = math.exp(lp_b - max_lp)
            total = exp_a + exp_b
            prob_a_sum += exp_a / total
            prob_b_sum += exp_b / total

        n = len(self.forks)
        return (prob_a_sum / n, prob_b_sum / n)

    def _vote_choice_idx(self) -> int:
        """Majority vote across forks."""
        choices = self.fork_choices()
        votes_a = sum(1 for c in choices if c == 0)
        votes_b = sum(1 for c in choices if c == 1)
        if votes_a > votes_b:
            return 0
        if votes_b > votes_a:
            return 1
        return -1

    def _weighted_vote_choice_idx(self) -> int:
        """Vote weighted by logprob difference magnitude."""
        weight_a = 0.0
        weight_b = 0.0

        choices = self.fork_choices()
        for i, fork in enumerate(self.forks):
            lp_a, lp_b = self._get_fork_logprobs(fork)
            diff = abs(lp_a - lp_b)
            if choices[i] == 0:
                weight_a += diff
            elif choices[i] == 1:
                weight_b += diff

        if weight_a > weight_b:
            return 0
        if weight_b > weight_a:
            return 1
        return -1


# ═══════════════════════════════════════════════════════════════════════════════
#  With Labels
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LabeledGroupedBinaryChoice(GroupedBinaryChoice, LabeledBinaryChoice):
    """GroupedBinaryChoice with semantic labels for each option.

    The labels attribute contains lists of labels for each option.
    labels[0] = ["a)", "[i]", "<x>", ...] for option A
    labels[1] = ["b)", "[ii]", "<y>", ...] for option B

    Each corresponding pair (labels[0][i], labels[1][i]) defines a fork.
    """

    labels: tuple[list[str], list[str]] | None = None

    # ── Labels Interface ─────────────────────────────────────────────────

    @property
    def chosen_label(self) -> str | None:
        """First label of the chosen option."""
        if self.labels is None:
            return None
        idx = self.choice_idx
        if idx == -1:
            return None
        return self.labels[idx][0] if self.labels[idx] else None

    @property
    def alternative_label(self) -> str | None:
        """First label of the alternative option."""
        if self.labels is None:
            return None
        idx = self.choice_idx
        if idx == -1:
            return None
        alt_idx = 1 - idx
        return self.labels[alt_idx][0] if self.labels[alt_idx] else None

    @property
    def chosen_labels(self) -> list[str]:
        """All labels for the chosen option."""
        if self.labels is None:
            return []
        idx = self.choice_idx
        if idx == -1:
            return []
        return list(self.labels[idx])

    @property
    def alternative_labels(self) -> list[str]:
        """All labels for the alternative option."""
        if self.labels is None:
            return []
        idx = self.choice_idx
        if idx == -1:
            return []
        return list(self.labels[1 - idx])

    @property
    def n_label_pairs(self) -> int:
        """Number of label pairs (should equal n_forks)."""
        if self.labels is None:
            return 0
        return min(len(self.labels[0]), len(self.labels[1]))

    def without_labels(self) -> GroupedBinaryChoice:
        """Strip labels, returning a plain GroupedBinaryChoice."""
        return GroupedBinaryChoice(
            tree=self.tree,
            label_pairs=self.label_pairs,
            aggregation=self.aggregation,
        )
