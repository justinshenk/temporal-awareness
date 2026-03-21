"""Metrics extracted from activation patching results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from ..common.base_schema import BaseSchema
from ..common.choice.grouped_binary_choice import ForkAggregation
from ..common.math import logaddexp
from ..common.patching_types import PatchingMode

from .intervened_choice import IntervenedChoice

# Label perspective types
LabelPerspective = Literal["clean", "corrupted", "combined"]

# Aggregation methods to plot (excluding vote methods which don't produce continuous metrics)
PLOT_AGGREGATION_METHODS: list[ForkAggregation] = [
    ForkAggregation.MEAN_LOGPROB,
    ForkAggregation.MEAN_NORMALIZED,
    ForkAggregation.SUM_LOGPROB,
    ForkAggregation.MIN_LOGPROB,
    ForkAggregation.MAX_LOGPROB,
]

# Default aggregation method for main plots
DEFAULT_AGGREGATION = ForkAggregation.MEAN_NORMALIZED


@dataclass
class IntervenedChoiceMetrics(BaseSchema):
    """Metrics extracted from an IntervenedChoice for visualization/analysis.

    Provides type-safe access to metrics with sensible defaults for missing data.

    Extraction modes:
    1. Aggregated (multilabel default): Uses aggregated logit_diff across all forks
       - recovery/disruption computed from aggregated values
       - All metrics are internally consistent
    2. Per-fork: Extracts metrics for a specific fork index
       - recovery/disruption computed from that fork's values only
    3. Combined: Uses logaddexp aggregation across forks
       - Semantically different from arithmetic aggregation
    """

    # Core metrics
    mode: PatchingMode = "denoising"
    recovery: float = 0.0
    disruption: float = 0.0
    flipped: bool = False
    label_perspective: LabelPerspective = "clean"
    n_labels: int = 1

    # Extraction metadata
    aggregation_method: str | None = None  # ForkAggregation name, or None for single-label
    fork_idx: int | None = None  # If extracted for specific fork

    # Logprob metrics
    logit_diff: float = 0.0
    logprob_short: float = -10.0
    logprob_long: float = -10.0

    # Probability metrics (derived from logprobs)
    prob_short: float = 0.0
    prob_long: float = 0.0

    # Raw logit metrics
    logit_short: float = 0.0
    logit_long: float = 0.0
    norm_logit_short: float = 0.0
    norm_logit_long: float = 0.0
    norm_logit_diff: float = 0.0

    # Baseline-relative metrics
    baseline_logit_diff: float = 0.0
    rel_logit_delta: float = 0.0  # (logit_diff - baseline) / abs(baseline)

    # Rank metrics
    reciprocal_rank_short: float = 0.0
    reciprocal_rank_long: float = 0.0

    # Fork distribution metrics
    fork_entropy: float = 0.0
    fork_diversity: float = 1.0  # 1.0 = one option dominates, 2.0 = balanced
    fork_simpson: float = 1.0

    # Vocabulary distribution metrics
    vocab_entropy: float = 0.0
    vocab_diversity: float = 1.0
    vocab_simpson: float = 1.0
    vocab_tcb: float = 0.0

    # Trajectory metrics
    inv_perplexity: float = 0.0
    traj_inv_perplexity_short: float = 0.0
    traj_inv_perplexity_long: float = 0.0

    # Intervention effect metrics (computed at extraction time based on mode)
    # These use "target - source" semantics so positive = toward target
    effect: float = 0.0  # recovery for denoising, disruption for noising
    effect_logit_diff: float = 0.0  # short-long for denoising, long-short for noising
    effect_norm_logit_diff: float = 0.0  # normalized version
    effect_rel_logit_delta: float = 0.0  # relative change from baseline
    effect_reciprocal_rank: float = 0.0  # rank of target option

    # Combined multilabel metrics (only populated for "combined" perspective)
    # These aggregate across both label systems using logaddexp
    combined_logit_short: float = 0.0  # logaddexp(clean_short, corrupt_short)
    combined_logit_long: float = 0.0  # logaddexp(clean_long, corrupt_long)
    combined_logit_diff: float = 0.0  # combined_short - combined_long
    combined_prob_short: float = 0.0  # exp(combined_logit_short) normalized
    combined_prob_long: float = 0.0  # exp(combined_logit_long) normalized

    @classmethod
    def from_choice(
        cls,
        choice: IntervenedChoice | None,
        label_perspective: LabelPerspective = "clean",
    ) -> IntervenedChoiceMetrics:
        """Extract metrics from an IntervenedChoice.

        For multilabel choices, this extracts AGGREGATED metrics using the default
        aggregation method (MEAN_NORMALIZED). All metrics are internally consistent.

        Args:
            choice: IntervenedChoice to extract metrics from, or None
            label_perspective: For combined perspective with multilabel, uses logaddexp.
                              Ignored for aggregated extraction (uses all forks).

        Returns:
            IntervenedChoiceMetrics with extracted values or defaults
        """
        if choice is None:
            return cls()

        n_labels = choice.n_labels

        # For combined perspective, use logaddexp aggregation
        if label_perspective == "combined" and n_labels > 1:
            return cls._extract_combined_metrics(choice)

        # For multilabel, use aggregated metrics (consistent recovery/logit_diff)
        if n_labels > 1:
            return cls._extract_aggregated_metrics(choice, DEFAULT_AGGREGATION)

        # Single-label: extract from fork 0
        return cls._extract_per_fork_metrics(choice, 0, "clean")

    @classmethod
    def from_choice_aggregated(
        cls,
        choice: IntervenedChoice | None,
        method: ForkAggregation = DEFAULT_AGGREGATION,
    ) -> IntervenedChoiceMetrics:
        """Extract aggregated metrics using a specific aggregation method.

        All metrics are internally consistent: recovery, disruption, logit_diff,
        and baseline_logit_diff all use the same aggregation method.

        Args:
            choice: IntervenedChoice to extract metrics from
            method: Aggregation method to use

        Returns:
            IntervenedChoiceMetrics with aggregated values
        """
        if choice is None:
            return cls()

        if choice.n_labels == 1:
            return cls._extract_per_fork_metrics(choice, 0, "clean")

        return cls._extract_aggregated_metrics(choice, method)

    @classmethod
    def from_choice_per_fork(
        cls,
        choice: IntervenedChoice | None,
        fork_idx: int,
    ) -> IntervenedChoiceMetrics:
        """Extract metrics for a specific fork (label pair).

        All metrics are internally consistent for that fork: recovery, disruption,
        and logit_diff are computed from that fork's values only.

        Args:
            choice: IntervenedChoice to extract metrics from
            fork_idx: Index of the fork to extract

        Returns:
            IntervenedChoiceMetrics for the specified fork
        """
        if choice is None:
            return cls()

        perspective: LabelPerspective = "clean" if fork_idx == 0 else "corrupted"
        return cls._extract_per_fork_metrics(choice, fork_idx, perspective)

    @classmethod
    def _extract_aggregated_metrics(
        cls,
        choice: IntervenedChoice,
        method: ForkAggregation,
    ) -> IntervenedChoiceMetrics:
        """Extract metrics using aggregated values across all forks.

        For multilabel, aggregates logprobs using the specified method,
        then computes recovery/disruption from the aggregated logit_diff.
        """
        from ..common.choice import GroupedBinaryChoice

        intervened = choice.intervened
        baseline = (
            choice.baseline_corrupted
            if choice.mode == "denoising"
            else choice.baseline_clean
        )

        # Get aggregated recovery/disruption using the method
        recovery = choice.get_recovery_by_method(method)
        disruption = choice.get_disruption_by_method(method)
        effect = recovery if choice.mode == "denoising" else disruption

        # Compute aggregated baseline logit_diff
        baseline_logit_diff = 0.0
        if baseline is not None and isinstance(baseline, GroupedBinaryChoice):
            baseline_lps = baseline._aggregated_logprobs_by_method(method)
            baseline_logit_diff = baseline_lps[0] - baseline_lps[1]
        elif baseline is not None:
            lps = baseline.divergent_logprobs
            baseline_logit_diff = lps[0] - lps[1]

        metrics = cls(
            mode=choice.mode,
            recovery=recovery,
            disruption=disruption,
            flipped=choice.flipped,
            baseline_logit_diff=baseline_logit_diff,
            effect=effect,
            label_perspective="clean",  # Aggregated doesn't have a single perspective
            n_labels=choice.n_labels,
            aggregation_method=method.value,
        )

        if intervened is None:
            return metrics

        # Get aggregated logprobs for intervened
        if isinstance(intervened, GroupedBinaryChoice):
            lps = intervened._aggregated_logprobs_by_method(method)
        else:
            lps = intervened.divergent_logprobs
        lp_short, lp_long = lps

        metrics.logprob_short = lp_short
        metrics.logprob_long = lp_long
        metrics.prob_short = math.exp(lp_short) if lp_short > -50 else 0.0
        metrics.prob_long = math.exp(lp_long) if lp_long > -50 else 0.0
        metrics.logit_diff = lp_short - lp_long
        metrics.effect_logit_diff = (
            metrics.logit_diff if metrics.mode == "denoising" else -metrics.logit_diff
        )

        # Compute relative logit delta (consistent aggregation)
        if abs(metrics.baseline_logit_diff) > 1e-6:
            metrics.rel_logit_delta = (
                metrics.logit_diff - metrics.baseline_logit_diff
            ) / abs(metrics.baseline_logit_diff)
            metrics.effect_rel_logit_delta = (
                metrics.rel_logit_delta
                if metrics.mode == "denoising"
                else -metrics.rel_logit_delta
            )

        # For aggregated metrics, we average fork-level metrics
        cls._extract_averaged_fork_metrics(metrics, intervened)

        return metrics

    @classmethod
    def _extract_averaged_fork_metrics(
        cls,
        metrics: IntervenedChoiceMetrics,
        intervened,
    ) -> None:
        """Extract and average fork-level metrics (entropy, diversity, etc.)."""
        from ..common.choice import GroupedBinaryChoice

        if not isinstance(intervened, GroupedBinaryChoice):
            # Single fork - extract directly
            tree = intervened.tree
            if tree.forks and tree.forks[0].analysis:
                fork_metrics = tree.forks[0].analysis.metrics
                metrics.fork_entropy = fork_metrics.fork_entropy
                metrics.fork_diversity = fork_metrics.fork_diversity
                metrics.fork_simpson = fork_metrics.fork_simpson
                metrics.reciprocal_rank_short = fork_metrics.reciprocal_rank_a
                metrics.reciprocal_rank_long = 1.0 - fork_metrics.reciprocal_rank_a + 0.5
                # Logits
                if fork_metrics.logits is not None:
                    metrics.logit_short, metrics.logit_long = fork_metrics.logits
                if fork_metrics.normalized_logits is not None:
                    metrics.norm_logit_short, metrics.norm_logit_long = fork_metrics.normalized_logits
                    metrics.norm_logit_diff = metrics.norm_logit_short - metrics.norm_logit_long
                    metrics.effect_norm_logit_diff = (
                        metrics.norm_logit_diff
                        if metrics.mode == "denoising"
                        else -metrics.norm_logit_diff
                    )
            if tree.nodes and tree.nodes[0].analysis:
                node_metrics = tree.nodes[0].analysis.metrics
                metrics.vocab_entropy = node_metrics.vocab_entropy
                metrics.vocab_diversity = node_metrics.vocab_diversity
                metrics.vocab_simpson = node_metrics.vocab_simpson
                metrics.vocab_tcb = node_metrics.vocab_tcb
            return

        # Multiple forks - average the metrics
        tree = intervened.tree
        if not tree.forks:
            return

        fork_entropies = []
        fork_diversities = []
        fork_simpsons = []
        reciprocal_ranks = []
        logits_short = []
        logits_long = []
        norm_logits_short = []
        norm_logits_long = []

        for fork in tree.forks:
            if fork.analysis:
                fm = fork.analysis.metrics
                fork_entropies.append(fm.fork_entropy)
                fork_diversities.append(fm.fork_diversity)
                fork_simpsons.append(fm.fork_simpson)
                reciprocal_ranks.append(fm.reciprocal_rank_a)
                if fm.logits is not None:
                    logits_short.append(fm.logits[0])
                    logits_long.append(fm.logits[1])
                if fm.normalized_logits is not None:
                    norm_logits_short.append(fm.normalized_logits[0])
                    norm_logits_long.append(fm.normalized_logits[1])

        if fork_entropies:
            metrics.fork_entropy = sum(fork_entropies) / len(fork_entropies)
            metrics.fork_diversity = sum(fork_diversities) / len(fork_diversities)
            metrics.fork_simpson = sum(fork_simpsons) / len(fork_simpsons)
            avg_rr = sum(reciprocal_ranks) / len(reciprocal_ranks)
            metrics.reciprocal_rank_short = avg_rr
            metrics.reciprocal_rank_long = 1.0 - avg_rr + 0.5
            metrics.effect_reciprocal_rank = (
                metrics.reciprocal_rank_short
                if metrics.mode == "denoising"
                else metrics.reciprocal_rank_long
            )

        # Average logits across forks
        if logits_short:
            metrics.logit_short = sum(logits_short) / len(logits_short)
            metrics.logit_long = sum(logits_long) / len(logits_long)
        if norm_logits_short:
            metrics.norm_logit_short = sum(norm_logits_short) / len(norm_logits_short)
            metrics.norm_logit_long = sum(norm_logits_long) / len(norm_logits_long)
            metrics.norm_logit_diff = metrics.norm_logit_short - metrics.norm_logit_long
            metrics.effect_norm_logit_diff = (
                metrics.norm_logit_diff
                if metrics.mode == "denoising"
                else -metrics.norm_logit_diff
            )

        # Node metrics - use first node (shared across forks)
        if tree.nodes and tree.nodes[0].analysis:
            node_metrics = tree.nodes[0].analysis.metrics
            metrics.vocab_entropy = node_metrics.vocab_entropy
            metrics.vocab_diversity = node_metrics.vocab_diversity
            metrics.vocab_simpson = node_metrics.vocab_simpson
            metrics.vocab_tcb = node_metrics.vocab_tcb

    @classmethod
    def _extract_per_fork_metrics(
        cls,
        choice: IntervenedChoice,
        fork_idx: int,
        label_perspective: LabelPerspective,
    ) -> IntervenedChoiceMetrics:
        """Extract metrics for a specific fork.

        All metrics are internally consistent for that fork.
        Handles both live choice objects and cached/loaded data.
        """
        from ..common.choice import GroupedBinaryChoice

        intervened = choice.intervened
        baseline = (
            choice.baseline_corrupted
            if choice.mode == "denoising"
            else choice.baseline_clean
        )

        # Check if we're working with cached data (no live choice objects)
        is_cached = choice.baseline_clean is None

        if is_cached:
            # Use cached data
            return cls._extract_per_fork_from_cached(choice, fork_idx, label_perspective)

        is_grouped = isinstance(intervened, GroupedBinaryChoice)
        actual_fork_idx = fork_idx if is_grouped else 0

        # Get per-fork recovery/disruption
        if is_grouped:
            recovery = choice.get_recovery_for_fork(fork_idx)
            disruption = choice.get_disruption_for_fork(fork_idx)
        else:
            recovery = choice.recovery
            disruption = choice.disruption
        effect = recovery if choice.mode == "denoising" else disruption

        # Extract baseline logit diff for this fork
        baseline_logit_diff = 0.0
        if baseline is not None:
            # Use .forks property which safely returns () if tree.forks is None
            baseline_forks = baseline.forks if isinstance(baseline, GroupedBinaryChoice) else ()
            if fork_idx < len(baseline_forks):
                fork = baseline_forks[fork_idx]
                orig_lps = (float(fork.next_token_logprobs[0]), float(fork.next_token_logprobs[1]))
            else:
                orig_lps = baseline.divergent_logprobs
            baseline_logit_diff = orig_lps[0] - orig_lps[1]

        metrics = cls(
            mode=choice.mode,
            recovery=recovery,
            disruption=disruption,
            flipped=choice.flipped,
            baseline_logit_diff=baseline_logit_diff,
            effect=effect,
            label_perspective=label_perspective,
            n_labels=choice.n_labels,
            fork_idx=fork_idx if is_grouped else None,
        )

        if intervened is None:
            return metrics

        tree = intervened.tree
        if not tree.forks or actual_fork_idx >= len(tree.forks):
            return metrics

        fork = tree.forks[actual_fork_idx]

        # Logprobs directly from fork
        lp_short, lp_long = float(fork.next_token_logprobs[0]), float(fork.next_token_logprobs[1])
        metrics.logprob_short = lp_short
        metrics.logprob_long = lp_long
        metrics.prob_short = math.exp(lp_short) if lp_short > -50 else 0.0
        metrics.prob_long = math.exp(lp_long) if lp_long > -50 else 0.0
        metrics.logit_diff = lp_short - lp_long

        # ForkMetrics from fork.analysis
        if fork.analysis is not None:
            fork_metrics = fork.analysis.metrics
            metrics.logit_diff = fork_metrics.logit_diff
            metrics.effect_logit_diff = (
                metrics.logit_diff if metrics.mode == "denoising" else -metrics.logit_diff
            )
            metrics.fork_entropy = fork_metrics.fork_entropy
            metrics.fork_diversity = fork_metrics.fork_diversity
            metrics.fork_simpson = fork_metrics.fork_simpson
            metrics.reciprocal_rank_short = fork_metrics.reciprocal_rank_a
            metrics.reciprocal_rank_long = 1.0 - fork_metrics.reciprocal_rank_a + 0.5
            metrics.effect_reciprocal_rank = (
                metrics.reciprocal_rank_short
                if metrics.mode == "denoising"
                else metrics.reciprocal_rank_long
            )
            if fork_metrics.logits is not None:
                metrics.logit_short, metrics.logit_long = fork_metrics.logits
            if fork_metrics.normalized_logits is not None:
                metrics.norm_logit_short, metrics.norm_logit_long = fork_metrics.normalized_logits
                metrics.norm_logit_diff = metrics.norm_logit_short - metrics.norm_logit_long
                metrics.effect_norm_logit_diff = (
                    metrics.norm_logit_diff
                    if metrics.mode == "denoising"
                    else -metrics.norm_logit_diff
                )

        # Compute relative logit delta
        if abs(metrics.baseline_logit_diff) > 1e-6:
            metrics.rel_logit_delta = (
                metrics.logit_diff - metrics.baseline_logit_diff
            ) / abs(metrics.baseline_logit_diff)
            metrics.effect_rel_logit_delta = (
                metrics.rel_logit_delta
                if metrics.mode == "denoising"
                else -metrics.rel_logit_delta
            )

        # NodeMetrics from first node (shared)
        if tree.nodes and tree.nodes[0].analysis:
            node_metrics = tree.nodes[0].analysis.metrics
            metrics.vocab_entropy = node_metrics.vocab_entropy
            metrics.vocab_diversity = node_metrics.vocab_diversity
            metrics.vocab_simpson = node_metrics.vocab_simpson
            metrics.vocab_tcb = node_metrics.vocab_tcb

        # Trajectory metrics - get trajectories directly
        # For grouped: trajs are [fork0_a, fork0_b, fork1_a, fork1_b, ...]
        traj_idx_a = 2 * actual_fork_idx if is_grouped else 0
        traj_idx_b = 2 * actual_fork_idx + 1 if is_grouped else 1

        if tree.trajs and traj_idx_b < len(tree.trajs):
            traj_a = tree.trajs[traj_idx_a]
            traj_b = tree.trajs[traj_idx_b]

            # Chosen trajectory based on logprob comparison
            chosen_traj = traj_a if lp_short >= lp_long else traj_b
            if chosen_traj.analysis and chosen_traj.analysis.full_traj:
                metrics.inv_perplexity = chosen_traj.analysis.full_traj.inv_perplexity

            # Per-trajectory inv_perplexity
            if traj_a.analysis and traj_a.analysis.continuation_only:
                metrics.traj_inv_perplexity_short = traj_a.analysis.continuation_only.inv_perplexity
            if traj_b.analysis and traj_b.analysis.continuation_only:
                metrics.traj_inv_perplexity_long = traj_b.analysis.continuation_only.inv_perplexity

        return metrics

    @classmethod
    def _extract_per_fork_from_cached(
        cls,
        choice: IntervenedChoice,
        fork_idx: int,
        label_perspective: LabelPerspective,
    ) -> IntervenedChoiceMetrics:
        """Extract per-fork metrics from cached/loaded IntervenedChoice.

        Used when choice.baseline_clean is None (data loaded from lightweight dict).
        """
        # Get per-fork recovery/disruption (these now use cached data internally)
        n_labels = choice.n_labels
        is_multilabel = n_labels > 1

        if is_multilabel:
            recovery = choice.get_recovery_for_fork(fork_idx)
            disruption = choice.get_disruption_for_fork(fork_idx)
        else:
            recovery = choice.recovery
            disruption = choice.disruption
        effect = recovery if choice.mode == "denoising" else disruption

        # Extract baseline logit diff for this fork from cached data
        baseline_logit_diff = 0.0
        if choice._cached_per_fork_logprobs and fork_idx < len(choice._cached_per_fork_logprobs[0]):
            # Get baseline based on mode (corrupted for denoising, clean for noising)
            baseline_idx = 1 if choice.mode == "denoising" else 0  # corrupted=1, clean=0
            lps = choice._cached_per_fork_logprobs[baseline_idx][fork_idx]
            baseline_logit_diff = lps[0] - lps[1]
            if choice.switched:
                baseline_logit_diff = -baseline_logit_diff
        elif choice._cached_logprobs:
            # Fall back to aggregated logprobs
            baseline_idx = 1 if choice.mode == "denoising" else 0
            lps = choice._cached_logprobs[baseline_idx]
            baseline_logit_diff = lps[0] - lps[1]
            if choice.switched:
                baseline_logit_diff = -baseline_logit_diff

        # Extract intervened logprobs for this fork
        lp_short, lp_long = -10.0, -10.0
        if choice._cached_per_fork_logprobs and fork_idx < len(choice._cached_per_fork_logprobs[2]):
            lps = choice._cached_per_fork_logprobs[2][fork_idx]  # intervened=2
            lp_short, lp_long = lps
        elif choice._cached_logprobs:
            lp_short, lp_long = choice._cached_logprobs[2]  # intervened=2

        logit_diff = lp_short - lp_long

        metrics = cls(
            mode=choice.mode,
            recovery=recovery,
            disruption=disruption,
            flipped=choice.flipped,
            baseline_logit_diff=baseline_logit_diff,
            effect=effect,
            label_perspective=label_perspective,
            n_labels=n_labels,
            fork_idx=fork_idx if is_multilabel else None,
            logprob_short=lp_short,
            logprob_long=lp_long,
            prob_short=math.exp(lp_short) if lp_short > -50 else 0.0,
            prob_long=math.exp(lp_long) if lp_long > -50 else 0.0,
            logit_diff=logit_diff,
            effect_logit_diff=logit_diff if choice.mode == "denoising" else -logit_diff,
        )

        # Compute relative logit delta
        if abs(metrics.baseline_logit_diff) > 1e-6:
            metrics.rel_logit_delta = (
                metrics.logit_diff - metrics.baseline_logit_diff
            ) / abs(metrics.baseline_logit_diff)
            metrics.effect_rel_logit_delta = (
                metrics.rel_logit_delta
                if metrics.mode == "denoising"
                else -metrics.rel_logit_delta
            )

        # Extract cached per-fork metrics (fork entropy, logits, etc.)
        if choice._cached_per_fork_metrics and fork_idx < len(choice._cached_per_fork_metrics):
            fm = choice._cached_per_fork_metrics[fork_idx]
            if "fork_entropy" in fm:
                metrics.fork_entropy = fm["fork_entropy"]
            if "fork_diversity" in fm:
                metrics.fork_diversity = fm["fork_diversity"]
            if "fork_simpson" in fm:
                metrics.fork_simpson = fm["fork_simpson"]
            if "reciprocal_rank_a" in fm:
                metrics.reciprocal_rank_short = fm["reciprocal_rank_a"]
                metrics.reciprocal_rank_long = 1.0 - fm["reciprocal_rank_a"] + 0.5
                metrics.effect_reciprocal_rank = (
                    metrics.reciprocal_rank_short
                    if metrics.mode == "denoising"
                    else metrics.reciprocal_rank_long
                )
            if "logits" in fm:
                metrics.logit_short, metrics.logit_long = fm["logits"]
            if "normalized_logits" in fm:
                metrics.norm_logit_short, metrics.norm_logit_long = fm["normalized_logits"]
                metrics.norm_logit_diff = metrics.norm_logit_short - metrics.norm_logit_long
                metrics.effect_norm_logit_diff = (
                    metrics.norm_logit_diff
                    if metrics.mode == "denoising"
                    else -metrics.norm_logit_diff
                )

        # Extract cached vocab metrics (shared across forks)
        if choice._cached_vocab_metrics:
            vm = choice._cached_vocab_metrics
            if "vocab_entropy" in vm:
                metrics.vocab_entropy = vm["vocab_entropy"]
            if "vocab_diversity" in vm:
                metrics.vocab_diversity = vm["vocab_diversity"]
            if "vocab_simpson" in vm:
                metrics.vocab_simpson = vm["vocab_simpson"]
            if "vocab_tcb" in vm:
                metrics.vocab_tcb = vm["vocab_tcb"]

        return metrics

    @classmethod
    def _extract_combined_metrics(
        cls,
        choice: IntervenedChoice,
    ) -> IntervenedChoiceMetrics:
        """Extract combined metrics using logaddexp aggregation across forks.

        This is semantically different from arithmetic aggregation methods.
        Uses logaddexp to combine logits, then computes recovery/disruption
        from the combined values.
        """
        from ..common.choice import GroupedBinaryChoice

        # Get combined recovery/disruption
        recovery = choice.get_recovery_combined()
        disruption = choice.get_disruption_combined()
        effect = recovery if choice.mode == "denoising" else disruption

        # Compute combined baseline logit_diff
        baseline = (
            choice.baseline_corrupted
            if choice.mode == "denoising"
            else choice.baseline_clean
        )
        baseline_logit_diff = choice._get_logit_diff_combined(baseline) if baseline else 0.0

        metrics = cls(
            mode=choice.mode,
            recovery=recovery,
            disruption=disruption,
            flipped=choice.flipped,
            baseline_logit_diff=baseline_logit_diff,
            effect=effect,
            label_perspective="combined",
            n_labels=choice.n_labels,
            aggregation_method="combined",
        )

        intervened = choice.intervened
        if intervened is None or not isinstance(intervened, GroupedBinaryChoice):
            return metrics

        if intervened.n_forks < 2:
            return metrics

        # Get vocab_logits from both forks
        clean_fork = intervened.tree.forks[0] if intervened.tree.forks else None
        corrupt_fork = (
            intervened.tree.forks[1] if len(intervened.tree.forks) > 1 else None
        )

        if not clean_fork or not corrupt_fork:
            return metrics

        clean_vocab = clean_fork.vocab_logits
        corrupt_vocab = corrupt_fork.vocab_logits

        if clean_vocab is None or corrupt_vocab is None:
            # Fall back to using logprobs if raw logits not available
            lp_clean_short, lp_clean_long = clean_fork.next_token_logprobs
            lp_corrupt_short, lp_corrupt_long = corrupt_fork.next_token_logprobs

            combined_short = logaddexp(float(lp_clean_short), float(lp_corrupt_short))
            combined_long = logaddexp(float(lp_clean_long), float(lp_corrupt_long))
        else:
            # Get token IDs from both forks
            clean_short_id, clean_long_id = clean_fork.next_token_ids
            corrupt_short_id, corrupt_long_id = corrupt_fork.next_token_ids

            # Get logits for each token
            clean_short_logit = clean_vocab[clean_short_id]
            clean_long_logit = clean_vocab[clean_long_id]
            corrupt_short_logit = corrupt_vocab[corrupt_short_id]
            corrupt_long_logit = corrupt_vocab[corrupt_long_id]

            combined_short = logaddexp(clean_short_logit, corrupt_short_logit)
            combined_long = logaddexp(clean_long_logit, corrupt_long_logit)

        metrics.combined_logit_short = combined_short
        metrics.combined_logit_long = combined_long
        metrics.combined_logit_diff = combined_short - combined_long

        # Convert to probabilities via softmax
        max_logit = max(combined_short, combined_long)
        exp_short = math.exp(combined_short - max_logit)
        exp_long = math.exp(combined_long - max_logit)
        total = exp_short + exp_long
        metrics.combined_prob_short = exp_short / total
        metrics.combined_prob_long = exp_long / total

        # Update main metrics to use combined values
        metrics.logprob_short = combined_short
        metrics.logprob_long = combined_long
        metrics.prob_short = metrics.combined_prob_short
        metrics.prob_long = metrics.combined_prob_long
        metrics.logit_diff = metrics.combined_logit_diff
        metrics.effect_logit_diff = (
            metrics.combined_logit_diff
            if metrics.mode == "denoising"
            else -metrics.combined_logit_diff
        )

        # Compute relative logit delta
        if abs(metrics.baseline_logit_diff) > 1e-6:
            metrics.rel_logit_delta = (
                metrics.logit_diff - metrics.baseline_logit_diff
            ) / abs(metrics.baseline_logit_diff)
            metrics.effect_rel_logit_delta = (
                metrics.rel_logit_delta
                if metrics.mode == "denoising"
                else -metrics.rel_logit_delta
            )

        # Average fork-level metrics
        cls._extract_averaged_fork_metrics(metrics, intervened)

        return metrics

    @classmethod
    def from_dict(cls, d: dict) -> IntervenedChoiceMetrics:
        """Create from dictionary."""
        return super().from_dict(d)
