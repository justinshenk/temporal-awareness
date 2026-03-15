"""Metrics extracted from activation patching results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..common.base_schema import BaseSchema
from ..common.math import logaddexp
from ..common.patching_types import PatchingMode

if TYPE_CHECKING:
    from .intervened_choice import IntervenedChoice

# Label perspective types
LabelPerspective = Literal["clean", "corrupted", "combined"]


@dataclass
class IntervenedChoiceMetrics(BaseSchema):
    """Metrics extracted from an IntervenedChoice for visualization/analysis.

    Provides type-safe access to metrics with sensible defaults for missing data.

    When multilabel patching is used (clean_labels != corrupted_labels), this can
    be extracted for different perspectives:
    - "clean": Metrics using clean label token IDs
    - "corrupted": Metrics using corrupted label token IDs
    - "combined": Aggregated metrics using logaddexp across both label systems

    The default from_choice() uses "clean" perspective for compatibility.
    Use from_choice_for_label() to extract specific perspectives.
    """

    # Core metrics
    mode: PatchingMode = "denoising"
    recovery: float = 0.0
    disruption: float = 0.0
    flipped: bool = False
    label_perspective: LabelPerspective = "clean"
    n_labels: int = 1

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

        Handles missing data gracefully with safe defaults.

        Args:
            choice: IntervenedChoice to extract metrics from, or None
            label_perspective: Which label system to use for metrics:
                - "clean": Use clean labels (index 0) - default
                - "corrupted": Use corrupted labels (index 1)
                - "combined": Aggregate across both label systems

        Returns:
            IntervenedChoiceMetrics with extracted values or defaults
        """
        if choice is None:
            return cls()

        n_labels = choice.n_labels

        # For combined perspective, extract and aggregate
        if label_perspective == "combined" and n_labels > 1:
            return cls._extract_combined_metrics(choice)

        # For single-label or specific perspective, extract from appropriate fork
        label_idx = 0 if label_perspective == "clean" else 1
        if label_idx >= n_labels:
            label_idx = 0  # Fallback if label doesn't exist

        return cls._extract_single_label_metrics(choice, label_idx, label_perspective)

    @classmethod
    def _extract_single_label_metrics(
        cls,
        choice: IntervenedChoice,
        label_idx: int,
        label_perspective: LabelPerspective,
    ) -> IntervenedChoiceMetrics:
        """Extract metrics for a specific label index."""
        from ..common.choice import GroupedBinaryChoice

        # Extract baseline logit diff from the appropriate baseline
        # Denoising runs corrupted, noising runs clean
        baseline_logit_diff = 0.0
        baseline = (
            choice.baseline_corrupted
            if choice.mode == "denoising"
            else choice.baseline_clean
        )
        if baseline is not None:
            if (
                isinstance(baseline, GroupedBinaryChoice)
                and label_idx < baseline.n_forks
            ):
                sub_choice = baseline.get_choice(label_idx)
                orig_lps = sub_choice.divergent_logprobs
            else:
                orig_lps = baseline.divergent_logprobs
            baseline_logit_diff = orig_lps[0] - orig_lps[1]

        # Compute effect based on mode (recovery for denoising, disruption for noising)
        effect = choice.recovery if choice.mode == "denoising" else choice.disruption

        # Start with defaults
        metrics = cls(
            mode=choice.mode,
            recovery=choice.recovery,
            disruption=choice.disruption,
            flipped=choice.flipped,
            baseline_logit_diff=baseline_logit_diff,
            effect=effect,
            label_perspective=label_perspective,
            n_labels=choice.n_labels,
        )

        # Get the intervened choice for this label
        intervened = choice.intervened
        if intervened is None:
            return metrics

        # Extract the specific label's choice if grouped
        if (
            isinstance(intervened, GroupedBinaryChoice)
            and label_idx < intervened.n_forks
        ):
            sub_intervened = intervened.get_choice(label_idx)
            tree = sub_intervened.tree
        else:
            sub_intervened = intervened
            tree = intervened.tree

        # Logprobs from divergent_logprobs (short=0, long=1)
        lp_short, lp_long = sub_intervened.divergent_logprobs
        metrics.logprob_short = lp_short
        metrics.logprob_long = lp_long
        metrics.prob_short = math.exp(lp_short) if lp_short > -50 else 0.0
        metrics.prob_long = math.exp(lp_long) if lp_long > -50 else 0.0
        metrics.logit_diff = lp_short - lp_long

        # ForkMetrics from tree.forks[0].analysis.metrics
        if tree is None:
            raise ValueError("intervened.tree is None - tree required for fork metrics")
        if not tree.forks:
            raise ValueError(
                f"tree.forks is empty - tree has {len(tree.nodes) if tree.nodes else 0} nodes but no forks"
            )
        if tree.forks[0].analysis is None:
            raise ValueError(f"tree.forks[0].analysis is None - fork={tree.forks[0]}")

        fork_metrics = tree.forks[0].analysis.metrics
        metrics.logit_diff = fork_metrics.logit_diff
        # effect_logit_diff: target - source (positive = toward target)
        metrics.effect_logit_diff = (
            metrics.logit_diff if metrics.mode == "denoising" else -metrics.logit_diff
        )
        metrics.fork_entropy = fork_metrics.fork_entropy
        metrics.fork_diversity = fork_metrics.fork_diversity
        metrics.fork_simpson = fork_metrics.fork_simpson
        # reciprocal_rank_a is for the A token (short)
        metrics.reciprocal_rank_short = fork_metrics.reciprocal_rank_a
        # For B token, it's the complement
        metrics.reciprocal_rank_long = 1.0 - fork_metrics.reciprocal_rank_a + 0.5
        # effect_reciprocal_rank: short for denoising (target=clean), long for noising (target=corrupt)
        metrics.effect_reciprocal_rank = (
            metrics.reciprocal_rank_short
            if metrics.mode == "denoising"
            else metrics.reciprocal_rank_long
        )
        # Raw logits and normalized logits
        if fork_metrics.logits is not None:
            metrics.logit_short, metrics.logit_long = fork_metrics.logits
        if fork_metrics.normalized_logits is not None:
            metrics.norm_logit_short, metrics.norm_logit_long = (
                fork_metrics.normalized_logits
            )
            metrics.norm_logit_diff = metrics.norm_logit_short - metrics.norm_logit_long
            # effect_norm_logit_diff: target - source (positive = toward target)
            metrics.effect_norm_logit_diff = (
                metrics.norm_logit_diff
                if metrics.mode == "denoising"
                else -metrics.norm_logit_diff
            )

        # Compute relative logit delta (change from baseline, normalized)
        if abs(metrics.baseline_logit_diff) > 1e-6:
            metrics.rel_logit_delta = (
                metrics.logit_diff - metrics.baseline_logit_diff
            ) / abs(metrics.baseline_logit_diff)
            # effect_rel_logit_delta: target - source (positive = toward target)
            metrics.effect_rel_logit_delta = (
                metrics.rel_logit_delta
                if metrics.mode == "denoising"
                else -metrics.rel_logit_delta
            )

        # NodeMetrics from tree.nodes[0].analysis.metrics
        if tree.nodes and tree.nodes[0].analysis:
            node_metrics = tree.nodes[0].analysis.metrics
            metrics.vocab_entropy = node_metrics.vocab_entropy
            metrics.vocab_diversity = node_metrics.vocab_diversity
            metrics.vocab_simpson = node_metrics.vocab_simpson
            metrics.vocab_tcb = node_metrics.vocab_tcb

        # TrajectoryMetrics from chosen_traj.analysis.full_traj
        chosen = sub_intervened.chosen_traj
        if chosen and chosen.analysis and chosen.analysis.full_traj:
            traj_metrics = chosen.analysis.full_traj
            metrics.inv_perplexity = traj_metrics.inv_perplexity

        # Trajectory inv_perplexity from continuation_only (trajs[0]=A, trajs[1]=B)
        if tree.trajs and len(tree.trajs) >= 2:
            traj_a = tree.trajs[0]
            traj_b = tree.trajs[1]
            if traj_a.analysis and traj_a.analysis.continuation_only:
                metrics.traj_inv_perplexity_short = (
                    traj_a.analysis.continuation_only.inv_perplexity
                )
            if traj_b.analysis and traj_b.analysis.continuation_only:
                metrics.traj_inv_perplexity_long = (
                    traj_b.analysis.continuation_only.inv_perplexity
                )

        return metrics

    @classmethod
    def _extract_combined_metrics(
        cls,
        choice: IntervenedChoice,
    ) -> IntervenedChoiceMetrics:
        """Extract combined metrics aggregating across all label systems.

        Uses logaddexp to combine logits across label systems:
        - combined_short = logaddexp(clean_short, corrupt_short)
        - combined_long = logaddexp(clean_long, corrupt_long)
        """
        from ..common.choice import GroupedBinaryChoice

        # Start with clean label metrics as base
        metrics = cls._extract_single_label_metrics(choice, 0, "combined")
        metrics.label_perspective = "combined"

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
            clean_choice = intervened.get_choice(0)
            corrupt_choice = intervened.get_choice(1)

            lp_clean_short, lp_clean_long = clean_choice.divergent_logprobs
            lp_corrupt_short, lp_corrupt_long = corrupt_choice.divergent_logprobs

            # Combined using logaddexp: log(exp(a) + exp(b))
            combined_short = logaddexp(lp_clean_short, lp_corrupt_short)
            combined_long = logaddexp(lp_clean_long, lp_corrupt_long)
        else:
            # Get token IDs from both forks
            clean_short_id, clean_long_id = clean_fork.next_token_ids
            corrupt_short_id, corrupt_long_id = corrupt_fork.next_token_ids

            # Get logits for each token
            clean_short_logit = clean_vocab[clean_short_id]
            clean_long_logit = clean_vocab[clean_long_id]
            corrupt_short_logit = corrupt_vocab[corrupt_short_id]
            corrupt_long_logit = corrupt_vocab[corrupt_long_id]

            # Combined: aggregate across both label systems
            # patient_logit = logaddexp(clean_short, corrupt_short)
            # impatient_logit = logaddexp(clean_long, corrupt_long)
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

        # Update effect metrics to use combined values
        metrics.logit_diff = metrics.combined_logit_diff
        metrics.effect_logit_diff = (
            metrics.combined_logit_diff
            if metrics.mode == "denoising"
            else -metrics.combined_logit_diff
        )

        return metrics

    @classmethod
    def from_dict(cls, d: dict) -> IntervenedChoiceMetrics:
        """Create from dictionary."""
        return super().from_dict(d)
