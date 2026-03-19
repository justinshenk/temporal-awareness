"""IntervenedChoice: result of a single activation patching intervention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from ..common.base_schema import BaseSchema
from ..common.choice import LabeledSimpleBinaryChoice, GroupedBinaryChoice
from ..common.choice.grouped_binary_choice import ForkAggregation
from ..common.math import logaddexp
from ..common.math.faithfulness_scores import compute_disruption, compute_recovery
from ..common.patching_types import PatchingMode

# Type alias for choices that can be single or grouped
ChoiceType = Union[LabeledSimpleBinaryChoice, GroupedBinaryChoice]


@dataclass
class IntervenedChoice(BaseSchema):
    """Single intervention result.

    Stores all three values needed for proper recovery/disruption calculation:
    - baseline_clean: Result from running clean input (no intervention)
    - baseline_corrupted: Result from running corrupted input (no intervention)
    - intervened: Result after applying intervention

    Each can be either LabeledSimpleBinaryChoice (single label pair) or
    GroupedBinaryChoice (multiple label pairs). When GroupedBinaryChoice,
    use .get_choice(i) to extract per-label LabeledSimpleBinaryChoice.

    Mode semantics:
    - denoising: Run corrupted text, inject clean activations -> recover clean behavior
    - noising: Run clean text, inject corrupted activations -> disrupt clean behavior
    """

    baseline_clean: ChoiceType | None = None
    baseline_corrupted: ChoiceType | None = None
    intervened: ChoiceType | None = None
    mode: PatchingMode = "denoising"
    switched: bool = False

    # Cached metrics for lightweight loading (when full objects are None)
    _cached_recovery: float | None = field(default=None, repr=False)
    _cached_disruption: float | None = field(default=None, repr=False)
    _cached_flipped: bool | None = field(default=None, repr=False)
    _cached_choice_idxs: tuple[int, int, int] | None = field(default=None, repr=False)
    _cached_logprobs: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = field(default=None, repr=False)
    _cached_n_labels: int | None = field(default=None, repr=False)

    # Tell _canon to call to_dict() directly instead of processing fields
    _use_custom_to_dict: bool = True

    @property
    def is_grouped(self) -> bool:
        """Whether this contains GroupedBinaryChoice (multiple label pairs)."""
        return isinstance(self.baseline_clean, GroupedBinaryChoice)

    @property
    def n_labels(self) -> int:
        """Number of label pairs."""
        if self._cached_n_labels is not None:
            return self._cached_n_labels
        if isinstance(self.baseline_clean, GroupedBinaryChoice):
            return self.baseline_clean.n_forks
        return 1

    def to_dict(self, **kwargs) -> dict:
        """Convert to lightweight dict with only metrics, not full trees."""
        if self.baseline_clean is None:
            return {
                "mode": self.mode.value if hasattr(self.mode, "value") else self.mode,
                "switched": self.switched,
                "n_labels": self._cached_n_labels or 1,
                "recovery": self._cached_recovery,
                "disruption": self._cached_disruption,
                "flipped": self._cached_flipped,
                "baseline_clean_choice_idx": self._cached_choice_idxs[0] if self._cached_choice_idxs else None,
                "baseline_corrupted_choice_idx": self._cached_choice_idxs[1] if self._cached_choice_idxs else None,
                "intervened_choice_idx": self._cached_choice_idxs[2] if self._cached_choice_idxs else None,
                "baseline_clean_logprobs": list(self._cached_logprobs[0]) if self._cached_logprobs else None,
                "baseline_corrupted_logprobs": list(self._cached_logprobs[1]) if self._cached_logprobs else None,
                "intervened_logprobs": list(self._cached_logprobs[2]) if self._cached_logprobs else None,
            }
        return {
            "mode": self.mode.value if hasattr(self.mode, "value") else self.mode,
            "switched": self.switched,
            "n_labels": self.n_labels,
            "recovery": self.recovery,
            "disruption": self.disruption,
            "flipped": self.flipped,
            "baseline_clean_choice_idx": self.baseline_clean.choice_idx,
            "baseline_corrupted_choice_idx": self.baseline_corrupted.choice_idx,
            "intervened_choice_idx": self.intervened.choice_idx,
            "baseline_clean_logprobs": list(self.baseline_clean.divergent_logprobs),
            "baseline_corrupted_logprobs": list(self.baseline_corrupted.divergent_logprobs),
            "intervened_logprobs": list(self.intervened.divergent_logprobs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> IntervenedChoice:
        """Load from dict, handling both full and lightweight formats."""
        if "recovery" in d and "baseline_clean" not in d:
            return cls(
                baseline_clean=None,
                baseline_corrupted=None,
                intervened=None,
                mode=d.get("mode", "denoising"),
                switched=d.get("switched", False),
                _cached_recovery=d.get("recovery"),
                _cached_disruption=d.get("disruption"),
                _cached_flipped=d.get("flipped"),
                _cached_choice_idxs=(
                    d.get("baseline_clean_choice_idx"),
                    d.get("baseline_corrupted_choice_idx"),
                    d.get("intervened_choice_idx"),
                ),
                _cached_logprobs=(
                    tuple(d.get("baseline_clean_logprobs", (0.0, 0.0))),
                    tuple(d.get("baseline_corrupted_logprobs", (0.0, 0.0))),
                    tuple(d.get("intervened_logprobs", (0.0, 0.0))),
                ),
                _cached_n_labels=d.get("n_labels", 1),
            )
        return super().from_dict(d)

    def pop_heavy(self) -> None:
        """Remove heavy data (trees) from choices to reduce memory."""
        if self.baseline_clean:
            self.baseline_clean.pop_heavy()
        if self.baseline_corrupted:
            self.baseline_corrupted.pop_heavy()
        if self.intervened:
            self.intervened.pop_heavy()

    def _get_logit_diff(self, choice: ChoiceType) -> float:
        """Get logit difference (A - B) from a choice, respecting switched flag."""
        logits = choice.divergent_logits
        if logits:
            if self.switched:
                return logits[1] - logits[0]
            return logits[0] - logits[1]
        lps = choice.divergent_logprobs
        if self.switched:
            return lps[1] - lps[0]
        return lps[0] - lps[1]

    @property
    def recovery(self) -> float:
        """Recovery score (aggregated across label pairs if grouped)."""
        if self._cached_recovery is not None:
            return self._cached_recovery

        y_clean = self._get_logit_diff(self.baseline_clean)
        y_corrupted = self._get_logit_diff(self.baseline_corrupted)
        y_intervened = self._get_logit_diff(self.intervened)
        return compute_recovery(y_intervened, y_clean, y_corrupted)

    @property
    def disruption(self) -> float:
        """Disruption score (aggregated across label pairs if grouped)."""
        if self._cached_disruption is not None:
            return self._cached_disruption

        y_clean = self._get_logit_diff(self.baseline_clean)
        y_corrupted = self._get_logit_diff(self.baseline_corrupted)
        y_intervened = self._get_logit_diff(self.intervened)
        return compute_disruption(y_intervened, y_clean, y_corrupted)

    @property
    def effect(self) -> float:
        """Effect metric: recovery for denoising, disruption for noising."""
        return self.recovery if self.mode == "denoising" else self.disruption

    @property
    def flipped(self) -> bool:
        """Whether the intervention flipped the choice."""
        if self._cached_flipped is not None:
            return self._cached_flipped
        if self.mode == "denoising":
            return self.baseline_corrupted.choice_idx != self.intervened.choice_idx
        return self.baseline_clean.choice_idx != self.intervened.choice_idx

    def switch(self) -> IntervenedChoice:
        """Swap clean↔corrupted semantics."""
        new_mode: PatchingMode = "noising" if self.mode == "denoising" else "denoising"
        if self.baseline_clean is None:
            new_choice_idxs = None
            if self._cached_choice_idxs:
                new_choice_idxs = (
                    self._cached_choice_idxs[1],
                    self._cached_choice_idxs[0],
                    self._cached_choice_idxs[2],
                )
            new_logprobs = None
            if self._cached_logprobs:
                new_logprobs = (
                    self._cached_logprobs[1],
                    self._cached_logprobs[0],
                    self._cached_logprobs[2],
                )
            return IntervenedChoice(
                baseline_clean=None,
                baseline_corrupted=None,
                intervened=None,
                mode=new_mode,
                switched=not self.switched,
                _cached_recovery=self._cached_disruption,
                _cached_disruption=self._cached_recovery,
                _cached_flipped=self._cached_flipped,
                _cached_choice_idxs=new_choice_idxs,
                _cached_logprobs=new_logprobs,
                _cached_n_labels=self._cached_n_labels,
            )
        return IntervenedChoice(
            baseline_clean=self.baseline_corrupted,
            baseline_corrupted=self.baseline_clean,
            intervened=self.intervened,
            mode=new_mode,
            switched=not self.switched,
        )

    # ── Per-Fork and Per-Method Access ────────────────────────────────────

    def _get_fork_logprobs(
        self, choice: ChoiceType, fork_idx: int
    ) -> tuple[float, float]:
        """Get logprobs from a specific fork of a GroupedBinaryChoice."""
        if not isinstance(choice, GroupedBinaryChoice):
            return choice.divergent_logprobs
        if not choice.tree.forks or fork_idx >= len(choice.tree.forks):
            return (0.0, 0.0)
        fork = choice.tree.forks[fork_idx]
        return (float(fork.next_token_logprobs[0]), float(fork.next_token_logprobs[1]))

    def _get_logit_diff_for_fork(self, choice: ChoiceType, fork_idx: int) -> float:
        """Get logit difference for a specific fork."""
        lps = self._get_fork_logprobs(choice, fork_idx)
        if self.switched:
            return lps[1] - lps[0]
        return lps[0] - lps[1]

    def _get_logit_diff_by_method(
        self, choice: ChoiceType, method: ForkAggregation
    ) -> float:
        """Get logit difference using a specific aggregation method."""
        if not isinstance(choice, GroupedBinaryChoice):
            return self._get_logit_diff(choice)
        lps = choice._aggregated_logprobs_by_method(method)
        if self.switched:
            return lps[1] - lps[0]
        return lps[0] - lps[1]

    def _get_logit_diff_combined(self, choice: ChoiceType) -> float:
        """Get logit difference using logaddexp combination across forks."""
        if not isinstance(choice, GroupedBinaryChoice) or choice.n_forks < 2:
            return self._get_logit_diff(choice)

        # Combine logprobs across forks using logaddexp
        lp_a_combined = None
        lp_b_combined = None
        for fork in choice.tree.forks:
            lp_a, lp_b = float(fork.next_token_logprobs[0]), float(fork.next_token_logprobs[1])
            if lp_a_combined is None:
                lp_a_combined, lp_b_combined = lp_a, lp_b
            else:
                lp_a_combined = logaddexp(lp_a_combined, lp_a)
                lp_b_combined = logaddexp(lp_b_combined, lp_b)

        if self.switched:
            return lp_b_combined - lp_a_combined
        return lp_a_combined - lp_b_combined

    def get_recovery_for_fork(self, fork_idx: int) -> float:
        """Get recovery score for a specific fork (label pair)."""
        if self.baseline_clean is None:
            return 0.0
        y_clean = self._get_logit_diff_for_fork(self.baseline_clean, fork_idx)
        y_corrupted = self._get_logit_diff_for_fork(self.baseline_corrupted, fork_idx)
        y_intervened = self._get_logit_diff_for_fork(self.intervened, fork_idx)
        return compute_recovery(y_intervened, y_clean, y_corrupted)

    def get_disruption_for_fork(self, fork_idx: int) -> float:
        """Get disruption score for a specific fork (label pair)."""
        if self.baseline_clean is None:
            return 0.0
        y_clean = self._get_logit_diff_for_fork(self.baseline_clean, fork_idx)
        y_corrupted = self._get_logit_diff_for_fork(self.baseline_corrupted, fork_idx)
        y_intervened = self._get_logit_diff_for_fork(self.intervened, fork_idx)
        return compute_disruption(y_intervened, y_clean, y_corrupted)

    def get_recovery_by_method(self, method: ForkAggregation) -> float:
        """Get recovery score using a specific aggregation method."""
        if self.baseline_clean is None:
            return 0.0
        y_clean = self._get_logit_diff_by_method(self.baseline_clean, method)
        y_corrupted = self._get_logit_diff_by_method(self.baseline_corrupted, method)
        y_intervened = self._get_logit_diff_by_method(self.intervened, method)
        return compute_recovery(y_intervened, y_clean, y_corrupted)

    def get_disruption_by_method(self, method: ForkAggregation) -> float:
        """Get disruption score using a specific aggregation method."""
        if self.baseline_clean is None:
            return 0.0
        y_clean = self._get_logit_diff_by_method(self.baseline_clean, method)
        y_corrupted = self._get_logit_diff_by_method(self.baseline_corrupted, method)
        y_intervened = self._get_logit_diff_by_method(self.intervened, method)
        return compute_disruption(y_intervened, y_clean, y_corrupted)

    def get_recovery_combined(self) -> float:
        """Get recovery score using logaddexp combination across forks."""
        if self.baseline_clean is None:
            return 0.0
        y_clean = self._get_logit_diff_combined(self.baseline_clean)
        y_corrupted = self._get_logit_diff_combined(self.baseline_corrupted)
        y_intervened = self._get_logit_diff_combined(self.intervened)
        return compute_recovery(y_intervened, y_clean, y_corrupted)

    def get_disruption_combined(self) -> float:
        """Get disruption score using logaddexp combination across forks."""
        if self.baseline_clean is None:
            return 0.0
        y_clean = self._get_logit_diff_combined(self.baseline_clean)
        y_corrupted = self._get_logit_diff_combined(self.baseline_corrupted)
        y_intervened = self._get_logit_diff_combined(self.intervened)
        return compute_disruption(y_intervened, y_clean, y_corrupted)
