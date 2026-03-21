"""Preference dataset analysis with detailed coherence breakdowns."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import TYPE_CHECKING

from ...common.logging import (
    log,
    log_banner,
    log_sub_banner,
    log_kv,
    log_divider,
)
from ..formatting.formatting_variation import (
    get_all_label_styles,
    get_simple_label_styles as _get_simple_label_styles,
)

if TYPE_CHECKING:
    from .preference_dataset import PreferenceDataset


def get_label_style_ids() -> dict[str, str]:
    """Get mapping from label_style string (e.g. 'a)_b)') to short ID (e.g. 'L1')."""
    result = {}
    for idx, (a, b) in enumerate(get_all_label_styles()):
        key = f"{a}_{b}"
        result[key] = f"L{idx + 1}"
    return result


def get_label_styles_display() -> dict[str, tuple[str, str]]:
    """Get mapping from short ID to label tuple."""
    result = {}
    for idx, (a, b) in enumerate(get_all_label_styles()):
        lid = f"L{idx + 1}"
        result[lid] = (a, b)
    return result


def get_simple_label_styles() -> set[str]:
    """Get set of simple label style strings."""
    return {f"{a}_{b}" for a, b in _get_simple_label_styles()}


def get_label_id(label_style: str) -> str:
    """Get short ID for a label style."""
    return get_label_style_ids().get(label_style, label_style[:10])


@dataclass
class CoherenceResult:
    """Result of a coherence computation."""

    n_coherent: int
    n_groups: int

    @property
    def pct(self) -> float | None:
        if self.n_groups == 0:
            return None
        return 100 * self.n_coherent / self.n_groups

    def __str__(self) -> str:
        if self.pct is None:
            return "N/A"
        return f"{self.n_coherent}/{self.n_groups} ({self.pct:.1f}%)"


@dataclass
class CoherenceBreakdown:
    """Coherence broken down by a grouping dimension."""

    dimension: str  # e.g., "label_style", "is_flipped"
    by_group: dict[str, CoherenceResult] = field(default_factory=dict)


@dataclass
class PreferenceAnalysis:
    """Complete analysis of a preference dataset."""

    dataset: "PreferenceDataset"

    # Basic stats
    n_total: int = 0
    n_short: int = 0
    n_long: int = 0
    n_rational: int = 0
    n_associated: int = 0

    # Overall coherence
    flip_coherence: CoherenceResult | None = None
    label_coherence: CoherenceResult | None = None
    unit_coherence: CoherenceResult | None = None
    spelling_coherence: CoherenceResult | None = None

    # Breakdowns: coherence of X, grouped by Y
    flip_by_label: CoherenceBreakdown | None = None
    flip_by_unit: CoherenceBreakdown | None = None
    flip_by_spelling: CoherenceBreakdown | None = None

    label_by_flip: CoherenceBreakdown | None = None
    label_by_unit: CoherenceBreakdown | None = None
    label_by_spelling: CoherenceBreakdown | None = None

    unit_by_flip: CoherenceBreakdown | None = None
    unit_by_label: CoherenceBreakdown | None = None
    unit_by_spelling: CoherenceBreakdown | None = None

    # Rationality breakdowns
    rationality_by_label: dict[str, tuple[int, int]] = field(default_factory=dict)
    rationality_by_flip: dict[str, tuple[int, int]] = field(default_factory=dict)
    rationality_by_unit: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Association breakdowns
    association_by_label: dict[str, tuple[int, int]] = field(default_factory=dict)
    association_by_flip: dict[str, tuple[int, int]] = field(default_factory=dict)
    association_by_unit: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Conditional breakdowns: (condition_dim, condition_val) -> {group_key: (n_good, n_total)}
    rationality_by_label_given: dict[tuple[str, str], dict[str, tuple[int, int]]] = (
        field(default_factory=dict)
    )
    association_by_label_given: dict[tuple[str, str], dict[str, tuple[int, int]]] = (
        field(default_factory=dict)
    )
    flip_coh_by_label_given: dict[tuple[str, str], dict[str, tuple[int, int]]] = field(
        default_factory=dict
    )
    label_coh_by_label_given: dict[tuple[str, str], dict[str, tuple[int, int]]] = field(
        default_factory=dict
    )
    unit_coh_by_label_given: dict[tuple[str, str], dict[str, tuple[int, int]]] = field(
        default_factory=dict
    )

    def print_summary(self) -> None:
        """Print the basic summary (rationality, association, overall coherence)."""
        log_banner(f"PreferenceDataset: {self.dataset.model_name}")
        log_kv("Samples", str(self.n_total))
        log_kv("Choices", f"short_term={self.n_short}, long_term={self.n_long}")
        log()

        # Rationality & Association
        log_sub_banner("RATIONALITY & ASSOCIATION")
        log()
        pct_rational = 100 * self.n_rational / self.n_total if self.n_total else 0
        pct_irrational = 100 - pct_rational
        n_irrational = self.n_total - self.n_rational

        log_kv("Rationality", "choice matches economic optimum given time horizon")
        log_kv(
            "  rational",
            f"{self.n_rational}/{self.n_total} ({pct_rational:.1f}%)",
            indent_str="",
        )
        log_kv(
            "  irrational",
            f"{n_irrational}/{self.n_total} ({pct_irrational:.1f}%)",
            indent_str="",
        )
        log()

        pct_associated = 100 * self.n_associated / self.n_total if self.n_total else 0
        pct_non_associated = 100 - pct_associated
        n_non_associated = self.n_total - self.n_associated

        log_kv("Association", "choice time is closest to horizon")
        log_kv(
            "  associated",
            f"{self.n_associated}/{self.n_total} ({pct_associated:.1f}%)",
            indent_str="",
        )
        log_kv(
            "  non-associated",
            f"{n_non_associated}/{self.n_total} ({pct_non_associated:.1f}%)",
            indent_str="",
        )
        log()

        # Overall coherence
        coherence_metrics = [
            ("flip", "same choice regardless of option order", self.flip_coherence),
            ("label", "same choice regardless of label style", self.label_coherence),
            (
                "unit",
                "same choice regardless of time unit display",
                self.unit_coherence,
            ),
            (
                "spelling",
                "same choice regardless of number format",
                self.spelling_coherence,
            ),
        ]

        has_coherence = any(
            m[2] is not None and m[2].pct is not None for m in coherence_metrics
        )
        if has_coherence:
            log_sub_banner(
                "COHERENCE (same answer for same content, different formatting)"
            )
            log()
            for name, desc, result in coherence_metrics:
                if result is not None and result.pct is not None:
                    warn = " ⚠️ LOW!" if result.pct < 90 else ""
                    log_kv(name, f"{result}{warn}")
            log()

        log_divider()

    def print_detailed(self) -> None:
        """Print detailed breakdowns."""
        self._print_all_tables()

    def _print_label_legend(self) -> None:
        """Print the label style legend as a table."""
        log_sub_banner("LABEL LEGEND")
        log()
        log("  ID │ Labels")
        log("  ───┼─────────────────────────────")
        for lid in sorted(get_label_styles_display().keys()):
            labels = get_label_styles_display()[lid]
            log(f"  {lid} │ {labels[0]} / {labels[1]}")
        log()

    def _fmt_pct(self, n: int, total: int, warn_below: float = 90) -> str:
        """Format a percentage with optional warning."""
        if total == 0:
            return "  -  "
        pct = 100 * n / total
        warn = "⚠" if pct < warn_below else " "
        return f"{pct:5.1f}%{warn}"

    def _coherence_to_tuple(
        self, breakdown: CoherenceBreakdown | None, key: str
    ) -> tuple[int, int]:
        """Get (n_good, n_total) from coherence breakdown."""
        if not breakdown or not breakdown.by_group or key not in breakdown.by_group:
            return (0, 0)
        r = breakdown.by_group[key]
        return (r.n_coherent, r.n_groups)

    def _print_table_by_label(self) -> None:
        """Print combined table by label style."""
        log_sub_banner("BY LABEL STYLE")
        log()

        # Header
        log("  Label │ Rational │ Associat │ Flip Coh │ Unit Coh")
        log("  ──────┼──────────┼──────────┼──────────┼──────────")

        for label_style in sorted(
            get_label_style_ids().keys(), key=lambda x: get_label_style_ids()[x]
        ):
            label_id = get_label_style_ids()[label_style]

            rat = self.rationality_by_label.get(label_style, (0, 0))
            assoc = self.association_by_label.get(label_style, (0, 0))
            flip = self._coherence_to_tuple(self.flip_by_label, label_style)
            unit = self._coherence_to_tuple(self.unit_by_label, label_style)

            # Skip if no data for this label
            if rat[1] == 0 and assoc[1] == 0:
                continue

            log(
                f"  {label_id:5} │ {self._fmt_pct(*rat)} │ {self._fmt_pct(*assoc)} │ {self._fmt_pct(*flip)} │ {self._fmt_pct(*unit)}"
            )

        log()

    def _print_table_by_order(self) -> None:
        """Print combined table by option order."""
        log_sub_banner("BY OPTION ORDER")
        log()

        log("  Order            │ Rational │ Associat │ Label Coh │ Unit Coh")
        log("  ─────────────────┼──────────┼──────────┼───────────┼──────────")

        for flip_val, display in [
            ("False", "short_term first"),
            ("True", "long_term first"),
        ]:
            rat = self.rationality_by_flip.get(flip_val, (0, 0))
            assoc = self.association_by_flip.get(flip_val, (0, 0))
            label_coh = self._coherence_to_tuple(self.label_by_flip, flip_val)
            unit_coh = self._coherence_to_tuple(self.unit_by_flip, flip_val)

            log(
                f"  {display:16} │ {self._fmt_pct(*rat)} │ {self._fmt_pct(*assoc)} │  {self._fmt_pct(*label_coh)} │ {self._fmt_pct(*unit_coh)}"
            )

        log()

    def _print_table_by_unit(self) -> None:
        """Print combined table by unit variation."""
        log_sub_banner("BY UNIT VARIATION")
        log()

        log("  Units            │ Rational │ Associat │ Flip Coh │ Label Coh")
        log("  ─────────────────┼──────────┼──────────┼──────────┼───────────")

        for unit_val, display in [
            ("False", "consistent units"),
            ("True", "varied units"),
        ]:
            rat = self.rationality_by_unit.get(unit_val, (0, 0))
            assoc = self.association_by_unit.get(unit_val, (0, 0))
            flip_coh = self._coherence_to_tuple(self.flip_by_unit, unit_val)
            label_coh = self._coherence_to_tuple(self.label_by_unit, unit_val)

            log(
                f"  {display:16} │ {self._fmt_pct(*rat)} │ {self._fmt_pct(*assoc)} │ {self._fmt_pct(*flip_coh)} │  {self._fmt_pct(*label_coh)}"
            )

        log()

    def _print_conditional_label_table(
        self, condition: str, cond_val: str, cond_display: str
    ) -> None:
        """Print label table conditioned on a specific value."""
        key = (condition, cond_val)

        rat_data = self.rationality_by_label_given.get(key, {})
        assoc_data = self.association_by_label_given.get(key, {})
        flip_data = self.flip_coh_by_label_given.get(key, {})
        unit_data = self.unit_coh_by_label_given.get(key, {})

        if not rat_data and not assoc_data:
            return

        log(f"  When {cond_display}:")

        # Show relevant coherence columns based on condition
        if condition == "has_time_unit_variation":
            # Can measure flip coherence, not unit coherence (already fixed)
            log("  Label │ Rational │ Associat │ Flip Coh")
            log("  ──────┼──────────┼──────────┼──────────")
            for label_style in sorted(
                get_label_style_ids().keys(), key=lambda x: get_label_style_ids()[x]
            ):
                label_id = get_label_style_ids()[label_style]
                rat = rat_data.get(label_style, (0, 0))
                assoc = assoc_data.get(label_style, (0, 0))
                flip = flip_data.get(label_style, (0, 0))
                if rat[1] == 0 and assoc[1] == 0:
                    continue
                log(
                    f"  {label_id:5} │ {self._fmt_pct(*rat)} │ {self._fmt_pct(*assoc)} │ {self._fmt_pct(*flip)}"
                )
        else:  # is_flipped
            # Can measure unit coherence, not flip coherence (already fixed)
            log("  Label │ Rational │ Associat │ Unit Coh")
            log("  ──────┼──────────┼──────────┼──────────")
            for label_style in sorted(
                get_label_style_ids().keys(), key=lambda x: get_label_style_ids()[x]
            ):
                label_id = get_label_style_ids()[label_style]
                rat = rat_data.get(label_style, (0, 0))
                assoc = assoc_data.get(label_style, (0, 0))
                unit = unit_data.get(label_style, (0, 0))
                if rat[1] == 0 and assoc[1] == 0:
                    continue
                log(
                    f"  {label_id:5} │ {self._fmt_pct(*rat)} │ {self._fmt_pct(*assoc)} │ {self._fmt_pct(*unit)}"
                )

        log()

    def _print_all_tables(self) -> None:
        """Print all breakdown tables."""
        self._print_label_legend()
        self._print_table_by_label()
        self._print_table_by_order()
        self._print_table_by_unit()

        # Conditional tables
        log_sub_banner("BY LABEL (conditioned)")
        log()
        self._print_conditional_label_table(
            "has_time_unit_variation", "False", "consistent units"
        )
        self._print_conditional_label_table(
            "has_time_unit_variation", "True", "varied units"
        )
        self._print_conditional_label_table("is_flipped", "False", "short_term first")
        self._print_conditional_label_table("is_flipped", "True", "long_term first")

        log_divider()

    def print_all(self) -> None:
        """Print full analysis."""
        self.print_summary()
        self.print_detailed()


def analyze_preferences(dataset: "PreferenceDataset") -> PreferenceAnalysis:
    """Analyze a preference dataset and return structured results.

    This function never raises exceptions - it returns partial results on error.
    """
    analysis = PreferenceAnalysis(dataset=dataset)

    try:
        prefs = dataset.preferences
        analysis.n_total = len(prefs)

        # Basic counts
        try:
            short, long = dataset.split_by_choice()
            analysis.n_short = len(short)
            analysis.n_long = len(long)
        except Exception:
            pass

        # Rationality and association
        try:
            analysis.n_rational = sum(1 for p in prefs if p.matches_rational is True)
            analysis.n_associated = sum(1 for p in prefs if p.matches_associated is True)
        except Exception:
            pass

        # Overall coherence
        try:
            flip_coh = dataset.get_flip_coherence()
            label_coh = dataset.get_label_coherence()
            unit_coh = dataset.get_unit_coherence()
            spelling_coh = dataset.get_spelling_coherence()

            if flip_coh[2] is not None:
                analysis.flip_coherence = CoherenceResult(flip_coh[0], flip_coh[1])
            if label_coh[2] is not None:
                analysis.label_coherence = CoherenceResult(label_coh[0], label_coh[1])
            if unit_coh[2] is not None:
                analysis.unit_coherence = CoherenceResult(unit_coh[0], unit_coh[1])
            if spelling_coh[2] is not None:
                analysis.spelling_coherence = CoherenceResult(spelling_coh[0], spelling_coh[1])
        except Exception:
            pass

        # Coherence breakdowns
        try:
            analysis.flip_by_label = _compute_coherence_breakdown(
                prefs,
                vary_dim="is_flipped",
                group_by="label_style",
                control_dims=["has_time_unit_variation", "has_spell_numbers"],
            )
            analysis.flip_by_unit = _compute_coherence_breakdown(
                prefs,
                vary_dim="is_flipped",
                group_by="has_time_unit_variation",
                control_dims=["label_style", "has_spell_numbers"],
            )
            analysis.flip_by_spelling = _compute_coherence_breakdown(
                prefs,
                vary_dim="is_flipped",
                group_by="has_spell_numbers",
                control_dims=["label_style", "has_time_unit_variation"],
            )

            analysis.label_by_flip = _compute_coherence_breakdown(
                prefs,
                vary_dim="label_style",
                group_by="is_flipped",
                control_dims=["has_time_unit_variation", "has_spell_numbers"],
            )
            analysis.label_by_unit = _compute_coherence_breakdown(
                prefs,
                vary_dim="label_style",
                group_by="has_time_unit_variation",
                control_dims=["is_flipped", "has_spell_numbers"],
            )

            analysis.unit_by_label = _compute_coherence_breakdown(
                prefs,
                vary_dim="has_time_unit_variation",
                group_by="label_style",
                control_dims=["is_flipped", "has_spell_numbers"],
            )
            analysis.unit_by_flip = _compute_coherence_breakdown(
                prefs,
                vary_dim="has_time_unit_variation",
                group_by="is_flipped",
                control_dims=["label_style", "has_spell_numbers"],
            )
        except Exception:
            pass

        # Rationality breakdowns
        try:
            analysis.rationality_by_label = _compute_metric_breakdown(
                prefs, "label_style", "matches_rational"
            )
            analysis.rationality_by_flip = _compute_metric_breakdown(
                prefs, "is_flipped", "matches_rational"
            )
            analysis.rationality_by_unit = _compute_metric_breakdown(
                prefs, "has_time_unit_variation", "matches_rational"
            )
        except Exception:
            pass

        # Association breakdowns
        try:
            analysis.association_by_label = _compute_metric_breakdown(
                prefs, "label_style", "matches_associated"
            )
            analysis.association_by_flip = _compute_metric_breakdown(
                prefs, "is_flipped", "matches_associated"
            )
            analysis.association_by_unit = _compute_metric_breakdown(
                prefs, "has_time_unit_variation", "matches_associated"
            )
        except Exception:
            pass

        # Conditional breakdowns: by label, conditioned on other dimensions
        try:
            for cond_dim, cond_vals in [
                ("has_time_unit_variation", ["False", "True"]),
                ("is_flipped", ["False", "True"]),
            ]:
                for cond_val in cond_vals:
                    key = (cond_dim, cond_val)
                    analysis.rationality_by_label_given[key] = _compute_metric_breakdown(
                        prefs, "label_style", "matches_rational", cond_dim, cond_val
                    )
                    analysis.association_by_label_given[key] = _compute_metric_breakdown(
                        prefs, "label_style", "matches_associated", cond_dim, cond_val
                    )
                    # Coherence breakdowns conditioned
                    analysis.flip_coh_by_label_given[key] = _compute_coherence_by_label_given(
                        prefs, "is_flipped", cond_dim, cond_val
                    )
                    analysis.label_coh_by_label_given[key] = _compute_coherence_by_label_given(
                        prefs, "label_style", cond_dim, cond_val
                    )
                    analysis.unit_coh_by_label_given[key] = _compute_coherence_by_label_given(
                        prefs, "has_time_unit_variation", cond_dim, cond_val
                    )
        except Exception:
            pass

    except Exception:
        pass

    return analysis


def _compute_coherence_breakdown(
    prefs: list,
    vary_dim: str,
    group_by: str,
    control_dims: list[str],
) -> CoherenceBreakdown:
    """Compute coherence of vary_dim, broken down by group_by, controlling for control_dims."""
    breakdown = CoherenceBreakdown(dimension=group_by)

    # Group samples by (content_key, group_by_value, control_values)
    # Then within each group, check if varying vary_dim gives same choice
    groups: dict[tuple, list] = defaultdict(list)

    for p in prefs:
        content_key = getattr(p, "content_key", None)
        if content_key is None:
            continue

        group_val = str(getattr(p, group_by, None))
        control_vals = tuple(str(getattr(p, d, None)) for d in control_dims)

        key = (content_key, group_val, control_vals)
        groups[key].append(p)

    # For each group_by value, compute coherence
    coherence_by_group: dict[str, list[bool]] = defaultdict(list)

    for (content_key, group_val, control_vals), samples in groups.items():
        # Get unique values of vary_dim in this group
        vary_vals = set(str(getattr(p, vary_dim, None)) for p in samples)
        if len(vary_vals) < 2:
            continue  # Need variation to measure coherence

        # Check if all samples made same choice
        choices = set(p.choice_idx for p in samples)
        is_coherent = len(choices) == 1

        coherence_by_group[group_val].append(is_coherent)

    # Aggregate
    for group_val, coherent_list in coherence_by_group.items():
        n_coherent = sum(coherent_list)
        n_groups = len(coherent_list)
        breakdown.by_group[group_val] = CoherenceResult(n_coherent, n_groups)

    return breakdown


def _compute_metric_breakdown(
    prefs: list,
    group_by: str,
    metric: str,
    condition_dim: str | None = None,
    condition_val: str | None = None,
) -> dict[str, tuple[int, int]]:
    """Compute a boolean metric rate broken down by a dimension.

    Args:
        prefs: List of preference samples
        group_by: Attribute to group by (e.g., "label_style", "is_flipped")
        metric: Boolean attribute to measure (e.g., "matches_rational", "matches_associated")
        condition_dim: Optional attribute to filter on
        condition_val: Required value of condition_dim (as string)
    """
    groups: dict[str, list] = defaultdict(list)

    for p in prefs:
        # Apply condition filter
        if condition_dim is not None:
            if str(getattr(p, condition_dim, None)) != condition_val:
                continue

        group_val = str(getattr(p, group_by, None))
        groups[group_val].append(p)

    result = {}
    for group_val, samples in groups.items():
        n_positive = sum(1 for p in samples if getattr(p, metric, None) is True)
        n_total = len(samples)
        result[group_val] = (n_positive, n_total)

    return result


def _compute_coherence_by_label_given(
    prefs: list,
    vary_dim: str,
    condition_dim: str,
    condition_val: str,
) -> dict[str, tuple[int, int]]:
    """Compute coherence of vary_dim, by label_style, given a condition.

    Args:
        prefs: List of preference samples
        vary_dim: Dimension being varied (e.g., "is_flipped")
        condition_dim: Dimension to filter on
        condition_val: Required value of condition_dim
    """
    # Filter prefs by condition
    filtered = [
        p for p in prefs if str(getattr(p, condition_dim, None)) == condition_val
    ]

    # Group by (content_key, label_style) and check coherence across vary_dim
    groups: dict[tuple, list] = defaultdict(list)

    for p in filtered:
        content_key = getattr(p, "content_key", None)
        if content_key is None:
            continue
        label_style = getattr(p, "label_style", None)
        key = (content_key, label_style)
        groups[key].append(p)

    # For each label_style, compute coherence
    coherence_by_label: dict[str, list[bool]] = defaultdict(list)

    for (content_key, label_style), samples in groups.items():
        # Need variation in vary_dim
        vary_vals = set(str(getattr(p, vary_dim, None)) for p in samples)
        if len(vary_vals) < 2:
            continue

        choices = set(p.choice_idx for p in samples)
        is_coherent = len(choices) == 1
        coherence_by_label[label_style].append(is_coherent)

    result = {}
    for label_style, coherent_list in coherence_by_label.items():
        n_coherent = sum(coherent_list)
        n_groups = len(coherent_list)
        result[label_style] = (n_coherent, n_groups)

    return result
