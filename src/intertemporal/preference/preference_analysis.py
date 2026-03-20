"""Preference dataset analysis with clean, focused output.

Analyzes choices by: rationality, association, reward, time, and horizon.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...common.logging import log

if TYPE_CHECKING:
    from .preference_dataset import PreferenceDataset
    from ..common.preference_types import PreferenceSample


# ═══════════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════════

WIDTH = 70


def _banner(title: str) -> None:
    log("═" * WIDTH)
    log(title)
    log("═" * WIDTH)


def _section(title: str) -> None:
    log("")
    log("─" * WIDTH)
    log(title)
    log("─" * WIDTH)


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "    -"
    return f"{100 * n / total:5.1f}%"


def _ratio(n: int, total: int) -> str:
    if total == 0:
        return "  -/-"
    return f"{n:3d}/{total:<3d}"


def _stat(n: int, total: int) -> str:
    return f"{_ratio(n, total)} ({_pct(n, total)})"


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis Result Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BucketStats:
    """Stats for a single bucket (e.g., horizon=1yr, reward=$100)."""

    n_short: int = 0
    n_long: int = 0
    n_rational: int = 0
    n_associated: int = 0

    @property
    def n_total(self) -> int:
        return self.n_short + self.n_long

    @property
    def short_pct(self) -> float:
        return 100 * self.n_short / self.n_total if self.n_total else 0

    @property
    def long_pct(self) -> float:
        return 100 * self.n_long / self.n_total if self.n_total else 0

    @property
    def rational_pct(self) -> float:
        return 100 * self.n_rational / self.n_total if self.n_total else 0

    @property
    def associated_pct(self) -> float:
        return 100 * self.n_associated / self.n_total if self.n_total else 0


@dataclass
class PreferenceAnalysis:
    """Complete preference analysis results."""

    model_name: str = ""
    n_total: int = 0

    # Overall stats
    overall: BucketStats = field(default_factory=BucketStats)

    # By horizon (in years)
    by_horizon: dict[float | None, BucketStats] = field(default_factory=dict)

    # By short-term reward
    by_short_reward: dict[float, BucketStats] = field(default_factory=dict)

    # By long-term reward
    by_long_reward: dict[float, BucketStats] = field(default_factory=dict)

    # By short-term time (years)
    by_short_time: dict[float, BucketStats] = field(default_factory=dict)

    # By long-term time (years)
    by_long_time: dict[float, BucketStats] = field(default_factory=dict)

    # By reward ratio (long/short)
    by_reward_ratio: dict[float, BucketStats] = field(default_factory=dict)

    def print_all(self) -> None:
        """Print the full analysis."""
        print_analysis(self)


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis Logic
# ═══════════════════════════════════════════════════════════════════════════════


def _get_horizon_years(sample: "PreferenceSample") -> float | None:
    """Extract horizon in years from sample."""
    if sample.time_horizon is None:
        return None
    if isinstance(sample.time_horizon, dict):
        # TimeValue stored as dict
        value = sample.time_horizon.get("value", 0)
        unit = sample.time_horizon.get("unit", "years")
        if unit == "days":
            return value / 365.25
        if unit == "weeks":
            return value / 52.18
        if unit == "months":
            return value / 12
        return value  # assume years
    return None


def _bucket_horizon(horizon: float | None) -> float | None:
    """Bucket horizon into meaningful groups."""
    if horizon is None:
        return None
    # Round to nice values: 0, 0.5, 1, 2, 5, 10, 20, 50, 100
    buckets = [0, 0.5, 1, 2, 5, 10, 20, 50, 100]
    for b in buckets:
        if horizon <= b + 0.01:
            return b
    return 100  # cap at 100


def _add_sample_to_stats(stats: BucketStats, sample: "PreferenceSample") -> None:
    """Add a sample's stats to a bucket."""
    if sample.choice_term == "short_term":
        stats.n_short += 1
    elif sample.choice_term == "long_term":
        stats.n_long += 1

    if sample.matches_rational is True:
        stats.n_rational += 1
    if sample.matches_associated is True:
        stats.n_associated += 1


def analyze_preferences(dataset: "PreferenceDataset") -> PreferenceAnalysis:
    """Analyze preference dataset and return structured results.

    Breaks down choices by:
    - Overall rationality and association
    - By time horizon (in years)
    - By reward amounts (short and long)
    - By delivery times (short and long)
    - By reward ratio
    """
    analysis = PreferenceAnalysis(model_name=dataset.model_name)
    prefs = dataset.preferences
    analysis.n_total = len(prefs)

    for sample in prefs:
        # Skip invalid samples
        if sample.choice_term not in ("short_term", "long_term"):
            continue

        # Overall
        _add_sample_to_stats(analysis.overall, sample)

        # By horizon
        horizon = _get_horizon_years(sample)
        bucket_h = _bucket_horizon(horizon)
        if bucket_h not in analysis.by_horizon:
            analysis.by_horizon[bucket_h] = BucketStats()
        _add_sample_to_stats(analysis.by_horizon[bucket_h], sample)

        # By rewards
        if sample.short_term_reward is not None:
            r = sample.short_term_reward
            if r not in analysis.by_short_reward:
                analysis.by_short_reward[r] = BucketStats()
            _add_sample_to_stats(analysis.by_short_reward[r], sample)

        if sample.long_term_reward is not None:
            r = sample.long_term_reward
            if r not in analysis.by_long_reward:
                analysis.by_long_reward[r] = BucketStats()
            _add_sample_to_stats(analysis.by_long_reward[r], sample)

        # By times
        if sample.short_term_time is not None:
            t = sample.short_term_time
            if t not in analysis.by_short_time:
                analysis.by_short_time[t] = BucketStats()
            _add_sample_to_stats(analysis.by_short_time[t], sample)

        if sample.long_term_time is not None:
            t = sample.long_term_time
            if t not in analysis.by_long_time:
                analysis.by_long_time[t] = BucketStats()
            _add_sample_to_stats(analysis.by_long_time[t], sample)

        # By reward ratio
        if sample.short_term_reward and sample.long_term_reward:
            ratio = round(sample.long_term_reward / sample.short_term_reward, 2)
            if ratio not in analysis.by_reward_ratio:
                analysis.by_reward_ratio[ratio] = BucketStats()
            _add_sample_to_stats(analysis.by_reward_ratio[ratio], sample)

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════


def _print_bucket_table(
    title: str,
    buckets: dict,
    key_label: str,
    key_format: str = "{}",
    sort_key=None,
) -> None:
    """Print a table of bucket stats."""
    if not buckets:
        return

    _section(title)
    log("")

    # Header
    log(f"  {key_label:>12} │  N   │ Short │ Long  │ Rational │ Associated")
    log(f"  {'─' * 12}─┼──────┼───────┼───────┼──────────┼───────────")

    # Sort keys
    keys = sorted(buckets.keys(), key=sort_key or (lambda x: (x is None, x or 0)))

    for key in keys:
        stats = buckets[key]
        if stats.n_total == 0:
            continue

        # Format key
        if key is None:
            key_str = "no horizon"
        else:
            key_str = key_format.format(key)

        log(
            f"  {key_str:>12} │ {stats.n_total:4d} │"
            f" {stats.short_pct:5.1f}% │ {stats.long_pct:5.1f}% │"
            f"   {stats.rational_pct:5.1f}% │    {stats.associated_pct:5.1f}%"
        )

    log("")


def print_analysis(analysis: PreferenceAnalysis) -> None:
    """Print the full analysis with clean formatting."""
    log("")
    _banner(f"PREFERENCE ANALYSIS: {analysis.model_name}")
    log("")

    # ─────────────────────────────────────────────────────────────────────────
    # Overall Summary
    # ─────────────────────────────────────────────────────────────────────────
    o = analysis.overall
    log(f"  Total samples: {analysis.n_total}")
    log("")
    log(f"  Choices:")
    log(f"    short_term: {_stat(o.n_short, o.n_total)}")
    log(f"    long_term:  {_stat(o.n_long, o.n_total)}")
    log("")
    log(f"  Rationality (choice matches economic optimum given horizon):")
    log(f"    rational:   {_stat(o.n_rational, o.n_total)}")
    log(f"    irrational: {_stat(o.n_total - o.n_rational, o.n_total)}")
    log("")
    log(f"  Association (choice time closest to horizon):")
    log(f"    associated:     {_stat(o.n_associated, o.n_total)}")
    log(f"    non-associated: {_stat(o.n_total - o.n_associated, o.n_total)}")

    # ─────────────────────────────────────────────────────────────────────────
    # By Horizon
    # ─────────────────────────────────────────────────────────────────────────
    _print_bucket_table(
        "BY TIME HORIZON (years)",
        analysis.by_horizon,
        "Horizon",
        "{:.1f}yr",
        sort_key=lambda x: (x is None, x or 0),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # By Reward
    # ─────────────────────────────────────────────────────────────────────────
    _print_bucket_table(
        "BY SHORT-TERM REWARD",
        analysis.by_short_reward,
        "Reward",
        "${:,.0f}",
    )

    _print_bucket_table(
        "BY LONG-TERM REWARD",
        analysis.by_long_reward,
        "Reward",
        "${:,.0f}",
    )

    _print_bucket_table(
        "BY REWARD RATIO (long/short)",
        analysis.by_reward_ratio,
        "Ratio",
        "{:.2f}x",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # By Time
    # ─────────────────────────────────────────────────────────────────────────
    _print_bucket_table(
        "BY SHORT-TERM DELIVERY TIME (years)",
        analysis.by_short_time,
        "Time",
        "{:.2f}yr",
    )

    _print_bucket_table(
        "BY LONG-TERM DELIVERY TIME (years)",
        analysis.by_long_time,
        "Time",
        "{:.2f}yr",
    )

    log("═" * WIDTH)
    log("")
