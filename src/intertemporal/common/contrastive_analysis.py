"""Contrastive pairs analysis with clean output.

Analyzes contrastive preference pairs by: horizon, rationality, confidence, content.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ...common.logging import log
from .contrastive_preferences import ContrastivePreferences


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
# Horizon Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _get_horizon_years(time_horizon: dict | None) -> float | None:
    """Extract horizon in years from time_horizon dict."""
    if time_horizon is None:
        return None
    if isinstance(time_horizon, dict):
        value = time_horizon.get("value")
        if value is None:
            return None
        unit = time_horizon.get("unit", "years")
        if unit == "days":
            return value / 365.25
        if unit == "weeks":
            return value / 52.18
        if unit == "months":
            return value / 12
        return value
    return None


def _has_horizon(time_horizon: dict | None) -> bool:
    """Check if time_horizon has an actual value."""
    return _get_horizon_years(time_horizon) is not None


def _format_horizon(h: float | None) -> str:
    """Format horizon for display."""
    if h is None:
        return "none"
    if h < 1:
        return f"{h * 12:.0f}mo"
    return f"{h:.0f}yr"


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ContrastivePairsAnalysis:
    """Analysis results for contrastive pairs."""

    n_pairs: int = 0

    # Horizon breakdown
    n_both_horizon: int = 0
    n_neither_horizon: int = 0
    n_same_horizon: int = 0
    n_different_horizon: int = 0

    # By horizon pair (short_horizon, long_horizon)
    by_horizon_pair: dict[tuple[float | None, float | None], int] = field(
        default_factory=dict
    )

    # Rationality
    n_both_rational: int = 0
    n_neither_rational: int = 0
    n_only_short_rational: int = 0
    n_only_long_rational: int = 0

    # Association
    n_both_associated: int = 0
    n_neither_associated: int = 0

    # Content variation
    n_same_rewards: int = 0
    n_same_times: int = 0
    n_same_labels: int = 0

    # Confidence
    confidence_buckets: dict[str, int] = field(default_factory=dict)
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    mean_confidence: float = 0.0

    # By reward
    by_short_reward: dict[float, int] = field(default_factory=dict)
    by_long_reward: dict[float, int] = field(default_factory=dict)
    by_reward_ratio: dict[float, int] = field(default_factory=dict)


def analyze_contrastive_pairs(
    pairs: list[ContrastivePreferences],
) -> ContrastivePairsAnalysis:
    """Analyze a list of contrastive preference pairs."""
    analysis = ContrastivePairsAnalysis()
    analysis.n_pairs = len(pairs)

    if not pairs:
        return analysis

    confidences = []

    for pair in pairs:
        # Horizon - use actual value extraction, not property checks
        h_short = _get_horizon_years(pair.short_term.time_horizon)
        h_long = _get_horizon_years(pair.long_term.time_horizon)
        has_short = h_short is not None
        has_long = h_long is not None

        if has_short and has_long:
            analysis.n_both_horizon += 1
            if h_short == h_long:
                analysis.n_same_horizon += 1
            else:
                analysis.n_different_horizon += 1
        if not has_short and not has_long:
            analysis.n_neither_horizon += 1
        key = (h_short, h_long)
        analysis.by_horizon_pair[key] = analysis.by_horizon_pair.get(key, 0) + 1

        # Rationality
        if pair.both_rational:
            analysis.n_both_rational += 1
        if pair.neither_rational:
            analysis.n_neither_rational += 1
        if pair.only_short_rational:
            analysis.n_only_short_rational += 1
        if pair.only_long_rational:
            analysis.n_only_long_rational += 1

        # Association
        if pair.both_associated:
            analysis.n_both_associated += 1
        if pair.neither_associated:
            analysis.n_neither_associated += 1

        # Content
        if pair.same_rewards:
            analysis.n_same_rewards += 1
        if pair.same_times:
            analysis.n_same_times += 1
        if pair.same_labels:
            analysis.n_same_labels += 1

        # Confidence
        conf = pair.min_choice_prob
        confidences.append(conf)

        # Bucket confidence
        if conf >= 0.9:
            bucket = "≥90%"
        elif conf >= 0.8:
            bucket = "80-90%"
        elif conf >= 0.7:
            bucket = "70-80%"
        elif conf >= 0.6:
            bucket = "60-70%"
        else:
            bucket = "<60%"
        analysis.confidence_buckets[bucket] = (
            analysis.confidence_buckets.get(bucket, 0) + 1
        )

        # By reward
        if pair.short_term.short_term_reward is not None:
            r = pair.short_term.short_term_reward
            analysis.by_short_reward[r] = analysis.by_short_reward.get(r, 0) + 1

        if pair.short_term.long_term_reward is not None:
            r = pair.short_term.long_term_reward
            analysis.by_long_reward[r] = analysis.by_long_reward.get(r, 0) + 1

        if pair.short_term.short_term_reward and pair.short_term.long_term_reward:
            ratio = round(
                pair.short_term.long_term_reward / pair.short_term.short_term_reward, 2
            )
            analysis.by_reward_ratio[ratio] = analysis.by_reward_ratio.get(ratio, 0) + 1

    # Confidence stats
    if confidences:
        analysis.min_confidence = min(confidences)
        analysis.max_confidence = max(confidences)
        analysis.mean_confidence = sum(confidences) / len(confidences)

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════


def print_contrastive_pairs(pairs: list[ContrastivePreferences]) -> None:
    """Print analysis of contrastive preference pairs."""
    analysis = analyze_contrastive_pairs(pairs)
    n = analysis.n_pairs

    log("")
    _banner("CONTRASTIVE PAIRS ANALYSIS")
    log("")
    log(f"  Total pairs: {n}")

    # ─────────────────────────────────────────────────────────────────────────
    # Confidence
    # ─────────────────────────────────────────────────────────────────────────
    _section("CONFIDENCE (min choice_prob per pair)")
    log("")
    log(f"  Range: {analysis.min_confidence:.1%} - {analysis.max_confidence:.1%}")
    log(f"  Mean:  {analysis.mean_confidence:.1%}")
    log("")
    log("  Distribution:")
    for bucket in ["≥90%", "80-90%", "70-80%", "60-70%", "<60%"]:
        count = analysis.confidence_buckets.get(bucket, 0)
        if count > 0:
            bar = "█" * (count * 40 // n) if n else ""
            log(f"    {bucket:>7}: {count:4d} ({100*count/n:5.1f}%) {bar}")

    # ─────────────────────────────────────────────────────────────────────────
    # Horizon
    # ─────────────────────────────────────────────────────────────────────────
    _section("HORIZON")
    log("")
    log(f"  Both have horizon:    {_stat(analysis.n_both_horizon, n)}")
    log(f"  Neither has horizon:  {_stat(analysis.n_neither_horizon, n)}")
    log(f"  Same horizon value:   {_stat(analysis.n_same_horizon, n)}")
    log(f"  Different horizons:   {_stat(analysis.n_different_horizon, n)}")

    # Horizon pair breakdown
    if analysis.by_horizon_pair:
        log("")
        log("  Horizon pairs (short_chooser → long_chooser):")
        log("")
        log("    Short H │ Long H  │ Count")
        log("    ────────┼─────────┼──────")

        sorted_pairs = sorted(
            analysis.by_horizon_pair.items(),
            key=lambda x: (x[0][0] is None, x[0][0] or 0, x[0][1] is None, x[0][1] or 0),
        )
        for (h_short, h_long), count in sorted_pairs:
            h_short_str = _format_horizon(h_short)
            h_long_str = _format_horizon(h_long)
            log(f"    {h_short_str:>7} │ {h_long_str:>7} │ {count:4d}")

    # ─────────────────────────────────────────────────────────────────────────
    # Rationality
    # ─────────────────────────────────────────────────────────────────────────
    _section("RATIONALITY")
    log("")
    log(f"  Both rational:        {_stat(analysis.n_both_rational, n)}")
    log(f"  Neither rational:     {_stat(analysis.n_neither_rational, n)}")
    log(f"  Only short rational:  {_stat(analysis.n_only_short_rational, n)}")
    log(f"  Only long rational:   {_stat(analysis.n_only_long_rational, n)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Association
    # ─────────────────────────────────────────────────────────────────────────
    _section("ASSOCIATION")
    log("")
    log(f"  Both associated:      {_stat(analysis.n_both_associated, n)}")
    log(f"  Neither associated:   {_stat(analysis.n_neither_associated, n)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Content Variation
    # ─────────────────────────────────────────────────────────────────────────
    _section("CONTENT VARIATION (what differs between pair members)")
    log("")
    log(f"  Same rewards:         {_stat(analysis.n_same_rewards, n)}")
    log(f"  Same delivery times:  {_stat(analysis.n_same_times, n)}")
    log(f"  Same labels:          {_stat(analysis.n_same_labels, n)}")

    # ─────────────────────────────────────────────────────────────────────────
    # By Reward
    # ─────────────────────────────────────────────────────────────────────────
    if analysis.by_reward_ratio:
        _section("BY REWARD RATIO (long/short)")
        log("")
        log("    Ratio │ Count")
        log("    ──────┼──────")
        for ratio in sorted(analysis.by_reward_ratio.keys()):
            count = analysis.by_reward_ratio[ratio]
            log(f"    {ratio:5.2f}x │ {count:4d}")

    log("")
    log("═" * WIDTH)
    log("")
