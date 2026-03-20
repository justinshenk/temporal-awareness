"""Contrastive pairs analysis with clean output.

Analyzes contrastive preference pairs by: horizon, rationality, confidence, content.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common.logging import log
from ...common.time_value import parse_horizon_years
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


def _get_horizon_years(time_horizon) -> float | None:
    """Extract horizon in years from time_horizon."""
    return parse_horizon_years(time_horizon)


def _bin_horizon(h: float | None) -> float | None:
    """Bin horizon to max 1-year granularity."""
    if h is None:
        return None
    if h < 1:
        # Monthly bins for < 1 year: 1mo, 2mo, 3mo, 6mo
        months = h * 12
        if months <= 1:
            return 1 / 12
        if months <= 2:
            return 2 / 12
        if months <= 3:
            return 3 / 12
        if months <= 6:
            return 6 / 12
        return 1.0
    # Yearly bins for >= 1 year
    return float(int(h))


def _format_horizon(h: float | None) -> str:
    """Format horizon for display."""
    if h is None:
        return "none"
    if h < 1:
        return f"{h * 12:.0f}mo"
    return f"{h:.0f}yr"


def _format_rational(val: bool | None) -> str:
    """Format rationality value."""
    if val is None:
        return "n/a"
    return "yes" if val else "no"


def _format_prob(val: float) -> str:
    """Format probability as percentage."""
    return f"{val:.1%}"


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
    n_only_short_horizon: int = 0
    n_only_long_horizon: int = 0
    n_same_horizon: int = 0
    n_different_horizon: int = 0

    # By horizon pair (short_horizon, long_horizon)
    by_horizon_pair: dict[tuple[float | None, float | None], int] = field(
        default_factory=dict
    )

    # Rationality (only counted when both values are not None)
    n_rationality_computable: int = 0
    n_both_rational: int = 0
    n_neither_rational: int = 0
    n_only_short_rational: int = 0
    n_only_long_rational: int = 0

    # Association (only counted when both values are not None)
    n_association_computable: int = 0
    n_both_associated: int = 0
    n_neither_associated: int = 0
    n_only_short_associated: int = 0
    n_only_long_associated: int = 0

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
        # Horizon
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
        elif has_short and not has_long:
            analysis.n_only_short_horizon += 1
        elif not has_short and has_long:
            analysis.n_only_long_horizon += 1
        else:
            analysis.n_neither_horizon += 1

        key = (_bin_horizon(h_short), _bin_horizon(h_long))
        analysis.by_horizon_pair[key] = analysis.by_horizon_pair.get(key, 0) + 1

        # Rationality - only count when both are computable (not None)
        r_short = pair.short_term.matches_rational
        r_long = pair.long_term.matches_rational
        if r_short is not None and r_long is not None:
            analysis.n_rationality_computable += 1
            if r_short and r_long:
                analysis.n_both_rational += 1
            elif not r_short and not r_long:
                analysis.n_neither_rational += 1
            elif r_short and not r_long:
                analysis.n_only_short_rational += 1
            else:
                analysis.n_only_long_rational += 1

        # Association - only count when both are computable
        a_short = pair.short_term.matches_associated
        a_long = pair.long_term.matches_associated
        if a_short is not None and a_long is not None:
            analysis.n_association_computable += 1
            if a_short and a_long:
                analysis.n_both_associated += 1
            elif not a_short and not a_long:
                analysis.n_neither_associated += 1
            elif a_short and not a_long:
                analysis.n_only_short_associated += 1
            else:
                analysis.n_only_long_associated += 1

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

        # By reward ratio
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


def _print_pair_details(pair: ContrastivePreferences, idx: int) -> None:
    """Print details of a single contrastive pair."""
    short = pair.short_term
    long = pair.long_term

    h_short = _get_horizon_years(short.time_horizon)
    h_long = _get_horizon_years(long.time_horizon)

    # Summary line: horizon difference -> choice difference
    h_short_str = _format_horizon(h_short)
    h_long_str = _format_horizon(h_long)
    log(f"  Pair {idx + 1}: horizon {h_short_str}→short vs {h_long_str}→long")

    log(f"    Short chooser: #{short.sample_idx} prob={_format_prob(short.choice_prob)} rational={_format_rational(short.matches_rational)}")
    log(f"    Long chooser:  #{long.sample_idx} prob={_format_prob(long.choice_prob)} rational={_format_rational(long.matches_rational)}")

    if short.short_term_reward and short.long_term_reward:
        ratio = short.long_term_reward / short.short_term_reward
        log(f"    Rewards: ${short.short_term_reward:,.0f} vs ${short.long_term_reward:,.0f} ({ratio:.1f}x)")

    # Show what varies between the pair
    diff = []
    if not pair.same_rewards:
        diff.append("rewards")
    if not pair.same_times:
        diff.append("times")
    if not pair.same_labels:
        diff.append("labels")

    if diff:
        log(f"    Differs: {', '.join(diff)}")


def print_contrastive_pairs(pairs: list[ContrastivePreferences]) -> None:
    """Print analysis of contrastive preference pairs."""
    n = len(pairs)

    log("")
    _banner("CONTRASTIVE PAIRS ANALYSIS")
    log("")
    log(f"  Total pairs: {n}")

    if n == 0:
        log("")
        log("  No pairs to analyze.")
        log("")
        return

    # For small numbers of pairs, just show individual details (aggregate stats are redundant)
    if n <= 5:
        log("")
        for i, pair in enumerate(pairs):
            _print_pair_details(pair, i)
            if i < n - 1:
                log("")
        log("")
        return

    # For larger datasets, show aggregate analysis
    analysis = analyze_contrastive_pairs(pairs)

    # ─────────────────────────────────────────────────────────────────────────
    # Confidence Summary
    # ─────────────────────────────────────────────────────────────────────────
    _section("CONFIDENCE")
    log("")
    if n == 1:
        log(f"  Min choice prob: {analysis.mean_confidence:.1%}")
    else:
        log(f"  Range: {analysis.min_confidence:.1%} - {analysis.max_confidence:.1%}")
        log(f"  Mean:  {analysis.mean_confidence:.1%}")
        log("")
        log("  Distribution:")
        for bucket in ["≥90%", "80-90%", "70-80%", "60-70%", "<60%"]:
            count = analysis.confidence_buckets.get(bucket, 0)
            if count > 0:
                bar = "█" * (count * 40 // n)
                log(f"    {bucket:>7}: {count:4d} ({100*count/n:5.1f}%) {bar}")

    # ─────────────────────────────────────────────────────────────────────────
    # Horizon Summary
    # ─────────────────────────────────────────────────────────────────────────
    _section("HORIZON")
    log("")
    log(f"  Both have horizon:      {_stat(analysis.n_both_horizon, n)}")
    log(f"  Only short has horizon: {_stat(analysis.n_only_short_horizon, n)}")
    log(f"  Only long has horizon:  {_stat(analysis.n_only_long_horizon, n)}")
    log(f"  Neither has horizon:    {_stat(analysis.n_neither_horizon, n)}")

    if analysis.n_both_horizon > 0:
        log("")
        log(f"  When both have horizon:")
        log(f"    Same value:     {_stat(analysis.n_same_horizon, analysis.n_both_horizon)}")
        log(f"    Different:      {_stat(analysis.n_different_horizon, analysis.n_both_horizon)}")

    # Horizon pair breakdown (only show if more than 1 unique pair and n > 5)
    if len(analysis.by_horizon_pair) > 1 and n > 5:
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
    # Rationality Summary
    # ─────────────────────────────────────────────────────────────────────────
    _section("RATIONALITY")
    log("")
    n_rat = analysis.n_rationality_computable
    n_not_computable = n - n_rat
    if n_not_computable > 0:
        log(f"  Computable (both have horizon): {_stat(n_rat, n)}")
    if n_rat > 0:
        log(f"  Both rational:      {_stat(analysis.n_both_rational, n_rat)}")
        log(f"  Neither rational:   {_stat(analysis.n_neither_rational, n_rat)}")
        log(f"  Only short rational:{_stat(analysis.n_only_short_rational, n_rat)}")
        log(f"  Only long rational: {_stat(analysis.n_only_long_rational, n_rat)}")
    elif n_not_computable == n:
        log("  Not computable (requires both samples to have horizon)")

    # ─────────────────────────────────────────────────────────────────────────
    # Association Summary
    # ─────────────────────────────────────────────────────────────────────────
    _section("ASSOCIATION")
    log("")
    n_assoc = analysis.n_association_computable
    n_not_computable = n - n_assoc
    if n_not_computable > 0:
        log(f"  Computable (both have horizon): {_stat(n_assoc, n)}")
    if n_assoc > 0:
        log(f"  Both associated:      {_stat(analysis.n_both_associated, n_assoc)}")
        log(f"  Neither associated:   {_stat(analysis.n_neither_associated, n_assoc)}")
        log(f"  Only short associated:{_stat(analysis.n_only_short_associated, n_assoc)}")
        log(f"  Only long associated: {_stat(analysis.n_only_long_associated, n_assoc)}")
    elif n_not_computable == n:
        log("  Not computable (requires both samples to have horizon)")

    # ─────────────────────────────────────────────────────────────────────────
    # Content Variation
    # ─────────────────────────────────────────────────────────────────────────
    _section("CONTENT VARIATION")
    log("")
    log(f"  Same rewards:        {_stat(analysis.n_same_rewards, n)}")
    log(f"  Same delivery times: {_stat(analysis.n_same_times, n)}")
    log(f"  Same labels:         {_stat(analysis.n_same_labels, n)}")

    # ─────────────────────────────────────────────────────────────────────────
    # By Reward Ratio (only if multiple ratios)
    # ─────────────────────────────────────────────────────────────────────────
    if len(analysis.by_reward_ratio) > 1:
        _section("BY REWARD RATIO")
        log("")
        log("    Ratio │ Count")
        log("    ──────┼──────")
        for ratio in sorted(analysis.by_reward_ratio.keys()):
            count = analysis.by_reward_ratio[ratio]
            log(f"    {ratio:5.2f}x │ {count:4d}")

    log("")
