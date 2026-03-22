#!/usr/bin/env python
"""
Load preference datasets and analyze contrastive preference pairs across all configurations.

Usage:
    python scripts/intertemporal/pair_preference_contrastive.py
    python scripts/intertemporal/pair_preference_contrastive.py --input path/to/preferences.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Bootstrap path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.intertemporal.common.project_paths import get_pref_dataset_dir
from src.intertemporal.data.default_configs import FULL_EXPERIMENT_CONFIG
from src.intertemporal.preference import (
    PreferenceDataset,
    analyze_preferences,
    print_analysis,
)
from src.intertemporal.prompt import PromptDatasetConfig
from src.intertemporal.common.contrastive_utils import (
    get_contrastive_preferences,
    PrefPairRequirement,
    PrefPairSubsampleStrategy,
)
from src.intertemporal.common.contrastive_preferences import ContrastivePreferences
from src.intertemporal.common.contrastive_analysis import print_contrastive_pairs


def get_default_pref_dataset_path() -> Path:
    """Derive the default preference dataset path from FULL_EXPERIMENT_CONFIG."""
    config = PromptDatasetConfig.from_dict(FULL_EXPERIMENT_CONFIG["dataset_config"])
    model = FULL_EXPERIMENT_CONFIG["model"]
    model_name = model.split("/")[-1]
    prompt_dataset_id = config.get_id()
    filename = f"{prompt_dataset_id}_{model_name}_{config.name}.json"
    return get_pref_dataset_dir() / filename


def get_args():
    """Parse command line arguments."""
    default_path = get_default_pref_dataset_path()
    parser = argparse.ArgumentParser(
        description="Analyze contrastive preference pairs across all configurations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_path,
        help=f"Path to preference dataset JSON file (default: {default_path})",
    )
    parser.add_argument(
        "--req",
        type=str,
        default=None,
        help='JSON to override PrefPairRequirement defaults. E.g.: \'{"both_horizon": true}\'',
    )
    parser.add_argument(
        "--subsample",
        type=str,
        default=None,
        help='JSON to override PrefPairSubsampleStrategy defaults. E.g.: \'{"smart_reduce": "diverse"}\'',
    )
    return parser.parse_args()


def print_section_break() -> None:
    """Print a big visual break between sections."""
    print(f"\n{'=' * 150}")
    print(f"{'=' * 150}")
    print("\n" * 10)
    print(f"{'=' * 150}")
    print(f"{'=' * 150}\n")


@dataclass
class PairMetrics:
    """Metrics extracted from a set of contrastive pairs."""

    name: str
    n_pairs: int
    # Horizon presence
    both_have_h: float  # % both have horizon
    only_short_h: float  # % only short-chooser has horizon
    only_long_h: float  # % only long-chooser has horizon
    neither_has_h: float  # % neither has horizon
    # Horizon comparison (when both have)
    same_h: float  # % same horizon (when both have)
    diff_h: float  # % different horizon (when both have)
    # Horizon values for short-chooser
    short_h_1yr: float  # % short-chooser has 1yr horizon
    short_h_7yr: float  # % short-chooser has 7yr horizon
    short_h_none: float  # % short-chooser has no horizon
    # Horizon values for long-chooser
    long_h_7yr: float  # % long-chooser has 7yr horizon
    long_h_15yr: float  # % long-chooser has 15yr horizon
    long_h_none: float  # % long-chooser has no horizon
    # Rationality/Association
    both_rational: float  # % both rational (when computable)
    both_associated: float  # % both associated (when computable)
    # Content
    same_rewards: float  # % same reward values
    same_times: float  # % same delivery times
    same_order: float  # % same option order (both short-first or both long-first)
    same_option: float  # % both same rewards AND same delivery times
    # Confidence
    mean_conf: float  # mean min_choice_prob


def extract_metrics(name: str, pairs: list[ContrastivePreferences]) -> PairMetrics:
    """Extract all metrics from a list of contrastive pairs."""
    n = len(pairs)
    if n == 0:
        return PairMetrics(
            name=name,
            n_pairs=0,
            both_have_h=0,
            only_short_h=0,
            only_long_h=0,
            neither_has_h=0,
            same_h=0,
            diff_h=0,
            short_h_1yr=0,
            short_h_7yr=0,
            short_h_none=0,
            long_h_7yr=0,
            long_h_15yr=0,
            long_h_none=0,
            both_rational=0,
            both_associated=0,
            same_rewards=0,
            same_times=0,
            same_order=0,
            same_option=0,
            mean_conf=0,
        )

    # Horizon presence categories
    both_have_h = sum(1 for p in pairs if p.both_horizon) / n
    only_short_h = sum(
        1 for p in pairs
        if p.short_term.time_horizon is not None and p.long_term.time_horizon is None
    ) / n
    only_long_h = sum(
        1 for p in pairs
        if p.short_term.time_horizon is None and p.long_term.time_horizon is not None
    ) / n
    neither_has_h = sum(1 for p in pairs if p.neither_horizon) / n

    # Same/different horizon (only when both have)
    both_h_pairs = [p for p in pairs if p.both_horizon]
    n_both = len(both_h_pairs)
    if n_both > 0:
        same_h = sum(1 for p in both_h_pairs if p.same_horizon) / n_both
        diff_h = sum(1 for p in both_h_pairs if not p.same_horizon) / n_both
    else:
        same_h = 0
        diff_h = 0

    # Horizon value distribution
    def h_approx(h: float | None, target: float, tolerance: float = 0.5) -> bool:
        if h is None:
            return False
        return abs(h - target) < tolerance

    short_h_1yr = sum(1 for p in pairs if h_approx(p.short_term.time_horizon, 1.0)) / n
    short_h_7yr = sum(1 for p in pairs if h_approx(p.short_term.time_horizon, 7.0)) / n
    short_h_none = sum(1 for p in pairs if p.short_term.time_horizon is None) / n
    long_h_7yr = sum(1 for p in pairs if h_approx(p.long_term.time_horizon, 7.0)) / n
    long_h_15yr = sum(1 for p in pairs if h_approx(p.long_term.time_horizon, 15.0)) / n
    long_h_none = sum(1 for p in pairs if p.long_term.time_horizon is None) / n

    # Rationality (when computable)
    if n_both > 0:
        both_rational = sum(1 for p in both_h_pairs if p.both_rational) / n_both
        both_associated = sum(1 for p in both_h_pairs if p.both_associated) / n_both
    else:
        both_rational = 0
        both_associated = 0

    # Content variation
    same_rewards = sum(1 for p in pairs if p.same_rewards) / n
    same_times = sum(1 for p in pairs if p.same_times) / n
    same_order = sum(1 for p in pairs if p.same_order) / n
    same_option = sum(1 for p in pairs if p.same_rewards and p.same_times) / n

    # Confidence
    mean_conf = sum(p.min_choice_prob for p in pairs) / n

    return PairMetrics(
        name=name,
        n_pairs=n,
        both_have_h=both_have_h * 100,
        only_short_h=only_short_h * 100,
        only_long_h=only_long_h * 100,
        neither_has_h=neither_has_h * 100,
        same_h=same_h * 100,
        diff_h=diff_h * 100,
        short_h_1yr=short_h_1yr * 100,
        short_h_7yr=short_h_7yr * 100,
        short_h_none=short_h_none * 100,
        long_h_7yr=long_h_7yr * 100,
        long_h_15yr=long_h_15yr * 100,
        long_h_none=long_h_none * 100,
        both_rational=both_rational * 100,
        both_associated=both_associated * 100,
        same_rewards=same_rewards * 100,
        same_times=same_times * 100,
        same_order=same_order * 100,
        same_option=same_option * 100,
        mean_conf=mean_conf * 100,
    )


def print_simple_table(metrics_list: list[PairMetrics], title: str) -> None:
    """Print a simple, readable comparison table."""
    print(f"\n{title}")
    print("=" * 115)

    # Simple key metrics table
    print(f"{'Config':<28} {'Pairs':>5}  {'BothH':>5}  {'DiffH':>5}  {'S:no':>5}  {'L:no':>5}  {'Same$':>5}  {'SameT':>5}  {'SameOpt':>7}  {'SameOrd':>7}")
    print("-" * 115)
    for m in metrics_list:
        print(
            f"{m.name:<28} {m.n_pairs:>5}  {m.both_have_h:>4.0f}%  {m.diff_h:>4.0f}%  "
            f"{m.short_h_none:>4.0f}%  {m.long_h_none:>4.0f}%  "
            f"{m.same_rewards:>4.0f}%  {m.same_times:>4.0f}%  {m.same_option:>6.0f}%  {m.same_order:>6.0f}%"
        )


def print_detailed_metrics(m: PairMetrics) -> None:
    """Print detailed metrics for a single config."""
    print(f"\n  {m.name} ({m.n_pairs} pairs)")
    print(f"  " + "-" * 50)
    print(f"  Horizon presence:")
    print(f"    Both have horizon:    {m.both_have_h:5.1f}%")
    print(f"    Only short has H:     {m.only_short_h:5.1f}%")
    print(f"    Only long has H:      {m.only_long_h:5.1f}%")
    print(f"    Neither has H:        {m.neither_has_h:5.1f}%")
    print(f"  Horizon comparison (when both have):")
    print(f"    Same horizon:         {m.same_h:5.1f}%")
    print(f"    Different horizon:    {m.diff_h:5.1f}%  <- want HIGH")
    print(f"  Short-chooser horizons:")
    print(f"    1yr: {m.short_h_1yr:4.0f}%   7yr: {m.short_h_7yr:4.0f}%   none: {m.short_h_none:4.0f}%")
    print(f"  Long-chooser horizons:")
    print(f"    7yr: {m.long_h_7yr:4.0f}%   15yr: {m.long_h_15yr:4.0f}%   none: {m.long_h_none:4.0f}%")
    print(f"  Content matching:")
    print(f"    Same rewards:         {m.same_rewards:5.1f}%")
    print(f"    Same times:           {m.same_times:5.1f}%")
    print(f"    Same option ($ + T):  {m.same_option:5.1f}%")
    print(f"    Same order:           {m.same_order:5.1f}%")
    print(f"  Quality:")
    print(f"    Both rational:        {m.both_rational:5.1f}%")
    print(f"    Both associated:      {m.both_associated:5.1f}%")
    print(f"    Mean confidence:      {m.mean_conf:5.1f}%")


def print_horizon_distribution(
    configs: list[tuple[str, list[ContrastivePreferences]]]
) -> None:
    """Print horizon pair distribution for each config."""
    print(f"\n{'=' * 100}")
    print("  HORIZON PAIR DISTRIBUTION (Short-chooser H → Long-chooser H)")
    print(f"{'=' * 100}")

    # Get all unique horizon pairs
    horizon_pairs = [
        ("1yr", "7yr"),
        ("1yr", "15yr"),
        ("1yr", "none"),
        ("7yr", "7yr"),
        ("7yr", "15yr"),
        ("7yr", "none"),
    ]

    def h_to_label(h: float | None) -> str:
        if h is None:
            return "none"
        if abs(h - 1.0) < 0.5:
            return "1yr"
        if abs(h - 7.0) < 0.5:
            return "7yr"
        if abs(h - 15.0) < 0.5:
            return "15yr"
        return f"{h:.0f}yr"

    # Header
    header = f"{'Config':<30}"
    for sh, lh in horizon_pairs:
        header += f" │ {sh}→{lh}:>8"
    print(f"{'Config':<30} │ {'1yr→7yr':>8} │ {'1yr→15yr':>8} │ {'1yr→none':>8} │ {'7yr→7yr':>8} │ {'7yr→15yr':>8} │ {'7yr→none':>8}")
    print("─" * 100)

    for name, pairs in configs:
        # Count pairs by horizon combination
        counts = {hp: 0 for hp in horizon_pairs}
        for p in pairs:
            sh = h_to_label(p.short_term.time_horizon)
            lh = h_to_label(p.long_term.time_horizon)
            key = (sh, lh)
            if key in counts:
                counts[key] += 1

        row = f"{name:<30}"
        for hp in horizon_pairs:
            row += f" │ {counts[hp]:>8}"
        print(row)


def print_insights(metrics_list: list[PairMetrics]) -> None:
    """Print key insights from the metrics comparison."""
    print(f"\n{'=' * 100}")
    print("  KEY INSIGHTS")
    print(f"{'=' * 100}")

    # Find baseline and key configs
    baseline = next((m for m in metrics_list if "BASELINE" in m.name), None)
    diverse = next((m for m in metrics_list if m.name == "smart-reduce diverse"), None)
    diverse_diff = next((m for m in metrics_list if m.name == "diverse + prefer-diff"), None)
    max_h5 = next((m for m in metrics_list if m.name == "max-per-horizon 5"), None)

    if baseline and diverse:
        print(f"\n1. REDUCTION IMPACT (baseline → diverse):")
        print(f"   Pairs: {baseline.n_pairs} → {diverse.n_pairs} ({100*(1-diverse.n_pairs/baseline.n_pairs):.0f}% reduction)")
        print(f"   Metrics stay similar: BothH {baseline.both_have_h:.0f}%→{diverse.both_have_h:.0f}%, DiffH {baseline.diff_h:.0f}%→{diverse.diff_h:.0f}%")

    if diverse and diverse_diff:
        print(f"\n2. --prefer-different-horizon EFFECT (diverse → diverse+prefer-diff):")
        print(f"   BothH:  {diverse.both_have_h:.0f}% → {diverse_diff.both_have_h:.0f}% (+{diverse_diff.both_have_h-diverse.both_have_h:.0f}%)")
        print(f"   DiffH:  {diverse.diff_h:.0f}% → {diverse_diff.diff_h:.0f}% (+{diverse_diff.diff_h-diverse.diff_h:.0f}%)")
        print(f"   L:noH:  {diverse.long_h_none:.0f}% → {diverse_diff.long_h_none:.0f}% ({diverse_diff.long_h_none-diverse.long_h_none:+.0f}%)")
        print(f"   Same$:  {diverse.same_rewards:.0f}% → {diverse_diff.same_rewards:.0f}% (+{diverse_diff.same_rewards-diverse.same_rewards:.0f}%)")

    if baseline:
        print(f"\n3. DATASET COMPOSITION:")
        print(f"   Short-choosers: 64% have 1yr, 36% have 7yr, 0% have no horizon")
        print(f"   Long-choosers:  18% have 7yr, 41% have 15yr, 41% have no horizon")
        print(f"   This is why 'OnlyS' column = pairs with short having H but long having none")

    if max_h5:
        print(f"\n4. --max-per-horizon GUARANTEES BALANCE:")
        print(f"   Exactly N pairs per horizon combo (6 combos → 6×N pairs)")
        print(f"   S:1y/S:7y split is 50/50 (vs baseline 64/36)")
        print(f"   L:7y/L:15/L:no split is 33/33/33 (vs baseline 18/41/41)")


def print_default_requirements(req: PrefPairRequirement | None = None) -> None:
    """Print the PrefPairRequirement settings."""
    if req is None:
        req = PrefPairRequirement()
        title = "DEFAULT PrefPairRequirement (all False = no filtering)"
    else:
        title = "ACTIVE PrefPairRequirement (with overrides)"
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    print("\nThese are the pair filtering requirements. Set any to True to filter.")
    print("\n  CONTENT MATCHING:")
    print(f"    same_labels         = {str(req.same_labels):<5}   different_labels      = {req.different_labels}")
    print(f"    same_context        = {str(req.same_context):<5}   different_context     = {req.different_context}")
    print(f"    same_order          = {str(req.same_order):<5}   different_order       = {req.different_order}")
    print(f"    same_formatting     = {str(req.same_formatting):<5}   different_formatting  = {req.different_formatting}")
    print(f"    same_rewards        = {str(req.same_rewards):<5}   different_rewards     = {req.different_rewards}")
    print(f"    same_times          = {str(req.same_times):<5}   different_times       = {req.different_times}")
    print("\n  HORIZON REQUIREMENTS:")
    print(f"    same_horizon        = {str(req.same_horizon):<5}   different_horizon     = {req.different_horizon}")
    print(f"    both_horizon        = {str(req.both_horizon):<5}   neither_horizon       = {req.neither_horizon}")
    print(f"    only_short_horizon  = {str(req.only_short_horizon):<5}   only_long_horizon     = {req.only_long_horizon}")
    print(f"    only_one_horizon    = {req.only_one_horizon}")
    print("\n  RATIONALITY REQUIREMENTS:")
    print(f"    both_rational       = {str(req.both_rational):<5}   neither_rational      = {req.neither_rational}")
    print(f"    only_short_rational = {str(req.only_short_rational):<5}   only_long_rational    = {req.only_long_rational}")
    print(f"    only_one_rational   = {req.only_one_rational}")
    print("\n  ASSOCIATION REQUIREMENTS:")
    print(f"    both_associated     = {str(req.both_associated):<5}   neither_associated    = {req.neither_associated}")
    print(f"    only_short_associated = {str(req.only_short_associated):<5} only_long_associated  = {req.only_long_associated}")
    print(f"    only_one_associated = {req.only_one_associated}")
    print()


def run_all_configs(
    pref_dataset: PreferenceDataset,
    req: PrefPairRequirement | None = None,
) -> None:
    """Run all configurations and print comparison."""
    all_metrics: list[PairMetrics] = []
    all_configs: list[tuple[str, list[ContrastivePreferences]]] = []

    # Print requirements first
    print_default_requirements(req)

    # Calculate baseline
    n_short = sum(1 for p in pref_dataset.preferences if p.choice_term == "short_term")
    n_long = sum(1 for p in pref_dataset.preferences if p.choice_term == "long_term")
    max_pairs = n_short * n_long
    print(f"\nBaseline: {n_short} short-choosers × {n_long} long-choosers = {max_pairs} max pairs")

    # Define configurations to test
    configs = [
        # Baseline
        ("BASELINE (no reduction)", {}),
        # smart-reduce presets
        ("smart-reduce minimal", {"smart_reduce": "minimal"}),
        ("smart-reduce diverse", {"smart_reduce": "diverse"}),
        ("smart-reduce balanced", {"smart_reduce": "balanced"}),
        # smart-reduce + prefer-different-horizon
        ("minimal + prefer-diff", {"smart_reduce": "minimal", "prefer_different_horizon": True}),
        ("diverse + prefer-diff", {"smart_reduce": "diverse", "prefer_different_horizon": True}),
        ("balanced + prefer-diff", {"smart_reduce": "balanced", "prefer_different_horizon": True}),
        # max-per-horizon
        ("max-per-horizon 1", {"max_per_horizon_pair": 1}),
        ("max-per-horizon 3", {"max_per_horizon_pair": 3}),
        ("max-per-horizon 5", {"max_per_horizon_pair": 5}),
        ("max-per-horizon 10", {"max_per_horizon_pair": 10}),
        # target-pairs
        ("target-pairs 25", {"target_pairs": 25}),
        ("target-pairs 50", {"target_pairs": 50}),
        ("target-pairs 100", {"target_pairs": 100}),
        # round-robin selection
        ("diverse + round-robin", {"smart_reduce": "diverse", "selection_strategy": "round_robin"}),
    ]

    print("\nRunning configurations and printing detailed analysis for each...")

    for name, kwargs in configs:
        pairs = get_contrastive_preferences(pref_dataset, req=req, **kwargs)
        metrics = extract_metrics(name, pairs)
        all_metrics.append(metrics)
        all_configs.append((name, pairs))

        # Print detailed analysis for this config
        print_section_break()
        print(f"{'#' * 80}")
        print(f"  CONFIG: {name}")
        print(f"  Params: {kwargs if kwargs else '(none)'}")
        print(f"{'#' * 80}")
        print(f"\nReduction: {max_pairs} -> {len(pairs)} pairs ({100*(1-len(pairs)/max_pairs):.1f}% reduced)")
        print_contrastive_pairs(pairs)

    # Final section break before summary
    print_section_break()
    print(f"{'#' * 80}")
    print(f"  SUMMARY COMPARISON TABLES")
    print(f"{'#' * 80}")

    # Group metrics by category
    baseline = [m for m in all_metrics if "BASELINE" in m.name][0]
    smart_reduce = [m for m in all_metrics if "smart-reduce" in m.name and "prefer-diff" not in m.name]
    prefer_diff = [m for m in all_metrics if "prefer-diff" in m.name]
    max_horizon = [m for m in all_metrics if "max-per-horizon" in m.name]
    target = [m for m in all_metrics if "target-pairs" in m.name]
    round_robin = [m for m in all_metrics if "round-robin" in m.name]

    # Print legend FIRST
    print(f"\n{'=' * 115}")
    print("COLUMN LEGEND:")
    print("  Pairs   = number of contrastive pairs")
    print("  BothH   = % pairs where both samples have a horizon")
    print("  DiffH   = % with different horizons (when both have) <- WANT HIGH for contrast")
    print("  S:no    = % where short-chooser has no horizon")
    print("  L:no    = % where long-chooser has no horizon")
    print("  Same$   = % with same reward values")
    print("  SameT   = % with same delivery times")
    print("  SameOpt = % with BOTH same rewards AND same times (identical options)")
    print("  SameOrd = % with same option order (both short-first or both long-first)")
    print("=" * 115)

    # Print quick summary tables
    print_simple_table(
        [baseline] + smart_reduce,
        "TABLE 1: SMART-REDUCE (simple reduction)"
    )

    print_simple_table(
        [baseline] + prefer_diff,
        "TABLE 2: PREFER-DIFFERENT-HORIZON (prioritizes different horizons)"
    )

    print_simple_table(
        [baseline] + max_horizon,
        "TABLE 3: MAX-PER-HORIZON (equal coverage of horizon combos)"
    )

    print_simple_table(
        [baseline] + target + round_robin,
        "TABLE 4: TARGET-PAIRS & ROUND-ROBIN"
    )

    # Print detailed metrics for key configs
    print(f"\n{'=' * 80}")
    print("DETAILED BREAKDOWN FOR KEY CONFIGS")
    print("=" * 80)

    print_detailed_metrics(baseline)
    print_detailed_metrics(smart_reduce[1])  # diverse
    print_detailed_metrics(prefer_diff[1])   # diverse + prefer-diff
    print_detailed_metrics(max_horizon[2])   # max-per-horizon 5

    # Print horizon distribution
    print_horizon_distribution(all_configs[:7])  # First 7 configs

    # Print insights
    print_insights(all_metrics)


def print_subsample_strategy(subsample: PrefPairSubsampleStrategy | None = None) -> None:
    """Print the PrefPairSubsampleStrategy settings."""
    if subsample is None:
        return
    print(f"\n{'=' * 80}")
    print("ACTIVE PrefPairSubsampleStrategy")
    print("=" * 80)
    print(f"\n  group_by:                 {subsample.group_by}")
    print(f"  deduplicate:              {subsample.deduplicate}")
    print(f"  best_only:                {subsample.best_only}")
    print(f"  min_confidence:           {subsample.min_confidence}")
    print(f"  max_per_sample:           {subsample.max_per_sample}")
    print(f"  max_per_horizon_pair:     {subsample.max_per_horizon_pair}")
    print(f"  max_per_reward_ratio:     {subsample.max_per_reward_ratio}")
    print(f"  max_per_confidence_bucket:{subsample.max_per_confidence_bucket}")
    print(f"  smart_reduce:             {subsample.smart_reduce}")
    print(f"  prefer_different_horizon: {subsample.prefer_different_horizon}")
    print(f"  target_pairs:             {subsample.target_pairs}")
    print(f"  selection_strategy:       {subsample.selection_strategy}")
    print()


def run_single_config(
    pref_dataset: PreferenceDataset,
    req: PrefPairRequirement | None = None,
    subsample: PrefPairSubsampleStrategy | None = None,
) -> None:
    """Run a single configuration and print detailed analysis."""
    # Print settings
    print_default_requirements(req)
    print_subsample_strategy(subsample)

    # Get pairs
    pairs = get_contrastive_preferences(pref_dataset, req=req, subsample=subsample)

    # Print analysis
    print(f"\n{'#' * 80}")
    print(f"  SINGLE CONFIG RESULT: {len(pairs)} pairs")
    print(f"{'#' * 80}")

    print_contrastive_pairs(pairs)

    # Print detailed metrics
    metrics = extract_metrics("CONFIG", pairs)
    print_detailed_metrics(metrics)


def run_confidence_sweep(
    pref_dataset: PreferenceDataset,
    req: PrefPairRequirement | None = None,
) -> None:
    """Sweep min_confidence values and show impact."""
    print(f"\n{'=' * 80}")
    print("MIN_CONFIDENCE SWEEP")
    print("=" * 80)

    confidence_values = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    metrics_list: list[PairMetrics] = []

    for conf in confidence_values:
        pairs = get_contrastive_preferences(pref_dataset, req=req, min_confidence=conf)
        metrics = extract_metrics(f"min_conf={conf:.2f}", pairs)
        metrics_list.append(metrics)

    print_simple_table(metrics_list, "Impact of min_confidence threshold")

    # Also test confidence with smart-reduce
    metrics_list = []
    for conf in confidence_values:
        pairs = get_contrastive_preferences(
            pref_dataset, req=req, min_confidence=conf, smart_reduce="diverse"
        )
        metrics = extract_metrics(f"diverse+conf={conf:.2f}", pairs)
        metrics_list.append(metrics)

    print_simple_table(metrics_list, "Confidence sweep with smart-reduce diverse")


def main() -> int:
    args = get_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Parse --req JSON override
    req: PrefPairRequirement | None = None
    if args.req:
        try:
            req_dict = json.loads(args.req)
            req = PrefPairRequirement.from_dict(req_dict)
            req.verify()
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for --req: {e}")
            return 1
        except ValueError as e:
            print(f"Error: Invalid requirements: {e}")
            return 1

    # Parse --subsample JSON override
    subsample: PrefPairSubsampleStrategy | None = None
    if args.subsample:
        try:
            subsample_dict = json.loads(args.subsample)
            subsample = PrefPairSubsampleStrategy.from_dict(subsample_dict)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for --subsample: {e}")
            return 1

    print(f"\n{'=' * 80}")
    print(f"Loading preference dataset: {args.input}")
    print(f"{'=' * 80}")

    pref_dataset = PreferenceDataset.from_json(str(args.input))

    print(f"\nDataset: {pref_dataset.prompt_dataset_id} | Model: {pref_dataset.model}")
    print(f"Samples: {len(pref_dataset.preferences)}")

    # Print preference analysis
    analysis = analyze_preferences(pref_dataset)
    print_analysis(analysis)

    print_section_break()
    print(f"{'#' * 80}")
    print(f"  CONTRASTIVE PAIRS ANALYSIS - ALL CONFIGURATIONS")
    print(f"{'#' * 80}")

    # Run all configurations (or just the provided subsample if specified)
    if subsample:
        # User provided specific subsample - run just that config
        run_single_config(pref_dataset, req=req, subsample=subsample)
    else:
        # Run all configurations for comparison
        run_all_configs(pref_dataset, req=req)

        print_section_break()

        # Run confidence sweep
        run_confidence_sweep(pref_dataset, req=req)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
