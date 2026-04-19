#!/usr/bin/env python
"""
Behavioral coherence analysis for intertemporal preference prompts.

Queries multiple models with all prompt variations, parses choices,
saves results, and generates comparative visualizations.

Usage:
    python scripts/intertemporal/coherent_behavior.py /path/to/config.json
    python scripts/intertemporal/coherent_behavior.py data/intertemporal/investment/investment_behave.json
    python scripts/intertemporal/coherent_behavior.py config.json --models "Qwen/Qwen3-4B" "anthropic:claude-haiku-4-5-20251001"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Bootstrap path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.intertemporal.data.default_configs import DEFAULT_MODEL
from src.intertemporal.prompt import PromptDatasetConfig, PromptDatasetGenerator

QWEN25_MODEL = "Qwen/Qwen2.5-3B-Instruct"
QWEN3_MODEL = "Qwen/Qwen3-4B"
CLAUDE_MODEL = "anthropic:claude-haiku-4-5-20251001"

ALL_MODELS = [QWEN25_MODEL, QWEN3_MODEL, DEFAULT_MODEL, CLAUDE_MODEL]

# Consistent color palette for up to 6 models
MODEL_PALETTE = [
    "#4C72B0",  # blue
    "#55A868",  # green
    "#DD8452",  # orange
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def short_model_name(model: str) -> str:
    """Human-friendly short name for display."""
    if model.startswith("anthropic:"):
        return model[10:]
    return model.split("/")[-1] if "/" in model else model


def _horizon_label(h_months: float | None) -> str:
    if h_months is None:
        return "None"
    if h_months < 12:
        return f"{h_months:.0f}mo"
    years = h_months / 12
    if years == int(years):
        return f"{int(years)}y"
    return f"{years:.1f}y"


def _save_fig(fig, output_dir: Path, name: str):
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------


def query_model(model_name: str, samples, choice_prefix: str) -> list[dict]:
    """Query a model with all samples, return list of {response, choice}."""
    from src.inference import ModelRunner

    runner = ModelRunner(model_name)
    # Use skip_thinking_prefix as prefilling for reasoning models
    # (same approach as PreferenceQuerier)
    prefilling = runner.skip_thinking_prefix
    results = []
    for i, sample in enumerate(samples):
        print(f"  [{short_model_name(model_name)}] sample {i + 1}/{len(samples)}")
        response = runner.generate(
            sample.text,
            max_new_tokens=256,
            temperature=0.0,
            prefilling=prefilling,
        )
        choice = parse_choice(response, sample, choice_prefix)
        results.append({"response": response, "choice": choice})
    return results


def parse_choice(response: str, sample, choice_prefix: str) -> str | None:
    """Parse model response to determine short_term or long_term choice.

    Returns 'short_term', 'long_term', or None if unparseable.
    """
    pair = sample.prompt.preference_pair
    short_label = pair.short_term.label.strip().rstrip(".")
    long_label = pair.long_term.label.strip().rstrip(".")

    # Strip <think>...</think> blocks from reasoning models
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Strip markdown bold/italic for matching
    clean_response = re.sub(r"\*+", "", response)

    # Strategy 1: Look for "I choose: <label>" pattern
    prefix_pattern = re.escape(choice_prefix.strip())
    match = re.search(prefix_pattern + r"\s*(.+?)[\.\n]", clean_response, re.IGNORECASE)
    if match:
        chosen_text = match.group(1).strip().rstrip(".")
        if _label_match(chosen_text, short_label):
            return "short_term"
        if _label_match(chosen_text, long_label):
            return "long_term"

    # Strategy 2: Check first line / first few words for label
    first_line = clean_response.strip().split("\n")[0].strip()
    if _label_match(first_line, short_label):
        return "short_term"
    if _label_match(first_line, long_label):
        return "long_term"

    # Strategy 3: Search anywhere in response for labels
    response_lower = response.lower()
    short_found = short_label.lower() in response_lower
    long_found = long_label.lower() in response_lower
    if short_found and not long_found:
        return "short_term"
    if long_found and not short_found:
        return "long_term"

    # Strategy 4: Look for "short" or "long" term mentions
    if "short-term" in response_lower or "short term" in response_lower:
        if "long-term" not in response_lower and "long term" not in response_lower:
            return "short_term"
    if "long-term" in response_lower or "long term" in response_lower:
        if "short-term" not in response_lower and "short term" not in response_lower:
            return "long_term"

    return None


def _label_match(text: str, label: str) -> bool:
    """Check if text matches or starts with a label."""
    text_clean = text.strip().lower().rstrip(".)")
    label_clean = label.strip().lower().rstrip(".)")
    return text_clean.startswith(label_clean) or label_clean.startswith(text_clean)


# ---------------------------------------------------------------------------
# Result building
# ---------------------------------------------------------------------------


def build_results(
    samples, model_names: list[str], all_model_results: dict[str, list[dict]]
) -> list[dict]:
    """Build the combined results list with per-model responses."""
    results = []
    for i, sample in enumerate(samples):
        pair = sample.prompt.preference_pair
        horizon = sample.prompt.time_horizon
        entry = {
            "sample_idx": sample.sample_idx,
            "prompt": sample.text,
            "time_horizon": horizon.to_dict() if horizon else None,
            "time_horizon_months": horizon.to_months() if horizon else None,
            "formatting_id": sample.formatting_id,
            "short_term_first": sample.short_term_first,
            "context_id": sample.context_id,
            "labels": [pair.short_term.label, pair.long_term.label],
            "short_term_reward": pair.short_term.reward.value,
            "long_term_reward": pair.long_term.reward.value,
            "short_term_time": pair.short_term.time.to_months(),
            "long_term_time": pair.long_term.time.to_months(),
        }
        for model_name in model_names:
            key = short_model_name(model_name)
            mr = all_model_results[model_name][i]
            entry[f"{key}_response"] = mr["response"]
            entry[f"{key}_choice"] = mr["choice"]
        results.append(entry)
    return results


def _get_choice(result: dict, model_key: str) -> str | None:
    return result.get(f"{model_key}_choice")


# ---------------------------------------------------------------------------
# Visualization: Coherence
# ---------------------------------------------------------------------------


def plot_coherence(results: list[dict], model_keys: list[str], output_dir: Path):
    """Does the response respect the time horizon?

    For each horizon, shows % choosing long-term per model.
    """
    horizon_data: dict[float | None, dict] = defaultdict(
        lambda: {mk: 0 for mk in model_keys} | {"total": 0}
    )
    for r in results:
        h = r["time_horizon_months"]
        horizon_data[h]["total"] += 1
        for mk in model_keys:
            if _get_choice(r, mk) == "long_term":
                horizon_data[h][mk] += 1

    horizons = sorted(horizon_data.keys(), key=lambda x: (x is not None, x or 0))
    labels = [_horizon_label(h) for h in horizons]

    n_models = len(model_keys)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(horizons))
    total_width = 0.8
    bar_width = total_width / n_models

    for idx, mk in enumerate(model_keys):
        pcts = []
        for h in horizons:
            d = horizon_data[h]
            total = d["total"]
            pcts.append(100 * d[mk] / total if total else 0)
        offset = (idx - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, pcts, bar_width, label=mk, color=MODEL_PALETTE[idx % len(MODEL_PALETTE)])

    st_time = results[0]["short_term_time"]
    lt_time = results[0]["long_term_time"]
    ax.axhline(50, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time Horizon")
    ax.set_ylabel("% Choosing Long-Term")
    ax.set_title("Coherence: Does Choice Respect Time Horizon?")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.text(
        0.02, 0.98,
        f"Short: ${results[0]['short_term_reward']:,.0f} in {_horizon_label(st_time)}\n"
        f"Long: ${results[0]['long_term_reward']:,.0f} in {_horizon_label(lt_time)}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    fig.tight_layout()
    _save_fig(fig, output_dir, "coherence")


# ---------------------------------------------------------------------------
# Visualization: Label Stability
# ---------------------------------------------------------------------------


def plot_label_stability(results: list[dict], model_keys: list[str], output_dir: Path):
    """Different labels, same order -- does the choice change?"""
    # Group: (horizon, short_term_first) -> per-model list of choices
    groups: dict[tuple, dict] = defaultdict(lambda: {mk: [] for mk in model_keys})
    for r in results:
        key = (r["time_horizon_months"], r["short_term_first"])
        for mk in model_keys:
            groups[key][mk].append(_get_choice(r, mk))

    horizons_seen = sorted(
        set(k[0] for k in groups.keys()), key=lambda x: (x is not None, x or 0)
    )

    per_model_stability: dict[str, list[float]] = {mk: [] for mk in model_keys}
    labels_out = []

    for h in horizons_seen:
        for mk in model_keys:
            choices_per_order = []
            for stf in [True, False]:
                key = (h, stf)
                if key in groups:
                    choices_per_order.append(groups[key][mk])
            agree = sum(1 for c in choices_per_order if len(set(c)) == 1)
            total = max(len(choices_per_order), 1)
            per_model_stability[mk].append(100 * agree / total)
        labels_out.append(_horizon_label(h))

    n_models = len(model_keys)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels_out))
    total_width = 0.8
    bar_width = total_width / n_models

    for idx, mk in enumerate(model_keys):
        offset = (idx - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, per_model_stability[mk], bar_width, label=mk, color=MODEL_PALETTE[idx % len(MODEL_PALETTE)])

    ax.set_xlabel("Time Horizon")
    ax.set_ylabel("% Groups with Stable Choice")
    ax.set_title("Label Stability: Different Labels, Same Order")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_out, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, output_dir, "label_stability")


# ---------------------------------------------------------------------------
# Visualization: Order Stability
# ---------------------------------------------------------------------------


def plot_order_stability(results: list[dict], model_keys: list[str], output_dir: Path):
    """Same labels, different order -- does flipping change the choice?"""
    # Group: (horizon, label_key) -> {True: per-model choice, False: per-model choice}
    groups: dict[tuple, dict] = defaultdict(dict)
    for r in results:
        label_key = tuple(sorted(r["labels"]))
        key = (r["time_horizon_months"], label_key)
        stf = r["short_term_first"]
        if stf not in groups[key]:
            groups[key][stf] = {mk: _get_choice(r, mk) for mk in model_keys}

    horizons_seen = sorted(
        set(k[0] for k in groups.keys()), key=lambda x: (x is not None, x or 0)
    )

    per_model_stable: dict[str, list[float]] = {mk: [] for mk in model_keys}
    labels_out = []

    for h in horizons_seen:
        model_match = {mk: 0 for mk in model_keys}
        total = 0
        for key, order_map in groups.items():
            if key[0] != h:
                continue
            if True in order_map and False in order_map:
                total += 1
                for mk in model_keys:
                    if order_map[True][mk] == order_map[False][mk]:
                        model_match[mk] += 1
        if total > 0:
            for mk in model_keys:
                per_model_stable[mk].append(100 * model_match[mk] / total)
            labels_out.append(_horizon_label(h))

    n_models = len(model_keys)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels_out))
    total_width = 0.8
    bar_width = total_width / n_models

    for idx, mk in enumerate(model_keys):
        offset = (idx - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, per_model_stable[mk], bar_width, label=mk, color=MODEL_PALETTE[idx % len(MODEL_PALETTE)])

    ax.set_xlabel("Time Horizon")
    ax.set_ylabel("% Pairs with Same Choice (Both Orders)")
    ax.set_title("Order Stability: Same Labels, Different Presentation Order")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_out, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, output_dir, "order_stability")


# ---------------------------------------------------------------------------
# Visualization: No-Horizon Analysis
# ---------------------------------------------------------------------------


def plot_no_horizon(results: list[dict], model_keys: list[str], output_dir: Path):
    """For samples without a time horizon: what's the default preference?"""
    no_horizon = [r for r in results if r["time_horizon_months"] is None]
    if not no_horizon:
        print("  No samples without time horizon -- skipping no_horizon plot.")
        return

    total = len(no_horizon)
    n_models = len(model_keys)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall preference per model
    ax = axes[0]
    pcts = []
    for mk in model_keys:
        long_count = sum(1 for r in no_horizon if _get_choice(r, mk) == "long_term")
        pcts.append(100 * long_count / total)
    ax.bar(model_keys, pcts, color=MODEL_PALETTE[: n_models])
    ax.set_ylabel("% Choosing Long-Term")
    ax.set_title(f"No-Horizon Default Preference (n={total})")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=30)

    # Right: by presentation order
    ax = axes[1]
    by_order: dict[str, dict] = defaultdict(lambda: {mk: 0 for mk in model_keys} | {"total": 0})
    for r in no_horizon:
        order_key = "ST-first" if r["short_term_first"] else "LT-first"
        by_order[order_key]["total"] += 1
        for mk in model_keys:
            if _get_choice(r, mk) == "long_term":
                by_order[order_key][mk] += 1

    order_keys = sorted(by_order.keys())
    x = np.arange(len(order_keys))
    total_width = 0.8
    bar_width = total_width / n_models
    for idx, mk in enumerate(model_keys):
        vals = [100 * by_order[ok][mk] / by_order[ok]["total"] for ok in order_keys]
        offset = (idx - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, vals, bar_width, label=mk, color=MODEL_PALETTE[idx % len(MODEL_PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels(order_keys)
    ax.set_ylabel("% Choosing Long-Term")
    ax.set_title("No-Horizon by Presentation Order")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, output_dir, "no_horizon")


# ---------------------------------------------------------------------------
# Visualization: Context Comparison
# ---------------------------------------------------------------------------


def plot_context_comparison(results: list[dict], model_keys: list[str], output_dir: Path):
    """For samples with different context_id but same options/horizon."""
    context_ids = set(r["context_id"] for r in results if r["context_id"] is not None)
    if len(context_ids) <= 1:
        print("  Only one context -- skipping context comparison.")
        return

    groups: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for r in results:
        key = (r["time_horizon_months"], r["short_term_first"], tuple(r["labels"]))
        cid = r["context_id"]
        groups[key][cid] = {mk: _get_choice(r, mk) for mk in model_keys}

    horizons_seen = sorted(
        set(k[0] for k in groups.keys()), key=lambda x: (x is not None, x or 0)
    )

    per_model_flip: dict[str, list[float]] = {mk: [] for mk in model_keys}
    labels_out = []

    for h in horizons_seen:
        model_flips = {mk: 0 for mk in model_keys}
        total_pairs = 0
        for key, ctx_map in groups.items():
            if key[0] != h:
                continue
            ctx_list = list(ctx_map.values())
            for i in range(len(ctx_list)):
                for j in range(i + 1, len(ctx_list)):
                    total_pairs += 1
                    for mk in model_keys:
                        if ctx_list[i][mk] != ctx_list[j][mk]:
                            model_flips[mk] += 1
        if total_pairs > 0:
            for mk in model_keys:
                per_model_flip[mk].append(100 * model_flips[mk] / total_pairs)
            labels_out.append(_horizon_label(h))

    if not labels_out:
        print("  No context pairs found -- skipping context comparison.")
        return

    n_models = len(model_keys)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels_out))
    total_width = 0.8
    bar_width = total_width / n_models
    for idx, mk in enumerate(model_keys):
        offset = (idx - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, per_model_flip[mk], bar_width, label=mk, color=MODEL_PALETTE[idx % len(MODEL_PALETTE)])
    ax.set_xlabel("Time Horizon")
    ax.set_ylabel("% Choice Flips Between Contexts")
    ax.set_title("Context Comparison: Same Options, Different Context")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_out, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, output_dir, "context_comparison")


# ---------------------------------------------------------------------------
# Visualization: Summary Heatmap
# ---------------------------------------------------------------------------


def plot_summary_heatmap(results: list[dict], model_keys: list[str], output_dir: Path):
    """Heatmap showing choice for every sample, side by side for all models."""
    sorted_results = sorted(
        results,
        key=lambda r: (
            r["time_horizon_months"] is not None,
            r["time_horizon_months"] or 0,
            r["short_term_first"],
            str(r["labels"]),
        ),
    )

    n = len(sorted_results)
    n_models = len(model_keys)
    cmap = ListedColormap(["#E74C3C", "#95A5A6", "#2ECC71"])

    fig, axes = plt.subplots(1, n_models, figsize=(3 * n_models + 2, max(6, n * 0.3)), sharey=True)
    if n_models == 1:
        axes = [axes]

    y_labels = []
    model_choice_data: dict[str, list[float]] = {mk: [] for mk in model_keys}

    for r in sorted_results:
        h_label = _horizon_label(r["time_horizon_months"])
        order = "ST-first" if r["short_term_first"] else "LT-first"
        lbl = r["labels"][0][:2]
        y_labels.append(f"H={h_label} | {order} | {lbl}")

        for mk in model_keys:
            c = _get_choice(r, mk)
            model_choice_data[mk].append(
                1 if c == "long_term" else 0 if c == "short_term" else 0.5
            )

    for ax, mk in zip(axes, model_keys):
        data = np.array(model_choice_data[mk]).reshape(-1, 1)
        ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(mk, fontsize=9)
        ax.set_xticks([])

    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(y_labels, fontsize=6)

    fig.text(0.5, 0.01, "Red=Short-Term | Green=Long-Term | Gray=Unparseable", ha="center", fontsize=8)
    fig.suptitle("Choice Summary Heatmap", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save_fig(fig, output_dir, "summary_heatmap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def get_args():
    parser = argparse.ArgumentParser(description="Behavioral coherence analysis")
    parser.add_argument(
        "config_json",
        type=Path,
        help="Path to a behave JSON config (e.g., investment_behave.json)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help=f"Model names to query (default: {', '.join(ALL_MODELS)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: out/behavioral/<config_name>/)",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()

    # 1. Load config and generate dataset
    config_path = args.config_json
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        return 1

    print(f"Loading config from {config_path}")
    config = PromptDatasetConfig.from_json(config_path)
    print(f"Config: {config.name}, horizons: {len(config.time_horizons)}")

    generator = PromptDatasetGenerator(config)
    dataset = generator.generate()
    samples = dataset.samples
    print(f"Generated {len(samples)} prompt samples")

    choice_prefix = config.prompt_format_config.get_response_prefix_before_choice()

    # 2. Set up output directory
    output_dir = args.output_dir or Path("out/behavioral") / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Determine models
    model_names = args.models if args.models else list(ALL_MODELS)
    model_keys = [short_model_name(m) for m in model_names]

    # 4. Query each model
    all_model_results: dict[str, list[dict]] = {}
    for model_name in model_names:
        print(f"\n--- Querying {short_model_name(model_name)} ---")
        all_model_results[model_name] = query_model(model_name, samples, choice_prefix)

    # 5. Build and save results
    results = build_results(samples, model_names, all_model_results)

    responses_path = output_dir / "responses.json"
    with open(responses_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved responses to {responses_path}")

    # 6. Print summary
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Results Summary ({total} samples)")
    print(f"{'='*60}")
    for mk in model_keys:
        long_count = sum(1 for r in results if _get_choice(r, mk) == "long_term")
        none_count = sum(1 for r in results if _get_choice(r, mk) is None)
        print(f"  {mk:>35}: {long_count}/{total} long-term ({100*long_count/total:.0f}%), {none_count} unparseable")

    # 7. Generate visualizations
    print("\nGenerating visualizations...")
    plot_coherence(results, model_keys, output_dir)
    plot_label_stability(results, model_keys, output_dir)
    plot_order_stability(results, model_keys, output_dir)
    plot_no_horizon(results, model_keys, output_dir)
    plot_context_comparison(results, model_keys, output_dir)
    plot_summary_heatmap(results, model_keys, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
