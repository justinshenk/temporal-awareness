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
from collections import Counter, defaultdict
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
    if model.startswith("openai:"):
        return model[7:]
    if model.startswith("gemini:"):
        return model[7:]
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


def query_model(
    model_name: str,
    samples,
    choice_prefix: str,
    reasoning: bool = False,
    cache_path: Path | None = None,
) -> list[dict]:
    """Query a model with all samples, return list of {response, choice}.

    When reasoning=True, prefilling is skipped (model is allowed to think) and
    max_new_tokens is raised to accommodate the <think>...</think> block.

    When cache_path is given, per-sample results are checkpointed after every
    sample. On restart, already-completed samples are loaded and skipped.
    """
    from src.inference import ModelRunner

    label = short_model_name(model_name) + ("-thinking" if reasoning else "")
    results: list[dict] = []
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        print(
            f"  [{label}] resumed from cache: {len(results)}/{len(samples)} done"
        )

    if len(results) >= len(samples):
        return results[: len(samples)]

    runner = ModelRunner(model_name)
    # OpenAI o-series are reasoning models that consume tokens internally
    # before emitting the final answer; need a high cap even outside
    # --reasoning mode (gpt-5.x is fine at 256).
    is_openai_o_series = model_name.startswith("openai:") and any(
        s in model_name.lower() for s in (":o1", ":o3", ":o4")
    )
    # Gemini 2.5 family ("flash"/"pro") uses internal thinking tokens; needs
    # a high max_output_tokens to leave budget for the visible answer.
    is_gemini_thinking = model_name.startswith("gemini:") and "2.5" in model_name
    if reasoning:
        prefilling = ""
        max_new_tokens = 4096
    elif is_openai_o_series or is_gemini_thinking:
        prefilling = runner.skip_thinking_prefix
        max_new_tokens = 4096
    else:
        prefilling = runner.skip_thinking_prefix
        max_new_tokens = 256

    start_idx = len(results)
    for i in range(start_idx, len(samples)):
        sample = samples[i]
        print(f"  [{label}] sample {i + 1}/{len(samples)}")
        response = runner.generate(
            sample.text,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            prefilling=prefilling,
        )
        choice = parse_choice(response, sample, choice_prefix)
        results.append({"response": response, "choice": choice})
        if cache_path is not None:
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(results, f, default=str)
            tmp.replace(cache_path)
    return results


CHOICE_PHRASES = [
    r"i\s+choose",
    r"i\s+would\s+choose",
    r"i\s+(?:will|'ll)\s+choose",
    r"i\s+pick",
    r"i\s+select",
    r"i\s+go\s+with",
    r"my\s+choice\s+is",
    r"my\s+answer\s+is",
    r"the\s+answer\s+is",
    r"final\s+answer\s*:?",
    r"answer\s*:",
    r"choice\s*:",
    r"option",
]


# Strong first-person commit phrases used by the unclosed-<think> salvage.
STRONG_COMMIT_PHRASES = [
    r"i\s*'ll\s+go\s+with",
    r"i\s+will\s+go\s+with",
    r"i\s+would\s+go\s+with",
    r"i\s+go\s+with",
    r"i\s+am\s+going\s+with",
    r"i'm\s+going\s+with",
    r"going\s+with",
    r"i\s*'ll\s+choose",
    r"i\s+will\s+choose",
    r"i\s+would\s+choose",
    r"i\s+choose",
    r"i\s*'ll\s+pick",
    r"i\s+will\s+pick",
    r"i\s+pick",
    r"i\s*'ll\s+select",
    r"i\s+will\s+select",
    r"i\s+select",
    r"my\s+(?:final\s+)?(?:choice|pick|selection)\s+is",
    r"final\s+answer\s*[:=]",
    r"my\s+(?:final\s+)?answer\s+is",
    r"i\s+conclude\s+(?:that\s+)?(?:the\s+)?(?:answer\s+)?(?:is\s+)?",
]

# Hedged third-person commit phrases used by the unclosed-<think> salvage.
WEAK_COMMIT_PHRASES = [
    r"i\s+(?:think|believe|guess)\s+the\s+answer\s+is",
    r"i\s+(?:think|believe)\s+the\s+correct\s+answer\s+is",
    r"i\s+(?:think|believe)\s+(?:that\s+)?the\s+answer\s+(?:would|should|might|must)\s+be",
    r"the\s+answer\s+is",
    r"the\s+correct\s+answer\s+is",
    r"the\s+answer\s+(?:would|should|might|must)\s+be",
    r"the\s+best\s+option\s+is",
    r"the\s+better\s+option\s+is",
    r"the\s+best\s+choice\s+is",
    r"the\s+better\s+choice\s+is",
    r"the\s+(?:better|best)\s+(?:investment|alternative)\s+is",
    r"so\s+the\s+answer\s+is",
    r"thus\s+the\s+answer\s+is",
    r"therefore[,]?\s+the\s+answer\s+is",
    r"hence\s+the\s+answer\s+is",
]


def parse_choice(response: str, sample, choice_prefix: str) -> str | None:
    """Parse model response to determine short_term or long_term choice.

    Returns 'short_term', 'long_term', or None if unparseable.

    Reasoning-mode failures: if `<think>` is open without a closing `</think>`,
    falls back to the unclosed-<think> SALVAGE strategy. The salvage looks
    for first-person ("I choose X", "I'll go with X") and hedged
    ("the answer is X") commitment phrases in the last 800 chars and uses
    label/descriptor matching (the descriptor mapping is dataset-specific;
    see `_english_descriptor_for`).

    False-positive risk: in cases where the model loops with conflicting
    commits ("I think X. But maybe Y. I think X."), we take the LAST strong
    commit (or last-3-agree weak commit). Single hedged commits are also
    accepted. This is more aggressive than refusing to guess at all and may
    occasionally miscategorize models that flip at the very end.
    """
    pair = sample.prompt.preference_pair
    short_label = pair.short_term.label.strip().rstrip(".")
    long_label = pair.long_term.label.strip().rstrip(".")

    if not response:
        return None

    # Reasoning models that ran over max_new_tokens leave <think> unclosed.
    # Try the salvage path before giving up.
    if "<think>" in response and "</think>" not in response:
        return _salvage_unclosed_think(response, short_label, long_label)

    # Strip <think>...</think>; what remains is the model's final answer.
    answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if not answer:
        return None

    clean = re.sub(r"[*_`]+", "", answer)

    # Strategy 1: original "I choose: <label>" pattern (with colon)
    prefix_pattern = re.escape(choice_prefix.strip())
    m = re.search(prefix_pattern + r"\s*(.+?)[\.\n]", clean, re.IGNORECASE)
    if m:
        chosen = m.group(1).strip().rstrip(".")
        if _label_match(chosen, short_label):
            return "short_term"
        if _label_match(chosen, long_label):
            return "long_term"

    # Strategy 2: phrase-based detection ("I choose X", "Option A", ...)
    phrase_result = _try_phrase_match(clean, short_label, long_label)
    if phrase_result is not None:
        return phrase_result

    # Strategy 3: first non-empty line starts with a label
    for line in clean.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if _label_match(line, short_label):
            return "short_term"
        if _label_match(line, long_label):
            return "long_term"
        break

    # Strategy 4: word-boundary search (only one label present)
    answer_lower = answer.lower()
    if _word_boundary_search(short_label, answer_lower) and not _word_boundary_search(long_label, answer_lower):
        return "short_term"
    if _word_boundary_search(long_label, answer_lower) and not _word_boundary_search(short_label, answer_lower):
        return "long_term"

    # Strategy 5: "short-term" / "long-term" phrasing
    has_short = "short-term" in answer_lower or "short term" in answer_lower
    has_long = "long-term" in answer_lower or "long term" in answer_lower
    if has_short and not has_long:
        return "short_term"
    if has_long and not has_short:
        return "long_term"

    return None


def _label_match(text: str, label: str) -> bool:
    """Check if text matches or starts with a label."""
    text_clean = text.strip().lower().rstrip(".)")
    label_clean = label.strip().lower().rstrip(".)")
    if not text_clean or not label_clean:
        return False
    return text_clean.startswith(label_clean) or label_clean.startswith(text_clean)


def _word_boundary_search(label: str, text_lower: str) -> bool:
    """Word-boundary search for label (avoids 'x' matching 'six', 'max', etc.)."""
    bare = label.strip().lower().rstrip(".)")
    if not bare:
        return False
    return re.search(r"\b" + re.escape(bare) + r"\b", text_lower) is not None


def _try_phrase_match(text: str, short_label: str, long_label: str) -> str | None:
    """Look for 'I choose X' / 'Answer: A' / 'Option B' patterns."""
    for phrase in CHOICE_PHRASES:
        pattern = (
            r"\b" + phrase + r"\b"
            r"\s*[:\-]?\s*"
            r"\(?\s*"
            r"([A-Za-z0-9]{1,3})"
            r"\s*[\.\)]?\s*\)?"
        )
        for m in re.finditer(pattern, text, re.IGNORECASE):
            token = m.group(1)
            if _label_match(token, short_label):
                return "short_term"
            if _label_match(token, long_label):
                return "long_term"
    return None


# ---------------------------------------------------------------------------
# Unclosed-<think> salvage helpers
# ---------------------------------------------------------------------------


def _english_descriptor_for(captured: str) -> str | None:
    """Map a captured chunk to 'short' or 'long' based on dataset descriptors.

    Hardcoded for the investment_behave schema:
        short_term = $20,000 in 6 months
        long_term  = $100k / $300k / $500k in 10 years

    Returns the FIRST (leftmost) descriptor mention, or None.
    Update if other reward/horizon schemas are introduced.
    """
    s = captured.lower()
    short_patterns = [
        re.compile(r"\$?\s*20[\s,]?000"),
        re.compile(r"\$?\s*20\s*k\b"),
        re.compile(r"\b6\s*[-\s]?month"),
        re.compile(r"\bsix\s*[-\s]?month"),
        re.compile(r"\bshort[-\s]term"),
    ]
    long_patterns = [
        re.compile(r"\$?\s*100[\s,]?000"),
        re.compile(r"\$?\s*300[\s,]?000"),
        re.compile(r"\$?\s*500[\s,]?000"),
        re.compile(r"\$?\s*100\s*k\b"),
        re.compile(r"\$?\s*300\s*k\b"),
        re.compile(r"\$?\s*500\s*k\b"),
        re.compile(r"\b10\s*[-\s]?year"),
        re.compile(r"\bten\s*[-\s]?year"),
        re.compile(r"\blong[-\s]term"),
    ]
    earliest: tuple[int, str] | None = None
    for pat in short_patterns:
        m = pat.search(s)
        if m and (earliest is None or m.start() < earliest[0]):
            earliest = (m.start(), "short")
    for pat in long_patterns:
        m = pat.search(s)
        if m and (earliest is None or m.start() < earliest[0]):
            earliest = (m.start(), "long")
    return earliest[1] if earliest else None


def _classify_capture(captured: str, short_label: str, long_label: str) -> str | None:
    """Classify a captured post-phrase chunk as 'short' or 'long'."""
    captured = captured.strip()
    by_token = None
    by_desc = _english_descriptor_for(captured)

    m = re.match(r"\(?\s*([A-Za-z0-9]{1,3})\s*[\.\)\]\"']?\b", captured)
    if m:
        token = m.group(1)
        if _label_match(token, short_label):
            by_token = "short"
        elif _label_match(token, long_label):
            by_token = "long"

    if by_token is None:
        m2 = re.search(r"\boption\s+([A-Za-z0-9]{1,3})\b", captured, re.IGNORECASE)
        if m2:
            token = m2.group(1)
            if _label_match(token, short_label):
                by_token = "short"
            elif _label_match(token, long_label):
                by_token = "long"

    if by_token is not None and by_desc is not None and by_token != by_desc:
        return None
    return by_token or by_desc


def _find_phrase_commits(
    text: str,
    phrases: list[str],
    short_label: str,
    long_label: str,
    window: int,
) -> list[tuple[int, str]]:
    """Find all phrase commits in the last `window` chars of `text`.
    Returns list of (position, kind) where kind is 'short' or 'long'."""
    text = re.sub(r"[*_`]+", "", text)
    tail = text[-window:]
    base = len(text) - len(tail)
    out: list[tuple[int, str]] = []
    for phrase in phrases:
        pattern = re.compile(
            r"\b" + phrase + r"\b"
            r"\s*[:\-]?\s*"
            r"(?:that\s+|to\s+(?:choose\s+|pick\s+|invest\s+in\s+|go\s+with\s+|select\s+)?|option\s+|the\s+)?"
            r"[\(\[\"']?"
            r"([\w\$%\.,\-\s]{1,80})",
            re.IGNORECASE,
        )
        for m in pattern.finditer(tail):
            kind = _classify_capture(m.group(1), short_label, long_label)
            if kind is not None:
                out.append((base + m.start(), kind))
    return out


def _salvage_unclosed_think(
    response: str, short_label: str, long_label: str, window: int = 800
) -> str | None:
    """Salvage a choice from a response that ran over max_new_tokens
    inside <think>...</think>. Tiered:
      - A1: a STRONG commit phrase exists; use the last one (with majority
        tiebreak across the last 3 if the last two disagree).
      - A2: only WEAK commit phrases; require last-3 agreement OR a clear
        last-5 majority OR the phrase appears exactly once.
    """
    strong = _find_phrase_commits(
        response, STRONG_COMMIT_PHRASES, short_label, long_label, window
    )
    weak = _find_phrase_commits(
        response, WEAK_COMMIT_PHRASES, short_label, long_label, window
    )

    if strong:
        strong.sort(key=lambda x: x[0])
        if len(strong) >= 2 and strong[-1][1] != strong[-2][1]:
            last_three = [s[1] for s in strong[-3:]]
            c = Counter(last_three)
            if c["short"] > c["long"]:
                return "short_term"
            if c["long"] > c["short"]:
                return "long_term"
        kind = strong[-1][1]
        return "short_term" if kind == "short" else "long_term"

    if weak:
        weak.sort(key=lambda x: x[0])
        last_three = [w[1] for w in weak[-3:]]
        if len(last_three) >= 2 and len(set(last_three)) == 1:
            kind = last_three[0]
            return "short_term" if kind == "short" else "long_term"
        if len(weak) >= 5:
            last_five = [w[1] for w in weak[-5:]]
            c = Counter(last_five)
            if c["short"] >= 4:
                return "short_term"
            if c["long"] >= 4:
                return "long_term"
        if len(weak) == 1:
            kind = weak[0][1]
            return "short_term" if kind == "short" else "long_term"

    return None


# ---------------------------------------------------------------------------
# Result building
# ---------------------------------------------------------------------------


def build_results(
    samples,
    model_names: list[str],
    all_model_results: dict[str, list[dict]],
    reasoning: bool = False,
) -> list[dict]:
    """Build the combined results list with per-model responses."""
    results = []
    suffix = "-thinking" if reasoning else ""
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
            key = short_model_name(model_name) + suffix
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
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable thinking/reasoning mode (skip prefilling, raise max_tokens, "
             "suffix output keys with -thinking).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of samples (for smoke tests).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable per-sample checkpointing (cache is on by default; "
             "cache files live at <output_dir>/_cache_<label>.json).",
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
    if args.limit is not None:
        samples = samples[: args.limit]
        print(f"Limiting to first {len(samples)} samples")
    print(f"Generated {len(samples)} prompt samples")

    choice_prefix = config.prompt_format_config.get_response_prefix_before_choice()

    # 2. Set up output directory
    output_dir = args.output_dir or Path("out/behavioral") / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Determine models
    model_names = args.models if args.models else list(ALL_MODELS)
    model_key_suffix = "-thinking" if args.reasoning else ""
    model_keys = [short_model_name(m) + model_key_suffix for m in model_names]

    # 4. Query each model
    all_model_results: dict[str, list[dict]] = {}
    for model_name in model_names:
        suffix = "-thinking" if args.reasoning else ""
        print(f"\n--- Querying {short_model_name(model_name)}{suffix} ---")
        cache_path = (
            None
            if args.no_cache
            else output_dir / f"_cache_{short_model_name(model_name)}{suffix}.json"
        )
        all_model_results[model_name] = query_model(
            model_name,
            samples,
            choice_prefix,
            reasoning=args.reasoning,
            cache_path=cache_path,
        )

    # 5. Build and save results
    results = build_results(
        samples, model_names, all_model_results, reasoning=args.reasoning
    )

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
