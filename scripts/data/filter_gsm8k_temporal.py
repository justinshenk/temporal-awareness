#!/usr/bin/env python3
"""
Filter GSM8K + HumanEval for temporal/time-related examples.

Medium-stakes dataset (math problem solving + code review).
Downloads from HuggingFace, filters for any problems mentioning
time-related concepts (rates, schedules, deadlines, ages, durations),
and saves the filtered subset as JSON for manual review.

No modifications to the original data — just filtering and saving.

Usage:
    pip install datasets --break-system-packages
    python scripts/data/filter_gsm8k_temporal.py
"""

import json
import re
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filtered_temporal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Temporal HORIZON keywords for math/code — focused on problems where
# time-based reasoning is CENTRAL to the problem, not incidental.
#
# For math: rate problems, scheduling problems, time-to-completion,
# compound interest, growth over time — these inherently involve
# temporal horizon reasoning.
#
# For code: timeout logic, scheduling, real-time constraints.
# ---------------------------------------------------------------------------
TEMPORAL_KEYWORDS = [
    # === Time-rate problems (math) — inherently about temporal reasoning ===
    r"\bper hour\b",
    r"\bper day\b",
    r"\bper week\b",
    r"\bper month\b",
    r"\bper year\b",
    r"\bper minute\b",
    r"\bper second\b",
    r"\bhow long\b",
    r"\bhow many (days|hours|minutes|years|weeks|months)\b",
    # === Scheduling / planning ===
    r"\bschedule\b",
    r"\bdeadline\b",
    r"\bovertime\b",
    r"\bshift\b",
    r"\bsemester\b",
    # === Growth / compound / projection over time ===
    r"\binterest rate\b",
    r"\bcompound\b",
    r"\bappreciat(e|ion|es|ing)\b",
    r"\bdepreciat(e|ion|es|ing)\b",
    r"\bgrowth rate\b",
    r"\binflation\b",
    r"\binvestment\b",
    # === Duration / elapsed time (core of the problem) ===
    r"\bduration\b",
    r"\belapsed\b",
    r"\btakes?\s+\d+\s+(hours?|minutes?|days?|weeks?|months?|years?)\b",
    r"\bspend(s|ing)?\s+\d+\s+(hours?|minutes?|days?|weeks?|months?)\b",
    # === Code temporal terms ===
    r"\btimeout\b",
    r"\blatency\b",
    r"\bthroughput\b",
    r"\bcron\b",
    r"\bscheduler\b",
    r"\breal[- ]?time\b",
    r"\bexpir(e|ation|ed|ing)\b",
    r"\bttl\b",
    r"\bbackoff\b",
    # === Horizon contrast ===
    r"\blong[- ]term\b",
    r"\bshort[- ]term\b",
    r"\bimmediate\b",
]

# Require at least 2 matches for math (avoids "how many hours" in a
# non-temporal arithmetic problem)
MIN_TEMPORAL_MATCHES = 2

TEMPORAL_PATTERN = re.compile("|".join(TEMPORAL_KEYWORDS), re.IGNORECASE)


def find_temporal_matches(text: str) -> list[str]:
    """Return all temporal keyword matches found in text."""
    return list(set(m.group() for m in TEMPORAL_PATTERN.finditer(text)))


def main():
    all_filtered = []

    # ---- GSM8K ----
    print("Loading GSM8K dataset...")
    ds_gsm = load_dataset("gsm8k", "main", split="train")
    print(f"Loaded {len(ds_gsm)} GSM8K examples")
    print(f"Fields: {list(ds_gsm[0].keys())}")

    gsm_filtered = []
    for i, example in enumerate(ds_gsm):
        text = f"{example['question']} {example['answer']}"
        matches = find_temporal_matches(text)
        if len(matches) >= MIN_TEMPORAL_MATCHES:
            gsm_filtered.append({
                "original_index": i,
                "source": "gsm8k",
                "temporal_matches": matches,
                "n_temporal_words": len(matches),
                "data": {k: v for k, v in example.items()},
            })

    print(f"GSM8K: {len(gsm_filtered)}/{len(ds_gsm)} contain temporal keywords "
          f"({len(gsm_filtered)/len(ds_gsm)*100:.0f}%)")
    all_filtered.extend(gsm_filtered)

    # ---- HumanEval ----
    print("\nLoading HumanEval dataset...")
    try:
        ds_he = load_dataset("openai_humaneval", split="test")
        print(f"Loaded {len(ds_he)} HumanEval examples")

        he_filtered = []
        for i, example in enumerate(ds_he):
            text = f"{example.get('prompt', '')} {example.get('canonical_solution', '')} {example.get('test', '')}"
            matches = find_temporal_matches(text)
            if len(matches) >= MIN_TEMPORAL_MATCHES:
                he_filtered.append({
                    "original_index": i,
                    "source": "humaneval",
                    "temporal_matches": matches,
                    "n_temporal_words": len(matches),
                    "data": {k: v for k, v in example.items()},
                })

        print(f"HumanEval: {len(he_filtered)}/{len(ds_he)} contain temporal keywords "
              f"({len(he_filtered)/len(ds_he)*100:.0f}%)")
        all_filtered.extend(he_filtered)

    except Exception as e:
        print(f"HumanEval load failed: {e}")

    # ---- MATH dataset (bonus — harder math problems) ----
    print("\nLoading MATH dataset...")
    try:
        ds_math = load_dataset("hendrycks/competition_math", split="train",
                               trust_remote_code=True)
        print(f"Loaded {len(ds_math)} MATH examples")

        math_filtered = []
        for i, example in enumerate(ds_math):
            text = f"{example.get('problem', '')} {example.get('solution', '')}"
            matches = find_temporal_matches(text)
            if len(matches) >= MIN_TEMPORAL_MATCHES:
                math_filtered.append({
                    "original_index": i,
                    "source": "competition_math",
                    "temporal_matches": matches,
                    "n_temporal_words": len(matches),
                    "data": {k: v for k, v in example.items()},
                })

        print(f"MATH: {len(math_filtered)}/{len(ds_math)} contain temporal keywords "
              f"({len(math_filtered)/len(ds_math)*100:.0f}%)")
        all_filtered.extend(math_filtered)

    except Exception as e:
        print(f"MATH dataset load failed: {e}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(all_filtered)} total temporal examples")
    print(f"{'='*60}")

    # Show distribution of temporal keywords
    keyword_counts = {}
    source_counts = {}
    for item in all_filtered:
        source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1
        for kw in item["temporal_matches"]:
            keyword_counts[kw.lower()] = keyword_counts.get(kw.lower(), 0) + 1

    print("\nBy source:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")

    print("\nTop 20 temporal keywords found:")
    for kw, count in sorted(keyword_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {kw}: {count}")

    # Save filtered results
    output_path = OUTPUT_DIR / "gsm8k_humaneval_math_temporal_filtered.json"
    with open(output_path, "w") as f:
        json.dump({
            "source_datasets": ["gsm8k/main", "openai_humaneval", "hendrycks/competition_math"],
            "total_filtered": len(all_filtered),
            "source_counts": source_counts,
            "temporal_keywords_used": TEMPORAL_KEYWORDS,
            "keyword_distribution": keyword_counts,
            "examples": all_filtered,
        }, f, indent=2, default=str)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Show a few examples from each source
    for src in ["gsm8k", "humaneval", "competition_math"]:
        examples = [x for x in all_filtered if x["source"] == src][:3]
        if examples:
            print(f"\n--- Sample {src} examples ---")
            for item in examples:
                data = item["data"]
                text = (data.get("question") or data.get("prompt") or
                        data.get("problem") or "")[:200]
                print(f"\n[{item['original_index']}] matches={item['temporal_matches']}")
                print(f"  Q: {text}...")


if __name__ == "__main__":
    main()
