#!/usr/bin/env python3
"""
Filter LegalBench for temporal/time-related examples.

High-stakes legal dataset. Downloads from HuggingFace, filters for any
tasks/examples mentioning time-related concepts (statutes of limitations,
deadlines, sentencing durations, etc.), and saves the filtered subset
as JSON for manual review.

No modifications to the original data — just filtering and saving.

Usage:
    pip install datasets --break-system-packages
    python scripts/data/filter_legalbench_temporal.py
"""

import json
import re
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filtered_temporal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Temporal HORIZON keywords — focused on planning horizon / urgency contrast,
# NOT generic time mentions like "filed on January 5" or "three years ago."
#
# Goal: find legal examples where temporal reasoning is central —
# statutes of limitations, sentencing duration, retroactivity,
# immediate vs long-term legal consequences.
# ---------------------------------------------------------------------------
TEMPORAL_KEYWORDS = [
    # === Horizon contrast ===
    r"\blong[- ]term\b",
    r"\bshort[- ]term\b",
    r"\bimmediate\b",
    r"\bpermanent\b",
    r"\btemporary\b",
    r"\blifetime\b",
    r"\blifelong\b",
    r"\bforeseeable future\b",
    # === Legal temporal structures (these are inherently about time horizons) ===
    r"\bstatute of limitations?\b",
    r"\blimitations? period\b",
    r"\bprescription period\b",
    r"\bstatutory period\b",
    r"\bfiling deadline\b",
    r"\bretroacti(ve|vely)\b",
    r"\bprospective(ly)?\b",
    r"\bex post facto\b",
    # === Sentencing / duration of consequence ===
    r"\bimprisonment\b",
    r"\bincarceration\b",
    r"\bprobation\b",
    r"\bparole\b",
    r"\blife sentence\b",
    r"\bminimum sentence\b",
    r"\bmaximum sentence\b",
    r"\bsentencing\b",
    # === Temporal status / vesting ===
    r"\bvest(ed|ing)\b",
    r"\bexpir(e|ation|ed|ing)\b",
    r"\beffective date\b",
    r"\bsunset\b",
    r"\bmoratorium\b",
    # === Urgency ===
    r"\burgent\b",
    r"\binjunction\b",
    r"\binterim\b",
    r"\bpreliminary\b",
    r"\bdelayed?\b",
    r"\bpostpone\b",
    r"\bstay(ed)?\b",
]

# Require at least 2 horizon keywords for an example to count
MIN_TEMPORAL_MATCHES = 2

TEMPORAL_PATTERN = re.compile("|".join(TEMPORAL_KEYWORDS), re.IGNORECASE)


def extract_text(example: dict) -> str:
    """Extract all searchable text from a LegalBench example."""
    parts = []
    for field in ["text", "question", "answer", "input", "output",
                  "instruction", "context", "passage", "label",
                  "premise", "hypothesis"]:
        val = example.get(field)
        if val is not None:
            if isinstance(val, (list, dict)):
                parts.append(json.dumps(val))
            else:
                parts.append(str(val))
    return " ".join(parts)


def find_temporal_matches(text: str) -> list[str]:
    """Return all temporal keyword matches found in text."""
    return list(set(m.group() for m in TEMPORAL_PATTERN.finditer(text)))


def main():
    print("Loading LegalBench dataset from HuggingFace...")
    print("LegalBench has many sub-tasks. Loading all available configs...\n")

    # LegalBench is organized by task — each task is a separate config
    # First, get the list of configs
    from datasets import get_dataset_config_names
    try:
        configs = get_dataset_config_names("nguha/legalbench")
        print(f"Found {len(configs)} LegalBench tasks/configs")
    except Exception as e:
        print(f"Could not list configs: {e}")
        print("Trying to load default split...")
        configs = [None]

    all_filtered = []
    task_stats = {}

    for config in configs:
        try:
            if config:
                ds = load_dataset("nguha/legalbench", config, split="test",
                                  trust_remote_code=True)
            else:
                ds = load_dataset("nguha/legalbench", split="test",
                                  trust_remote_code=True)

            task_filtered = []
            for i, example in enumerate(ds):
                text = extract_text(example)
                matches = find_temporal_matches(text)
                if len(matches) >= MIN_TEMPORAL_MATCHES:
                    task_filtered.append({
                        "original_index": i,
                        "task": config or "default",
                        "temporal_matches": matches,
                        "n_temporal_words": len(matches),
                        "data": {k: v for k, v in example.items()},
                    })

            if task_filtered:
                task_stats[config or "default"] = {
                    "total": len(ds),
                    "temporal": len(task_filtered),
                    "pct": len(task_filtered) / len(ds) * 100,
                }
                all_filtered.extend(task_filtered)
                print(f"  [{config}] {len(task_filtered)}/{len(ds)} temporal "
                      f"({len(task_filtered)/len(ds)*100:.0f}%)")

        except Exception as e:
            print(f"  [{config}] SKIP: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(all_filtered)} total temporal examples across "
          f"{len(task_stats)} tasks")
    print(f"{'='*60}")

    # Show distribution of temporal keywords
    keyword_counts = {}
    for item in all_filtered:
        for kw in item["temporal_matches"]:
            keyword_counts[kw.lower()] = keyword_counts.get(kw.lower(), 0) + 1

    print("\nTop 20 temporal keywords found:")
    for kw, count in sorted(keyword_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {kw}: {count}")

    # Show top tasks by temporal content
    print("\nTop 10 tasks by temporal example count:")
    for task, stats in sorted(task_stats.items(), key=lambda x: -x[1]["temporal"])[:10]:
        print(f"  {task}: {stats['temporal']}/{stats['total']} ({stats['pct']:.0f}%)")

    # Save filtered results
    output_path = OUTPUT_DIR / "legalbench_temporal_filtered.json"
    with open(output_path, "w") as f:
        json.dump({
            "source_dataset": "nguha/legalbench",
            "total_tasks": len(configs),
            "tasks_with_temporal": len(task_stats),
            "total_filtered": len(all_filtered),
            "temporal_keywords_used": TEMPORAL_KEYWORDS,
            "keyword_distribution": keyword_counts,
            "task_stats": task_stats,
            "examples": all_filtered,
        }, f, indent=2, default=str)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Show a few examples
    print("\n--- Sample filtered examples ---")
    for item in all_filtered[:5]:
        data = item["data"]
        text = extract_text(data)[:200]
        print(f"\n[{item['task']}/{item['original_index']}] "
              f"matches={item['temporal_matches']}")
        print(f"  Text: {text}...")


if __name__ == "__main__":
    main()
