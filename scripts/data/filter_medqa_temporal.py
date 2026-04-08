#!/usr/bin/env python3
"""
Filter MedQA (USMLE-style) for temporal/time-related examples.

High-stakes medical dataset. Downloads from HuggingFace, filters for any
questions or answer options mentioning time-related concepts, and saves
the filtered subset as JSON for manual review.

No modifications to the original data — just filtering and saving.

Usage:
    pip install datasets --break-system-packages
    python scripts/data/filter_medqa_temporal.py
"""

import json
import re
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filtered_temporal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Temporal HORIZON keywords — focused on planning horizon / urgency contrast,
# NOT generic time mentions like "23-year-old" or "3 months ago".
#
# Goal: find questions where the *reasoning* involves a temporal horizon
# (acute vs chronic management, immediate intervention vs long-term planning,
# short-term vs long-term prognosis).
# ---------------------------------------------------------------------------
TEMPORAL_KEYWORDS = [
    # === Horizon contrast (the most valuable — direct short vs long) ===
    r"\bacute\b",
    r"\bchronic\b",
    r"\blong[- ]term\b",
    r"\bshort[- ]term\b",
    r"\bimmediate\b",
    r"\blong[- ]range\b",
    r"\bshort[- ]range\b",
    r"\bend[- ]of[- ]life\b",
    r"\bpalliative\b",
    r"\bterminal\b",
    r"\blifelong\b",
    # === Medical planning / prognosis ===
    r"\bprognosis\b",
    r"\bfollow[- ]up\b",
    r"\bmanagement plan\b",
    r"\btreatment plan\b",
    r"\blong[- ]term management\b",
    r"\blong[- ]term prognosis\b",
    r"\blong[- ]term complication\b",
    r"\blong[- ]term outcome\b",
    r"\blong[- ]term treatment\b",
    r"\bshort[- ]term management\b",
    r"\bpreventive\b",
    r"\bprevention\b",
    r"\bprophylaxis\b",
    r"\bscreening\b",
    r"\bearly detection\b",
    # === Temporal trajectory / progression ===
    r"\bprogression\b",
    r"\bprogressive\b",
    r"\binsidious\b",
    r"\bgradual\b",
    r"\brapid onset\b",
    r"\bsudden onset\b",
    r"\bremission\b",
    r"\brelapse\b",
    r"\brecurrence\b",
    # === Urgency level ===
    r"\bemergency\b",
    r"\blife[- ]threatening\b",
    r"\burgent\b",
    r"\bstat\b",
    r"\bdelayed\b",
]

# Require at least this many horizon keywords to count as a temporal example.
# This filters out questions that just happen to mention "acute" once in a
# list of symptoms vs questions where temporal reasoning is central.
MIN_TEMPORAL_MATCHES = 2

TEMPORAL_PATTERN = re.compile("|".join(TEMPORAL_KEYWORDS), re.IGNORECASE)


def extract_text(example: dict) -> str:
    """Extract all searchable text from a MedQA example."""
    parts = []
    # Different MedQA formats have different field names
    for field in ["question", "sent1", "sent2", "ending0", "ending1",
                  "ending2", "ending3", "ending4", "answer", "exp",
                  "options", "meta_info", "answer_idx"]:
        val = example.get(field)
        if val is not None:
            if isinstance(val, dict):
                parts.extend(str(v) for v in val.values())
            elif isinstance(val, list):
                parts.extend(str(v) for v in val)
            else:
                parts.append(str(val))
    return " ".join(parts)


def find_temporal_matches(text: str) -> list[str]:
    """Return all temporal keyword matches found in text."""
    return list(set(m.group() for m in TEMPORAL_PATTERN.finditer(text)))


def main():
    print("Loading MedQA dataset from HuggingFace...")
    # Try the USMLE 4-option version first
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        dataset_name = "GBaker/MedQA-USMLE-4-options"
    except Exception:
        ds = load_dataset("bigbio/med_qa", "med_qa_en_source", split="train")
        dataset_name = "bigbio/med_qa"

    print(f"Loaded {len(ds)} examples from {dataset_name}")
    print(f"Fields: {list(ds[0].keys())}")
    print(f"\nSample example:\n{json.dumps(ds[0], indent=2, default=str)[:1000]}")

    # Filter for temporal content — require MIN_TEMPORAL_MATCHES horizon keywords
    filtered = []
    for i, example in enumerate(ds):
        text = extract_text(example)
        matches = find_temporal_matches(text)
        if len(matches) >= MIN_TEMPORAL_MATCHES:
            filtered.append({
                "original_index": i,
                "temporal_matches": matches,
                "n_temporal_words": len(matches),
                "data": {k: v for k, v in example.items()},
            })

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(filtered)} / {len(ds)} examples contain temporal keywords")
    print(f"{'='*60}")

    # Show distribution of temporal keywords
    keyword_counts = {}
    for item in filtered:
        for kw in item["temporal_matches"]:
            keyword_counts[kw.lower()] = keyword_counts.get(kw.lower(), 0) + 1

    print("\nTop 20 temporal keywords found:")
    for kw, count in sorted(keyword_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {kw}: {count}")

    # Save filtered results
    output_path = OUTPUT_DIR / "medqa_temporal_filtered.json"
    with open(output_path, "w") as f:
        json.dump({
            "source_dataset": dataset_name,
            "total_examples": len(ds),
            "filtered_count": len(filtered),
            "temporal_keywords_used": TEMPORAL_KEYWORDS,
            "keyword_distribution": keyword_counts,
            "examples": filtered,
        }, f, indent=2, default=str)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Show a few examples
    print("\n--- Sample filtered examples ---")
    for item in filtered[:5]:
        data = item["data"]
        question = data.get("question", data.get("sent1", "N/A"))[:200]
        print(f"\n[{item['original_index']}] matches={item['temporal_matches']}")
        print(f"  Q: {question}...")


if __name__ == "__main__":
    main()
