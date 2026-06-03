#!/usr/bin/env python3
"""
Download low-stakes dataset for temporal awareness experiments:
AG News (4-class topic classification)

Usage (make sure venv is active):
    source .venv/bin/activate
    python scripts/data/download_low_stakes.py
"""

import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filtered_temporal"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_ag_news():
    """Download AG News dataset via HuggingFace datasets library."""
    print("=" * 60)
    print("Downloading AG News dataset...")
    print("=" * 60)

    print("  Loading from HuggingFace... ", end="", flush=True)
    ds = load_dataset("ag_news")
    print("done")

    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    examples = []
    for split_name in ds:
        for i, item in enumerate(ds[split_name]):
            examples.append({
                "original_index": i,
                "split": split_name,
                "text": item["text"],
                "label": item["label"],
                "label_name": label_map[item["label"]],
            })

    print(f"  Total examples: {len(examples)}")
    print(f"  Splits: {list(ds.keys())}")
    print(f"  Labels: {label_map}")

    # Show distribution
    from collections import Counter
    label_dist = Counter(ex["label_name"] for ex in examples)
    print(f"  Label distribution: {dict(label_dist)}")

    if examples:
        print("\n  Sample:")
        print(f"    text: {examples[0]['text'][:120]}...")
        print(f"    label: {examples[0]['label_name']}")

    output = {
        "source_dataset": "ag_news",
        "source_url": "https://huggingface.co/datasets/ag_news",
        "description": "AG News topic classification (4-class) - simple non-temporal classification for low-stakes baseline",
        "total_examples": len(examples),
        "stake_level": "low",
        "label_map": label_map,
        "examples": examples,
    }

    out_path = DATA_DIR / "ag_news_classification.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return len(examples)


if __name__ == "__main__":
    print("Downloading low-stakes dataset\n")
    n = download_ag_news()
    print(f"\nDONE — AG News: {n} examples")
