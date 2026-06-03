#!/usr/bin/env python3
"""
Organize all patience degradation datasets into a clean, consistent format.

Reads from data/filtered_temporal/ (raw downloads + audited files)
Writes to data/processed/patience_degradation/ (experiment-ready)

Each output file has:
  - metadata (source, description, stake_level, n_examples)
  - examples list with standardized fields:
      - id: unique identifier
      - question: the prompt/task for the model
      - answer: expected answer
      - options: (if MCQ) dict of options
      - category: sub-category within the dataset
"""

import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

INPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filtered_temporal"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "patience_degradation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_ag_news():
    """Low-stakes: AG News text classification."""
    print("=" * 60)
    print("LOW-STAKES: AG News")
    print("=" * 60)

    with open(INPUT_DIR / "ag_news_classification.json") as f:
        raw = json.load(f)

    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    # Use test split only (7600 examples, balanced)
    test_examples = [ex for ex in raw["examples"] if ex["split"] == "test"]

    examples = []
    for ex in test_examples:
        examples.append({
            "id": f"agnews_{ex['original_index']}",
            "question": f"Classify the following news article into one of these categories: World, Sports, Business, Sci/Tech.\n\nArticle: {ex['text']}",
            "answer": label_map[ex["label"]],
            "options": {"A": "World", "B": "Sports", "C": "Business", "D": "Sci/Tech"},
            "category": label_map[ex["label"]],
        })

    # Verify balance
    cats = Counter(ex["category"] for ex in examples)
    print(f"  Examples: {len(examples)} (test split)")
    print(f"  Categories: {dict(cats)}")

    output = {
        "dataset": "ag_news",
        "stake_level": "low",
        "task_type": "text_classification",
        "description": "4-class news topic classification. Simple, non-temporal tasks with minimal cognitive load. Baseline for probe stability under trivial repetition.",
        "n_examples": len(examples),
        "category_distribution": dict(cats),
        "examples": examples,
    }

    out_path = OUTPUT_DIR / "low_stakes_ag_news.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return len(examples)


def process_tram():
    """Medium-stakes (temporal): TRAM arithmetic MCQ only."""
    print("\n" + "=" * 60)
    print("MEDIUM-STAKES (temporal): TRAM Arithmetic")
    print("=" * 60)

    with open(INPUT_DIR / "tram_arithmetic.json") as f:
        raw = json.load(f)

    # Keep only MCQ format (not short-answer duplicates)
    mcq_examples = [ex for ex in raw["examples"] if ex["source_file"] == "arithmetic_mcq.csv"]

    examples = []
    for i, ex in enumerate(mcq_examples):
        examples.append({
            "id": f"tram_arith_{i}",
            "question": ex["Question"],
            "answer": ex["Answer"],  # e.g. "C"
            "options": {
                "A": ex.get("Option A", ""),
                "B": ex.get("Option B", ""),
                "C": ex.get("Option C", ""),
                "D": ex.get("Option D", ""),
            },
            "category": ex.get("Category", "unknown"),
        })

    cats = Counter(ex["category"] for ex in examples)
    print(f"  Examples: {len(examples)} (MCQ only, removed SAQ duplicates)")
    print("  Categories:")
    for cat, count in cats.most_common():
        print(f"    {cat}: {count}")

    output = {
        "dataset": "tram_arithmetic",
        "source": "TRAM-Benchmark (ACL 2024)",
        "stake_level": "medium",
        "task_type": "temporal_arithmetic_mcq",
        "description": "Temporal arithmetic reasoning — duration computation, time differences, date calculations, time zone conversions. Tests explicit temporal computation under repetition.",
        "n_examples": len(examples),
        "category_distribution": dict(cats.most_common()),
        "examples": examples,
    }

    out_path = OUTPUT_DIR / "medium_stakes_tram_arithmetic.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return len(examples)


def process_mbpp():
    """Medium-stakes (non-temporal control): MBPP code generation."""
    print("\n" + "=" * 60)
    print("MEDIUM-STAKES (control): MBPP Code Generation")
    print("=" * 60)

    with open(INPUT_DIR / "mbpp_code_review.json") as f:
        raw = json.load(f)

    # Use test split (500 examples) - standard evaluation set
    test_examples = [ex for ex in raw["examples"] if ex["split"] == "test"]

    examples = []
    for ex in test_examples:
        tests_str = "\n".join(ex["test_list"]) if isinstance(ex["test_list"], list) else str(ex["test_list"])
        examples.append({
            "id": f"mbpp_{ex['task_id']}",
            "question": f"{ex['text']}\n\nTest cases:\n{tests_str}",
            "answer": ex["code"],
            "options": None,  # open-ended code generation
            "category": "code_generation",
        })

    print(f"  Examples: {len(examples)} (test split)")

    output = {
        "dataset": "mbpp",
        "source": "google-research-datasets/mbpp",
        "stake_level": "medium",
        "task_type": "code_generation",
        "description": "Python code generation problems. Cognitively demanding but non-temporal control: monitors whether temporal representations drift under sustained cognitive load without temporal content.",
        "n_examples": len(examples),
        "examples": examples,
    }

    out_path = OUTPUT_DIR / "medium_stakes_mbpp_code.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return len(examples)


def process_medqa():
    """High-stakes: Curated MedQA temporal reasoning."""
    print("\n" + "=" * 60)
    print("HIGH-STAKES: MedQA Temporal (curated)")
    print("=" * 60)

    with open(INPUT_DIR / "medqa_temporal_curated.json") as f:
        raw = json.load(f)

    examples = []
    for i, ex in enumerate(raw["examples"]):
        examples.append({
            "id": f"medqa_temporal_{i}",
            "question": ex["data"]["question"],
            "answer": ex["data"]["answer"],
            "answer_idx": ex["data"].get("answer_idx", ""),
            "options": ex["data"].get("options", {}),
            "category": ex.get("temporal_reasoning_category", "unknown"),
        })

    cats = Counter(ex["category"] for ex in examples)
    print(f"  Examples: {len(examples)} (triple-filtered)")
    print("  Temporal reasoning categories:")
    for cat, count in cats.most_common():
        print(f"    {cat}: {count}")

    output = {
        "dataset": "medqa_temporal_curated",
        "source": "GBaker/MedQA-USMLE-4-options",
        "stake_level": "high",
        "task_type": "medical_temporal_reasoning_mcq",
        "description": "Triple-filtered MedQA subset (10178 → 993 → 452 → 128). Every example requires genuine temporal horizon reasoning: choosing between immediate vs long-term interventions, screening timing, follow-up planning, prognosis assessment, and prevention strategies. Degradation here maps to clinically meaningful errors.",
        "filtering_pipeline": raw.get("filtering_pipeline", []),
        "n_examples": len(examples),
        "category_distribution": dict(cats.most_common()),
        "examples": examples,
    }

    out_path = OUTPUT_DIR / "high_stakes_medqa_temporal.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return len(examples)


if __name__ == "__main__":
    print("Organizing all datasets for patience degradation experiments\n")

    n_ag = process_ag_news()
    n_tram = process_tram()
    n_mbpp = process_mbpp()
    n_medqa = process_medqa()

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"  Low-stakes:    AG News           {n_ag:>6} examples  (test split)")
    print(f"  Medium-stakes: TRAM Arithmetic   {n_tram:>6} examples  (MCQ only)")
    print(f"  Medium-stakes: MBPP Code         {n_mbpp:>6} examples  (test split)")
    print(f"  High-stakes:   MedQA Temporal    {n_medqa:>6} examples  (curated)")
    print(f"\n  All saved to: {OUTPUT_DIR}/")
    print("  Files:")
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        print(f"    {f.name} ({f.stat().st_size / 1024:.1f} KB)")
