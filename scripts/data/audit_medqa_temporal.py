#!/usr/bin/env python3
"""
Quality audit of MedQA temporal-filtered dataset.

Problem: The current filter catches incidental temporal keywords like
"emergency" (71% of examples) and "acute" (diagnostic labels, not temporal
reasoning). This script:

1. Categorizes examples by temporal reasoning quality
2. Produces a cleaned high-quality subset for high-stakes experiments
3. Outputs stats for the paper

Usage:
    python scripts/data/audit_medqa_temporal.py
"""

import json
import re
import random
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filtered_temporal"

# Keywords that are almost always incidental (not requiring temporal reasoning)
INCIDENTAL_KEYWORDS = {
    "emergency",      # "presents to the emergency department" - just a setting
    "stat",           # often "stat" as in immediately, but rarely the focus
}

# Keywords that strongly suggest actual temporal reasoning in the question
STRONG_TEMPORAL_KEYWORDS = {
    "follow-up", "follow up",
    "long-term", "long term",
    "short-term", "short term",
    "prognosis",
    "progression",
    "screening",
    "prevention", "preventive",
    "prophylaxis",
    "palliative",
    "end-of-life", "end of life",
    "lifelong",
    "recurrence",
    "relapse",
    "remission",
}

# Keywords that need context - could be temporal reasoning or just labels
CONTEXT_DEPENDENT = {
    "acute",        # "acute pancreatitis" (label) vs "acute vs chronic management" (temporal)
    "chronic",      # same issue
    "progressive",  # "progressive disease" (description) vs "how to manage progression" (temporal)
    "immediate",    # "immediate treatment" could go either way
    "delayed",      # similar
    "gradual",      # usually descriptive
    "sudden onset", # usually descriptive
    "rapid onset",  # usually descriptive
    "urgent",       # could be setting or reasoning
    "terminal",     # could be diagnosis or planning
    "life-threatening",  # usually descriptive
    "insidious",    # descriptive
}


def classify_example(example):
    """
    Classify a MedQA example by temporal reasoning quality.

    Returns:
        "strong"  - Question clearly requires temporal reasoning
        "moderate" - Has temporal elements but may not require temporal reasoning
        "weak"    - Temporal keywords are incidental (setting/labels)
    """
    matches = set(m.lower() for m in example.get("temporal_matches", []))
    question = example.get("data", {}).get("question", "").lower()

    # Check if ONLY incidental keywords matched
    non_incidental = matches - INCIDENTAL_KEYWORDS
    if not non_incidental:
        return "weak"

    # Check for strong temporal keywords
    strong_matches = non_incidental & STRONG_TEMPORAL_KEYWORDS
    if strong_matches:
        # Extra check: does the question actually ASK about temporal aspects?
        temporal_question_patterns = [
            r"next\s+(?:step|best|appropriate)",
            r"most\s+(?:appropriate|likely)\s+next",
            r"long[- ]term\s+(?:management|treatment|complication|outcome|prognosis)",
            r"follow[- ]up",
            r"when\s+should",
            r"how\s+(?:long|often|soon)",
            r"prognosis",
            r"screening",
            r"prevention",
            r"prophylaxis",
            r"management\s+plan",
            r"treatment\s+plan",
            r"most\s+likely\s+(?:develop|progress|outcome)",
            r"risk\s+of\s+(?:developing|recurrence)",
        ]
        for pattern in temporal_question_patterns:
            if re.search(pattern, question):
                return "strong"
        return "moderate"

    # Only context-dependent keywords
    context_matches = non_incidental & CONTEXT_DEPENDENT
    if context_matches:
        # Check if the question involves temporal contrast
        contrast_patterns = [
            r"acute\s+(?:vs|versus|or)\s+chronic",
            r"chronic\s+(?:vs|versus|or)\s+acute",
            r"immediate\s+(?:vs|versus|or)\s+(?:long|delayed)",
            r"next\s+(?:step|best|appropriate)",
            r"most\s+(?:appropriate|likely)\s+next",
            r"management",
            r"treatment\s+(?:plan|option|approach)",
        ]
        for pattern in contrast_patterns:
            if re.search(pattern, question):
                return "moderate"
        return "weak"

    return "weak"


def main():
    # Load data
    medqa_path = DATA_DIR / "medqa_temporal_filtered.json"
    print(f"Loading {medqa_path}...")
    with open(medqa_path) as f:
        data = json.load(f)

    examples = data["examples"]
    print(f"Total examples: {len(examples)}")

    # Classify all examples
    print("\nClassifying examples by temporal reasoning quality...\n")
    classifications = {"strong": [], "moderate": [], "weak": []}

    for ex in examples:
        quality = classify_example(ex)
        classifications[quality].append(ex)

    print("=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    for level, exs in classifications.items():
        pct = len(exs) / len(examples) * 100
        print(f"  {level:>10}: {len(exs):>4} ({pct:.1f}%)")

    # Show keyword distribution per class
    for level in ["strong", "moderate", "weak"]:
        exs = classifications[level]
        if not exs:
            continue
        all_kw = []
        for ex in exs:
            all_kw.extend(m.lower() for m in ex.get("temporal_matches", []))
        kw_dist = Counter(all_kw).most_common(10)
        print(f"\n  Top keywords in '{level}':")
        for kw, count in kw_dist:
            print(f"    {kw}: {count}")

    # Show sample examples from each class
    print("\n" + "=" * 60)
    print("SAMPLE EXAMPLES")
    print("=" * 60)

    for level in ["strong", "moderate", "weak"]:
        exs = classifications[level]
        if not exs:
            continue
        print(f"\n--- {level.upper()} ({len(exs)} total) ---")
        samples = random.sample(exs, min(3, len(exs)))
        for s in samples:
            q = s["data"]["question"]
            matches = s["temporal_matches"]
            print(f"\n  Matches: {matches}")
            print(f"  Q: {q[:200]}...")
            print(f"  A: {s['data']['answer']}")

    # Build cleaned high-quality subset
    print("\n" + "=" * 60)
    print("BUILDING CLEANED SUBSET")
    print("=" * 60)

    # Use strong + moderate examples
    cleaned = classifications["strong"] + classifications["moderate"]
    print(f"  Strong + Moderate: {len(cleaned)} examples")

    # If we have enough, great. If not, note it.
    if len(cleaned) >= 100:
        print(f"  => Sufficient for high-stakes experiments (need ~100-200)")
    else:
        print(f"  => WARNING: Only {len(cleaned)} examples. May need to loosen criteria.")
        print(f"     Consider adding select 'weak' examples with manual review.")

    # Save cleaned subset
    cleaned_output = {
        "source_dataset": "GBaker/MedQA-USMLE-4-options (temporal-filtered, quality-audited)",
        "source_url": "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options",
        "description": "Quality-audited MedQA temporal reasoning subset. Filtered to examples that "
                       "require genuine temporal reasoning (treatment timing, prognosis, follow-up, "
                       "acute vs chronic management) rather than incidental temporal keywords.",
        "original_filtered_count": len(examples),
        "audit_results": {
            "strong": len(classifications["strong"]),
            "moderate": len(classifications["moderate"]),
            "weak": len(classifications["weak"]),
        },
        "total_examples": len(cleaned),
        "stake_level": "high",
        "examples": cleaned,
    }

    out_path = DATA_DIR / "medqa_temporal_audited.json"
    with open(out_path, 'w') as f:
        json.dump(cleaned_output, f, indent=2)
    print(f"\n  Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Also save a small random sample for manual inspection
    sample_size = min(50, len(cleaned))
    manual_sample = random.sample(cleaned, sample_size)
    sample_output = {
        "description": "Random sample of audited MedQA examples for manual inspection",
        "sample_size": sample_size,
        "from_total": len(cleaned),
        "examples": manual_sample,
    }

    sample_path = DATA_DIR / "medqa_manual_review_sample.json"
    with open(sample_path, 'w') as f:
        json.dump(sample_output, f, indent=2)
    print(f"  Manual review sample: {sample_path} ({sample_size} examples)")

    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"  Original filtered:  {len(examples)}")
    print(f"  After quality audit: {len(cleaned)} (strong: {len(classifications['strong'])}, moderate: {len(classifications['moderate'])})")
    print(f"  Removed as weak:    {len(classifications['weak'])}")
    print(f"  Manual review sample saved ({sample_size} examples)")
    print(f"\n  For NeurIPS: {len(cleaned)} high-quality examples is {'sufficient' if len(cleaned) >= 100 else 'INSUFFICIENT — needs attention'}")


if __name__ == "__main__":
    random.seed(42)
    main()
