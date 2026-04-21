#!/usr/bin/env python3
"""
Create a low-stakes TRAM ordering dataset for patience degradation experiments.

The TRAM benchmark (Wang et al., ACL 2024) includes temporal ordering as one of
its subtasks. This script generates temporal ordering MCQs following the TRAM
format: simple "which comes first/last" questions about days, months, and
chronological sequences.

These are LOW STAKES because:
  - No domain expertise required
  - Simple factual ordering (days of week, months of year)
  - Minimal cognitive load — tests pattern-following under repetition

Output format matches the processed patience_degradation dataset schema.
"""

import json
import random
from pathlib import Path

random.seed(42)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / \
    "data" / "processed" / "patience_degradation" / "low_stakes_tram_ordering.json"


def generate_day_ordering_questions(n: int = 500) -> list[dict]:
    """Generate 'which day comes first/last in the week' MCQs."""
    questions = []
    templates = [
        ("Which of these days comes earliest in the week?", "earliest"),
        ("Which of these days comes latest in the week?", "latest"),
        ("Which day is closer to the start of the week?", "earliest"),
        ("Which day is closer to the end of the week?", "latest"),
        ("Of the following, which day occurs first in a standard week?", "earliest"),
    ]

    for i in range(n):
        template, mode = templates[i % len(templates)]
        # Pick 4 distinct days
        chosen = random.sample(DAYS, 4)
        indices = [DAYS.index(d) for d in chosen]

        if mode == "earliest":
            correct_idx = indices.index(min(indices))
        else:
            correct_idx = indices.index(max(indices))

        answer_letter = chr(65 + correct_idx)
        options = {chr(65 + j): chosen[j] for j in range(4)}

        questions.append({
            "id": f"tram_order_day_{i}",
            "question": template,
            "answer": answer_letter,
            "options": options,
            "category": "Day Ordering",
        })

    return questions


def generate_month_ordering_questions(n: int = 500) -> list[dict]:
    """Generate 'which month comes first/last in the year' MCQs."""
    questions = []
    templates = [
        ("Which of these months comes earliest in the year?", "earliest"),
        ("Which of these months comes latest in the year?", "latest"),
        ("Which month occurs first in the calendar year?", "earliest"),
        ("Of the following months, which is closest to December?", "latest"),
        ("Which of these months is closest to January?", "earliest"),
    ]

    for i in range(n):
        template, mode = templates[i % len(templates)]
        chosen = random.sample(MONTHS, 4)
        indices = [MONTHS.index(m) for m in chosen]

        if mode == "earliest":
            correct_idx = indices.index(min(indices))
        else:
            correct_idx = indices.index(max(indices))

        answer_letter = chr(65 + correct_idx)
        options = {chr(65 + j): chosen[j] for j in range(4)}

        questions.append({
            "id": f"tram_order_month_{i}",
            "question": template,
            "answer": answer_letter,
            "options": options,
            "category": "Month Ordering",
        })

    return questions


def generate_sequence_ordering_questions(n: int = 500) -> list[dict]:
    """Generate temporal sequence ordering questions."""
    questions = []

    # Decade ordering
    decades = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
    # Century ordering
    centuries = [
        "15th century", "16th century", "17th century", "18th century",
        "19th century", "20th century", "21st century",
    ]
    # Time-of-day ordering
    times_of_day = ["dawn", "morning", "noon", "afternoon", "evening", "night", "midnight"]
    # Season ordering (Northern Hemisphere calendar)
    season_months = [
        ("Winter (Dec-Feb)", 0), ("Spring (Mar-May)", 1),
        ("Summer (Jun-Aug)", 2), ("Autumn (Sep-Nov)", 3),
    ]

    pool = [
        (decades, "Decade Ordering"),
        (centuries, "Century Ordering"),
        (times_of_day, "Time-of-Day Ordering"),
    ]

    templates_earliest = [
        "Which of these comes earliest?",
        "Which occurred first chronologically?",
        "Which of these is the earliest?",
    ]
    templates_latest = [
        "Which of these comes latest?",
        "Which occurred most recently?",
        "Which of these is the latest?",
    ]

    for i in range(n):
        seq, category = pool[i % len(pool)]
        mode = "earliest" if i % 2 == 0 else "latest"

        chosen = random.sample(seq, min(4, len(seq)))
        indices = [seq.index(item) for item in chosen]

        if mode == "earliest":
            correct_idx = indices.index(min(indices))
            template = templates_earliest[i % len(templates_earliest)]
        else:
            correct_idx = indices.index(max(indices))
            template = templates_latest[i % len(templates_latest)]

        answer_letter = chr(65 + correct_idx)
        options = {chr(65 + j): chosen[j] for j in range(len(chosen))}

        questions.append({
            "id": f"tram_order_seq_{i}",
            "question": template,
            "answer": answer_letter,
            "options": options,
            "category": category,
        })

    return questions


def generate_relative_ordering_questions(n: int = 500) -> list[dict]:
    """Generate 'does X come before or after Y' questions."""
    questions = []

    all_pairs = []
    # Day pairs
    for i in range(len(DAYS)):
        for j in range(i + 1, len(DAYS)):
            all_pairs.append((DAYS[i], DAYS[j], "day"))
    # Month pairs
    for i in range(len(MONTHS)):
        for j in range(i + 1, len(MONTHS)):
            all_pairs.append((MONTHS[i], MONTHS[j], "month"))

    random.shuffle(all_pairs)

    for i in range(n):
        first, second, ptype = all_pairs[i % len(all_pairs)]

        # Question format: "Does X come before or after Y?"
        question = f"In the standard calendar, does {first} come before or after {second}?"
        options = {
            "A": f"{first} comes before {second}",
            "B": f"{first} comes after {second}",
            "C": "They occur at the same time",
            "D": "Cannot be determined",
        }
        # first always comes before second by construction
        answer = "A"

        questions.append({
            "id": f"tram_order_rel_{i}",
            "question": question,
            "answer": answer,
            "options": options,
            "category": f"Relative {ptype.capitalize()} Ordering",
        })

    return questions


def main():
    day_qs = generate_day_ordering_questions(400)
    month_qs = generate_month_ordering_questions(400)
    seq_qs = generate_sequence_ordering_questions(400)
    rel_qs = generate_relative_ordering_questions(400)

    all_examples = day_qs + month_qs + seq_qs + rel_qs
    random.shuffle(all_examples)

    # Category distribution
    cat_dist = {}
    for ex in all_examples:
        cat = ex["category"]
        cat_dist[cat] = cat_dist.get(cat, 0) + 1

    dataset = {
        "dataset": "tram_ordering",
        "source": "TRAM-Benchmark (ACL 2024) — ordering subtask",
        "stake_level": "low",
        "task_type": "temporal_ordering_mcq",
        "description": (
            "Temporal ordering reasoning — comparing chronological positions of "
            "days, months, decades, centuries, and times of day. Simple factual "
            "ordering with minimal cognitive load, replacing AG News as the "
            "low-stakes benchmark for repetition degradation experiments."
        ),
        "n_examples": len(all_examples),
        "category_distribution": cat_dist,
        "examples": all_examples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Created {len(all_examples)} examples → {OUTPUT_PATH}")
    print(f"Categories: {json.dumps(cat_dist, indent=2)}")


if __name__ == "__main__":
    main()
