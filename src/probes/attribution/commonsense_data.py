"""LLM-Adapters commonsense data glue: load, format, extract, score (torch-free).

All sets (commonsense_170k train; boolq / piqa / ARC-Challenge / … eval) share one schema —
``{"instruction", "input"(empty), "output": "the correct answer is X", "answer": "X"}`` — and the
pyreft commonsense task formats prompts as literally ``"%s\\n" % instruction`` (no alpaca
wrapper). Predictions are read as the word following ``"the correct answer is"`` in the
generation, matching the paper's trigger-token extraction.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PROMPT_TEMPLATE = "%s\n"
ANSWER_TRIGGER = "the correct answer is"


def load_commonsense_json(path: str | Path) -> list[dict]:
    """The LLM-Adapters files are a single JSON list of items."""
    return json.loads(Path(path).read_text())


def format_prompt(item: dict) -> str:
    return PROMPT_TEMPLATE % item["instruction"]


def format_target(item: dict) -> str:
    return item["output"]


def subset_examples(data: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic shuffled subset (not the file head — the 170k file is grouped by source)."""
    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n]


def extract_answer(text: str) -> str:
    """The word after the trigger, lowercased and stripped of punctuation; '' when absent."""
    lowered = text.lower()
    idx = lowered.find(ANSWER_TRIGGER)
    if idx < 0:
        return ""
    rest = lowered[idx + len(ANSWER_TRIGGER):].split()
    return rest[0].strip(".,;:!?\"'") if rest else ""


def score_predictions(preds: list[str], golds: list[str]) -> float:
    """Exact-match accuracy against the gold ``answer`` fields."""
    if len(preds) != len(golds):
        raise ValueError(f"length mismatch: {len(preds)} preds vs {len(golds)} golds")
    return sum(p == g.lower() for p, g in zip(preds, golds)) / len(golds)
