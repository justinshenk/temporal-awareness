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

COMMONSENSE_DIR = Path("data/commonsense")
TRAIN_FILE = "commonsense_170k.json"


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


def commonsense_problems(split: str, n: int, skip: int = 0, seed: int | None = None,
                         data_dir: str | Path = COMMONSENSE_DIR) -> list[tuple[str, str]]:
    """Return ``n`` ``(instruction, answer)`` pairs from a commonsense split, skipping ``skip``.

    ``split`` names an eval set (``boolq`` / ``piqa`` / ``ARC-Challenge`` / … → ``{split}_test.json``)
    or ``train`` (the commonsense-170k file the donor was fitted on). ``seed`` is accepted for task
    registry signature parity and deliberately **ignored**: the files are read in order, and the
    contrast cache stores *indices* into this scan, so a seed-dependent order would misindex every
    later phase.
    """
    directory = Path(data_dir)
    path = directory / (TRAIN_FILE if split == "train" else f"{split}_test.json")
    if not path.exists():
        available = sorted(p.name for p in directory.glob("*.json")) if directory.exists() else []
        raise FileNotFoundError(
            f"no commonsense split {split!r} at {path} (available in {directory}: {available}); "
            f"run scripts.attribution.download_commonsense_data")
    data = load_commonsense_json(path)
    return [(item["instruction"], item["answer"]) for item in data[skip:skip + n]]


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
