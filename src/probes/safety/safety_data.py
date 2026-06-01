"""Harmful / harmless prompt sources for refusal-direction analysis.

Harmful prompts: ``mlabonne/harmful_behaviors`` (ungated AdvBench mirror, 520 items).
Harmless prompts: ``tatsu-lab/alpaca`` instructions with no extra input field.

The dataset-reading is thin; the field-extraction is pure and unit-tested.
"""

from __future__ import annotations

from datasets import load_dataset

HARMFUL_DATASET = "mlabonne/harmful_behaviors"
HARMLESS_DATASET = "tatsu-lab/alpaca"


def harmful_texts(rows) -> list[str]:
    """Pull harmful instruction strings from AdvBench-style rows (``text`` field)."""
    return [r["text"].strip() for r in rows if r.get("text", "").strip()]


def harmless_instructions(rows) -> list[str]:
    """Pull standalone Alpaca instructions (no ``input`` context) — harmless prompts."""
    return [
        r["instruction"].strip()
        for r in rows
        if r.get("instruction", "").strip() and not r.get("input", "").strip()
    ]


def load_harmful() -> list[str]:
    return harmful_texts(load_dataset(HARMFUL_DATASET, split="train"))


def load_harmless() -> list[str]:
    return harmless_instructions(load_dataset(HARMLESS_DATASET, split="train"))
