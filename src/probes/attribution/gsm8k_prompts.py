"""MetaMath prompt format + GSM8K answer scoring (Llama-2 base has no chat template).

Faithful to meta-math/MetaMath ``eval_gsm8k.py`` so steered/base/LoRA accuracies are
comparable to the published numbers: the Alpaca instruction prompt ends with
``### Response: Let's think step by step.`` (the model continues the CoT from there),
and the predicted number is parsed from the ``The answer is: `` tail with fraction support.
"""

from __future__ import annotations

import re
from fractions import Fraction

METAMATH_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
)

_NUM_RE = re.compile(r"[\-+]?\d*[\.,/]?\d+")


def metamath_prompt(question: str) -> str:
    """The exact MetaMath GSM8K eval prompt for one question (CoT tokens follow this prefix)."""
    return METAMATH_TEMPLATE.format(instruction=question)


def gsm8k_gold_answer(answer_field: str) -> float:
    """Parse the gold final answer from a GSM8K ``answer`` field (text after ``####``)."""
    tail = answer_field.split("#### ")[1].strip()
    return float(tail.replace(",", ""))


def extract_pred_number(completion: str) -> float | None:
    """Parse the model's predicted number from the ``The answer is: `` tail (MetaMath-faithful)."""
    parts = completion.split("The answer is: ")
    if len(parts) <= 1:
        return None
    match = _NUM_RE.search(parts[-1].strip())
    if not match:
        return None
    token = match.group()
    if "/" in token:
        try:
            return float(Fraction(token))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def numeric_match(pred: float | None, gold: float, tol: float = 1e-4) -> bool:
    """True iff a prediction was parsed and matches the gold answer within ``tol``."""
    return pred is not None and abs(pred - gold) < tol
