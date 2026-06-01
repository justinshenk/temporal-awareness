"""Deterministic DDXPlus MCQ case construction, shared by training and extraction.

Mirrors the case logic in ``scripts/context_fatigue/run_ddxplus_mcq.py`` (load
cases, keep those whose gold pathology is in the top-N differential, shuffle the
options, record the gold letter) but as importable, testable functions so the
LoRA training set and the shift-extraction eval set are built the same way and on
disjoint index splits.

The dataset object is passed in (not loaded here) so these functions unit-test
against a tiny fake dataset with no network access.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema
from src.probes.ddxplus import OPTION_LABELS, SYSTEM_PROMPT, format_case_mcq


def chat_messages(prompt_text: str) -> list[dict[str, str]]:
    """Chat messages for one case, folding the system prompt into the user turn.

    Gemma-2's chat template has no ``system`` role, so the instruction is
    prepended to the user content. Used identically by training and extraction so
    the prediction site (final prompt token) matches across both.
    """
    return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt_text}"}]


@dataclass
class MCQCase(BaseSchema):
    """One DDXPlus multiple-choice case (clean, single-case)."""

    source_index: int = 0
    prompt_text: str = ""
    gold_letter: str = ""
    option_names: list[str] = field(default_factory=list)


def select_valid_indices(ds, n_options: int) -> list[int]:
    """Indices whose gold pathology appears in the top-``n_options`` differential."""
    valid = []
    for i in range(len(ds)):
        row = ds[i]
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        names = [d[0] for d in ddx[:n_options]]
        if row["PATHOLOGY"] in names:
            valid.append(i)
    return valid


def disjoint_split(
    indices: list[int], n_train: int, n_eval: int, seed: int
) -> tuple[list[int], list[int]]:
    """Shuffle ``indices`` and carve off non-overlapping train / eval slices."""
    if n_train + n_eval > len(indices):
        raise ValueError(
            f"n_train + n_eval ({n_train + n_eval}) exceeds available cases ({len(indices)})"
        )
    pool = list(indices)
    random.Random(seed).shuffle(pool)
    return pool[:n_train], pool[n_train : n_train + n_eval]


def icl_messages(tokenizer, fillers, final_messages, max_ctx: int, fill_target: float):
    """Build an ICL chat: as many filler DDXPlus Q&A turns as fit, then ``final_messages``.

    Args:
        tokenizer: used only to measure token budget via the chat template.
        fillers: ``MCQCase`` list providing user(case)/assistant(letter) prior turns.
        final_messages: the final turn(s) to append (already-built message dicts), e.g.
            a medical eval case or a harmful prompt as a plain user turn.
        max_ctx, fill_target: stop adding fillers once the running length exceeds
            ``max_ctx * fill_target`` tokens.
    """
    messages: list[dict[str, str]] = []
    budget = int(max_ctx * fill_target)
    for fc in fillers:
        trial = messages + chat_messages(fc.prompt_text) + [
            {"role": "assistant", "content": fc.gold_letter}
        ]
        n = len(tokenizer.apply_chat_template(trial, add_generation_prompt=False, tokenize=True))
        if n > budget:
            break
        messages = trial
    return messages + list(final_messages)


def build_case(ds, idx: int, evidence_db: dict, n_options: int, rng: random.Random) -> MCQCase:
    """Build one MCQ case at dataset index ``idx`` with options shuffled by ``rng``."""
    row = ds[idx]
    pathology = row["PATHOLOGY"]
    ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
    option_names = [d[0] for d in ddx[:n_options]]
    if pathology not in option_names:
        raise ValueError(f"index {idx}: gold {pathology!r} not in top-{n_options} differential")

    shuffled = option_names[:]
    rng.shuffle(shuffled)
    gold_pos = shuffled.index(pathology)
    gold_letter = OPTION_LABELS[gold_pos]

    prompt_text = format_case_mcq(
        row["AGE"], row["SEX"], row["INITIAL_EVIDENCE"], row["EVIDENCES"],
        evidence_db, shuffled,
    )
    return MCQCase(
        source_index=idx,
        prompt_text=prompt_text,
        gold_letter=gold_letter,
        option_names=shuffled,
    )


def build_cases(
    ds, indices: list[int], evidence_db: dict, n_options: int, seed: int
) -> list[MCQCase]:
    """Build cases for every index, with a single seeded RNG for option order."""
    rng = random.Random(seed)
    return [build_case(ds, idx, evidence_db, n_options, rng) for idx in indices]


def permute_labels(cases: list[MCQCase], seed: int) -> list[MCQCase]:
    """Shuffled-label control: reassign gold letters across cases (derangement-ish).

    The prompts are unchanged; only the supervised target letter is permuted, so a
    LoRA trained on these learns no real DDXPlus signal. Used to test whether the
    LoRA↔ICL subspace alignment is task-specific rather than a generic finetune
    artifact.
    """
    letters = [c.gold_letter for c in cases]
    random.Random(seed).shuffle(letters)
    return [
        MCQCase(
            source_index=c.source_index,
            prompt_text=c.prompt_text,
            gold_letter=letter,
            option_names=c.option_names,
        )
        for c, letter in zip(cases, letters)
    ]
