"""MuSiQue open-book multi-hop QA data: HF loader, training-item builder, response-only encode.

Open-book: the gold *supporting* passages are placed in the instruction (the analogue of GSM8K's
in-problem numbers), so the LoRA learns multi-hop *composition* over given facts, not parametric
recall. Bridges the MuSiQue HF schema (``dgslibisey/MuSiQue``) to two consumers:

* the LoRA training pipeline — :func:`encode_multihop` produces prompt+target token ids with the
  prompt length kept, fed through ``train_loreft_commonsense.collate_left_padded`` for response-only
  CE labels (byte-identical masking to the commonsense/LoReFT arms);
* the attribution drivers — :func:`multihop_problems` yields ``(instruction_string, gold)`` pairs
  where ``instruction_string`` carries the passages and is wrapped by
  ``multihop_prompt_from_instruction`` (one-arg, MetaMath-isomorphic), so the drivers treat a
  multi-hop problem exactly like a GSM8K question. ``gold`` is a dict (answer + aliases) because
  :func:`multihop_prompts.answer_match` scores against aliases, unlike GSM8K's numeric match.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from datasets import load_dataset

from src.probes.attribution.multihop_prompts import (
    build_instruction,
    format_multihop_solution,
    multihop_prompt_from_instruction,
)

MUSIQUE_HF = "dgslibisey/MuSiQue"


def supporting_passages(item: dict) -> List[Tuple[str, str]]:
    """Gold supporting ``(title, text)`` passages in original order (the open-book context)."""
    return [(p["title"], p["paragraph_text"]) for p in item["paragraphs"] if p.get("is_supporting")]


def problem_instruction(item: dict) -> str:
    """The driver-facing "question": the instruction body with gold supporting passages inlined."""
    return build_instruction(item["question"], supporting_passages(item))


def gold_target(item: dict) -> str:
    """Supervised CoT response: leading newline + worked chain + answer marker.

    The leading ``"\\n"`` is the join separator (prompt ends at ``Let's think step by step.``), so the
    supervised continuation matches what the model generates at inference (cf.
    ``gsm8k_prompts.metamath_fewshot_prompt``, which joins template and solution with ``"\\n"``).
    """
    return "\n" + format_multihop_solution(item["question_decomposition"])


def load_musique(split: str, n: int | None = None, skip: int = 0,
                 answerable_only: bool = True, seed: int | None = None) -> List[dict]:
    """Raw MuSiQue items: optionally answerable-only, shuffled by ``seed``, then ``[skip:skip+n]``."""
    ds = load_dataset(MUSIQUE_HF, split=split)
    items = [ds[i] for i in range(len(ds))]
    if answerable_only:
        items = [it for it in items if it.get("answerable", True)]
    if seed is not None:
        random.Random(seed).shuffle(items)
    return items[skip:(skip + n) if n is not None else None]


def encode_multihop(tok, items: List[dict], max_len: int) -> List[dict]:
    """Per item: prompt ids (with BOS) + target ids + eos, truncated; keep the prompt length.

    Mirrors ``train_loreft_commonsense.encode_examples`` but for the multi-hop prompt/target pair.
    """
    out = []
    for it in items:
        prompt = multihop_prompt_from_instruction(problem_instruction(it))
        p_ids = tok(prompt, add_special_tokens=True).input_ids
        t_ids = tok(gold_target(it), add_special_tokens=False).input_ids + [tok.eos_token_id]
        full = (p_ids + t_ids)[:max_len]
        if len(full) <= len(p_ids):                 # truncation ate the whole response
            continue
        out.append({"ids": full, "plen": len(p_ids)})
    return out


def multihop_problems(split: str, n: int, skip: int = 0,
                      seed: int | None = None) -> List[Tuple[str, dict]]:
    """``(instruction_string, gold)`` pairs for the drivers; ``gold = {"answer", "aliases"}``.

    ``instruction_string`` is the open-book instruction (passages inlined); the driver wraps it with
    ``multihop_prompt_from_instruction``. Gold is a dict (not a float) so ``answer_match`` can use the
    MuSiQue answer aliases.
    """
    items = load_musique(split, n, skip, answerable_only=True, seed=seed)
    return [(problem_instruction(it),
             {"answer": it["answer"], "aliases": list(it.get("answer_aliases", []))})
            for it in items]
