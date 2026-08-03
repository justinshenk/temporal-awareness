"""Shared model loading, task data, CoT generation, and accuracy eval for the
primal-ridge attribution scripts (collect / fit / steer).

The procedure apparatus was written against GSM8K; :data:`TASKS` generalizes it to a second
multi-step procedure (MuSiQue open-book multi-hop QA) for the generality test, by naming the
four places a task enters: how problems are loaded, how a problem becomes a prompt, how a
completion is scored, and how gold is displayed. GSM8K stays the default everywhere, so the
existing drivers and their committed results are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.attribution.gram_accumulator import GramAccumulator
from src.probes.attribution.gsm8k_prompts import (
    extract_pred_number,
    gsm8k_gold_answer,
    metamath_prompt,
    numeric_match,
)
from src.probes.attribution.multihop_data import multihop_problems
from src.probes.attribution.multihop_prompts import (
    answer_match,
    extract_pred_answer,
    multihop_prompt_from_instruction,
)


def load_base_and_lora(cfg) -> tuple:
    """Load tokenizer, base model, and the LoRA-wrapped model on one base instance.

    ``base`` and the adapter share weights/layers, so a single ``PerTokenResidualCapture``
    on ``base`` sees both forwards; toggle with ``lora.disable_adapter()``.
    """
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=cfg["device"]).eval()
    lora = PeftModel.from_pretrained(base, cfg["adapter"]).eval()
    return tok, base, lora


def gsm8k_problems(split: str, n: int, skip: int = 0) -> list[tuple[str, float]]:
    """Return ``n`` (question, gold_float) pairs from a GSM8K split, skipping the first ``skip``."""
    ds = load_dataset("gsm8k", "main", split=split)
    out = []
    for i in range(skip, min(skip + n, len(ds))):
        out.append((ds[i]["question"], gsm8k_gold_answer(ds[i]["answer"])))
    return out


def gsm8k_demos(n: int, split: str = "train", skip: int = 0) -> list[tuple[str, str]]:
    """Return ``n`` (question, raw_answer_field) demonstration pairs for few-shot ICL prompts."""
    ds = load_dataset("gsm8k", "main", split=split)
    return [(ds[i]["question"], ds[i]["answer"]) for i in range(skip, min(skip + n, len(ds)))]


@dataclass(frozen=True)
class TaskSpec:
    """The four seams by which a multi-step procedure enters the attribution apparatus.

    ``problems(split, n, skip, seed)`` returns ``(question, gold)`` pairs, where ``question`` is
    whatever ``prompt`` consumes (a bare GSM8K question; a MuSiQue open-book instruction with the
    passages already inlined) and ``gold`` is whatever ``score`` consumes (a float; a dict with
    answer + aliases). Drivers stay task-agnostic by going through this object.
    """

    name: str
    problems: Callable[..., list[tuple[str, Any]]]
    prompt: Callable[[str], str]
    score: Callable[[str, Any], bool]
    format_gold: Callable[[Any], str]


def _gsm8k_score(completion: str, gold: float) -> bool:
    return bool(numeric_match(extract_pred_number(completion), gold))


def _multihop_score(completion: str, gold: dict) -> bool:
    return bool(answer_match(extract_pred_answer(completion), gold["answer"], gold.get("aliases", ())))


def _multihop_problems(split: str, n: int, skip: int = 0, seed: int | None = None):
    """MuSiQue problems in the driver contract; ``seed`` must match the contrast-set build."""
    return multihop_problems(split, n, skip=skip, seed=seed)


def _gsm8k_problems_seeded(split: str, n: int, skip: int = 0, seed: int | None = None):
    """GSM8K is read in file order; ``seed`` is accepted for signature parity and ignored."""
    return gsm8k_problems(split, n, skip=skip)


TASKS: dict[str, TaskSpec] = {
    "gsm8k": TaskSpec(
        name="gsm8k",
        problems=_gsm8k_problems_seeded,
        prompt=metamath_prompt,
        score=_gsm8k_score,
        format_gold=lambda g: f"{g:g}",
    ),
    "multihop": TaskSpec(
        name="multihop",
        problems=_multihop_problems,
        prompt=multihop_prompt_from_instruction,
        score=_multihop_score,
        format_gold=lambda g: str(g["answer"]),
    ),
}


def get_task(name: str = "gsm8k") -> TaskSpec:
    """Look up a task spec by name, failing loudly on an unknown task."""
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known tasks: {sorted(TASKS)}")
    return TASKS[name]


def prompt_token_ids(tokenizer, question: str, device, task: str | TaskSpec = "gsm8k") -> torch.Tensor:
    spec = task if isinstance(task, TaskSpec) else get_task(task)
    ids = tokenizer(spec.prompt(question), return_tensors="pt").input_ids
    return ids.to(device)


@torch.no_grad()
def generate_cot_ids(model, tokenizer, question: str, device, max_new: int,
                     task: str | TaskSpec = "gsm8k") -> tuple[torch.Tensor, int]:
    """Greedy-generate a CoT; return ``(full_ids (1,L), prompt_len)`` for teacher-forcing."""
    prompt_ids = prompt_token_ids(tokenizer, question, device, task)
    out = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return out, prompt_ids.shape[1]


@torch.no_grad()
def manifold_bases(acc_dir, layers, k: int, device, which: str = "base") -> dict[int, torch.Tensor]:
    """Top-``k`` manifold basis per layer from the stored accumulators (float32 columns).

    ``which`` selects the token-second-moment manifold: ``base`` (Σaaᵀ), ``lora``
    (Σ(a+δ)(a+δ)ᵀ), or ``union`` (top-k of base⊕lora). Used as the projection subspace for
    the steering manifold probes. Returns ``{layer: V (d, k)}``.
    """
    bases = {}
    for l in layers:
        acc = GramAccumulator.from_state_dict(torch.load(Path(acc_dir) / f"train_L{l}.pt"), device=device)
        bases[l] = acc.manifold_basis(k, which).to(torch.float32)
    return bases


@torch.no_grad()
def task_accuracy(model, tokenizer, problems: list[tuple[str, Any]], device, max_new: int,
                  task: str | TaskSpec = "gsm8k") -> float:
    """Greedy-generate from the task's prompt and score parsed answers against gold."""
    spec = task if isinstance(task, TaskSpec) else get_task(task)
    correct = 0
    for question, gold in problems:
        prompt_ids = prompt_token_ids(tokenizer, question, device, spec)
        out = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        correct += spec.score(text, gold)
    return correct / len(problems)


def gsm8k_accuracy(model, tokenizer, problems: list[tuple[str, float]], device, max_new: int) -> float:
    """GSM8K accuracy — the task-fixed alias kept for the existing arithmetic drivers."""
    return task_accuracy(model, tokenizer, problems, device, max_new, "gsm8k")


@torch.no_grad()
def build_contrast_set(base, lora, tok, problems, device, max_new, cache_path: Path,
                       task: str | TaskSpec = "gsm8k"):
    """Greedy-eval base and donor on one prompt; keep base-wrong / donor-right problems.

    That set is the recoverable budget every later phase measures against, so it is cached as
    ``{indices, base_acc, lora_acc, n_eval}`` and reloaded verbatim by the oracle/ladder drivers.
    Returns ``(indices, base_acc, lora_acc)``.
    """
    spec = task if isinstance(task, TaskSpec) else get_task(task)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        print(f"[contrast] loaded {len(cached['indices'])} problems from {cache_path}", flush=True)
        return cached["indices"], cached["base_acc"], cached["lora_acc"]

    indices, base_ok, lora_ok = [], 0, 0
    for i, (question, gold) in enumerate(problems):
        prompt_ids = prompt_token_ids(tok, question, device, spec)
        with lora.disable_adapter():
            b_out = base.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        l_out = lora.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        b_ok = spec.score(tok.decode(b_out[0][prompt_ids.shape[1]:], skip_special_tokens=True), gold)
        l_ok = spec.score(tok.decode(l_out[0][prompt_ids.shape[1]:], skip_special_tokens=True), gold)
        base_ok += b_ok
        lora_ok += l_ok
        if l_ok and not b_ok:
            indices.append(i)
        print(f"  [{i+1}/{len(problems)}] base_ok={b_ok} lora_ok={l_ok} "
              f"({'KEEP' if (l_ok and not b_ok) else 'skip'})", flush=True)

    base_acc, lora_acc = base_ok / len(problems), lora_ok / len(problems)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"indices": indices, "base_acc": base_acc, "lora_acc": lora_acc, "n_eval": len(problems)},
        indent=2))
    print(f"[contrast] {len(indices)} base-fail/donor-solve; base={base_acc:.3f} donor={lora_acc:.3f}",
          flush=True)
    return indices, base_acc, lora_acc


def load_contrast(cfg, task: str | TaskSpec = "gsm8k") -> list[tuple[str, Any]]:
    """Rehydrate the cached contrast set into ``(question, gold)`` tuples.

    The cache stores indices into a deterministic scan (``build_contrast_set``), so the scan is
    re-materialized with the task's own loader at the cached ``n_eval`` — the seed must match the
    one the cache was built with (both come from ``cfg['seed']``).
    """
    spec = task if isinstance(task, TaskSpec) else get_task(task)
    out = cfg["output"]
    cache = Path(out.get("contrast_json") or Path(out["steer_json"]).parent / "lockstep_contrast_set.json")
    meta = json.loads(cache.read_text())
    scan = spec.problems(cfg["eval"]["split"], meta["n_eval"], skip=0, seed=cfg["seed"])
    return [tuple(scan[i]) for i in meta["indices"]]
