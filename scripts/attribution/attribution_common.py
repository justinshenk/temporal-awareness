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
from typing import Any, Callable, Sequence

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.attribution.chain_token_roles import (
    ROLE_COMPUTED,
    ROLE_COPIED_DIGIT,
    ROLE_FINAL_ANSWER,
    ROLE_HOP_ANSWER,
    ROLE_SCAFFOLD,
    ROLE_SUB_QUESTION,
    gsm8k_token_roles,
    multihop_token_roles,
)
from src.probes.attribution.gram_accumulator import GramAccumulator
from src.probes.attribution.gsm8k_prompts import (
    extract_pred_number,
    gsm8k_gold_answer,
    metamath_prompt,
    numeric_match,
)
from src.probes.attribution.multihop_data import gold_chain, multihop_problems
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
class GoldLensSpec:
    """How a task feeds the gold-token teacher-forced lens (plan-vs-execute, P4).

    ``gold_chain(gold)`` returns the supervised continuation to teacher-force (joined to the prompt
    verbatim, leading separator included), or is ``None`` when the task has no in-format gold chain
    and the lens must teacher-force the *donor's own* generated CoT instead — GSM8K's case, since
    its dataset CoT is not in MetaMath format. ``token_roles(tok, ids, prompt_len, gold)`` labels
    every token of the teacher-forced sequence, and ``role_classes`` names the reported groupings
    over those labels.

    ``contrasts`` lists the ``(label, class_a, class_b)`` role-class differences the verdict rests
    on. They are reported as problem-clustered bootstrap intervals rather than bare point
    estimates: tokens are not independent within a problem, so a token-level interval would badly
    overstate the evidence for a gap.
    """

    token_roles: Callable[[Any, Sequence[int], int, Any], list[dict]]
    role_classes: dict[str, Callable[[dict], bool]]
    gold_chain: Callable[[Any], str] | None = None
    contrasts: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    """The seams by which a multi-step procedure enters the attribution apparatus.

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
    lens: GoldLensSpec | None = None      # only the gold-token lens (P4) needs this seam


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


# Reported groupings for the gold-token lens. GSM8K's four reproduce the committed E1b table
# (``digit`` is the union of the two digit roles); multi-hop's split the chain into plan
# (sub-question), execution (hop answer, also split by hop index since hop >= 2 is the one that
# needs composition), the copied restatement, and format scaffold.
_GSM8K_ROLE_CLASSES: dict[str, Callable[[dict], bool]] = {
    "all": lambda r: True,
    "digit": lambda r: r["role"] in (ROLE_COMPUTED, ROLE_COPIED_DIGIT),
    "computed (result of =)": lambda r: r["role"] == ROLE_COMPUTED,
    "copied digit (not computed)": lambda r: r["role"] == ROLE_COPIED_DIGIT,
}

_MULTIHOP_ROLE_CLASSES: dict[str, Callable[[dict], bool]] = {
    "all": lambda r: True,
    "sub_question (plan)": lambda r: r["role"] == ROLE_SUB_QUESTION,
    "hop_answer (execute)": lambda r: r["role"] == ROLE_HOP_ANSWER,
    "hop_answer hop 1": lambda r: r["role"] == ROLE_HOP_ANSWER and r["hop"] == 1,
    "hop_answer hop >= 2": lambda r: r["role"] == ROLE_HOP_ANSWER and (r["hop"] or 0) >= 2,
    "final_answer (copy)": lambda r: r["role"] == ROLE_FINAL_ANSWER,
    "scaffold (format)": lambda r: r["role"] == ROLE_SCAFFOLD,
}

# The differences each task's verdict rests on, interval-estimated with problems as the
# resampling unit. GSM8K asks whether base is *better* on the tokens it must compute than on the
# chain at large (E1b's claim); multi-hop asks whether execution beats planning (H_plan vs
# H_exec), and whether the later, composed hops are harder than the first.
_GSM8K_CONTRASTS = (
    ("computed - all", "computed (result of =)", "all"),
    ("computed - copied digit", "computed (result of =)", "copied digit (not computed)"),
)

_MULTIHOP_CONTRASTS = (
    ("execute - plan", "hop_answer (execute)", "sub_question (plan)"),
    ("execute - all", "hop_answer (execute)", "all"),
    ("hop >= 2 - hop 1", "hop_answer hop >= 2", "hop_answer hop 1"),
    ("copy - execute", "final_answer (copy)", "hop_answer (execute)"),
)


TASKS: dict[str, TaskSpec] = {
    "gsm8k": TaskSpec(
        name="gsm8k",
        problems=_gsm8k_problems_seeded,
        prompt=metamath_prompt,
        score=_gsm8k_score,
        format_gold=lambda g: f"{g:g}",
        lens=GoldLensSpec(token_roles=gsm8k_token_roles, role_classes=_GSM8K_ROLE_CLASSES,
                          contrasts=_GSM8K_CONTRASTS),
    ),
    "multihop": TaskSpec(
        name="multihop",
        problems=_multihop_problems,
        prompt=multihop_prompt_from_instruction,
        score=_multihop_score,
        format_gold=lambda g: str(g["answer"]),
        lens=GoldLensSpec(token_roles=multihop_token_roles,
                          role_classes=_MULTIHOP_ROLE_CLASSES,
                          gold_chain=lambda g: gold_chain(g["decomposition"]),
                          contrasts=_MULTIHOP_CONTRASTS),
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


def gold_chain_ids(tokenizer, question: str, gold: Any, device,
                   task: str | TaskSpec = "gsm8k") -> tuple[torch.Tensor, int]:
    """Build ``prompt + gold chain + eos`` ids for teacher-forcing; return ``(full_ids, prompt_len)``.

    The join reproduces training (``multihop_data.encode_multihop``) exactly: the prompt is tokenized
    with its BOS, the chain without special tokens, and EOS is appended — so the teacher-forced
    sequence is the one the donor was supervised on.
    """
    spec = task if isinstance(task, TaskSpec) else get_task(task)
    if spec.lens.gold_chain is None:
        raise ValueError(f"task {spec.name!r} has no in-format gold chain to teacher-force")
    prompt_ids = prompt_token_ids(tokenizer, question, device, spec)
    chain_ids = tokenizer(spec.lens.gold_chain(gold), add_special_tokens=False).input_ids
    tail = torch.tensor([chain_ids + [tokenizer.eos_token_id]], device=device)
    return torch.cat([prompt_ids, tail], dim=1), prompt_ids.shape[1]


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
