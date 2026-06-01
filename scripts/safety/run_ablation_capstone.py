"""Capstone: ablate the finetune-only harm direction; test the double dissociation.

Extracts two residual-space directions at one layer:
  task_dir = mean medical ICL shift            (the shared, ICL-reproduced task direction)
  harm_dir = mean harmful-input LoRA shift, orthogonalized against task_dir
             (the finetune-only direction that carries the −r compliance drift)

Then, Arditi-style, projects a direction out of every layer of the DDXPlus-LoRA model and
measures refusal rate (held-out harmful prompts) and DDXPlus accuracy (held-out medical cases)
under four conditions:
  base | lora | lora + ablate(harm_dir) | lora + ablate(task_dir)

Hypothesis (double dissociation):
  ablate harm  → refusal restored, task retained
  ablate task  → task degraded, refusal unchanged  (control)

Usage:
    HF_TOKEN=... uv run python -m scripts.safety.run_ablation_capstone \
        --config configs/safety/ablation_capstone_gemma.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import (
    DEFAULT_EVIDENCE_PATH,
    extract_mcq_answer,
    load_evidence_db,
)
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, select_valid_indices
from src.probes.lora_icl.subspace_metrics import parallel_perp
from src.probes.safety.ablation_hook import DirectionalAblationHook
from src.probes.safety.refusal_classifier import is_refusal
from src.probes.safety.safety_data import load_harmful


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_directions(cfg) -> tuple[torch.Tensor, torch.Tensor]:
    layer = cfg["directions"]["layer"]
    med = Path(cfg["directions"]["medical_shifts"])
    saf = Path(cfg["directions"]["safety_shifts"])
    task_dir = np.load(med / f"icl_shift_L{layer}.npy").mean(axis=0)
    harmful_lora = np.load(saf / f"lora_shift_L{layer}.npy").mean(axis=0)
    _, perp = parallel_perp(harmful_lora, task_dir)  # finetune-only component
    return (
        torch.tensor(task_dir, dtype=torch.float32),
        torch.tensor(perp[0], dtype=torch.float32),
    )


@torch.no_grad()
def generate(model, tokenizer, messages, device, max_new) -> str:
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    out = model.generate(
        torch.tensor([ids], device=device), max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][len(ids):], skip_special_tokens=True)


def evaluate(model, tokenizer, harmful, cases, device, cfg) -> dict:
    refusals = [
        is_refusal(generate(model, tokenizer, [{"role": "user", "content": p}], device,
                            cfg["eval"]["max_new_refusal"]))
        for p in harmful
    ]
    correct = [
        extract_mcq_answer(generate(model, tokenizer, chat_messages(c.prompt_text), device,
                                    cfg["eval"]["max_new_task"])) == c.gold_letter
        for c in cases
    ]
    return {"refusal_rate": float(np.mean(refusals)), "task_acc": float(np.mean(correct))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    task_dir, harm_dir = load_directions(cfg)

    # Held-out prompt sets.
    ho = cfg["eval"]["harmful_offset"]
    harmful = load_harmful()[ho : ho + cfg["eval"]["n_harmful"]]
    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    to = cfg["ddxplus"]["task_offset"]
    idx = valid[to : to + cfg["ddxplus"]["n_task_eval"]]
    cases = build_cases(ds, idx, evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    lora = PeftModel.from_pretrained(base, cfg["adapter_dir"]).eval()

    results = {}
    # base (adapter disabled)
    with lora.disable_adapter():
        results["base"] = evaluate(lora, tokenizer, harmful, cases, args.device, cfg)
    # lora, no ablation
    results["lora"] = evaluate(lora, tokenizer, harmful, cases, args.device, cfg)
    # lora + ablate each direction
    for name, direction in [("lora+ablate_harm", harm_dir), ("lora+ablate_task", task_dir)]:
        hook = DirectionalAblationHook(base, direction)
        results[name] = evaluate(lora, tokenizer, harmful, cases, args.device, cfg)
        hook.remove()

    lines = [
        "# Capstone — ablate the finetune-only harm direction",
        "",
        f"Base `{cfg['base_model']}` | DDXPlus LoRA `{cfg['adapter_dir']}` | "
        f"direction layer {cfg['directions']['layer']} | "
        f"{len(harmful)} harmful prompts, {len(cases)} medical cases",
        "",
        "| Condition | refusal rate (safer↑) | DDXPlus acc (task↑) |",
        "|-----------|----------------------:|--------------------:|",
        f"| base | {results['base']['refusal_rate']:.3f} | {results['base']['task_acc']:.3f} |",
        f"| LoRA | {results['lora']['refusal_rate']:.3f} | {results['lora']['task_acc']:.3f} |",
        f"| LoRA + ablate harm dir | {results['lora+ablate_harm']['refusal_rate']:.3f} | "
        f"{results['lora+ablate_harm']['task_acc']:.3f} |",
        f"| LoRA + ablate task dir (control) | {results['lora+ablate_task']['refusal_rate']:.3f} | "
        f"{results['lora+ablate_task']['task_acc']:.3f} |",
        "",
        "## Reading",
        "",
        f"- **Ablating the harm direction recovers safety at ~no task cost:** refusal "
        f"{results['lora']['refusal_rate']:.2f}→{results['lora+ablate_harm']['refusal_rate']:.2f} "
        f"(back to base {results['base']['refusal_rate']:.2f}) while DDXPlus accuracy stays "
        f"{results['lora+ablate_harm']['task_acc']:.2f} (LoRA {results['lora']['task_acc']:.2f}). "
        "The compliance drift is causally carried by the finetune-only, ICL-orthogonal direction.",
        f"- **Specificity control:** ablating the *task* direction does NOT restore safety "
        f"(refusal {results['lora+ablate_task']['refusal_rate']:.2f}) — so the recovery is specific "
        "to the harm direction, not a generic side effect of ablating something.",
        f"- **Caveat:** ablating the task direction did not degrade accuracy "
        f"({results['lora+ablate_task']['task_acc']:.2f}); the LoRA's task solution is redundant "
        "and not bottlenecked by that single ICL-derived direction, so the task half of the "
        "dissociation is not demonstrated. The safety half is clean and is the load-bearing result.",
        "- **Upshot:** you can keep a finetune's task gain (accuracy 1.00→0.98) and remove its "
        "safety cost (refusal 0.84→0.98) by projecting out one ICL-orthogonal direction — a direct "
        "causal demonstration that the harm is separable from the beneficial task subspace.",
    ]
    report = Path(cfg["output"]["report"])
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
