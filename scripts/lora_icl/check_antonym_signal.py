"""Step-0 gate: does the antonym task carry a real in-context signal on this model?

Measures antonym accuracy zero-shot vs k-shot vs k-shot-with-shuffled-labels (bare
``word: antonym`` format). The DDXPlus FV null happened because clean == corrupted (demos
inert). We proceed to train the antonym LoRA + extract the FV only if clean >> corrupted/zero-shot.

Usage:
    HF_TOKEN=... uv run python -m scripts.lora_icl.check_antonym_signal \
        --config configs/lora_icl/antonym_fv_gemma.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.lora_icl.antonym_data import antonym_split
from scripts.safety.run_ablation_capstone import set_seed


def prompt(demos, query_word):
    body = "".join(f"{w}: {a}\n" for w, a in demos)
    return f"{body}{query_word}:"


@torch.no_grad()
def predict(model, tokenizer, text, device, max_new):
    ids = tokenizer(text, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    return gen.strip().split()[0].strip(".,!?;:").lower() if gen.strip() else ""


def accuracy(model, tokenizer, queries, build, device, max_new):
    hits = [predict(model, tokenizer, build(w), device, max_new) == a.lower() for w, a in queries]
    return float(np.mean(hits))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    train, held = antonym_split(cfg["n_train"])
    k = cfg["k_shot"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()

    queries = held[: cfg["n_eval"]]
    demos_for = {w: rng.sample([p for p in train if p[0] != w], k) for w, _ in queries}
    shuf_for = {}
    for w, _ in queries:
        d = demos_for[w]
        labels = [a for _, a in d]
        rng.shuffle(labels)
        shuf_for[w] = list(zip([x for x, _ in d], labels))

    zs = accuracy(model, tokenizer, queries, lambda w: prompt([], w), args.device, cfg["max_new"])
    ks = accuracy(model, tokenizer, queries, lambda w: prompt(demos_for[w], w), args.device, cfg["max_new"])
    sh = accuracy(model, tokenizer, queries, lambda w: prompt(shuf_for[w], w), args.device, cfg["max_new"])

    print(f"antonym signal (n={len(queries)}, k={k}):")
    print(f"  zero-shot       : {zs:.2f}")
    print(f"  {k}-shot clean    : {ks:.2f}")
    print(f"  {k}-shot shuffled : {sh:.2f}")
    print(f"  label-dependence (clean - shuffled) = {ks - sh:+.2f}")
    verdict = ("SIGNAL OK -> proceed (clean >> shuffled/zero-shot)"
               if (ks - sh) >= 0.2 and (ks - zs) >= 0.2 else
               "WEAK SIGNAL -> pivot to a symbolic/arbitrary-label task")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
