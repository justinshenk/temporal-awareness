"""Qualitative side-by-side: base vs. LoRA vs. (optionally) steered greedy generation
on one GSM8K problem.

    uv run python -m scripts.attribution.show_generation \
        --config configs/attribution/metamath_llama2_gsm8k.yaml [--index 0 --max-new 400]

Add steering to see what the primal-ridge maps actually do to generation:

    ... --steer-maps _smoke --steer-alphas 0.5,1.0           # all 32 layers, joint
    ... --steer-maps _smoke --steer-layers 31 --steer-alphas 1.0   # single layer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import (
    gsm8k_problems,
    load_base_and_lora,
    manifold_bases,
    prompt_token_ids,
)
from src.probes.attribution.gsm8k_prompts import extract_pred_number, numeric_match
from src.probes.safety.steering_hook import LinearPrimalSteerHook


@torch.no_grad()
def generate_text(model, tokenizer, question, device, max_new) -> str:
    prompt_ids = prompt_token_ids(tokenizer, question, device)
    out = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)


def report(label, text, gold) -> None:
    pred = extract_pred_number(text)
    print(f"\n----- {label} -----\n{text}")
    print(f"  -> parsed={pred}  correct={numeric_match(pred, gold)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--steer-maps", default=None, help="maps_dir suffix, e.g. _smoke (enables steering)")
    ap.add_argument("--steer-layers", default=None, help="comma list (default all num_layers)")
    ap.add_argument("--steer-alphas", default="0.5,1.0")
    ap.add_argument("--steer-norm-preserve", action="store_true",
                    help="rescale each steered position back to its original norm")
    ap.add_argument("--steer-project-k", type=int, default=None,
                    help="project injected delta onto top-k manifold basis")
    ap.add_argument("--steer-project-manifold", default="base", choices=["base", "lora", "union"],
                    help="which manifold to project onto (base/lora/union)")
    ap.add_argument("--steer-prefill-only", action="store_true",
                    help="steer the prompt pass only; let generation run un-reinjected")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device, max_new = cfg["device"], args.max_new

    tokenizer, base, lora = load_base_and_lora(cfg)
    question, gold = gsm8k_problems(cfg["eval"]["split"], 1, skip=args.index)[0]

    bar = "=" * 80
    print(f"{bar}\nQUESTION (test #{args.index}):\n{question}\n\nGOLD: {gold}\n{bar}")

    with lora.disable_adapter():
        report("BASE (adapter disabled)", generate_text(base, tokenizer, question, device, max_new), gold)
    report("LoRA (MetaMath adapter)", generate_text(lora, tokenizer, question, device, max_new), gold)

    if args.steer_maps is not None:
        maps_dir = Path(cfg["output"]["maps_dir"] + args.steer_maps)
        layers = ([int(x) for x in args.steer_layers.split(",")] if args.steer_layers
                  else list(range(cfg["num_layers"])))
        maps = {l: torch.load(maps_dir / f"W_L{l}.pt")["W"] for l in layers}
        bases = None
        if args.steer_project_k:
            acc_dir = cfg["output"]["acc_dir"] + args.steer_maps
            bases = manifold_bases(acc_dir, layers, args.steer_project_k, device,
                                   which=args.steer_project_manifold)
        tag = "joint-all-layer" if len(layers) == cfg["num_layers"] else f"L{layers}"
        np_tag = " norm-preserve" if args.steer_norm_preserve else ""
        pj_tag = f" project-k={args.steer_project_k}/{args.steer_project_manifold}" if args.steer_project_k else ""
        pf_tag = " prefill-only" if args.steer_prefill_only else ""
        for alpha in (float(x) for x in args.steer_alphas.split(",")):
            with lora.disable_adapter():
                hook = LinearPrimalSteerHook(base, maps, alpha, norm_preserve=args.steer_norm_preserve,
                                             project_bases=bases, prefill_only=args.steer_prefill_only)
                text = generate_text(base, tokenizer, question, device, max_new)
                hook.remove()
            report(f"STEERED {tag}{np_tag}{pj_tag}{pf_tag} a={alpha}", text, gold)


if __name__ == "__main__":
    main()
