"""Phase 1: stream per-token (a_t, δ_t) over GSM8K CoT into per-layer GramAccumulators.

For each problem: greedy-generate a CoT with the LoRA, then teacher-force that exact
token sequence through base (adapter disabled) and LoRA (adapter on), capturing all-token
residuals at every layer in two passes. a_t = base residual over the CoT slice; δ_t =
LoRA − base. Accumulate into a train and a held-out set of GramAccumulators, checkpointed
to disk for the cheap Phase-2 lambda sweep.

    uv run python -m scripts.attribution.collect_cot_residuals \
        --config configs/attribution/metamath_llama2_gsm8k.yaml [--n-fit 8 --n-te 4 --max-new 64]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import generate_cot_ids, gsm8k_problems, load_base_and_lora
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.cot_collection import assemble_blocks, cot_token_slice
from src.probes.attribution.gram_accumulator import GramAccumulator
from src.probes.extraction import PerTokenResidualCapture


@torch.no_grad()
def teacher_force_capture(model, capture, full_ids) -> dict[int, torch.Tensor]:
    capture.clear()
    with capture.capturing():
        model(full_ids, use_cache=False)
    return {l: t.clone() for l, t in capture.captured.items()}


def collect_split(problems, lora, base, capture, tokenizer, layers, dim, device, accum_dev, max_new, tag):
    accs = {l: GramAccumulator(dim, device=accum_dev, layer=l) for l in layers}
    n_tok_total = 0
    for i, (question, _gold) in enumerate(problems):
        full_ids, prompt_len = generate_cot_ids(lora, tokenizer, question, device, max_new)
        cot_len = full_ids.shape[1] - prompt_len
        if cot_len <= 0:
            continue
        sl = cot_token_slice(prompt_len, full_ids.shape[1])
        with lora.disable_adapter():
            base_cap = teacher_force_capture(base, capture, full_ids)
        lora_cap = teacher_force_capture(lora, capture, full_ids)
        for l in layers:
            a, d = assemble_blocks(base_cap, lora_cap, l, sl)
            accs[l].update(a, d)
        n_tok_total += cot_len
        if (i + 1) % 10 == 0 or i == len(problems) - 1:
            print(f"  [{tag}] {i+1}/{len(problems)} problems, {n_tok_total} CoT tokens", flush=True)
    return accs, n_tok_total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-fit", type=int, default=None)
    ap.add_argument("--n-te", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--out-suffix", default="", help="subdir suffix for smoke runs, e.g. _smoke")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    n_fit = args.n_fit or cfg["collect"]["n_fit"]
    n_te = args.n_te or cfg["collect"]["n_te"]
    max_new = args.max_new or cfg["collect"]["max_new"]
    device, accum_dev = cfg["device"], cfg["accum_device"]
    layers = list(range(cfg["num_layers"]))
    dim = cfg["hidden_dim"]

    print(f"Loading {cfg['base_model']} + adapter {cfg['adapter']} ...", flush=True)
    tokenizer, base, lora = load_base_and_lora(cfg)
    capture = PerTokenResidualCapture(base, layers)

    split = cfg["collect"]["split"]
    fit_problems = gsm8k_problems(split, n_fit, skip=0)
    te_problems = gsm8k_problems(split, n_te, skip=n_fit)  # disjoint
    print(f"fit={len(fit_problems)} held-out={len(te_problems)} problems (max_new={max_new})", flush=True)

    acc_tr, ntr = collect_split(fit_problems, lora, base, capture, tokenizer, layers, dim, device, accum_dev, max_new, "train")
    acc_te, nte = collect_split(te_problems, lora, base, capture, tokenizer, layers, dim, device, accum_dev, max_new, "heldout")
    capture.remove()

    out_dir = Path(cfg["output"]["acc_dir"] + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    for l in layers:
        torch.save(acc_tr[l].state_dict(), out_dir / f"train_L{l}.pt")
        torch.save(acc_te[l].state_dict(), out_dir / f"heldout_L{l}.pt")
    meta = {"base_model": cfg["base_model"], "adapter": cfg["adapter"], "n_fit": len(fit_problems),
            "n_te": len(te_problems), "max_new": max_new, "layers": layers, "hidden_dim": dim,
            "train_cot_tokens": ntr, "heldout_cot_tokens": nte,
            "train_tokens_per_layer": acc_tr[layers[0]].n_tokens}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {len(layers)*2} accumulators + meta.json to {out_dir}", flush=True)
    print(f"train CoT tokens={ntr}  held-out CoT tokens={nte}", flush=True)


if __name__ == "__main__":
    main()
