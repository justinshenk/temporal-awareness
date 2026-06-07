"""Per-context local refit: a fresh ridge map per problem, fit on its OWN prompt prefix.

Every static steering result so far used ONE global map W (fit across all problems, off-policy)
and recovered 0.00. This closes a different gap: instead of one-size-fits-all, fit a fresh
per-layer ridge map on each problem's own prompt-prefix activations — teacher-force just the
prompt through base and LoRA, take (a, δ) over the prompt positions, solve the ridge at that
layer's global λ* — then steer THAT problem's CoT generation with its own map.

  - local maps recover accuracy where the global map gave 0.00 → the wall was the global map's
    generality; a context-tuned map transfers.
  - local maps still 0.00 → locality doesn't help; the failure is not map generality.

No answer/CoT leakage: the map sees only the question prompt, never the LoRA's solution.

    uv run python -m scripts.attribution.local_refit_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml \
        --maps-suffix _smoke [--n-eval 30 --alphas 0.5,1.0 --lam global]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import (
    gsm8k_accuracy,
    gsm8k_problems,
    load_base_and_lora,
    prompt_token_ids,
)
from scripts.attribution.collect_cot_residuals import teacher_force_capture
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.cot_collection import assemble_blocks
from src.probes.attribution.gram_accumulator import GramAccumulator
from src.probes.attribution.gsm8k_prompts import extract_pred_number, numeric_match
from src.probes.extraction import PerTokenResidualCapture
from src.probes.safety.steering_hook import LinearPrimalSteerHook


@torch.no_grad()
def fit_local_maps(base, lora, capture, prompt_ids, layers, dim, accum_dev, lam_by_layer) -> dict:
    """Fit a per-layer ridge map on this prompt's own (a, δ) prefix tokens (slice [0:prompt_len])."""
    with lora.disable_adapter():
        base_cap = teacher_force_capture(base, capture, prompt_ids)
    lora_cap = teacher_force_capture(lora, capture, prompt_ids)
    sl = slice(0, prompt_ids.shape[1])
    maps = {}
    for l in layers:
        a, d = assemble_blocks(base_cap, lora_cap, l, sl)
        acc = GramAccumulator(dim, device=accum_dev, layer=l)
        acc.update(a, d)
        maps[l] = acc.solve(lam_by_layer[l]).W.to(torch.float32)
        del acc
    return maps


@torch.no_grad()
def local_refit_accuracy(base, lora, capture, tokenizer, problems, layers, dim, device,
                         accum_dev, lam_by_layer, alphas, max_new) -> dict:
    """Per problem fit its own maps once, then steer+generate+score at every alpha.

    Returns ``{alpha: accuracy}`` (maps depend on λ, not α, so the costly fit is shared).
    """
    correct = {a: 0 for a in alphas}
    for question, gold in problems:
        prompt_ids = prompt_token_ids(tokenizer, question, device)
        maps = fit_local_maps(base, lora, capture, prompt_ids, layers, dim, accum_dev, lam_by_layer)
        for alpha in alphas:
            with lora.disable_adapter():
                hook = LinearPrimalSteerHook(base, maps, alpha)
                out = base.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                hook.remove()
            text = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
            correct[alpha] += int(numeric_match(extract_pred_number(text), gold))
    return {a: correct[a] / len(problems) for a in alphas}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--maps-suffix", default="_smoke", help="global maps dir suffix (for λ* per layer)")
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--alphas", default="0.5,1.0", help="comma list of joint-steer alpha values")
    ap.add_argument("--layers", default=None, help="comma list for joint injection (default all)")
    ap.add_argument("--lams", default="1,100,global",
                    help="comma list; each is a float (fixed λ at every layer) or 'global' (reuse λ*)")
    ap.add_argument("--base-acc", type=float, default=None)
    ap.add_argument("--lora-acc", type=float, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, accum_dev = cfg["device"], cfg["accum_device"]
    dim, max_new = cfg["hidden_dim"], cfg["eval"]["max_new"]
    alphas = [float(x) for x in args.alphas.split(",")]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(cfg["num_layers"])))

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tokenizer, base, lora = load_base_and_lora(cfg)
    capture = PerTokenResidualCapture(base, layers)
    problems = gsm8k_problems(cfg["eval"]["split"], args.n_eval, skip=0)

    maps_dir = Path(cfg["output"]["maps_dir"] + args.maps_suffix)
    global_lam = {l: torch.load(maps_dir / f"W_L{l}.pt")["lam"] for l in layers}
    lam_specs = args.lams.split(",")

    def lam_by_layer(spec: str) -> dict:
        return dict(global_lam) if spec == "global" else {l: float(spec) for l in layers}

    print(f"Per-context local refit: prompt-prefix fit, joint over {len(layers)} layers, "
          f"λ-sweep {lam_specs}; eval {len(problems)} GSM8K {cfg['eval']['split']} problems", flush=True)

    if args.base_acc is not None and args.lora_acc is not None:
        base_acc, lora_acc = args.base_acc, args.lora_acc
        print(f"REFERENCE (supplied) base={base_acc:.3f}  LoRA={lora_acc:.3f}", flush=True)
    else:
        with lora.disable_adapter():
            base_acc = gsm8k_accuracy(base, tokenizer, problems, device, max_new)
        lora_acc = gsm8k_accuracy(lora, tokenizer, problems, device, max_new)
        print(f"REFERENCE base={base_acc:.3f}  LoRA={lora_acc:.3f}", flush=True)

    budget = lora_acc - base_acc
    results = {"base_acc": base_acc, "lora_acc": lora_acc, "budget": budget,
               "n_eval": len(problems), "lams": lam_specs, "layers": sorted(layers), "by_lam": {}}
    for spec in lam_specs:
        accs = local_refit_accuracy(base, lora, capture, tokenizer, problems, layers, dim,
                                    device, accum_dev, lam_by_layer(spec), alphas, max_new)
        results["by_lam"][spec] = {}
        for alpha in alphas:
            acc = accs[alpha]
            recov = (acc - base_acc) / budget if budget > 0 else None
            results["by_lam"][spec][alpha] = {"steer_acc": acc, "recovery": recov}
            rec_s = f"{recov:+.2f}" if recov is not None else "n/a"
            print(f"  local-refit λ={spec:>6} a={alpha}: steer_acc={acc:.3f}  recovery={rec_s}", flush=True)

    out = Path(cfg["output"]["steer_json"].replace("steer_results.json", "local_refit_gsm8k.json"))
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
