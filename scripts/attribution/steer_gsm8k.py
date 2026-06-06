"""Phase 3: steer the base model with α·W_L·a at each layer; measure GSM8K accuracy.

References (computed once on the same n_eval problems): base accuracy (weak, gate
[0.00,0.08]) and LoRA accuracy (the budget, gate [0.36,0.46]). Then per layer, inject the
Phase-2 W at lambda* into the BASE model (adapter disabled) at all positions and re-score.
Builds the decisive 2×2 input (held-out R² from Phase 2 × steered-acc recovery here).

    uv run python -m scripts.attribution.steer_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml [--maps-suffix _smoke --n-eval 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import gsm8k_accuracy, gsm8k_problems, load_base_and_lora
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.safety.steering_hook import LinearPrimalSteerHook


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--maps-suffix", default="")
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--layers", default=None, help="comma list to restrict (default all)")
    ap.add_argument("--alphas", default=None, help="comma list of alpha values, e.g. 0.5,1.0")
    ap.add_argument("--joint", action="store_true",
                    help="inject ALL --layers maps simultaneously in one hook (full-adapter test)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = cfg["device"]
    n_eval = args.n_eval or cfg["eval"]["n_eval"]
    max_new = cfg["eval"]["max_new"]
    alphas = ([float(x) for x in args.alphas.split(",")] if args.alphas
              else cfg["eval"]["alphas"])
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(cfg["num_layers"])))

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tokenizer, base, lora = load_base_and_lora(cfg)
    problems = gsm8k_problems(cfg["eval"]["split"], n_eval, skip=0)
    print(f"Evaluating on {len(problems)} GSM8K {cfg['eval']['split']} problems (max_new={max_new})", flush=True)

    with lora.disable_adapter():
        base_acc = gsm8k_accuracy(base, tokenizer, problems, device, max_new)
    lora_acc = gsm8k_accuracy(lora, tokenizer, problems, device, max_new)
    print(f"\nREFERENCE  base={base_acc:.3f} (gate 0.00-0.08)  LoRA={lora_acc:.3f} (gate 0.36-0.46)", flush=True)

    maps_dir = Path(cfg["output"]["maps_dir"] + args.maps_suffix)
    results = {"base_acc": base_acc, "lora_acc": lora_acc, "budget": lora_acc - base_acc,
               "n_eval": len(problems), "alphas": alphas, "joint": None, "per_layer": {}}

    def recovery(acc: float) -> float:
        return (acc - base_acc) / (lora_acc - base_acc) if lora_acc > base_acc else 0.0

    if args.joint:
        maps, lam_by_layer = {}, {}
        for l in layers:
            rec = torch.load(maps_dir / f"W_L{l}.pt")
            maps[l], lam_by_layer[l] = rec["W"], rec["lam"]
        print(f"\nJOINT all-layer injection over {len(maps)} layers: {sorted(maps)}", flush=True)
        results["joint"] = {"layers": sorted(maps), "lam_star": lam_by_layer, "alpha": {}}
        for alpha in alphas:
            with lora.disable_adapter():
                hook = LinearPrimalSteerHook(base, maps, alpha)
                acc = gsm8k_accuracy(base, tokenizer, problems, device, max_new)
                hook.remove()
            results["joint"]["alpha"][alpha] = {"steer_acc": acc, "recovery": recovery(acc)}
            print(f"  JOINT a={alpha}: steer_acc={acc:.3f}  recovery={recovery(acc):+.2f}", flush=True)
    else:
        for l in layers:
            rec = torch.load(maps_dir / f"W_L{l}.pt")
            W, lam_star = rec["W"], rec["lam"]
            results["per_layer"][l] = {"lam_star": lam_star, "alpha": {}}
            for alpha in alphas:
                with lora.disable_adapter():
                    hook = LinearPrimalSteerHook(base, {l: W}, alpha)
                    acc = gsm8k_accuracy(base, tokenizer, problems, device, max_new)
                    hook.remove()
                results["per_layer"][l]["alpha"][alpha] = {"steer_acc": acc, "recovery": recovery(acc)}
                print(f"  L{l:2d} a={alpha}: steer_acc={acc:.3f}  recovery={recovery(acc):+.2f}  (lambda*={lam_star:.2e})", flush=True)

    suffix = args.maps_suffix + ("_joint" if args.joint else "")
    Path(cfg["output"]["steer_json"].replace(".json", f"{suffix}.json")).write_text(
        json.dumps(results, indent=2, default=float))
    print(f"\nSaved {cfg['output']['steer_json'].replace('.json', f'{suffix}.json')}")


if __name__ == "__main__":
    main()
