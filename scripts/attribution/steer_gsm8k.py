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

from scripts.attribution.attribution_common import (
    get_task,
    load_base_and_lora,
    manifold_bases,
    task_accuracy,
)
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
    ap.add_argument("--norm-preserve", action="store_true",
                    help="rescale each steered position back to its original norm (magnitude-vs-direction probe)")
    ap.add_argument("--project-k", type=int, default=None,
                    help="project injected delta onto top-k manifold basis (manifold probe)")
    ap.add_argument("--project-manifold", default="base", choices=["base", "lora", "union"],
                    help="which manifold to project the delta onto (base=Σaaᵀ, lora=Σ(a+δ)(a+δ)ᵀ)")
    ap.add_argument("--prefill-only", action="store_true",
                    help="steer the prompt pass only; skip every decode step (temporal-axis probe)")
    ap.add_argument("--base-acc", type=float, default=None,
                    help="skip base reference recompute and use this value (deterministic refs)")
    ap.add_argument("--lora-acc", type=float, default=None,
                    help="skip LoRA reference recompute and use this value (deterministic refs)")
    ap.add_argument("--task", default=None, help="task registry key (default: config 'task' or gsm8k)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    task = get_task(args.task or cfg.get("task", "gsm8k"))
    device = cfg["device"]
    n_eval = args.n_eval or cfg["eval"]["n_eval"]
    max_new = cfg["eval"]["max_new"]
    alphas = ([float(x) for x in args.alphas.split(",")] if args.alphas
              else cfg["eval"]["alphas"])
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(cfg["num_layers"])))

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tokenizer, base, lora = load_base_and_lora(cfg)
    problems = task.problems(cfg["eval"]["split"], n_eval, skip=0, seed=cfg["seed"])
    print(f"Evaluating on {len(problems)} {task.name} {cfg['eval']['split']} problems (max_new={max_new})", flush=True)

    if args.base_acc is not None and args.lora_acc is not None:
        base_acc, lora_acc = args.base_acc, args.lora_acc
        print(f"\nREFERENCE (supplied) base={base_acc:.3f}  LoRA={lora_acc:.3f}", flush=True)
    else:
        with lora.disable_adapter():
            base_acc = task_accuracy(base, tokenizer, problems, device, max_new, task)
        lora_acc = task_accuracy(lora, tokenizer, problems, device, max_new, task)
        print(f"\nREFERENCE  base={base_acc:.3f} (gate 0.00-0.08)  LoRA={lora_acc:.3f} (gate 0.36-0.46)", flush=True)

    maps_dir = Path(cfg["output"]["maps_dir"] + args.maps_suffix)
    bases = (manifold_bases(cfg["output"]["acc_dir"] + args.maps_suffix, layers, args.project_k,
                            device, which=args.project_manifold)
             if args.project_k else None)
    hook_kw = dict(norm_preserve=args.norm_preserve, project_bases=bases, prefill_only=args.prefill_only)
    results = {"base_acc": base_acc, "lora_acc": lora_acc, "budget": lora_acc - base_acc,
               "n_eval": len(problems), "alphas": alphas, "norm_preserve": args.norm_preserve,
               "project_k": args.project_k, "project_manifold": args.project_manifold,
               "prefill_only": args.prefill_only, "joint": None, "per_layer": {}}

    def recovery(acc: float) -> float:
        return (acc - base_acc) / (lora_acc - base_acc) if lora_acc > base_acc else 0.0

    suffix = (args.maps_suffix + ("_joint" if args.joint else "")
              + ("_normpreserve" if args.norm_preserve else "")
              + (f"_projk{args.project_k}{args.project_manifold}" if args.project_k else "")
              + ("_prefill" if args.prefill_only else ""))
    out_path = Path(cfg["output"]["steer_json"].replace(".json", f"{suffix}.json"))

    def save() -> None:
        out_path.write_text(json.dumps(results, indent=2, default=float))

    if args.joint:
        maps, lam_by_layer = {}, {}
        for l in layers:
            rec = torch.load(maps_dir / f"W_L{l}.pt")
            maps[l], lam_by_layer[l] = rec["W"], rec["lam"]
        print(f"\nJOINT all-layer injection over {len(maps)} layers: {sorted(maps)}", flush=True)
        results["joint"] = {"layers": sorted(maps), "lam_star": lam_by_layer, "alpha": {}}
        for alpha in alphas:
            gens = []
            with lora.disable_adapter():
                hook = LinearPrimalSteerHook(base, maps, alpha, **hook_kw)
                acc = task_accuracy(base, tokenizer, problems, device, max_new, task, records=gens)
                hook.remove()
            results["joint"]["alpha"][alpha] = {"steer_acc": acc, "recovery": recovery(acc),
                                               "generations": gens}
            print(f"  JOINT a={alpha}: steer_acc={acc:.3f}  recovery={recovery(acc):+.2f}", flush=True)
            save()
    else:
        for l in layers:
            rec = torch.load(maps_dir / f"W_L{l}.pt")
            W, lam_star = rec["W"], rec["lam"]
            results["per_layer"][l] = {"lam_star": lam_star, "alpha": {}}
            for alpha in alphas:
                gens = []
                with lora.disable_adapter():
                    hook = LinearPrimalSteerHook(base, {l: W}, alpha, **hook_kw)
                    acc = task_accuracy(base, tokenizer, problems, device, max_new, task,
                                        records=gens)
                    hook.remove()
                results["per_layer"][l]["alpha"][alpha] = {"steer_acc": acc,
                                                           "recovery": recovery(acc),
                                                           "generations": gens}
                print(f"  L{l:2d} a={alpha}: steer_acc={acc:.3f}  recovery={recovery(acc):+.2f}  (lambda*={lam_star:.2e})", flush=True)
                # A steer sweep is hours long; write after every cell so a death leaves an artifact
                # rather than the log-only cells that cost the alpha grid its JSON.
                save()

    save()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
