"""On-policy DAgger refit: does closing the off-policy loop rescue static steering?

Every map so far was fit OFF-POLICY — on sequences the LoRA (expert) generated, with clean-base
activations — but deployed on the STEERED base's own rollout, which is off-distribution. DAgger
closes that loop:

  round k:  roll out with the current steered map  →  on-policy CoT sequences
            label each visited state with the expert: a = clean base residual, δ = LoRA − base
            AGGREGATE those (a, δ) into the dataset (seeded with the original off-policy stats)
            refit W on the union  →  steer + eval

Aggregation = adding to GramAccumulators seeded from the saved off-policy ``train_L*.pt`` (the "D"
in DAgger Dataset Aggregation). λ reuses each layer's global λ* (fixed). If accuracy climbs over
rounds, the wall was the off-policy distribution; if it stays ~0, on-policy data doesn't help and
the linear-injection mechanism is the wall.

    uv run python -m scripts.attribution.dagger_refit_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml \
        --maps-suffix _smoke [--rounds 2 --n-fit 48 --n-eval 30 --alpha 1.0 --rollout-max 256]
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
    prompt_token_ids,
    task_accuracy,
)
from scripts.attribution.collect_cot_residuals import teacher_force_capture
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.cot_collection import assemble_blocks, cot_token_slice
from src.probes.attribution.gram_accumulator import GramAccumulator
from src.probes.extraction import PerTokenResidualCapture
from src.probes.safety.steering_hook import LinearPrimalSteerHook


@torch.no_grad()
def rollout_and_label(base, lora, capture, tokenizer, problems, maps, alpha, layers, device,
                      accs, rollout_max, task) -> tuple[int, int]:
    """Steer base with ``maps``, generate on-policy CoT, label with LoRA δ, accumulate into ``accs``."""
    n_seq = n_tok = 0
    for question, _gold in problems:
        pid = prompt_token_ids(tokenizer, question, device, task)
        with lora.disable_adapter():
            hook = LinearPrimalSteerHook(base, maps, alpha)
            full_ids = base.generate(pid, max_new_tokens=rollout_max, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            hook.remove()
        cot_len = full_ids.shape[1] - pid.shape[1]
        if cot_len <= 0:
            continue
        sl = cot_token_slice(pid.shape[1], full_ids.shape[1])
        with lora.disable_adapter():
            base_cap = teacher_force_capture(base, capture, full_ids)
        lora_cap = teacher_force_capture(lora, capture, full_ids)
        for l in layers:
            a, d = assemble_blocks(base_cap, lora_cap, l, sl)
            accs[l].update(a, d)
        n_seq += 1
        n_tok += cot_len
    return n_seq, n_tok


@torch.no_grad()
def steered_eval(base, lora, tokenizer, problems, maps, alpha, device, max_new, task) -> float:
    with lora.disable_adapter():
        hook = LinearPrimalSteerHook(base, maps, alpha)
        acc = task_accuracy(base, tokenizer, problems, device, max_new, task)
        hook.remove()
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--maps-suffix", default="_smoke")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-fit", type=int, default=48, help="on-policy rollout problems per round (train)")
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--rollout-max", type=int, default=256)
    ap.add_argument("--eval-max", type=int, default=256)
    ap.add_argument("--task", default=None, help="task registry key (default: config 'task' or gsm8k)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    task = get_task(args.task or cfg.get("task", "gsm8k"))
    device, accum_dev, dim = cfg["device"], cfg["accum_device"], cfg["hidden_dim"]
    layers = list(range(cfg["num_layers"]))

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tokenizer, base, lora = load_base_and_lora(cfg)
    capture = PerTokenResidualCapture(base, layers)
    train_problems = task.problems(cfg["collect"]["split"], args.n_fit, skip=0, seed=cfg["seed"])
    eval_problems = task.problems(cfg["eval"]["split"], args.n_eval, skip=0, seed=cfg["seed"])

    acc_dir = Path(cfg["output"]["acc_dir"] + args.maps_suffix)
    maps_dir = Path(cfg["output"]["maps_dir"] + args.maps_suffix)
    print(f"Seeding aggregation from off-policy accumulators in {acc_dir} ...", flush=True)
    accs = {l: GramAccumulator.from_state_dict(torch.load(acc_dir / f"train_L{l}.pt"), device=accum_dev)
            for l in layers}
    recs = {l: torch.load(maps_dir / f"W_L{l}.pt") for l in layers}
    global_lam = {l: recs[l]["lam"] for l in layers}
    maps = {l: recs[l]["W"].to(torch.float32) for l in layers}

    with lora.disable_adapter():
        base_acc = task_accuracy(base, tokenizer, eval_problems, device, args.eval_max, task)
    lora_acc = task_accuracy(lora, tokenizer, eval_problems, device, args.eval_max, task)
    budget = lora_acc - base_acc
    print(f"REFERENCE base={base_acc:.3f}  LoRA={lora_acc:.3f}  budget={budget:+.3f}  "
          f"(n_eval={len(eval_problems)}, eval_max={args.eval_max})", flush=True)

    def recovery(a):
        return (a - base_acc) / budget if budget > 0 else None

    def rec_str(a):
        r = recovery(a)
        return f"{r:+.2f}" if r is not None else "n/a"

    seed_tokens = sum(accs[l].n_tokens for l in layers) // len(layers)
    acc0 = steered_eval(base, lora, tokenizer, eval_problems, maps, args.alpha, device, args.eval_max, task)
    print(f"\n[round 0]  seed=off-policy global map  acc={acc0:.3f}  recovery={rec_str(acc0)}  "
          f"(agg tokens/layer={seed_tokens})", flush=True)
    rounds = [{"round": 0, "source": "off-policy-global", "agg_tokens_per_layer": seed_tokens,
               "steer_acc": acc0, "recovery": recovery(acc0)}]

    for k in range(1, args.rounds + 1):
        n_seq, n_tok = rollout_and_label(base, lora, capture, tokenizer, train_problems, maps,
                                         args.alpha, layers, device, accs, args.rollout_max, task)
        maps = {l: accs[l].solve(global_lam[l]).W.to(torch.float32) for l in layers}
        agg_tokens = sum(accs[l].n_tokens for l in layers) // len(layers)
        acc = steered_eval(base, lora, tokenizer, eval_problems, maps, args.alpha, device, args.eval_max, task)
        print(f"[round {k}]  +{n_seq} on-policy seqs ({n_tok} tok)  agg tokens/layer={agg_tokens}  "
              f"acc={acc:.3f}  recovery={rec_str(acc)}", flush=True)
        rounds.append({"round": k, "source": "on-policy-aggregated", "on_policy_seqs": n_seq,
                       "on_policy_tokens": n_tok, "agg_tokens_per_layer": agg_tokens,
                       "steer_acc": acc, "recovery": recovery(acc)})

    results = {"base_acc": base_acc, "lora_acc": lora_acc, "budget": budget, "alpha": args.alpha,
               "n_fit": args.n_fit, "n_eval": len(eval_problems), "rollout_max": args.rollout_max,
               "eval_max": args.eval_max, "rounds": rounds}
    out = Path(cfg["output"]["steer_json"]).parent / f"dagger_refit_{task.name}.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
