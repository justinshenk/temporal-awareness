"""Does ONE vector, pooled across problems, install a register? (the CAA-style test)

The corrected `mean_delta` floor showed a single constant shift recovers 0.82 of the L20 oracle's
0.99 on commonsense — but that vector is estimated from *the problem's own* donor trajectory, so it
only shows the required shift has no temporal structure **within** a problem. This script asks the
stronger, and much more useful, question: estimate **one** vector by pooling δ across a set of
*estimation* problems, then inject that same constant for every *evaluation* problem. That is the
object CAA/ActAdd-style steering actually offers, and it is the register-side claim §4 and §10 rest on.

Estimation and evaluation problems are **disjoint** — the estimation set is taken from the contrast
cache *after* the evaluated slice — so the number is out-of-sample, not a restatement of the fit.

    uv run python -m scripts.attribution.global_register_vector \
        --config configs/attribution/commonsense_llama2.yaml --task commonsense \
        --layer 20 --n-estimate 100 --n-contrast 100 --alphas 0.5,1.0,1.5

Reuses `AdditionSteeringHook` (the existing additive-steering hook) and `task_accuracy`, so the
evaluation path is the ordinary generation path — no lockstep, no donor forward at inference. That
is the point: if this works, installing the register needs no oracle at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import (
    generate_cot_ids,
    get_task,
    load_base_and_lora,
    load_contrast,
    task_accuracy,
)
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.lockstep_oracle import generated_rows
from src.probes.extraction import PerTokenResidualCapture
from src.probes.safety.steering_hook import AdditionSteeringHook


@torch.no_grad()
def estimate_global_vector(base, lora, tok, problems, device, layer, max_new, task):
    """Pool the mean generated-position δ at ``layer`` over ``problems`` into one vector.

    Per problem: let the **donor** generate (so the trajectory is the one the register produces),
    then run base and donor forwards over that same sequence and average δ over the generated
    positions only. Problem means are averaged with equal weight, so a long generation cannot
    dominate the pooled direction.
    """
    capture = PerTokenResidualCapture(base, [layer])
    per_problem = []
    for i, (question, _gold) in enumerate(problems):
        ids, plen = generate_cot_ids(lora, tok, question, device, max_new, task)
        if ids.shape[1] <= plen:
            continue
        capture.clear()
        with capture.capturing():
            base(ids, use_cache=False)                      # adapter ON → donor residual
        h_lora = capture.captured[layer]
        capture.clear()
        with lora.disable_adapter(), capture.capturing():
            base(ids, use_cache=False)                      # adapter OFF → base residual
        a = capture.captured[layer]
        per_problem.append(generated_rows(h_lora - a, plen).mean(dim=0))
        if (i + 1) % 20 == 0:
            print(f"  estimated on {i+1}/{len(problems)} problems", flush=True)
    capture.remove()
    stacked = torch.stack(per_problem)
    pooled = stacked.mean(dim=0)
    # How aligned are the per-problem vectors? A high mean cosine to the pooled direction means one
    # vector can serve the task; a low one means the per-problem win does not generalize.
    cos = torch.nn.functional.cosine_similarity(stacked, pooled.unsqueeze(0), dim=1)
    return pooled, {"n_problems": len(per_problem), "pooled_norm": float(pooled.norm()),
                    "mean_per_problem_norm": float(stacked.norm(dim=1).mean()),
                    "mean_cosine_to_pooled": float(cos.mean()),
                    "min_cosine_to_pooled": float(cos.min())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default=None)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-estimate", type=int, default=100)
    ap.add_argument("--n-contrast", type=int, default=100)
    ap.add_argument("--alphas", default="0.5,1.0,1.5")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, L = cfg["device"], args.layer
    task = get_task(args.task or cfg.get("task", "gsm8k"))
    alphas = [float(x) for x in args.alphas.split(",")]

    print(f"Loading {cfg['base_model']} + adapter (task={task.name}) ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)

    contrast = load_contrast(cfg, task)
    evaluated = contrast[:args.n_contrast]
    estimation = contrast[args.n_contrast:args.n_contrast + args.n_estimate]
    if not estimation:
        raise SystemExit("no disjoint estimation problems left in the contrast cache")
    print(f"estimate on {len(estimation)} problems, evaluate on {len(evaluated)} (disjoint)",
          flush=True)

    vec, stats = estimate_global_vector(base, lora, tok, estimation, device, L, args.max_new, task)
    print(f"[vector] {stats}", flush=True)

    results = {"task": task.name, "layer": L, "n_contrast": len(evaluated), "max_new": args.max_new,
               "vector_stats": stats, "per_alpha": {}}
    with lora.disable_adapter():
        for alpha in alphas:
            hook = AdditionSteeringHook(base, {L: (vec * alpha)})
            acc = task_accuracy(base, tok, evaluated, device, args.max_new, task)
            hook.remove()
            # base is 0 and donor 1 on the contrast set by construction, so accuracy IS recovery
            results["per_alpha"][alpha] = {"acc": acc, "recovery": acc}
            print(f"  alpha={alpha:<5} acc={acc:.3f} recovery={acc:+.3f}", flush=True)

    out_dir = Path(cfg["output"].get("dir") or Path(cfg["output"]["steer_json"]).parent)
    out_path = Path(args.out) if args.out else out_dir / f"global_register_vector_{task.name}_L{L}.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
