"""Short-output control: does static injection transfer compute over a short trajectory?

GSM8K (multi-step CoT) → all-step joint steering recovers 0.00 of the LoRA budget. Here the
problems are single-operation arithmetic, run under two prompt MODES that vary how long a
trajectory the model must self-maintain (the prompt's steering machinery is identical to GSM8K):

  - ``direct``: the response is forced to start with ``The answer is: `` → the model must emit
    the number immediately, with NO self-maintained CoT (the pure single-step compute probe).
  - ``cot``: the natural ``Let's think step by step`` format → one short reasoning step (~80
    tokens) given a budget where the LoRA actually finishes.

Read against GSM8K's multi-step 0.00:
  - steered recovery rises as the trajectory shortens (cot > direct ~ wins) → the wall is the
    long self-maintained CoT, not the compute.
  - steered recovery stays ~0 even direct/short → the compute itself does not transfer.

    uv run python -m scripts.attribution.short_arithmetic \
        --config configs/attribution/metamath_llama2_gsm8k.yaml \
        --maps-suffix _smoke [--n 30 --modes direct,cot --alphas 0.5,1.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import load_base_and_lora
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.arithmetic_problems import DEFAULT_TIERS, generate_arithmetic_problems
from src.probes.attribution.gsm8k_prompts import (
    extract_pred_number,
    metamath_direct_prompt,
    metamath_prompt,
    numeric_match,
)
from src.probes.safety.steering_hook import LinearPrimalSteerHook

# mode -> (prompt_fn, max_new, direct?). ``direct`` outputs are a couple tokens; ``cot`` needs
# room for one short worked step plus the answer marker.
MODES = {
    "direct": (metamath_direct_prompt, 12, True),
    "cot": (metamath_prompt, 256, False),
}


@torch.no_grad()
def score_by_tier(model, tokenizer, problems, device, prompt_fn, max_new, direct) -> tuple[float, dict]:
    """Greedy-generate each problem under ``prompt_fn``; return (overall_acc, {tier: acc}).

    In ``direct`` mode the answer marker is already in the prompt, so the prediction is the
    leading number of the completion (recovered by re-prepending the marker for the parser).
    """
    tally: dict[str, list[int]] = {}
    correct = 0
    for question, gold, tier in problems:
        ids = tokenizer(prompt_fn(question), return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        pred = extract_pred_number("The answer is: " + text if direct else text)
        ok = int(numeric_match(pred, gold))
        correct += ok
        slot = tally.setdefault(tier, [0, 0])
        slot[0] += ok
        slot[1] += 1
    return correct / len(problems), {t: c / n for t, (c, n) in tally.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--maps-suffix", default="_smoke")
    ap.add_argument("--n", type=int, default=30, help="problems (split round-robin over tiers)")
    ap.add_argument("--seed", type=int, default=None, help="problem-generation seed (default cfg seed)")
    ap.add_argument("--modes", default="direct,cot", help="comma list from {direct, cot}")
    ap.add_argument("--alphas", default="0.5,1.0", help="comma list of joint-steer alpha values")
    ap.add_argument("--layers", default=None, help="comma list for joint injection (default all)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    seed = args.seed if args.seed is not None else cfg["seed"]
    device = cfg["device"]
    modes = [m for m in args.modes.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(cfg["num_layers"])))

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tokenizer, base, lora = load_base_and_lora(cfg)
    problems = generate_arithmetic_problems(args.n, seed)
    print(f"{len(problems)} single-op problems over tiers {DEFAULT_TIERS}; "
          f"modes {modes}; joint over {len(layers)} layers", flush=True)

    maps_dir = Path(cfg["output"]["maps_dir"] + args.maps_suffix)
    maps = {l: torch.load(maps_dir / f"W_L{l}.pt")["W"] for l in layers}

    results = {"base_model": cfg["base_model"], "n": len(problems), "seed": seed,
               "tiers": list(DEFAULT_TIERS), "alphas": alphas, "layers": sorted(maps),
               "by_mode": {}}

    for mode in modes:
        prompt_fn, max_new, direct = MODES[mode]
        with lora.disable_adapter():
            base_acc, base_tier = score_by_tier(base, tokenizer, problems, device, prompt_fn, max_new, direct)
        lora_acc, lora_tier = score_by_tier(lora, tokenizer, problems, device, prompt_fn, max_new, direct)
        budget = lora_acc - base_acc
        print(f"\n[{mode}] max_new={max_new}  base={base_acc:.3f} {base_tier}  "
              f"LoRA={lora_acc:.3f} {lora_tier}  budget={budget:+.3f}", flush=True)

        entry = {"max_new": max_new, "base": {"acc": base_acc, "by_tier": base_tier},
                 "lora": {"acc": lora_acc, "by_tier": lora_tier}, "budget": budget, "steer": {}}
        for alpha in alphas:
            with lora.disable_adapter():
                hook = LinearPrimalSteerHook(base, maps, alpha)
                acc, tier = score_by_tier(base, tokenizer, problems, device, prompt_fn, max_new, direct)
                hook.remove()
            recov = (acc - base_acc) / budget if budget > 0 else None
            entry["steer"][alpha] = {"acc": acc, "recovery": recov, "by_tier": tier}
            rec_s = f"{recov:+.2f}" if recov is not None else "n/a"
            print(f"    steer a={alpha}: acc={acc:.3f}  recovery={rec_s}  {tier}", flush=True)
        results["by_mode"][mode] = entry

    out = Path(cfg["output"]["steer_json"].replace("steer_results.json", "short_arithmetic.json"))
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
