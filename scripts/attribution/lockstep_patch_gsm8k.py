"""Lockstep dual-forward residual oracle on GSM8K (Step 1, the linchpin).

For each problem in the base-fails / LoRA-solves contrast set, greedily decode the BASE model
while, at every step, overwriting its residual at a chosen layer set with the LoRA model's true
residual on the same base-generated context (see ``src/probes/attribution/lockstep_oracle.py``).

    # validate the apparatus (AC1): all-layers control must reproduce LoRA exactly
    uv run python -m scripts.attribution.lockstep_patch_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --validate --n-contrast 3

    # single-layer recovery sweep (the headline) + cumulative (depth transition)
    uv run python -m scripts.attribution.lockstep_patch_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml \
        --mode single --layers 0,4,8,12,16,20,24,28,31

Modes: ``control`` (all layers; positive control), ``single`` (inject layer L alone, sweep),
``cumulative`` (inject layers 0..L, sweep). recovery = (acc-base)/(lora-base).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import (
    build_contrast_set,
    generate_cot_ids,
    get_task,
    load_base_and_lora,
    prompt_token_ids,
)
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.lockstep_oracle import (
    OverwriteResidualHook,
    control_injection,
    lockstep_generate,
)
from src.probes.extraction import PerTokenResidualCapture


def make_lockstep_fns(base, lora, capture, inject, inject_layers, control=None, generator=None,
                      control_alpha=1.0, prompt_len=0):
    """Wire the shared base/adapter into the two callables ``lockstep_generate`` consumes.

    With ``control`` set the oracle's true residual is replaced by a content-destroyed shift of the
    same magnitude (see ``lockstep_oracle.control_injection``) — the empirical floor a k-way task
    needs, where an intervention that merely garbles decoding still scores at chance. It costs a
    second forward per step, since the floor is defined on δ and so needs the base residual too.
    """
    def capture_residuals(S):
        capture.clear()
        with capture.capturing():
            base(S, use_cache=False)  # adapter ON → LoRA residuals on base's context
        lora_resid = {li: capture.captured[li] for li in inject_layers}
        if control is None:
            return lora_resid
        capture.clear()
        with lora.disable_adapter(), capture.capturing():
            base(S, use_cache=False)  # adapter OFF → base residual a, on the same context
        return {li: control_injection(capture.captured[li], lora_resid[li], control, generator,
                                     control_alpha, prompt_len)
                for li in inject_layers}

    def base_logits(S):
        with inject.injecting(), lora.disable_adapter():
            return base(S, use_cache=False).logits

    return capture_residuals, base_logits


def lockstep_eval(base, lora, tok, problems, device, inject_layers, max_new, task,
                  control=None, generator=None, control_alpha=1.0):
    """Lockstep-decode every problem; return (accuracy, per-problem correctness list)."""
    capture = PerTokenResidualCapture(base, inject_layers)
    inject = OverwriteResidualHook(base, inject_layers)
    cap_fn, logit_fn = make_lockstep_fns(base, lora, capture, inject, inject_layers,
                                         control, generator, control_alpha)
    eos = tok.eos_token_id
    per = []
    for i, (q, gold) in enumerate(problems):
        prompt_ids = prompt_token_ids(tok, q, device, task)
        if control:
            # Rebuilt per problem: the control statistic is taken over GENERATED positions, so it
            # needs this problem's prompt length. Averaging over all positions dilutes it ~2.7x
            # and yields an intervention that leaves decoding byte-identical to base.
            cap_fn, logit_fn = make_lockstep_fns(base, lora, capture, inject, inject_layers,
                                                 control, generator, control_alpha,
                                                 prompt_ids.shape[1])
        out = lockstep_generate(prompt_ids, cap_fn, logit_fn, inject,
                                inject_layers, max_new, eos, device)
        text = tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        ok = task.score(text, gold)
        per.append(ok)
        print(f"    [{i+1}/{len(problems)}] gold={task.format_gold(gold)} ok={ok}", flush=True)
    capture.remove()
    inject.remove()
    return sum(per) / len(per), per


def run_validate(base, lora, tok, problems, device, num_layers, max_new, n, task):
    """AC1: all-layers lockstep must reproduce LoRA's greedy answers exactly."""
    sample = problems[:n]
    print(f"\n[validate] all-layers lockstep vs LoRA greedy on {len(sample)} problems", flush=True)
    lora_ok = []
    for q, gold in sample:
        ids, plen = generate_cot_ids(lora, tok, q, device, max_new, task)
        txt = tok.decode(ids[0][plen:], skip_special_tokens=True)
        lora_ok.append(task.score(txt, gold))
    ctrl_acc, ctrl_ok = lockstep_eval(base, lora, tok, sample, device, list(range(num_layers)),
                                      max_new, task)
    lora_acc = sum(lora_ok) / len(lora_ok)
    passed = ctrl_ok == lora_ok
    print(f"[validate] lora_acc={lora_acc:.3f} control_acc={ctrl_acc:.3f} "
          f"per-problem-match={passed}  {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["control", "single", "cumulative"], default="single")
    ap.add_argument("--layers", default=None, help="comma list (default: all, evenly for sweep)")
    ap.add_argument("--n-eval", type=int, default=60, help="problems scanned to build contrast set")
    ap.add_argument("--n-contrast", type=int, default=None, help="cap on contrast-set size used")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--validate", action="store_true", help="run AC1 check and exit")
    ap.add_argument("--task", default=None, help="task registry key (default: config 'task' or gsm8k)")
    ap.add_argument("--control", choices=["mean_delta", "shuffle_positions", "random_matched"],
                    default=None,
                    help="replace the true residual with a content-destroyed shift (empirical floor)")
    ap.add_argument("--control-alpha", type=float, default=1.0,
                    help="scale the control shift; separates a bad direction from a bad magnitude")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, num_layers = cfg["device"], cfg["num_layers"]
    task = get_task(args.task or cfg.get("task", "gsm8k"))

    print(f"Loading {cfg['base_model']} + adapter (task={task.name}) ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)
    scan = task.problems(cfg["eval"]["split"], args.n_eval, skip=0, seed=cfg["seed"])

    # Reuse the task's own contrast cache when the config names one (the multi-hop P0 gate writes
    # it in this exact schema); otherwise fall back to the driver-local GSM8K cache.
    out_dir = Path(cfg["output"].get("dir") or Path(cfg["output"]["steer_json"]).parent)
    cache_path = Path(cfg["output"].get("contrast_json") or out_dir / "lockstep_contrast_set.json")
    indices, base_acc, lora_acc = build_contrast_set(
        base, lora, tok, scan, device, args.max_new, cache_path, task)
    contrast = [tuple(scan[i]) for i in indices]
    if args.n_contrast:
        contrast = contrast[:args.n_contrast]
    print(f"Using {len(contrast)} contrast problems (base={base_acc:.3f} lora={lora_acc:.3f})", flush=True)

    if args.validate:
        ok = run_validate(base, lora, tok, contrast, device, num_layers, args.max_new,
                          args.n_contrast or 3, task)
        raise SystemExit(0 if ok else 1)

    # The contrast set is base-fails / LoRA-solves by construction, so on it base accuracy is 0
    # and LoRA accuracy is 1: the recoverable budget spans the full [0,1] range and recovery is
    # simply the contrast-set accuracy. (``base_acc``/``lora_acc`` below are the full-scan numbers,
    # kept only for context — do NOT use them as the recovery denominator.)
    def recovery(acc: float) -> float:
        return acc

    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(0, num_layers, 4)) + [num_layers - 1])

    # Seeded once per run: the floor must be reproducible, and every problem must draw from the
    # same stream so a lucky permutation cannot be mistaken for a layer effect.
    generator = torch.Generator().manual_seed(cfg["seed"]) if args.control else None

    results = {"task": task.name, "mode": args.mode, "control_mode": args.control,
               "control_alpha": args.control_alpha if args.control else None,
               "scan_base_acc": base_acc, "scan_lora_acc": lora_acc,
               "contrast_base_acc": 0.0, "contrast_lora_acc": 1.0,
               "n_contrast": len(contrast), "max_new": args.max_new, "contrast_indices": indices,
               "per_layer": {}}

    if args.mode == "control":
        print(f"\n[control] all {num_layers} layers (must recover ≈ lora)", flush=True)
        acc, _ = lockstep_eval(base, lora, tok, contrast, device, list(range(num_layers)),
                               args.max_new, task, args.control, generator, args.control_alpha)
        results["control"] = {"acc": acc, "recovery": recovery(acc)}
        print(f"  control acc={acc:.3f} recovery={recovery(acc):+.3f}", flush=True)
    else:
        for L in layers:
            inj = [L] if args.mode == "single" else list(range(L + 1))
            print(f"\n[{args.mode}] L={L} (inject {len(inj)} layer(s))", flush=True)
            acc, _ = lockstep_eval(base, lora, tok, contrast, device, inj, args.max_new, task,
                                   args.control, generator, args.control_alpha)
            results["per_layer"][L] = {"acc": acc, "recovery": recovery(acc)}
            print(f"  L{L:2d} acc={acc:.3f} recovery={recovery(acc):+.3f}", flush=True)

    # GSM8K keeps its original filenames so the committed arithmetic results stay addressable.
    # A control run always gets its own name — silently overwriting a real sweep with its floor is
    # the output hazard that cost the alpha grid its artifact.
    stem = f"lockstep_{args.mode}" if task.name == "gsm8k" else f"lockstep_{task.name}_{args.mode}"
    if args.control:
        stem = f"{stem}_{args.control}"
        if args.control_alpha != 1.0:
            stem = f"{stem}_a{args.control_alpha:g}"
    out_path = Path(args.out) if args.out else out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
