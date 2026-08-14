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
    N_GENERATION_RECORDS,
    build_contrast_set,
    generate_cot_ids,
    get_task,
    load_base_and_lora,
    prompt_token_ids,
)
from scripts.attribution.global_register_vector import estimate_global_vector
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.lockstep_oracle import (
    OverwriteResidualHook,
    control_injection,
    generated_rows,
    lockstep_generate,
)
from src.probes.extraction import PerTokenResidualCapture


def make_lockstep_fns(base, lora, capture, inject, inject_layers, control=None, generator=None,
                      control_alpha=1.0, prompt_len=0, control_positions="all", vector=None,
                      cosines=None):
    """Wire the shared base/adapter into the two callables ``lockstep_generate`` consumes.

    With ``control`` set the oracle's true residual is replaced by a content-destroyed shift of the
    same magnitude (see ``lockstep_oracle.control_injection``) — the empirical floor a k-way task
    needs, where an intervention that merely garbles decoding still scores at chance. It costs a
    second forward per step, since the floor is defined on δ and so needs the base residual too.

    ``vector`` supplies the ``fixed_vector`` mode's frozen constant. ``cosines``, if given, collects
    its per-step cosine to the live running mean — the "is it even the same direction?" diagnostic
    that separates a failed *vector* from a failed *delivery path*.
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
        if cosines is not None and vector is not None:
            li = inject_layers[0]
            live = generated_rows(lora_resid[li] - capture.captured[li], prompt_len).mean(dim=0)
            cosines.append(float(torch.nn.functional.cosine_similarity(
                vector.to(live.device, live.dtype), live, dim=0)))
        return {li: control_injection(capture.captured[li], lora_resid[li], control, generator,
                                      control_alpha, prompt_len, control_positions, vector)
                for li in inject_layers}

    def base_logits(S):
        with inject.injecting(), lora.disable_adapter():
            return base(S, use_cache=False).logits

    return capture_residuals, base_logits


def lockstep_eval(base, lora, tok, problems, device, inject_layers, max_new, task,
                  control=None, generator=None, control_alpha=1.0, control_positions="all",
                  fixed_vector=None, n_records=N_GENERATION_RECORDS):
    """Lockstep-decode every problem; return (accuracy, per-problem correctness, generations).

    The generations are returned, not just printed, because every retraction in this work so far
    was caught by decoding text *after* a number had already become an argument. Persisting the
    first ``n_records`` alongside the accuracy makes "read the generations before the verdict"
    structural rather than a rule someone has to remember.
    """
    capture = PerTokenResidualCapture(base, inject_layers)
    inject = OverwriteResidualHook(base, inject_layers)
    cap_fn, logit_fn = make_lockstep_fns(base, lora, capture, inject, inject_layers,
                                         control, generator, control_alpha)
    eos = tok.eos_token_id
    per, records, cosines = [], [], []
    for i, (q, gold) in enumerate(problems):
        prompt_ids = prompt_token_ids(tok, q, device, task)
        if control:
            # Rebuilt per problem: the control statistic is taken over GENERATED positions, so it
            # needs this problem's prompt length. Averaging over all positions dilutes it ~2.7x
            # and yields an intervention that leaves decoding byte-identical to base.
            vec = fixed_vector(q) if callable(fixed_vector) else fixed_vector
            cap_fn, logit_fn = make_lockstep_fns(base, lora, capture, inject, inject_layers,
                                                 control, generator, control_alpha,
                                                 prompt_ids.shape[1], control_positions, vec,
                                                 cosines)
        out = lockstep_generate(prompt_ids, cap_fn, logit_fn, inject,
                                inject_layers, max_new, eos, device)
        text = tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        ok = task.score(text, gold)
        per.append(ok)
        if len(records) < n_records:
            records.append({"gold": task.format_gold(gold), "ok": ok,
                            "prompt_len": int(prompt_ids.shape[1]),
                            "n_generated": int(out.shape[1] - prompt_ids.shape[1]),
                            "generation": text})
        print(f"    [{i+1}/{len(problems)}] gold={task.format_gold(gold)} ok={ok}", flush=True)
    capture.remove()
    inject.remove()
    diag = {"vector_cosine_to_live_mean": _summarize(cosines)} if cosines else {}
    return sum(per) / len(per), per, {"generations": records, **diag}


def _summarize(xs: list[float]) -> dict:
    t = torch.tensor(xs)
    return {"n": len(xs), "mean": float(t.mean()), "min": float(t.min()), "max": float(t.max())}


def build_fixed_vector(args, base, lora, tok, contrast_all, device, layers, task):
    """Freeze the constant that ``fixed_vector`` injects, and record where it came from.

    ``mean_delta`` reads 0.820, but it re-estimates itself from a live donor forward at *every*
    decode step, and the early steps are near-oracle by construction: at step 1 there are no
    generated rows at all, so the statistic falls back to the whole sequence; at step 2 the "mean"
    **is** the true δ of the first generated token; at step 3 it is the mean of two true δs. Those
    early tokens are the trigger phrase, which is exactly the span that decides the score. So 0.820
    may be measuring a near-oracle injection over the deciding span rather than "a constant
    suffices". Freezing the donor's whole-trajectory mean and injecting it at every step removes the
    loop while holding the delivery path identical — the one variable that matters.

    ``per_problem`` is the direct comparison to ``mean_delta`` (same problem, same layer, same
    delivery, no loop). ``pooled`` is the CAA-style claim: one vector for the whole task, estimated
    on problems disjoint from the evaluated slice.
    """
    if len(layers) != 1:
        raise SystemExit(f"--control fixed_vector needs exactly one --layers value, got {layers}")
    layer = layers[0]
    if args.fixed_vector == "per_problem":
        def per_problem(question):
            vec, _ = estimate_global_vector(base, lora, tok, [(question, None)], device, layer,
                                            args.max_new, task)
            return vec
        return per_problem
    estimation = contrast_all[args.n_contrast:args.n_contrast + args.n_estimate]
    if not estimation:
        raise SystemExit("no disjoint estimation problems left in the contrast cache")
    print(f"[fixed_vector] pooling one vector over {len(estimation)} disjoint problems @L{layer}",
          flush=True)
    vec, stats = estimate_global_vector(base, lora, tok, estimation, device, layer, args.max_new,
                                        task)
    print(f"[fixed_vector] {stats}", flush=True)
    return vec


def run_validate(base, lora, tok, problems, device, num_layers, max_new, n, task):
    """AC1: all-layers lockstep must reproduce LoRA's greedy answers exactly."""
    sample = problems[:n]
    print(f"\n[validate] all-layers lockstep vs LoRA greedy on {len(sample)} problems", flush=True)
    lora_ok = []
    for q, gold in sample:
        ids, plen = generate_cot_ids(lora, tok, q, device, max_new, task)
        txt = tok.decode(ids[0][plen:], skip_special_tokens=True)
        lora_ok.append(task.score(txt, gold))
    ctrl_acc, ctrl_ok, _ = lockstep_eval(base, lora, tok, sample, device, list(range(num_layers)),
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
    ap.add_argument("--control", choices=["mean_delta", "shuffle_positions", "random_matched",
                                          "random_constant", "fixed_vector"], default=None,
                    help="replace the true residual with a content-destroyed shift (empirical floor)")
    ap.add_argument("--control-alpha", type=float, default=1.0,
                    help="scale the control shift; separates a bad direction from a bad magnitude")
    ap.add_argument("--control-positions", choices=["all", "generated", "prompt"], default="all",
                    help="where the control shift lands; separates re-encoding the prompt from "
                         "steering the generation (~150 prompt tokens vs ~7 generated ones)")
    ap.add_argument("--fixed-vector", choices=["per_problem", "pooled"], default="per_problem",
                    help="source of the frozen constant for --control fixed_vector: this problem's "
                         "own donor-trajectory mean, or one vector pooled over disjoint problems")
    ap.add_argument("--n-estimate", type=int, default=100,
                    help="problems pooled for --fixed-vector pooled (taken after the evaluated "
                         "slice, so estimation and evaluation stay disjoint)")
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
    contrast_all = [tuple(scan[i]) for i in indices]
    contrast = contrast_all[:args.n_contrast] if args.n_contrast else contrast_all
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
               "control_positions": args.control_positions if args.control else None,
               "fixed_vector_source": args.fixed_vector if args.control == "fixed_vector" else None,
               "scan_base_acc": base_acc, "scan_lora_acc": lora_acc,
               "contrast_base_acc": 0.0, "contrast_lora_acc": 1.0,
               "n_contrast": len(contrast), "max_new": args.max_new, "contrast_indices": indices,
               "per_layer": {}}

    # GSM8K keeps its original filenames so the committed arithmetic results stay addressable.
    # A control run always gets its own name — silently overwriting a real sweep with its floor is
    # the output hazard that cost the alpha grid its artifact.
    stem = f"lockstep_{args.mode}" if task.name == "gsm8k" else f"lockstep_{task.name}_{args.mode}"
    if args.control:
        stem = f"{stem}_{args.control}"
        if args.control == "fixed_vector":
            stem = f"{stem}_{args.fixed_vector}"
        if args.control_positions != "all":
            stem = f"{stem}_{args.control_positions}"
        if args.control_alpha != 1.0:
            stem = f"{stem}_a{args.control_alpha:g}"
    out_path = Path(args.out) if args.out else out_dir / f"{stem}.json"

    def save() -> None:
        """Write after every cell. The alpha grid reached 9 of 12 and left no artifact because
        this happened only at the end; a killed run must still be able to cite what it measured."""
        out_path.write_text(json.dumps(results, indent=2, default=float))

    fixed_vector = build_fixed_vector(args, base, lora, tok, contrast_all, device, layers, task) \
        if args.control == "fixed_vector" else None
    if args.control == "fixed_vector":
        results["fixed_vector_layer"] = layers[0]

    if args.mode == "control":
        print(f"\n[control] all {num_layers} layers (must recover ≈ lora)", flush=True)
        acc, _, extra = lockstep_eval(base, lora, tok, contrast, device, list(range(num_layers)),
                                      args.max_new, task, args.control, generator,
                                      args.control_alpha, args.control_positions, fixed_vector)
        results["control"] = {"acc": acc, "recovery": recovery(acc), **extra}
        print(f"  control acc={acc:.3f} recovery={recovery(acc):+.3f}", flush=True)
        save()
    else:
        for L in layers:
            inj = [L] if args.mode == "single" else list(range(L + 1))
            print(f"\n[{args.mode}] L={L} (inject {len(inj)} layer(s))", flush=True)
            acc, _, extra = lockstep_eval(base, lora, tok, contrast, device, inj, args.max_new,
                                          task, args.control, generator, args.control_alpha,
                                          args.control_positions, fixed_vector)
            results["per_layer"][L] = {"acc": acc, "recovery": recovery(acc), **extra}
            print(f"  L{L:2d} acc={acc:.3f} recovery={recovery(acc):+.3f}", flush=True)
            save()

    save()
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
