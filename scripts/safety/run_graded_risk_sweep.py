"""Route-dependent safety across a graded-distance-to-refusal content axis.

Holds the task FUNCTION fixed (4-option MCQ) and sweeps four content buckets B0<B1<B2<B3
at increasing topical proximity to the refusal axis r (MMLU neutral → WMDP hazardous).
Per bucket it runs the three route arms and a gradient gate:

  0. GRADIENT GATE   bucket-mean cos(a_i, r) at the prediction site — must be monotone
                     B0<B1<B2<B3 (the design premise). Aborts unless --force.
  1. HEADROOM        zero-shot vs few-shot MCQ accuracy (does steering have a gain to move?).
  2. ICL erosion     fill harmful-prompt context with k bucket shots; ΔRefusal + cos(shift,r).
  3. STEER (breadth) fit a per-layer input-conditional map W from the bucket's ICL shifts,
                     measure task transfer (W·a) and refusal; mean-vector baseline too.
  4. LoRA            attach the bucket's adapter (train_graded_lora.py first); ΔRefusal + cos.

All base-model measurements run BEFORE any LoRA adapter is attached (PeftModel wraps the
base in-place — the recurring base-contamination trap).

    HF_TOKEN=... uv run python -m scripts.safety.run_graded_risk_sweep \
        --config configs/safety/graded_risk_qwen.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import capture_resid, generate_reply, prompt_ids, set_seed, user_turn
from scripts.safety.run_lora_distill import ridge_maps
from scripts.safety.run_route_safety_sweep import condition_metrics
from src.probes.extraction import PerTokenResidualCapture
from src.probes.safety.graded_risk_data import (
    is_monotone_increasing,
    load_buckets,
    mean_cosine_to_dir,
)
from src.probes.safety.mcq_icl import chat_mcq, icl_mcq, mcq_accuracy
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless
from src.probes.safety.steering_hook import AdditionSteeringHook, LinearConditionalSteerHook


@torch.no_grad()
def capture_pooled(model, capture, ids, device):
    """Mean-pooled and last-token residuals per layer for one prompt.

    The gradient gate measures content proximity to r via the MEAN-pooled prompt residual
    (the topic lives in the question tokens; the last token is the format token "Answer:").
    Both are recorded so the premise can be read either way.
    """
    capture.clear()
    with capture.capturing():
        model(torch.tensor([ids], device=device), use_cache=False)
    mean = {L: hs.numpy().mean(axis=0) for L, hs in capture.captured.items()}
    last = {L: hs.numpy()[-1] for L, hs in capture.captured.items()}
    return mean, last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="continue even if the gradient gate fails")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    layers = cfg["extract"]["layers"]
    mc, ft = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"]
    grad_layer = cfg["extract"]["grad_layer"]
    max_new = cfg["eval"]["max_new"]
    buckets_order = cfg["buckets"]
    d = cfg["data"]
    icl_shots = cfg["doses"]["icl_shots"]
    st = cfg["steer"]
    lambdas, alphas, few = st["lambdas"], st["alphas"], st["few"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_root = Path(cfg["output"]["adapter_root"])

    # ── prompt sets ──────────────────────────────────────────────────────────
    harmful, harmless = load_harmful(), load_harmless()
    n_dir, n_eval = cfg["direction"]["n_harmful"], cfg["eval"]["n_harmful"]
    harmful_fit = harmful[:n_dir]
    harmful_eval = harmful[n_dir:n_dir + n_eval]
    harmless_fit = harmless[:cfg["direction"]["n_harmless"]]

    print("Loading buckets ...")
    buckets = load_buckets(buckets_order, cfg["seed"], d["n_filler"], d["n_fit"], d["n_eval"], d["n_train"])
    for b in buckets_order:
        print(f"  {b}: {buckets[b].n_available} rows available")

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(msgs):
        return capture_resid(base, capture, prompt_ids(tokenizer, msgs), args.device)

    def gen(ids):
        return generate_reply(base, tokenizer, ids, args.device, max_new)

    def base_acc_for(cases, hook_factory=None, k=0, fillers=None):
        hook = hook_factory() if hook_factory else None
        acc, n = mcq_accuracy(tokenizer, cases, gen, mc, max_new, fillers=fillers, k=k, fill_target=ft)
        if hook:
            hook.remove()
        return acc, n

    def steered_refusal(hook_factory):
        hook = hook_factory()
        rate = refusal_rate([gen(prompt_ids(tokenizer, user_turn(p))) for p in harmful_eval])
        hook.remove()
        return rate

    # ── refusal direction r (base, clean harmful vs harmless) ────────────────
    h_res = [resid(user_turn(p)) for p in harmful_fit]
    s_res = [resid(user_turn(p)) for p in harmless_fit]
    r_by_layer = {L: refusal_direction(np.stack([x[L] for x in h_res]),
                                       np.stack([x[L] for x in s_res])) for L in layers}

    # ── shared baseline on the held-out harmful set ──────────────────────────
    base_resids = [resid(user_turn(p)) for p in harmful_eval]
    base_refusals = [gen(prompt_ids(tokenizer, user_turn(p))) for p in harmful_eval]
    base_rate = refusal_rate(base_refusals)
    print(f"base refusal={base_rate:.3f} (n={len(harmful_eval)})")

    # ── 0. GRADIENT GATE: bucket cos(content, r); gate on the mean-pooled measure ─
    gradient, gradient_lasttok = {}, {}
    for b in buckets_order:
        sample = buckets[b].eval[:d["grad_sample"]]
        pooled = [capture_pooled(base, capture, prompt_ids(tokenizer, chat_mcq(c.prompt_text)), args.device)
                  for c in sample]
        means = {L: np.stack([p[0][L] for p in pooled]) for L in layers}
        lasts = {L: np.stack([p[1][L] for p in pooled]) for L in layers}
        gradient[b] = {L: mean_cosine_to_dir(means[L], r_by_layer[L]) for L in layers}
        gradient_lasttok[b] = {L: mean_cosine_to_dir(lasts[L], r_by_layer[L]) for L in layers}
        print(f"  grad {b}: meanpool cos(a,r)@L{grad_layer}={gradient[b][grad_layer]:+.4f}  "
              f"lasttok={gradient_lasttok[b][grad_layer]:+.4f}")
    grad_seq = [gradient[b][grad_layer] for b in buckets_order]
    gate_pass = is_monotone_increasing(grad_seq)
    print(f"GRADIENT GATE (meanpool) @L{grad_layer}: {'PASS' if gate_pass else 'FAIL'}  "
          f"seq={['%+.4f' % v for v in grad_seq]}")
    (out_dir / "graded_gradient.json").write_text(json.dumps(
        {"grad_layer": grad_layer, "buckets": buckets_order, "gradient": gradient,
         "gradient_lasttok": gradient_lasttok, "seq_at_grad_layer": grad_seq,
         "monotone": gate_pass}, indent=2))
    if not gate_pass and not args.force:
        print("Gradient gate FAILED — aborting (pass --force to gather the sweep anyway).")
        capture.remove()
        sys.exit(2)

    # ── 1–3. per-bucket base-model arms ──────────────────────────────────────
    results = {"base_model": cfg["base_model"], "base_refusal": base_rate,
               "n_eval_harmful": len(harmful_eval), "layers": layers, "grad_layer": grad_layer,
               "gradient": gradient, "gradient_lasttok": gradient_lasttok,
               "gate_pass": gate_pass, "buckets": {}}

    for b in buckets_order:
        bd = buckets[b]
        rec = {"cos_a_r": gradient[b]}
        print(f"\n=== bucket {b} ===")

        # 1. headroom
        zs, _ = base_acc_for(bd.eval, k=0)
        fs, _ = base_acc_for(bd.eval, k=few, fillers=bd.filler)
        rec["headroom"] = {"zeroshot_acc": zs, "fewshot_acc": fs, "few": few}
        print(f"  headroom: zero-shot={zs:.3f}  {few}-shot={fs:.3f}")

        # 2. ICL erosion arm
        rec["icl"] = []
        for k in icl_shots:
            resids, refus, n_shots = [], [], None
            for p in harmful_eval:
                msgs = icl_mcq(tokenizer, bd.filler[:k], user_turn(p), mc, ft)
                n_shots = (len(msgs) - 1) // 2
                ids = prompt_ids(tokenizer, msgs)
                resids.append(capture_resid(base, capture, ids, args.device))
                refus.append(gen(ids))
            rate, per_layer, _ = condition_metrics(resids, refus, base_resids, base_rate, layers, r_by_layer)
            acc, _ = base_acc_for(bd.eval, k=k, fillers=bd.filler)
            rec["icl"].append({"dose": k, "n_shots": n_shots, "refusal_rate": rate,
                               "delta_refusal": base_rate - rate, "task_acc": acc,
                               "task_gain": (acc - zs) if not np.isnan(acc) else float("nan"),
                               "per_layer": per_layer})
            print(f"  ICL k={k} (~{n_shots} shots): refusal={rate:.3f} dRefusal={base_rate-rate:+.3f} acc={acc:.3f}")

        # 3. steering / breadth arm — fit per-case ICL shifts on the fit slice
        A, ICL = {L: [] for L in layers}, {L: [] for L in layers}
        for c in bd.fit:
            a = resid(chat_mcq(c.prompt_text))
            i = resid(icl_mcq(tokenizer, bd.filler[:few], chat_mcq(c.prompt_text), mc, ft))
            for L in layers:
                A[L].append(a[L]); ICL[L].append(i[L])
        A = {L: np.stack(A[L]) for L in layers}
        Delta = {L: np.stack(ICL[L]) - A[L] for L in layers}
        d_mean = {L: Delta[L].mean(0) for L in layers}
        rec["steer"] = {"base_acc": zs, "runs": []}
        for mode, lt in [("all", False), ("last", True)]:
            for a in alphas:
                mk = lambda a=a, lt=lt: AdditionSteeringHook(
                    base, {L: torch.tensor(a * d_mean[L]) for L in layers}, last_token=lt)
                acc_m, _ = base_acc_for(bd.eval, mk)
                ref_m = steered_refusal(mk)
                rec["steer"]["runs"].append({"method": "meanvec", "mode": mode, "alpha": a,
                                             "acc": acc_m, "refusal": ref_m})
                print(f"  steer meanvec[{mode}] a={a}: acc={acc_m:.3f} refusal={ref_m:.3f}")
        for lam in lambdas:
            maps = ridge_maps(A, Delta, layers, lam)
            for mode, lt in [("all", False), ("last", True)]:
                for a in alphas:
                    mk = lambda a=a, lt=lt, maps=maps: LinearConditionalSteerHook(
                        base, maps, a, last_token=lt)
                    acc_w, _ = base_acc_for(bd.eval, mk)
                    ref_w = steered_refusal(mk)
                    rec["steer"]["runs"].append({"method": "condmap", "mode": mode, "lambda": lam,
                                                 "alpha": a, "acc": acc_w, "refusal": ref_w})
                    print(f"  steer condmap[{mode}] lam={lam} a={a}: acc={acc_w:.3f} refusal={ref_w:.3f}")

        results["buckets"][b] = rec

    # ── 4. LoRA arm (attach adapters AFTER all base work) ────────────────────
    lora_model = None
    for b in buckets_order:
        adir = adapter_root / b
        if not adir.exists():
            print(f"  [skip LoRA] adapter missing: {adir}")
            continue
        if lora_model is None:
            lora_model = PeftModel.from_pretrained(base, str(adir), adapter_name=b)
        else:
            lora_model.load_adapter(str(adir), adapter_name=b)
        lora_model.set_adapter(b)
        resids = [capture_resid(lora_model, capture, prompt_ids(tokenizer, user_turn(p)), args.device)
                  for p in harmful_eval]
        refus = [generate_reply(lora_model, tokenizer, prompt_ids(tokenizer, user_turn(p)), args.device, max_new)
                 for p in harmful_eval]
        rate, per_layer, _ = condition_metrics(resids, refus, base_resids, base_rate, layers, r_by_layer)
        acc, _ = mcq_accuracy(tokenizer, buckets[b].eval,
                              lambda ids: generate_reply(lora_model, tokenizer, ids, args.device, max_new),
                              mc, max_new)
        zs = results["buckets"][b]["headroom"]["zeroshot_acc"]
        results["buckets"][b]["lora"] = {"refusal_rate": rate, "delta_refusal": base_rate - rate,
                                         "task_acc": acc, "task_gain": acc - zs, "per_layer": per_layer}
        print(f"  LoRA {b}: refusal={rate:.3f} dRefusal={base_rate-rate:+.3f} acc={acc:.3f}")
    capture.remove()

    (out_dir / "graded_risk_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_dir}/graded_risk_sweep.json")


if __name__ == "__main__":
    main()
