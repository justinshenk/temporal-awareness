"""Benign-twin probe: deconfound distance-to-fit from length, and test Corollary 1.

Extends run_steer_ood_probe.py with a BENIGN TWIN at each format/length:

  far    harmful=AdvBench          benign=Alpaca one-liner
  near   harmful=med-harmful       benign=med one-liner (drug dosing/side-effects)
  indist harmful=DDX block+harm    benign=DDX block+"diagnose this patient" (= the fit distribution)

Why: (1) distance-to-fit is otherwise confounded with length (indist long, far/near short);
(2) Corollary 1 — the dual-form map W·a = Σᵢ(a·aᵢ)Cᵢ depends on a only through k(a), so it CANNOT
distinguish harmful from benign at the same format. So if ‖W·a‖ is equally large on indist-harm and
indist-benign, the map applies the full task shift regardless of harm, and the verdict rides on
whether refusal holds. We also run the ICL and LoRA routes on the harmful buckets — the
weights-vs-activations stress test on in-distribution harm.

    HF_TOKEN=... uv run python -m scripts.safety.run_steer_twin_probe \
        --config configs/safety/route_safety_qwen.yaml --layer 21 \
        --adapter results/safety/qwen_sweep/adapter_d75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import capture_resid, generate_reply, prompt_ids, set_seed, user_turn
from scripts.safety.run_lora_distill import ridge_maps
from scripts.safety.run_steer_ood_probe import orthogonalize_maps
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, icl_messages, select_valid_indices
from src.probes.lora_icl.subspace_metrics import vector_cosine
from src.probes.safety.graded_harmful_data import (
    far_benign_messages,
    far_messages,
    indist_benign_messages,
    indist_interleaved_messages,
    indist_messages,
    near_benign_messages,
    near_messages,
)
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless
from src.probes.safety.steering_hook import LinearConditionalSteerHook


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--n-fit", type=int, default=120)
    ap.add_argument("--n-harmful", type=int, default=30)
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--few", type=int, default=4)
    ap.add_argument("--icl-k", type=int, default=8, help="DDXPlus demos prepended for the ICL route")
    ap.add_argument("--adapter", default="results/safety/qwen_sweep/adapter_d75",
                    help="DDXPlus LoRA adapter for the weight-route stress test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/safety/twin_probe")
    ap.add_argument("--mediators-only", action="store_true",
                    help="compute cos(W·a,r)/‖W·a‖/cos_fit for all buckets and skip all generation (fast)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers = cfg["extract"]["layers"]
    mc, ft, max_new = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"], cfg["eval"]["max_new"]
    L = args.layer
    alphas = [float(a) for a in args.alphas.split(",")]
    a_ref = max(alphas)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    harmful, harmless = load_harmful(), load_harmless()
    n_dir, n_dirless = cfg["direction"]["n_harmful"], cfg["direction"]["n_harmless"]
    harmful_fit, harmless_fit = harmful[:n_dir], harmless[:n_dirless]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf = cfg["ddxplus"]["n_filler"]
    fillers = build_cases(ds, valid[:nf], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    fit_cases = build_cases(ds, valid[nf:nf + args.n_fit], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    indist_src = build_cases(ds, valid[nf + args.n_fit:nf + args.n_fit + args.n_harmful],
                             evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(msgs):
        return capture_resid(base, capture, prompt_ids(tokenizer, msgs), args.device)

    def gen_refusal(model, msgs_list, hook_factory=None, k=0):
        hook = hook_factory() if hook_factory else None
        outs = []
        for m in msgs_list:
            packed = icl_messages(tokenizer, fillers[:k], m, mc, ft) if k else m
            outs.append(generate_reply(model, tokenizer, prompt_ids(tokenizer, packed), args.device, max_new))
        if hook:
            hook.remove()
        return refusal_rate(outs)

    # refusal direction r + the conditional map W (DDXPlus ICL shifts)
    h_res = [resid(user_turn(p)) for p in harmful_fit]
    s_res = [resid(user_turn(p)) for p in harmless_fit]
    r_by_layer = {l: refusal_direction(np.stack([x[l] for x in h_res]),
                                       np.stack([x[l] for x in s_res])) for l in layers}
    A, ICL = {l: [] for l in layers}, {l: [] for l in layers}
    for c in fit_cases:
        a = resid(chat_messages(c.prompt_text))
        i = resid(icl_messages(tokenizer, fillers[:args.few], chat_messages(c.prompt_text), mc, ft))
        for l in layers:
            A[l].append(a[l]); ICL[l].append(i[l])
    A = {l: np.stack(A[l]) for l in layers}
    Delta = {l: np.stack(ICL[l]) - A[l] for l in layers}
    centroid = {l: A[l].mean(0) for l in layers}
    maps = ridge_maps(A, Delta, layers, 1.0)
    maps_orth = orthogonalize_maps(maps, r_by_layer)
    maps_np = {l: (At.numpy(), C.numpy()) for l, (At, C) in maps.items()}

    def w_dot_a(a_vec, l):
        At, C = maps_np[l]
        return (a_vec @ At) @ C                                  # unit-alpha dual-form W·a

    # six buckets: harmful + benign twin at each format
    buckets = {
        "far_harm": far_messages(args.n_harmful, skip=n_dir),
        "far_ben": far_benign_messages(args.n_harmful, harmless, skip=n_dirless),
        "near_harm": near_messages(args.n_harmful),
        "near_ben": near_benign_messages(args.n_harmful),
        "indist_harm": indist_messages(indist_src, args.n_harmful),
        "indist_wove": indist_interleaved_messages(indist_src, args.n_harmful),
        "indist_ben": indist_benign_messages(indist_src, args.n_harmful),
    }
    order = ["far_harm", "far_ben", "near_harm", "near_ben",
             "indist_harm", "indist_wove", "indist_ben"]
    harmful_buckets = ["far_harm", "near_harm", "indist_harm", "indist_wove"]

    results = {"base_model": cfg["base_model"], "layer": L, "alphas": alphas,
               "adapter": args.adapter, "icl_k": args.icl_k, "n_harmful": args.n_harmful, "buckets": {}}

    # ── all base-model measurements first (before wrapping in PeftModel) ──────
    for name in order:
        msgs_list = buckets[name]
        acts = [resid(m) for m in msgs_list]
        cos_fit = float(np.mean([vector_cosine(a[L], centroid[L]) for a in acts]))
        shifts = [w_dot_a(a[L], L) for a in acts]
        rel_norm = float(np.mean([np.linalg.norm(s) / (np.linalg.norm(a[L]) + 1e-8)
                                  for s, a in zip(shifts, acts)]))
        cos_r = float(np.mean([vector_cosine(s, r_by_layer[L]) for s in shifts]))
        if args.mediators_only:                                  # fast path: direction + magnitude, no generation
            results["buckets"][name] = {"cos_fit": cos_fit, "wa_relnorm": rel_norm, "cos_wa_r": cos_r}
            print(f"  {name:>11}: cos_fit={cos_fit:+.3f} ‖Wa‖/‖a‖={rel_norm:.3f} cos(Wa,r)={cos_r:+.3f}")
            continue
        base_ref = gen_refusal(base, msgs_list)
        rec = {"cos_fit": cos_fit, "wa_relnorm": rel_norm, "cos_wa_r": cos_r, "base_refusal": base_ref}
        if name in harmful_buckets:                              # steered + ICL routes only where refusal is the signal
            rec["steer"] = {}
            for al in alphas:
                rec["steer"][f"naive_a{al}"] = gen_refusal(
                    base, msgs_list, lambda al=al: LinearConditionalSteerHook(base, maps, al))
                rec["steer"][f"orth_a{al}"] = gen_refusal(
                    base, msgs_list, lambda al=al: LinearConditionalSteerHook(base, maps_orth, al))
            rec["icl_refusal"] = gen_refusal(base, msgs_list, k=args.icl_k)
        results["buckets"][name] = rec
        extra = (f" base_ref={base_ref:.2f} naive(a{a_ref})={rec['steer'][f'naive_a{a_ref}']:.2f} "
                 f"orth(a{a_ref})={rec['steer'][f'orth_a{a_ref}']:.2f} icl(k{args.icl_k})={rec['icl_refusal']:.2f}"
                 if name in harmful_buckets else f" base_ref={base_ref:.2f}")
        print(f"  {name:>11}: cos_fit={cos_fit:+.3f} ‖Wa‖/‖a‖={rel_norm:.3f} cos(Wa,r)={cos_r:+.3f}|{extra}")

    # ── weight route LAST (PeftModel wraps base in-place) ────────────────────
    if not args.mediators_only and Path(args.adapter).exists():
        lora = PeftModel.from_pretrained(base, args.adapter).eval()
        for name in harmful_buckets:
            results["buckets"][name]["lora_refusal"] = gen_refusal(lora, buckets[name])
            print(f"  LoRA {name}: refusal={results['buckets'][name]['lora_refusal']:.2f}")
    else:
        print(f"  [skip LoRA] adapter missing: {args.adapter}")
    capture.remove()

    # ── Corollary-1 contrast + validity ─────────────────────────────────────
    pairs = [("far", "far_harm", "far_ben"), ("near", "near_harm", "near_ben"),
             ("indist_suffix", "indist_harm", "indist_ben"),
             ("indist_woven", "indist_wove", "indist_ben")]
    print("\nCorollary-1 contrast (‖W·a‖ harm vs benign at same format; should match):")
    for fmt, h, b in pairs:
        wh, wb = results["buckets"][h]["wa_relnorm"], results["buckets"][b]["wa_relnorm"]
        print(f"  {fmt:>7}: harm {wh:.3f} vs benign {wb:.3f}  (ratio {wh / wb:.2f})")
    wa_harm = [results["buckets"][h]["wa_relnorm"] for _, h, _ in pairs]
    valid = wa_harm[2] >= 0.8 * max(wa_harm)                     # indist-harm not knocked OOD
    print(f"validity (‖W·a‖ on indist-harm not collapsed): {valid}  seq={['%.3f' % v for v in wa_harm]}")
    print("cos(W·a, r) — direction selectivity (harm vs benign at same format):")
    for fmt, h, b in pairs:
        print(f"  {fmt:>7}: harm {results['buckets'][h]['cos_wa_r']:+.3f} vs benign {results['buckets'][b]['cos_wa_r']:+.3f}")
    if not args.mediators_only:
        ih = results["buckets"]["indist_harm"]
        print(f"ΔRefusal on indist-harm @a{a_ref}: naive {ih['base_refusal'] - ih['steer'][f'naive_a{a_ref}']:+.2f}  "
              f"orth {ih['base_refusal'] - ih['steer'][f'orth_a{a_ref}']:+.2f}  ICL {ih['base_refusal'] - ih['icl_refusal']:+.2f}")
    results["validity"] = valid
    (out_dir / "twin_probe.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {out_dir}/twin_probe.json")


if __name__ == "__main__":
    main()
