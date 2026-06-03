"""Confirm the shared task axis, then remove the LoRA's refusal side-effect using ICL.

ICL gives the "clean" task move on harmful prompts (task gain, ~no refusal erosion). The
LoRA move = that shared task component + a LoRA-specific residual. We:

  1. CONFIRM geometry, per layer (directions fit on a held-out harmful split):
       u  = unit mean ICL shift            (task axis)
       w  = unit (mean LoRA shift  minus its u-component)   (LoRA-specific residual)
     and report cos(icl,lora) (shared component), cos(u, r) (≈0 expected: task off the
     refusal axis), cos(w, r) (large expected: the LoRA residual IS the refusal axis).

  2. REMOVE it: ablate w (the LoRA-minus-ICL direction) from the LoRA model's residual
     stream and check refusal is restored while DDXPlus accuracy is preserved. Controls:
     ablate r (known recipe) and a random direction (specificity).

    uv run python -m scripts.safety.run_icl_guided_ablation \
        --config configs/safety/route_safety_qwen.yaml \
        --adapter results/safety/qwen_sweep/adapter_d600 --icl-k 16
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import (
    capture_resid,
    generate_reply,
    prompt_ids,
    set_seed,
    user_turn,
)
from scripts.safety.run_route_safety_sweep import ddxplus_accuracy
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, icl_messages, select_valid_indices
from src.probes.lora_icl.subspace_metrics import vector_cosine
from src.probes.safety.ablation_hook import DirectionalAblationHook
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def mean_shift(variant_resids, base_resids, layer):
    return np.mean([v[layer] - b[layer] for v, b in zip(variant_resids, base_resids)], axis=0)


def refusal_on(model, tokenizer, prompts, device, max_new):
    return refusal_rate([generate_reply(model, tokenizer, prompt_ids(tokenizer, user_turn(p)), device, max_new)
                         for p in prompts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default="results/safety/qwen_sweep/adapter_d600")
    ap.add_argument("--icl-k", type=int, default=16)
    ap.add_argument("--n-fit", type=int, default=40, help="harmful prompts to fit u/w/r")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers = cfg["extract"]["layers"]
    mc, ft, max_new = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"], cfg["eval"]["max_new"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    harmful, harmless = load_harmful(), load_harmless()
    # disjoint slices: direction-fit (r), shift-fit (u/w), held-out eval
    harmful_rfit = harmful[:cfg["direction"]["n_harmful"]]
    harmful_sfit = harmful[cfg["direction"]["n_harmful"]:cfg["direction"]["n_harmful"] + args.n_fit]
    es = cfg["direction"]["n_harmful"] + args.n_fit
    harmful_eval = harmful[es:es + cfg["eval"]["n_harmful"]]
    harmless_rfit = harmless[:cfg["direction"]["n_harmless"]]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf, nt = cfg["ddxplus"]["n_filler"], cfg["ddxplus"]["n_task_eval"]
    fillers = build_cases(ds, valid[:nf], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    task_eval = build_cases(ds, valid[nf:nf + nt], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(model, p, ctx_k=0):
        msgs = icl_messages(tokenizer, fillers[:ctx_k], user_turn(p), mc, ft) if ctx_k else user_turn(p)
        return capture_resid(model, capture, prompt_ids(tokenizer, msgs), args.device)

    # r direction (base, harmful vs harmless)
    hr = [resid(base, p) for p in harmful_rfit]
    sr = [resid(base, p) for p in harmless_rfit]
    r_by_layer = {L: refusal_direction(np.stack([x[L] for x in hr]), np.stack([x[L] for x in sr])) for L in layers}

    # shift-fit residuals: base_clean, ICL (k shots), LoRA — on the shift-fit harmful split
    base_fit = [resid(base, p) for p in harmful_sfit]
    icl_fit = [resid(base, p, args.icl_k) for p in harmful_sfit]
    lora_model = PeftModel.from_pretrained(base, args.adapter).eval()
    lora_fit = [resid(lora_model, p) for p in harmful_sfit]

    geom, u_by, w_by = [], {}, {}
    for L in layers:
        icl_m = mean_shift(icl_fit, base_fit, L)
        lora_m = mean_shift(lora_fit, base_fit, L)
        u = unit(icl_m)
        perp = lora_m - np.dot(lora_m, u) * u
        w = unit(perp)
        u_by[L], w_by[L] = u, w
        geom.append({
            "layer": L,
            "cos_icl_lora": vector_cosine(icl_m, lora_m),
            "cos_u_r": vector_cosine(u, r_by_layer[L]),
            "cos_w_r": vector_cosine(w, r_by_layer[L]),
            "lora_frac_along_u": float(abs(np.dot(lora_m, u)) / (np.linalg.norm(lora_m) + 1e-9)),
            "lora_frac_perp": float(np.linalg.norm(perp) / (np.linalg.norm(lora_m) + 1e-9)),
        })
        print(f"  L{L:2d}: cos(icl,lora)={geom[-1]['cos_icl_lora']:+.3f} "
              f"cos(u,r)={geom[-1]['cos_u_r']:+.3f} cos(w,r)={geom[-1]['cos_w_r']:+.3f} "
              f"|perp|/|lora|={geom[-1]['lora_frac_perp']:.2f}")
    capture.remove()

    # ablation layer = strongest |cos(w, r)|
    star = max(layers, key=lambda L: abs(next(g["cos_w_r"] for g in geom if g["layer"] == L)))
    rng = np.random.default_rng(cfg["seed"])
    rand_dir = unit(rng.standard_normal(w_by[star].shape[0]))
    directions = {"none": None, "ablate_w": w_by[star], "ablate_r": r_by_layer[star], "ablate_random": rand_dir}

    def measure(model):
        return {"refusal": refusal_on(model, tokenizer, harmful_eval, args.device, max_new),
                "acc": ddxplus_accuracy(model, tokenizer, task_eval, args.device, mc)[0]}

    conditions = {}
    # TRUE base requires disabling the adapter (PeftModel wrapped `base` in place).
    with lora_model.disable_adapter():
        conditions["base"] = measure(lora_model)
        print(f"  base: refusal={conditions['base']['refusal']:.3f} acc={conditions['base']['acc']:.3f}")
        # sanity: ablating r on the base should DROP refusal (proves the hook bites)
        hook = DirectionalAblationHook(base, torch.tensor(r_by_layer[star]))
        conditions["base_ablate_r_sanity"] = measure(lora_model)
        hook.remove()
        print(f"  base+ablate_r (sanity, expect refusal DROP): "
              f"refusal={conditions['base_ablate_r_sanity']['refusal']:.3f}")
    # LoRA conditions (adapter active)
    for name, d in directions.items():
        hook = None if d is None else DirectionalAblationHook(base, torch.tensor(d))
        conditions[f"lora_{name}"] = measure(lora_model)
        if hook:
            hook.remove()
        print(f"  LoRA [{name}]: refusal={conditions[f'lora_{name}']['refusal']:.3f} "
              f"acc={conditions[f'lora_{name}']['acc']:.3f}")

    result = {"base_model": cfg["base_model"], "adapter": args.adapter, "icl_k": args.icl_k,
              "ablation_layer": star, "n_eval": len(harmful_eval), "geometry": geom, "conditions": conditions}
    (out_dir / "icl_guided_ablation.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out_dir}/icl_guided_ablation.json (ablation layer L{star})")


if __name__ == "__main__":
    main()
