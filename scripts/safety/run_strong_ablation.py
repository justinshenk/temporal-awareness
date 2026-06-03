"""Stronger (per-layer, all-layer) ICL-guided ablation, on weaker + stronger LoRA.

Fixes the underpowered single-direction ablation (which only took base refusal 0.98->0.60).
Computes per-layer directions at ALL decoder layers and ablates each layer's own
direction. Targets the threshold-erosion 75ex LoRA (less weight rewrite) and the 600ex
for contrast. ŵ is the ICL-guided direction (LoRA move minus its ICL-aligned part); r is
the label-based refusal direction (control); random is the specificity control.

    uv run python -m scripts.safety.run_strong_ablation \
        --config configs/safety/route_safety_qwen.yaml --icl-k 16
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
from src.probes.safety.ablation_hook import PerLayerAblationHook
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless

ADAPTERS = {"75": "results/safety/qwen_sweep/adapter_d75",
            "600": "results/safety/qwen_sweep/adapter_d600"}


def unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--icl-k", type=int, default=16)
    ap.add_argument("--n-fit", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    mc, ft, max_new = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"], cfg["eval"]["max_new"]
    out_dir = Path(cfg["output"]["dir"])

    harmful, harmless = load_harmful(), load_harmless()
    nh = cfg["direction"]["n_harmful"]
    h_rfit = harmful[:nh]
    h_sfit = harmful[nh:nh + args.n_fit]
    es = nh + args.n_fit
    h_eval = harmful[es:es + cfg["eval"]["n_harmful"]]
    s_rfit = harmless[:cfg["direction"]["n_harmless"]]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf, nt = cfg["ddxplus"]["n_filler"], cfg["ddxplus"]["n_task_eval"]
    fillers = build_cases(ds, valid[:nf], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    task_eval = build_cases(ds, valid[nf:nf + nt], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])

    all_layers = list(range(28))
    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, all_layers)

    def resid(model, p, k=0):
        msgs = icl_messages(tokenizer, fillers[:k], user_turn(p), mc, ft) if k else user_turn(p)
        return capture_resid(model, capture, prompt_ids(tokenizer, msgs), args.device)

    # r (label-based) and u (ICL task direction) at every layer
    hr = [resid(base, p) for p in h_rfit]
    sr = [resid(base, p) for p in s_rfit]
    r_all = {L: refusal_direction(np.stack([x[L] for x in hr]), np.stack([x[L] for x in sr])) for L in all_layers}
    base_fit = [resid(base, p) for p in h_sfit]
    icl_fit = [resid(base, p, args.icl_k) for p in h_sfit]
    u_all = {L: unit(np.mean([i[L] - b[L] for i, b in zip(icl_fit, base_fit)], axis=0)) for L in all_layers}

    def measure(model):
        return {"refusal": refusal_rate([generate_reply(model, tokenizer, prompt_ids(tokenizer, user_turn(p)),
                                                         args.device, max_new) for p in h_eval]),
                "acc": ddxplus_accuracy(model, tokenizer, task_eval, args.device, mc)[0]}

    results = {}
    # Build one wrapped model; switch adapters via load/set.
    lora_model = None
    # base reference + per-layer r sanity (computed once, adapter disabled later)
    for tag, adapter in ADAPTERS.items():
        name = f"d{tag}"
        if lora_model is None:
            lora_model = PeftModel.from_pretrained(base, adapter, adapter_name=name).eval()
        else:
            lora_model.load_adapter(adapter, adapter_name=name)
        lora_model.set_adapter(name)

        lora_fit = [resid(lora_model, p) for p in h_sfit]
        w_all = {}
        for L in all_layers:
            lora_m = np.mean([lo[L] - b[L] for lo, b in zip(lora_fit, base_fit)], axis=0)
            w_all[L] = unit(lora_m - np.dot(lora_m, u_all[L]) * u_all[L])

        rng = np.random.default_rng(cfg["seed"])
        rand = {L: unit(rng.standard_normal(w_all[L].shape[0])) for L in all_layers}
        dirsets = {"none": None, "ablate_w": w_all, "ablate_r": r_all, "ablate_random": rand}

        res = {}
        if tag == "75":  # base reference + sanity once
            with lora_model.disable_adapter():
                res["base"] = measure(lora_model)
                hook = PerLayerAblationHook(base, {L: torch.tensor(r_all[L]) for L in all_layers})
                res["base_ablate_r_sanity"] = measure(lora_model)
                hook.remove()
            print(f"  base refusal={res['base']['refusal']:.3f} | base+ablate_r(sanity)="
                  f"{res['base_ablate_r_sanity']['refusal']:.3f} (expect big drop)")

        for dname, dirs in dirsets.items():
            hook = None if dirs is None else PerLayerAblationHook(base, {L: torch.tensor(dirs[L]) for L in all_layers})
            res[dname] = measure(lora_model)
            if hook:
                hook.remove()
            print(f"  LoRA-{tag} [{dname}]: refusal={res[dname]['refusal']:.3f} acc={res[dname]['acc']:.3f}")
        results[tag] = res

    capture.remove()
    out = {"base_model": cfg["base_model"], "icl_k": args.icl_k, "n_eval": len(h_eval), "results": results}
    (out_dir / "strong_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_dir}/strong_ablation.json")


if __name__ == "__main__":
    main()
