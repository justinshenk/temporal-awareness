"""Learn the linear route-alignment map W: Δh_L (LoRA shift) -> Δh_B (base ICL shift).

For a set of prompts, per layer:
    Δh_B = resid(base, prompt+ICL) - resid(base, prompt)     (activation-route / ICL shift)
    Δh_L = resid(LoRA, prompt)     - resid(base, prompt)     (weight-route / LoRA shift)
Fit  W Δh_L ≈ Δh_B  by ridge (dual form), on a held-out split. Then test the prediction
that, on harmful prompts (where the routes diverge), W annihilates the refusal direction:
‖W r̂‖ ≪ ‖r̂‖ — i.e. the route difference IS the refusal component, stated linearly.

Reported per layer (held-out): relative residual, cos(W Δh_L, Δh_B) vs the identity baseline
cos(Δh_L, Δh_B), and ‖W r̂‖/‖r̂‖.

    uv run python -m scripts.safety.run_route_map --config configs/safety/route_safety_qwen.yaml \
        --adapter results/safety/qwen_sweep/adapter_d600
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

from scripts.safety.extract_refusal_shifts import capture_resid, prompt_ids, set_seed, user_turn
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, icl_messages, select_valid_indices
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default="results/safety/qwen_sweep/adapter_d600")
    ap.add_argument("--icl-k", type=int, default=16)
    ap.add_argument("--n-fit", type=int, default=80)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, mc, ft = cfg["extract"]["layers"], cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"]

    harmful, harmless = load_harmful(), load_harmless()
    nh = cfg["direction"]["n_harmful"]
    h_fit = harmful[nh:nh + args.n_fit]
    h_eval = harmful[nh + args.n_fit:nh + args.n_fit + args.n_eval]
    h_rfit, s_rfit = harmful[:nh], harmless[:cfg["direction"]["n_harmless"]]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    fillers = build_cases(ds, valid[:cfg["ddxplus"]["n_filler"]], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(model, p, icl=False):
        msgs = icl_messages(tokenizer, fillers[:args.icl_k], user_turn(p), mc, ft) if icl else user_turn(p)
        return capture_resid(model, capture, prompt_ids(tokenizer, msgs), args.device)

    # refusal direction r (base, harmful vs harmless)
    hr = [resid(base, p) for p in h_rfit]
    sr = [resid(base, p) for p in s_rfit]
    r = {L: refusal_direction(np.stack([x[L] for x in hr]), np.stack([x[L] for x in sr])) for L in layers}

    # BASE resids must be collected BEFORE wrapping (PeftModel injects LoRA into `base`).
    def base_pairs(prompts):
        return [(resid(base, p), resid(base, p, icl=True)) for p in prompts]  # (clean, icl)
    fit_base, eval_base = base_pairs(h_fit), base_pairs(h_eval)

    lora = PeftModel.from_pretrained(base, args.adapter).eval()
    fit_lclean = [resid(lora, p) for p in h_fit]
    eval_lclean = [resid(lora, p) for p in h_eval]
    capture.remove()

    def build(base_list, lclean_list):
        dB, dL = {L: [] for L in layers}, {L: [] for L in layers}
        for (bclean, bicl), lclean in zip(base_list, lclean_list):
            for L in layers:
                dB[L].append(bicl[L] - bclean[L])      # Δh_B = ICL shift (base)
                dL[L].append(lclean[L] - bclean[L])    # Δh_L = LoRA shift
        return {L: np.stack(dB[L]) for L in layers}, {L: np.stack(dL[L]) for L in layers}

    HB_fit, HL_fit = build(fit_base, fit_lclean)
    HB_ev, HL_ev = build(eval_base, eval_lclean)
    for d in (HB_fit, HL_fit, HB_ev, HL_ev):       # bf16 LoRA forwards can emit NaN/Inf
        for L in layers:
            d[L] = np.nan_to_num(d[L], nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    for L in layers:
        HL = HL_fit[L].astype(np.float64)            # (n,d)
        HB = HB_fit[L].astype(np.float64)
        K = HL @ HL.T                                # (n,n)
        reg = args.lam * np.trace(K) / K.shape[0]
        if not np.isfinite(reg) or reg <= 0:
            reg = 1.0
        M = np.linalg.inv(K + reg * np.eye(K.shape[0]))
        # W = HBᵀ (Kᵀ?) ... dual ridge: W x = HBᵀ-cols ... use W·x = HB.T @ (M @ (HL @ x))
        def W_apply(x, HL=HL, HB=HB, M=M):
            return HB.T @ (M @ (HL @ x))             # (d,)

        rhat = r[L] / np.linalg.norm(r[L])
        # held-out: cos(W Δh_L, Δh_B) + residual; and the r-component of the shifts in vs out
        cos_W, cos_id, res, cos_in_r, cos_out_r = [], [], [], [], []
        for i in range(HB_ev[L].shape[0]):
            dl, db = HL_ev[L][i].astype(np.float64), HB_ev[L][i].astype(np.float64)
            pred = W_apply(dl)
            cos_W.append(cos(pred, db))
            cos_id.append(cos(dl, db))
            res.append(np.linalg.norm(pred - db) / (np.linalg.norm(db) + 1e-9))
            cos_in_r.append(cos(dl, rhat))     # refusal alignment of the LoRA shift (in)
            cos_out_r.append(cos(pred, rhat))  # refusal alignment after the map (out)
        wr = W_apply(rhat)
        # random-direction baseline: W's typical gain on a unit vector (to normalize ‖W r̂‖)
        rng = np.random.default_rng(cfg["seed"] + L)
        rand_gains = []
        for _ in range(20):
            u = rng.standard_normal(rhat.shape[0])
            u /= np.linalg.norm(u)
            rand_gains.append(np.linalg.norm(W_apply(u)))
        baseline_gain = float(np.mean(rand_gains))
        # task direction = unit mean ICL shift; how much does W preserve it?
        dtask = HB.mean(0)
        dtask = dtask / (np.linalg.norm(dtask) + 1e-9)
        rows.append({
            "layer": L,
            "cos_id(dL,dB)": float(np.mean(cos_id)),
            "cos_W(WdL,dB)": float(np.mean(cos_W)),
            "rel_residual": float(np.mean(res)),
            "cos_in_r(dL,r)": float(np.mean(cos_in_r)),           # refusal alignment going IN
            "cos_out_r(WdL,r)": float(np.mean(cos_out_r)),        # refusal alignment coming OUT
            "Wr_over_baseline": float(np.linalg.norm(wr) / (baseline_gain + 1e-9)),
        })
        print(f"  L{L:2d}: cos_id={rows[-1]['cos_id(dL,dB)']:+.3f} -> cos_W={rows[-1]['cos_W(WdL,dB)']:+.3f} "
              f"resid={rows[-1]['rel_residual']:.2f} | cos(dL,r)={rows[-1]['cos_in_r(dL,r)']:+.3f} "
              f"-> cos(WdL,r)={rows[-1]['cos_out_r(WdL,r)']:+.3f}")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "route_map.json").write_text(json.dumps({"lam": args.lam, "icl_k": args.icl_k, "per_layer": rows}, indent=2))
    print(f"\nSaved {out_dir}/route_map.json")


if __name__ == "__main__":
    main()
