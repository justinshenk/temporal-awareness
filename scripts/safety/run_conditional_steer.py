"""Approach B: input-conditional steering to beat the fixed-vector 0.65 ceiling.

A constant mean task vector captures only the average task move. Here we fit, per layer, a
closed-form linear map W (ridge, dual form) that predicts EACH case's task shift from its
own activation, from the per-case ICL shifts:
    Δ_i = resid(base, case_i WITH ICL) − resid(base, case_i clean)   (target shift)
    a_i = resid(base, case_i clean)                                  (input)
    W   = Aᵀ (A Aᵀ + λI)⁻¹ Δ    →   steer(new) = α · (a_new W)        (input-conditional)
Stored factored as Aᵀ (d×n) and C=(AAᵀ+λI)⁻¹Δ (n×d), so the hook is two small matmuls.

Compares base, the constant mean-vector steer (the 0.65 baseline), and the conditional
map over a small (λ, α) grid — measuring BOTH task accuracy and refusal (the safety
tradeoff: more input-conditioning ⇒ closer to LoRA ⇒ may re-acquire erosion).

    uv run python -m scripts.safety.run_conditional_steer --config configs/safety/route_safety_qwen.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import capture_resid, generate_reply, prompt_ids, set_seed, user_turn
from scripts.safety.run_route_safety_sweep import ddxplus_accuracy
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, icl_messages, select_valid_indices
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.safety_data import load_harmful
from src.probes.safety.steering_hook import AdditionSteeringHook


class LinearConditionalSteerHook:
    """Add α·(hs @ Aᵀ @ C) per position — an input-conditional task-shift prediction."""

    def __init__(self, model, maps, alpha, last_token=False):
        self.maps = {li: (At.to(torch.float32), C.to(torch.float32)) for li, (At, C) in maps.items()}
        self.alpha, self.last_token, self.enabled, self._hooks = alpha, last_token, True, []
        for li in self.maps:
            self._hooks.append(model.model.layers[li].register_forward_hook(self._mk(li)))

    def _mk(self, li):
        def hook(module, inp, out):
            if not self.enabled:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            At, C = (t.to(hs.device, hs.dtype) for t in self.maps[li])
            if self.last_token:
                if hs.shape[1] <= 1:
                    return out
                hs = hs.clone()
                hs[:, -1, :] = hs[:, -1, :] + self.alpha * ((hs[:, -1, :] @ At) @ C)
            else:
                hs = hs + self.alpha * ((hs @ At) @ C)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--few", type=int, default=4)
    ap.add_argument("--n-fit", type=int, default=120)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--n-harmful", type=int, default=30)
    ap.add_argument("--lambdas", default="0.3,1,3")
    ap.add_argument("--sv-ks", default="", help="singular-vector steering ranks, e.g. 4,16")
    ap.add_argument("--modes", default="all,last", help="all,last")
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, mc, ft, max_new = cfg["extract"]["layers"], cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"], cfg["eval"]["max_new"]
    lambdas = [float(x) for x in args.lambdas.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    nh = cfg["direction"]["n_harmful"]
    h_eval = load_harmful()[nh:nh + args.n_harmful]
    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf = cfg["ddxplus"]["n_filler"]
    bc = lambda lo, hi: build_cases(ds, valid[lo:hi], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    fillers = bc(0, nf)
    ev = bc(nf, nf + args.n_eval)                       # the good split (base ~0.139)
    fit = bc(nf + args.n_eval, nf + args.n_eval + args.n_fit)

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(case, k=0):
        final = chat_messages(case.prompt_text)
        msgs = icl_messages(tokenizer, fillers[:k], final, mc, ft) if k else final
        return capture_resid(base, capture, prompt_ids(tokenizer, msgs), args.device)

    A = {L: np.stack([resid(c)[L] for c in fit]) for L in layers}        # (n,d) base acts
    # recompute icl per case once (reuse the same forward set by caching)
    icl = {L: [] for L in layers}
    for c in fit:
        r = resid(c, args.few)
        for L in layers:
            icl[L].append(r[L])
    Delta = {L: np.stack(icl[L]) - A[L] for L in layers}                  # (n,d) shifts
    capture.remove()
    d_mean = {L: Delta[L].mean(0) for L in layers}

    def measure():
        return {"task_acc": ddxplus_accuracy(base, tokenizer, ev, args.device, mc)[0],
                "refusal": refusal_rate([generate_reply(base, tokenizer, prompt_ids(tokenizer, user_turn(p)),
                                                        args.device, max_new) for p in h_eval])}

    # (2) input-conditional ridge maps, once per lambda (dual form).
    maps_by_lam = {}
    for lam in lambdas:
        maps = {}
        for L in layers:
            AL = A[L].astype(np.float64)
            K = AL @ AL.T
            reg = lam * np.trace(K) / K.shape[0]
            C = np.linalg.solve(K + reg * np.eye(K.shape[0]), Delta[L].astype(np.float64))  # (n,d)
            maps[L] = (torch.tensor(AL.T), torch.tensor(C))             # Aᵀ (d,n), C (n,d)
        maps_by_lam[lam] = maps

    # (3) singular-vector steering: project the mean shift onto the top-k right singular
    # subspace of the shift set — a fixed, denoised low-rank version of the mean vector.
    sv_ks = [int(k) for k in args.sv_ks.split(",")] if args.sv_ks else []
    sv_by_k = {}
    for k in sv_ks:
        vecs = {}
        for L in layers:
            _, _, Vt = np.linalg.svd(Delta[L].astype(np.float64), full_matrices=False)
            Vk = Vt[:k].T                                              # (d,k) top-k right singular vecs
            vecs[L] = Vk @ (Vk.T @ d_mean[L])                          # project mean onto top-k subspace
        sv_by_k[k] = vecs

    results = {"base": measure()}
    print(f"  base: {results['base']}")
    for mode in args.modes.split(","):                                  # all-position and/or last-token
        lt = mode == "last"
        for a in alphas:                                               # (1) constant mean vector
            sh = AdditionSteeringHook(base, {L: torch.tensor(a * d_mean[L]) for L in layers}, last_token=lt)
            results[f"meanvec_{mode}_a{a}"] = measure()
            sh.remove()
            print(f"  meanvec[{mode}] a={a}: {results[f'meanvec_{mode}_a{a}']}")
        for lam in lambdas:                                            # (2) input-conditional map
            for a in alphas:
                hook = LinearConditionalSteerHook(base, maps_by_lam[lam], a, last_token=lt)
                results[f"cond_{mode}_lam{lam}_a{a}"] = measure()
                hook.remove()
                print(f"  cond[{mode}] lam={lam} a={a}: {results[f'cond_{mode}_lam{lam}_a{a}']}")
        for k in sv_ks:                                               # (3) singular-vector steering
            for a in alphas:
                sh = AdditionSteeringHook(base, {L: torch.tensor(a * sv_by_k[k][L]) for L in layers}, last_token=lt)
                results[f"singvec_{mode}_k{k}_a{a}"] = measure()
                sh.remove()
                print(f"  singvec[{mode}] k={k} a={a}: {results[f'singvec_{mode}_k{k}_a{a}']}")

    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output"]["dir"]) / "conditional_steer.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {cfg['output']['dir']}/conditional_steer.json")


if __name__ == "__main__":
    main()
