"""Why does task-vector steering cliff at high alpha — overwrite or off-manifold?

Diagnostics (no generation):
  1. Norm trajectory: ‖α·d‖ / ‖natural activation‖ per layer vs α (steer dominating ⇒ overwrite).
  2. Off-manifold energy: α·‖off-subspace(d)‖ vs the natural off-subspace scatter ‖off(a)‖,
     where the subspace is the top-k PCA of natural activations at that layer.
Fix / disambiguation (generation): re-measure task accuracy + refusal under
  - plain additive steering,
  - norm-preserving steering (rescale to original norm),
  - projection steering (project back onto the natural activation subspace),
across α. If a variant removes the high-α accuracy cliff, that names the cause and widens
the usable band.

Uses the ICL task vector d_icl_few (no finetuning needed).

    uv run python -m scripts.safety.run_steering_manifold --config configs/safety/route_safety_qwen.yaml
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
from src.probes.safety.steering_hook import (
    AdditionSteeringHook,
    NormPreservingSteeringHook,
    ProjectionSteeringHook,
)


def pca_basis(acts, var=0.90):
    """Top-k PCA (mean, V (d,k)) capturing >= var of variance from rows of `acts` (n,d)."""
    A = np.asarray(acts, np.float64)
    mean = A.mean(0)
    _, S, Vt = np.linalg.svd(A - mean, full_matrices=False)
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(np.searchsorted(cum, var) + 1)
    return mean, Vt[:k].T, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--few", type=int, default=4)
    ap.add_argument("--n-fit", type=int, default=40)
    ap.add_argument("--n-pca", type=int, default=100)
    ap.add_argument("--n-task", type=int, default=40)
    ap.add_argument("--n-harmful", type=int, default=30)
    ap.add_argument("--alphas", default="1,2,4")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, mc, ft, max_new = cfg["extract"]["layers"], cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"], cfg["eval"]["max_new"]
    alphas = [float(a) for a in args.alphas.split(",")]

    nh = cfg["direction"]["n_harmful"]
    h_eval = load_harmful()[nh:nh + args.n_harmful]
    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf = cfg["ddxplus"]["n_filler"]
    bc = lambda lo, hi: build_cases(ds, valid[lo:hi], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    fillers = bc(0, nf)
    fit = bc(nf, nf + args.n_fit)
    # task_eval kept right after fit (matches the run where steering peaks ~0.65), so the
    # fix can be tested against a real accuracy peak; PCA cases drawn afterward (disjoint).
    e0 = nf + args.n_fit
    task_eval = bc(e0, e0 + args.n_task)
    pca_cases = bc(e0 + args.n_task, e0 + args.n_task + args.n_pca)

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

    base_fit = [resid(c) for c in fit]
    icl_fit = [resid(c, args.few) for c in fit]
    d = {L: np.mean([i[L] - b[L] for i, b in zip(icl_fit, base_fit)], axis=0) for L in layers}
    nat = {L: [] for L in layers}
    for c in pca_cases:
        r = resid(c)
        for L in layers:
            nat[L].append(r[L])
    capture.remove()

    # ── diagnostics ──
    diag, bases = {}, {}
    for L in layers:
        A = np.stack(nat[L])
        mean, V, k = pca_basis(A)
        bases[L] = (torch.tensor(mean), torch.tensor(V))
        a_norm = float(np.mean(np.linalg.norm(A, axis=1)))
        off_a = float(np.mean(np.linalg.norm((A - mean) - (A - mean) @ V @ V.T, axis=1)))  # natural off-manifold scatter
        d_off = float(np.linalg.norm(d[L] - V @ (V.T @ d[L])))                              # off-subspace part of d
        d_norm = float(np.linalg.norm(d[L]))
        diag[L] = {"pca_k": k, "a_norm": a_norm, "d_norm": d_norm, "off_a": off_a,
                   "d_off_frac": d_off / (d_norm + 1e-9),
                   "norm_ratio": {a: a * d_norm / a_norm for a in alphas},
                   "offmanifold_ratio": {a: a * d_off / (off_a + 1e-9) for a in alphas}}
        print(f"  L{L:2d}: k={k} ‖a‖={a_norm:.1f} ‖d‖={d_norm:.1f} d_off_frac={d_off/d_norm:.2f} "
              f"| norm_ratio@a={ {a: round(v,2) for a,v in diag[L]['norm_ratio'].items()} } "
              f"| offmanifold_ratio={ {a: round(v,1) for a,v in diag[L]['offmanifold_ratio'].items()} }")

    # ── fix eval ──
    def measure():
        return {"task_acc": ddxplus_accuracy(base, tokenizer, task_eval, args.device, mc)[0],
                "refusal": refusal_rate([generate_reply(base, tokenizer, prompt_ids(tokenizer, user_turn(p)),
                                                        args.device, max_new) for p in h_eval])}

    results = {"base": measure()}
    print(f"  base: {results['base']}")
    # projection-only sanity (alpha=0): does projecting natural acts hurt the task?
    h = ProjectionSteeringHook(base, {L: torch.zeros_like(torch.tensor(d[L])) for L in layers}, bases)
    results["projection_only_a0"] = measure()
    h.remove()
    print(f"  projection_only (a=0 sanity): {results['projection_only_a0']}")

    # Instantiate ONE hook at a time — a list literal would register all three at once
    # (each hook attaches on construction), stacking them during every measurement.
    def make_hook(name, vecs):
        if name == "plain":
            return AdditionSteeringHook(base, vecs)
        if name == "normpreserve":
            return NormPreservingSteeringHook(base, vecs)
        return ProjectionSteeringHook(base, vecs, bases)

    for a in alphas:
        vecs = {L: torch.tensor(a * d[L]) for L in layers}
        for name in ("plain", "normpreserve", "projection"):
            hook = make_hook(name, vecs)
            results[f"{name}_a{a}"] = measure()
            hook.remove()
            print(f"  {name} a={a}: {results[f'{name}_a{a}']}")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "steering_manifold.json").write_text(json.dumps({"diagnostics": diag, "fix": results}, indent=2))
    print(f"\nSaved {out_dir}/steering_manifold.json")


if __name__ == "__main__":
    main()
