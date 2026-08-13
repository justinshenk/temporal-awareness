"""Locate the capability-critical directions of δ in variance-space (oracle PCA-band ablation).

Collect base-trajectory δ at layer L, PCA it, then run the lockstep oracle injecting **only** the
part of the true shift inside one PCA band (``a + Π_band(δ_true)``). If the high-energy band (format)
recovers, computation is high-variance (the MLP had it open-loop → its 0% is closed-loop). If only the
low-energy tail recovers, computation is low-variance → an MSE fit misses it → whitening is the lever.

    uv run python -m scripts.attribution.lockstep_pca_band \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 \
        --bands full,top8,top64,top256,tail64 --n-pca 100 --n-contrast 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import (
    get_task,
    load_base_and_lora,
    load_contrast,
    prompt_token_ids,
)
from scripts.attribution.nonlinear_delta_gsm8k import collect_base_traj
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.delta_subspace import energy_fraction, pca_bands
from src.probes.attribution.lockstep_oracle import (
    OverwriteResidualHook,
    lockstep_generate,
    projected_injection,
)
from src.probes.extraction import PerTokenResidualCapture


def band_columns(V: torch.Tensor, spec: str) -> torch.Tensor:
    """Resolve a band spec (``full`` / ``topK`` / ``tailK``) to eigenvector columns of ``V`` (d, d)."""
    d = V.shape[1]
    if spec == "full":
        return V
    if spec.startswith("top"):
        return V[:, :int(spec[3:])]
    if spec.startswith("tail"):
        return V[:, int(spec[4:]):]
    raise ValueError(f"bad band spec: {spec}")


def make_band_fns(base, lora, capture, inject, L, Vband):
    def capture_residuals(S):
        capture.clear()
        with lora.disable_adapter(), capture.capturing():
            base(S, use_cache=False)                 # base residual a
        a = capture.captured[L]
        capture.clear()
        with capture.capturing():
            base(S, use_cache=False)                 # LoRA residual on same context
        lora_resid = capture.captured[L]
        return {L: projected_injection(a, lora_resid, Vband.to(a.dtype))}

    def base_logits(S):
        with inject.injecting(), lora.disable_adapter():
            return base(S, use_cache=False).logits

    return capture_residuals, base_logits


def eval_band(base, lora, tok, problems, device, L, Vband, max_new, task):
    capture = PerTokenResidualCapture(base, [L])
    inject = OverwriteResidualHook(base, [L])
    cap_fn, logit_fn = make_band_fns(base, lora, capture, inject, L, Vband)
    eos = tok.eos_token_id
    correct = 0
    for q, gold in problems:
        pid = prompt_token_ids(tok, q, device, task)
        out = lockstep_generate(pid, cap_fn, logit_fn, inject, [L], max_new, eos, device)
        text = tok.decode(out[0][pid.shape[1]:], skip_special_tokens=True)
        correct += bool(task.score(text, gold))
    capture.remove()
    inject.remove()
    return correct / len(problems)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--bands", default="full,top8,top64,top256,tail64")
    ap.add_argument("--n-pca", type=int, default=100, help="train problems for δ PCA")
    ap.add_argument("--n-contrast", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--task", default=None, help="task registry key (default: config 'task' or gsm8k)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, L = cfg["device"], args.layer
    bands = args.bands.split(",")
    task = get_task(args.task or cfg.get("task", "gsm8k"))

    print(f"Loading {cfg['base_model']} + adapter (task={task.name}) ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)

    pca_problems = task.problems(cfg.get("collect", {}).get("split", "train"), args.n_pca,
                                 skip=0, seed=cfg["seed"])
    print(f"Collecting δ at L{L} over {len(pca_problems)} problems for PCA ...", flush=True)
    _, D = collect_base_traj(base, lora, tok, pca_problems, device, L, args.max_new, task)
    print(f"  PCA over {D.shape[0]} tokens ...", flush=True)
    V, lam = pca_bands(D)
    V = V.to(torch.float32)

    contrast = load_contrast(cfg, task)[:args.n_contrast]
    results = {"task": task.name, "layer": L, "n_pca_tokens": int(D.shape[0]),
               "n_contrast": len(contrast), "max_new": args.max_new, "bands": {}}
    print(f"\nband        energy_frac   recovery   ({len(contrast)} contrast problems)", flush=True)
    for spec in bands:
        Vb = band_columns(V, spec)
        k = Vb.shape[1]
        ef = (1.0 if spec == "full"
              else energy_fraction(lam, k) if spec.startswith("top")
              else 1.0 - energy_fraction(lam, V.shape[1] - k))
        acc = eval_band(base, lora, tok, contrast, device, L, Vb, args.max_new, task)
        results["bands"][spec] = {"k": int(k), "energy_frac": float(ef), "recovery": acc}
        print(f"{spec:10s}  {ef:6.3f}        {acc:.3f}", flush=True)

    # GSM8K keeps its original filename so the committed band cliff stays addressable.
    stem = f"lockstep_pca_band_L{L}" if task.name == "gsm8k" else f"lockstep_pca_band_{task.name}_L{L}"
    out_path = Path(args.out) if args.out else Path(cfg["output"]["steer_json"]).parent / f"{stem}.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
