"""Can a nonlinear δ estimator clear the steering bar the linear ridge can't?

End to end at one layer L: (1) collect base-trajectory ``(a, δ=lora−base)`` over train problems;
(2) fit a DeltaMLP ``f(a)≈δ``; (3) report val geometry vs the ridge baseline; (4) measure GSM8K
recovery on the test contrast set by steering base with ``a + f(a)`` (NonlinearSteerHook, ordinary
KV-cache generation). Baselines printed alongside: base (0), LoRA (1 on contrast), ridge steer
(≈0.05), oracle (≈0.75 at L20).

    uv run python -m scripts.attribution.nonlinear_delta_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n-train 200 --hidden 4096
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.attribution.attribution_common import (
    generate_cot_ids,
    get_task,
    load_base_and_lora,
    load_contrast,
    task_accuracy,
)
from scripts.attribution.collect_cot_residuals import teacher_force_capture
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.cot_collection import assemble_blocks, cot_token_slice
from src.probes.attribution.nonlinear_estimator import NonlinearSteerHook, fit_delta_mlp
from src.probes.attribution.shift_geometry import shift_geometry
from src.probes.safety.steering_hook import LinearPrimalSteerHook
from src.probes.extraction import PerTokenResidualCapture


def collect_base_traj(base, lora, tok, problems, device, L, max_new, task="gsm8k"):
    """Stack base-trajectory (a, δ) at layer L over the CoT tokens of every problem."""
    capture = PerTokenResidualCapture(base, [L])
    A, D = [], []
    for i, (q, _gold) in enumerate(problems):
        with lora.disable_adapter():
            full_ids, plen = generate_cot_ids(base, tok, q, device, max_new, task)
        if full_ids.shape[1] - plen <= 0:
            continue
        sl = cot_token_slice(plen, full_ids.shape[1])
        with lora.disable_adapter():
            base_cap = teacher_force_capture(base, capture, full_ids)
        lora_cap = teacher_force_capture(lora, capture, full_ids)
        a, d = assemble_blocks(base_cap, lora_cap, L, sl)
        A.append(a.float())
        D.append(d.float())
        if (i + 1) % 25 == 0:
            print(f"  collected {i+1}/{len(problems)} problems", flush=True)
    capture.remove()
    return torch.cat(A), torch.cat(D)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=200, help="train-split problems for fitting f")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--n-contrast", type=int, default=20)
    ap.add_argument("--task", default=None, help="task registry key (default: config 'task' or gsm8k)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    task = get_task(args.task or cfg.get("task", "gsm8k"))
    device, L = cfg["device"], args.layer

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)

    # --- collect base-trajectory shifts on the train split (disjoint from the test contrast set) ---
    train_problems = task.problems(cfg["collect"]["split"], args.n_train, skip=0, seed=cfg["seed"])
    print(f"Collecting base-trajectory (a, δ) at L{L} over {len(train_problems)} train problems ...", flush=True)
    A, D = collect_base_traj(base, lora, tok, train_problems, device, L, args.max_new, task)
    n = A.shape[0]
    n_val = max(1, n // 10)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    a_tr, d_tr, a_val, d_val = A[tr_idx], D[tr_idx], A[val_idx], D[val_idx]
    print(f"  {n} CoT tokens → {len(tr_idx)} train / {len(val_idx)} val", flush=True)

    # --- fit the nonlinear estimator ---
    print(f"Fitting DeltaMLP(hidden={args.hidden}) ...", flush=True)
    mlp, fit_metrics = fit_delta_mlp(a_tr, d_tr, a_val, d_val, hidden=args.hidden,
                                     epochs=args.epochs, device=device, seed=0, verbose=True)
    print(f"  best val: cos={fit_metrics['val_cosine']:+.3f} R²={fit_metrics['val_r2']:+.3f} "
          f"(epoch {fit_metrics['epoch']})", flush=True)

    # --- ridge baseline geometry on the same val tokens ---
    W = torch.load(Path(cfg["output"]["maps_dir"]) / f"W_L{L}.pt")["W"].to(torch.float32)
    ridge_pred = (a_val.float() @ W.T.to(a_val.device)).cpu().numpy()
    ridge_geo = shift_geometry(ridge_pred, d_val.cpu().numpy())
    print(f"  ridge baseline (same val): cos={ridge_geo['mean_cosine']:+.3f} R²={ridge_geo['r2']:+.3f}", flush=True)

    # --- recovery on the test contrast set ---
    contrast = load_contrast(cfg, task)[:args.n_contrast]
    print(f"\nRecovery on {len(contrast)} contrast problems (max_new={args.max_new}):", flush=True)
    with lora.disable_adapter():
        base_acc = task_accuracy(base, tok, contrast, device, args.max_new, task)
    lora_acc = task_accuracy(lora, tok, contrast, device, args.max_new, task)

    rhook = LinearPrimalSteerHook(base, {L: W}, 1.0)
    with lora.disable_adapter():
        ridge_acc = task_accuracy(base, tok, contrast, device, args.max_new, task)
    rhook.remove()

    nhook = NonlinearSteerHook(base, mlp, L, alpha=1.0)
    with lora.disable_adapter():
        nl_acc = task_accuracy(base, tok, contrast, device, args.max_new, task)
    nhook.remove()

    results = {
        "task": task.name, "layer": L, "n_train_problems": len(train_problems), "n_train_tokens": int(len(tr_idx)),
        "hidden": args.hidden, "n_contrast": len(contrast), "max_new": args.max_new,
        "geometry_val": {"nonlinear": {"cos": fit_metrics["val_cosine"], "r2": fit_metrics["val_r2"]},
                          "ridge": {"cos": ridge_geo["mean_cosine"], "r2": ridge_geo["r2"]}},
        "recovery": {"base": base_acc, "ridge_steer": ridge_acc, "nonlinear_steer": nl_acc,
                     "lora": lora_acc},
    }
    print("\n=== SUMMARY ===", flush=True)
    print(f"  geometry(val)  ridge: cos={ridge_geo['mean_cosine']:+.3f} R²={ridge_geo['r2']:+.3f}"
          f"   nonlinear: cos={fit_metrics['val_cosine']:+.3f} R²={fit_metrics['val_r2']:+.3f}", flush=True)
    print(f"  recovery       base={base_acc:.3f}  ridge_steer={ridge_acc:.3f}  "
          f"nonlinear_steer={nl_acc:.3f}  lora={lora_acc:.3f}", flush=True)

    stem = f"nonlinear_delta_L{L}" if task.name == "gsm8k" else f"nonlinear_delta_{task.name}_L{L}"
    out_path = Path(args.out) if args.out else Path(cfg["output"]["steer_json"]).parent / f"{stem}.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    torch.save(mlp.state_dict(), Path(cfg["output"]["maps_dir"]) / f"delta_mlp_L{L}.pt")
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
