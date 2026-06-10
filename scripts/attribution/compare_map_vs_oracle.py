"""Why does W·a (steering, 0% recovery) fail where true-δ (lockstep oracle, ~75%) succeeds?

For each contrast problem, take BASE's own greedy CoT (the site where steering acts), and at each
layer compare the fitted-map shift ``δ_pred = a·Wᵀ`` against the true shift ``δ_true = lora−base``
(captured by teacher-forcing the LoRA on base's tokens). Aggregate the per-token geometry
(cosine, magnitude ratio, R²) over all CoT tokens. Low cosine/R² here — despite the map's decent
held-out R² on the LoRA's *own* CoT distribution — localizes the steering null to the estimator:
the linear map cannot reach the true shift at base's acting site.

    uv run python -m scripts.attribution.compare_map_vs_oracle \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layers 8,12,16,20,24,28
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
    gsm8k_problems,
    load_base_and_lora,
)
from scripts.attribution.collect_cot_residuals import teacher_force_capture
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.cot_collection import assemble_blocks, cot_token_slice
from src.probes.attribution.shift_geometry import shift_geometry
from src.probes.extraction import PerTokenResidualCapture


def load_contrast_problems(cfg, n_eval_fallback: int):
    """Reuse the lockstep contrast set (base-fail/LoRA-solve indices) if cached."""
    cache = Path(cfg["output"]["steer_json"]).parent / "lockstep_contrast_set.json"
    meta = json.loads(cache.read_text())
    scan = gsm8k_problems(cfg["eval"]["split"], meta["n_eval"], skip=0)
    return [tuple(scan[i]) for i in meta["indices"]], meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layers", default="8,12,16,20,24,28,31")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--n-contrast", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = cfg["device"]
    layers = [int(x) for x in args.layers.split(",")]

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)
    contrast, meta = load_contrast_problems(cfg, args.n_contrast)
    contrast = contrast[:args.n_contrast]
    print(f"{len(contrast)} contrast problems (base-fail/LoRA-solve)", flush=True)

    maps_dir = Path(cfg["output"]["maps_dir"])
    W = {L: torch.load(maps_dir / f"W_L{L}.pt")["W"].to(torch.float32) for L in layers}
    sweep = json.loads(Path(cfg["output"]["sweep_json"]).read_text())
    r2_te = {sw["layer"]: sw["r2_te_star"] for sw in sweep["layers"]}

    capture = PerTokenResidualCapture(base, layers)
    pred_rows = {L: [] for L in layers}
    true_rows = {L: [] for L in layers}

    for i, (q, _gold) in enumerate(contrast):
        with lora.disable_adapter():
            full_ids, plen = generate_cot_ids(base, tok, q, device, args.max_new)  # base's own CoT
        if full_ids.shape[1] - plen <= 0:
            continue
        sl = cot_token_slice(plen, full_ids.shape[1])
        with lora.disable_adapter():
            base_cap = teacher_force_capture(base, capture, full_ids)
        lora_cap = teacher_force_capture(lora, capture, full_ids)
        for L in layers:
            a, d_true = assemble_blocks(base_cap, lora_cap, L, sl)  # a=base resid, d_true=lora−base
            d_pred = a.to(torch.float32) @ W[L].T
            pred_rows[L].append(d_pred.numpy())
            true_rows[L].append(d_true.to(torch.float32).numpy())
        print(f"  [{i+1}/{len(contrast)}] CoT tokens={full_ids.shape[1]-plen}", flush=True)
    capture.remove()

    results = {"n_contrast": len(contrast), "max_new": args.max_new, "per_layer": {}}
    print("\nL : cos(W·a, δ_true)  ‖W·a‖/‖δ_true‖   R²(base-traj)   R²_te(lora-CoT)", flush=True)
    for L in layers:
        g = shift_geometry(np.concatenate(pred_rows[L]), np.concatenate(true_rows[L]))
        g["r2_te_lora_cot"] = r2_te.get(L)
        results["per_layer"][L] = g
        print(f"{L:2d}: {g['mean_cosine']:+.3f}            {g['median_norm_ratio']:.3f}"
              f"          {g['r2']:+.3f}         {r2_te.get(L):+.3f}", flush=True)

    out_path = Path(args.out) if args.out else maps_dir.parent / "map_vs_oracle.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
