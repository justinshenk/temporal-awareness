"""DAS vs PCA: does a task-loss-trained δ-subspace recover where the variance band can't?

For each rank r in a grid, learn an orthonormal R (d×r) by gradient descent on a behavioral loss —
inject ``a + Π_R(δ_true)`` at layer L (frozen base, every position, teacher-forced on the base
trajectory) and minimise CE against the LoRA's greedy next-token decisions. Then evaluate R with the
*identical* closed-loop lockstep oracle used for the PCA bands (``eval_band``), and print DAS-R@r
against the PCA-top-r anchor at matched rank. If DAS-R@64 ≫ PCA-top64 (=0), the capability directions
are low-rank but low-variance — found only by task-loss search.

    uv run python -m scripts.attribution.das_subspace_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 \
        --ranks 8,64,256,512 --n-train 100 --epochs 40 --n-contrast 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import generate_cot_ids, gsm8k_problems, load_base_and_lora, load_contrast
from scripts.attribution.collect_cot_residuals import teacher_force_capture
from scripts.attribution.lockstep_pca_band import eval_band
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.cot_collection import cot_token_slice
from src.probes.attribution.das_subspace import OrthoSubspace, inject_value, subspace_lm_loss
from src.probes.attribution.lockstep_oracle import OverwriteResidualHook
from src.probes.extraction import PerTokenResidualCapture


def build_train_cache(base, lora, tok, problems, device, L, max_new):
    """Per problem: base trajectory ``full_ids``; cached ``a_L``, ``δ_L``, and LoRA greedy targets.

    Targets are ``argmax`` of the LoRA's logits at each position given the *base* context — exactly
    the per-step decision the oracle injection reproduces, so this supervises the closed-loop quantity.
    """
    capture = PerTokenResidualCapture(base, [L])
    cache = []
    for i, (q, _gold) in enumerate(problems):
        with lora.disable_adapter():
            full_ids, plen = generate_cot_ids(base, tok, q, device, max_new)
        if full_ids.shape[1] - plen <= 0:
            continue
        with lora.disable_adapter():
            base_cap = teacher_force_capture(base, capture, full_ids)
        capture.clear()
        with capture.capturing(), torch.no_grad():
            lora_logits = lora(full_ids, use_cache=False).logits
        lora_L = capture.captured[L]
        a = base_cap[L].to(device)                       # (seq, d) f32
        delta = (lora_L.to(device) - a)                  # (seq, d) f32
        target = lora_logits[0].argmax(-1).to(device)    # (seq,) greedy LoRA tokens
        cache.append({"ids": full_ids, "plen": plen, "a": a, "delta": delta, "target": target})
        if (i + 1) % 25 == 0:
            print(f"  cached {i+1}/{len(problems)} problems", flush=True)
    capture.remove()
    return cache


def train_subspace(base, lora, cache, device, L, r, d, epochs, lr, seed):
    """Learn an orthonormal R (d×r) by the behavioral CE loss; return R (cpu f32) + loss trace."""
    sub = OrthoSubspace(d, r, seed=seed).to(device)
    inject = OverwriteResidualHook(base, [L])
    opt = torch.optim.Adam(sub.parameters(), lr=lr)
    order = torch.randperm(len(cache), generator=torch.Generator().manual_seed(seed))
    trace = []
    for ep in range(epochs):
        perm = order[torch.randperm(len(cache), generator=torch.Generator().manual_seed(seed + ep + 1))]
        tot = 0.0
        for j in perm.tolist():
            item = cache[j]
            R = sub()
            h = inject_value(item["a"], item["delta"], R)        # (seq, d) f32, grad in R
            inject.set_values({L: h})
            with inject.injecting(), lora.disable_adapter():
                logits = base(item["ids"].to(device), use_cache=False).logits
            loss = subspace_lm_loss(logits.float(), item["target"], item["plen"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        trace.append(tot / len(cache))
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"    [r={r}] epoch {ep+1}/{epochs}  mean CE={trace[-1]:.4f}", flush=True)
    inject.remove()
    R = sub().detach().cpu()
    return R, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--ranks", default="8,64,256,512")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--n-contrast", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--pca-json", default=None, help="PCA anchor (default: lockstep_pca_band_L{L}.json)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, L, d = cfg["device"], args.layer, cfg["hidden_dim"]
    ranks = [int(x) for x in args.ranks.split(",")]

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)
    for p in lora.parameters():                          # freeze everything; only R trains
        p.requires_grad_(False)
    base.eval()

    out_dir = Path(cfg["output"]["steer_json"]).parent
    pca_path = Path(args.pca_json) if args.pca_json else out_dir / f"lockstep_pca_band_L{L}.json"
    pca = json.loads(pca_path.read_text())["bands"] if pca_path.exists() else {}

    train_problems = gsm8k_problems("train", args.n_train, skip=0)
    print(f"Building train cache at L{L} over {len(train_problems)} problems ...", flush=True)
    cache = build_train_cache(base, lora, tok, train_problems, device, L, args.max_new)
    print(f"  {len(cache)} usable problems cached", flush=True)

    contrast = load_contrast(cfg)[:args.n_contrast]
    results = {"layer": L, "n_train": len(cache), "n_contrast": len(contrast), "ranks": {}}
    print(f"\nrank   DAS-R recovery   PCA-top-r recovery   ({len(contrast)} contrast problems)", flush=True)
    for r in ranks:
        print(f"\n[train] rank {r} ...", flush=True)
        R, trace = train_subspace(base, lora, cache, device, L, r, d, args.epochs, args.lr, cfg["seed"])
        ortho_err = float((R.T @ R - torch.eye(r)).abs().max())
        das_acc = eval_band(base, lora, tok, contrast, device, L, R.to(torch.float32), args.max_new)
        pca_acc = pca.get(f"top{r}", {}).get("recovery")
        results["ranks"][r] = {"das_recovery": das_acc, "pca_recovery": pca_acc,
                               "final_ce": trace[-1], "ortho_err": ortho_err, "loss_trace": trace}
        pca_str = f"{pca_acc:.3f}" if pca_acc is not None else "  n/a"
        print(f"{r:5d}   {das_acc:.3f}            {pca_str}   (CE={trace[-1]:.3f}, ‖RᵀR−I‖={ortho_err:.1e})",
              flush=True)

    out_path = Path(args.out) if args.out else out_dir / f"das_subspace_L{L}.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
