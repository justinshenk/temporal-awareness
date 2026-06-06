"""Phase 2: per-layer primal-ridge lambda sweep (cheap closed-form linear algebra).

Loads the train/held-out GramAccumulators from Phase 1; for each layer and each lambda
computes in-sample R² (asserting the two-form cross-check) and held-out R²_te; picks
lambda* = argmax R²_te (the genuine learnable-signal level); saves the float32 W at
lambda* per layer for the Phase-3 steering eval.

    uv run python -m scripts.attribution.fit_ridge_sweep \
        --config configs/attribution/metamath_llama2_gsm8k.yaml [--acc-suffix _smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.probes.attribution.gram_accumulator import GramAccumulator
from src.probes.attribution.ridge_fit import LayerSweep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--acc-suffix", default="")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    layers = list(range(cfg["num_layers"]))
    device = cfg["accum_device"]
    lo, hi, num = cfg["sweep"]["lambdas_logspace"]
    lambdas = [float(x) for x in np.logspace(lo, hi, int(num))]
    rtol = float(cfg["sweep"]["crosscheck_rtol"])

    acc_dir = Path(cfg["output"]["acc_dir"] + args.acc_suffix)
    maps_dir = Path(cfg["output"]["maps_dir"] + args.acc_suffix)
    maps_dir.mkdir(parents=True, exist_ok=True)

    sweeps = []
    for l in layers:
        tr = GramAccumulator.from_state_dict(torch.load(acc_dir / f"train_L{l}.pt"), device=device)
        te = GramAccumulator.from_state_dict(torch.load(acc_dir / f"heldout_L{l}.pt"), device=device)
        r2_in, r2_te, cc_err = [], [], []
        best_W, best_r2_te, best_lam = None, -1e30, lambdas[0]
        for lam in lambdas:
            fit = tr.solve(lam)
            assert fit.crosscheck_abs_err < rtol * max(fit.s, 1.0), (
                f"L{l} lam={lam}: cross-check {fit.crosscheck_abs_err:.3e} >= {rtol}*s")
            ho = te.score(fit.W)
            r2_in.append(fit.r2_insample)
            r2_te.append(ho.r2_te)
            cc_err.append(fit.crosscheck_abs_err)
            if ho.r2_te > best_r2_te:
                best_r2_te, best_lam, best_W = ho.r2_te, lam, fit.W
        sw = LayerSweep(layer=l, lambdas=lambdas, r2_insample=r2_in, r2_te=r2_te,
                        crosscheck_abs_err=cc_err, lam_star=best_lam, r2_te_star=best_r2_te)
        sweeps.append(sw)
        torch.save({"layer": l, "lam": best_lam, "W": best_W.to(torch.float32).cpu()},
                   maps_dir / f"W_L{l}.pt")
        print(f"  L{l:2d}: lambda*={best_lam:.2e}  R2_te*={best_r2_te:+.4f}  "
              f"R2_in@lambda*={r2_in[lambdas.index(best_lam)]:+.4f}", flush=True)

    out = {"lambdas": lambdas, "layers": [sw.to_dict() for sw in sweeps]}
    Path(cfg["output"]["sweep_json"].replace(".json", f"{args.acc_suffix}.json")).write_text(
        json.dumps(out, indent=2, default=float))
    best = max(sweeps, key=lambda s: s.r2_te_star)
    print(f"\nPeak held-out R²_te = {best.r2_te_star:+.4f} at layer {best.layer} "
          f"(lambda*={best.lam_star:.2e})")
    print(f"Saved per-layer W to {maps_dir} and sweep to {cfg['output']['sweep_json']}")


if __name__ == "__main__":
    main()
