"""How much of the LoRA shift δ lives in the base vs. LoRA activation manifold?

For each layer, builds the top-k manifold basis V from the stored CoT-token second moments
(base ``Σaaᵀ``, lora ``Σ(a+δ)(a+δ)ᵀ``, or their union) and reports the fraction of δ energy
it captures: ``tr(Vᵀ Gd V) / Σ‖δ‖²``. This decides whether the base-manifold projection used
in the steering probe is stripping real signal:

  base-survival high  → projection keeps most of δ; base-only steering probe is defensible.
  base-survival low   → δ uses directions the base manifold doesn't span (a finding in itself);
                        compare to lora-survival — if lora-survival ≫ base-survival, the LoRA's
                        shift occupies new representational territory.

    uv run python -m scripts.attribution.manifold_overlap \
        --config configs/attribution/metamath_llama2_gsm8k.yaml [--acc-suffix _smoke --ks 128,512,1024]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.probes.attribution.gram_accumulator import GramAccumulator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--acc-suffix", default="_smoke")
    ap.add_argument("--ks", default="128,512,1024", help="comma list of subspace dims")
    ap.add_argument("--layers", default=None, help="comma list (default all)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg["accum_device"]
    ks = [int(x) for x in args.ks.split(",")]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(cfg["num_layers"])))
    acc_dir = Path(cfg["output"]["acc_dir"] + args.acc_suffix)

    rows = []
    header = "layer  " + "  ".join(f"k={k:<4d}[base/lora/union]" for k in ks)
    print(header, flush=True)
    for l in layers:
        acc = GramAccumulator.from_state_dict(torch.load(acc_dir / f"train_L{l}.pt"), device=device)
        rec = {"layer": l}
        cells = []
        for k in ks:
            surv = {w: acc.delta_survival(acc.manifold_basis(k, w)) for w in ("base", "lora", "union")}
            rec[f"k{k}"] = surv
            cells.append(f"{surv['base']:.3f}/{surv['lora']:.3f}/{surv['union']:.3f}")
        rows.append(rec)
        print(f"L{l:2d}    " + "  ".join(f"{c:>20s}" for c in cells), flush=True)

    out = Path(cfg["output"]["sweep_json"].replace("sweep.json", f"manifold_overlap{args.acc_suffix}.json"))
    out.write_text(json.dumps({"ks": ks, "layers": rows}, indent=2, default=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
