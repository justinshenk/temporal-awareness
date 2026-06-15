"""Phase B: do LoRA and LoReFT make the SAME activation edit on commonsense?

For a shared sample of commonsense prompts, capture per-layer residual activations at LoReFT's
f7+l7 positions under three configs — base, LoRA (PEFT weights), LoReFT (PositionEditHook) — then
compare each method's edit ``δ_method = h_method − h_base`` per layer:

  * ``cosine``     — mean per-token cosine between δ_LoRA and δ_LoReFT (``shift_geometry``);
  * ``cka_delta``  — linear CKA of the two edit sets (``activation_similarity.linear_cka``);
  * ``subspace``   — top-k PCA-subspace overlap of the two edit sets (``subspace_overlap``);
  * ``cka_h``      — (secondary) linear CKA of the steered representations h_LoRA vs h_LoReFT.

High cosine/CKA/overlap ⇒ the two methods converge on the same representational edit; low ⇒ they
reach the same commonsense behavior by different routes. Capture is batch=1 (the capture hook's
assumption); the LoReFT edit hook is registered BEFORE the capture hook so capture reads the
post-edit residual, and the LoRA pass reuses the same capture on the same base layer modules
(PEFT wraps the inner projections, not the decoder-block objects).

    uv run python -m scripts.attribution.compare_lora_loreft_commonsense \
        --config configs/attribution/loreft_commonsense_llama2.yaml \
        --lora results/attribution/loreft_commonsense/lora_commonsense \
        --loreft results/attribution/loreft_commonsense/loreft_interventions.pt \
        --n-eval 60 --layers 4,8,12,16,20,24,28
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from scripts.attribution.eval_commonsense_suite import load_interventions
from scripts.attribution.train_loreft_commonsense import load_frozen_base
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.activation_similarity import linear_cka, subspace_overlap
from src.probes.attribution.commonsense_data import format_prompt, load_commonsense_json
from src.probes.attribution.loreft_intervention import intervention_locations
from src.probes.attribution.shift_geometry import mean_row_cosine
from src.probes.extraction import PerTokenResidualCapture


def sample_prompts(dcfg, datasets, n_eval):
    """A balanced sample across the eval datasets: first ``n_eval // len(datasets)`` of each."""
    per = max(1, n_eval // len(datasets))
    items = []
    for name in datasets:
        ds = load_commonsense_json(Path(dcfg["dir"]) / f"{name}_test.json")[:per]
        items.extend(ds)
    return items


@torch.no_grad()
def capture_rows(forward, capture, layers, pos):
    """Run ``forward()`` under capture and return ``{layer: rows[pos]}`` (len(pos), d) on CPU."""
    capture.clear()
    with capture.capturing():
        forward()
    return {li: capture.captured[li][pos] for li in layers}


def collect(base, tok, items, device, layers, n_prefix, n_suffix, loreft_path, lora_path):
    """Per-layer stacked rows for base / loreft / lora at the f7+l7 positions of each prompt."""
    loreft_hook = load_interventions(loreft_path, base, device)      # edit hook (registered first)
    capture = PerTokenResidualCapture(base, layers)                  # fires AFTER the edit hook
    rows = {tag: {li: [] for li in layers} for tag in ("base", "loreft", "lora")}

    encoded = []
    for it in items:
        ids = tok(format_prompt(it), return_tensors="pt").input_ids.to(device)
        plen = ids.shape[1]
        pos = intervention_locations(plen, n_prefix, n_suffix, offset=0)
        encoded.append((ids, pos))
        base_rows = capture_rows(lambda: base(ids, use_cache=False), capture, layers, pos)  # no inject
        loreft_hook.set_locations(torch.tensor([pos], device=device))
        loreft_rows = capture_rows(
            lambda: _inject(loreft_hook, base, ids), capture, layers, pos)
        for li in layers:
            rows["base"][li].append(base_rows[li])
            rows["loreft"][li].append(loreft_rows[li])

    loreft_hook.remove()
    lora = PeftModel.from_pretrained(base, lora_path).eval()         # adapter into the same layers
    for ids, pos in encoded:
        lora_rows = capture_rows(lambda: lora(ids, use_cache=False), capture, layers, pos)
        for li in layers:
            rows["lora"][li].append(lora_rows[li])

    capture.remove()
    return {tag: {li: torch.cat(rows[tag][li], dim=0) for li in layers} for tag in rows}


def _inject(hook, base, ids):
    with hook.injecting():
        base(ids, use_cache=False)


def compare(rows, layers, k):
    """Per-layer edit-similarity metrics between LoRA and LoReFT."""
    out = {}
    for li in layers:
        b, l, r = rows["base"][li], rows["lora"][li], rows["loreft"][li]
        dL, dR = (l - b), (r - b)
        out[li] = {
            "n_tokens": int(dL.shape[0]),
            "cosine": mean_row_cosine(dL.numpy(), dR.numpy()),
            "cka_delta": linear_cka(dL, dR),
            "subspace_overlap": subspace_overlap(dL, dR, k),
            "cka_h": linear_cka(l, r),
            "median_norm_lora": float(dL.norm(dim=1).median()),
            "median_norm_loreft": float(dR.norm(dim=1).median()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--lora", required=True, help="PEFT adapter dir from train_lora_commonsense.py")
    ap.add_argument("--loreft", required=True, help="LoReFT checkpoint from train_loreft_commonsense.py")
    ap.add_argument("--n-eval", type=int, default=60, help="total prompts, split across datasets")
    ap.add_argument("--layers", default="4,8,12,16,20,24,28", help="probe layer subset (comma list)")
    ap.add_argument("--k", type=int, default=8, help="PCA-subspace dim for the overlap metric")
    ap.add_argument("--datasets", default=None, help="override data.eval_datasets")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, dcfg, icfg = cfg["device"], cfg["data"], cfg["intervention"]
    datasets = args.datasets.split(",") if args.datasets else dcfg["eval_datasets"]
    layers = [int(x) for x in args.layers.split(",")]

    print(f"Loading {cfg['base_model']} ...", flush=True)
    tok, base = load_frozen_base(cfg)
    items = sample_prompts(dcfg, datasets, args.n_eval)
    print(f"{len(items)} prompts across {datasets}; probe layers {layers}", flush=True)

    rows = collect(base, tok, items, device, layers, icfg["n_prefix"], icfg["n_suffix"],
                   args.loreft, args.lora)
    metrics = compare(rows, layers, args.k)

    results = {"layers": layers, "k": args.k, "n_prompts": len(items), "datasets": datasets,
               "lora": args.lora, "loreft": args.loreft, "per_layer": metrics}
    print(f"\n{'layer':>5} {'cosine':>8} {'cka_delta':>10} {'subspace':>9} {'cka_h':>7}  "
          f"{'|δ|_lora':>9} {'|δ|_loreft':>11}", flush=True)
    for li in layers:
        m = metrics[li]
        print(f"{li:>5} {m['cosine']:>8.3f} {m['cka_delta']:>10.3f} {m['subspace_overlap']:>9.3f} "
              f"{m['cka_h']:>7.3f}  {m['median_norm_lora']:>9.2f} {m['median_norm_loreft']:>11.2f}",
              flush=True)

    out_path = (Path(args.out) if args.out
                else Path(cfg["output"]["dir"]) / "lora_vs_loreft_similarity.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
