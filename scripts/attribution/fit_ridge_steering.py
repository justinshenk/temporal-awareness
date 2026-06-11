"""Distil a trained LoReFT intervention into a ridge-fit rank-r conditional affine map.

For each layer, over the f7+l7 prompt positions, we collect paired residual-stream rows from two
forwards of the frozen base on a subset of commonsense-train prompts: ``h_base`` (clean, no hook) and
``h_steered`` (the LoReFT post-edit value). One :class:`PairStats` per layer streams the sufficient
statistics; after all batches :func:`fit_ridge_rank` produces the rank-r map ``δ̂(h)`` that
``RidgeEdit`` applies as ``h' = h + δ̂(h)``. Raw rows are never stored — memory is per-layer ``d×d``.

    uv run python -m scripts.attribution.fit_ridge_steering \
        --config configs/attribution/loreft_commonsense_llama2.yaml \
        --loreft results/attribution/loreft_commonsense/loreft_interventions.pt \
        --n-fit 4000 --rank 8 --ridge-lambda 1.0

Saves ``{output.dir}/ridge_maps.pt`` ({"meta": …, "maps": {layer: {P,Q,mean_x,mean_d}}}) for
``eval_commonsense_suite.py --interventions ridge:<path>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from scripts.attribution.train_loreft_commonsense import (
    collate_left_padded,
    encode_examples,
    load_frozen_base,
)
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.commonsense_data import (
    load_commonsense_json,
    subset_examples,
)
from src.probes.attribution.loreft_intervention import (
    LoReFTIntervention,
    PositionEditHook,
)
from src.probes.attribution.ridge_steering_map import (
    PairStats,
    fit_residual,
    fit_ridge_rank,
)


def load_loreft(path: str, model, device) -> tuple[PositionEditHook, dict]:
    """Rebuild the trained ``{layer: LoReFTIntervention}`` and its :class:`PositionEditHook`."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    interventions = {}
    for li, state in ckpt["state"].items():
        iv = LoReFTIntervention(meta["hidden_dim"], meta["rank"]).to(device)
        iv.load_state_dict(state)
        iv.eval()
        interventions[int(li)] = iv
    return PositionEditHook(model, interventions), meta


class LocationCapture:
    """Temporary forward hooks that copy ``output[0][bidx, loc]`` of each layer into a buffer.

    Registered AFTER the LoReFT :class:`PositionEditHook`, so on the steered pass the captured rows
    are LoReFT's post-edit values; on the clean pass (LoReFT disabled) they are the base residuals.
    """

    def __init__(self, model, layers: list[int]):
        self.captured: dict[int, torch.Tensor] = {}
        self.locations: torch.Tensor | None = None
        self._hooks = [model.model.layers[li].register_forward_hook(self._make_hook(li))
                       for li in layers]

    def set_locations(self, locations: torch.Tensor) -> None:
        self.locations = locations

    def _make_hook(self, li: int):
        def hook_fn(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            loc = self.locations.to(hs.device)
            if hs.shape[1] <= int(loc.max()):
                return
            bidx = torch.arange(hs.shape[0], device=hs.device).unsqueeze(1).expand_as(loc)
            rows = hs[bidx, loc]                            # (batch, n_pos, d)
            self.captured[li] = rows.reshape(-1, rows.shape[-1]).detach().float()

        return hook_fn

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


@torch.no_grad()
def collect_stats(model, tok, encoded, hook, pre_capture, post_capture, layers, icfg,
                  batch_size, device, fit_input):
    """Stream paired (input, steered-post) rows into one :class:`PairStats` per layer.

    ``fit_input='clean'`` pairs the clean residual with the steered post-edit value (the map must
    reconstruct the steered trajectory from off-distribution clean input). ``fit_input='steered'``
    pairs the steered *pre-edit* value (what the map actually sees at inference) with the post-edit
    value, so the target δ is exactly LoReFT's own rank-r edit on its own input.
    """
    stats = {li: PairStats(model.config.hidden_size, device) for li in layers}
    n_batches = (len(encoded) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch = encoded[bi * batch_size:(bi + 1) * batch_size]
        ids, mask, locs, _ = collate_left_padded(
            batch, tok.pad_token_id, icfg["n_prefix"], icfg["n_suffix"], device)
        pre_capture.set_locations(locs)
        post_capture.set_locations(locs)

        clean = None
        if fit_input == "clean":
            model(ids, attention_mask=mask)                # clean: LoReFT hook disabled
            clean = {li: post_capture.captured[li] for li in layers}

        hook.set_locations(locs)
        with hook.injecting():
            model(ids, attention_mask=mask)                # steered: pre_capture=pre-edit, post=post-edit
        for li in layers:
            src = clean[li] if fit_input == "clean" else pre_capture.captured[li]
            stats[li].update(src, post_capture.captured[li])
        print(f"  batch {bi + 1}/{n_batches}", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--loreft", required=True, help="trained LoReFT checkpoint to distil")
    ap.add_argument("--n-fit", type=int, default=None, help="prompts to fit on (default data.n_train)")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--fit-input", choices=["clean", "steered"], default="clean",
                    help="regress δ from the clean residual (distribution-shifted) or the steered "
                         "pre-edit value (the input the map sees at inference)")
    ap.add_argument("--out", default=None, help="override output path")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, d = cfg["device"], cfg["hidden_dim"]
    dcfg, icfg = cfg["data"], cfg["intervention"]
    n_fit = args.n_fit or dcfg["n_train"]

    print(f"Loading {cfg['base_model']} (frozen bf16) ...", flush=True)
    tok, model = load_frozen_base(cfg)
    layers = sorted(int(li) for li in
                    torch.load(args.loreft, map_location="cpu", weights_only=True)["state"])
    pre_capture = LocationCapture(model, layers)           # BEFORE LoReFT hook: sees pre-edit value
    hook, lmeta = load_loreft(args.loreft, model, device)
    post_capture = LocationCapture(model, layers)          # AFTER LoReFT hook: sees post-edit value

    data = load_commonsense_json(Path(dcfg["dir"]) / dcfg["train_file"])
    items = subset_examples(data, n_fit, seed=cfg["seed"])
    encoded = encode_examples(tok, items, dcfg["max_len"])
    print(f"{len(encoded)} usable prompts (of {n_fit} sampled), fitting {len(layers)} layers "
          f"at rank {args.rank}, λ={args.ridge_lambda}, fit-input={args.fit_input}", flush=True)

    stats = collect_stats(model, tok, encoded, hook, pre_capture, post_capture, layers, icfg,
                          cfg["eval"]["batch_size"], device, args.fit_input)
    pre_capture.remove()
    post_capture.remove()
    hook.remove()

    maps, residuals = {}, []
    for li in layers:
        state = stats[li].state()
        P, Q, mean_x, mean_d = fit_ridge_rank(state, args.rank, args.ridge_lambda)
        maps[li] = {"P": P, "Q": Q, "mean_x": mean_x, "mean_d": mean_d}
        rel = fit_residual(state, P, Q, mean_x, mean_d)
        residuals.append(rel)
        print(f"  layer {li:2d}  rel ‖δ−δ̂‖/‖δ‖ = {rel:.4f}", flush=True)
    print(f"fit residual range: {min(residuals):.4f} .. {max(residuals):.4f}", flush=True)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "ridge_maps.pt"
    torch.save({"meta": {"base_model": cfg["base_model"], "hidden_dim": d,
                         "num_layers": cfg["num_layers"], "rank": args.rank,
                         "ridge_lambda": args.ridge_lambda, "fit_input": args.fit_input,
                         "n_fit": len(encoded),
                         "n_prefix": icfg["n_prefix"], "n_suffix": icfg["n_suffix"],
                         "loreft": args.loreft, "loreft_meta": lmeta},
                "maps": maps}, out_path)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
