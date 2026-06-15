"""On-policy cross-model transplant of a trained LoReFT commonsense capability.

A donor (Llama-2-7B *base*) owns a trained LoReFT edit; a recipient (Llama-2-7B-*chat*) never saw
it. Because both share the Llama-2 tokenizer, the same commonsense prompt yields row-corresponding
residual streams at the f7+l7 prompt positions, so a per-layer orthogonal-Procrustes bridge ``A_L``
(recipient→donor) lets the recipient apply the donor's edit (see
:mod:`src.probes.attribution.crossmodel_align`).

The experiment contrasts **off-policy** and **on-policy** alignment:

* ``--phase clean`` (off-policy): fit ``A_L`` from CLEAN paired activations — donor clean rows ↔
  recipient clean rows. The bridge never sees the steered distribution it will operate on.
* ``--phase onpolicy`` (DAgger, 1 round): with the *clean* transplant already active in the
  recipient, collect the recipient's STEERED **pre-edit** rows; with LoReFT active in the donor,
  collect the donor's STEERED **post-edit** rows; refit ``A_L`` on these steered↔steered pairs.

Donor and recipient are loaded one at a time to bound GPU memory: we first run every donor forward
(caching the captured rows on CPU), free the donor, then run every recipient forward, then stream the
paired rows into one :class:`CrossCovStats` per layer and fit.

    uv run python -m scripts.attribution.transplant_loreft --phase clean \
        --config configs/attribution/loreft_commonsense_llama2.yaml \
        --loreft results/attribution/loreft_commonsense/loreft_interventions.pt \
        --recipient NousResearch/Llama-2-7b-chat-hf --n-fit 2000 \
        --out results/attribution/loreft_commonsense/transplant_clean.pt

    uv run python -m scripts.attribution.transplant_loreft --phase onpolicy \
        --config ... --loreft ... --recipient ... --n-fit 2000 \
        --clean-transplant results/attribution/loreft_commonsense/transplant_clean.pt \
        --out results/attribution/loreft_commonsense/transplant_onpolicy.pt

Saves ``{"meta": …, "align": {layer: {A, mean_R, mean_D}}}`` for
``eval_commonsense_suite.py --interventions transplant:<path>``.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.attribution.eval_commonsense_suite import load_transplant
from scripts.attribution.fit_ridge_steering import LocationCapture, load_loreft
from scripts.attribution.train_loreft_commonsense import collate_left_padded, encode_examples
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.commonsense_data import load_commonsense_json, subset_examples
from src.probes.attribution.crossmodel_align import (
    CrossCovStats,
    affine_residual,
    fit_affine_bridge,
    fit_procrustes,
    procrustes_residual,
)


def load_frozen(model_name: str, device: str):
    """Frozen bf16 model + left-padded tokenizer (shared Llama-2 vocab for donor and recipient)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def build_batches(tok, encoded, icfg, batch_size, device):
    """Per-batch ``(ids, mask, locs)`` on ``device`` (locations identical across models)."""
    n_batches = (len(encoded) + batch_size - 1) // batch_size
    batches = []
    for bi in range(n_batches):
        chunk = encoded[bi * batch_size:(bi + 1) * batch_size]
        ids, mask, locs, _ = collate_left_padded(
            chunk, tok.pad_token_id, icfg["n_prefix"], icfg["n_suffix"], device)
        batches.append((ids, mask, locs))
    return batches


@torch.no_grad()
def capture_rows(model, batches, layers, hook, capture):
    """Run every batch, returning ``[{layer: rows}]`` (CPU f32) of the post-``capture`` residuals.

    ``hook`` is the active intervention (LoReFT for the donor, the clean transplant for the recipient
    on-policy pass) or ``None`` for a clean pass. ``capture`` is registered relative to ``hook`` by
    the caller (AFTER for donor post-edit, BEFORE for recipient pre-edit).
    """
    per_batch = []
    for bi, (ids, mask, locs) in enumerate(batches):
        capture.set_locations(locs)
        if hook is None:
            model(ids, attention_mask=mask)
        else:
            hook.set_locations(locs)
            with hook.injecting():
                model(ids, attention_mask=mask)
        per_batch.append({li: capture.captured[li].cpu() for li in layers})
        print(f"  batch {bi + 1}/{len(batches)}", flush=True)
    return per_batch


def fit_alignment(recipient_rows, donor_rows, layers, d):
    """Stream paired rows into one :class:`CrossCovStats` per layer and fit ``A_L``."""
    stats = {li: CrossCovStats(d) for li in layers}
    for rec_batch, don_batch in zip(recipient_rows, donor_rows):
        for li in layers:
            stats[li].update(rec_batch[li], don_batch[li])
    align, residuals = {}, []
    for li in layers:
        state = stats[li].state()
        A, mean_R, mean_D = fit_procrustes(state)
        align[li] = {"A": A, "mean_R": mean_R, "mean_D": mean_D}
        rel = procrustes_residual(state, A, mean_R, mean_D)
        residuals.append(rel)
        print(f"  layer {li:2d}  procrustes residual ‖X_Dᶜ−A X_Rᶜ‖/‖X_Dᶜ‖ = {rel:.4f}", flush=True)
    print(f"alignment residual range: {min(residuals):.4f} .. {max(residuals):.4f}", flush=True)
    return align


def split_stats(recipient_rows, donor_rows, layers, d, val_every):
    """Train/val :class:`CrossCovStats` per layer; every ``val_every``-th batch is held out."""
    train = {li: CrossCovStats(d) for li in layers}
    val = {li: CrossCovStats(d) for li in layers}
    for bi, (rec_batch, don_batch) in enumerate(zip(recipient_rows, donor_rows)):
        dest = val if bi % val_every == val_every - 1 else train
        for li in layers:
            dest[li].update(rec_batch[li], don_batch[li])
    return train, val


def fit_affine_alignment(recipient_rows, donor_rows, layers, d, lam_sweep, ranks, val_every,
                         fit_device):
    """Affine vs Procrustes bridge on the same rows, scored on HELD-OUT batches.

    Per layer: fit the orthogonal Procrustes map and full-rank affine ridge maps (one per λ in
    ``lam_sweep``) on the train batches and score both with :func:`affine_residual` on the val
    batches (Procrustes is the affine map ``W_F = Aᵀ``, so one residual function compares them on
    identical footing). λ* is chosen by the median held-out residual across layers; the saved maps
    and the rank sweep (SVD truncations of the λ* map) use λ*. Fits run on ``fit_device`` —
    float64 ``d×d`` solves are far faster on GPU and the models are already freed.
    """
    train, val = split_stats(recipient_rows, donor_rows, layers, d, val_every)
    tstates = {li: {k: v.to(fit_device) for k, v in train[li].state().items()} for li in layers}
    vstates = {li: {k: v.to(fit_device) for k, v in val[li].state().items()} for li in layers}
    print(f"held-out split: every {val_every}th batch; fitting on {fit_device}", flush=True)

    per_lam = {lam: [] for lam in lam_sweep}                       # held-out residual per layer
    procrustes_res = []
    for li in layers:
        A, mean_R, mean_D = fit_procrustes(tstates[li])
        pres = affine_residual(vstates[li], A.T, mean_R, mean_D)
        procrustes_res.append(pres)
        cells = []
        for lam in lam_sweep:
            W_F, _, mR, mD = fit_affine_bridge(tstates[li], lam=lam)
            per_lam[lam].append(affine_residual(vstates[li], W_F, mR, mD))
            cells.append(f"λ={lam:g} {per_lam[lam][-1]:.4f}")
        print(f"  layer {li:2d}  heldout: procrustes {pres:.4f} | affine " + "  ".join(cells),
              flush=True)

    medians = {lam: sorted(rs)[len(rs) // 2] for lam, rs in per_lam.items()}
    lam_star = min(medians, key=medians.get)
    print(f"λ* = {lam_star:g} (median held-out residual {medians[lam_star]:.4f}; "
          f"procrustes median {sorted(procrustes_res)[len(procrustes_res) // 2]:.4f})", flush=True)

    align = {}
    for li in layers:
        W_F, W_G, mean_R, mean_D = fit_affine_bridge(tstates[li], lam=lam_star)
        align[li] = {"W_F": W_F.cpu(), "W_G": W_G.cpu(),
                     "mean_R": mean_R.cpu(), "mean_D": mean_D.cpu()}
        cells = []
        for rank in ranks:
            W_r, _, mR, mD = fit_affine_bridge(tstates[li], lam=lam_star, rank=rank)
            cells.append(f"r{rank} {affine_residual(vstates[li], W_r, mR, mD):.4f}")
        full = affine_residual(vstates[li], W_F, mean_R, mean_D)
        print(f"  layer {li:2d}  rank sweep @λ*: " + "  ".join(cells) + f"  full {full:.4f}",
              flush=True)
    return align, lam_star


def free(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--loreft", required=True, help="donor's trained LoReFT checkpoint")
    ap.add_argument("--recipient", required=True, help="recipient model (e.g. the chat model)")
    ap.add_argument("--phase", choices=["clean", "onpolicy"], required=True)
    ap.add_argument("--clean-transplant", default=None,
                    help="phase=onpolicy: the clean transplant to run live in the recipient")
    ap.add_argument("--n-fit", type=int, default=2000, help="prompts to align on")
    ap.add_argument("--bridge", choices=["procrustes", "affine"], default="procrustes",
                    help="orthogonal Procrustes (in-sample residual) or ridge-fit affine maps "
                         "(held-out residual surface vs the Procrustes baseline)")
    ap.add_argument("--lam-sweep", default="1e-6,1e-4,1e-2,1e0",
                    help="bridge=affine: relative ridge λ values for the held-out surface")
    ap.add_argument("--ranks", default="8,64,512",
                    help="bridge=affine: SVD-truncation ranks for the held-out surface at λ*")
    ap.add_argument("--val-every", type=int, default=10,
                    help="bridge=affine: every k-th batch is held out for residual scoring")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.phase == "onpolicy" and not args.clean_transplant:
        ap.error("--phase onpolicy requires --clean-transplant")

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, d = cfg["device"], cfg["hidden_dim"]
    dcfg, icfg = cfg["data"], cfg["intervention"]
    donor_name = cfg["base_model"]
    layers = sorted(int(li) for li in
                    torch.load(args.loreft, map_location="cpu", weights_only=True)["state"])

    data = load_commonsense_json(Path(dcfg["dir"]) / dcfg["train_file"])
    items = subset_examples(data, args.n_fit, seed=cfg["seed"])

    # ---- Donor pass: capture donor rows (clean for phase=clean, LoReFT post-edit for onpolicy). ----
    print(f"[donor] loading {donor_name} (frozen bf16) ...", flush=True)
    tok, donor = load_frozen(donor_name, device)
    encoded = encode_examples(tok, items, dcfg["max_len"])
    print(f"{len(encoded)} usable prompts; aligning {len(layers)} layers, phase={args.phase}",
          flush=True)
    batches = build_batches(tok, encoded, icfg, cfg["eval"]["batch_size"], device)

    if args.phase == "clean":
        capture = LocationCapture(donor, layers)
        donor_rows = capture_rows(donor, batches, layers, None, capture)
        capture.remove()
    else:
        hook, _ = load_loreft(args.loreft, donor, device)          # LoReFT PositionEditHook
        capture = LocationCapture(donor, layers)                   # AFTER LoReFT ⇒ post-edit rows
        donor_rows = capture_rows(donor, batches, layers, hook, capture)
        capture.remove()
        hook.remove()
    del batches
    free(donor)

    # ---- Recipient pass: capture recipient rows (clean, or steered pre-edit for onpolicy). ----
    print(f"[recipient] loading {args.recipient} (frozen bf16) ...", flush=True)
    rtok, recipient = load_frozen(args.recipient, device)
    rencoded = encode_examples(rtok, items, dcfg["max_len"])
    if len(rencoded) != len(encoded):
        raise RuntimeError(f"tokenizer mismatch: donor {len(encoded)} vs recipient {len(rencoded)} "
                           "usable prompts — rows would not correspond")
    rbatches = build_batches(rtok, rencoded, icfg, cfg["eval"]["batch_size"], device)

    if args.phase == "clean":
        capture = LocationCapture(recipient, layers)
        recipient_rows = capture_rows(recipient, rbatches, layers, None, capture)
        capture.remove()
    else:
        capture = LocationCapture(recipient, layers)               # BEFORE transplant ⇒ pre-edit rows
        hook = load_transplant(args.clean_transplant, recipient, device)
        recipient_rows = capture_rows(recipient, rbatches, layers, hook, capture)
        capture.remove()
        hook.remove()
    del rbatches
    free(recipient)

    meta = {"phase": args.phase, "donor": donor_name, "recipient": args.recipient,
            "loreft": args.loreft, "hidden_dim": d, "num_layers": cfg["num_layers"],
            "n_fit": len(encoded), "n_prefix": icfg["n_prefix"], "n_suffix": icfg["n_suffix"],
            "clean_transplant": args.clean_transplant, "bridge": args.bridge}
    if args.bridge == "procrustes":
        align = fit_alignment(recipient_rows, donor_rows, layers, d)
    else:
        lam_sweep = [float(x) for x in args.lam_sweep.split(",")]
        ranks = [int(x) for x in args.ranks.split(",")]
        align, lam_star = fit_affine_alignment(recipient_rows, donor_rows, layers, d, lam_sweep,
                                               ranks, args.val_every, device)
        meta.update({"lam": lam_star, "val_every": args.val_every})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "align": align}, out_path)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
