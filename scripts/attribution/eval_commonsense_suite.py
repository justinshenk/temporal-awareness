"""Eval base vs LoReFT-steered Llama-2-7B on the LLM-Adapters commonsense test splits.

Greedy decoding, 32 new tokens, answer = the word after "the correct answer is" (the paper's
trigger extraction), exact-match against the gold ``answer`` field. With ``--interventions none``
this is the base zero-shot anchor; with a checkpoint from ``train_loreft_commonsense.py`` the
interventions edit the f7+l7 prompt positions at prefill (decode steps pass through untouched).

    uv run python -m scripts.attribution.eval_commonsense_suite \
        --config configs/attribution/loreft_commonsense_llama2.yaml --interventions none
    uv run python -m scripts.attribution.eval_commonsense_suite \
        --config configs/attribution/loreft_commonsense_llama2.yaml \
        --interventions results/attribution/loreft_commonsense/loreft_interventions.pt

Writes ``{output.dir}/eval_{tag}.json`` with per-item generations for transcript inspection.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from scripts.attribution.train_loreft_commonsense import load_frozen_base
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.commonsense_data import (
    extract_answer,
    format_prompt,
    load_commonsense_json,
    score_predictions,
)
from src.probes.attribution.loreft_intervention import (
    LoReFTIntervention,
    PositionEditHook,
    intervention_locations,
)
from src.probes.attribution.crossmodel_align import TransplantEdit
from src.probes.attribution.ridge_steering_map import RidgeEdit

RIDGE_PREFIX = "ridge:"
TRANSPLANT_PREFIX = "transplant:"


def load_interventions(path: str, model, device) -> PositionEditHook:
    """LoReFT checkpoint, a ridge map (``ridge:``), or a cross-model transplant (``transplant:``)."""
    if path.startswith(RIDGE_PREFIX):
        return load_ridge_maps(path[len(RIDGE_PREFIX):], model, device)
    if path.startswith(TRANSPLANT_PREFIX):
        return load_transplant(path[len(TRANSPLANT_PREFIX):], model, device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    interventions = {}
    for li, state in ckpt["state"].items():
        iv = LoReFTIntervention(meta["hidden_dim"], meta["rank"]).to(device)
        iv.load_state_dict(state)
        iv.eval()
        interventions[int(li)] = iv
    print(f"Loaded {len(interventions)} layer interventions (rank {meta['rank']}, "
          f"trained on {meta['n_train']} examples)", flush=True)
    return PositionEditHook(model, interventions)


def load_ridge_maps(path: str, model, device) -> PositionEditHook:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    interventions = {}
    for li, m in ckpt["maps"].items():
        edit = RidgeEdit(m["P"], m["Q"], m["mean_x"], m["mean_d"]).to(device)
        edit.eval()
        interventions[int(li)] = edit
    print(f"Loaded {len(interventions)} ridge steering maps (rank {meta['rank']}, "
          f"λ={meta['ridge_lambda']}, fit on {meta['n_fit']} prompts)", flush=True)
    return PositionEditHook(model, interventions)


def load_transplant(path: str, model, device) -> PositionEditHook:
    """Cross-model transplant: per-layer Procrustes bridge wrapping the donor's LoReFT modules.

    The transplant file stores the alignment ``{A, mean_R, mean_D}`` per layer plus the donor LoReFT
    checkpoint path; we rebuild each donor :class:`LoReFTIntervention` and wrap it in a
    :class:`TransplantEdit`, which the recipient model applies through :class:`PositionEditHook`.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    donor_ckpt = torch.load(meta["loreft"], map_location=device, weights_only=True)
    lmeta = donor_ckpt["meta"]
    interventions = {}
    for li, al in ckpt["align"].items():
        donor = LoReFTIntervention(lmeta["hidden_dim"], lmeta["rank"]).to(device)
        donor.load_state_dict(donor_ckpt["state"][li])
        donor.eval()
        edit = TransplantEdit(donor, al["A"], al["mean_R"], al["mean_D"]).to(device)
        edit.eval()
        interventions[int(li)] = edit
    print(f"Loaded {len(interventions)} transplant edits ({meta['phase']} alignment, "
          f"fit on {meta['n_fit']} prompts; donor {lmeta['base_model']})", flush=True)
    return PositionEditHook(model, interventions)


@torch.no_grad()
def eval_dataset(model, tok, items: list[dict], hook: PositionEditHook | None,
                 n_prefix: int, n_suffix: int, batch_size: int, max_new: int,
                 device) -> dict:
    preds, records = [], []
    for s in range(0, len(items), batch_size):
        batch = items[s:s + batch_size]
        enc = tok([format_prompt(it) for it in batch], return_tensors="pt",
                  padding=True).to(device)
        width = enc.input_ids.shape[1]
        if hook is not None:
            locs = []
            for i in range(len(batch)):
                plen = int(enc.attention_mask[i].sum())
                locs.append(intervention_locations(plen, n_prefix, n_suffix, offset=width - plen))
            n_loc = max(len(lo) for lo in locs)
            hook.set_locations(torch.tensor([lo + [lo[-1]] * (n_loc - len(lo)) for lo in locs],
                                            device=device))
        ctx = hook.injecting() if hook is not None else nullcontext()
        with ctx:
            out = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                                 pad_token_id=tok.pad_token_id)
        for i, it in enumerate(batch):
            gen = tok.decode(out[i, width:], skip_special_tokens=True)
            pred = extract_answer(gen)
            preds.append(pred)
            records.append({"gold": it["answer"], "pred": pred, "gen": gen})
        print(f"  {min(s + batch_size, len(items))}/{len(items)}", flush=True)
    golds = [it["answer"] for it in items]
    return {"accuracy": score_predictions(preds, golds), "n": len(items), "records": records}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--interventions", required=True,
                    help="checkpoint path from train_loreft_commonsense.py, or 'none' for base")
    ap.add_argument("--datasets", default=None, help="override data.eval_datasets (comma list)")
    ap.add_argument("--limit", type=int, default=None, help="override eval.limit")
    ap.add_argument("--base-model", default=None,
                    help="override cfg['base_model'] (e.g. the recipient chat model for transplants)")
    ap.add_argument("--tag", default=None, help="output tag (default: base|loreft|ridge|transplant)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.base_model:
        cfg["base_model"] = args.base_model
    set_seed(cfg["seed"])
    device = cfg["device"]
    dcfg, icfg, ecfg = cfg["data"], cfg["intervention"], cfg["eval"]
    datasets = args.datasets.split(",") if args.datasets else dcfg["eval_datasets"]
    limit = args.limit if args.limit is not None else ecfg["limit"]
    steered = args.interventions != "none"
    if not steered:
        default_tag = "base"
    elif args.interventions.startswith(RIDGE_PREFIX):
        default_tag = "ridge"
    elif args.interventions.startswith(TRANSPLANT_PREFIX):
        default_tag = "transplant"
    else:
        default_tag = "loreft"
    tag = args.tag or default_tag

    print(f"Loading {cfg['base_model']} ...", flush=True)
    tok, model = load_frozen_base(cfg)
    hook = load_interventions(args.interventions, model, device) if steered else None

    results = {"tag": tag, "interventions": args.interventions, "datasets": {}}
    for name in datasets:
        items = load_commonsense_json(Path(dcfg["dir"]) / f"{name}_test.json")
        if limit:
            items = items[:limit]
        print(f"\n[{name}] {len(items)} items", flush=True)
        res = eval_dataset(model, tok, items, hook, icfg["n_prefix"], icfg["n_suffix"],
                           ecfg["batch_size"], ecfg["max_new"], device)
        results["datasets"][name] = res
        print(f"[{name}] accuracy = {res['accuracy']:.4f}", flush=True)

    if hook is not None:
        hook.remove()
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n{'dataset':<16}accuracy", flush=True)
    for name, res in results["datasets"].items():
        print(f"{name:<16}{res['accuracy']:.4f}", flush=True)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
