"""Train LoReFT interventions on commonsense-170k (subset-first reproduction, Llama-2-7B).

Paper recipe (pyreft examples/loreft, commonsense task): one rank-8 intervention per layer at ALL
layers, applied to the first 7 + last 7 prompt tokens (weights shared across positions), lr 9e-4,
6 epochs, batch 16 × grad-accum 2, warmup 0.1, dropout 0. The base model is frozen bf16; only the
interventions (≈2.1M params, f32) train, on CE over the response tokens ("the correct answer is X").

    uv run python -m scripts.attribution.train_loreft_commonsense \
        --config configs/attribution/loreft_commonsense_llama2.yaml

Saves ``{output.dir}/loreft_interventions.pt`` ({"meta": …, "state": {layer: state_dict}}) for
``eval_commonsense_suite.py``. Reduce ``--batch-size`` (and raise ``--grad-accum``) if OOM.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.commonsense_data import (
    format_prompt,
    format_target,
    load_commonsense_json,
    subset_examples,
)
from src.probes.attribution.loreft_intervention import (
    LoReFTIntervention,
    PositionEditHook,
    intervention_locations,
    response_labels,
)


def load_frozen_base(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=cfg["device"]).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def encode_examples(tok, items: list[dict], max_len: int) -> list[dict]:
    """Per item: prompt ids (with BOS) + target ids + eos, truncated; keep the prompt length."""
    out = []
    for it in items:
        p_ids = tok(format_prompt(it), add_special_tokens=True).input_ids
        t_ids = tok(format_target(it), add_special_tokens=False).input_ids + [tok.eos_token_id]
        full = (p_ids + t_ids)[:max_len]
        if len(full) <= len(p_ids):                 # truncation ate the whole response
            continue
        out.append({"ids": full, "plen": len(p_ids)})
    return out


def collate_left_padded(batch: list[dict], pad_id: int, n_prefix: int, n_suffix: int, device):
    """Left-pad to a common width; per-sample f+l locations offset by the pad, labels on response.

    Ragged location lists are padded by repeating their last index (duplicate writes are benign —
    the hook computes every edit from the pre-edit row).
    """
    width = max(len(b["ids"]) for b in batch)
    ids, mask, locs, starts = [], [], [], []
    for b in batch:
        pad = width - len(b["ids"])
        ids.append([pad_id] * pad + b["ids"])
        mask.append([0] * pad + [1] * len(b["ids"]))
        locs.append(intervention_locations(b["plen"], n_prefix, n_suffix, offset=pad))
        starts.append(pad + b["plen"])
    n_loc = max(len(lo) for lo in locs)
    locs = [lo + [lo[-1]] * (n_loc - len(lo)) for lo in locs]
    ids = torch.tensor(ids, device=device)
    labels = response_labels(ids.cpu(), torch.tensor(starts)).to(device)
    return ids, torch.tensor(mask, device=device), torch.tensor(locs, device=device), labels


def linear_warmup_decay(opt, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-train", type=int, default=None, help="override data.n_train")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="override train.grad_accum")
    ap.add_argument("--out", default=None, help="override checkpoint path")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, d = cfg["device"], cfg["hidden_dim"]
    dcfg, icfg, tcfg = cfg["data"], cfg["intervention"], cfg["train"]
    n_train = args.n_train or dcfg["n_train"]
    epochs = args.epochs or tcfg["epochs"]
    batch_size = args.batch_size or tcfg["batch_size"]
    grad_accum = args.grad_accum or tcfg["grad_accum"]

    print(f"Loading {cfg['base_model']} (frozen bf16) ...", flush=True)
    tok, model = load_frozen_base(cfg)

    data = load_commonsense_json(Path(dcfg["dir"]) / dcfg["train_file"])
    items = subset_examples(data, n_train, seed=cfg["seed"])
    encoded = encode_examples(tok, items, dcfg["max_len"])
    print(f"{len(encoded)} usable examples (of {n_train} sampled from {len(data)})", flush=True)

    interventions = {li: LoReFTIntervention(d, icfg["rank"], seed=cfg["seed"] + li).to(device)
                     for li in range(cfg["num_layers"])}
    hook = PositionEditHook(model, interventions)
    params = [p for iv in interventions.values() for p in iv.parameters()]
    n_params = sum(p.numel() for p in params)
    print(f"{len(interventions)} layer interventions, rank {icfg['rank']}, "
          f"{n_params:,} trainable params", flush=True)

    n_batches = math.ceil(len(encoded) / batch_size)
    total_steps = math.ceil(n_batches * epochs / grad_accum)
    opt = torch.optim.AdamW(params, lr=tcfg["lr"])
    sched = linear_warmup_decay(opt, total_steps, tcfg["warmup_ratio"])

    for ep in range(epochs):
        order = torch.randperm(len(encoded),
                               generator=torch.Generator().manual_seed(cfg["seed"] + ep)).tolist()
        running, pending = 0.0, 0
        for bi in range(n_batches):
            batch = [encoded[j] for j in order[bi * batch_size:(bi + 1) * batch_size]]
            ids, mask, locs, labels = collate_left_padded(
                batch, tok.pad_token_id, icfg["n_prefix"], icfg["n_suffix"], device)
            hook.set_locations(locs)
            with hook.injecting():
                loss = model(ids, attention_mask=mask, labels=labels).loss
            (loss / grad_accum).backward()
            pending += 1
            if pending == grad_accum or bi == n_batches - 1:
                opt.step()
                sched.step()
                opt.zero_grad()
                pending = 0
            running += float(loss.detach())
            if (bi + 1) % tcfg["log_every"] == 0:
                print(f"  epoch {ep+1}/{epochs}  batch {bi+1}/{n_batches}  "
                      f"mean CE={running / (bi + 1):.4f}  lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)
        print(f"epoch {ep+1}/{epochs} done  mean CE={running / n_batches:.4f}", flush=True)

    hook.remove()
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "loreft_interventions.pt"
    torch.save({"meta": {"base_model": cfg["base_model"], "hidden_dim": d,
                         "num_layers": cfg["num_layers"], **icfg,
                         "n_train": len(encoded), "epochs": epochs, "seed": cfg["seed"]},
                "state": {li: iv.state_dict() for li, iv in interventions.items()}}, out_path)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
