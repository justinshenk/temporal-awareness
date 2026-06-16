"""Train a LoRA donor on MuSiQue open-book multi-hop QA (Llama-2-7B) — Phase 0 of the generality test.

Mirrors ``train_lora_commonsense.py`` (same LLM-Adapters recipe, frozen bf16 base, response-only CE
via ``collate_left_padded``) but on the MuSiQue *composition* task: prompt = supporting passages +
question + CoT lead-in, target = worked hop chain ending ``The answer is: X``. Saves a standard PEFT
adapter to ``{output.lora_dir}`` for the attribution drivers (``--task multihop``).

    uv run python -m scripts.attribution.train_lora_multihop \
        --config configs/attribution/multihop_llama2.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from peft import get_peft_model

from scripts.attribution.train_lora_commonsense import build_lora_config
from scripts.attribution.train_loreft_commonsense import (
    collate_left_padded,
    linear_warmup_decay,
    load_frozen_base,
)
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.multihop_data import encode_multihop, load_musique


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-train", type=int, default=None, help="override data.n_train")
    ap.add_argument("--epochs", type=int, default=None, help="override lora.epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="override lora.batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="override lora.grad_accum")
    ap.add_argument("--out", default=None, help="override adapter output dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = cfg["device"]
    dcfg, icfg, lcfg = cfg["data"], cfg["intervention"], cfg["lora"]
    n_train = args.n_train or dcfg["n_train"]
    epochs = args.epochs or lcfg["epochs"]
    batch_size = args.batch_size or lcfg["batch_size"]
    grad_accum = args.grad_accum or lcfg["grad_accum"]

    print(f"Loading {cfg['base_model']} (frozen bf16) ...", flush=True)
    tok, model = load_frozen_base(cfg)
    model = get_peft_model(model, build_lora_config(lcfg))
    model.train()
    model.print_trainable_parameters()

    items = load_musique(dcfg["train_split"], n_train, seed=cfg["seed"])
    encoded = encode_multihop(tok, items, dcfg["max_len"])
    print(f"{len(encoded)} usable examples (of {len(items)} answerable sampled)", flush=True)

    n_batches = math.ceil(len(encoded) / batch_size)
    total_steps = math.ceil(n_batches * epochs / grad_accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lcfg["lr"])
    sched = linear_warmup_decay(opt, total_steps, lcfg["warmup_ratio"])

    for ep in range(epochs):
        order = torch.randperm(len(encoded),
                               generator=torch.Generator().manual_seed(cfg["seed"] + ep)).tolist()
        running = 0.0
        pending = 0
        for bi in range(n_batches):
            batch = [encoded[j] for j in order[bi * batch_size:(bi + 1) * batch_size]]
            ids, mask, _locs, labels = collate_left_padded(
                batch, tok.pad_token_id, icfg["n_prefix"], icfg["n_suffix"], device)
            loss = model(ids, attention_mask=mask, labels=labels).loss
            (loss / grad_accum).backward()
            pending += 1
            if pending == grad_accum or bi == n_batches - 1:
                opt.step()
                sched.step()
                opt.zero_grad()
                pending = 0
            running += float(loss.detach())
            if (bi + 1) % lcfg["log_every"] == 0:
                print(f"  epoch {ep+1}/{epochs}  batch {bi+1}/{n_batches}  "
                      f"mean CE={running / (bi + 1):.4f}  lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)
        print(f"epoch {ep+1}/{epochs} done  mean CE={running / n_batches:.4f}", flush=True)

    out_dir = Path(args.out) if args.out else Path(cfg["output"]["lora_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    print(f"Saved adapter to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
