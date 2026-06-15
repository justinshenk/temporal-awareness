"""Train a matched LoRA on commonsense-170k (subset-first), Llama-2-7B — the LoReFT comparison arm.

Same 20k subset, prompt template (``"%s\n"``), target (``"the correct answer is X"``), base model,
and response-only CE labels as ``train_loreft_commonsense.py`` (it reuses that script's
``encode_examples`` and ``collate_left_padded``, so the supervised signal is byte-identical). The
ONLY difference is the adaptation method: a low-rank weight delta (LoRA, the LLM-Adapters config the
ReFT paper benchmarks against) instead of a subspace representation edit (LoReFT). This gives an
apples-to-apples pair for the activation-similarity comparison.

    uv run python -m scripts.attribution.train_lora_commonsense \
        --config configs/attribution/loreft_commonsense_llama2.yaml

Saves a standard PEFT adapter to ``{output.dir}/lora_commonsense/`` (adapter_config.json +
adapter_model.safetensors) for ``eval_commonsense_suite.py --lora <dir>``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model

from scripts.attribution.train_loreft_commonsense import (
    collate_left_padded,
    encode_examples,
    linear_warmup_decay,
    load_frozen_base,
)
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.commonsense_data import (
    load_commonsense_json,
    subset_examples,
)


def build_lora_config(lcfg: dict) -> LoraConfig:
    """A causal-LM LoRA config from the YAML ``lora:`` block (LLM-Adapters commonsense recipe)."""
    return LoraConfig(
        r=lcfg["rank"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=list(lcfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


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
    tok, model = load_frozen_base(cfg)                       # all params frozen, .eval()
    model = get_peft_model(model, build_lora_config(lcfg))   # injects trainable LoRA adapters
    model.train()
    model.print_trainable_parameters()

    data = load_commonsense_json(Path(dcfg["dir"]) / dcfg["train_file"])
    items = subset_examples(data, n_train, seed=cfg["seed"])  # same seed/subset as the LoReFT
    encoded = encode_examples(tok, items, dcfg["max_len"])
    print(f"{len(encoded)} usable examples (of {n_train} sampled from {len(data)})", flush=True)

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
            # reuse the LoReFT collate (response-only CE labels identical); locs are unused for LoRA
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

    out_dir = Path(args.out) if args.out else Path(cfg["output"]["dir"]) / "lora_commonsense"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    print(f"Saved adapter to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
