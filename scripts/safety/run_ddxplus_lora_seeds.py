"""DDXPlus LoRA across seeds: is the big task gain + refusal erosion robust? (mirror of
run_medmcqa_lora; the headline route-dependence result, seed-checked).

    uv run python -m scripts.safety.run_ddxplus_lora_seeds --config configs/safety/route_safety_qwen.yaml --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.lora_icl.train_ddxplus_lora import collate, encode_example
from scripts.safety.extract_refusal_shifts import generate_reply, prompt_ids, set_seed, user_turn
from scripts.safety.run_route_safety_sweep import ddxplus_accuracy
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.lora_icl.ddxplus_cases import build_cases, select_valid_indices
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.safety_data import load_harmful


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--n-harmful", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="ddxplus_lora.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    mc, max_new, n_o = cfg["extract"]["max_ctx"], cfg["eval"]["max_new"], cfg["ddxplus"]["n_options"]

    nh = cfg["direction"]["n_harmful"]
    h_eval = load_harmful()[nh:nh + args.n_harmful]
    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, n_o)
    random.Random(cfg["seed"]).shuffle(valid)
    train = build_cases(ds, valid[:args.n_train], evidence_db, n_o, cfg["seed"])
    ev = build_cases(ds, valid[args.n_train:args.n_train + args.n_eval], evidence_db, n_o, cfg["seed"])

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device)

    def refusal(m):
        return refusal_rate([generate_reply(m, tokenizer, prompt_ids(tokenizer, user_turn(p)), args.device, max_new)
                             for p in h_eval])

    model.eval()
    base_acc = ddxplus_accuracy(model, tokenizer, ev, args.device, mc)[0]
    base_ref = refusal(model)
    print(f"  base: acc={base_acc:.3f} refusal={base_ref:.3f}")

    examples = [encode_example(tokenizer, c.prompt_text, c.gold_letter, 1024) for c in train]
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.train()
    loader = DataLoader(examples, batch_size=8, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            (model(**batch).loss / 2).backward()
            if (i + 1) % 2 == 0:
                optim.step()
                optim.zero_grad()
        optim.step()
        optim.zero_grad()
        print(f"  epoch {epoch} done")

    model.eval()
    lora_acc = ddxplus_accuracy(model, tokenizer, ev, args.device, mc)[0]
    lora_ref = refusal(model)
    res = {"seed": cfg["seed"], "base_acc": base_acc, "base_refusal": base_ref,
           "lora_acc": lora_acc, "lora_refusal": lora_ref,
           "lora_task_gain": lora_acc - base_acc, "lora_refusal_drop": base_ref - lora_ref}
    print(f"  LoRA: acc={lora_acc:.3f} (gain {lora_acc-base_acc:+.3f}) refusal={lora_ref:.3f} (drop {base_ref-lora_ref:+.3f})")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / args.out).write_text(json.dumps(res, indent=2))
    print(f"\nSaved {out_dir}/{args.out}")


if __name__ == "__main__":
    main()
