"""Train a MedMCQA LoRA and measure its task gain + refusal — the route-dependence (weight
route) on the second dataset, to compare against ICL/steering (activation route).

    uv run python -m scripts.safety.run_medmcqa_lora --config configs/safety/route_safety_qwen.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.lora_icl.train_ddxplus_lora import collate
from scripts.safety.extract_refusal_shifts import generate_reply, prompt_ids, set_seed, user_turn
from scripts.safety.run_medmcqa_pipeline import chat, medmcqa_cases, parse4
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.safety_data import load_harmful


def encode(tokenizer, prompt_text, gold, max_seq_len):
    pids = tokenizer.apply_chat_template(chat(prompt_text), add_generation_prompt=True, tokenize=True)
    if not isinstance(pids, list):
        pids = pids["input_ids"]
    aids = tokenizer(gold, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return {"input_ids": (pids + aids)[:max_seq_len], "labels": ([-100] * len(pids) + aids)[:max_seq_len]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--n-harmful", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None, help="override cfg seed (data shuffle + init)")
    ap.add_argument("--out", default="medmcqa_lora.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    mc, max_new = cfg["extract"]["max_ctx"], cfg["eval"]["max_new"]

    nh = cfg["direction"]["n_harmful"]
    h_eval = load_harmful()[nh:nh + args.n_harmful]
    mm = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=cfg["seed"])
    train = medmcqa_cases(mm, range(args.n_train))
    ev = medmcqa_cases(mm, range(args.n_train, args.n_train + args.n_eval))

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device)

    def acc(m):
        c = n = 0
        for case in ev:
            ids = prompt_ids(tokenizer, chat(case.prompt_text))
            pred = parse4(generate_reply(m, tokenizer, ids, args.device, max_new))
            if pred:
                n += 1
                c += int(pred == case.gold_letter)
        return c / n if n else float("nan")

    def refusal(m):
        return refusal_rate([generate_reply(m, tokenizer, prompt_ids(tokenizer, user_turn(p)), args.device, max_new)
                             for p in h_eval])

    model.eval()
    base_acc, base_ref = acc(model), refusal(model)
    print(f"  base: acc={base_acc:.3f} refusal={base_ref:.3f}")

    examples = [encode(tokenizer, c.prompt_text, c.gold_letter, 1024) for c in train]
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
    lora_acc, lora_ref = acc(model), refusal(model)
    res = {"base_acc": base_acc, "base_refusal": base_ref, "lora_acc": lora_acc, "lora_refusal": lora_ref,
           "lora_task_gain": lora_acc - base_acc, "lora_refusal_drop": base_ref - lora_ref}
    print(f"  LoRA: acc={lora_acc:.3f} (gain {lora_acc-base_acc:+.3f}) refusal={lora_ref:.3f} (drop {base_ref-lora_ref:+.3f})")

    res["seed"] = cfg["seed"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / args.out).write_text(json.dumps(res, indent=2))
    print(f"\nSaved {out_dir}/{args.out}")


if __name__ == "__main__":
    main()
