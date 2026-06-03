"""ICL accuracy on the SAME MedMCQA slice the LoRA was evaluated on (cases 600:650), to
disambiguate the LoRA's ~0 gain: is the slice headroom-free (ICL also flat) or does ICL help
where the LoRA didn't?

    uv run python -m scripts.safety.run_medmcqa_icl_check --config configs/safety/route_safety_qwen.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import generate_reply, prompt_ids, set_seed
from scripts.safety.run_medmcqa_pipeline import chat, icl, medmcqa_cases, parse4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-train", type=int, default=600)   # LoRA used 0:600; eval was 600:650
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--few", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    mc, ft, max_new = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"], cfg["eval"]["max_new"]

    mm = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=cfg["seed"])
    ev = medmcqa_cases(mm, range(args.n_train, args.n_train + args.n_eval))           # same as LoRA eval
    fill = medmcqa_cases(mm, range(args.n_train + args.n_eval, args.n_train + args.n_eval + 40))

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()

    def acc(k):
        c = n = 0
        for case in ev:
            msgs = icl(tokenizer, fill[:k], chat(case.prompt_text), mc, ft) if k else chat(case.prompt_text)
            ids = prompt_ids(tokenizer, msgs)
            if len(ids) > mc - max_new:
                continue
            pred = parse4(generate_reply(base, tokenizer, ids, args.device, max_new))
            if pred:
                n += 1
                c += int(pred == case.gold_letter)
        return c / n if n else float("nan")

    print(f"  same slice (600:650): zero-shot={acc(0):.3f}  {args.few}-shot ICL={acc(args.few):.3f}")


if __name__ == "__main__":
    main()
