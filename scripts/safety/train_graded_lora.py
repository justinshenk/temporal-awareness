"""Train a LoRA adapter on one graded-risk content bucket (the weight route).

For bucket ``Bx`` this finetunes the base model on that bucket's MCQ train slice (same
4-option letter-answer format as the ICL/steering arms), so the route sweep can compare
the safety cost of the WEIGHT route against the ACTIVATION route at matched content.

    HF_TOKEN=... uv run python -m scripts.safety.train_graded_lora \
        --config configs/safety/graded_risk_qwen.yaml --bucket B3

The adapter is written to ``{output.adapter_root}/{bucket}`` (gitignored). Train every
bucket before running ``run_graded_risk_sweep.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from scripts.lora_icl.train_ddxplus_lora import encode_example, fit_lora, set_seed
from src.probes.safety.graded_risk_data import BUCKETS, load_buckets
from src.probes.safety.mcq_icl import chat_mcq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--bucket", required=True, choices=BUCKETS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    d = cfg["data"]

    out_dir = Path(cfg["output"]["adapter_root"]) / args.bucket
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading bucket {args.bucket} train cases ...")
    bucket = load_buckets([args.bucket], cfg["seed"], d["n_filler"], d["n_fit"],
                          d["n_eval"], d["n_train"])[args.bucket]
    print(f"  {len(bucket.train)} train cases (bucket has {bucket.n_available} rows)")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [
        encode_example(tokenizer, c.prompt_text, c.gold_letter, cfg["train"]["max_seq_len"],
                       chat_fn=chat_mcq)
        for c in bucket.train
    ]
    fit_lora(cfg, examples, out_dir, args.device, tokenizer)


if __name__ == "__main__":
    main()
