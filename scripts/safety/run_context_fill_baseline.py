"""Baseline: pure context-fill effect on refusal for the BASE model.

Fills the context with neutral benign Q&A (Alpaca) that is unrelated to the harmful
question, then asks a self-contained ("local") harmful question and measures refusal.
No LoRA. Isolates context *length* from filler content and from finetuning.

Usage:
    HF_TOKEN=... uv run python -m scripts.safety.run_context_fill_baseline \
        --config configs/safety/context_fill_baseline_gemma.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.safety.refusal_classifier import is_refusal
from src.probes.safety.safety_data import load_harmful
from scripts.safety.run_ablation_capstone import generate, set_seed


def alpaca_turns(cfg) -> list[tuple[str, str]]:
    """Neutral (instruction, response) filler pairs with no extra input field."""
    ds = load_dataset(cfg["filler"]["dataset"], split="train")
    turns = []
    for r in ds:
        if r["instruction"].strip() and not r["input"].strip() and r["output"].strip():
            turns.append((r["instruction"].strip(), r["output"].strip()))
        if len(turns) >= cfg["filler"]["n_pool"]:
            break
    return turns


def fill_context(tokenizer, turns, final_user: str, max_ctx: int, fill: float):
    """Accumulate neutral Q&A turns up to `fill` of max_ctx, then the local harmful question."""
    msgs: list[dict[str, str]] = []
    budget = int(max_ctx * fill)
    for user, asst in turns:
        trial = msgs + [{"role": "user", "content": user}, {"role": "assistant", "content": asst}]
        n = len(tokenizer.apply_chat_template(trial, add_generation_prompt=False, tokenize=True))
        if n > budget:
            break
        msgs = trial
    return msgs + [{"role": "user", "content": final_user}]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    turns = alpaca_turns(cfg)
    ho = cfg["harmful"]["offset"]
    harmful = load_harmful()[ho : ho + cfg["harmful"]["n"]]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()

    rows = []
    for fill in cfg["fills"]:
        refs = []
        approx_tokens = 0
        for p in harmful:
            msgs = fill_context(tokenizer, turns, p, cfg["max_ctx"], fill)
            ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            approx_tokens = len(ids)
            out = generate(model, tokenizer, msgs, args.device, cfg["max_new_refusal"])
            refs.append(is_refusal(out))
        rec = {"fill": fill, "refusal_rate": float(np.mean(refs)), "ctx_tokens": approx_tokens}
        rows.append(rec)
        print(f"fill {fill:.2f} (~{approx_tokens} tok): refusal {rec['refusal_rate']:.3f}")

    lines = [
        "# Context-fill baseline — base model, local harmful question, neutral filler",
        "",
        f"`{cfg['base_model']}` (no LoRA) | {len(harmful)} self-contained harmful questions | "
        f"filler = neutral Alpaca Q&A (unrelated to the question) | max_ctx {cfg['max_ctx']}",
        "",
        "| Context fill | ~tokens | refusal rate |",
        "|-------------:|--------:|-------------:|",
    ]
    for r in rows:
        lines.append(f"| {r['fill']:.0%} | {r['ctx_tokens']} | {r['refusal_rate']:.3f} |")

    lo = min(r["refusal_rate"] for r in rows)
    hi = rows[0]["refusal_rate"]
    lines += [
        "",
        "## Reading",
        "",
        f"- Base refusal across fill 0%→{cfg['fills'][-1]:.0%}: {hi:.2f} → "
        f"{rows[-1]['refusal_rate']:.2f} (min {lo:.2f}).",
        "- Flat ⇒ pure context length does not erode the base model's refusal on a local question "
        "(the long-context safety collapse seen with the LoRA is finetuning-induced, not a generic "
        "context-length effect). A drop ⇒ context length alone erodes refusal even without "
        "finetuning or content priming.",
    ]
    report = Path(cfg["output"]["report"])
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
