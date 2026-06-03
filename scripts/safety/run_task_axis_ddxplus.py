"""Confirm the shared task axis ON TASK prompts (DDXPlus), Qwen — the clean contrast
to the harmful-prompt result where ICL and LoRA shifts were ~orthogonal.

At the DDXPlus prediction site, per layer:
    icl_shift  = resid(base, case WITH k DDXPlus demos) - resid(base, case clean)
    lora_shift = resid(LoRA, case clean)                - resid(base, case clean)
Report cos(mean icl_shift, mean lora_shift). High here (but ~0 on harmful prompts) =>
ICL and finetuning share the TASK direction; they diverge only OFF-task (where the LoRA
carries the refusal side-component).

    uv run python -m scripts.safety.run_task_axis_ddxplus \
        --config configs/safety/route_safety_qwen.yaml \
        --adapter results/safety/qwen_sweep/adapter_d600 --icl-k 16 --n-eval 40
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import capture_resid, prompt_ids, set_seed
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import (
    build_cases,
    chat_messages,
    icl_messages,
    select_valid_indices,
)
from src.probes.lora_icl.subspace_metrics import vector_cosine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default="results/safety/qwen_sweep/adapter_d600")
    ap.add_argument("--icl-k", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers = cfg["extract"]["layers"]
    mc, ft = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf = cfg["ddxplus"]["n_filler"]
    fillers = build_cases(ds, valid[:nf], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    # eval cases disjoint from the filler pool
    eval_cases = build_cases(ds, valid[nf:nf + args.n_eval], evidence_db,
                             cfg["ddxplus"]["n_options"], cfg["seed"])

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(model, case, k=0):
        final = chat_messages(case.prompt_text)
        msgs = icl_messages(tokenizer, fillers[:k], final, mc, ft) if k else final
        return capture_resid(model, capture, prompt_ids(tokenizer, msgs), args.device)

    base_clean = [resid(base, c) for c in eval_cases]
    icl = [resid(base, c, args.icl_k) for c in eval_cases]
    lora_model = PeftModel.from_pretrained(base, args.adapter).eval()
    lora = [resid(lora_model, c) for c in eval_cases]
    capture.remove()

    rows = []
    for L in layers:
        icl_m = np.mean([i[L] - b[L] for i, b in zip(icl, base_clean)], axis=0)
        lora_m = np.mean([lo[L] - b[L] for lo, b in zip(lora, base_clean)], axis=0)
        rows.append({"layer": L, "cos_icl_lora": vector_cosine(icl_m, lora_m)})
        print(f"  L{L:2d}: cos(icl_shift, lora_shift) on DDXPlus = {rows[-1]['cos_icl_lora']:+.3f}")

    out = {"base_model": cfg["base_model"], "adapter": args.adapter, "icl_k": args.icl_k,
           "n_eval": len(eval_cases), "per_layer": rows,
           "peak_cos": max(r["cos_icl_lora"] for r in rows)}
    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output"]["dir"]) / "task_axis_ddxplus.json").write_text(json.dumps(out, indent=2))
    print(f"\npeak cos = {out['peak_cos']:+.3f} (contrast: ~0 on harmful prompts)")


if __name__ == "__main__":
    main()
