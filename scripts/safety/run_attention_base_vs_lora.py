"""Does the LoRA finetune significantly change attention vs the base model?

Run base and LoRA on the SAME prompts (held-out harmful + DDXPlus task), capture the
last-query-token attention at every layer/head, and compare per head with
``1 - cos(base_attn, lora_attn)`` (same prompt => same key positions => comparable).
Reports the per-layer divergence profile and the most-changed heads, separately for
harmful and task prompts — to see whether the LoRA reroutes attention, and whether the
change concentrates at the late layers where the refusal erosion lives.

    uv run python -m scripts.safety.run_attention_base_vs_lora \
        --config configs/safety/route_safety_qwen.yaml --adapter results/safety/qwen_sweep/adapter_d600
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

from scripts.safety.extract_refusal_shifts import prompt_ids, set_seed, user_turn
from src.probes.context_fatigue.attention_capture import SelectiveAttentionCapture
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, select_valid_indices
from src.probes.safety.safety_data import load_harmful


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default="results/safety/qwen_sweep/adapter_d600")
    ap.add_argument("--n-harmful", type=int, default=30)
    ap.add_argument("--n-task", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers = list(range(28))

    nh = cfg["direction"]["n_harmful"]
    harmful = load_harmful()[nh:nh + args.n_harmful]
    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf = cfg["ddxplus"]["n_filler"]
    task = build_cases(ds, valid[nf:nf + args.n_task], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = SelectiveAttentionCapture(base, layers)
    lora_model = PeftModel.from_pretrained(base, args.adapter).eval()

    def attn(prompt_msgs):
        ids = prompt_ids(tokenizer, prompt_msgs)
        capture.clear()
        capture.enabled = True
        with torch.no_grad():
            lora_model(torch.tensor([ids], device=args.device), use_cache=False)  # adapter state set by caller
        capture.enabled = False
        return {L: capture.captured[L].numpy() for L in layers}  # (n_heads, seq_len)

    def divergence_set(prompts_msgs):
        # per (layer, head): mean over prompts of 1 - cos(base_attn, lora_attn)
        per_head = {L: [] for L in layers}
        for pm in prompts_msgs:
            with lora_model.disable_adapter():
                b = attn(pm)
            la = attn(pm)
            for L in layers:
                bh, lh = b[L], la[L]
                num = (bh * lh).sum(1)
                den = np.linalg.norm(bh, axis=1) * np.linalg.norm(lh, axis=1) + 1e-9
                per_head[L].append(1.0 - num / den)  # (n_heads,)
        return {L: np.mean(per_head[L], axis=0) for L in layers}  # (n_heads,)

    out = {}
    for name, prompts in [("harmful", [user_turn(p) for p in harmful]),
                          ("task", [chat_messages(c.prompt_text) for c in task])]:
        div = divergence_set(prompts)
        layer_mean = {L: float(div[L].mean()) for L in layers}
        # top changed heads overall
        flat = sorted(((float(div[L][h]), L, int(h)) for L in layers for h in range(div[L].shape[0])),
                      reverse=True)
        out[name] = {
            "layer_mean_divergence": layer_mean,
            "overall_mean": float(np.mean([div[L].mean() for L in layers])),
            "top_heads": [{"layer": L, "head": h, "divergence": round(d, 3)} for d, L, h in flat[:10]],
        }
        print(f"\n=== {name} === overall mean 1-cos = {out[name]['overall_mean']:.3f}")
        print("  per-layer mean divergence:")
        for L in layers[::4] + [layers[-1]]:
            print(f"    L{L:2d}: {layer_mean[L]:.3f}")
        print("  top-5 changed heads:", [(t['layer'], t['head'], t['divergence']) for t in out[name]['top_heads'][:5]])
    capture.remove()

    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output"]["dir"]) / "attention_base_vs_lora.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {cfg['output']['dir']}/attention_base_vs_lora.json")


if __name__ == "__main__":
    main()
