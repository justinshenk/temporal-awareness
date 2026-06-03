"""Does the task-vector steering (the finetuning alternative) preserve safety?

The load-bearing claim: additive steering of the ICL task vector confers DDXPlus skill
WITHOUT eroding refusal — unlike LoRA finetuning. The task vector is added to EVERY prompt
(incl. harmful), so we measure both:
  - DDXPlus accuracy on task prompts (the gain)
  - refusal rate on held-out harmful prompts (should stay ~base, since the task direction
    is off the refusal axis)
across steering strengths, vs no-steer base and vs the LoRA finetune.

Uses the ICL task direction (few-shot, no finetuning needed) — d_icl_few.

    uv run python -m scripts.safety.run_steering_safety \
        --config configs/safety/route_safety_qwen.yaml
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

from scripts.safety.extract_refusal_shifts import capture_resid, generate_reply, prompt_ids, set_seed, user_turn
from scripts.safety.run_route_safety_sweep import ddxplus_accuracy
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, icl_messages, select_valid_indices
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless
from src.probes.safety.steering_hook import AdditionSteeringHook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default="results/safety/qwen_sweep/adapter_d600")
    ap.add_argument("--few", type=int, default=4)
    ap.add_argument("--n-fit", type=int, default=40)
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers = cfg["extract"]["layers"]
    mc, ft, max_new = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"], cfg["eval"]["max_new"]
    alphas = [float(a) for a in args.alphas.split(",")]

    nh, ns = cfg["direction"]["n_harmful"], cfg["direction"]["n_harmless"]
    harmful = load_harmful()
    h_rfit = harmful[:nh]
    h_eval = harmful[nh:nh + cfg["eval"]["n_harmful"]]
    s_rfit = load_harmless()[:ns]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf, nt = cfg["ddxplus"]["n_filler"], cfg["ddxplus"]["n_task_eval"]
    fillers = build_cases(ds, valid[:nf], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    fit = build_cases(ds, valid[nf:nf + args.n_fit], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    task_eval = build_cases(ds, valid[nf + args.n_fit:nf + args.n_fit + nt], evidence_db,
                            cfg["ddxplus"]["n_options"], cfg["seed"])

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(case, k=0):
        final = chat_messages(case.prompt_text)
        msgs = icl_messages(tokenizer, fillers[:k], final, mc, ft) if k else final
        return capture_resid(base, capture, prompt_ids(tokenizer, msgs), args.device)

    base_fit = [resid(c) for c in fit]
    icl_fit = [resid(c, args.few) for c in fit]
    d = {L: np.mean([i[L] - b[L] for i, b in zip(icl_fit, base_fit)], axis=0) for L in layers}
    # refusal direction r, and the task vector with its r-component projected out
    hr = [capture_resid(base, capture, prompt_ids(tokenizer, user_turn(p)), args.device) for p in h_rfit]
    sr = [capture_resid(base, capture, prompt_ids(tokenizer, user_turn(p)), args.device) for p in s_rfit]
    capture.remove()
    r = {L: refusal_direction(np.stack([x[L] for x in hr]), np.stack([x[L] for x in sr])) for L in layers}
    d_orth = {L: d[L] - np.dot(d[L], r[L]) / np.dot(r[L], r[L]) * r[L] for L in layers}

    def refusal_now():
        return refusal_rate([generate_reply(base, tokenizer, prompt_ids(tokenizer, user_turn(p)),
                                            args.device, max_new) for p in h_eval])

    results = {}
    results["base"] = {"task_acc": ddxplus_accuracy(base, tokenizer, task_eval, args.device, mc)[0],
                       "refusal": refusal_now()}
    print(f"  base: acc={results['base']['task_acc']:.3f} refusal={results['base']['refusal']:.3f}")
    for name, vec in [("steer", d), ("steer_orth", d_orth)]:
        for a in alphas:
            hook = AdditionSteeringHook(base, {L: torch.tensor(a * vec[L]) for L in layers})
            key = f"{name}_a{a}"
            results[key] = {"task_acc": ddxplus_accuracy(base, tokenizer, task_eval, args.device, mc)[0],
                            "refusal": refusal_now()}
            hook.remove()
            print(f"  {key}: acc={results[key]['task_acc']:.3f} refusal={results[key]['refusal']:.3f}")

    # LoRA finetune reference (task gain WITH safety erosion)
    lora = PeftModel.from_pretrained(base, args.adapter).eval()
    results["lora_finetune"] = {"task_acc": ddxplus_accuracy(lora, tokenizer, task_eval, args.device, mc)[0],
                                "refusal": refusal_rate([generate_reply(lora, tokenizer,
                                    prompt_ids(tokenizer, user_turn(p)), args.device, max_new) for p in h_eval])}
    print(f"  LoRA finetune: acc={results['lora_finetune']['task_acc']:.3f} "
          f"refusal={results['lora_finetune']['refusal']:.3f}")

    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output"]["dir"]) / "steering_safety.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {cfg['output']['dir']}/steering_safety.json")


if __name__ == "__main__":
    main()
