"""Domain shift: does task-vector steering generalize to MMLU, and stay safe cross-domain?

DDXPlus zero-shot was poor (0.14-0.32) — that headroom is why ICL/steering helped. MMLU is
general knowledge Qwen likely already knows, so we first check headroom (zero-shot vs
few-shot ICL). Then steer MMLU with:
  - an MMLU-native task vector (within-domain),
  - the DDXPlus medical task vector (cross-domain transfer),
each all-position and last-token, measuring MMLU accuracy AND refusal (does steering an
out-of-domain vector erode safety?).

    uv run python -m scripts.safety.run_mmlu_transfer --config configs/safety/route_safety_qwen.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import capture_resid, generate_reply, prompt_ids, set_seed, user_turn
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, icl_messages, select_valid_indices
from src.probes.safety.mcq_icl import chat_mcq, icl_mcq, mcq_accuracy, mcq_cases
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.safety_data import load_harmful
from src.probes.safety.steering_hook import AdditionSteeringHook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--few", type=int, default=5)
    ap.add_argument("--n-fit", type=int, default=80)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--n-filler", type=int, default=60)
    ap.add_argument("--n-harmful", type=int, default=25)
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, mc, ft, max_new = cfg["extract"]["layers"], cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"], cfg["eval"]["max_new"]
    alphas = [float(a) for a in args.alphas.split(",")]

    nh = cfg["direction"]["n_harmful"]
    h_eval = load_harmful()[nh:nh + args.n_harmful]

    # MMLU cases
    mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=cfg["seed"])
    mm_fill = mcq_cases(mm, range(args.n_filler))
    mm_fit = mcq_cases(mm, range(args.n_filler, args.n_filler + args.n_fit))
    mm_eval = mcq_cases(mm, range(args.n_filler + args.n_fit, args.n_filler + args.n_fit + args.n_eval))

    # DDXPlus cases (for the cross-domain medical vector)
    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    dd = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(dd, cfg["ddxplus"]["n_options"])
    dd_fill = build_cases(dd, valid[:cfg["ddxplus"]["n_filler"]], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    dd_fit = build_cases(dd, valid[cfg["ddxplus"]["n_filler"]:cfg["ddxplus"]["n_filler"] + args.n_fit],
                         evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(msgs):
        return capture_resid(base, capture, prompt_ids(tokenizer, msgs), args.device)

    def mean_shift(cases, chat_fn, icl_fn, fillers):
        base_r = [resid(chat_fn(c.prompt_text)) for c in cases]
        icl_r = [resid(icl_fn(tokenizer, fillers[:args.few], chat_fn(c.prompt_text), mc, ft)) for c in cases]
        return {L: np.mean([i[L] - b[L] for i, b in zip(icl_r, base_r)], axis=0) for L in layers}

    d_mmlu = mean_shift(mm_fit, chat_mcq, icl_mcq, mm_fill)
    d_ddx = mean_shift(dd_fit, chat_messages, icl_messages, dd_fill)
    capture.remove()

    def mmlu_acc(k=0, hook_factory=None):
        hook = hook_factory() if hook_factory else None
        acc, _ = mcq_accuracy(tokenizer, mm_eval,
                              lambda ids: generate_reply(base, tokenizer, ids, args.device, max_new),
                              mc, max_new, fillers=mm_fill, k=k, fill_target=ft)
        if hook:
            hook.remove()
        return acc

    def refusal_now(hook_factory=None):
        hook = hook_factory() if hook_factory else None
        r = refusal_rate([generate_reply(base, tokenizer, prompt_ids(tokenizer, user_turn(p)), args.device, max_new)
                          for p in h_eval])
        if hook:
            hook.remove()
        return r

    results = {"mmlu_zeroshot": {"acc": mmlu_acc(0)},
               f"mmlu_icl_{args.few}shot": {"acc": mmlu_acc(args.few)},
               "base_refusal": refusal_now()}
    print(f"  MMLU zero-shot acc={results['mmlu_zeroshot']['acc']:.3f} | "
          f"{args.few}-shot ICL acc={results[f'mmlu_icl_{args.few}shot']['acc']:.3f} | base refusal={results['base_refusal']:.3f}")

    def steer(vec, a, last):
        return lambda: AdditionSteeringHook(base, {L: torch.tensor(a * vec[L]) for L in layers}, last_token=last)

    for name, vec in [("mmlu_vec", d_mmlu), ("ddx_vec", d_ddx)]:
        for a in alphas:
            for mode, last in [("all", False), ("last", True)]:
                key = f"steer_{name}_{mode}_a{a}"
                results[key] = {"acc": mmlu_acc(0, steer(vec, a, last)), "refusal": refusal_now(steer(vec, a, last))}
                print(f"  {key}: acc={results[key]['acc']:.3f} refusal={results[key]['refusal']:.3f}")

    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output"]["dir"]) / "mmlu_transfer.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {cfg['output']['dir']}/mmlu_transfer.json")


if __name__ == "__main__":
    main()
