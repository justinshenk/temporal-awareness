"""Generalization: run the core pipeline (headroom -> task-vector steering -> safety) on a
SECOND dataset, MedMCQA (real medical-exam MCQ; distinct from DDXPlus's synthetic format).

  1. headroom: zero-shot vs few-shot ICL accuracy (does the model need the task?).
  2. transfer: fit the mean ICL task vector, steer the base, measure accuracy.
  3. safety: refusal under the steer (does steering a NEW task's vector erode refusal?).

    uv run python -m scripts.safety.run_medmcqa_pipeline --config configs/safety/route_safety_qwen.yaml
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import capture_resid, generate_reply, prompt_ids, set_seed, user_turn
from src.probes.extraction import PerTokenResidualCapture
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless
from src.probes.safety.steering_hook import AdditionSteeringHook

L4 = ["A", "B", "C", "D"]
SYS = "Answer the following multiple-choice question with the letter of the correct option."


@dataclass
class MQ:
    prompt_text: str
    gold_letter: str


def medmcqa_cases(ds, idxs):
    out = []
    for i in idxs:
        r = ds[i]
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        lines = [r["question"]] + [f"{L4[j]}) {opts[j]}" for j in range(4)] + ["\nAnswer:"]
        out.append(MQ("\n".join(lines), L4[r["cop"]]))
    return out


def chat(p):
    return [{"role": "user", "content": f"{SYS}\n\n{p}"}]


def icl(tokenizer, fillers, final, mc, ft):
    msgs, budget = [], int(mc * ft)
    for fc in fillers:
        trial = msgs + chat(fc.prompt_text) + [{"role": "assistant", "content": fc.gold_letter}]
        if len(tokenizer.apply_chat_template(trial, add_generation_prompt=False, tokenize=True)) > budget:
            break
        msgs = trial
    return msgs + list(final)


def parse4(t):
    t = t.strip().upper()
    if t and t[0] in "ABCD":
        return t[0]
    m = re.findall(r"\b([A-D])\b", t)
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--few", type=int, default=5)
    ap.add_argument("--n-fit", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--n-filler", type=int, default=40)
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
    h_rfit, s_rfit = load_harmful()[:nh], load_harmless()[:cfg["direction"]["n_harmless"]]

    mm = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=cfg["seed"])
    fill = medmcqa_cases(mm, range(args.n_filler))
    fit = medmcqa_cases(mm, range(args.n_filler, args.n_filler + args.n_fit))
    ev = medmcqa_cases(mm, range(args.n_filler + args.n_fit, args.n_filler + args.n_fit + args.n_eval))

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(msgs):
        return capture_resid(base, capture, prompt_ids(tokenizer, msgs), args.device)

    hr = [resid(user_turn(p)) for p in h_rfit]
    sr = [resid(user_turn(p)) for p in s_rfit]
    rdir = {L: refusal_direction(np.stack([x[L] for x in hr]), np.stack([x[L] for x in sr])) for L in layers}
    bfit = [resid(chat(c.prompt_text)) for c in fit]
    ifit = [resid(icl(tokenizer, fill[:args.few], chat(c.prompt_text), mc, ft)) for c in fit]
    d = {L: np.mean([i[L] - b[L] for i, b in zip(ifit, bfit)], axis=0) for L in layers}
    d_orth = {L: d[L] - np.dot(d[L], rdir[L]) / np.dot(rdir[L], rdir[L]) * rdir[L] for L in layers}
    capture.remove()

    def acc(k=0, hook=None):
        correct = n = 0
        for c in ev:
            msgs = icl(tokenizer, fill[:k], chat(c.prompt_text), mc, ft) if k else chat(c.prompt_text)
            ids = prompt_ids(tokenizer, msgs)
            if len(ids) > mc - max_new:
                continue
            pred = parse4(generate_reply(base, tokenizer, ids, args.device, max_new))
            if pred:
                n += 1
                correct += int(pred == c.gold_letter)
        return correct / n if n else float("nan")

    def refusal(hook):
        return refusal_rate([generate_reply(base, tokenizer, prompt_ids(tokenizer, user_turn(p)), args.device, max_new)
                             for p in h_eval])

    res = {"zeroshot": acc(0), f"icl_{args.few}shot": acc(args.few), "base_refusal": refusal(None)}
    print(f"  HEADROOM: zero-shot={res['zeroshot']:.3f}  {args.few}-shot ICL={res[f'icl_{args.few}shot']:.3f}  base refusal={res['base_refusal']:.3f}")
    for name, vec in [("steer", d), ("steer_orth", d_orth)]:
        for a in alphas:
            h = AdditionSteeringHook(base, {L: torch.tensor(a * vec[L]) for L in layers})
            res[f"{name}_a{a}"] = {"acc": acc(0), "refusal": refusal(h)}
            h.remove()
            print(f"  {name} a={a}: acc={res[f'{name}_a{a}']['acc']:.3f} refusal={res[f'{name}_a{a}']['refusal']:.3f}")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "medmcqa_pipeline.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved {out_dir}/medmcqa_pipeline.json")


if __name__ == "__main__":
    main()
