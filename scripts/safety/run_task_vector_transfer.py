"""Transfer the ICL task gain to a FRESH (zero-shot) chat via additive steering.

Full activation patching fails because the representation drifts across context fill
(the many-shot ICL activation lives in a drifted regime; transplanting it into a fresh
prompt mismatches). Additive steering instead adds the task *direction* to the fresh
prompt's residual stream, preserving its own state.

The direction matters because raw icl_shift = task component + context-drift component:
  d_lora     : mean LoRA shift — drift-free by construction (oracle; shares task axis)
  d_icl_few  : mean ICL shift, FEW shots — little drift (no-finetuning option)
  d_icl_many : mean ICL shift, MANY shots — drift-contaminated
We steer zero-shot DDXPlus with each (sweep alpha) and compare to a last-token patch.

    uv run python -m scripts.safety.run_task_vector_transfer \
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

from scripts.safety.extract_refusal_shifts import capture_resid, prompt_ids, set_seed
from scripts.safety.run_route_safety_sweep import ddxplus_accuracy, parse_letter
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, icl_messages, select_valid_indices
from src.probes.safety.steering_hook import AdditionSteeringHook


class LastTokenPatchHook:
    """Overwrite the prefill last-token residual with a per-case vector (the patch baseline)."""

    def __init__(self, model, vectors):
        self.vectors = {li: v.detach().float() for li, v in vectors.items()}
        self._hooks = [model.model.layers[li].register_forward_hook(self._mk(li)) for li in self.vectors]

    def _mk(self, li):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] > 1:  # prefill only
                hs = hs.clone()
                hs[:, -1, :] = self.vectors[li].to(hs.device, hs.dtype)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default="results/safety/qwen_sweep/adapter_d600")
    ap.add_argument("--few", type=int, default=4)
    ap.add_argument("--many", type=int, default=48)
    ap.add_argument("--n-fit", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers = cfg["extract"]["layers"]
    mc, ft = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"]
    alphas = [float(a) for a in args.alphas.split(",")]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf = cfg["ddxplus"]["n_filler"]
    fillers = build_cases(ds, valid[:nf], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    fit = build_cases(ds, valid[nf:nf + args.n_fit], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    ev = build_cases(ds, valid[nf + args.n_fit:nf + args.n_fit + args.n_eval], evidence_db,
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

    # Fit-set mean shifts -> task directions (kept at natural magnitude).
    base_fit = [resid(base, c) for c in fit]
    icl_few_fit = [resid(base, c, args.few) for c in fit]
    icl_many_fit = [resid(base, c, args.many) for c in fit]
    lora_model = PeftModel.from_pretrained(base, args.adapter).eval()
    lora_fit = [resid(lora_model, c) for c in fit]

    def mean_shift(var, bse):
        return {L: np.mean([v[L] - b[L] for v, b in zip(var, bse)], axis=0) for L in layers}

    dirs = {"d_lora": mean_shift(lora_fit, base_fit),
            "d_icl_few": mean_shift(icl_few_fit, base_fit),
            "d_icl_many": mean_shift(icl_many_fit, base_fit)}
    capture.remove()

    with lora_model.disable_adapter():
        results = {"baselines": {
            "zeroshot": ddxplus_accuracy(lora_model, tokenizer, ev, args.device, mc)[0],
            f"icl_{args.few}shot": ddxplus_accuracy(lora_model, tokenizer, ev, args.device, mc, fillers, args.few, ft)[0],
            f"icl_{args.many}shot": ddxplus_accuracy(lora_model, tokenizer, ev, args.device, mc, fillers, args.many, ft)[0],
        }}
        print(f"  baselines: {results['baselines']}")

        # Additive steering on zero-shot prompts.
        results["steering"] = {}
        for dname, d in dirs.items():
            results["steering"][dname] = {}
            for a in alphas:
                hook = AdditionSteeringHook(base, {L: torch.tensor(a * d[L]) for L in layers})
                acc = ddxplus_accuracy(lora_model, tokenizer, ev, args.device, mc)[0]
                hook.remove()
                results["steering"][dname][f"alpha_{a}"] = acc
                print(f"  steer {dname} a={a}: acc={acc:.3f}")

    # Patch baseline: per-case overwrite of the prefill last-token residual with the
    # case's own many-shot ICL residual (the "transplant" that drift should break).
    capture2 = PerTokenResidualCapture(base, layers)
    correct = n = 0
    with lora_model.disable_adapter():
        for c in ev:
            capture2.clear()
            with capture2.capturing():
                base(torch.tensor([prompt_ids(tokenizer, icl_messages(
                    tokenizer, fillers[:args.many], chat_messages(c.prompt_text), mc, ft))],
                    device=args.device), use_cache=False)
            patch = {L: torch.tensor(capture2.captured[L][-1].numpy()) for L in layers}
            hook = LastTokenPatchHook(base, patch)
            ids = prompt_ids(tokenizer, chat_messages(c.prompt_text))
            out = base.generate(torch.tensor([ids], device=args.device), max_new_tokens=6,
                               do_sample=False, pad_token_id=tokenizer.eos_token_id)
            hook.remove()
            pred = parse_letter(tokenizer.decode(out[0][len(ids):], skip_special_tokens=True))
            if pred:
                n += 1
                correct += int(pred == c.gold_letter)
    capture2.remove()
    results["patch_lasttoken"] = {"acc": (correct / n if n else None), "n": n}
    print(f"  patch last-token (transplant): acc={results['patch_lasttoken']['acc']} (n={n})")

    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output"]["dir"]) / "task_vector_transfer.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {cfg['output']['dir']}/task_vector_transfer.json")


if __name__ == "__main__":
    main()
