"""The conditionality index's computation pole: constant mean-shift self-steer on GSM8K.

Same estimator as the DDXPlus arm in ``scripts/lora_icl/run_lora_map_transfer.py``: the mean
over the eval panel of resid(base+LoRA) − resid(base) at the final prompt token, injected
decode-time (final prefill position + every decode step) at a single layer, no adapter. On
DDXPlus that constant recovers 0.66--0.71 of the adapter's floor→ceiling gap in two models; the
register-vs-procedure account predicts it recovers ≈0 here, where the capability is multi-step
computation the base cannot do. Prior rungs bound *conditional* linear estimators (ridge maps,
DAgger, DAS) at ≤0.23 recovery; this is the strictly weaker constant rung, run with the exact
protocol of the register pole so the index is comparable across tasks.

    uv run python -m scripts.attribution.selfsteer_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.attribution.attribution_common import gsm8k_problems, load_base_and_lora
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.gsm8k_prompts import (
    extract_pred_number,
    metamath_prompt,
    numeric_match,
)
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.shift_extraction import last_token_residual
from src.probes.safety.steering_hook import AdditionSteeringHook

STEER_LAYERS = [8, 16, 20, 24]  # relative depths matching the DDXPlus arm's {7,14,18,21}/28,
                                # plus L20, the attribution program's canonical layer

_STOP_MARKERS = ("\n\n### Instruction", "\nBelow is an instruction", "\n\nBelow is an instruction")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/attribution/metamath_llama2_gsm8k.yaml")
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--out-dir", default="results/attribution/selfsteer_gsm8k")
    return p.parse_args()


@torch.no_grad()
def eval_accuracy(model, tok, problems, device, max_new) -> dict:
    correct, parsed, preds = 0, 0, []
    for question, gold in problems:
        ids = tok(metamath_prompt(question), return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        for marker in _STOP_MARKERS:
            text = text.split(marker)[0]
        pred = extract_pred_number(text)
        preds.append(pred)
        parsed += pred is not None
        correct += pred is not None and numeric_match(pred, gold)
    n = len(problems)
    return {"n": n, "accuracy": correct / n, "parse_rate": parsed / n, "preds": preds}


@torch.no_grad()
def capture_final_states(model, capture, tok, problems, device) -> np.ndarray:
    rows = []
    for question, _ in problems:
        ids = tok(metamath_prompt(question), return_tensors="pt").input_ids.to(device)
        capture.clear()
        with capture.capturing():
            model(ids, use_cache=False)
        site = last_token_residual(capture.captured)
        rows.append(np.stack([site[li] for li in capture.layers]).astype(np.float32))
    return np.stack(rows)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = cfg["device"]
    problems = gsm8k_problems(cfg["eval"]["split"], args.n_eval, skip=0)
    (out / "gold.json").write_text(json.dumps([g for _, g in problems]))

    tok, base, lora = load_base_and_lora(cfg)
    n_layers = len(base.model.layers)
    capture = PerTokenResidualCapture(base, list(range(n_layers)))
    evals = {}

    print("capturing base/lora final-token states + floor/ceiling", flush=True)
    with lora.disable_adapter():
        base_states = capture_final_states(base, capture, tok, problems, device)
        evals["floor"] = eval_accuracy(lora, tok, problems, device, args.max_new)
    print(f"  floor acc={evals['floor']['accuracy']:.3f} "
          f"parse={evals['floor']['parse_rate']:.2f}", flush=True)
    lora_states = capture_final_states(lora, capture, tok, problems, device)
    evals["ceiling"] = eval_accuracy(lora, tok, problems, device, args.max_new)
    print(f"  ceiling acc={evals['ceiling']['accuracy']:.3f} "
          f"parse={evals['ceiling']['parse_rate']:.2f}", flush=True)
    capture.remove()

    delta = (lora_states.astype(np.float64) - base_states).mean(axis=0)
    np.save(out / "delta_metamath.npy", delta)
    cos = [float(np.dot(a, delta[li]) / (np.linalg.norm(a) * np.linalg.norm(delta[li])))
           for li in STEER_LAYERS
           for a in (lora_states[:, li].astype(np.float64) - base_states[:, li].astype(np.float64))]
    print(f"  per-case shift alignment (pooled over steer layers): "
          f"mean cos={np.mean(cos):.3f}", flush=True)

    with lora.disable_adapter():
        for li in STEER_LAYERS:
            vec = torch.tensor(delta[li], dtype=torch.float32)
            hook = AdditionSteeringHook(base, {li: vec}, decode_time=True)
            arm = f"selfsteer_L{li}"
            evals[arm] = eval_accuracy(lora, tok, problems, device, args.max_new)
            hook.remove()
            print(f"  {arm} acc={evals[arm]['accuracy']:.3f} "
                  f"parse={evals[arm]['parse_rate']:.2f}", flush=True)

    floor, ceil = evals["floor"]["accuracy"], evals["ceiling"]["accuracy"]
    best = max(evals[f"selfsteer_L{li}"]["accuracy"] for li in STEER_LAYERS)
    evals["conditionality_index"] = {
        "recovered_fraction_best": (best - floor) / (ceil - floor) if ceil > floor else None,
        "floor": floor, "ceiling": ceil, "best_selfsteer": best,
    }
    (out / "selfsteer_evals.json").write_text(json.dumps(evals, indent=2))
    print(f"\nrecovered fraction (best layer): "
          f"{evals['conditionality_index']['recovered_fraction_best']:.3f}", flush=True)


if __name__ == "__main__":
    main()
