"""Capstone: how exact must the injected shift be to recover capability?

Lockstep-decode base at layer L, injecting ``W·a + t·(δ_true − W·a)`` every step (see
``blended_injection``): ``t=0`` is α=1 steering (expected ≈0%), ``t=1`` is the true-δ oracle
(expected ≈0.75 at L20). Sweeping ``t`` traces recovery vs shift fidelity — a sharp rise only near
``t=1`` proves the procedure needs a near-exact shift, which is why a 0.6-cosine ridge map fails.

    uv run python -m scripts.attribution.lockstep_fidelity_sweep \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --ts 0,0.5,0.8,0.9,1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import (
    gsm8k_problems,
    load_base_and_lora,
    prompt_token_ids,
)
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.gsm8k_prompts import extract_pred_number, numeric_match
from src.probes.attribution.lockstep_oracle import (
    OverwriteResidualHook,
    blended_injection,
    lockstep_generate,
)
from src.probes.extraction import PerTokenResidualCapture


def make_blend_fns(base, lora, capture, inject, L, W, t):
    """capture_residuals returns the blended injection value; base_logits is the patched base pass."""
    def capture_residuals(S):
        capture.clear()
        with lora.disable_adapter(), capture.capturing():
            base(S, use_cache=False)               # base residual a
        a = capture.captured[L]
        capture.clear()
        with capture.capturing():
            base(S, use_cache=False)               # LoRA residual on the same context
        lora_resid = capture.captured[L]
        return {L: blended_injection(a, lora_resid, W.to(a.dtype), t)}

    def base_logits(S):
        with inject.injecting(), lora.disable_adapter():
            return base(S, use_cache=False).logits

    return capture_residuals, base_logits


def eval_blend(base, lora, tok, problems, device, L, W, t, max_new):
    capture = PerTokenResidualCapture(base, [L])
    inject = OverwriteResidualHook(base, [L])
    cap_fn, logit_fn = make_blend_fns(base, lora, capture, inject, L, W, t)
    eos = tok.eos_token_id
    correct = 0
    for q, gold in problems:
        prompt_ids = prompt_token_ids(tok, q, device)
        out = lockstep_generate(prompt_ids, cap_fn, logit_fn, inject, [L], max_new, eos, device)
        text = tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        correct += bool(numeric_match(extract_pred_number(text), gold))
    capture.remove()
    inject.remove()
    return correct / len(problems)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--ts", default="0,0.5,0.8,0.9,1.0")
    ap.add_argument("--n-contrast", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, L = cfg["device"], args.layer
    ts = [float(x) for x in args.ts.split(",")]

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)
    W = torch.load(Path(cfg["output"]["maps_dir"]) / f"W_L{L}.pt")["W"].to(torch.float32)

    cache = Path(cfg["output"]["steer_json"]).parent / "lockstep_contrast_set.json"
    meta = json.loads(cache.read_text())
    scan = gsm8k_problems(cfg["eval"]["split"], meta["n_eval"], skip=0)
    contrast = [tuple(scan[i]) for i in meta["indices"]][:args.n_contrast]
    print(f"L={L}, {len(contrast)} contrast problems, ts={ts}", flush=True)

    results = {"layer": L, "n_contrast": len(contrast), "max_new": args.max_new, "recovery": {}}
    print("\n t   : recovery (=acc on contrast)", flush=True)
    for t in ts:
        acc = eval_blend(base, lora, tok, contrast, device, L, W, t, args.max_new)
        results["recovery"][t] = acc
        print(f" {t:.2f}: {acc:.3f}", flush=True)

    out_path = Path(args.out) if args.out else Path(cfg["output"]["steer_json"]).parent / f"lockstep_fidelity_L{L}.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
