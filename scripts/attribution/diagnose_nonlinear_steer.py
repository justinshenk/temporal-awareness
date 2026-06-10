"""Diagnose why nonlinear-steer recovers 0%: is it coherent-but-wrong (on-manifold) or gibberish
(off-manifold coherence collapse)? Print base / ridge-steer / nonlinear-steer generations, and the
injected-shift norm ratio ‖f(a)‖/‖a‖ at the acting site.

    uv run python -m scripts.attribution.diagnose_nonlinear_steer \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import gsm8k_problems, load_base_and_lora, prompt_token_ids
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.nonlinear_estimator import DeltaMLP, NonlinearSteerHook
from src.probes.safety.steering_hook import LinearPrimalSteerHook


def gen(model, tok, prompt_ids, max_new):
    out = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, L = cfg["device"], args.layer
    tok, base, lora = load_base_and_lora(cfg)

    maps_dir = Path(cfg["output"]["maps_dir"])
    W = torch.load(maps_dir / f"W_L{L}.pt")["W"].to(torch.float32)
    sd = torch.load(maps_dir / f"delta_mlp_L{L}.pt")
    hidden = sd["net.1.weight"].shape[0]
    mlp = DeltaMLP(W.shape[0], hidden).to(device).to(torch.float32)
    mlp.load_state_dict(sd)
    mlp.eval()

    problems = gsm8k_problems(cfg["eval"]["split"], 8, skip=0)[:args.n]
    for q, gold in problems:
        pid = prompt_token_ids(tok, q, device)
        print("\n" + "=" * 80)
        print(f"Q: {q[:90]}...  gold={gold:g}")
        with lora.disable_adapter():
            print(f"\n[BASE]\n{gen(base, tok, pid, args.max_new)[:400]}")
        rh = LinearPrimalSteerHook(base, {L: W}, 1.0)
        with lora.disable_adapter():
            print(f"\n[RIDGE-STEER]\n{gen(base, tok, pid, args.max_new)[:400]}")
        rh.remove()
        nh = NonlinearSteerHook(base, mlp, L, alpha=1.0)
        with lora.disable_adapter():
            print(f"\n[NONLINEAR-STEER]\n{gen(base, tok, pid, args.max_new)[:400]}")
        nh.remove()
        print(f"\n[LORA]\n{gen(lora, tok, pid, args.max_new)[:400]}")


if __name__ == "__main__":
    main()
