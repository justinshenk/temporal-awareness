"""Why is steered accuracy not reproducible? Show the steered MCQ logits are knife-edge.

Two measurements on zero-shot DDXPlus, base vs task-vector-steered (α=1):
  1. Margin distribution: gap between the top-1 and top-2 option-letter logits. A tiny
     margin means a hair separates two answers — so ~1e-2 numerical jitter flips the pick.
  2. Perturbation-flip test: inject gaussian noise of relative scale ε (≈ bf16 jitter) at
     the steered layers and count how many answers flip vs the noise-free run. If steered
     flips en masse where base shrugs it off, the steered state is numerically fragile —
     which is exactly how two "deterministic" greedy runs can disagree by 0.65 vs 0.22.

    uv run python -m scripts.safety.run_steer_margin --config configs/safety/route_safety_qwen.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import capture_resid, prompt_ids, set_seed
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, icl_messages, select_valid_indices
from src.probes.safety.steering_hook import AdditionSteeringHook

LETTERS = ["A", "B", "C", "D", "E"]


class RelNoiseHook:
    """Add gaussian noise of relative L2 scale eps (per position) to each named layer."""

    def __init__(self, model, layers, eps):
        self.eps, self.enabled, self._hooks = eps, True, []
        for li in layers:
            self._hooks.append(model.model.layers[li].register_forward_hook(self._mk()))

    def _mk(self):
        def hook(module, inp, out):
            if not self.enabled:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            n = torch.randn_like(hs)
            n = n / (n.norm(dim=-1, keepdim=True) + 1e-6) * hs.norm(dim=-1, keepdim=True) * self.eps
            hs = hs + n
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--few", type=int, default=4)
    ap.add_argument("--n-fit", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--epsilons", default="0.003,0.01,0.03")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, mc, ft = cfg["extract"]["layers"], cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"]
    epsilons = [float(e) for e in args.epsilons.split(",")]

    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["ddxplus"]["dataset"], split=cfg["ddxplus"]["split"])
    valid = select_valid_indices(ds, cfg["ddxplus"]["n_options"])
    nf = cfg["ddxplus"]["n_filler"]
    bc = lambda lo, hi: build_cases(ds, valid[lo:hi], evidence_db, cfg["ddxplus"]["n_options"], cfg["seed"])
    fillers, fit = bc(0, nf), bc(nf, nf + args.n_fit)
    ev = bc(nf + args.n_fit, nf + args.n_fit + args.n_eval)

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    letter_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]

    capture = PerTokenResidualCapture(base, layers)
    def resid(case, k=0):
        final = chat_messages(case.prompt_text)
        msgs = icl_messages(tokenizer, fillers[:k], final, mc, ft) if k else final
        return capture_resid(base, capture, prompt_ids(tokenizer, msgs), args.device)
    d = {L: np.mean([i[L] - b[L] for i, b in zip([resid(c, args.few) for c in fit],
                                                 [resid(c) for c in fit])], axis=0) for L in layers}
    capture.remove()

    ev_ids = [prompt_ids(tokenizer, chat_messages(c.prompt_text)) for c in ev]

    @torch.no_grad()
    def predict():
        preds, margins = [], []
        for ids in ev_ids:
            lg = base(torch.tensor([ids], device=args.device)).logits[0, -1]
            opt = lg[letter_ids].float().cpu().numpy()
            order = np.argsort(opt)[::-1]
            preds.append(int(order[0]))
            margins.append(float(opt[order[0]] - opt[order[1]]))
        return preds, margins

    out = {}
    for cond, steer in [("base", False), (f"steered_a{args.alpha}", True)]:
        sh = AdditionSteeringHook(base, {L: torch.tensor(args.alpha * d[L]) for L in layers}) if steer else None
        ref_preds, margins = predict()
        flips = {}
        for eps in epsilons:
            rates = []
            for _ in range(args.draws):
                nh = RelNoiseHook(base, layers, eps)
                p, _ = predict()
                nh.remove()
                rates.append(float(np.mean([a != b for a, b in zip(p, ref_preds)])))
            flips[eps] = float(np.mean(rates))
        if sh:
            sh.remove()
        out[cond] = {
            "median_margin": float(np.median(margins)),
            "frac_margin_lt_0.5": float(np.mean(np.array(margins) < 0.5)),
            "frac_margin_lt_1.0": float(np.mean(np.array(margins) < 1.0)),
            "flip_rate_by_eps": flips,
        }
        print(f"  {cond}: median_margin={out[cond]['median_margin']:.3f} "
              f"frac<0.5={out[cond]['frac_margin_lt_0.5']:.2f} "
              f"flip_rate={ {e: round(r,3) for e,r in flips.items()} }")

    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output"]["dir"]) / "steer_margin.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {cfg['output']['dir']}/steer_margin.json")


if __name__ == "__main__":
    main()
