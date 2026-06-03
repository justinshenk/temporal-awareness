"""Is the steered accuracy stable WITHIN one process? (decisive determinism test)

Compute the task vector once, then evaluate base and steered DDXPlus accuracy several
times in the same process. If steered accuracy is identical across repeats, the forward
is intra-process deterministic and the cross-run 0.65-vs-0.22 gap was a *deterministic*
difference between runs (different d / eval), not numerical noise. If it varies, the
forward itself is nondeterministic.

    uv run python -m scripts.safety.run_steer_repro --config configs/safety/route_safety_qwen.yaml
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
from scripts.safety.run_route_safety_sweep import ddxplus_accuracy
from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import build_cases, chat_messages, icl_messages, select_valid_indices
from src.probes.safety.steering_hook import AdditionSteeringHook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--few", type=int, default=4)
    ap.add_argument("--n-fit", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    layers, mc, ft = cfg["extract"]["layers"], cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"]
    alphas = [float(a) for a in args.alphas.split(",")]

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
    capture = PerTokenResidualCapture(base, layers)

    def resid(case, k=0):
        final = chat_messages(case.prompt_text)
        msgs = icl_messages(tokenizer, fillers[:k], final, mc, ft) if k else final
        return capture_resid(base, capture, prompt_ids(tokenizer, msgs), args.device)

    d = {L: np.mean([i[L] - b[L] for i, b in zip([resid(c, args.few) for c in fit],
                                                 [resid(c) for c in fit])], axis=0) for L in layers}
    capture.remove()
    d_norms = {L: float(np.linalg.norm(d[L])) for L in layers}

    out = {"d_norms": d_norms, "runs": {}}
    out["runs"]["base"] = [ddxplus_accuracy(base, tokenizer, ev, args.device, mc)[0] for _ in range(args.repeats)]
    print(f"  base acc repeats: {out['runs']['base']}")
    for a in alphas:
        accs = []
        for _ in range(args.repeats):
            sh = AdditionSteeringHook(base, {L: torch.tensor(a * d[L]) for L in layers})
            accs.append(ddxplus_accuracy(base, tokenizer, ev, args.device, mc)[0])
            sh.remove()
        out["runs"][f"steered_a{a}"] = accs
        print(f"  steered a={a} acc repeats: {accs}")

    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output"]["dir"]) / "steer_repro.json").write_text(json.dumps(out, indent=2))
    print(f"  d_norms: { {L: round(v,1) for L,v in d_norms.items()} }")
    print(f"\nSaved {cfg['output']['dir']}/steer_repro.json")


if __name__ == "__main__":
    main()
