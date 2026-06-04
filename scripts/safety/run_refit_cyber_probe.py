"""OOD-luck + δ=0 refit on a real, non-medical, lower-base-refusal domain (cyber/code).

Same logic as run_steer_refit_probe but domain = cybersecurity/code, all real data:
  fit task = Alpaca code instruction->output (ICL "coding assistant" task vector),
  in-domain harmful = AdvBench cyber prompts (hack/malware/steal), lower base refusal than medical.

Maps from the benign ICL shift Δ:
  naive  = ridge(A_benign, Δ)
  orth   = ridge(A_benign, Δ ⊥ r)
  refit  = ridge([A_benign; A_harm], [Δ; 0])      (teach δ=0 on cyber-harmful in-domain)

Reported on HELD-OUT cyber-harmful (refusal dynamics) AND held-out benign code (the SELECTIVITY check:
refit must keep a large ‖W·a‖ on benign while killing it on harmful — free-text has no clean MCQ acc).

    HF_TOKEN=... uv run python -m scripts.safety.run_refit_cyber_probe \
        --config configs/safety/route_safety_qwen.yaml --layer 21 --seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.safety.extract_refusal_shifts import capture_resid, generate_reply, prompt_ids, set_seed
from scripts.safety.run_lora_distill import ridge_maps
from scripts.safety.run_steer_ood_probe import orthogonalize_maps
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.subspace_metrics import vector_cosine
from src.probes.safety.cyber_domain_data import DOMAINS, icl_freetext, load_domain, user_turn
from src.probes.safety.refusal_classifier import refusal_rate
from src.probes.safety.refusal_direction import refusal_direction
from src.probes.safety.safety_data import load_harmful, load_harmless
from src.probes.safety.steering_hook import LinearConditionalSteerHook

# Vicuna v1.5 has no chat template in its tokenizer — supply the official one.
VICUNA_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + ' ' }}"
    "{% else %}{{ 'A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user\\'s questions. ' }}{% endif %}"
    "{% for m in messages %}{% if m['role'] == 'user' %}{{ 'USER: ' + m['content'] + ' ' }}"
    "{% elif m['role'] == 'assistant' %}{{ 'ASSISTANT: ' + m['content'] + '</s>' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}")
CHAT_TEMPLATES = {"vicuna": VICUNA_TEMPLATE}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--domain", choices=list(DOMAINS), default="cyber")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--n-fit", type=int, default=80)
    ap.add_argument("--n-demos", type=int, default=8)
    ap.add_argument("--n-harm-fit", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/safety/refit_cyber")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    layers = cfg["extract"]["layers"]
    mc, ft = cfg["extract"]["max_ctx"], cfg["extract"]["icl_fill_target"]
    L, mn = args.layer, args.max_new
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # generic refusal direction r (full harmful vs harmless, not just cyber)
    harmful_fit = load_harmful()[:cfg["direction"]["n_harmful"]]
    harmless_fit = load_harmless()[:cfg["direction"]["n_harmless"]]
    cyber_harm, code_pairs = load_domain(args.domain)
    print(f"[{args.domain}] in-domain harmful: {len(cyber_harm)} | benign instruction/output pairs: {len(code_pairs)}")

    print(f"Loading {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if cfg.get("chat_template") in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[cfg["chat_template"]]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
    capture = PerTokenResidualCapture(base, layers)

    def resid(msgs):
        return capture_resid(base, capture, prompt_ids(tokenizer, msgs), args.device)

    def refusal(prompts, hook_factory=None):
        hook = hook_factory() if hook_factory else None
        out = refusal_rate([generate_reply(base, tokenizer, prompt_ids(tokenizer, user_turn(p)), args.device, mn)
                            for p in prompts])
        if hook:
            hook.remove()
        return out

    hr = [resid(user_turn(p)) for p in harmful_fit]
    sr = [resid(user_turn(p)) for p in harmless_fit]
    r_by = {l: refusal_direction(np.stack([x[l] for x in hr]), np.stack([x[l] for x in sr])) for l in layers}

    def relnorm_cos(maps, acts):
        At, C = maps[L][0].numpy(), maps[L][1].numpy()
        rel = float(np.mean([np.linalg.norm((a[L] @ At) @ C) / (np.linalg.norm(a[L]) + 1e-8) for a in acts]))
        cr = float(np.mean([vector_cosine((a[L] @ At) @ C, r_by[L]) for a in acts]))
        return rel, cr

    seed_results = []
    for seed in seeds:
        set_seed(seed)
        rng = random.Random(seed)
        pairs = list(code_pairs); rng.shuffle(pairs)
        harm = list(cyber_harm); rng.shuffle(harm)
        demos = pairs[:args.n_demos]
        fit_instr = [i for i, _ in pairs[args.n_demos:args.n_demos + args.n_fit]]
        benign_eval = [i for i, _ in pairs[args.n_demos + args.n_fit:args.n_demos + args.n_fit + args.n_eval]]
        harm_fit = harm[:args.n_harm_fit]
        harm_eval = harm[args.n_harm_fit:args.n_harm_fit + args.n_eval]

        # benign fit: A_benign + Δ (ICL coding task vector)
        Ab = {l: [] for l in layers}; Db = {l: [] for l in layers}
        for instr in fit_instr:
            x = resid(user_turn(instr))
            y = resid(icl_freetext(tokenizer, demos, instr, mc, ft))
            for l in layers:
                Ab[l].append(x[l]); Db[l].append(y[l] - x[l])
        Ab = {l: np.stack(Ab[l]) for l in layers}; Db = {l: np.stack(Db[l]) for l in layers}
        Ah = {l: np.stack([resid(user_turn(p))[l] for p in harm_fit]) for l in layers}

        W_naive = ridge_maps(Ab, Db, layers, 1.0)
        W_orth = orthogonalize_maps(W_naive, r_by)
        A_aug = {l: np.concatenate([Ab[l], Ah[l]], 0) for l in layers}
        D_aug = {l: np.concatenate([Db[l], np.zeros_like(Ah[l])], 0) for l in layers}
        W_refit = ridge_maps(A_aug, D_aug, layers, 1.0)

        harm_acts = [resid(user_turn(p)) for p in harm_eval]
        benign_acts = [resid(user_turn(p)) for p in benign_eval]
        base_ref = refusal(harm_eval)
        rec = {"seed": seed, "base_refusal_harm": base_ref, "maps": {}}
        print(f"  seed{seed}: base refusal on cyber-harm = {base_ref:.2f}")
        for name, maps in [("naive", W_naive), ("orth", W_orth), ("refit", W_refit)]:
            rel_h, cr_h = relnorm_cos(maps, harm_acts)
            rel_b, _ = relnorm_cos(maps, benign_acts)               # selectivity: keep shift on benign
            ref = refusal(harm_eval, lambda m=maps: LinearConditionalSteerHook(base, m, args.alpha))
            rec["maps"][name] = {"wa_harm": rel_h, "cos_wa_r_harm": cr_h, "wa_benign": rel_b,
                                 "steer_refusal_harm": ref, "delta_refusal": base_ref - ref}
            print(f"    {name:>6}: ‖Wa‖harm={rel_h:.3f} ‖Wa‖benign={rel_b:.3f} cos(Wa,r)={cr_h:+.3f} "
                  f"refusal {base_ref:.2f}->{ref:.2f}")
        seed_results.append(rec)
    capture.remove()

    def avg(name, key):
        return float(np.mean([s["maps"][name][key] for s in seed_results]))
    bref = float(np.mean([s["base_refusal_harm"] for s in seed_results]))
    print(f"\n=== mean over seeds (held-out cyber-harmful, base refusal {bref:.2f}) ===")
    for name in ("naive", "orth", "refit"):
        print(f"  {name:>6}: ‖Wa‖harm={avg(name,'wa_harm'):.3f} ‖Wa‖benign={avg(name,'wa_benign'):.3f} "
              f"cos(Wa,r)={avg(name,'cos_wa_r_harm'):+.3f} steer_refusal={avg(name,'steer_refusal_harm'):.2f}")
    selective = avg("refit", "wa_benign") > 3 * avg("refit", "wa_harm")
    print(f"refit is selective (keeps shift on benign, kills it on harmful): {selective}")

    results = {"layer": L, "alpha": args.alpha, "seeds": seeds, "base_refusal_harm": bref,
               "per_seed": seed_results, "refit_selective": selective}
    (out_dir / "refit_cyber.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"Saved {out_dir}/refit_cyber.json")


if __name__ == "__main__":
    main()
