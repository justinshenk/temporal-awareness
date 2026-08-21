"""Cross-model LoRA capability transmission through per-layer linear activation maps.

Does the DDXPlus capability of a LoRA trained on Qwen2.5-7B-Instruct transmit to the untrained
Qwen2.5-1.5B-Instruct through ridge maps between their residual spaces? Brief:
``tasks/lora_map_transfer_execution.md``. Four sequential phases, one process each:

    capture-donor      7B: map-corpus + eval-panel residuals (base / +LoRA / +shuffled-LoRA),
                       floor + ceiling accuracies, and the self-steer sanity arms (base + its
                       own mean shift, no map) — if those are null, transmission is dead at home.
    capture-recipient  1.5B: map-corpus residuals, floor + own-adapter ceiling accuracies.
    fit-maps           CPU: per-layer ridge maps donor→recipient on the shared corpus,
                       ridge strength chosen per layer on the held-out split.
    run-steering       1.5B: decode-time steering with the mapped shift at several layers and
                       doses, against norm-matched random controls per (layer, dose) and the
                       mapped *shuffled-adapter* shift (format-not-task control).

Both models have 28 decoder layers, so layer pairing is identity; maps are 3584→1536. The eval
panel (100 test-split cases) and map corpus (400 disjoint test-split cases) are the seeded
``disjoint_split`` slices, so every arm in every phase scores the identical items.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import DEFAULT_EVIDENCE_PATH, load_evidence_db
from src.probes.extraction import PerTokenResidualCapture
from src.probes.lora_icl.ddxplus_cases import (
    build_cases,
    chat_messages,
    disjoint_split,
    icl_messages,
    select_valid_indices,
)
from src.probes.lora_icl.linear_map_transfer import (
    LinearMap,
    fit_linear_map,
    norm_matched_random,
)
from src.probes.lora_icl.shift_extraction import last_token_residual
from src.probes.safety.steering_hook import AdditionSteeringHook

STEER_LAYERS = [7, 14, 18, 21]
ALPHAS = [1.0, 2.0]
LAM_GRID = [1e-1, 1e1, 1e3, 1e5]
N_MAP_CORPUS = 400


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["capture-donor", "capture-recipient",
                                     "fit-maps", "run-steering", "recipient-selfsteer",
                                     "icl-route", "dose-curve"])
    p.add_argument("--donor-config", default="configs/lora_icl/ddxplus_qwen_lora.yaml")
    p.add_argument("--recipient-config", default="configs/lora_icl/ddxplus_qwen1.5b_lora.yaml")
    p.add_argument("--out-dir", default="results/lora_icl/map_transfer")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new", type=int, default=6)
    return p.parse_args()


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def build_panels(cfg):
    """Eval panel (100 cases) + map corpus (400 cases), disjoint, from the test split."""
    evidence_db = load_evidence_db(DEFAULT_EVIDENCE_PATH)
    ds = load_dataset(cfg["data"]["dataset"], split=cfg["data"]["eval_split"])
    valid = select_valid_indices(ds, cfg["data"]["n_options"])
    corpus_pool, eval_idx = disjoint_split(
        valid, cfg["data"]["n_train_cases"], cfg["data"]["n_eval_cases"], cfg["seed"])
    eval_cases = build_cases(ds, eval_idx, evidence_db, cfg["data"]["n_options"], cfg["seed"])
    corpus = build_cases(ds, corpus_pool[:N_MAP_CORPUS], evidence_db,
                         cfg["data"]["n_options"], cfg["seed"])
    return eval_cases, corpus


def prompt_ids(tokenizer, case, device):
    ids = tokenizer.apply_chat_template(
        chat_messages(case.prompt_text), add_generation_prompt=True, tokenize=True)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return torch.tensor([ids], device=device)


@torch.no_grad()
def capture_states(model, capture, tokenizer, cases, device) -> np.ndarray:
    """(n, n_layers, d) float32 final-token residuals, layer-ordered by capture.layers."""
    rows = []
    for c in cases:
        capture.clear()
        with capture.capturing():
            model(prompt_ids(tokenizer, c, device), use_cache=False)
        site = last_token_residual(capture.captured)
        rows.append(np.stack([site[li] for li in capture.layers]).astype(np.float32))
    return np.stack(rows)


@torch.no_grad()
def eval_accuracy(model, tokenizer, cases, device, max_new) -> dict:
    correct, parsed, preds = 0, 0, []
    for c in cases:
        ids = prompt_ids(tokenizer, c, device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"[A-E]", text)
        pred = m.group(0) if m else None
        preds.append(pred)
        parsed += pred is not None
        correct += pred == c.gold_letter
    n = len(cases)
    return {"n": n, "accuracy": correct / n, "parse_rate": parsed / n, "preds": preds}


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    donor_cfg = load_cfg(args.donor_config)
    recip_cfg = load_cfg(args.recipient_config)
    eval_cases, corpus = build_panels(donor_cfg)
    golds = [c.gold_letter for c in eval_cases]
    (out / "eval_gold.json").write_text(json.dumps(golds))

    if args.phase == "capture-donor":
        tok = AutoTokenizer.from_pretrained(donor_cfg["base_model"])
        base = AutoModelForCausalLM.from_pretrained(
            donor_cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
        n_layers = len(base.model.layers)
        capture = PerTokenResidualCapture(base, list(range(n_layers)))
        evals = {}

        print("donor: map-corpus states + eval base states + floor", flush=True)
        np.save(out / "donor_corpus_states.npy",
                capture_states(base, capture, tok, corpus, args.device))
        base_eval = capture_states(base, capture, tok, eval_cases, args.device)
        np.save(out / "donor_eval_base.npy", base_eval)
        evals["donor_floor"] = eval_accuracy(base, tok, eval_cases, args.device, args.max_new)
        print(f"  donor_floor acc={evals['donor_floor']['accuracy']:.3f}", flush=True)

        lora = PeftModel.from_pretrained(base, donor_cfg["output"]["adapter_dir"],
                                         adapter_name="real").eval()
        lora.load_adapter(donor_cfg["output"]["shuffled_adapter_dir"], adapter_name="shuffled")
        deltas = {}
        for tag in ("real", "shuffled"):
            lora.set_adapter(tag)
            states = capture_states(lora, capture, tok, eval_cases, args.device)
            np.save(out / f"donor_eval_lora_{tag}.npy", states)
            deltas[tag] = (states.astype(np.float64) - base_eval).mean(axis=0)
            evals[f"donor_ceiling_{tag}"] = eval_accuracy(
                lora, tok, eval_cases, args.device, args.max_new)
            print(f"  donor_ceiling_{tag} acc={evals[f'donor_ceiling_{tag}']['accuracy']:.3f}",
                  flush=True)
        np.save(out / "delta_real.npy", deltas["real"])
        np.save(out / "delta_shuffled.npy", deltas["shuffled"])
        capture.remove()

        # Self-steer sanity: the donor's own mean shift, no map. If this is null the
        # mean-shift summary cannot carry the capability even at home.
        with lora.disable_adapter():
            for li in STEER_LAYERS:
                vec = torch.tensor(deltas["real"][li], dtype=torch.float32)
                hook = AdditionSteeringHook(base, {li: vec}, decode_time=True)
                evals[f"donor_selfsteer_L{li}"] = eval_accuracy(
                    lora, tok, eval_cases, args.device, args.max_new)
                hook.remove()
                print(f"  donor_selfsteer_L{li} "
                      f"acc={evals[f'donor_selfsteer_L{li}']['accuracy']:.3f}", flush=True)
        (out / "donor_evals.json").write_text(json.dumps(evals, indent=2))

    elif args.phase == "capture-recipient":
        tok = AutoTokenizer.from_pretrained(recip_cfg["base_model"])
        base = AutoModelForCausalLM.from_pretrained(
            recip_cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
        capture = PerTokenResidualCapture(base, list(range(len(base.model.layers))))
        evals = {}
        print("recipient: map-corpus states + floor + ceiling", flush=True)
        np.save(out / "recipient_corpus_states.npy",
                capture_states(base, capture, tok, corpus, args.device))
        capture.remove()
        evals["recipient_floor"] = eval_accuracy(base, tok, eval_cases, args.device,
                                                 args.max_new)
        print(f"  recipient_floor acc={evals['recipient_floor']['accuracy']:.3f}", flush=True)
        lora = PeftModel.from_pretrained(base, recip_cfg["output"]["adapter_dir"]).eval()
        evals["recipient_ceiling"] = eval_accuracy(lora, tok, eval_cases, args.device,
                                                   args.max_new)
        print(f"  recipient_ceiling acc={evals['recipient_ceiling']['accuracy']:.3f}",
              flush=True)
        (out / "recipient_evals.json").write_text(json.dumps(evals, indent=2))

    elif args.phase == "fit-maps":
        donor = np.load(out / "donor_corpus_states.npy").astype(np.float64)
        recip = np.load(out / "recipient_corpus_states.npy").astype(np.float64)
        assert donor.shape[0] == recip.shape[0], (donor.shape, recip.shape)
        profile = {}
        for li in range(donor.shape[1]):
            best = None
            for lam in LAM_GRID:
                m = fit_linear_map(donor[:, li], recip[:, li], lam=lam)
                if best is None or m.r2_holdout > best[1].r2_holdout:
                    best = (lam, m)
            lam, m = best
            profile[li] = {"lam": lam, "r2_holdout": m.r2_holdout}
            if li in STEER_LAYERS:
                np.savez(out / f"map_L{li}.npz", **m.to_arrays())
            print(f"  L{li:2d} lam={lam:g} holdout R2={m.r2_holdout:.4f}", flush=True)
        (out / "map_r2_profile.json").write_text(json.dumps(profile, indent=2))

    elif args.phase == "run-steering":
        tok = AutoTokenizer.from_pretrained(recip_cfg["base_model"])
        base = AutoModelForCausalLM.from_pretrained(
            recip_cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
        deltas = {"real": np.load(out / "delta_real.npy"),
                  "shuffled": np.load(out / "delta_shuffled.npy")}
        evals = {}
        for li in STEER_LAYERS:
            m = LinearMap.from_arrays(**np.load(out / f"map_L{li}.npz"))
            mapped = {tag: m.map_shift(deltas[tag][li]) for tag in deltas}
            for alpha in ALPHAS:
                arms = {
                    f"transfer_real_L{li}_a{alpha:g}": mapped["real"] * alpha,
                    f"transfer_shuffled_L{li}_a{alpha:g}": mapped["shuffled"] * alpha,
                    f"rand_control_L{li}_a{alpha:g}": norm_matched_random(
                        mapped["real"] * alpha, seed=1000 + li * 10 + int(alpha)),
                }
                for arm, vec in arms.items():
                    hook = AdditionSteeringHook(
                        base, {li: torch.tensor(vec, dtype=torch.float32)}, decode_time=True)
                    evals[arm] = eval_accuracy(base, tok, eval_cases, args.device,
                                               args.max_new)
                    hook.remove()
                    print(f"  {arm} acc={evals[arm]['accuracy']:.3f} "
                          f"parse={evals[arm]['parse_rate']:.2f}", flush=True)
        (out / "steering_evals.json").write_text(json.dumps(evals, indent=2))

    elif args.phase == "recipient-selfsteer":
        # Interpretive control for a transfer null: steer the recipient with its OWN adapter's
        # mean shift. If this is null too, the failure is the recipient's steerability (or the
        # mean-shift summary at 1.5B scale), not the cross-model map.
        tok = AutoTokenizer.from_pretrained(recip_cfg["base_model"])
        base = AutoModelForCausalLM.from_pretrained(
            recip_cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
        capture = PerTokenResidualCapture(base, list(range(len(base.model.layers))))
        base_eval = capture_states(base, capture, tok, eval_cases, args.device)
        lora = PeftModel.from_pretrained(base, recip_cfg["output"]["adapter_dir"]).eval()
        lora_eval = capture_states(lora, capture, tok, eval_cases, args.device)
        capture.remove()
        delta = (lora_eval.astype(np.float64) - base_eval).mean(axis=0)
        np.save(out / "delta_recipient_own.npy", delta)
        evals = {}
        with lora.disable_adapter():
            for li in STEER_LAYERS:
                for alpha in ALPHAS:
                    vec = torch.tensor(delta[li] * alpha, dtype=torch.float32)
                    hook = AdditionSteeringHook(base, {li: vec}, decode_time=True)
                    arm = f"recip_selfsteer_L{li}_a{alpha:g}"
                    evals[arm] = eval_accuracy(lora, tok, eval_cases, args.device,
                                               args.max_new)
                    hook.remove()
                    print(f"  {arm} acc={evals[arm]['accuracy']:.3f} "
                          f"parse={evals[arm]['parse_rate']:.2f}", flush=True)
        (out / "recipient_selfsteer_evals.json").write_text(json.dumps(evals, indent=2))

    elif args.phase == "icl-route":
        # E-A2: does the context route install the same direction as the weight route? Capture
        # the donor's ICL shift (same case, with vs without accumulated demonstrations), compare
        # its mean to the LoRA shift's mean per layer, and steer *clean* prompts with the mean
        # ICL shift — dose-matched to the LoRA delta whose behavioral effect is known (0.73).
        tok = AutoTokenizer.from_pretrained(donor_cfg["base_model"])
        base = AutoModelForCausalLM.from_pretrained(
            donor_cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
        n_layers = len(base.model.layers)
        capture = PerTokenResidualCapture(base, list(range(n_layers)))
        fillers = corpus  # disjoint from the eval panel by construction
        max_ctx, fill_target = 4096, 0.85

        def icl_ids(case):
            msgs = icl_messages(tok, fillers, chat_messages(case.prompt_text),
                                max_ctx, fill_target)
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            return torch.tensor([ids], device=args.device)

        print("icl-route: capturing ICL states + icl ceiling", flush=True)
        icl_rows = []
        correct = 0
        for c in eval_cases:
            ids = icl_ids(c)
            capture.clear()
            with capture.capturing(), torch.no_grad():
                # decoder only: the LM head's full-sequence fp32 logits are ~5 GB at this
                # context length and the capture never reads them
                base.model(ids, use_cache=False)
            site = last_token_residual(capture.captured)
            icl_rows.append(np.stack([site[li] for li in capture.layers]).astype(np.float32))
            with torch.no_grad():
                gen = base.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
            text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            m = re.search(r"[A-E]", text)
            correct += (m.group(0) if m else None) == c.gold_letter
            torch.cuda.empty_cache()
        capture.remove()
        icl_states = np.stack(icl_rows)
        base_eval = np.load(out / "donor_eval_base.npy").astype(np.float64)
        icl_delta = (icl_states.astype(np.float64) - base_eval).mean(axis=0)
        np.save(out / "delta_icl.npy", icl_delta)
        lora_delta = np.load(out / "delta_real.npy")
        evals = {"icl_ceiling": {"n": len(eval_cases), "accuracy": correct / len(eval_cases)}}
        print(f"  icl_ceiling acc={correct / len(eval_cases):.3f}", flush=True)
        cos_profile = {}
        for li in range(n_layers):
            a, b = icl_delta[li], lora_delta[li]
            cos_profile[li] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        (out / "icl_lora_cos_profile.json").write_text(json.dumps(cos_profile, indent=2))
        print("  cos(icl, lora) at steer layers: "
              + "  ".join(f"L{li}={cos_profile[li]:+.3f}" for li in STEER_LAYERS), flush=True)

        for li in STEER_LAYERS:
            raw = icl_delta[li]
            matched = raw * (np.linalg.norm(lora_delta[li]) / np.linalg.norm(raw))
            for tag, vec in (("raw", raw), ("normmatched", matched)):
                hook = AdditionSteeringHook(
                    base, {li: torch.tensor(vec, dtype=torch.float32)}, decode_time=True)
                arm = f"iclsteer_L{li}_{tag}"
                evals[arm] = eval_accuracy(base, tok, eval_cases, args.device, args.max_new)
                hook.remove()
                print(f"  {arm} acc={evals[arm]['accuracy']:.3f} "
                      f"parse={evals[arm]['parse_rate']:.2f}", flush=True)
        (out / "icl_route_evals.json").write_text(json.dumps(evals, indent=2))

    elif args.phase == "dose-curve":
        # E-A3: the conditionality index against LoRA training-set size. Nested seeded slices
        # ([:25] c [:75] c [:225] c [:600]) trained separately; per dose: ceiling, mean shift,
        # self-steer at the two best layers, plus the shift's norm and its cosine to the full-
        # dose shift. Hypothesis on record: the constant/register component is learned first.
        doses = [25, 75, 225]
        adapter_dirs = {d: f"results/safety/qwen_sweep/adapter_d{d}" for d in doses}
        tok = AutoTokenizer.from_pretrained(donor_cfg["base_model"])
        base = AutoModelForCausalLM.from_pretrained(
            donor_cfg["base_model"], torch_dtype=torch.bfloat16, device_map=args.device).eval()
        capture = PerTokenResidualCapture(base, list(range(len(base.model.layers))))
        base_eval = np.load(out / "donor_eval_base.npy").astype(np.float64)
        delta_600 = np.load(out / "delta_real.npy")
        first = doses[0]
        lora = PeftModel.from_pretrained(base, adapter_dirs[first],
                                         adapter_name=f"d{first}").eval()
        for d in doses[1:]:
            lora.load_adapter(adapter_dirs[d], adapter_name=f"d{d}")
        evals = {}
        for d in doses:
            lora.set_adapter(f"d{d}")
            states = capture_states(lora, capture, tok, eval_cases, args.device)
            delta = (states.astype(np.float64) - base_eval).mean(axis=0)
            np.save(out / f"delta_d{d}.npy", delta)
            evals[f"ceiling_d{d}"] = eval_accuracy(lora, tok, eval_cases, args.device,
                                                   args.max_new)
            print(f"  ceiling_d{d} acc={evals[f'ceiling_d{d}']['accuracy']:.3f}", flush=True)
            with lora.disable_adapter():
                for li in (18, 21):
                    hook = AdditionSteeringHook(
                        base, {li: torch.tensor(delta[li], dtype=torch.float32)},
                        decode_time=True)
                    arm = f"selfsteer_d{d}_L{li}"
                    evals[arm] = eval_accuracy(lora, tok, eval_cases, args.device,
                                               args.max_new)
                    hook.remove()
                    print(f"  {arm} acc={evals[arm]['accuracy']:.3f}", flush=True)
            for li in (18, 21):
                a, b = delta[li], delta_600[li]
                evals[f"geometry_d{d}_L{li}"] = {
                    "norm": float(np.linalg.norm(a)),
                    "norm_d600": float(np.linalg.norm(b)),
                    "cos_to_d600": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                }
        capture.remove()
        (out / "dose_curve_evals.json").write_text(json.dumps(evals, indent=2))

    print("phase done:", args.phase)


if __name__ == "__main__":
    main()
