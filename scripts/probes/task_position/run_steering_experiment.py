"""Causal steering experiment for the within_task_fraction probe.

For each test trace, runs Gemma-9B-IT three times:
  - no_steer (baseline)
  - early_steer (force probe readout to 0.1 at every token)
  - late_steer (force probe readout to 0.9 at every token)
At each case's prediction-site token, records option_probs / pred / correct.
Compares per-condition accuracy and calibration gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import OPTION_LABELS
from src.probes.task_position.probes import RidgeProbe
from src.probes.task_position.steering import ProbeSteeringHook

CONDITIONS = [
    ("no_steer", None),
    ("early_steer", 0.1),
    ("late_steer", 0.9),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument(
        "--activations",
        default="results/probes/task_position/gemma-9b-it/activations.pt",
    )
    p.add_argument(
        "--probe",
        default="results/probes/task_position/gemma-9b-it/probes/within_task_fraction_L10.pkl",
    )
    p.add_argument(
        "--correctness",
        default="results/probes/task_position/gemma-9b-it/correctness.json",
    )
    p.add_argument(
        "--split",
        default="results/probes/task_position/gemma-9b-it/probes/split.json",
    )
    p.add_argument(
        "--out",
        default="results/probes/task_position/2026-04-13-v2-steering.md",
    )
    p.add_argument("--steer-layer", type=int, default=10)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def get_option_token_ids(tokenizer):
    out = {}
    for letter in OPTION_LABELS:
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(ids) == 1:
            out[letter] = ids[0]
        else:
            out[letter] = tokenizer.encode(letter, add_special_tokens=False)[0]
    return out


def main():
    args = parse_args()

    print(f"Loading {args.activations}...")
    blob = torch.load(args.activations, weights_only=False)
    traces = blob["traces"]

    with open(args.split) as f:
        split = json.load(f)
    test_ids = set(int(x) for x in split["test_ids"])

    with open(args.correctness) as f:
        correctness_by_trace = json.load(f)

    print(f"Loading probe {args.probe}...")
    probe = RidgeProbe.load(args.probe)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    option_token_ids = get_option_token_ids(tokenizer)
    option_id_tensor = torch.tensor(
        [option_token_ids[l] for l in OPTION_LABELS], device=args.device
    )

    hook = ProbeSteeringHook(model, layer=args.steer_layer, probe=probe, target=0.0)

    # Sanity check: verify prediction_site points to a "model is about to speak"
    # position by decoding a window of tokens around it.
    first_test_trace = next((t for t in traces if t["trace_id"] in test_ids), None)
    if first_test_trace is not None:
        first_cr = correctness_by_trace.get(str(first_test_trace["trace_id"]), [])
        if first_cr and "prediction_site" in first_cr[0]:
            ps = first_cr[0]["prediction_site"]
            tokens_window = first_test_trace["tokens"][max(0, ps - 5) : ps + 6]
            decoded = tokenizer.decode(tokens_window)
            print(f"\n[Sanity check] First test trace prediction_site={ps}")
            print(f"  tokens[{max(0, ps-5)}:{ps+6}] decoded: {repr(decoded)}")
            # The window should end near a "start_of_turn>model" marker
            if "<start_of_turn>" not in decoded and "model" not in decoded.lower():
                raise RuntimeError(
                    f"Prediction site sanity check FAILED. Expected tokens near "
                    f"'<start_of_turn>model', got: {repr(decoded)}\n"
                    "The prediction_site offset is wrong — abort."
                )
            print("  [OK] Looks like a model-is-about-to-speak position.\n")
        else:
            raise RuntimeError(
                "correctness.json records are missing 'prediction_site' field. "
                "Re-run extract_activations.py --eval-correctness first."
            )

    records = []
    for t in traces:
        if t["trace_id"] not in test_ids:
            continue
        cr = correctness_by_trace.get(str(t["trace_id"]), [])
        if not cr:
            continue
        tokens = t["tokens"]
        input_ids = torch.tensor([tokens], device=args.device)

        for cond_name, target in CONDITIONS:
            if target is None:
                hook.enabled = False
                with torch.no_grad():
                    out = model(input_ids, use_cache=False)
            else:
                hook.target = float(target)
                with hook.steering(), torch.no_grad():
                    out = model(input_ids, use_cache=False)
            logits = out.logits[0]  # (seq, vocab)
            for case in cr:
                ci = case["case_index"]
                pos = case["prediction_site"]
                option_logits = logits[pos][option_id_tensor].float()
                probs = torch.softmax(option_logits, dim=0).cpu().numpy()
                option_probs = {l: float(probs[i]) for i, l in enumerate(OPTION_LABELS)}
                pred = max(option_probs, key=option_probs.get)
                records.append(
                    {
                        "trace_id": t["trace_id"],
                        "case_index": ci,
                        "gold": case["gold"],
                        "condition": cond_name,
                        "pred": pred,
                        "correct": int(pred == case["gold"]),
                        "confidence": option_probs[pred],
                    }
                )
            del out, logits
            torch.cuda.empty_cache()

        print(f"  trace {t['trace_id']}: done all 3 conditions")

    hook.remove()

    df = pd.DataFrame(records)

    summary = (
        df.groupby("condition")
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    summary["calibration_gap"] = summary["mean_confidence"] - summary["accuracy"]
    print("\nSummary:")
    print(summary.to_string(index=False))

    pivot = df.pivot_table(
        index=["trace_id", "case_index"], columns="condition", values="correct"
    ).reset_index()
    print("\nFlip stats (case-level):")
    if "no_steer" in pivot.columns and "early_steer" in pivot.columns:
        flips_to_correct = int(((pivot["no_steer"] == 0) & (pivot["early_steer"] == 1)).sum())
        flips_to_wrong = int(((pivot["no_steer"] == 1) & (pivot["early_steer"] == 0)).sum())
        print(f"  early_steer: {flips_to_correct} wrong→right, {flips_to_wrong} right→wrong")
    if "no_steer" in pivot.columns and "late_steer" in pivot.columns:
        flips_to_correct = int(((pivot["no_steer"] == 0) & (pivot["late_steer"] == 1)).sum())
        flips_to_wrong = int(((pivot["no_steer"] == 1) & (pivot["late_steer"] == 0)).sum())
        print(f"  late_steer:  {flips_to_correct} wrong→right, {flips_to_wrong} right→wrong")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Causal steering experiment (v2 stretch)\n\n")
    lines.append(
        "Steers the L10 residual stream so that the trained `within_task_fraction` "
        "probe's readout at every token equals a target value, then re-runs Gemma-9B-IT "
        "on the v1 test traces. Three conditions: `no_steer` (baseline), `early_steer` "
        "(target=0.1, 'pretend you just started this case'), `late_steer` (target=0.9, "
        "'pretend you're about to finish this case'). All other context unchanged.\n\n"
    )
    lines.append("## Per-condition summary\n\n")
    lines.append(summary.to_markdown(index=False) + "\n\n")
    lines.append("## Per-case records (head)\n\n")
    lines.append(df.head(20).to_markdown(index=False) + "\n\n")
    lines.append(
        "## Interpretation\n\n"
        "If `early_steer` lowers the calibration gap relative to `no_steer`, "
        "AND `late_steer` raises it symmetrically, then the within_task_fraction "
        "direction is causally implicated in the fatigue calibration gap. If only "
        "one direction has an effect, or both perturb in the same direction, the "
        "story is messier — but still informative about how the model uses this "
        "representation.\n"
    )
    out_path.write_text("".join(lines))
    print(f"\nWrote {out_path}")

    df.to_csv(
        out_path.parent / "2026-04-13-v2-steering-records.csv", index=False
    )


if __name__ == "__main__":
    main()
