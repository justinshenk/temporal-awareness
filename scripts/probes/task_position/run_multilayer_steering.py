"""Multi-layer causal steering experiment for the within_task_fraction probe.

Tests whether single-layer null result was due to compensation by downstream
layers, by hooking probes at multiple layers simultaneously and steering all
of them in concert.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import OPTION_LABELS
from src.probes.task_position.probes import RidgeProbe
from src.probes.task_position.steering import ProbeSteeringHook

CONDITIONS = [
    ("no_steer", None, []),
    ("early_L10", 0.1, [10]),
    ("late_L10", 0.9, [10]),
    ("early_L10_L20", 0.1, [10, 20]),
    ("late_L10_L20", 0.9, [10, 20]),
    ("early_L10_L20_L30", 0.1, [10, 20, 30]),
    ("late_L10_L20_L30", 0.9, [10, 20, 30]),
    ("early_all_late", 0.1, [10, 20, 30, 41]),
    ("late_all_late", 0.9, [10, 20, 30, 41]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument(
        "--activations",
        default="results/probes/task_position/gemma-9b-it/activations.pt",
    )
    p.add_argument(
        "--probe-dir",
        default="results/probes/task_position/gemma-9b-it/probes",
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
        default="results/probes/task_position/2026-04-13-v3-multilayer-steering.md",
    )
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


def load_probes(probe_dir: Path, layers: list[int]) -> dict[int, RidgeProbe]:
    probes = {}
    for layer in layers:
        path = probe_dir / f"within_task_fraction_L{layer}.pkl"
        probes[layer] = RidgeProbe.load(path)
    return probes


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

    # Verify prediction_site is available (sanity check from v2 fix)
    sample_record = next(
        c for tid in correctness_by_trace for c in correctness_by_trace[tid]
    )
    if "prediction_site" not in sample_record:
        raise SystemExit(
            "correctness.json missing 'prediction_site' field — re-run extraction"
        )

    all_steer_layers = sorted({li for _, _, layers in CONDITIONS for li in layers})
    print(f"Loading probes for layers {all_steer_layers}...")
    probes = load_probes(Path(args.probe_dir), all_steer_layers)

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

    # Register one hook per steerable layer up front. Each hook starts with
    # enabled=False; per-condition we set targets and toggle the relevant ones on.
    hooks: dict[int, ProbeSteeringHook] = {}
    for layer in all_steer_layers:
        hooks[layer] = ProbeSteeringHook(
            model, layer=layer, probe=probes[layer], target=0.0
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

        for cond_name, target, steer_layers in CONDITIONS:
            # Reset enabled state for all hooks
            for h in hooks.values():
                h.enabled = False

            if target is None:
                with torch.no_grad():
                    out = model(input_ids, use_cache=False)
            else:
                # Set target on the layers we're steering, then enable them.
                with ExitStack() as stack:
                    for layer in steer_layers:
                        hooks[layer].target = float(target)
                        stack.enter_context(hooks[layer].steering())
                    with torch.no_grad():
                        out = model(input_ids, use_cache=False)

            logits = out.logits[0]
            for case in cr:
                pos = case["prediction_site"]
                option_logits = logits[pos][option_id_tensor].float()
                probs = torch.softmax(option_logits, dim=0).cpu().numpy()
                option_probs = {
                    l: float(probs[i]) for i, l in enumerate(OPTION_LABELS)
                }
                pred = max(option_probs, key=option_probs.get)
                records.append(
                    {
                        "trace_id": t["trace_id"],
                        "case_index": case["case_index"],
                        "gold": case["gold"],
                        "condition": cond_name,
                        "pred": pred,
                        "correct": int(pred == case["gold"]),
                        "confidence": option_probs[pred],
                    }
                )
            del out, logits
            torch.cuda.empty_cache()

        print(f"  trace {t['trace_id']}: done all {len(CONDITIONS)} conditions")

    for h in hooks.values():
        h.remove()

    df = pd.DataFrame(records)

    # Per-condition summary
    summary = (
        df.groupby("condition", sort=False)
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

    # Flip stats vs no_steer baseline
    pivot = df.pivot_table(
        index=["trace_id", "case_index"],
        columns="condition",
        values="correct",
    ).reset_index()

    print("\nFlip stats vs no_steer:")
    flip_rows = []
    for cond_name, _, _ in CONDITIONS:
        if cond_name == "no_steer":
            continue
        if cond_name not in pivot.columns or "no_steer" not in pivot.columns:
            continue
        wrong_to_right = int(
            ((pivot["no_steer"] == 0) & (pivot[cond_name] == 1)).sum()
        )
        right_to_wrong = int(
            ((pivot["no_steer"] == 1) & (pivot[cond_name] == 0)).sum()
        )
        flip_rows.append(
            {
                "condition": cond_name,
                "wrong_to_right": wrong_to_right,
                "right_to_wrong": right_to_wrong,
                "net_flips": wrong_to_right - right_to_wrong,
            }
        )
        print(
            f"  {cond_name}: {wrong_to_right} wrong→right, {right_to_wrong} right→wrong"
        )
    flips_df = pd.DataFrame(flip_rows)

    # Mean confidence shift vs no_steer
    pivot_conf = df.pivot_table(
        index=["trace_id", "case_index"],
        columns="condition",
        values="confidence",
    ).reset_index()
    conf_rows = []
    for cond_name, _, _ in CONDITIONS:
        if cond_name == "no_steer" or cond_name not in pivot_conf.columns:
            continue
        diff = (pivot_conf[cond_name] - pivot_conf["no_steer"]).mean()
        conf_rows.append({"condition": cond_name, "mean_confidence_shift": float(diff)})
    conf_df = pd.DataFrame(conf_rows)

    # Write markdown
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Multi-layer causal steering experiment (v3-1)\n\n")
    lines.append(
        "Extends the v2 single-layer steering experiment by hooking the "
        "within_task_fraction probe at multiple layers simultaneously. The v2 "
        "single-layer null could have been an artifact of downstream-layer "
        "compensation; v3-1 tests whether broader-stack steering punches through.\n\n"
    )
    lines.append(
        "Each condition steers all listed layers using their own trained "
        "within_task_fraction probe direction, pushing every token's readout "
        "to the target value at every steered layer simultaneously.\n\n"
    )
    lines.append("## Per-condition summary\n\n")
    lines.append(summary.to_markdown(index=False) + "\n\n")
    lines.append("## Confidence shift vs no_steer (mean per case)\n\n")
    lines.append(conf_df.to_markdown(index=False) + "\n\n")
    lines.append("## Case-level flips vs no_steer\n\n")
    lines.append(flips_df.to_markdown(index=False) + "\n\n")
    lines.append(
        "## Interpretation\n\n"
        "Compare each early/late condition to no_steer. If the steering effect "
        "scales with the number of steered layers (e.g., L10 alone is null, "
        "L10+L20 has small effect, all-layer has large effect), then the v2 "
        "null was a single-layer compensation artifact and the direction IS "
        "causal — just hard to perturb in isolation.\n\n"
        "If even the full-stack steering produces near-zero effects, the "
        "v2 conclusion stands: the within_task_fraction direction at L10 is a "
        "*readout* of something that drives the calibration gap, not the "
        "*cause* of it. The probe captures a real internal signal but not a "
        "lever.\n"
    )
    out_path.write_text("".join(lines))
    print(f"\nWrote {out_path}")

    df.to_csv(
        out_path.parent / "2026-04-13-v3-multilayer-steering-records.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
