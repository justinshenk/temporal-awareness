"""Head-level ablation experiment for the within_task_fraction probe.

Phase 1: Ranks L20 heads by per-head `att_current_case` correlation with the
L10 within_task_fraction probe readout.
Phase 2: Ablates top/bottom/random heads via o_proj column zeroing.
Phase 3: Measures per-condition accuracy, confidence, and A4-style binned
calibration gap on the v1 test traces.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.ddxplus import OPTION_LABELS
from src.probes.task_position.probes import RidgeProbe

ATTN_LAYER = 20
N_BINS = 5
RANDOM_SEED = 42


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
        default="results/probes/task_position/2026-04-13-v4-head-ablation.md",
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


def per_head_att_current_case(
    attn_all_heads: torch.Tensor, case_boundaries: list[int]
) -> np.ndarray:
    """Compute per-token att_current_case for each head.

    Args:
        attn_all_heads: (n_heads, seq, seq) causal attention weights (CPU float32)
        case_boundaries: sorted list of case start positions

    Returns:
        (n_heads, seq) float32
    """
    n_heads, seq, _ = attn_all_heads.shape
    out = np.zeros((n_heads, seq), dtype=np.float32)
    for t in range(seq):
        idx = bisect.bisect_right(case_boundaries, t) - 1
        start = case_boundaries[idx] if idx >= 0 else 0
        out[:, t] = attn_all_heads[:, t, start : t + 1].sum(dim=1).numpy()
    return out


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-10 or y.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


@contextmanager
def ablate_heads(model, layer: int, heads: list[int], head_dim: int):
    """Temporarily zero the o_proj weight columns for the listed heads."""
    m = model.model.layers[layer].self_attn.o_proj
    original = m.weight.data.clone()
    try:
        with torch.no_grad():
            for h in heads:
                m.weight.data[:, h * head_dim : (h + 1) * head_dim] = 0.0
        yield
    finally:
        with torch.no_grad():
            m.weight.data.copy_(original)


def run_forward(model, input_ids: torch.Tensor, output_attentions: bool = False):
    with torch.no_grad():
        out = model(
            input_ids,
            use_cache=False,
            output_attentions=output_attentions,
            return_dict=True,
        )
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
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    print(f"Gemma config: n_heads={n_heads}, head_dim={head_dim}, hidden={cfg.hidden_size}")

    option_token_ids = get_option_token_ids(tokenizer)
    option_id_tensor = torch.tensor(
        [option_token_ids[l] for l in OPTION_LABELS], device=args.device
    )

    test_traces = [t for t in traces if t["trace_id"] in test_ids]

    # ── Phase 1: Rank heads by per-head att_current_case correlation ──
    print("\nPhase 1: per-head att_current_case correlation at L20")
    per_head_correlations = np.zeros((n_heads, len(test_traces)), dtype=np.float32)
    for ti, t in enumerate(test_traces):
        tokens = t["tokens"]
        boundaries = list(t["case_boundaries"])
        input_ids = torch.tensor([tokens], device=args.device)

        print(f"  trace {t['trace_id']}: forward pass with attentions")
        out = run_forward(model, input_ids, output_attentions=True)
        attn = out.attentions[ATTN_LAYER][0].float().cpu()  # (n_heads, seq, seq)
        features = per_head_att_current_case(attn, boundaries)  # (n_heads, seq)
        l10_resid = t["activations"][10].numpy().astype(np.float32)
        readout = probe.predict(l10_resid)
        for h in range(n_heads):
            per_head_correlations[h, ti] = safe_pearson(features[h], readout)
        del out, attn
        torch.cuda.empty_cache()

    mean_head_r = per_head_correlations.mean(axis=1)
    print("\nPer-head mean correlation (sorted):")
    ranking = sorted(range(n_heads), key=lambda h: -mean_head_r[h])
    for rank, h in enumerate(ranking):
        print(f"  rank {rank:2d}  head {h:2d}  mean r = {mean_head_r[h]:+.3f}")

    top1 = [ranking[0]]
    top3 = ranking[:3]
    top5 = ranking[:5]
    bottom3 = ranking[-3:]
    rng = random.Random(RANDOM_SEED)
    random3 = sorted(rng.sample(range(n_heads), 3))
    print(f"\ntop1={top1} top3={top3} top5={top5} bottom3={bottom3} random3={random3}")

    conditions = [
        ("no_ablation", []),
        ("ablate_top1", top1),
        ("ablate_top3", top3),
        ("ablate_top5", top5),
        ("ablate_bottom3", bottom3),
        ("ablate_random3", random3),
    ]

    # ── Phase 2+3: per-condition forward passes and per-case records ──
    print("\nPhase 2+3: per-condition forward passes")
    records = []
    for t in test_traces:
        tokens = t["tokens"]
        cr = correctness_by_trace.get(str(t["trace_id"]), [])
        if not cr:
            continue
        input_ids = torch.tensor([tokens], device=args.device)

        # Compute L10 readout at each prediction site ONCE (L10 is upstream of L20)
        l10_resid = t["activations"][10].numpy().astype(np.float32)
        case_readouts = {}
        for case in cr:
            pos = case["prediction_site"]
            case_readouts[case["case_index"]] = float(
                probe.predict(l10_resid[pos : pos + 1])[0]
            )

        for cond_name, heads in conditions:
            if not heads:
                out = run_forward(model, input_ids)
            else:
                with ablate_heads(model, ATTN_LAYER, heads, head_dim):
                    out = run_forward(model, input_ids)
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
                        "wf_readout": case_readouts[case["case_index"]],
                    }
                )
            del out, logits
            torch.cuda.empty_cache()
        print(f"  trace {t['trace_id']}: done all {len(conditions)} conditions")

    df = pd.DataFrame(records)

    # ── Phase 4: analysis ──
    print("\nPhase 4: analysis")
    global_summary = (
        df.groupby("condition", sort=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    global_summary["calibration_gap"] = (
        global_summary["mean_confidence"] - global_summary["accuracy"]
    )
    print("\nGlobal summary:")
    print(global_summary.to_string(index=False))

    # A4-style binning: qcut by wf_readout using no_ablation's readouts to
    # define bin edges, then apply those edges to every condition so bins are
    # comparable. Since wf_readout is computed from L10 (upstream of L20), it
    # should be identical across conditions — we confirm this by using the
    # baseline condition's readouts.
    baseline = df[df["condition"] == "no_ablation"].copy()
    baseline["bin"] = pd.qcut(
        baseline["wf_readout"], q=N_BINS, labels=False, duplicates="drop"
    )
    # Map (trace_id, case_index) -> bin
    bin_map = {
        (row["trace_id"], row["case_index"]): int(row["bin"])
        for _, row in baseline.iterrows()
    }
    df["bin"] = df.apply(
        lambda r: bin_map.get((r["trace_id"], r["case_index"]), -1), axis=1
    )
    df_binned = df[df["bin"] >= 0]

    binned = (
        df_binned.groupby(["condition", "bin"], sort=False)
        .agg(
            n=("correct", "size"),
            confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    binned["gap"] = binned["confidence"] - binned["accuracy"]
    print("\nBinned calibration by condition:")
    print(binned.to_string(index=False))

    # Slope of gap vs bin index per condition
    slopes = []
    for cond_name, _ in conditions:
        sub = binned[binned["condition"] == cond_name]
        if len(sub) < 2:
            slopes.append({"condition": cond_name, "gap_slope": float("nan")})
            continue
        xs = sub["bin"].to_numpy(dtype=np.float64)
        ys = sub["gap"].to_numpy(dtype=np.float64)
        slope = float(np.polyfit(xs, ys, deg=1)[0])
        slopes.append({"condition": cond_name, "gap_slope": slope})
    slopes_df = pd.DataFrame(slopes)
    print("\nGap-vs-bin-index slope (positive = monotone fatigue pattern survives):")
    print(slopes_df.to_string(index=False))

    # Flip stats vs no_ablation
    pivot = df.pivot_table(
        index=["trace_id", "case_index"], columns="condition", values="correct"
    ).reset_index()
    flip_rows = []
    for cond_name, _ in conditions:
        if cond_name == "no_ablation" or cond_name not in pivot.columns:
            continue
        wr = int(((pivot["no_ablation"] == 0) & (pivot[cond_name] == 1)).sum())
        rw = int(((pivot["no_ablation"] == 1) & (pivot[cond_name] == 0)).sum())
        flip_rows.append(
            {
                "condition": cond_name,
                "wrong_to_right": wr,
                "right_to_wrong": rw,
                "net_flips": wr - rw,
            }
        )
    flips_df = pd.DataFrame(flip_rows)
    print("\nFlip stats vs no_ablation:")
    print(flips_df.to_string(index=False))

    # ── Write markdown ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    head_rank_df = pd.DataFrame(
        {
            "rank": list(range(n_heads)),
            "head": ranking,
            "mean_correlation": [float(mean_head_r[h]) for h in ranking],
        }
    )

    lines = []
    lines.append("# Head-level ablation experiment (v4)\n\n")
    lines.append(
        "Tests whether the L20 heads that drive `att_current_case` are causally "
        "upstream of the v1 calibration gap. Phase 1 ranks the 16 L20 heads by "
        "their per-head `att_current_case` correlation with the L10 "
        "within_task_fraction probe readout. Phase 2 ablates top/bottom/random "
        "heads by zeroing the corresponding input columns of `o_proj.weight`. "
        "Phase 3 measures per-case accuracy and confidence under each ablation. "
        "Phase 4 compares the global calibration gap and the A4-style binned gap "
        "pattern across conditions.\n\n"
        "A flattening of the gap-vs-bin slope under ablation is strong evidence "
        "that the ablated heads are causally upstream of the A4 pattern. A "
        "null result (slope unchanged) means the ablation removed the probe-"
        "predictive feature but not the calibration gap — meaning the gap is "
        "driven by something else entirely.\n\n"
    )
    lines.append("## Phase 1: per-head L20 ranking\n\n")
    lines.append(head_rank_df.to_markdown(index=False) + "\n\n")
    lines.append(
        f"Ablation sets: top1={top1}, top3={top3}, top5={top5}, bottom3={bottom3}, random3={random3}\n\n"
    )
    lines.append("## Global summary per condition\n\n")
    lines.append(global_summary.to_markdown(index=False) + "\n\n")
    lines.append("## Gap-vs-bin slope per condition\n\n")
    lines.append(slopes_df.to_markdown(index=False) + "\n\n")
    lines.append(
        "Baseline slope is the expected A4 monotone pattern on this test set. "
        "Ablation conditions that drop the slope toward 0 have attacked the "
        "mechanism producing the pattern.\n\n"
    )
    lines.append("## A4-style binned calibration by condition\n\n")
    lines.append(binned.to_markdown(index=False) + "\n\n")
    lines.append("## Flip stats vs no_ablation\n\n")
    lines.append(flips_df.to_markdown(index=False) + "\n\n")
    lines.append(
        "## Interpretation\n\n"
        "Read this in order:\n"
        "1. Does the global calibration gap change under ablation? If no change "
        "at top5 either, the heads are not carrying the gap signal.\n"
        "2. Does the gap-vs-bin slope flatten under ablation? If yes (and not "
        "for bottom3/random3), the ablation targets are causally upstream of "
        "the A4 pattern. If no, the A4 pattern survives the ablation and is "
        "driven by something the ablation didn't touch.\n"
        "3. Do the targeted ablations (top1/3/5) differ meaningfully from the "
        "random3 control? If they behave identically, the effect is non-"
        "specific.\n"
    )
    out_path.write_text("".join(lines))
    print(f"\nWrote {out_path}")

    df.to_csv(
        out_path.parent / "2026-04-13-v4-head-ablation-records.csv", index=False
    )


if __name__ == "__main__":
    main()
