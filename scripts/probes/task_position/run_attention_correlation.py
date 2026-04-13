"""Hunt the upstream variable U for the within_task_fraction probe readout.

Captures per-token attention summary features from Gemma-9B-IT at L10 and L20
on the v1 test traces, then correlates each feature with the trained
within_task_fraction probe readout. Identifies the strongest external
correlate of the position-belief direction, which is a candidate for U.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.task_position.probes import RidgeProbe

ATTN_LAYERS = [10, 20]


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
        "--split",
        default="results/probes/task_position/gemma-9b-it/probes/split.json",
    )
    p.add_argument(
        "--out",
        default="results/probes/task_position/2026-04-13-v3-attention-correlation.md",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def compute_attention_features(
    attn_seq: torch.Tensor,
    case_boundaries: list[int],
) -> dict[str, np.ndarray]:
    """Compute per-token attention summary features from one layer's attention.

    Args:
        attn_seq: tensor of shape (n_heads, seq, seq) on cpu, float32
        case_boundaries: sorted list of case start tokens

    Returns:
        dict mapping feature name -> (seq,) array
    """
    # Average over heads
    a = attn_seq.mean(dim=0)  # (seq, seq)
    seq = a.shape[0]

    att_bos = a[:, 0].numpy()
    upper = min(50, seq)
    att_first_50 = a[:, :upper].sum(dim=1).numpy()

    att_current_case = np.zeros(seq, dtype=np.float32)
    for t in range(seq):
        idx = bisect.bisect_right(case_boundaries, t) - 1
        if idx < 0:
            start = 0
        else:
            start = case_boundaries[idx]
        att_current_case[t] = a[t, start : t + 1].sum().item()

    win = 100
    att_recent = np.zeros(seq, dtype=np.float32)
    for t in range(seq):
        start = max(0, t - win)
        att_recent[t] = a[t, start : t + 1].sum().item()

    # Causal attention entropy: attention is masked (zeros above diagonal).
    # Add a tiny epsilon to avoid log(0) on the masked positions; they
    # contribute 0 because attn_value * log(attn_value+eps) ≈ 0 when attn is 0.
    eps = 1e-12
    a_eps = a + eps
    ent = -(a * torch.log(a_eps)).sum(dim=1).numpy()

    return {
        "att_bos": att_bos,
        "att_first_50": att_first_50,
        "att_current_case": att_current_case,
        "att_recent_window_100": att_recent,
        "att_entropy": ent,
    }


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-10 or y.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    args = parse_args()

    print(f"Loading {args.activations}...")
    blob = torch.load(args.activations, weights_only=False)
    traces = blob["traces"]

    with open(args.split) as f:
        split = json.load(f)
    test_ids = set(int(x) for x in split["test_ids"])

    print(f"Loading probe {args.probe}...")
    probe = RidgeProbe.load(args.probe)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",  # eager attn so output_attentions works
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rows = []
    for t in traces:
        if t["trace_id"] not in test_ids:
            continue
        tokens = t["tokens"]
        boundaries = list(t["case_boundaries"])
        n_tokens = len(tokens)
        input_ids = torch.tensor([tokens], device=args.device)

        print(f"\nTrace {t['trace_id']}: forward pass with attentions ({n_tokens} tokens)")
        with torch.no_grad():
            out = model(
                input_ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )

        l10_resid = t["activations"][10].numpy().astype(np.float32)
        readout = probe.predict(l10_resid)

        for layer in ATTN_LAYERS:
            attn = out.attentions[layer][0].float().cpu()  # (n_heads, seq, seq)
            features = compute_attention_features(attn, boundaries)
            for name, arr in features.items():
                rows.append(
                    {
                        "trace_id": t["trace_id"],
                        "attn_layer": layer,
                        "feature": name,
                        "pearson_r": pearson_safe(arr, readout),
                    }
                )
            del attn
            torch.cuda.empty_cache()

        del out, input_ids
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    # Aggregate: mean and abs-mean per (layer, feature) across traces
    summary = (
        df.groupby(["attn_layer", "feature"])
        .agg(
            mean_r=("pearson_r", "mean"),
            mean_abs_r=("pearson_r", lambda s: float(np.abs(s).mean())),
            min_r=("pearson_r", "min"),
            max_r=("pearson_r", "max"),
        )
        .reset_index()
        .sort_values("mean_abs_r", ascending=False)
    )

    print("\nPer-trace correlations:")
    print(df.to_string(index=False))
    print("\nSummary (sorted by mean |r|):")
    print(summary.to_string(index=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Hunting the upstream variable U (v3-5)\n\n")
    lines.append(
        "v1 trained a within_task_fraction probe at Gemma-9B-IT L10 with R² 0.954. "
        "v2 and v3-1 found that steering this direction (single-layer or multi-layer) "
        "has no causal effect on accuracy, confidence, or the calibration gap. The "
        "direction is a *readout* of some upstream variable U, not a lever. v3-5 "
        "scans candidate Us by capturing per-token attention features at L10 and L20 "
        "and correlating them with the probe readout across the v1 test traces.\n\n"
        "Candidate features:\n"
        "- `att_bos`: attention to position 0 (BOS / known sink)\n"
        "- `att_first_50`: sum of attention to positions [0, 50] (chat header region)\n"
        "- `att_current_case`: attention to the current case content (boundary..t)\n"
        "- `att_recent_window_100`: attention to the last 100 tokens (recency)\n"
        "- `att_entropy`: Shannon entropy of the token's attention distribution\n\n"
    )
    lines.append("## Per-trace correlations\n\n")
    lines.append(df.to_markdown(index=False) + "\n\n")
    lines.append("## Summary (sorted by mean |Pearson r| across traces)\n\n")
    lines.append(summary.to_markdown(index=False) + "\n\n")
    lines.append(
        "## Interpretation\n\n"
        "The strongest correlate (top of the summary table) is the leading "
        "candidate for U. A correlation > 0.5 indicates a substantial linear "
        "relationship between that attention feature and the position-belief "
        "readout. A correlation > 0.8 means the attention feature contains most "
        "of the same information as the linear readout, just expressed in a "
        "different basis.\n\n"
        "Note that strong correlation here is necessary but not sufficient for "
        "causation. To upgrade a candidate to a real cause, the next step would "
        "be a head-level ablation: knock out the heads contributing most to "
        "that attention feature, and see if the calibration gap collapses.\n"
    )
    out_path.write_text("".join(lines))
    print(f"\nWrote {out_path}")

    df.to_csv(
        out_path.parent / "2026-04-13-v3-attention-correlation-records.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
