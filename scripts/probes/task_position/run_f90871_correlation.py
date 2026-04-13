"""Correlate F90871 SAE feature with the tokens_until_boundary probe readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from src.probes.task_position.probes import RidgeProbe
from src.probes.task_position.sae_features import (
    F90871_INDEX,
    encode_features,
    load_gemma_scope_l20_sae,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--activations",
        default="results/probes/task_position/gemma-9b-it/activations.pt",
    )
    p.add_argument(
        "--probe",
        default="results/probes/task_position/gemma-9b-it/probes/tokens_until_boundary_L10.pkl",
    )
    p.add_argument(
        "--split",
        default="results/probes/task_position/gemma-9b-it/probes/split.json",
    )
    p.add_argument(
        "--out",
        default="results/probes/task_position/2026-04-13-v2-f90871-correlation.md",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.activations}...")
    blob = torch.load(args.activations, weights_only=False)
    traces = blob["traces"]
    layers = blob["layers"]
    if 10 not in layers or 20 not in layers:
        raise SystemExit(f"Need layers 10 and 20 in saved activations; got {layers}")

    with open(args.split) as f:
        split = json.load(f)
    test_ids = set(int(x) for x in split["test_ids"])

    print(f"Loading probe {args.probe}...")
    probe = RidgeProbe.load(args.probe)

    print("Loading Gemma-Scope L20 SAE (gemma-scope-9b-it-res)...")
    sae = load_gemma_scope_l20_sae(device=args.device)

    rows = []
    for t in traces:
        if t["trace_id"] not in test_ids:
            continue
        l10 = t["activations"][10]
        l20 = t["activations"][20]
        n_tokens = l10.shape[0]

        readout = probe.predict(l10.numpy().astype(np.float32))
        f_activations = encode_features(
            sae, l20, feature_indices=[F90871_INDEX], device=args.device
        )[:, 0]

        full_r, full_p = pearsonr(f_activations, readout)
        half = n_tokens // 2
        early_r, _ = pearsonr(f_activations[:half], readout[:half])
        late_r, _ = pearsonr(f_activations[half:], readout[half:])

        rows.append(
            {
                "trace_id": t["trace_id"],
                "n_tokens": int(n_tokens),
                "pearson_full": float(full_r),
                "pearson_p": float(full_p),
                "pearson_early_half": float(early_r),
                "pearson_late_half": float(late_r),
                "decay": float(early_r - late_r),
                "f90871_mean": float(f_activations.mean()),
                "f90871_std": float(f_activations.std()),
            }
        )

    df = pd.DataFrame(rows)
    print("\nPer-trace results:")
    print(df.to_string(index=False))

    summary = {
        "mean_pearson_full": float(df["pearson_full"].mean()),
        "mean_pearson_early": float(df["pearson_early_half"].mean()),
        "mean_pearson_late": float(df["pearson_late_half"].mean()),
        "mean_decay": float(df["decay"].mean()),
        "n_test_traces": int(len(df)),
    }
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# F90871 ↔ tokens_until_boundary correlation (v2 stretch)\n\n")
    lines.append(
        "Tests whether the Gemma-Scope SAE feature F90871 (a 'document-boundary "
        "detector' identified in the existing context-fatigue writeup) is the "
        "mechanistic correlate of the trained `tokens_until_boundary` probe at L10. "
        "F90871 is computed by applying the SAE encoder to saved L20 residual stream "
        "activations from v1; the probe readout is computed from saved L10 residuals.\n\n"
    )
    lines.append("## Per-trace correlations (test split, n=4 traces)\n\n")
    lines.append(df.to_markdown(index=False) + "\n\n")
    lines.append("## Summary\n\n")
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v:.4f}\n")
    lines.append(
        "\n## Interpretation\n\n"
        "A positive `pearson_full` means F90871 activation tracks the probe readout "
        "(higher F90871 ↔ further from boundary, OR vice-versa depending on the "
        "probe sign). A positive `decay` (early > late) supports the writeup's "
        "claim that F90871 is suppressed as context accumulates: the boundary "
        "detector loses its signal late in the trace.\n"
    )
    out.write_text("".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
