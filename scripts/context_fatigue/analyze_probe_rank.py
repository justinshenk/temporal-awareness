"""Rank of the linear mode code at one layer, by iterative direction removal.

Probe 2 (``analyze_format_probes.py``) reads the upcoming reply's compliance from the final
pre-generation state. This script asks how many linear directions that code spans: fit the
probe at one stack layer, record its LOO-CV AUC, backproject the full-data LDA weights into
activation space (exactly as the direction export does), project that direction out of every
state, and refit. AUC falling to chance after k removals = the code has linear rank ≈ k.

Same pipeline, data path, and labels as Probe 2 — StandardScaler → PCA → LDA, LOO-CV at the
episode level, gsm8k mixed cells joined to the erosion run's ``turns.csv``. CPU-only; runs on
the committed ``e6_format_probes/`` captures.

    .venv/bin/python scripts/context_fatigue/analyze_probe_rank.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_format_probes import lda_pipeline, load_depth, loo_auc


def load_probe2_layer(capture_dir: Path, turns_csv: Path, depths, stack_layer: int):
    turns = pd.read_csv(turns_csv)[["depth", "probe", "fully_compliant"]]
    xs, ys = [], []
    for depth in depths:
        states, rows = load_depth(capture_dir, "gsm8k", depth)
        rows = rows.merge(turns[turns.depth == depth], on=["depth", "probe"], how="left")
        keep = rows.fully_compliant.notna().values
        xs.append(states[keep][:, stack_layer])
        ys.append(rows.fully_compliant[keep].astype(int).values)
    return np.concatenate(xs), np.concatenate(ys)


def full_fit_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pipe = lda_pipeline(len(y), x.shape[1]).fit(x, y)
    scaler, pca, lda = pipe.named_steps.values()
    w = (lda.coef_ @ pca.components_).ravel() / scaler.scale_
    return w / np.linalg.norm(w)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", default="results/context_fatigue/e6_format_probes")
    p.add_argument("--gsm8k-turns", default="results/context_fatigue/e6_gsm8k/turns.csv")
    p.add_argument("--probe2-depths", type=int, nargs="+", default=[12, 15])
    p.add_argument("--stack-layer", type=int, default=21)
    p.add_argument("--rounds", type=int, default=3,
                   help="number of AUCs to report: before removal, then after each projection")
    args = p.parse_args()
    capture_dir = Path(args.capture_dir)

    x, y = load_probe2_layer(capture_dir, Path(args.gsm8k_turns), args.probe2_depths,
                             args.stack_layer)
    print(f"Probe-2 rank at stack L{args.stack_layer}: n={len(y)} "
          f"({int(y.sum())} compliant), depths {args.probe2_depths}")

    aucs = []
    for r in range(args.rounds):
        auc = loo_auc(x, y)
        aucs.append(float(auc))
        print(f"  after {r} direction(s) removed: LOO-AUC = {auc:.3f}")
        if r < args.rounds - 1:
            u = full_fit_direction(x, y)
            x = x - np.outer(x @ u, u)

    dest = capture_dir / f"probe_rank_L{args.stack_layer}.json"
    dest.write_text(json.dumps({
        "stack_layer": args.stack_layer, "depths": args.probe2_depths,
        "n": int(len(y)), "n_compliant": int(y.sum()),
        "loo_auc_by_directions_removed": aucs}, indent=1))
    print(f"Saved to {dest}")


if __name__ == "__main__":
    main()
