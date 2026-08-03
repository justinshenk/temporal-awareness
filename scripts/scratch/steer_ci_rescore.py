#!/usr/bin/env python
"""Stage 4: re-score the best steering cell with per-prompt logging.

For one model: load the stored CAA + control vectors (local HF mirror), rebuild
the same 20 held-out eval prompts in both label orders, score baseline / best-cell
steering / matched-norm random control, and write the per-prompt logp differences
next to the stored means so the sanity gate can compare them.

The gate is the point: if the recomputed S, S_ctrl and baseline do not reproduce
the stored campaign values, the per-prompt numbers describe a different run and
must not be used for confidence intervals.

Usage:
    uv run python scripts/scratch/steer_ci_rescore.py \
        --model mistralai/Mistral-7B-Instruct-v0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "intertemporal"))

import steer_turn_preference as stp  # noqa: E402
from src.inference import ModelBackend, ModelRunner  # noqa: E402
from src.inference.interventions import Intervention, InterventionTarget  # noqa: E402

MIRROR = REPO / "out" / "hf_new" / "steering" / "extreme_sweep"
OUT_BASE = REPO / "out" / "steering_ci"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    short = args.model.split("/")[-1]
    mirror_dir = MIRROR / short
    out_dir = OUT_BASE / short
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.load(open(mirror_dir / "steering_summary.json"))
    assert summary["model"] == args.model, (summary["model"], args.model)
    best = summary["best"]
    layer, alpha = int(best["layer"]), float(best["alpha"])

    caa = np.load(mirror_dir / "caa_vectors.npz")
    ctrl = np.load(mirror_dir / "control_vectors.npz")
    steer_vec = caa[f"layer_{layer}"]
    ctrl_vec = ctrl[f"layer_{layer}"]
    for name, vec in (("caa", steer_vec), ("control", ctrl_vec)):
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, f"{name} vector not unit norm: {norm}"

    eval_pairs = sorted(stp.load_pairs(stp.EVAL_DATA), key=lambda p: p["id"])
    eval_pairs = eval_pairs[: summary["n_eval_pairs"]]
    assert [p["id"] for p in eval_pairs] == summary["eval_pair_ids"], "eval ids differ"

    # Match the stored campaign runs exactly: TransformerLens, no weight
    # processing, bfloat16.
    runner = ModelRunner(
        args.model,
        backend=ModelBackend.TRANSFORMERLENS,
        process_weights=False,
        dtype=torch.bfloat16,
    )
    assert runner.n_layers == summary["n_layers"], (runner.n_layers, summary["n_layers"])
    assert runner.d_model == steer_vec.shape[0]

    items = stp.build_eval_items(runner, eval_pairs)

    def make_intervention(vec: np.ndarray) -> Intervention:
        return Intervention(
            layer=layer,
            mode="add",
            values=(alpha * vec).astype(np.float32),
            target=InterventionTarget.all(),
            component="resid_post",
        )

    print(f"Scoring baseline for {short}...")
    baseline = stp.score_items(runner, items, None, args.batch_size)
    print(f"Scoring steering L{layer} alpha={alpha:g}...")
    steer = stp.score_items(runner, items, make_intervention(steer_vec), args.batch_size)
    print("Scoring matched-norm random control...")
    control = stp.score_items(runner, items, make_intervention(ctrl_vec), args.batch_size)

    stored_best_row = next(
        r for r in summary["rows"] if r["layer"] == layer and float(r["alpha"]) == alpha
    )
    result = {
        "model": args.model,
        "backend": "transformerlens",
        "process_weights": False,
        "dtype": str(runner.dtype),
        "device": runner.device,
        "layer": layer,
        "alpha": alpha,
        "n_eval_pairs": len(eval_pairs),
        "eval_pair_ids": [p["id"] for p in eval_pairs],
        "recomputed": {
            "baseline": {
                "S": baseline.S, "long_A": baseline.long_A, "long_B": baseline.long_B,
                "per_prompt": baseline.per_prompt,
            },
            "steer": {
                "S": steer.S, "long_A": steer.long_A, "long_B": steer.long_B,
                "per_prompt": steer.per_prompt,
            },
            "control": {
                "S": control.S, "long_A": control.long_A, "long_B": control.long_B,
                "per_prompt": control.per_prompt,
            },
        },
        "stored": {"baseline": summary["baseline"], "best_row": stored_best_row},
        "gate": {
            "delta_S": steer.S - stored_best_row["S"],
            "delta_S_ctrl": control.S - stored_best_row["S_ctrl"],
            "delta_baseline": baseline.S - summary["baseline"]["S"],
        },
    }
    out_path = out_dir / "rescore.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print(
        f"GATE {short}: S {steer.S:.4f} (stored {stored_best_row['S']:.4f}) | "
        f"S_ctrl {control.S:.4f} (stored {stored_best_row['S_ctrl']:.4f}) | "
        f"baseline {baseline.S:.4f} (stored {summary['baseline']['S']:.4f})"
    )

    runner.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
