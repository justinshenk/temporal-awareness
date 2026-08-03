#!/usr/bin/env python
"""Stage 4: paired bootstrap CIs from the per-prompt steering re-scores.

Reads out/steering_ci/<model>/rescore.json and computes 10k-resample bootstrap
intervals for S_steer, S_ctrl and the paired difference S_steer - S_ctrl.
Resampling is over eval prompts within each label order, and the same resampled
prompt indices are used for every condition, so the steer-minus-control contrast
stays paired. The counterbalanced definition S = (mean(long_A) + mean(long_B))/2
is preserved in every resample.

CPU only. Run after steer_ci_rescore.py has written every model you want.

Usage:
    uv run python scripts/scratch/steer_ci_bootstrap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "out" / "steering_ci"
N_BOOT = 10_000
SEED = 0

MODELS = [
    "Qwen3-4B-Instruct-2507",
    "Llama-3.1-8B-Instruct",
    "gemma-2-9b-it",
    "Mistral-7B-Instruct-v0.3",
]


def s_of(per_prompt: dict[str, list[float]], idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    a = np.asarray(per_prompt["long_A"])[idx_a]
    b = np.asarray(per_prompt["long_B"])[idx_b]
    return (a.mean() + b.mean()) / 2.0


def main() -> int:
    rng = np.random.default_rng(SEED)
    out = {}
    for model in MODELS:
        path = BASE / model / "rescore.json"
        if not path.exists():
            print(f"SKIP {model}: {path} missing")
            continue
        r = json.load(open(path))
        rec = r["recomputed"]
        n_a = len(rec["steer"]["per_prompt"]["long_A"])
        n_b = len(rec["steer"]["per_prompt"]["long_B"])

        # The stored mean must come from these very lists, or the CI describes
        # something other than the reported point estimate.
        for cond in ("baseline", "steer", "control"):
            pp = rec[cond]["per_prompt"]
            s = (np.mean(pp["long_A"]) + np.mean(pp["long_B"])) / 2.0
            assert abs(s - rec[cond]["S"]) < 1e-9, (model, cond, s, rec[cond]["S"])

        stats = {"S_steer": [], "S_ctrl": [], "S_base": [], "diff_steer_ctrl": []}
        for _ in range(N_BOOT):
            ia = rng.integers(0, n_a, n_a)
            ib = rng.integers(0, n_b, n_b)
            s_steer = s_of(rec["steer"]["per_prompt"], ia, ib)
            s_ctrl = s_of(rec["control"]["per_prompt"], ia, ib)
            s_base = s_of(rec["baseline"]["per_prompt"], ia, ib)
            stats["S_steer"].append(s_steer)
            stats["S_ctrl"].append(s_ctrl)
            stats["S_base"].append(s_base)
            stats["diff_steer_ctrl"].append(s_steer - s_ctrl)

        entry = {
            "n_prompts_per_order": {"long_A": n_a, "long_B": n_b},
            "n_boot": N_BOOT,
            "seed": SEED,
            "layer": r["layer"],
            "alpha": r["alpha"],
            "point": {
                "S_steer": rec["steer"]["S"],
                "S_ctrl": rec["control"]["S"],
                "S_base": rec["baseline"]["S"],
                "diff_steer_ctrl": rec["steer"]["S"] - rec["control"]["S"],
            },
            "ci95": {
                k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                for k, v in stats.items()
            },
            "p_diff_gt_0": float(np.mean(np.asarray(stats["diff_steer_ctrl"]) > 0)),
            "gate_vs_stored": r["gate"],
        }
        out[model] = entry
        print(
            f"{model}: S={entry['point']['S_steer']:.3f} "
            f"CI[{entry['ci95']['S_steer'][0]:.3f},{entry['ci95']['S_steer'][1]:.3f}] | "
            f"ctrl={entry['point']['S_ctrl']:.3f} "
            f"CI[{entry['ci95']['S_ctrl'][0]:.3f},{entry['ci95']['S_ctrl'][1]:.3f}] | "
            f"diff CI[{entry['ci95']['diff_steer_ctrl'][0]:.3f},"
            f"{entry['ci95']['diff_steer_ctrl'][1]:.3f}] "
            f"P(diff>0)={entry['p_diff_gt_0']:.4f}"
        )

    BASE.mkdir(parents=True, exist_ok=True)
    out_path = BASE / "steering_bootstrap_cis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
