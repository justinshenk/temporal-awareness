"""Plot recovery vs the fraction of decode steps patched by the L20 oracle — one or more tasks.

Reads ``temporal_oracle_L{L}.json`` (and its per-task siblings). The periodic(k) gates trace a
recovery-vs-fraction-patched curve (k=1 is the full oracle at frac=1.0); the structural gates
(result_only / planning_only / reasoning_only / step_boundary) are plotted as labelled points. A
planning-side point sitting near the full oracle at a small fraction — with the answer-side point
down near base — locates the transported trajectory state in the structural tokens rather than in
the arithmetic (the E1b prediction).

Passing several JSONs overlays them, which is how the paper's F1 compares the two procedures: the
knee is the claim, and two tasks falling off the same cliff is the evidence that it is not
arithmetic-specific. Every point carries a percentile-bootstrap 95% interval over problems
(``src/common/bootstrap_stats``) — a bare 0.00 is a point estimate, not a bounded null.

    # single task (unchanged)
    uv run python -m scripts.attribution.plot_temporal_oracle \
        --json results/attribution/temporal_oracle_L20.json

    # F1: both procedures overlaid
    uv run python -m scripts.attribution.plot_temporal_oracle \
        --json results/attribution/temporal_oracle_L20.json,\
results/attribution/temporal_oracle_multihop_L20.json \
        --labels GSM8K,MuSiQue --out papers/register_vs_procedure/figures/f1_temporal_density.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.bootstrap_stats import bootstrap_interval

# Categorical slots 1 and 2 of the reference palette, in fixed order (blue, orange) — a pair
# validated for CVD separation. Assigned by task identity, never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

# Structural gates are named per task (GSM8K plans then computes; multi-hop reasons then answers).
# Marker shape carries the *role* so the reader can compare across tasks without reading colour.
STRUCTURAL_MARKERS = {
    "planning_only": ("^", "planning only"),
    "reasoning_only": ("^", "reasoning only"),
    "result_only": ("v", "result only"),
    "answer_only": ("v", "answer only"),
    "step_boundary": ("s", "step boundary"),
}


def gate_interval(gate: dict) -> tuple[float, float, float]:
    """(estimate, lo, hi) for one gate, bootstrapped over problems.

    Falls back to the recorded point estimate with a zero-width interval when the run predates
    ``per_problem`` — better a visibly absent interval than a fabricated one.
    """
    per = gate.get("per_problem")
    if not per:
        r = float(gate["recovery"])
        return r, r, r
    iv = bootstrap_interval(np.asarray(per, dtype=float), np.mean)
    return iv.estimate, iv.lo, iv.hi


def periodic_series(gates: dict) -> tuple[list, list, list, list]:
    """The periodic(k) curve, ordered by fraction patched, with asymmetric error bars."""
    items = sorted(((g, v) for g, v in gates.items() if g.startswith("periodic_")),
                   key=lambda kv: kv[1]["frac_patched"])
    xs, ys, los, his = [], [], [], []
    for _, v in items:
        est, lo, hi = gate_interval(v)
        xs.append(v["frac_patched"])
        ys.append(est)
        los.append(max(0.0, est - lo))
        his.append(max(0.0, hi - est))
    return xs, ys, los, his


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, help="comma-separated list; several overlay")
    ap.add_argument("--labels", default=None, help="comma-separated series labels (default: task name)")
    ap.add_argument("--structural", action="store_true", default=True,
                    help="also plot the structural gates as points")
    ap.add_argument("--no-structural", dest="structural", action="store_false")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = [Path(p) for p in args.json.split(",")]
    datasets = [json.loads(p.read_text()) for p in paths]
    labels = (args.labels.split(",") if args.labels
              else [d.get("task", "gsm8k") for d in datasets])
    if len(labels) != len(datasets):
        raise SystemExit(f"{len(labels)} labels for {len(datasets)} json files")

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for i, (data, label) in enumerate(zip(datasets, labels)):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        gates = data["gates"]
        xs, ys, los, his = periodic_series(gates)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o-", color=color, lw=2.0, ms=8,
                    capsize=3, elinewidth=1.2, zorder=3,
                    label=f"{label} — periodic(k), n={data['n_contrast']}")

        if not args.structural:
            continue
        # Structural gates carry their value in the LEGEND, not in inline text. Several of them
        # land on top of each other at (low fraction, 0.00), where inline labels overprint into
        # an unreadable smear — the legend costs one line each and always resolves.
        for name, (marker, pretty) in STRUCTURAL_MARKERS.items():
            if name not in gates:
                continue
            v = gates[name]
            est, lo, hi = gate_interval(v)
            ax.errorbar([v["frac_patched"]], [est], yerr=[[max(0.0, est - lo)], [max(0.0, hi - est)]],
                        fmt=marker, color=color, ms=9, capsize=3, elinewidth=1.2,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=4,
                        label=f"{label} — {pretty} ({est:.2f} @ {v['frac_patched']:.0%})")

    # k labels come from the first series only: the fractions coincide across tasks by
    # construction (the gate is defined on step index), so labelling both would just overprint.
    # Offsets alternate because k>=4 crowds into x<0.25 where same-height labels would collide.
    periodic_first = sorted(((g, v) for g, v in datasets[0]["gates"].items()
                             if g.startswith("periodic_")),
                            key=lambda kv: kv[1]["frac_patched"])
    for j, (name, v) in enumerate(periodic_first):
        ax.annotate(name.replace("periodic_", "k="), (v["frac_patched"], v["recovery"]),
                    textcoords="offset points", xytext=(0, 13 if j % 2 else -17), ha="center",
                    fontsize=7.5, color=MUTED)

    ax.set_xlabel(f"fraction of decode steps patched with the full-δ L{datasets[0]['layer']} oracle",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("recovery on the contrast set", fontsize=9.5, color=INK)
    ax.set_xlim(-0.06, 1.10)
    ax.set_ylim(-0.14, 1.06)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.legend(loc="upper left", fontsize=7.8, framealpha=0.95, edgecolor=GRID,
              handletextpad=0.5, labelspacing=0.35, borderpad=0.6)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path("results/figures") / f"temporal_oracle_L{datasets[0]['layer']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
