"""Plot Phase 1: GSM8K recovery vs the fraction of decode steps patched by the L20 oracle.

Reads ``temporal_oracle_L{L}.json``. The periodic(k) gates trace a recovery-vs-fraction-patched
curve (k=1 is the full oracle at frac=1.0); the structural gates (result_only / planning_only /
step_boundary) are plotted as labelled points. A planning_only point sitting near the full oracle at
a small fraction — with result_only down near base — locates the transported trajectory state in the
structural/planning tokens, not the arithmetic (the E1b prediction).

    uv run python -m scripts.attribution.plot_temporal_oracle \
        --json results/attribution/temporal_oracle_L20.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    L = data["layer"]
    gates = data["gates"]

    periodic = sorted(((g, v) for g, v in gates.items() if g.startswith("periodic_")),
                      key=lambda kv: kv[1]["frac_patched"])
    px = [v["frac_patched"] for _, v in periodic]
    py = [v["recovery"] for _, v in periodic]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(px, py, "o-", color="#2c3e50", lw=2.0, ms=7, label="periodic(k) gate", zorder=2)
    for (name, v) in periodic:
        ax.annotate(name.replace("periodic_", "k="), (v["frac_patched"], v["recovery"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5,
                    color="#2c3e50")

    colors = {"result_only": "#c0392b", "planning_only": "#27ae60", "step_boundary": "#8e44ad"}
    for name, color in colors.items():
        if name not in gates:
            continue
        v = gates[name]
        ax.scatter([v["frac_patched"]], [v["recovery"]], color=color, s=90, zorder=4,
                   edgecolor="white", linewidth=1.2, label=name)
        ax.annotate(f"{name}\n({v['recovery']:.2f})", (v["frac_patched"], v["recovery"]),
                    textcoords="offset points", xytext=(8, -4), ha="left", fontsize=8, color=color)

    ax.set_xlabel("fraction of decode steps patched with the full-δ L%d oracle" % L)
    ax.set_ylabel(f"GSM8K recovery on the contrast set (n={data['n_contrast']})")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(f"Temporal decomposition of the L{L} oracle: the trajectory state is DENSE\n"
                 f"only ≥94%-step patching recovers (planning_only); every sparser gate ⇒ 0",
                 fontsize=10.5)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path("results/figures") / f"temporal_oracle_L{L}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
