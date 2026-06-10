"""Plot DAS-R vs PCA-top-r recovery at matched rank: can task-loss find what variance can't?

Reads ``das_subspace_L{L}.json`` and overlays the task-loss-trained subspace's recovery against the
variance-selected (PCA) band at each rank. A gap (DAS above PCA) means the capability directions are
low-rank but low-variance — found only by behavioral search.

    uv run python -m scripts.attribution.plot_das_subspace --json results/attribution/das_subspace_L20.json
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
    items = sorted(((int(r), v) for r, v in data["ranks"].items()), key=lambda kv: kv[0])
    ranks = [r for r, _ in items]
    das = [v["das_recovery"] for _, v in items]
    pca = [v.get("pca_recovery") for _, v in items]
    ce = [v.get("final_ce") for _, v in items]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ranks, das, "o-", color="#8e44ad", lw=2.4, ms=9, label="DAS-R recovery (task-loss subspace)", zorder=3)
    ax.plot(ranks, [p if p is not None else float("nan") for p in pca], "s--", color="#c0392b",
            lw=2.0, ms=7, label="PCA top-r recovery (variance subspace)", zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("subspace rank r")
    ax.set_ylabel("closed-loop oracle recovery (contrast acc)")
    ax.set_ylim(-0.03, 1.0)
    ax.grid(True, which="both", alpha=0.25)

    for r, d_, p_ in zip(ranks, das, pca):
        ax.annotate(f"{d_:.2f}", (r, d_), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8, color="#8e44ad")
        if p_ is not None:
            ax.annotate(f"{p_:.2f}", (r, p_), textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=8, color="#c0392b")

    # CE on a twin axis: the dissociation — DAS drives teacher-forced CE → 0 yet recovers nothing.
    ax2 = ax.twinx()
    ax2.plot(ranks, ce, "^:", color="#16a085", lw=1.6, ms=7, alpha=0.8,
             label="DAS teacher-forced CE (↓ = better fit)")
    ax2.set_ylabel("DAS teacher-forced CE", color="#16a085")
    ax2.tick_params(axis="y", labelcolor="#16a085")
    ax2.set_ylim(0, max(c for c in ce if c is not None) * 1.1)
    for r, c_ in zip(ranks, ce):
        ax2.annotate(f"{c_:.2f}", (r, c_), textcoords="offset points", xytext=(6, 2), ha="left",
                     fontsize=7.5, color="#16a085")

    lines, labs = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labs + lab2, loc="center left", fontsize=9, framealpha=0.95)
    ax.set_title(f"Task-loss subspace search does NOT beat variance (L{L}, oracle injection)\n"
                 f"DAS recovers 0 at every rank — even at r=512/CE=0.04 where PCA recovers 0.45",
                 fontsize=10.5)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path("results/figures") / f"das_vs_pca_L{L}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
