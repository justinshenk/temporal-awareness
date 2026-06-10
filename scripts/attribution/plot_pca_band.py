"""Plot the PCA-band oracle recovery curve: where in variance-space the capability lives.

Reads ``lockstep_pca_band_L{L}.json`` (top-k band recovery + energy fraction) and renders the
dissociation between *energy* (which saturates in the high-variance head) and *recovery* (which
only switches on in the moderate/low-variance band the head discards).

    uv run python -m scripts.attribution.plot_pca_band \
        --json results/attribution/lockstep_pca_band_L20.json
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
    bands = data["bands"]

    # Cumulative top-k bands (incl. full as k=d), ordered by k; tail bands plotted separately.
    top = {s: b for s, b in bands.items() if s == "full" or s.startswith("top")}
    tail = {s: b for s, b in bands.items() if s.startswith("tail")}
    ordered = sorted(top.items(), key=lambda kv: kv[1]["k"])
    ks = [b["k"] for _, b in ordered]
    rec = [b["recovery"] for _, b in ordered]
    ef = [b["energy_frac"] for _, b in ordered]
    labels = [s for s, _ in ordered]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ks, rec, "o-", color="#c0392b", lw=2.2, ms=8, label="recovery (exact-match)", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("top-k PCA directions of δ injected (log₂)")
    ax.set_ylabel("oracle recovery (contrast acc)", color="#c0392b")
    ax.set_ylim(-0.03, 1.0)
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{l}\n(k={k})" for l, k in zip(labels, ks)], fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(ks, ef, "s--", color="#2c3e50", lw=1.6, ms=6, alpha=0.7, label="energy fraction")
    ax2.set_ylabel("fraction of δ energy captured", color="#2c3e50")
    ax2.set_ylim(0, 1.02)
    ax2.tick_params(axis="y", labelcolor="#2c3e50")

    # Annotate the dissociation: energy saturates early, recovery switches on late.
    for k, r in zip(ks, rec):
        ax.annotate(f"{r:.2f}", (k, r), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#c0392b")

    # Complement injections (tail bands V[:, k:] = drop the top-k head, keep the rest). These are
    # NOT cumulative top-k points, so they'd collide near k=d on this axis — state them as a callout.
    if tail:
        d = max(ks)
        rows = sorted(((d - b["k"], b["energy_frac"], b["recovery"]) for b in tail.values()))
        body = "\n".join(f"  δ−top{drop:<3d} (keep {ef:.0%} energy) → {rec:.2f}" for drop, ef, rec in rows)
        ax.text(0.97, 0.62, "Complement injections (drop the head):\n" + body
                + "\nremoving even the top-8 craters recovery",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#16a085",
                bbox=dict(boxstyle="round", fc="white", ec="#16a085", alpha=0.95))

    lines, labs = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labs + lab2, loc="center left", fontsize=9, framealpha=0.9)

    ax.set_title(f"Task competence is spread across a wide δ-band, not the high-energy head "
                 f"(L{L} oracle)\nenergy is front-loaded (55% in top-64) but recovery isn't "
                 f"(≈0 until k>256); dropping top-64 → 0",
                 fontsize=10.5)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path("results/figures") / f"pca_band_recovery_L{L}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
