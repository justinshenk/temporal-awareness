#!/usr/bin/env python
"""Figure-7-style PCA at the turn-transition tokens, for any model/domain.

Rows are the chat-suffix tokens; the left column colors by preference
(orange = long, teal = short), the right by discrete time-horizon category
(seconds..millennia, gray = no horizon) — the paper's layout, regenerated
from a --turn-only geometry run.

    python scripts/intertemporal/plot_turn_geometry.py <run_dir> [--layers 27]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA

LONG_COLOR, SHORT_COLOR = "#d95f02", "#1b7a8a"  # orange = long, teal = short

# Horizon categories by months-value threshold, colored blue -> red like Fig. 7.
HORIZON_BINS = [
    ("Seconds", 3e-4, "#2166ac"),
    ("Hours", 2e-2, "#4393c3"),
    ("Days", 2e-1, "#66c2a5"),
    ("Weeks", 0.9, "#a6d96a"),
    ("Months", 11.0, "#fee08b"),
    ("Years", 110.0, "#fdae61"),
    ("Decades", 1100.0, "#f46d43"),
    ("Centuries", 11000.0, "#d73027"),
    ("Millennia", float("inf"), "#a50026"),
]
NO_HORIZON_COLOR = "#8c8c8c"


def horizon_color(months):
    if months is None or not np.isfinite(months):
        return NO_HORIZON_COLOR
    for _, upper, color in HORIZON_BINS:
        if months <= upper:
            return color
    return HORIZON_BINS[-1][2]


def load_samples(run_dir: Path) -> list[dict]:
    out = []
    for sdir in sorted((run_dir / "data" / "samples").glob("sample_*")):
        try:
            mapping = json.load(open(sdir / "position_mapping.json"))
            choice = json.load(open(sdir / "choice.json"))
        except (OSError, json.JSONDecodeError):
            continue
        pos = [
            (p["abs_pos"], p.get("decoded_token", "?"))
            for p in mapping.get("positions", [])
            if p.get("format_pos") in ("chat_suffix", "chat_suffix_tail")
        ]
        if not pos:
            continue
        out.append(
            {
                "dir": sdir,
                "pos": sorted(pos),
                "chose_long": bool(choice.get("chose_long_term")),
                "months": choice.get("time_horizon_months"),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--layers", type=str, default=None)
    ap.add_argument("--components", type=str, default="resid_post")
    ap.add_argument("--max-rows", type=int, default=4, help="turn tokens to show")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir / "analysis" / "turn_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.load(open(run_dir / "summary.json"))
    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else summary.get("layers", [])
    )

    samples = load_samples(run_dir)
    print(f"loaded {len(samples)} samples")
    if not samples:
        return 1

    written = []
    for layer in layers:
        for comp in args.components.split(","):
            # Row per suffix token, skipping whitespace-only tokens (the paper
            # shows im_end, \n, im_start, assistant; we keep \n rows only when
            # among the first max_rows structural positions).
            n_rel = min(args.max_rows, max(len(s["pos"]) for s in samples))
            rows = []
            for rel in range(n_rel):
                X, cl, hm, tok = [], [], [], "?"
                for s in samples:
                    if rel >= len(s["pos"]):
                        continue
                    abs_pos, tok_ = s["pos"][rel]
                    f = s["dir"] / f"L{layer}" / f"{comp}_{abs_pos}.npy"
                    if not f.exists():
                        continue
                    X.append(np.load(f))
                    cl.append(s["chose_long"])
                    hm.append(s["months"])
                    tok = tok_
                if len(X) < 50:
                    continue
                emb = PCA(n_components=2).fit_transform(
                    np.stack(X).astype(np.float32)
                )
                rows.append((tok, emb, np.array(cl), hm))
            if not rows:
                continue

            fig, axes = plt.subplots(
                len(rows), 2, figsize=(9.2, 2.9 * len(rows)), squeeze=False
            )
            for r, (tok, emb, cl, hm) in enumerate(rows):
                axL, axR = axes[r]
                axL.scatter(
                    emb[cl, 0], emb[cl, 1], s=5, alpha=0.55, c=LONG_COLOR, lw=0
                )
                axL.scatter(
                    emb[~cl, 0], emb[~cl, 1], s=5, alpha=0.55, c=SHORT_COLOR, lw=0
                )
                colors = [horizon_color(m) for m in hm]
                axR.scatter(emb[:, 0], emb[:, 1], s=5, alpha=0.6, c=colors, lw=0)
                label = tok.replace("\n", "\\n")
                axR.text(
                    1.04, 0.5, f"{label!r}", transform=axR.transAxes,
                    fontsize=11, family="monospace", va="center",
                )
                for ax in (axL, axR):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_color("#cccccc")

            pref_handles = [
                Line2D([], [], marker="o", ls="", color=LONG_COLOR, label="Long"),
                Line2D([], [], marker="o", ls="", color=SHORT_COLOR, label="Short"),
            ]
            hor_handles = [
                Line2D([], [], marker="s", ls="", color=c, label=n)
                for n, _, c in HORIZON_BINS
            ] + [Line2D([], [], marker="s", ls="", color=NO_HORIZON_COLOR, label="No Horizon")]
            axes[0][0].legend(handles=pref_handles, fontsize=8, loc="upper left",
                              bbox_to_anchor=(-0.42, 1.0), frameon=False)
            axes[min(1, len(rows) - 1)][0].legend(
                handles=hor_handles, fontsize=6.5, loc="upper left",
                bbox_to_anchor=(-0.42, 1.0), frameon=False, markerscale=0.8,
            )
            fig.suptitle(f"Layer-{layer} {comp} PCA at the turn-transition tokens",
                         fontsize=12)
            fig.tight_layout(rect=(0.04, 0, 0.97, 0.97))
            path = out_dir / f"fig7_L{layer}_{comp}.png"
            fig.savefig(path, dpi=160, facecolor="white")
            plt.close(fig)
            written.append(path)
            print("wrote", path)

    print(f"{len(written)} figure(s)")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
