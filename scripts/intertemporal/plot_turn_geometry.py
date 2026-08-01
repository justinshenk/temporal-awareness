#!/usr/bin/env python
"""PCA of turn-transition activations, colored by preference and horizon.

Reads a --turn-only geometry run and renders one figure per (layer, component):
a row of PCA scatters, one panel per chat-suffix token, colored by the model's
chosen option and by time horizon. This is the change-of-turn collapse figure
(paper Figs. 4-5) for an arbitrary model and domain.

    python scripts/intertemporal/plot_turn_geometry.py out/geo/climate_geometry \
        --out-dir out/geo/climate_geometry/analysis/turn_plots
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Accessible two-class palette (blue/orange) and a perceptual ramp for horizon.
SHORT_COLOR, LONG_COLOR = "#4477aa", "#ee6677"
HORIZON_CMAP = "viridis"


def load_samples(run_dir: Path) -> list[dict]:
    samples = []
    for sdir in sorted((run_dir / "data" / "samples").glob("sample_*")):
        try:
            with open(sdir / "position_mapping.json") as fh:
                mapping = json.load(fh)
            with open(sdir / "choice.json") as fh:
                choice = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        suffix_pos = [
            p["abs_pos"]
            for p in mapping.get("positions", [])
            if p.get("format_pos") in ("chat_suffix", "chat_suffix_tail")
        ]
        tokens = {
            p["abs_pos"]: p.get("decoded_token", "?")
            for p in mapping.get("positions", [])
        }
        if not suffix_pos:
            continue
        samples.append(
            {
                "dir": sdir,
                "suffix_pos": sorted(suffix_pos),
                "tokens": tokens,
                "chose_long": bool(choice.get("chose_long_term")),
                "horizon_months": choice.get("time_horizon_months"),
            }
        )
    return samples


def collect_matrix(
    samples: list[dict], layer: int, component: str, rel: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    rows, chose, horizon, token = [], [], [], "?"
    for s in samples:
        if rel >= len(s["suffix_pos"]):
            continue
        abs_pos = s["suffix_pos"][rel]
        path = s["dir"] / f"L{layer}" / f"{component}_{abs_pos}.npy"
        if not path.exists():
            continue
        rows.append(np.load(path))
        chose.append(s["chose_long"])
        horizon.append(s["horizon_months"] or np.nan)
        token = s["tokens"].get(abs_pos, "?")
    if not rows:
        return np.empty((0, 0)), np.empty(0), np.empty(0), token
    return np.stack(rows), np.array(chose), np.array(horizon, dtype=float), token


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--layers", type=str, default=None, help="comma list; default: all in summary")
    ap.add_argument("--components", type=str, default="resid_post,attn_out")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir / "analysis" / "turn_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "summary.json") as fh:
        summary = json.load(fh)
    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else summary.get("layers", [])
    )
    components = args.components.split(",")

    samples = load_samples(run_dir)
    print(f"loaded {len(samples)} samples from {run_dir}")
    if not samples:
        return 1

    n_rel = max(len(s["suffix_pos"]) for s in samples)
    written = []
    for layer in layers:
        for comp in components:
            panels = []
            for rel in range(n_rel):
                X, chose, horizon, token = collect_matrix(samples, layer, comp, rel)
                if X.shape[0] < 20:
                    continue
                emb = PCA(n_components=2).fit_transform(X.astype(np.float32))
                panels.append((rel, token, emb, chose, horizon))
            if not panels:
                continue

            fig, axes = plt.subplots(
                2, len(panels), figsize=(3.4 * len(panels), 6.4), squeeze=False
            )
            for col, (rel, token, emb, chose, horizon) in enumerate(panels):
                ax = axes[0][col]
                for mask, color, label in (
                    (~chose, SHORT_COLOR, "short"),
                    (chose, LONG_COLOR, "long"),
                ):
                    ax.scatter(
                        emb[mask, 0], emb[mask, 1], s=4, alpha=0.5, c=color, label=label
                    )
                ax.set_title(f"r{rel} {token!r}", fontsize=9)
                if col == 0:
                    ax.set_ylabel("choice")
                    ax.legend(fontsize=7, markerscale=2)
                ax.set_xticks([])
                ax.set_yticks([])

                ax = axes[1][col]
                finite = np.isfinite(horizon)
                if finite.any():
                    sc = ax.scatter(
                        emb[finite, 0],
                        emb[finite, 1],
                        s=4,
                        alpha=0.6,
                        c=np.log10(horizon[finite] + 1e-3),
                        cmap=HORIZON_CMAP,
                    )
                    if col == len(panels) - 1:
                        fig.colorbar(sc, ax=ax, label="log10 horizon (months)")
                ax.scatter(
                    emb[~finite, 0], emb[~finite, 1], s=4, alpha=0.3, c="#999999"
                )
                if col == 0:
                    ax.set_ylabel("horizon")
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(f"L{layer} {comp} — turn-transition PCA", fontsize=11)
            fig.tight_layout()
            path = out_dir / f"L{layer}_{comp}_turn_pca.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
            print(f"wrote {path}")

    print(f"{len(written)} figure(s) -> {out_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
