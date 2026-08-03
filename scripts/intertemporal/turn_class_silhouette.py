#!/usr/bin/env python
"""Per-layer separation of the two choice classes at the turn-transition tokens.

The Fig-7 story is qualitative: at the change of turn the two preference
classes collapse into one cloud, and by the role token they split into two.
This script measures that with a silhouette score, so a temporal run and a
risk run can be compared on the same scale.

For every layer and every turn-transition token it loads the activations of
all samples, projects them with a 2-component PCA (the space the figure
shows), and scores the two classes with the silhouette coefficient. The score
in the full activation space is reported beside it, so a verdict never rests
on the projection alone.

    python scripts/intertemporal/turn_class_silhouette.py <run_dir> \
        --components resid_post --out <run_dir>/analysis/turn_silhouette.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_turn_geometry import load_samples  # noqa: E402


def score_position(
    samples: list[dict], layer: int, comp: str, rel: int, n_pca: int
) -> dict | None:
    """Silhouette of the two choice classes at one layer and one turn token."""
    vectors, labels, token = [], [], "?"
    for sample in samples:
        if rel >= len(sample["pos"]):
            continue
        abs_pos, token_ = sample["pos"][rel]
        path = sample["dir"] / f"L{layer}" / f"{comp}_{abs_pos}.npy"
        if not path.exists():
            continue
        vectors.append(np.load(path))
        labels.append(sample["chose_long"])
        token = token_
    if len(vectors) < 50:
        return None

    X = np.stack(vectors).astype(np.float32)
    y = np.asarray(labels)
    if len(np.unique(y)) < 2 or min(int(y.sum()), int((~y).sum())) < 10:
        return None

    pca = PCA(n_components=n_pca)
    emb = pca.fit_transform(X)
    return {
        "layer": layer,
        "component": comp,
        "rel_pos": rel,
        "token": token,
        "n": int(X.shape[0]),
        "n_long": int(y.sum()),
        "n_short": int((~y).sum()),
        "silhouette_pca": float(silhouette_score(emb, y)),
        "silhouette_full": float(silhouette_score(X, y)),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--components", default="resid_post")
    parser.add_argument("--layers", default=None, help="comma separated; default: summary.json")
    parser.add_argument("--max-rel", type=int, default=4, help="turn tokens to score")
    parser.add_argument("--pca-components", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    summary = json.load(open(run_dir / "summary.json"))
    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else sorted(summary.get("layers", []))
    )
    samples = load_samples(run_dir)
    if not samples:
        print(f"no samples with turn positions under {run_dir}")
        return 1
    n_rel = min(args.max_rel, max(len(s["pos"]) for s in samples))
    print(f"{len(samples)} samples, {len(layers)} layers, {n_rel} turn tokens")

    rows = []
    for comp in args.components.split(","):
        for layer in layers:
            for rel in range(n_rel):
                row = score_position(samples, layer, comp, rel, args.pca_components)
                if row is None:
                    continue
                rows.append(row)
                print(
                    f"  L{row['layer']:<3} {row['component']:<10} rel={row['rel_pos']} "
                    f"{row['token']!r:<16} n={row['n']:<5} "
                    f"sil_pca={row['silhouette_pca']:+.4f} "
                    f"sil_full={row['silhouette_full']:+.4f}"
                )
    if not rows:
        print("no layer/position had enough samples in both classes")
        return 1

    out_path = args.out or run_dir / "analysis" / "turn_silhouette.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_rel = max(r["rel_pos"] for r in rows)
    final_rows = [r for r in rows if r["rel_pos"] == final_rel]
    best = max(final_rows, key=lambda r: r["silhouette_pca"])
    with open(out_path, "w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "model": summary.get("model"),
                "dataset": summary.get("config", {}).get("name"),
                "n_samples": len(samples),
                "pca_components": args.pca_components,
                "final_rel_pos": final_rel,
                "best_at_final_token": best,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}")
    print(
        f"final token rel={final_rel} {best['token']!r}: best silhouette "
        f"{best['silhouette_pca']:+.4f} at L{best['layer']} ({best['component']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
