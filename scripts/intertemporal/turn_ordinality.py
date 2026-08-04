#!/usr/bin/env python
"""Does the first principal component order a swept quantity at the turn tokens?

The paper's geometric claim is that a swept quantity lays out along a gradient
in the residual stream, not merely that it is decodable. This script measures
that directly: at every layer and every turn-transition token it fits a
2-component PCA over all samples and asks how well PC1 orders the swept levels.

Two numbers per cell, because they answer different questions. The Spearman
correlation asks whether the order is right at all, and it is invariant to any
monotone rescaling of the swept quantity, so log-odds and raw probability give
the same value. The Pearson correlation against the log-scaled quantity asks
the stronger question of whether the levels are laid out evenly, which is what
"a gradient from seconds to millennia" actually asserts.

The sign of PC1 is arbitrary, so magnitudes are what count.

    python scripts/intertemporal/turn_ordinality.py <run_dir> --ordinal probability
    python scripts/intertemporal/turn_ordinality.py <run_dir> --ordinal horizon
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_turn_geometry import load_samples  # noqa: E402

PROBABILITY_RE = re.compile(r"probability of (\d+) percent")


def probability_of(sample_dir: Path) -> float | None:
    """Gamble probability, read from the rendered prompt itself."""
    try:
        text = json.load(open(sample_dir / "prompt_sample.json")).get("text", "")
    except (OSError, json.JSONDecodeError):
        return None
    match = PROBABILITY_RE.search(text)
    return float(match.group(1)) if match else None


def horizon_of(sample_dir: Path) -> float | None:
    """Time horizon in months, from the stored choice record."""
    try:
        months = json.load(open(sample_dir / "choice.json")).get("time_horizon_months")
    except (OSError, json.JSONDecodeError):
        return None
    return float(months) if months else None


def log_scale(values: np.ndarray, kind: str) -> np.ndarray:
    """Put the swept quantity on the scale its gradient is claimed to be even on."""
    if kind == "probability":
        p = np.clip(values / 100.0, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))
    return np.log10(values)


def score_cell(
    samples: list[dict], layer: int, comp: str, rel: int, kind: str
) -> dict | None:
    """PC1 ordinality and class silhouette at one layer and one turn token."""
    vectors, ordinals, labels, token = [], [], [], "?"
    for sample in samples:
        if rel >= len(sample["pos"]) or sample["ordinal"] is None:
            continue
        abs_pos, token_ = sample["pos"][rel]
        path = sample["dir"] / f"L{layer}" / f"{comp}_{abs_pos}.npy"
        if not path.exists():
            continue
        vectors.append(np.load(path))
        ordinals.append(sample["ordinal"])
        labels.append(sample["chose_long"])
        token = token_
    if len(vectors) < 50:
        return None

    X = np.stack(vectors).astype(np.float32)
    v = np.asarray(ordinals, dtype=np.float64)
    y = np.asarray(labels)
    levels = np.unique(v)
    if len(levels) < 3:
        return None

    pca = PCA(n_components=2)
    emb = pca.fit_transform(X)
    pc1 = emb[:, 0]
    v_log = log_scale(v, kind)

    rho = spearmanr(pc1, v).statistic
    r_log = pearsonr(pc1, v_log).statistic
    level_means = np.array([pc1[v == lv].mean() for lv in levels])
    rho_levels = spearmanr(level_means, levels).statistic
    r_levels = pearsonr(level_means, log_scale(levels, kind)).statistic

    # Separation of the two choice classes, for reference against the
    # binary-split measurement.
    sil = None
    if len(np.unique(y)) == 2 and min(int(y.sum()), int((~y).sum())) >= 10:
        sil = float(silhouette_score(emb, y))

    return {
        "layer": layer,
        "component": comp,
        "rel_pos": rel,
        "token": token,
        "n": int(X.shape[0]),
        "n_levels": int(len(levels)),
        "spearman_pc1": float(rho),
        "abs_spearman_pc1": float(abs(rho)),
        "pearson_pc1_log": float(r_log),
        "abs_pearson_pc1_log": float(abs(r_log)),
        "spearman_level_means": float(rho_levels),
        "abs_spearman_level_means": float(abs(rho_levels)),
        "pearson_level_means_log": float(r_levels),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "silhouette_choice_pca": sil,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--ordinal", choices=["probability", "horizon"], required=True)
    parser.add_argument("--components", default="resid_post")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--max-rel", type=int, default=4)
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
    read_ordinal = probability_of if args.ordinal == "probability" else horizon_of
    for sample in samples:
        sample["ordinal"] = read_ordinal(sample["dir"])
    usable = [s for s in samples if s["ordinal"] is not None]
    if not usable:
        print(f"no sample carried a readable {args.ordinal}")
        return 1
    levels = sorted({s["ordinal"] for s in usable})
    print(
        f"{len(usable)} of {len(samples)} samples carry a {args.ordinal}; "
        f"{len(levels)} levels: {levels if len(levels) <= 25 else '...'}"
    )
    n_rel = min(args.max_rel, max(len(s["pos"]) for s in usable))

    rows = []
    for comp in args.components.split(","):
        for layer in layers:
            for rel in range(n_rel):
                row = score_cell(usable, layer, comp, rel, args.ordinal)
                if row is None:
                    continue
                rows.append(row)
                print(
                    f"  L{row['layer']:<3} rel={row['rel_pos']} {row['token']!r:<16} "
                    f"n={row['n']:<5} |rho|={row['abs_spearman_pc1']:.3f} "
                    f"|r_log|={row['abs_pearson_pc1_log']:.3f} "
                    f"|rho_levels|={row['abs_spearman_level_means']:.3f}"
                )
    if not rows:
        print("no layer/position had enough samples")
        return 1

    out_path = args.out or run_dir / "analysis" / f"turn_ordinality_{args.ordinal}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = max(rows, key=lambda r: r["abs_spearman_pc1"])
    with open(out_path, "w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "ordinal": args.ordinal,
                "n_samples": len(usable),
                "levels": levels if len(levels) <= 50 else None,
                "n_levels": len(levels),
                "best_cell": best,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out_path}")
    print(
        f"strongest ordering: L{best['layer']} rel={best['rel_pos']} "
        f"{best['token']!r} |rho|={best['abs_spearman_pc1']:.3f} "
        f"|r_log|={best['abs_pearson_pc1_log']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
