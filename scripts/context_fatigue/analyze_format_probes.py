"""Analysis for the E6 residual-stream probes (capture: ``run_format_probes.py``).

Probe methodology follows Dongre et al. (arXiv:2605.12922), the closest prior art, so the
numbers are comparable: StandardScaler → PCA (≤50 components, clipped to the fold's rank) → LDA,
leave-one-out cross-validation at the episode level, and a 200-shuffle permutation null at the
peak layer (labels shuffled, the full pipeline re-run, p = fraction of null AUCs ≥ observed).

- **Probe 1**: per layer, trained on depth 0 to separate format-system from neutral-system
  states, tested at every mmlu depth. Flat AUC across depth = the instruction stays decodable
  while compliance is 0.000. The permutation null shuffles the *training* labels and re-fits.
- **Probe 2**: within gsm8k's mixed cells (depths 12 and 15), LOO-CV per layer predicting
  whether the reply will comply, labels joined from the committed run's ``turns.csv`` (the
  rebuild is bit-identical, verified via the spans re-runs). Above-chance AUC before the first
  generated token = the mode is set in the residual state; the layer profile localizes it.

    .venv/bin/python scripts/context_fatigue/analyze_format_probes.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_PERMUTATIONS = 200


def load_depth(out_dir: Path, filler: str, depth: int):
    z = np.load(out_dir / f"{filler}_d{depth}.npz")
    return z["states"], pd.DataFrame(json.loads(str(z["rows"])))


def lda_pipeline(n_train: int, n_features: int):
    k = min(50, n_train - 1, n_features)
    return make_pipeline(StandardScaler(), PCA(n_components=k),
                         LinearDiscriminantAnalysis())


def _degenerate(x) -> bool:
    """True when the features carry no variance — the input-embedding row at the shared final
    prompt position is byte-identical across samples, which would crash LDA's SVD. Returning
    chance keeps layer 0 as the baseline row Dongre et al. use it as."""
    return float(x.std(axis=0).max()) < 1e-8


def transfer_auc(x_train, y_train, x_test, y_test) -> float:
    if _degenerate(x_train):
        return 0.5
    clf = lda_pipeline(len(y_train), x_train.shape[1]).fit(x_train, y_train)
    return roc_auc_score(y_test, clf.decision_function(x_test))


def loo_auc(x, y) -> float:
    if _degenerate(x):
        return 0.5
    scores = np.empty(len(y))
    for tr, te in LeaveOneOut().split(x):
        clf = lda_pipeline(len(tr), x.shape[1]).fit(x[tr], y[tr])
        scores[te] = clf.decision_function(x[te])
    return roc_auc_score(y, scores)


def probe1_instruction_presence(out_dir: Path, depths, rng):
    states0, rows0 = load_depth(out_dir, "mmlu", 0)
    y0 = (rows0.variant == "format").astype(int).values
    n_layers = states0.shape[1]
    print("Probe 1 — instruction presence (train depth 0, test each depth)")
    results = {}
    for depth in depths:
        states, rows = load_depth(out_dir, "mmlu", depth)
        y = (rows.variant == "format").astype(int).values
        aucs = [transfer_auc(states0[:, li], y0, states[:, li], y)
                for li in range(n_layers)]
        best = int(np.argmax(aucs))
        null = np.array([transfer_auc(states0[:, best], rng.permutation(y0),
                                      states[:, best], y)
                         for _ in range(N_PERMUTATIONS)])
        p = float((null >= aucs[best]).mean())
        results[depth] = {"auc_by_layer": [float(a) for a in aucs], "best_layer": best,
                          "best_auc": float(aucs[best]), "mean_auc": float(np.mean(aucs)),
                          "perm_p_at_best": p, "perm_null_mean": float(null.mean()),
                          "n": int(len(y))}
        print(f"  depth {depth:2d}: best L{best:2d} AUC={aucs[best]:.3f} "
              f"(mean {np.mean(aucs):.3f}, perm null {null.mean():.3f}, p={p:.3f}, n={len(y)})")
    return results


def probe2_mode_visibility(out_dir: Path, turns_csv: Path, depths, rng):
    turns = pd.read_csv(turns_csv)[["depth", "probe", "fully_compliant"]]
    xs, ys = [], []
    for depth in depths:
        states, rows = load_depth(out_dir, "gsm8k", depth)
        rows = rows.merge(turns[turns.depth == depth], on=["depth", "probe"], how="left")
        keep = rows.fully_compliant.notna().values
        xs.append(states[keep])
        ys.append(rows.fully_compliant[keep].astype(int).values)
    x, y = np.concatenate(xs), np.concatenate(ys)
    n_layers = x.shape[1]
    print(f"Probe 2 — will the reply comply? gsm8k depths {depths}, "
          f"n={len(y)} ({int(y.sum())} compliant), LOO-CV")
    aucs = [loo_auc(x[:, li], y) for li in range(n_layers)]
    best = int(np.argmax(aucs))
    null = np.array([loo_auc(x[:, best], rng.permutation(y))
                     for _ in range(N_PERMUTATIONS)])
    p = float((null >= aucs[best]).mean())
    first_high = next((li for li, a in enumerate(aucs) if a > 0.8), None)
    print(f"  best L{best:2d} LOO-AUC={aucs[best]:.3f} "
          f"(perm null {null.mean():.3f}, p={p:.3f}); first layer with AUC>0.8: {first_high}")
    return {"auc_by_layer": [float(a) for a in aucs], "best_layer": best,
            "best_auc": float(aucs[best]), "perm_p_at_best": p,
            "perm_null_mean": float(null.mean()), "first_layer_above_0.8": first_high,
            "n": int(len(y)), "n_compliant": int(y.sum())}


def vector_geometry(out_dir: Path, mmlu_deep: int, gsm8k_deep: int):
    """Cosines between the candidate mode vectors — one per demonstrated style, per layer.

    High cosine between the mmlu and gsm8k mean-diff vectors = a shared precedent axis;
    near-orthogonal = style-specific vectors. Computed for every layer so the steering layer's
    value is read in context.
    """
    def meandiff(filler, deep):
        s_deep, r_deep = load_depth(out_dir, filler, deep)
        s0, r0 = load_depth(out_dir, filler, 0)
        keep_d = (r_deep.variant == "format").values
        keep_0 = (r0.variant == "format").values
        return s_deep[keep_d].mean(axis=0) - s0[keep_0].mean(axis=0)

    vm, vg = meandiff("mmlu", mmlu_deep), meandiff("gsm8k", gsm8k_deep)
    cos = []
    for li in range(vm.shape[0]):
        denom = np.linalg.norm(vm[li]) * np.linalg.norm(vg[li])
        cos.append(float(np.dot(vm[li], vg[li]) / denom) if denom > 1e-12 else 0.0)
    best = int(np.argmax(np.abs(cos)))
    print(f"Vector geometry — cos(mmlu d{mmlu_deep}−d0, gsm8k d{gsm8k_deep}−d0) by layer: "
          f"median {np.median(cos):+.3f}, max |cos| {cos[best]:+.3f} at L{best}")
    return {"cos_by_layer": cos, "median_cos": float(np.median(cos))}


def export_probe2_direction(out_dir: Path, turns_csv: Path, depths, stack_layer: int,
                            dest: Path):
    """Refit Probe 2's pipeline on the full data at one layer and backproject the LDA weights
    into activation space: decision(x) = w_lda · PCA(scale(x)), so in x-coordinates
    w = (w_lda @ components) / scale. This is the direction that *reads* the mode; saving it
    lets the steering driver test whether it can also *remove* it."""
    turns = pd.read_csv(turns_csv)[["depth", "probe", "fully_compliant"]]
    xs, ys = [], []
    for depth in depths:
        states, rows = load_depth(out_dir, "gsm8k", depth)
        rows = rows.merge(turns[turns.depth == depth], on=["depth", "probe"], how="left")
        keep = rows.fully_compliant.notna().values
        xs.append(states[keep][:, stack_layer])
        ys.append(rows.fully_compliant[keep].astype(int).values)
    x, y = np.concatenate(xs), np.concatenate(ys)
    pipe = lda_pipeline(len(y), x.shape[1]).fit(x, y)
    scaler, pca, lda = pipe.named_steps.values()
    w = (lda.coef_ @ pca.components_).ravel() / scaler.scale_
    np.save(dest, w.astype(np.float32))
    print(f"probe-2 direction @stack L{stack_layer}: |w|={np.linalg.norm(w):.3f} -> {dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", default="results/context_fatigue/e6_format_probes")
    p.add_argument("--gsm8k-turns", default="results/context_fatigue/e6_gsm8k/turns.csv")
    p.add_argument("--mmlu-depths", type=int, nargs="+",
                   default=[0, 3, 7, 14, 21, 28, 35, 42])
    p.add_argument("--probe2-depths", type=int, nargs="+", default=[12, 15])
    p.add_argument("--export-direction", default=None,
                   help="skip the analyses; refit Probe 2 at --direction-layer on all data and "
                        "save the backprojected direction to this .npy path")
    p.add_argument("--direction-layer", type=int, default=21)
    args = p.parse_args()
    out_dir = Path(args.capture_dir)
    rng = np.random.default_rng(0)

    if args.export_direction:
        export_probe2_direction(out_dir, Path(args.gsm8k_turns), args.probe2_depths,
                                args.direction_layer, Path(args.export_direction))
        return

    results = {"probe1": probe1_instruction_presence(out_dir, args.mmlu_depths, rng),
               "probe2": probe2_mode_visibility(out_dir, Path(args.gsm8k_turns),
                                                args.probe2_depths, rng),
               "geometry": vector_geometry(out_dir, max(args.mmlu_depths), 15)}
    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=1))
    print(f"Saved to {out_dir}/probe_results.json", flush=True)


if __name__ == "__main__":
    main()
