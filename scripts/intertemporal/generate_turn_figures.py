"""Regenerate the NeurReps/paper geometry figures from the v2 extractions.

Per run: per-token PCA panels at the shown layer, PC1 fans across stored
layers, and the silhouette-over-layers curve at the final turn token.
Signs of successive layers' PC1 are aligned so fan lines do not flip.

    python out/pca_browser/gen_v2_figures.py <run> <shown_layer> [n_samples]
"""

import json, random, sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

LONG_C, SHORT_C = "#d95f02", "#1b7a8a"
HORIZON_BINS = [
    ("Seconds", 3e-4, "#2166ac"), ("Hours", 2e-2, "#4393c3"),
    ("Days", 2e-1, "#66c2a5"), ("Weeks", 0.9, "#a6d96a"),
    ("Months", 11.0, "#fee08b"), ("Years", 110.0, "#fdae61"),
    ("Decades", 1100.0, "#f46d43"), ("Centuries", 11000.0, "#d73027"),
    ("Millennia", float("inf"), "#a50026"),
]

def horizon_color(years):
    months = years * 12.0
    for _, hi, c in HORIZON_BINS:
        if months < hi:
            return c
    return HORIZON_BINS[-1][2]

run, shown = sys.argv[1], int(sys.argv[2])
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 600
work = Path("out/pca_browser/ex") / run
sd = next(work.glob("*/data/samples"))
samples = sorted(sd.glob("sample_*"))
random.seed(0)
sel = random.sample(samples, min(NS, len(samples)))

store = {}
for s in sel:
    try:
        pm = json.load(open(s / "position_mapping.json"))["named_positions"]
        cj = json.load(open(s / "choice.json"))
    except Exception:
        continue
    h = cj.get("time_horizon_years")
    if h is None or h <= 0:
        continue
    lt = 1 if cj.get("chose_long_term") else 0
    turn = list(pm.get("chat_suffix", [])) + list(pm.get("chat_suffix_tail", []))
    for r, ap in enumerate(turn):
        for ld in s.glob("L*"):
            f = ld / f"resid_post_{ap}.npy"
            if f.exists():
                store.setdefault((int(ld.name[1:]), r), []).append(
                    (np.load(f).astype(np.float32), h, lt))

nr = max(r for _, r in store) + 1
layers = sorted({L for L, _ in store})
LABELS = {
    "qwen3_4b_investment": ["<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n"],
    "qwen35_4b_startup": ["<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n"],
    "gemma2_9b_climate": ["<end_of_turn>", "\\n", "<start_of_turn>", "model", "\\n"],
    "llama31_8b_health": ["<|eot_id|>", "<|start_header_id|>", "assistant",
                          "<|end_header_id|>", "\\n\\n"],
    "mistral7b_education": ["[/INST]"],
}[run]

out = Path("out/fig_v2/neurreps"); out.mkdir(parents=True, exist_ok=True)

# ---- per-token projections with sign alignment across layers -------------
proj = {}   # (L, r) -> (Z[n,2], h[n], lt[n])
for r in range(nr):
    prev = None
    for L in layers:
        V = store.get((L, r))
        if not V:
            continue
        X = np.stack([v[0] for v in V])
        h = np.array([v[1] for v in V]); lt = np.array([v[2] for v in V], bool)
        X = X - X.mean(0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Z = X @ Vt[:2].T
        if prev is not None and len(prev) == len(Z):
            if np.corrcoef(Z[:, 0], prev)[0, 1] < 0:
                Z[:, 0] = -Z[:, 0]
        prev = Z[:, 0]
        proj[(L, r)] = (Z, h, lt)

# ---- panels at the shown layer (one file per token per coloring) ---------
for r in range(nr):
    if (shown, r) not in proj:
        continue
    Z, h, lt = proj[(shown, r)]
    for kind in ("term_chosen", "time_scale"):
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        if kind == "term_chosen":
            ax.scatter(Z[lt, 0], Z[lt, 1], s=6, alpha=0.6, c=LONG_C, lw=0)
            ax.scatter(Z[~lt, 0], Z[~lt, 1], s=6, alpha=0.6, c=SHORT_C, lw=0)
        else:
            ax.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.65,
                       c=[horizon_color(y) for y in h], lw=0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        fig.tight_layout()
        fig.savefig(out / f"{run}_L{shown}__chat_suffix_{r}__{kind}.pdf")
        plt.close(fig)

# ---- fans ----------------------------------------------------------------
fig, axes = plt.subplots(1, nr, figsize=(3.0 * nr, 3.0), squeeze=False)
for r in range(nr):
    ax = axes[0][r]
    Ls = [L for L in layers if (L, r) in proj]
    n = min(len(proj[(L, r)][0]) for L in Ls)
    tr = np.stack([proj[(L, r)][0][:n, 0] for L in Ls])   # [nlayers, n]
    lt = proj[(Ls[0], r)][2][:n]
    xs = np.arange(len(Ls))
    for i in range(n):
        ax.plot(xs, tr[:, i], lw=0.25, alpha=0.28,
                color=LONG_C if lt[i] else SHORT_C)
    ax.set_title(repr(LABELS[r]), fontsize=9, family="monospace")
    ax.set_xticks(xs[:: max(1, len(Ls) // 5)])
    ax.set_xticklabels([str(L) for L in Ls[:: max(1, len(Ls) // 5)]], fontsize=6)
    ax.set_xlabel("Layer", fontsize=7)
    if r == 0:
        ax.set_ylabel("PC1 projection", fontsize=7)
        ax.plot([], [], color=LONG_C, label="Long")
        ax.plot([], [], color=SHORT_C, label="Short")
        ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(out / f"fan_{run}.pdf")
plt.close(fig)

# ---- silhouette over layers at the final turn token ----------------------
rfin = nr - 1
sil = {}
for L in layers:
    if (L, rfin) not in proj:
        continue
    Z, h, lt = proj[(L, rfin)]
    if 1 < lt.sum() < len(lt) - 1:
        sil[L] = float(silhouette_score(Z, lt))
json.dump({"run": run, "final_token": LABELS[rfin], "shown": shown,
           "silhouette": sil},
          open(out / f"sil_{run}.json", "w"), indent=1)
print(f"[{run}] n={len(sel)} silhouette at {LABELS[rfin]!r}: "
      + " ".join(f"L{L}={v:+.3f}" for L, v in sorted(sil.items())))
print(f"[{run}] wrote panels shown=L{shown}, fan, sil")
