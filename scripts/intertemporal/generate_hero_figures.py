"""Three-component PCA renders in the original hero style, from v2 data.

Produces, for one run and layer: a single large horizon-colored manifold
per requested token (the hero), and a rows-by-two grid with preference on
the left and horizon on the right.

    python out/pca_browser/gen_hero_figures.py <run> <layer> [n_samples]
"""

import json, random, sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

LONG_C, SHORT_C = "#c65d3b", "#2e7d8a"
BINS = [  # upper bound in years
    ("Seconds", 60 / 31557600, "#3b64d8"),
    ("Minutes", 3600 / 31557600, "#4a9fd8"),
    ("Hours", 86400 / 31557600, "#3fa9a0"),
    ("Days", 604800 / 31557600, "#58b368"),
    ("Weeks", 1 / 12, "#a5c452"),
    ("Months", 1.0, "#e3c33f"),
    ("Years", 10.0, "#e39440"),
    ("Decades", 100.0, "#d65f2c"),
    ("Centuries", float("inf"), "#c03a2b"),
]
NOH = ("No Horizon", "#5d6b7a")

def bin_of(y):
    if y is None or y <= 0:
        return NOH[0]
    for name, hi, _ in BINS:
        if y < hi:
            return name
    return BINS[-1][0]

COLOR = {n: c for n, _, c in BINS} | {NOH[0]: NOH[1]}

run, layer = sys.argv[1], int(sys.argv[2])
EL = float(sys.argv[4]) if len(sys.argv) > 4 else 18
AZ = float(sys.argv[5]) if len(sys.argv) > 5 else -55
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
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
    lt = 1 if cj.get("chose_long_term") else 0
    turn = list(pm.get("chat_suffix", [])) + list(pm.get("chat_suffix_tail", []))
    for r, ap in enumerate(turn):
        f = s / f"L{layer}" / f"resid_post_{ap}.npy"
        if f.exists():
            store.setdefault(r, []).append((np.load(f).astype(np.float32), h, lt))

LABELS = {
    "qwen3_4b_investment": ["<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n"],
}.get(run, [str(r) for r in sorted(store)])

out = Path("out/fig_v2/hero"); out.mkdir(parents=True, exist_ok=True)

def pca3(V):
    X = np.stack([v[0] for v in V]); X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:3].T

def scatter3(ax, Z, colors, size):
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=colors, s=size, alpha=0.85,
               linewidths=0, depthshade=True)
    ax.set_axis_off()
    ax.view_init(elev=EL, azim=AZ)

# ---- heroes --------------------------------------------------------------
for r in sorted(store):
    V = store[r]
    Z = pca3(V)
    cats = [bin_of(v[1]) for v in V]
    fig = plt.figure(figsize=(10.5, 6.8), facecolor="#f2f4f6")
    ax = fig.add_subplot(111, projection="3d", facecolor="#f2f4f6")
    scatter3(ax, Z, [COLOR[c] for c in cats], 46)
    present = [n for n, _, _ in BINS if n in set(cats)] + ([NOH[0]] if NOH[0] in cats else [])
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=COLOR[n], label=n)
                       for n in present],
              loc="upper right", bbox_to_anchor=(1.12, 0.95), fontsize=10,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"hero_{run}_L{layer}_r{r}_e{int(EL)}a{int(AZ)}.png", dpi=160,
                facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"hero r{r} ({LABELS[r]!r}) n={len(V)}")

# ---- grid: rows = tokens, cols = preference | horizon --------------------
rows = sorted(store)
fig = plt.figure(figsize=(8.2, 2.55 * len(rows)), facecolor="white")
for i, r in enumerate(rows):
    V = store[r]
    Z = pca3(V)
    lt = np.array([v[2] for v in V], bool)
    cats = [bin_of(v[1]) for v in V]
    axL = fig.add_subplot(len(rows), 2, 2 * i + 1, projection="3d")
    scatter3(axL, Z, [LONG_C if x else SHORT_C for x in lt], 15)
    axR = fig.add_subplot(len(rows), 2, 2 * i + 2, projection="3d")
    scatter3(axR, Z, [COLOR[c] for c in cats], 15)
    axR.text2D(1.02, 0.5, repr(LABELS[r]), transform=axR.transAxes,
               fontsize=9, family="monospace", va="center")
fig.subplots_adjust(left=0, right=0.86, top=1.02, bottom=-0.02,
                    hspace=-0.28, wspace=-0.08)
fig.savefig(out / f"grid_{run}_L{layer}.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"grid rows={rows}")
