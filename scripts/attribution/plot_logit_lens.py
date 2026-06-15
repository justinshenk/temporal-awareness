"""Plot E1: where the emitted answer token crystallizes across depth at the L20 patch site.

Reads ``logit_lens_patch_L{L}.json`` and, for each mode (patch / lora / base), renders the median
logit-lens rank of the emitted **answer-bearing** token versus readout layer. A curve that is already
at rank 0 by L20 and stays flat means the answer is present at the patch site (communication); a curve
that descends to rank 0 only at deeper layers means those layers compute it.

    uv run python -m scripts.attribution.plot_logit_lens \
        --json results/attribution/logit_lens_patch_L20.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def median(xs):
    s = sorted(xs)
    n = len(s)
    return float("nan") if n == 0 else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    L = data["layer"]
    layers = data["readout_layers"]
    colors = {"patch": "#c0392b", "lora": "#2c3e50", "base": "#7f8c8d"}
    styles = {"patch": "o-", "lora": "s--", "base": "^:"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for mode, md in data["modes"].items():
        ans = [r for p in md["problems"] for r in p["steps"] if r["answer_bearing"]]
        med = [median([r["rank"][str(li)] for r in ans]) for li in layers]
        ax.plot(layers, med, styles.get(mode, "o-"), color=colors.get(mode, None), lw=2.0, ms=7,
                label=f"{mode} (n={len(ans)} ans-tokens; top-1@L{L}: {md['answer_top1_at_L']})")

    ax.axvline(L, color="#16a085", ls=":", alpha=0.6)
    ax.set_yscale("symlog")
    ax.set_xlabel("readout layer (logit lens)")
    ax.set_ylabel("median rank of emitted answer token (0 = argmax; symlog)")
    ax.set_xticks(layers)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.set_title(f"Where does the answer token crystallize? (L{L} patch site)\n"
                 f"rank 0 already at L{L} ⇒ answer present (communication); "
                 f"descends only deeper ⇒ downstream computation",
                 fontsize=10.5)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path("results/figures") / f"logit_lens_patch_L{L}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
