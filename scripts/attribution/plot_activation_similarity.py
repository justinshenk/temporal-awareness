"""Plot per-layer LoRA-vs-LoReFT edit similarity on commonsense.

Reads ``lora_vs_loreft_similarity.json`` and renders the per-layer cosine / linear-CKA(δ) /
subspace-overlap / CKA(h) curves. High flat curves ⇒ the two methods converge on the same
representational edit; curves near zero ⇒ same behavior, different routes.

    uv run python -m scripts.attribution.plot_activation_similarity \
        --json results/attribution/loreft_commonsense/lora_vs_loreft_similarity.json
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
    layers = data["layers"]
    pl = data["per_layer"]

    def series(key):
        return [pl[str(li)][key] if str(li) in pl else pl[li][key] for li in layers]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    curves = [("cosine", "mean per-token cosine(δ)", "#c0392b", "o-"),
              ("cka_delta", "linear CKA(δ_LoRA, δ_LoReFT)", "#2c3e50", "s-"),
              ("subspace_overlap", f"top-{data['k']} subspace overlap(δ)", "#27ae60", "^-"),
              ("cka_h", "linear CKA(h_LoRA, h_LoReFT)", "#8e44ad", "d--")]
    for key, label, color, style in curves:
        ax.plot(layers, series(key), style, color=color, lw=2.0, ms=7, label=label)

    ax.set_xlabel("decoder layer (probe subset)")
    ax.set_ylabel("similarity")
    ax.set_ylim(-0.15, 1.05)
    ax.set_xticks(layers)
    ax.axhline(0.0, color="gray", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_title(f"Do LoRA and LoReFT make the same commonsense edit?  "
                 f"(n={data['n_prompts']} prompts, f7+l7 positions)\n"
                 f"high ⇒ same representational edit; ~0 ⇒ same behavior, different routes",
                 fontsize=10.5)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path("results/figures") / "lora_vs_loreft_similarity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
