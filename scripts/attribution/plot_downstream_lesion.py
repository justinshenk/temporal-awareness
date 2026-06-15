"""Plot E3: recovery vs cumulative downstream identity-ablation, patch vs LoRA-natural.

Reads ``downstream_lesion_L{L}.json`` and renders recovery as the top k downstream layers (L+1..N)
are turned into no-ops. The two curves adjudicate compute-vs-communicate: if both collapse together,
the natively-capable LoRA also needs those layers (the answer is NOT sitting at L20 — base+patch
recovery is genuine downstream computation); if the patched curve stays high under heavy ablation, the
answer is already at L20 and base's 21–31 merely transcribe it.

    uv run python -m scripts.attribution.plot_downstream_lesion \
        --json results/attribution/downstream_lesion_L20.json
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
    levels = sorted(data["levels"].items(), key=lambda kv: int(kv[0]))
    ks = [int(k) for k, _ in levels]
    rp = [v["recovery_patch"] for _, v in levels]
    rl = [v["recovery_lora"] for _, v in levels]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ks, rp, "o-", color="#c0392b", lw=2.2, ms=8, label="base + L%d full-δ patch" % L, zorder=3)
    ax.plot(ks, rl, "s--", color="#2c3e50", lw=1.8, ms=7, alpha=0.8, label="LoRA-natural (control)")
    ax.set_xlabel(f"# top downstream layers identity-ablated (of {len(data['downstream_layers'])}: "
                  f"L{data['downstream_layers'][0]}..L{data['downstream_layers'][-1]})")
    ax.set_ylabel(f"GSM8K recovery on the contrast set (n={data['n_contrast']})")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.25)
    for k, r in zip(ks, rp):
        ax.annotate(f"{r:.2f}", (k, r), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#c0392b")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(f"Is base's downstream load-bearing, or a readout? (L{L} full-δ oracle)\n"
                 f"patch and LoRA-natural collapsing together ⇒ 21–31 compute; "
                 f"patch staying flat ⇒ answer already at L{L}",
                 fontsize=10.5)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path("results/figures") / f"downstream_lesion_L{L}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
