"""F2 — single-layer lockstep oracle recovery by layer, tasks overlaid.

Reads one or more ``lockstep_*_single.json`` sweeps and overlays their recovery-vs-layer curves.
The claim the figure carries is the onset/plateau contrast: the register's curve rises at L16 and
saturates, both procedures stay near zero until L20. Layers 28/31 are excluded by default — the
hook overwrites the block output, so L31 is the all-layers control in disguise.

    uv run python -m scripts.attribution.plot_oracle_sweep \
        --json results/attribution/lockstep_commonsense_single.json,\
results/attribution/lockstep_multihop_single.json,\
results/attribution/lockstep_gsm8k_single_sweep.json \
        --labels "commonsense (register)",MuSiQue,GSM8K \
        --out papers/register_vs_procedure/paper/figures/f2_oracle_sweep.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed identity order: register, then the two procedures (never recycled)
SERIES = ["#3b6fb5", "#c26f2e", "#5a5f6e"]
MARKERS = ["o", "s", "^"]
GRID = "#d5d8de"


def load_sweep(path: Path, exclude_degenerate: tuple[int, ...] = ()) -> tuple[list[int], list[float]]:
    d = json.loads(Path(path).read_text())
    items = sorted(((int(k), float(v["recovery"])) for k, v in d["per_layer"].items()
                    if int(k) not in exclude_degenerate))
    return [k for k, _ in items], [v for _, v in items]


def render_sweep(paths: list[Path], labels: list[str], out: Path,
                 exclude_degenerate: tuple[int, ...] = (28, 31)) -> None:
    if len(paths) != len(labels):
        raise ValueError(f"{len(paths)} sweeps but {len(labels)} labels")

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for i, (path, label) in enumerate(zip(paths, labels)):
        layers, rec = load_sweep(path, exclude_degenerate)
        ax.plot(layers, rec, marker=MARKERS[i % len(MARKERS)], ms=5, lw=2,
                color=SERIES[i % len(SERIES)], label=label, zorder=3)

    ax.set_xlabel("layer patched (single-layer lockstep oracle)", fontsize=9)
    ax.set_ylabel("recovery of donor budget", fontsize=9)
    ax.set_ylim(-0.04, 1.04)
    ax.tick_params(labelsize=8)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="comma list of lockstep sweep JSONs")
    ap.add_argument("--labels", required=True, help="comma list of series labels")
    ap.add_argument("--out", type=Path,
                    default=Path("papers/register_vs_procedure/paper/figures/f2_oracle_sweep.pdf"))
    ap.add_argument("--include-degenerate", action="store_true",
                    help="keep layers 28/31 (block-output overwrite)")
    args = ap.parse_args()
    paths = [Path(p) for p in args.json.split(",")]
    labels = args.labels.split(",")
    render_sweep(paths, labels, args.out,
                 exclude_degenerate=() if args.include_degenerate else (28, 31))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
