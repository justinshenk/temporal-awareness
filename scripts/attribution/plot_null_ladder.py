"""F3 — the procedure-null ladder as a dot-and-interval range plot against the oracle.

Reads ``null_bounds.json`` (from ``bound_procedure_nulls``): each rung's recovery point estimate
with its exact-binomial 95% interval, drawn as a dot at the estimate and a whisker to the upper
bound. The lockstep oracle is a dashed reference line — the positive control every rung is bounded
against. DAS appears once (largest rank); the per-rank rows are identical bounds.

    uv run python -m scripts.attribution.plot_null_ladder \
        --json results/attribution/null_bounds.json \
        --out papers/register_vs_procedure/paper/figures/f3_null_ladder.pdf
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1f2430"
ACCENT = "#3b6fb5"
GRID = "#d5d8de"


def dedupe_das(rows: list[dict]) -> list[dict]:
    """Keep one DAS row — the largest rank — since every rank shares the same bound."""
    das = [r for r in rows if r["rung"].startswith("DAS")]
    if not das:
        return list(rows)
    def rank(r: dict) -> int:
        m = re.search(r"r=(\d+)", r["rung"])
        return int(m.group(1)) if m else 0
    keep = max(das, key=rank)
    return [r for r in rows if not r["rung"].startswith("DAS")] + [keep]


def render_ladder(json_path: Path, out: Path, oracle: float,
                  oracle_label: str = "lockstep oracle") -> None:
    rows = dedupe_das(json.loads(Path(json_path).read_text()))
    if not rows:
        raise ValueError(f"no rungs in {json_path}")

    fig, ax = plt.subplots(figsize=(5.4, 0.55 * len(rows) + 0.9))
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        ax.plot([r["recovery_lo"], r["recovery_hi"]], [y, y], color=INK, lw=2,
                solid_capstyle="butt", zorder=3)
        ax.plot([r["recovery"]], [y], "o", color=INK, ms=6, zorder=4)
        ax.annotate(f"n={r['n']}", (r["recovery_hi"], y), textcoords="offset points",
                    xytext=(6, -3), fontsize=7.5, color="#6a7080")

    ax.axvline(oracle, color=ACCENT, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.text(oracle + 0.015, 0.12, f"{oracle_label} = {oracle:.2f}",
            fontsize=8, color=ACCENT, ha="left", va="center")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["rung"] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("recovery of donor budget (95% bound)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=Path("results/attribution/null_bounds.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("papers/register_vs_procedure/paper/figures/f3_null_ladder.pdf"))
    ap.add_argument("--oracle", type=float, default=0.75)
    ap.add_argument("--oracle-label", default="lockstep oracle @L20")
    args = ap.parse_args()
    render_ladder(args.json, args.out, args.oracle, args.oracle_label)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
