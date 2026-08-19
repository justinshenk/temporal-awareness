"""Per-head attention analysis for the distance ladder (E1) and the competition sweep (E3).

Two questions the head-averaged shares cannot answer:

1. **Is the drain uniform across heads?** If a few heads carry the evidence and lose most of the
   mass, the averaged share is a blunt summary of what displacement does.
2. **Does competition move heads without moving their mean?** The paper's second mechanism rests
   on the evidence's share not changing between arms. A mean can hold still while heads
   redistribute, and only the unreduced per-head shares can rule that out.

Pure re-analysis of ``heads.csv`` written by the two drivers under ``--per-head``. No GPU.

    uv run python scripts/context_fatigue/analyze_head_structure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.probes.context_fatigue.head_analysis import (
    head_concentration,
    paired_head_contrasts,
    redistribution_test,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
E1_HEADS = REPO_ROOT / "results" / "context_fatigue" / "e1_heads" / "heads.csv"
E3_HEADS = REPO_ROOT / "results" / "context_fatigue" / "e3_heads" / "heads.csv"
OUT_JSON = REPO_ROOT / "results" / "context_fatigue" / "head_structure.json"

E1_ARMS = ["local", "back_2", "back_5", "back_10", "back_20"]
# 32 heads: a per-head interval at 0.05 expects ~1.6 false positives across the family.
BONFERRONI = 0.05 / 32


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Re-run the driver with --per-head; the head columns are dropped by "
            f"span_share() before turns.csv is written, so they cannot be recovered from it.")
    return pd.read_csv(path)


def _balanced(df: pd.DataFrame, arms) -> pd.DataFrame:
    """Probes present in every arm, so per-head contrasts compare one item set."""
    per_probe = df.groupby("probe")["arm"].nunique()
    keep = per_probe[per_probe == len(arms)].index
    return df[df["probe"].isin(keep)]


def analyse_drain(df: pd.DataFrame) -> dict:
    """Concentration per arm, and the per-head local -> back_20 drain."""
    out = {"concentration": {}, "head_mean_share": {}}
    for arm in E1_ARMS:
        per_head = (df[df["arm"] == arm].groupby("head")["evidence_share"].mean()
                    .sort_index().to_numpy())
        out["head_mean_share"][arm] = [round(float(v), 6) for v in per_head]
        out["concentration"][arm] = head_concentration(per_head).to_dict()
    local = pd.Series(out["head_mean_share"]["local"])
    back = pd.Series(out["head_mean_share"]["back_20"])
    drain = local - back
    out["drain"] = {
        "mean": float(drain.mean()),
        "min": float(drain.min()),
        "max": float(drain.max()),
        "n_heads_losing": int((drain > 0).sum()),
        "n_heads_gaining": int((drain < 0).sum()),
        "top4_share_of_total_drain": float(drain.sort_values(ascending=False)[:4].sum()
                                           / drain.sum()),
        "corr_local_share_with_drain": float(local.corr(drain)),
        "fractional_drain_min": float((drain / local).min()),
        "fractional_drain_max": float((drain / local).max()),
    }
    contrasts = paired_head_contrasts(df, "local", "back_20", alpha=BONFERRONI)
    out["drain"]["n_heads_significant_bonferroni"] = sum(c.delta.excludes_zero()
                                                         for c in contrasts)
    out["drain"]["n_heads"] = len(contrasts)
    return out


def analyse_redistribution(df: pd.DataFrame, pairs) -> dict:
    out = {}
    for a, b in pairs:
        loose = redistribution_test(df, a, b)
        strict = redistribution_test(df, a, b, alpha=BONFERRONI)
        row = loose.to_dict()
        row["n_heads_excluding_zero_bonferroni"] = strict.n_heads_excluding_zero
        out[f"{a}-{b}"] = row
    return out


def main() -> None:
    e1 = _balanced(_require(E1_HEADS), E1_ARMS)
    e3 = _balanced(_require(E3_HEADS), ["disjoint", "random", "near_dup"])

    payload = {
        "e1_distance": analyse_drain(e1),
        "e1_redistribution": analyse_redistribution(e1, [("local", "back_20")]),
        "e3_competition": analyse_redistribution(
            e3, [("random", "near_dup"), ("disjoint", "near_dup"), ("random", "disjoint")]),
        "e3_concentration": {
            arm: head_concentration(
                e3[e3["arm"] == arm].groupby("head")["evidence_share"].mean().sort_index()
                .to_numpy()).to_dict()
            for arm in ["disjoint", "random", "near_dup"]
        },
        "n_probes": {"e1": int(e1["probe"].nunique()), "e3": int(e3["probe"].nunique())},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    d = payload["e1_distance"]
    print("E1 — per-head evidence share")
    for arm in E1_ARMS:
        c = d["concentration"][arm]
        print(f"  {arm:9s} effective heads {c['effective_heads']:5.2f}/32   "
              f"top-4 fraction {c['top4_fraction']:.3f}")
    dr = d["drain"]
    print(f"\n  drain local->back_20: {dr['n_heads_losing']}/{dr['n_heads']} heads lose mass, "
          f"{dr['n_heads_significant_bonferroni']} significant at Bonferroni")
    print(f"  top-4 heads carry {dr['top4_share_of_total_drain']:.1%} of the total drain")
    print(f"  corr(local share, drain) = {dr['corr_local_share_with_drain']:+.3f}")
    print(f"  fractional drain per head spans {dr['fractional_drain_min']:.1%} to "
          f"{dr['fractional_drain_max']:.1%}")

    print("\nRedistribution (mean delta vs mean |per-head delta|)")
    for label, block in [("E1 local-back_20", payload["e1_redistribution"]),
                         ("E3", payload["e3_competition"])]:
        for name, r in block.items():
            ratio = r["redistribution_ratio"]
            print(f"  [{label}] {name:22s} mean {r['mean_delta']:+.5f}  "
                  f"mean|d| {r['mean_abs_delta']:.5f} (null {r['null_mean_abs_delta']:.5f}, "
                  f"p={r['p_value']:.4f})  ratio {ratio:6.2f}  "
                  f"heads!=0 {r['n_heads_excluding_zero']:2d} (bonf "
                  f"{r['n_heads_excluding_zero_bonferroni']:2d})/{r['n_heads']}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
