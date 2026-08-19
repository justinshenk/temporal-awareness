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
    drain_shape,
    evidence_head_profile,
    head_concentration,
    paired_head_contrasts,
    redistribution_test,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
E1_HEADS = REPO_ROOT / "results" / "context_fatigue" / "e1_heads_all" / "heads.csv"
E1_TURNS = REPO_ROOT / "results" / "context_fatigue" / "e1_heads_all" / "turns.csv"
E3_HEADS = REPO_ROOT / "results" / "context_fatigue" / "e3_heads_all" / "heads.csv"
REFERENCE_LAYER = 24
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
    shape = drain_shape(local.to_numpy(), back.to_numpy())
    out["drain"] = {
        "mean": float(drain.mean()),
        "n_heads_losing": int((drain > 0).sum()),
        "n_heads_gaining": int((drain < 0).sum()),
        "fractional_drain_min": float((drain / local).min()),
        "fractional_drain_max": float((drain / local).max()),
        # Level vs *absolute* drain is ~1 by arithmetic when every head loses a similar fraction,
        # so it is deliberately not reported: the fractional correlation inside `shape` is the
        # one that says whether high-mass heads are special.
        **shape.to_dict(),
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


def sweep_layers(e1: pd.DataFrame, e3: pd.DataFrame, span_fraction: float) -> dict:
    """Per-layer head structure, so no conclusion is scoped to one slice of the model.

    A head that specializes in the evidence is what would make the head-averaged share a blunt
    summary, and there is no reason to expect it at the reference layer specifically -- the extra
    layers ride on the same forward, so restricting to one was never a measurement decision.
    """
    rows, best = [], []
    for layer in sorted(e1["layer"].unique()):
        le1 = e1[e1["layer"] == layer]
        prof = evidence_head_profile(le1, "local", span_fraction)
        local = le1[le1.arm == "local"].groupby("head")["evidence_share"].mean().sort_index()
        back = le1[le1.arm == "back_20"].groupby("head")["evidence_share"].mean().sort_index()
        shape = drain_shape(local.to_numpy(), back.to_numpy())
        redis = redistribution_test(e3[e3["layer"] == layer], "random", "near_dup")
        rows.append({
            "layer": int(layer),
            "max_enrichment": float(prof["enrichment"].max()),
            "n_heads_enriched": int((prof["enrichment"] > 1).sum()),
            "mean_evidence_share": float(prof["evidence_share"].mean()),
            "effective_heads": head_concentration(
                prof["evidence_share"].to_numpy()).effective_heads,
            "fractional_drain_mean": shape.fractional_drain_mean,
            "uniform_odds_r2": shape.r2,
            "competition_mean_delta": redis.mean_delta,
            "competition_mean_abs_delta": redis.mean_abs_delta,
            "competition_ratio": redis.redistribution_ratio,
            "competition_p": redis.p_value,
        })
        for head, r in prof.iterrows():
            best.append({"layer": int(layer), "head": int(head),
                         "evidence_share": float(r["evidence_share"]),
                         "question_share": float(r["question_share"]),
                         "enrichment": float(r["enrichment"]),
                         "ev_over_q": float(r["ev_over_q"])})
    best.sort(key=lambda r: -r["enrichment"])
    return {"per_layer": rows, "top_heads_global": best[:15],
            "n_heads_total": len(best),
            "n_heads_enriched_global": sum(1 for r in best if r["enrichment"] > 1)}


def main() -> None:
    e1_all = _balanced(_require(E1_HEADS), E1_ARMS)
    e3_all = _balanced(_require(E3_HEADS), ["disjoint", "random", "near_dup"])
    e1 = e1_all[e1_all["layer"] == REFERENCE_LAYER]
    e3 = e3_all[e3_all["layer"] == REFERENCE_LAYER]

    turns = pd.read_csv(E1_TURNS)
    span_fraction = float((turns["evidence_tokens"] / turns["ctx_tokens"]).mean())
    profile = evidence_head_profile(e1, "local", span_fraction).reset_index()

    payload = {
        "span_fraction": span_fraction,
        "n_heads_enriched": int((profile["enrichment"] > 1).sum()),
        "evidence_heads": profile.round(6).to_dict("records"),
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
        "reference_layer": REFERENCE_LAYER,
        "layers": sweep_layers(e1_all, e3_all, span_fraction),
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
    print(f"  fractional drain {dr['fractional_drain_mean']:.1%} +- {dr['fractional_drain_sd']:.1%}, "
          f"spanning {dr['fractional_drain_min']:.1%} to {dr['fractional_drain_max']:.1%}")
    print(f"  corr(level, FRACTIONAL drain) = {dr['corr_level_with_fractional_drain']:+.3f} "
          f"-- high-mass heads are not special")
    print(f"  one uniform odds-scale ({dr['best_bias_nats']:+.3f} nats) reproduces the per-head "
          f"pattern at R2={dr['r2']:.3f}")
    print(f"  per-head implied bias sd {dr['implied_bias_sd']:.3f} nats, range "
          f"{dr['implied_bias_min']:+.2f} to {dr['implied_bias_max']:+.2f}")

    prof = payload["evidence_heads"]
    print(f"\n  evidence span is {payload['span_fraction']:.1%} of context; a uniform head scores "
          f"{payload['span_fraction']:.4f}")
    print(f"  heads with enrichment > 1 (concentrating on the evidence): "
          f"{payload['n_heads_enriched']}/32")
    print("  top 5 by evidence share at local:")
    for row in prof[:5]:
        print(f"    head {row['head']:2d}  ev {row['evidence_share']:.4f}  q {row['question_share']:.4f}  "
              f"ev/q {row['ev_over_q']:.2f}  enrich {row['enrichment']:.2f}")

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
    ls = payload["layers"]
    print(f"\nAll {ls['n_heads_total']} heads across {len(ls['per_layer'])} layers")
    print(f"  heads concentrating on the evidence (enrichment > 1): "
          f"{ls['n_heads_enriched_global']}/{ls['n_heads_total']}")
    print("  top 10 by enrichment, anywhere in the model:")
    for r in ls["top_heads_global"][:10]:
        print(f"    L{r['layer']:2d}H{r['head']:2d}  enrich {r['enrichment']:5.2f}  "
              f"ev {r['evidence_share']:.4f}  q {r['question_share']:.4f}  "
              f"ev/q {r['ev_over_q']:6.2f}")
    print("\n  per-layer (evidence share, drain, competition reallocation):")
    print("   layer  meanEv  effHeads  fracDrain  unifR2   compMeanD  compMeanAbsD  ratio")
    for r in ls["per_layer"]:
        print(f"    {r['layer']:3d}  {r['mean_evidence_share']:.4f}  {r['effective_heads']:7.2f}  "
              f"{r['fractional_drain_mean']:8.1%}  {r['uniform_odds_r2']:6.3f}  "
              f"{r['competition_mean_delta']:+.5f}  {r['competition_mean_abs_delta']:11.5f}  "
              f"{r['competition_ratio']:6.2f}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
