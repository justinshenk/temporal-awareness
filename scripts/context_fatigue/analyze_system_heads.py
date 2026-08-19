"""Which heads read the system prompt, and what accumulation does to them.

The paper reports an instruction-adherence null: a canary system instruction stays at ceiling as
context fills. That null is a floor effect on an easy check, and it says nothing about whether the
mechanism that *would* degrade compliance is intact. This asks the mechanistic question directly —
which of the 1,024 heads hold attention on the system prompt at cold start, and how much of that
attention survives a full context.

Also profiles the current-query span, so the same heads can be compared with and without context.

Pure re-analysis of `results/olmo_attention_all/attention_stats.csv`. No GPU.

    uv run python scripts/context_fatigue/analyze_system_heads.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.probes.context_fatigue.head_analysis import span_profile_by_fill

REPO_ROOT = Path(__file__).resolve().parents[2]
STATS = REPO_ROOT / "results" / "olmo_attention_all" / "attention_stats.csv"
OUT_JSON = REPO_ROOT / "results" / "context_fatigue" / "system_heads.json"
SPANS = ["frac_system", "frac_current_query", "frac_recent_cases"]


def main() -> None:
    if not STATS.exists():
        raise FileNotFoundError(
            f"missing {STATS}. Run run_olmo_attention.py with --layers 0,1,...,31 first; the "
            f"committed 5-layer artifact cannot answer a whole-model question.")
    df = pd.read_csv(STATS)
    payload = {"n_layers": int(df["layer"].nunique()), "n_heads_total": int(len(
        df[["layer", "head"]].drop_duplicates())), "fill_range": [float(df.context_fill.min()),
                                                                  float(df.context_fill.max())]}

    for span in SPANS:
        prof = span_profile_by_fill(df, span)
        payload[span] = {
            "total_cold": float(prof["cold"].mean()),
            "total_full": float(prof["full"].mean()),
            "biggest_losses": prof.head(12).round(5).to_dict("records"),
            "n_heads_losing_over_0p05": int((prof["delta"] < -0.05).sum()),
            "n_heads_losing_over_0p10": int((prof["delta"] < -0.10).sum()),
        }
        # The heads that *held* the span at cold start, whether or not they moved most.
        holders = prof.sort_values("cold", ascending=False).head(12)
        payload[span]["biggest_holders_cold"] = holders.round(5).to_dict("records")

    OUT_JSON.write_text(json.dumps(payload, indent=2))

    print(f"{payload['n_heads_total']} heads over {payload['n_layers']} layers | "
          f"fill {payload['fill_range'][0]:.2f}-{payload['fill_range'][1]:.2f}")
    for span in SPANS:
        p = payload[span]
        print(f"\n=== {span} ===")
        print(f"  mean over all heads: cold {p['total_cold']:.4f} -> full {p['total_full']:.4f} "
              f"(change {p['total_full'] - p['total_cold']:+.4f})")
        print(f"  heads losing more than 0.05 absolute: {p['n_heads_losing_over_0p05']}; "
              f"more than 0.10: {p['n_heads_losing_over_0p10']}")
        print("  heads holding the most at cold start:")
        for r in p["biggest_holders_cold"][:8]:
            print(f"    L{int(r['layer']):2d}H{int(r['head']):2d}  cold {r['cold']:.4f} -> "
                  f"full {r['full']:.4f}   change {r['delta']:+.4f}   r(fill) "
                  f"{r['corr_with_fill']:+.2f}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
