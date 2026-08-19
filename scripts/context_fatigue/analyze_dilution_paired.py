"""Recompute every paired contrast in the dilution program with the correct estimator.

Why this exists
---------------
E1c/E1d/E1e/E1f all measure the **same item** under several conditions, and their reports say so
("paired n = 174"). But the intervals were produced by :func:`arm_accuracy_gap`, which resamples
the two arms *independently*. That estimator is right for independent arms and wrong here: it
charges the interval for between-item difficulty variance that cancels inside a within-item
contrast, inflating the CI by roughly 2.5x on this data.

The correction is not cosmetic. Under the paired estimator E1d's necessity contrast
(``back_20_clamped - back_20``) moves from "not significant" to significant, which changes that
experiment's verdict, and 6 of E1f's 7 dose contrasts become significant.

Reads only committed artifacts; writes ``results/context_fatigue/dilution_paired.json`` so the
corrected numbers in E1_MECHANISM.md each trace to a file.
"""

import json
from pathlib import Path

import pandas as pd

from src.probes.context_fatigue.dilution_analysis import arm_accuracy_gap, paired_accuracy_gap

RESULTS = Path(__file__).resolve().parents[2] / "results" / "context_fatigue"

E1F_LEVELS = ["0.036", "0.032", "0.029", "0.025", "0.020", "0.016", "0.012"]


def _item_key(df: pd.DataFrame) -> pd.Series:
    """A within-run item identifier. E1c/E1d key on (session, depth, pathology); E1e/E1f on probe."""
    if "probe" in df.columns:
        return df["probe"]
    cols = ["session", "filler_turns", "pathology"]
    return df[cols].astype(str).agg("_".join, axis=1)


def _pivot(path: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.assign(item=_item_key(df))
    return df.pivot_table(index="item", columns=column, values="correct").dropna()


def _contrast(piv: pd.DataFrame, a: str, b: str) -> dict:
    paired = paired_accuracy_gap(piv[a], piv[b])
    unpaired = arm_accuracy_gap(piv[a], piv[b])
    return {
        "contrast": f"{a} - {b}", "n": int(len(piv)),
        "estimate": round(paired.estimate, 4),
        "lo": round(paired.lo, 4), "hi": round(paired.hi, 4),
        "significant": paired.excludes_zero(),
        "unpaired_lo": round(unpaired.lo, 4), "unpaired_hi": round(unpaired.hi, 4),
        "unpaired_significant": unpaired.excludes_zero(),
    }


def main() -> None:
    out: dict[str, list[dict]] = {}

    e1c = _pivot(RESULTS / "e1c_evidence_clamp" / "turns.csv", "condition")
    out["e1c_sufficiency"] = [_contrast(e1c, "local", "local_clamped"),
                              _contrast(e1c, "local_clamped", "back_20"),
                              _contrast(e1c, "local", "back_20")]

    e1d = _pivot(RESULTS / "e1d_evidence_rescue" / "turns.csv", "condition")
    out["e1d_necessity"] = [_contrast(e1d, "back_20_clamped", "back_20"),
                            _contrast(e1d, "local", "back_20_clamped"),
                            _contrast(e1d, "local", "back_20")]

    e1e = _pivot(RESULTS / "e1e_dissociation" / "turns.csv", "arm")
    out["e1e_dissociation"] = [_contrast(e1e, "local", "turns5_short"),
                               _contrast(e1e, "local", "turns5_long"),
                               _contrast(e1e, "local", "turns20_short"),
                               _contrast(e1e, "turns5_short", "turns5_long"),
                               _contrast(e1e, "turns5_long", "turns20_short")]

    # E1f: levels above an item's natural share are unreachable, so only the balanced panel --
    # the items present at *every* level -- is a like-for-like dose-response.
    raw = pd.read_csv(RESULTS / "e1f_share_knee" / "turns.csv")
    counts = raw.groupby("probe")["level"].nunique()
    balanced = raw[raw["probe"].isin(counts[counts == raw["level"].nunique()].index)]
    e1f = balanced.pivot_table(index="probe", columns="level", values="correct").dropna()
    out["e1f_vs_natural"] = [_contrast(e1f, "natural", lv) for lv in E1F_LEVELS]
    order = ["natural", *E1F_LEVELS]
    out["e1f_adjacent"] = [_contrast(e1f, a, b) for a, b in zip(order[:-1], order[1:])]

    path = RESULTS / "dilution_paired.json"
    path.write_text(json.dumps(out, indent=2))

    for block, rows in out.items():
        print(f"\n=== {block} ===")
        for r in rows:
            flag = "SIG" if r["significant"] else "   "
            moved = "  <-- changes verdict" if r["significant"] != r["unpaired_significant"] else ""
            print(f"  {r['contrast']:32s} n={r['n']:4d} {r['estimate']:+.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] {flag}  "
                  f"(unpaired [{r['unpaired_lo']:+.4f},{r['unpaired_hi']:+.4f}]){moved}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
