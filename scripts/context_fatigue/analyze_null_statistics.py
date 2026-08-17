"""Interval estimates for the context-fatigue nulls + the calibration-gap figure.

Turns the extended abstract's point-estimate nulls into bounded claims and measures the
confidently-wrong gap it names as the real hazard. Pure re-analysis of committed per-case data
— no model calls, no GPU.

    uv run python scripts/context_fatigue/analyze_null_statistics.py

Writes ``results/context_fatigue/null_statistics.json``, a companion markdown summary, and
``context_fatigue_paper/figures/calibration_gap.pdf``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.probes.context_fatigue.null_statistics import (
    FILL_LABELS,
    attention_inversion_stats,
    calibration_by_fill,
    fill_slope_stats,
    final_bin_stats,
    layer_generality,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# The pooled stream (original 12 sessions at fill<=0.88 + the 2026-08-17 top-bin batch: 15
# sessions/mode at fill<=0.93, disjoint seeds, overflow-guarded). Falls back to the original
# artifact when the pooled file is absent so the script stays runnable from a fresh clone.
POOLED_TURNS = REPO_ROOT / "results" / "random_context_topbin" / "turns_pooled.csv"
TURNS = POOLED_TURNS if POOLED_TURNS.exists() else None
OUT_JSON = REPO_ROOT / "results" / "context_fatigue" / "null_statistics.json"
OUT_MD = REPO_ROOT / "results" / "context_fatigue" / "NULL_STATISTICS.md"
FIG_PATH = REPO_ROOT / "context_fatigue_paper" / "figures" / "calibration_gap.pdf"


def render_markdown(payload: dict) -> str:
    """A reviewer-facing summary of every interval, in the order the paper needs them."""
    lines = ["# Interval estimates for the context-fatigue nulls", ""]

    lines += ["## 1. Accuracy vs context fill (the null, with power)", ""]
    lines += ["| stream | n | accuracy | corr(correct, fill) 95% CI | upper-half minus lower-half | decline excluded above |",
              "|---|---:|---:|---|---|---:|"]
    for mode, s in payload["fill_slope"].items():
        r, d = s["corr_fill"], s["diff_high_minus_low"]
        lines.append(
            f"| {mode} | {s['n']} | {s['accuracy']:.3f} | "
            f"{r['estimate']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
            f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | "
            f"{s['max_decline_excluded_at_95']:.3f} |"
        )
    lines += ["", "The final column is the equivalence bound: accuracy declines larger than this "
                  "(in proportion points, upper vs lower half of the context) are excluded at 95%.", ""]

    if "fill_slope_le80" in payload:
        lines += ["### Scoped to the first 80% of context (the paper's flat claim)", ""]
        lines += ["| stream | n | accuracy | corr(correct, fill) 95% CI | upper-half minus lower-half | decline excluded above |",
                  "|---|---:|---:|---|---|---:|"]
        for mode, s in payload["fill_slope_le80"].items():
            r, d = s["corr_fill"], s["diff_high_minus_low"]
            lines.append(
                f"| {mode} | {s['n']} | {s['accuracy']:.3f} | "
                f"{r['estimate']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | "
                f"{s['max_decline_excluded_at_95']:.3f} |"
            )
        lines += ["", "The flat claim is scoped to fill < 0.8; the ≥0.8 region is quantified in "
                      "§2 below, where the random-stream dip is a real effect, not a bound.", ""]

    lines += ["## 2. The top-fill-bin dip", ""]
    lines += ["| stream | n top bin | acc top bin | acc rest | difference 95% CI | significant |",
              "|---|---:|---:|---:|---|---|"]
    for mode, s in payload["final_bin"].items():
        d = s["diff_top_minus_rest"]
        lines.append(
            f"| {mode} | {s['n_top_bin']} | {s['accuracy_top_bin']:.3f} | {s['accuracy_rest']:.3f} | "
            f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {'yes' if s['significant'] else 'no'} |"
        )
    lines.append("")

    inv = payload["attention_inversion"]
    lines += [f"## 3. Per-case inversion with intervals ({inv['model']}, layer {inv['layer']})", ""]
    lines += ["| fill bin | n cases | correct | wrong | delta (wrong-correct) 95% CI | significant |",
              "|---|---:|---:|---:|---|---|"]
    for label in FILL_LABELS:
        b = inv["bins"].get(label)
        if not b:
            continue
        d = b["delta_wrong_minus_correct"]
        lines.append(
            f"| {label} | {b['n_cases']} | {b['mean_correct']:.3f} | {b['mean_wrong']:.3f} | "
            f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {'yes' if b['significant'] else 'no'} |"
        )
    p = inv["pooled_delta"]
    lines += [f"| **pooled** | {p['n']} | | | **{p['estimate']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]** | "
              f"{'yes' if inv['pooled_significant'] else 'no'} |", ""]
    lines += [f"{inv['n_significant_bins']} of {len(inv['bins'])} individual bins reach significance; "
              "the pooled estimate is the claim to lead with.", ""]

    lines += ["### Layer generality", "", "| layer | pooled delta 95% CI | significant |", "|---|---|---|"]
    for layer, s in payload["layer_generality"].items():
        d = s["pooled_delta"]
        lines.append(f"| {layer} | {d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | "
                     f"{'yes' if s['significant'] else 'no'} |")
    lines.append("")

    cal = payload["calibration"]
    lines += ["## 4. The confidently-wrong gap", "",
              f"Pooled DDXPlus MCQ streams, n={cal['n_total']} cases "
              f"({', '.join(f'{k}={v}' for k, v in cal['streams'].items())}).", ""]
    lines += ["| fill bin | n | accuracy | confidence | gap | confidence on wrong answers 95% CI |",
              "|---|---:|---:|---:|---:|---|"]
    for label in FILL_LABELS:
        b = cal["bins"].get(label)
        if not b:
            continue
        cw = b["confidence_on_wrong"]
        cw_txt = (f"{cw['estimate']:.3f} [{cw['lo']:.3f}, {cw['hi']:.3f}] (n={cw['n']})"
                  if cw else "n/a")
        lines.append(
            f"| {label} | {b['n']} | {b['accuracy']:.3f} | {b['confidence']:.3f} | "
            f"{b['gap_confidence_minus_accuracy']:+.3f} | {cw_txt} |"
        )
    c1, c2 = cal["corr_confidence_fill"], cal["corr_confidence_fill_wrong_only"]
    lines += ["",
              f"corr(confidence, fill) = {c1['estimate']:+.3f} [{c1['lo']:+.3f}, {c1['hi']:+.3f}], n={c1['n']}",
              "",
              f"corr(confidence, fill) on **wrong answers only** = {c2['estimate']:+.3f} "
              f"[{c2['lo']:+.3f}, {c2['hi']:+.3f}], n={c2['n']}",
              ""]
    return "\n".join(lines)


def make_calibration_figure(cal: dict, path: Path) -> None:
    """Accuracy vs confidence by fill bin — the gap is the confidently-wrong hazard."""
    labels = [lab for lab in FILL_LABELS if lab in cal["bins"]]
    acc = [cal["bins"][lab]["accuracy"] for lab in labels]
    conf = [cal["bins"][lab]["confidence"] for lab in labels]
    wrong = [cal["bins"][lab]["confidence_on_wrong"] for lab in labels]

    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    x = range(len(labels))
    ax.plot(x, conf, "o-", color="#c44e52", label="confidence")
    ax.plot(x, acc, "s-", color="#4c72b0", label="accuracy")
    if all(w is not None for w in wrong):
        est = [w["estimate"] for w in wrong]
        lo = [w["estimate"] - w["lo"] for w in wrong]
        hi = [w["hi"] - w["estimate"] for w in wrong]
        ax.errorbar(x, est, yerr=[lo, hi], fmt="^--", color="#dd8452", capsize=3,
                    label="confidence | wrong")
    ax.fill_between(x, acc, conf, color="#c44e52", alpha=0.12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("context fill")
    ax.set_ylabel("probability")
    ax.set_title("The confidently-wrong gap widens", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    payload = {
        "fill_slope": fill_slope_stats(TURNS),
        "fill_slope_le80": fill_slope_stats(TURNS, max_fill=0.8),
        "final_bin": final_bin_stats(TURNS),
        "attention_inversion": attention_inversion_stats(),
        "layer_generality": layer_generality(),
        "calibration": calibration_by_fill(),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    OUT_MD.write_text(render_markdown(payload))
    make_calibration_figure(payload["calibration"], FIG_PATH)

    print(render_markdown(payload))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}\nwrote {FIG_PATH}")


if __name__ == "__main__":
    main()
