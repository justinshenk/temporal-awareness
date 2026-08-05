"""Interval estimates for the context-fatigue nulls, and the calibration gap.

The extended abstract's central claims are *nulls* (accuracy flat as context fills, adherence
flat) reported as point estimates. A null stated without an interval invites the standard
"underpowered" objection, so this module turns each into a bounded claim: not "we found no
decay" but "a decay larger than X is excluded at 95%".

Three analyses, all re-using committed per-case data (no model calls):

* :func:`fill_slope_stats` — accuracy vs context fill (``results/random_context/turns.csv``),
  point-biserial ``r`` with a Fisher-z interval plus a bootstrap interval on the
  high-minus-low fill accuracy difference. The degradation-side interval limit is the
  equivalence bound the paper can assert.
* :func:`attention_inversion_stats` — the per-case inversion of Table~2
  (``results/olmo_attention_{model}/attention_stats.csv``): wrong-minus-correct current-query
  attention per fill bin, with bootstrap intervals, so the claim is not resting on one bin.
* :func:`calibration_by_fill` — the confidently-wrong gap the abstract names as the real
  hazard: per-case confidence against correctness as context fills. Only the DDXPlus MCQ
  streams carry confidence *and* correctness *and* fill, so n is small and reported per stream
  as well as pooled.

Bootstrap intervals are seeded (:data:`SEED`) so every reported number is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.common.base_schema import BaseSchema

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results"
DATA = REPO_ROOT / "data" / "context_fatigue"

SEED = 42
N_BOOT = 10000

# Quarter-width fill bins, matching paper_figures.FILL_EDGES / the Table 2 binning.
FILL_EDGES = [0.0, 0.25, 0.50, 0.75, 1.01]
FILL_LABELS = ["0-25%", "25-50%", "50-75%", "75-100%"]

# DDXPlus MCQ streams carrying per-case confidence, correctness and fill.
MCQ_STREAMS = {
    "llama8b": DATA / "ddxplus_mcq_llama8b" / "ddxplus_mcq_turns.csv",
    "llama8b_reversed": DATA / "ddxplus_mcq_llama8b_reversed" / "ddxplus_mcq_turns.csv",
    "qwen7b_reversed": DATA / "ddxplus_mcq_qwen7b_reversed" / "ddxplus_mcq_turns.csv",
    "qwen7b_verbose": DATA / "ddxplus_mcq_verbose" / "ddxplus_mcq_turns.csv",
}


@dataclass
class Interval(BaseSchema):
    """A point estimate with a 95% interval and the n it rests on."""

    estimate: float
    lo: float
    hi: float
    n: int

    def excludes_zero(self) -> bool:
        return (self.lo > 0) or (self.hi < 0)

    def render(self, digits: int = 3) -> str:
        return f"{self.estimate:+.{digits}f} [{self.lo:+.{digits}f}, {self.hi:+.{digits}f}] (n={self.n})"


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def bootstrap_interval(values: np.ndarray, statistic, alpha: float = 0.05) -> Interval:
    """Percentile bootstrap interval for ``statistic`` over rows of ``values``."""
    rng = _rng()
    n = len(values)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        draws[b] = statistic(values[idx])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(statistic(values)), float(lo), float(hi), n)


def fisher_z_interval(r: float, n: int, alpha: float = 0.05) -> Interval:
    """Closed-form Fisher-z interval for a correlation coefficient."""
    if n < 4:
        return Interval(r, float("nan"), float("nan"), n)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return Interval(r, float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se)), n)


def _diff_high_low(arr: np.ndarray, split: float = 0.5) -> float:
    """Accuracy in the upper fill half minus the lower half (columns: fill, correct)."""
    fill, correct = arr[:, 0], arr[:, 1]
    hi, lo = correct[fill >= split], correct[fill < split]
    if len(hi) == 0 or len(lo) == 0:
        return float("nan")
    return float(hi.mean() - lo.mean())


def fill_slope_stats(path: Path | None = None) -> dict[str, dict]:
    """Accuracy-vs-fill interval estimates per stream mode (random / coherent).

    Returns, per mode: the point-biserial correlation with a Fisher-z interval, the
    upper-minus-lower-half accuracy difference with a bootstrap interval, and the resulting
    equivalence bound — the largest *decline* (in accuracy points across the context) that the
    data still permit at 95%.
    """
    df = pd.read_csv(path or (RESULTS / "random_context" / "turns.csv"))
    out: dict[str, dict] = {}
    for mode, g in df.groupby("mode"):
        fill = g["context_fill"].to_numpy(float)
        correct = g["correct"].to_numpy(float)
        r = float(np.corrcoef(fill, correct)[0, 1])
        r_int = fisher_z_interval(r, len(g))
        diff = bootstrap_interval(np.column_stack([fill, correct]), _diff_high_low)
        out[str(mode)] = {
            "n": int(len(g)),
            "accuracy": float(correct.mean()),
            "corr_fill": r_int.to_dict(),
            "diff_high_minus_low": diff.to_dict(),
            # the worst decline compatible with the data: the negative end of the interval
            "max_decline_excluded_at_95": float(-diff.lo),
            "flat_supported": bool(not diff.excludes_zero() or diff.estimate > 0),
        }
    return out


def final_bin_stats(path: Path | None = None) -> dict[str, dict]:
    """Is the last-bin dip real? Top-fill-bin accuracy minus the rest, with an interval.

    The abstract restricts its flat claim to the first ~80% of context because the top bin
    dips in both streams; this quantifies whether that dip survives its own small n.
    """
    df = pd.read_csv(path or (RESULTS / "random_context" / "turns.csv"))
    out: dict[str, dict] = {}
    for mode, g in df.groupby("mode"):
        fill = g["context_fill"].to_numpy(float)
        correct = g["correct"].to_numpy(float)
        arr = np.column_stack([fill, correct])
        diff = bootstrap_interval(arr, lambda a: _diff_high_low(a, split=0.8))
        top = correct[fill >= 0.8]
        out[str(mode)] = {
            "n_top_bin": int(len(top)),
            "accuracy_top_bin": float(top.mean()) if len(top) else float("nan"),
            "accuracy_rest": float(correct[fill < 0.8].mean()),
            "diff_top_minus_rest": diff.to_dict(),
            "significant": bool(diff.excludes_zero()),
        }
    return out


def _per_case_attention(model: str, layer: int) -> pd.DataFrame:
    """One row per (session, case) at ``layer``: head-averaged attention + correctness."""
    df = pd.read_csv(RESULTS / f"olmo_attention_{model}" / "attention_stats.csv")
    layer_df = df[df["layer"] == layer]
    per_case = (
        layer_df.groupby(["session", "case"])
        .agg(
            context_fill=("context_fill", "first"),
            correct=("correct", "first"),
            frac_current_query=("frac_current_query", "mean"),
            attention_entropy=("attention_entropy", "mean"),
        )
        .reset_index()
    )
    per_case["fill_bin"] = pd.cut(
        per_case["context_fill"], bins=FILL_EDGES, labels=FILL_LABELS, include_lowest=True
    )
    return per_case


def attention_inversion_stats(model: str = "instruct", layer: int = 24) -> dict:
    """Table 2 with intervals: wrong-minus-correct current-query attention, per fill bin.

    Heads are averaged within a case first, so n is the number of cases (not cases x heads) and
    the interval is not inflated by head-level pseudo-replication.
    """
    per_case = _per_case_attention(model, layer)

    def delta(arr: np.ndarray) -> float:
        correct, value = arr[:, 0], arr[:, 1]
        wrong_v, right_v = value[correct == 0], value[correct == 1]
        if len(wrong_v) == 0 or len(right_v) == 0:
            return float("nan")
        return float(wrong_v.mean() - right_v.mean())

    bins: dict[str, dict] = {}
    for label in FILL_LABELS:
        g = per_case[per_case["fill_bin"] == label]
        if g.empty:
            continue
        arr = np.column_stack([g["correct"].to_numpy(float), g["frac_current_query"].to_numpy(float)])
        interval = bootstrap_interval(arr, delta)
        bins[label] = {
            "n_cases": int(len(g)),
            "n_wrong": int((g["correct"] == 0).sum()),
            "n_correct": int((g["correct"] == 1).sum()),
            "mean_correct": float(g.loc[g["correct"] == 1, "frac_current_query"].mean()),
            "mean_wrong": float(g.loc[g["correct"] == 0, "frac_current_query"].mean()),
            "delta_wrong_minus_correct": interval.to_dict(),
            "significant": bool(interval.excludes_zero()),
        }

    pooled_arr = np.column_stack(
        [per_case["correct"].to_numpy(float), per_case["frac_current_query"].to_numpy(float)]
    )
    pooled = bootstrap_interval(pooled_arr, delta)
    return {
        "model": model,
        "layer": layer,
        "bins": bins,
        "pooled_delta": pooled.to_dict(),
        "pooled_significant": bool(pooled.excludes_zero()),
        "n_significant_bins": sum(b["significant"] for b in bins.values()),
    }


def layer_generality(model: str = "instruct", layers: tuple[int, ...] = (8, 16, 24, 31)) -> dict:
    """Pooled inversion delta at several layers — is layer 24 special or representative?"""
    out = {}
    for layer in layers:
        try:
            stats_at_layer = attention_inversion_stats(model, layer)
        except (KeyError, ValueError):
            continue
        if stats_at_layer["bins"]:
            out[str(layer)] = {
                "pooled_delta": stats_at_layer["pooled_delta"],
                "significant": stats_at_layer["pooled_significant"],
            }
    return out


def load_mcq_streams() -> pd.DataFrame:
    """Pooled DDXPlus MCQ cases with per-case confidence, correctness and fill."""
    frames = []
    for name, path in MCQ_STREAMS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["stream"] = name
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    # exp(mean per-token logprob): the geometric-mean token probability of the emitted answer
    pooled["confidence"] = np.exp(pooled["mean_logprob"].to_numpy(float))
    pooled["fill_bin"] = pd.cut(
        pooled["context_fill"], bins=FILL_EDGES, labels=FILL_LABELS, include_lowest=True
    )
    return pooled


def calibration_by_fill() -> dict:
    """The confidently-wrong gap: confidence minus accuracy as context fills.

    Reports per fill bin the mean confidence, the accuracy, their gap (the calibration error
    with sign), and the mean confidence *on wrong answers* — the quantity the abstract calls
    the hazard. Small n is intrinsic here (only the MCQ streams carry all three fields), so
    both per-stream and pooled numbers are returned.
    """
    df = load_mcq_streams()
    bins: dict[str, dict] = {}
    for label in FILL_LABELS:
        g = df[df["fill_bin"] == label]
        if g.empty:
            continue
        conf = g["confidence"].to_numpy(float)
        correct = g["correct"].to_numpy(float)
        wrong_conf = conf[correct == 0]
        bins[label] = {
            "n": int(len(g)),
            "n_wrong": int((correct == 0).sum()),
            "accuracy": float(correct.mean()),
            "confidence": float(conf.mean()),
            "gap_confidence_minus_accuracy": float(conf.mean() - correct.mean()),
            "confidence_on_wrong": (
                bootstrap_interval(wrong_conf.reshape(-1, 1), lambda a: float(a.mean())).to_dict()
                if len(wrong_conf) > 1
                else None
            ),
            "mean_entropy": float(g["mean_entropy"].mean()),
        }

    fill = df["context_fill"].to_numpy(float)
    gap_trend = fisher_z_interval(
        float(np.corrcoef(fill, df["confidence"].to_numpy(float))[0, 1]), len(df)
    )
    wrong = df[df["correct"] == 0]
    wrong_trend = fisher_z_interval(
        float(np.corrcoef(wrong["context_fill"].to_numpy(float),
                          wrong["confidence"].to_numpy(float))[0, 1]),
        len(wrong),
    )
    return {
        "n_total": int(len(df)),
        "streams": {s: int(n) for s, n in df["stream"].value_counts().items()},
        "bins": bins,
        "corr_confidence_fill": gap_trend.to_dict(),
        "corr_confidence_fill_wrong_only": wrong_trend.to_dict(),
    }
