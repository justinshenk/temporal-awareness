"""Extreme and inconsistent discount-rate probe (follow-up to the paper's App. O).

Titrates the delayed amount X at which a model becomes indifferent between an
immediate reward R and X paid after delay d, via forced A/B choice decided by
teacher-forced logprob comparison. At the boundary the hyperbolic rate is
k = (X/R - 1) / d_years. The probe targets two failure modes only:

  1. Extreme rates: no boundary within a 20x cap, k > 0.5/yr, or k < 1e-4/yr.
  2. Inconsistency: magnitude-effect ordering flips across delays, indifference
     points that do not increase with delay, and preference reversals when the
     A/B labels are swapped.

Usage:
    uv run python scripts/discount/extreme_discount_probe.py
    uv run python scripts/discount/extreme_discount_probe.py --stub
    uv run python scripts/discount/extreme_discount_probe.py --models google/gemma-2-9b-it
"""

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformer_lens.loading_from_pretrained import get_official_model_name

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binary_choice import BinaryChoiceRunner  # noqa: E402
from src.inference.backends import ModelBackend  # noqa: E402

DELAYS = [
    ("1 day", 1.0 / 365.25),
    ("1 week", 7.0 / 365.25),
    ("1 month", 30.44 / 365.25),
    ("6 months", 182.6 / 365.25),
    ("1 year", 1.0),
    ("5 years", 5.0),
    ("20 years", 20.0),
]
REWARDS = [50, 500, 5000]
MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
CAP_MULT = 20
K_HIGH = 0.5
K_LOW = 1e-4
# First three categorical slots of the validated reference palette (all-pairs safe).
R_COLORS = {50: "#2a78d6", 500: "#eb6834", 5000: "#1baf7a"}


# -------------------- Prompting --------------------


def fmt_dollars(x):
    if abs(x - round(x)) < 0.005:
        return f"${round(x):,}"
    return f"${x:,.2f}"


def build_prompt(immediate, delayed, delay_label, swapped):
    opt_now = f"receive {fmt_dollars(immediate)} today"
    opt_later = f"receive {fmt_dollars(delayed)} in {delay_label}"
    first, second = (opt_later, opt_now) if swapped else (opt_now, opt_later)
    return (
        "You must choose between two options.\n"
        f"Option A: {first}.\n"
        f"Option B: {second}.\n"
        "Which option do you choose? Answer with exactly one letter, A or B."
    )


class ModelChooser:
    """Forced A/B choice via teacher-forced logprob comparison."""

    def __init__(self, runner):
        self.runner = runner

    def picks_delayed(self, immediate, delayed, delay_label, swapped):
        prompt = build_prompt(immediate, delayed, delay_label, swapped)
        choice = self.runner.choose(prompt, choice_prefix="", labels=("A", "B"))
        lp_a, lp_b = choice.divergent_logprobs
        picked_b = lp_b > lp_a
        picked_delayed = picked_b != swapped  # unswapped: B is delayed; swapped: A is delayed
        return picked_delayed, float(lp_a), float(lp_b)


class StubChooser:
    """Deterministic hyperbolic chooser for validating search logic without a model."""

    K_BY_R = {50: 0.08, 500: 0.05, 5000: 0.03}

    def picks_delayed(self, immediate, delayed, delay_label, swapped):
        d_years = dict(DELAYS)[delay_label]
        threshold = immediate * (1.0 + self.K_BY_R[immediate] * d_years)
        picked_delayed = delayed > threshold
        return picked_delayed, 0.0, -1.0

    def true_boundary(self, immediate, d_years):
        return immediate * (1.0 + self.K_BY_R[immediate] * d_years)


# -------------------- Titration --------------------


def titrate_cell(chooser, reward, delay_label, d_years, steps, records):
    """Binary-search the smallest delayed X the model prefers over `reward` today."""
    n_reversals = 0

    def ask(x):
        nonlocal n_reversals
        picked, lp_a, lp_b = chooser.picks_delayed(reward, x, delay_label, swapped=False)
        picked_s, lp_a_s, lp_b_s = chooser.picks_delayed(reward, x, delay_label, swapped=True)
        for swapped, p, a, b in ((False, picked, lp_a, lp_b), (True, picked_s, lp_a_s, lp_b_s)):
            records.append(
                dict(delay=delay_label, delay_years=d_years, reward=reward, delayed_x=x,
                     swapped=swapped, picked_delayed=p, logprob_a=a, logprob_b=b)
            )
        n_reversals += int(picked != picked_s)
        return picked

    lo, hi = float(reward), float(reward * CAP_MULT)
    base = dict(reward=reward, delay=delay_label, delay_years=d_years,
                k_lower_bound=None, k_bracket=None)
    if ask(lo):
        # Prefers the delayed option even with zero premium: k <= 0.
        return dict(base, status="always_delayed", X_boundary=lo, k=0.0,
                    n_reversals=n_reversals)
    if not ask(hi):
        # Still prefers immediate at the 20x cap: k exceeds the searchable range.
        return dict(base, status="no_boundary", X_boundary=None, k=None,
                    k_lower_bound=(CAP_MULT - 1) / d_years, n_reversals=n_reversals)
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        if ask(mid):
            hi = mid
        else:
            lo = mid
    x_boundary = (lo + hi) / 2.0
    k = (x_boundary / reward - 1.0) / d_years
    bracket = [(lo / reward - 1.0) / d_years, (hi / reward - 1.0) / d_years]
    return dict(base, status="ok", X_boundary=x_boundary, k=k, k_bracket=bracket,
                n_reversals=n_reversals, search_lo=lo, search_hi=hi)


def flag_cell(cell):
    """Extreme-rate flags, robust to search resolution via the final bracket."""
    flags = []
    if cell["status"] == "no_boundary":
        flags += ["no_boundary_within_cap", "extreme_high"]
        return flags
    k_lo, k_hi = cell["k_bracket"] if cell["k_bracket"] else (cell["k"], cell["k"])
    if k_lo > K_HIGH:
        flags.append("extreme_high")
    if k_hi < K_LOW:
        flags.append("extreme_low")
    return flags


# -------------------- Inconsistency metrics --------------------


def effective_k(cell):
    return cell["k_lower_bound"] if cell["status"] == "no_boundary" else cell["k"]


def magnitude_reversals(cells):
    """Ordering flips of k(R=50) vs k(R=5000) across delays."""
    orderings = []
    for label, _ in DELAYS:
        k_small = effective_k(cells[(label, REWARDS[0])])
        k_large = effective_k(cells[(label, REWARDS[-1])])
        if k_small == k_large:
            continue
        orderings.append(dict(delay=label, k_small=k_small, k_large=k_large,
                              small_gt_large=k_small > k_large))
    n_flips = sum(
        1 for i in range(len(orderings) - 1)
        if orderings[i]["small_gt_large"] != orderings[i + 1]["small_gt_large"]
    )
    return dict(orderings=orderings, n_flips=n_flips)


def nonmonotonicity(cells):
    """Indifference X should increase with delay; count strict decreases per R."""
    out = {}
    for reward in REWARDS:
        seq = []
        for label, _ in DELAYS:
            cell = cells[(label, reward)]
            x = cell["X_boundary"] if cell["status"] != "no_boundary" else float(reward * CAP_MULT)
            seq.append(dict(delay=label, x=x))
        n_violations = sum(1 for i in range(len(seq) - 1) if seq[i + 1]["x"] < seq[i]["x"] - 1e-9)
        out[str(reward)] = dict(sequence=seq, n_violations=n_violations)
    return out


# -------------------- Model driver --------------------


def select_backend(model_name):
    if model_name == "Qwen/Qwen3-4B-Instruct-2507":
        return ModelBackend.TRANSFORMERLENS  # mapped to the Qwen3-4B config inside ModelRunner
    try:
        get_official_model_name(model_name)
        return ModelBackend.TRANSFORMERLENS
    except ValueError:
        return ModelBackend.HUGGINGFACE


def run_model(model_name, steps, device):
    if model_name == "stub":
        chooser, runner, backend = StubChooser(), None, "stub"
    else:
        backend = select_backend(model_name)
        runner = BinaryChoiceRunner(model_name, device=device, backend=backend)
        chooser = ModelChooser(runner)

    records, cells, cell_list = [], {}, []
    t0 = time.time()
    for reward in REWARDS:
        for label, d_years in DELAYS:
            cell = titrate_cell(chooser, reward, label, d_years, steps, records)
            cell["flags"] = flag_cell(cell)
            cells[(label, reward)] = cell
            cell_list.append(cell)
            k_txt = "none" if cell["k"] is None else f"{cell['k']:.5f}"
            print(f"  R=${reward:<5d} d={label:<9s} status={cell['status']:<14s} "
                  f"k={k_txt} flags={cell['flags']}", flush=True)

    result = dict(
        model=model_name,
        backend=str(backend),
        titration_steps=steps,
        cap_multiplier=CAP_MULT,
        k_high_threshold=K_HIGH,
        k_low_threshold=K_LOW,
        cells=cell_list,
        inconsistency=dict(
            magnitude_effect=magnitude_reversals(cells),
            nonmonotonicity=nonmonotonicity(cells),
            n_preference_reversals=sum(c["n_reversals"] for c in cell_list),
            n_queries=len(records) // 2,
        ),
        choices=records,
        elapsed_sec=time.time() - t0,
    )

    if isinstance(chooser, StubChooser):
        validate_stub(chooser, cells)
    if runner is not None:
        runner.unload()
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def validate_stub(chooser, cells):
    n_bad = 0
    for (label, reward), cell in cells.items():
        if cell["status"] != "ok":
            continue
        x_true = chooser.true_boundary(reward, dict(DELAYS)[label])
        if not (cell["search_lo"] <= x_true <= cell["search_hi"]):
            n_bad += 1
            print(f"STUB FAIL: {label} R={reward}: true {x_true} outside "
                  f"[{cell['search_lo']}, {cell['search_hi']}]")
    print(f"Stub validation: {'PASS' if n_bad == 0 else f'{n_bad} FAILURES'}")


# -------------------- Outputs --------------------


def write_summary_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "delay", "delay_years", "R", "X_boundary", "k",
                         "k_lower_bound", "status", "n_reversals", "flags"])
        for res in results:
            for cell in res["cells"]:
                writer.writerow([
                    res["model"], cell["delay"], f"{cell['delay_years']:.6f}",
                    cell["reward"],
                    "" if cell["X_boundary"] is None else f"{cell['X_boundary']:.2f}",
                    "" if cell["k"] is None else f"{cell['k']:.6f}",
                    "" if cell["k_lower_bound"] is None else f"{cell['k_lower_bound']:.6f}",
                    cell["status"], cell["n_reversals"], ";".join(cell["flags"]),
                ])


def plot_k_vs_delay(results, path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), sharey=True, squeeze=False)
    y_floor = K_LOW / 10.0
    for ax, res in zip(axes[0], results):
        for reward in REWARDS:
            xs, ys, flagged = [], [], []
            for cell in res["cells"]:
                if cell["reward"] != reward:
                    continue
                k = effective_k(cell)
                xs.append(cell["delay_years"])
                ys.append(max(k, y_floor))
                flagged.append(bool(cell["flags"]))
            ax.plot(xs, ys, color=R_COLORS[reward], lw=2, marker="o", ms=5,
                    label=f"R=${reward}")
            fx = [x for x, fl in zip(xs, flagged) if fl]
            fy = [y for y, fl in zip(ys, flagged) if fl]
            ax.plot(fx, fy, ls="none", marker="x", ms=9, mew=2, color="#0b0b0b")
        ax.axhline(K_HIGH, color="#52514e", lw=1, ls="--")
        ax.axhline(K_LOW, color="#52514e", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(res["model"].split("/")[-1], fontsize=9)
        ax.set_xlabel("delay (years)")
        ax.grid(alpha=0.2, which="both")
    axes[0][0].set_ylabel("boundary k (1/yr)")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Hyperbolic discount rate at indifference (x = flagged extreme)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------- Main --------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "out" / "extreme_discount"))
    parser.add_argument("--stub", action="store_true",
                        help="Validate prompts and search logic with a deterministic chooser.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = ["stub"] if args.stub else args.models

    results = []
    for model_name in models:
        print(f"\n=== {model_name} ===", flush=True)
        res = run_model(model_name, args.steps, args.device)
        json_path = out_dir / f"{model_name.replace('/', '_')}.json"
        json_path.write_text(json.dumps(res, indent=2))
        print(f"Wrote {json_path} ({res['elapsed_sec']:.1f}s, "
              f"reversals={res['inconsistency']['n_preference_reversals']})", flush=True)
        results.append(res)

    write_summary_csv(results, out_dir / "extreme_discount_summary.csv")
    plot_k_vs_delay(results, out_dir / "extreme_discount_k_vs_delay.png")
    print(f"\nSummary CSV and figure written under {out_dir}")


if __name__ == "__main__":
    main()
