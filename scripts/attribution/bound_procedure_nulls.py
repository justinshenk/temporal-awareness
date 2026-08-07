"""Attach exact binomial bounds to every rung of the procedure null ladder.

Reads the committed artifacts, recovers each rung's hit count from its stored accuracy and n, and
prints the ladder with intervals on the recovery scale. CPU only, no model, no network.

    uv run python -m scripts.attribution.bound_procedure_nulls

Writes ``results/attribution/null_bounds.json`` so the paper's table has one source.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.common.null_intervals import bounded_null_from_rate

RESULTS = Path("results/attribution")


def _load(name: str) -> dict:
    with open(RESULTS / f"{name}.json") as fh:
        return json.load(fh)


def _global_ridge() -> tuple[str, float, int, float, float]:
    """The baseline rung. Its n and references come from the local-refit artifact's global arm."""
    d = _load("local_refit_gsm8k")
    return ("global primal-ridge map", d["by_lam"]["global"]["1.0"]["steer_acc"],
            d["n_eval"], d["base_acc"], d["lora_acc"])


def _local_refit() -> tuple[str, float, int, float, float]:
    d = _load("local_refit_gsm8k")
    return ("per-context local refit", d["by_lam"]["1"]["1.0"]["steer_acc"],
            d["n_eval"], d["base_acc"], d["lora_acc"])


def _dagger() -> tuple[str, float, int, float, float]:
    d = _load("dagger_refit_gsm8k")
    final = d["rounds"][-1]
    return (f"on-policy DAgger (round {final['round']})", final["steer_acc"],
            d["n_eval"], d["base_acc"], d["lora_acc"])


def _short_arithmetic() -> tuple[str, float, int, float, float]:
    """The ``cot`` arm, not ``direct``.

    ``direct`` forces a zero-step answer and there the donor scores *below* base
    (0.767 -> 0.600, a negative budget), so it carries no recovery claim; the writeup's
    trajectory-length argument rests on the one-step ``cot`` arm (base 0.000 -> donor 0.733).
    """
    d = _load("short_arithmetic")
    cot = d["by_mode"]["cot"]
    return ("short-output arithmetic (1-step)", cot["steer"]["1.0"]["acc"],
            d["n"], cot["base"]["acc"], cot["lora"]["acc"])


def _das() -> list[tuple[str, float, int, float, float]]:
    """DAS is already on the recovery scale over a contrast set, so the budget is 1.0."""
    d = _load("das_subspace_L20")
    n = d["n_contrast"]
    return [(f"DAS task-loss subspace (r={r})", spec["das_recovery"], n, 0.0, 1.0)
            for r, spec in d["ranks"].items()]


def collect_rungs() -> list[tuple[str, float, int, float, float]]:
    return [_global_ridge(), _short_arithmetic(), _local_refit(), _dagger(), *_das()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "null_bounds.json")
    args = parser.parse_args()

    rows = []
    print(f"{'rung':<34} {'recovery':>8}  {'95% interval':>18}  {'n':>4}")
    print("-" * 72)
    for label, acc, n, base, lora in collect_rungs():
        bound = bounded_null_from_rate(acc, n, base_acc=base, lora_acc=lora)
        rows.append({"rung": label, **asdict(bound)})
        interval = f"[{bound.recovery_lo:+.3f}, {bound.recovery_hi:+.3f}]"
        print(f"{label:<34} {bound.recovery:>+8.3f}  {interval:>18}  {n:>4}")

    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    print("\nNOTE: intervals treat the base and donor accuracies as known constants, so they "
          "cover\nsampling error in the steered run only. State that wherever these appear.")


if __name__ == "__main__":
    main()
