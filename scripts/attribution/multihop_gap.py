"""Base-vs-donor gap and contrast set on MuSiQue open-book multi-hop QA — the Phase 0 go/no-go gate.

The multi-hop analogue of ``lockstep_patch_gsm8k.build_contrast_set``: greedily decode the base and
the LoRA donor from the *identical* open-book prompt (one weight set, toggled with
``disable_adapter()``) and keep the base-fails / donor-solves problems. That set is the recoverable
budget every later phase measures against, so it is cached in the GSM8K schema
(``{indices, base_acc, lora_acc, n_eval}``) for the task-parameterized drivers.

The gate: the donor must clear the base by at least ``--min-contrast`` problems. Too small a gap and
the generality test has nothing to recover, which is itself a reportable finding (H_arith survives by
default) — see ``tasks/current_task.md``.

    uv run python -m scripts.attribution.multihop_gap \
        --config configs/attribution/multihop_llama2.yaml --n-eval 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import build_contrast_set, get_task, load_base_and_lora
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.multihop_data import multihop_problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-eval", type=int, default=None, help="override eval.n_eval (gate scan size)")
    ap.add_argument("--max-new", type=int, default=None, help="override eval.max_new")
    ap.add_argument("--min-contrast", type=int, default=80,
                    help="gate: minimum base-fail/donor-solve problems to proceed to P1")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, dcfg, ecfg, ocfg = cfg["device"], cfg["data"], cfg["eval"], cfg["output"]
    n_eval = args.n_eval or ecfg["n_eval"]
    max_new = args.max_new or ecfg["max_new"]

    print(f"Loading {cfg['base_model']} + donor {cfg['adapter']} ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)

    problems = multihop_problems(dcfg["contrast_split"], n_eval, seed=cfg["seed"])
    print(f"{len(problems)} MuSiQue problems from split '{dcfg['contrast_split']}'", flush=True)

    indices, base_acc, lora_acc = build_contrast_set(
        base, lora, tok, problems, device, max_new, Path(ocfg["contrast_json"]),
        get_task(cfg.get("task", "multihop")))

    passed = len(indices) >= args.min_contrast
    summary = {
        "task": "multihop",
        "base_model": cfg["base_model"],
        "adapter": cfg["adapter"],
        "split": dcfg["contrast_split"],
        "n_eval": len(problems),
        "max_new": max_new,
        "base_acc": base_acc,
        "donor_acc": lora_acc,
        "gap": lora_acc - base_acc,
        "n_contrast": len(indices),
        "min_contrast": args.min_contrast,
        "gate_passed": passed,
    }
    gap_path = Path(ocfg["gap_json"])
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    gap_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== P0 GATE ===\nbase={base_acc:.3f}  donor={lora_acc:.3f}  gap={lora_acc-base_acc:+.3f}"
          f"\ncontrast set: {len(indices)} (need >={args.min_contrast})"
          f"\nverdict: {'PASS -> proceed to P1' if passed else 'FAIL -> stop and report'}"
          f"\nwrote {gap_path} and {ocfg['contrast_json']}", flush=True)


if __name__ == "__main__":
    main()
