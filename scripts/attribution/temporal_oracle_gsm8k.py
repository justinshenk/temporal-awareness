"""Phase 1 — temporal decomposition of the L20 oracle: how time-sparse can the patch be?

The full-δ L20 lockstep oracle overwrites base's residual with the LoRA's at EVERY decode step and
recovers 0.75 on the base-fails/LoRA-solves contrast set. This sweeps the *time* axis no prior run
touched (``lockstep_fidelity_sweep`` varied magnitude, ``lockstep_pca_band`` varied dimensions):
patch only the steps a **gate** selects and measure recovery vs the fraction of steps patched.

  * periodic(k) — patch every k-th step (k=1 = full oracle, the positive control → must hit 0.75);
  * result_only — patch only computed-result tokens (digits after '=');
  * planning_only — its exact complement (operators, step text, boundaries);
  * step_boundary — patch only the first token of each line (the thinnest scaffold).

E1b predicts result_only ≈ base (base already computes results at 96.8% TF) and planning_only ≈ the
full oracle: i.e. the transported trajectory state is structural/planning, not arithmetic.

    uv run python -m scripts.attribution.temporal_oracle_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n-contrast 20 \
        --gates periodic:1,2,3,4,6,8 result_only planning_only step_boundary

Reuses the lockstep oracle wiring (``OverwriteResidualHook``, ``PerTokenResidualCapture``) and the
contrast set; adds the per-step gates (``src/probes/attribution/temporal_gate.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.attribution.attribution_common import get_task, load_base_and_lora, load_contrast, prompt_token_ids
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.lockstep_oracle import OverwriteResidualHook
from src.probes.attribution.multihop_prompts import answer_span_gate
from src.probes.attribution.temporal_gate import (
    gated_lockstep_generate,
    periodic,
    planning_only,
    result_only,
    step_boundary,
)
from src.probes.extraction import PerTokenResidualCapture


def build_gates(specs):
    """Parse CLI gate specs into ``[(name, gate), ...]``. ``periodic:1,2,4`` expands per period.

    ``result_only``/``planning_only`` partition GSM8K CoT by computed-result spans;
    ``answer_only``/``reasoning_only`` are the multi-hop structural analogue (final-answer span
    vs the hop chain before it).
    """
    gates = []
    named = {"result_only": result_only, "planning_only": planning_only,
             "step_boundary": step_boundary,
             "answer_only": answer_span_gate(),
             "reasoning_only": answer_span_gate(complement=True)}
    for spec in specs:
        if spec.startswith("periodic:"):
            for k in spec.split(":", 1)[1].split(","):
                gates.append((f"periodic_{int(k)}", periodic(int(k))))
        elif spec in named:
            gates.append((spec, named[spec]))
        else:
            raise ValueError(f"unknown gate spec: {spec}")
    return gates


def make_oracle_fns(base, lora, capture, inject, L):
    """Full-δ capture (overwrite L with the LoRA's residual) and the patched base-logits pass."""
    def capture_residuals(S):
        capture.clear()
        with capture.capturing():
            base(S, use_cache=False)                       # adapter ON → LoRA residual on base ctx
        return {L: capture.captured[L]}

    def base_logits(S):
        with inject.injecting(), lora.disable_adapter():
            return base(S, use_cache=False).logits

    return capture_residuals, base_logits


def run_gate(base, lora, tok, problems, device, L, inject, gate, max_new, task):
    """Gated lockstep-decode every problem; return (recovery, frac_patched, per_problem)."""
    capture = PerTokenResidualCapture(base, [L])
    cap_fn, logit_fn = make_oracle_fns(base, lora, capture, inject, L)

    def decode_token(i):
        return tok.decode([i])

    eos = tok.eos_token_id
    per, patched, steps = [], 0, 0
    for q, gold in problems:
        pid = prompt_token_ids(tok, q, device, task)
        out, mask = gated_lockstep_generate(pid, cap_fn, logit_fn, inject, [L], gate,
                                             decode_token, max_new, eos, device)
        text = tok.decode(out[0][pid.shape[1]:], skip_special_tokens=True)
        per.append(task.score(text, gold))
        patched += sum(mask)
        steps += len(mask)
    capture.remove()
    return sum(per) / len(per), (patched / steps if steps else 0.0), per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-contrast", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--gates", nargs="+",
                    default=["periodic:1,2,3,4,6,8", "result_only", "planning_only", "step_boundary"])
    ap.add_argument("--task", default=None, help="task registry key (default: config 'task' or gsm8k)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    task = get_task(args.task or cfg.get("task", "gsm8k"))
    device, L = cfg["device"], args.layer

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)
    contrast = load_contrast(cfg, task)[:args.n_contrast]
    gates = build_gates(args.gates)
    inject = OverwriteResidualHook(base, [L])

    results = {"task": task.name, "layer": L, "n_contrast": len(contrast),
               "max_new": args.max_new, "gates": {}}
    print(f"\n{'gate':16s} {'frac_patched':>12s} {'recovery':>9s}", flush=True)
    for name, gate in gates:
        rec, frac, per = run_gate(base, lora, tok, contrast, device, L, inject, gate, args.max_new, task)
        results["gates"][name] = {"recovery": rec, "frac_patched": frac, "per_problem": per}
        print(f"{name:16s} {frac:>12.3f} {rec:>9.3f}", flush=True)

    inject.remove()
    stem = f"temporal_oracle_L{L}" if task.name == "gsm8k" else f"temporal_oracle_{task.name}_L{L}"
    out_path = (Path(args.out) if args.out
                else Path(cfg["output"]["steer_json"]).parent / f"{stem}.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
