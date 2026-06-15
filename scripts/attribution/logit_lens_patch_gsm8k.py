"""E1 — where does the answer token crystallize across depth at the L20 patch site?

Runs three closed-loop trajectories on the base-fails/LoRA-solves contrast set and, at every decode
step, logit-lenses the residual at readout layers {20,22,…,31} for the token actually emitted that
step — recording its per-layer rank and logit:
- **patch**: base with the full-δ L20 oracle (overwrite L20 with the LoRA's residual), base weights 21+;
- **lora**: the LoRA-natural trajectory (its own residuals);
- **base**: the base-only trajectory.

The headline restricts to **answer-bearing tokens** (decoded token contains a digit). If the emitted
answer token is already top-1 at L20 and its rank barely moves to L31, the answer is present at the
patch site and base's 21–31 transcribe it (communication). If it is buried at L20 and climbs to top-1
only deeper, base's 21–31 compute it. Caveat (documented in the writeup): logit-lens top-1 at L20 is
*sufficient but not necessary* for communication — base's 21–31 are a nonlinear decoder — so this is a
conservative lower bound; the causal complement is the downstream-lesion experiment (E3).

    uv run python -m scripts.attribution.logit_lens_patch_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n-contrast 20
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import load_base_and_lora, prompt_token_ids
from scripts.attribution.nonlinear_delta_gsm8k import load_contrast
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.gsm8k_prompts import extract_pred_number, numeric_match
from src.probes.attribution.logit_lens import LogitLens
from src.probes.attribution.lockstep_oracle import OverwriteResidualHook, lockstep_generate
from src.probes.extraction import PerTokenResidualCapture


def token_is_answer_bearing(tok, token_id: int) -> bool:
    """A token contributes to the numeric answer if its decoded text contains a digit."""
    return any(c.isdigit() for c in tok.decode([token_id]))


def lens_trajectory(mode, base, lora, tok, question, device, L, readout_layers, inject, readout,
                    lens, max_new):
    """One closed-loop trajectory; returns (generated_text, [per-step lens records]).

    ``mode`` ∈ {"patch","lora","base"} selects injection + which adapter state runs the readout
    forward. Records are appended inside ``base_logits`` for the token that step emits, so they use the
    exact emitted token ``lockstep_generate`` selects (no duplicated decode logic).
    """
    records: list[dict] = []
    inject_layers = [L] if mode == "patch" else []

    def capture_residuals(S):
        if mode != "patch":
            return {}
        readout.clear()
        with readout.capturing():
            base(S, use_cache=False)                       # LoRA residual on the same context
        return {L: readout.captured[L]}                    # full oracle injection value (overwrite)

    def base_logits(S):
        readout.clear()
        adapter_ctx = nullcontext() if mode == "lora" else lora.disable_adapter()
        inject_ctx = inject.injecting() if mode == "patch" else nullcontext()
        with adapter_ctx, readout.capturing(), inject_ctx:
            logits = base(S, use_cache=False).logits
        emitted = int(logits[0, -1].argmax())
        rec = {"token_id": emitted, "token": tok.decode([emitted]),
               "answer_bearing": token_is_answer_bearing(tok, emitted), "rank": {}, "logit": {}}
        for li in readout_layers:
            rank, lg = lens.rank_of(readout.captured[li], emitted, row=-1)
            rec["rank"][str(li)] = rank
            rec["logit"][str(li)] = lg
        records.append(rec)
        return logits

    pid = prompt_token_ids(tok, question, device)
    out = lockstep_generate(pid, capture_residuals, base_logits, inject, inject_layers,
                            max_new, tok.eos_token_id, device)
    text = tok.decode(out[0][pid.shape[1]:], skip_special_tokens=True)
    return text, records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-contrast", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--readout-layers", default=None,
                    help="comma-separated layers to lens; default L,L+2,…,N-1 plus the last layer")
    ap.add_argument("--modes", default="patch,lora,base")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, L = cfg["device"], args.layer

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)
    n_layers = base.config.num_hidden_layers
    readout_layers = ([int(x) for x in args.readout_layers.split(",")]
                      if args.readout_layers
                      else sorted({*range(L, n_layers, 2), n_layers - 1}))

    inject = OverwriteResidualHook(base, [L])              # registered before readout
    readout = PerTokenResidualCapture(base, readout_layers)
    lens = LogitLens(base)
    contrast = load_contrast(cfg)[:args.n_contrast]
    modes = args.modes.split(",")

    results = {"layer": L, "n_contrast": len(contrast), "max_new": args.max_new,
               "readout_layers": readout_layers, "modes": {}}
    for mode in modes:
        print(f"\n=== mode={mode} ===", flush=True)
        problems = []
        for q, gold in contrast:
            text, records = lens_trajectory(mode, base, lora, tok, q, device, L, readout_layers,
                                            inject, readout, lens, args.max_new)
            problems.append({"gold": gold, "correct": bool(numeric_match(extract_pred_number(text), gold)),
                             "steps": records})
        ans = [r for p in problems for r in p["steps"] if r["answer_bearing"]]
        top1_at_L = sum(r["rank"][str(L)] == 0 for r in ans)
        print(f"  answer-bearing tokens: {len(ans)}; already top-1 at L{L}: {top1_at_L}", flush=True)
        results["modes"][mode] = {"problems": problems, "n_answer_tokens": len(ans),
                                  "answer_top1_at_L": top1_at_L}

    inject.remove()
    readout.remove()
    out_path = (Path(args.out) if args.out
                else Path(cfg["output"]["steer_json"]).parent / f"logit_lens_patch_L{L}.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
