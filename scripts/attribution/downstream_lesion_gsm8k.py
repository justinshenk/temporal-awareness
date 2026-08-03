"""E3 — is base's downstream load-bearing computation, or a transcription channel?

Apply the full-δ L20 oracle (overwrite base's L20 residual with the LoRA's, every decode step) and
then *identity-ablate* base layers L+1..N cumulatively from the top ({31}, {30,31}, …, {21..31}),
measuring closed-loop GSM8K recovery on the base-fails/LoRA-solves contrast set at each ablation level.

The control is the **LoRA-natural** model on the same problems under the identical ablation (base
solves ≈nothing, so a base-solvable control set is empty). The two curves adjudicate the
compute-vs-communicate question:
- recovery survives heavy ablation AND the LoRA-natural curve does too → the answer is already at L20,
  21–31 are a readout (communication);
- recovery collapses with ablation, tracking the LoRA-natural collapse → the natively-capable model
  also needs 21–31, so base+patch recovery is base doing the analogous downstream computation, and the
  matched curves rule out "ablation merely breaks generic fluency".

    uv run python -m scripts.attribution.downstream_lesion_gsm8k \
        --config configs/attribution/metamath_llama2_gsm8k.yaml --layer 20 --n-contrast 20

Reuses the lockstep oracle (`OverwriteResidualHook`, `lockstep_generate`) and the contrast set; adds
`IdentityAblationHook` (`src/probes/attribution/layer_ablation.py`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.attribution.attribution_common import load_base_and_lora, load_contrast, prompt_token_ids
from scripts.safety.extract_refusal_shifts import set_seed
from src.probes.attribution.gsm8k_prompts import extract_pred_number, numeric_match
from src.probes.attribution.layer_ablation import IdentityAblationHook
from src.probes.attribution.lockstep_oracle import (
    OverwriteResidualHook,
    lockstep_generate,
)
from src.probes.extraction import PerTokenResidualCapture


def make_full_oracle_fns(base, lora, capture, inject, ablate, L):
    """Capture fn (full δ = overwrite L with the LoRA's residual) and an ablation-aware base_logits."""

    def capture_residuals(S):
        capture.clear()
        with capture.capturing():
            base(S, use_cache=False)                       # LoRA residual on the same context
        return {L: capture.captured[L]}                    # full oracle: inject the LoRA's L20

    def base_logits(S):
        with inject.injecting(), ablate.ablating(), lora.disable_adapter():
            return base(S, use_cache=False).logits

    return capture_residuals, base_logits


def oracle_recovery(base, lora, tok, problems, device, L, inject, ablate, active, max_new):
    capture = PerTokenResidualCapture(base, [L])
    cap_fn, logit_fn = make_full_oracle_fns(base, lora, capture, inject, ablate, L)
    ablate.set_layers(active)
    eos = tok.eos_token_id
    correct = 0
    for q, gold in problems:
        pid = prompt_token_ids(tok, q, device)
        out = lockstep_generate(pid, cap_fn, logit_fn, inject, [L], max_new, eos, device)
        text = tok.decode(out[0][pid.shape[1]:], skip_special_tokens=True)
        correct += bool(numeric_match(extract_pred_number(text), gold))
    capture.remove()
    return correct / len(problems)


@torch.no_grad()
def lora_recovery(lora, tok, problems, device, ablate, active, max_new):
    """LoRA-natural greedy decode under the same ablation (adapter on, no injection)."""
    ablate.set_layers(active)
    pad = tok.pad_token_id or tok.eos_token_id
    correct = 0
    for q, gold in problems:
        pid = prompt_token_ids(tok, q, device)
        with ablate.ablating():
            out = lora.generate(pid, max_new_tokens=max_new, do_sample=False, pad_token_id=pad)
        text = tok.decode(out[0][pid.shape[1]:], skip_special_tokens=True)
        correct += bool(numeric_match(extract_pred_number(text), gold))
    return correct / len(problems)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-contrast", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--ablation-levels", default=None,
                    help="comma-separated k values (#top downstream layers ablated); default 0..N-L-1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device, L = cfg["device"], args.layer

    print(f"Loading {cfg['base_model']} + adapter ...", flush=True)
    tok, base, lora = load_base_and_lora(cfg)
    n_layers = base.config.num_hidden_layers
    downstream = list(range(L + 1, n_layers))              # 21..31 for L=20
    levels = ([int(x) for x in args.ablation_levels.split(",")]
              if args.ablation_levels else list(range(len(downstream) + 1)))

    contrast = load_contrast(cfg)[:args.n_contrast]
    inject = OverwriteResidualHook(base, [L])              # registered before ablate
    ablate = IdentityAblationHook(base, downstream)

    results = {"layer": L, "n_contrast": len(contrast), "max_new": args.max_new,
               "downstream_layers": downstream, "levels": {}}
    print(f"\nk   ablated layers          recovery_patch  recovery_lora", flush=True)
    for k in levels:
        active = downstream[len(downstream) - k:]          # top-k closest to output
        rp = oracle_recovery(base, lora, tok, contrast, device, L, inject, ablate, active, args.max_new)
        rl = lora_recovery(lora, tok, contrast, device, ablate, active, args.max_new)
        results["levels"][str(k)] = {"ablated": active, "recovery_patch": rp, "recovery_lora": rl}
        print(f"{k:<3d} {str(active):22s}  {rp:.3f}           {rl:.3f}", flush=True)

    inject.remove()
    ablate.remove()
    out_path = (Path(args.out) if args.out
                else Path(cfg["output"]["steer_json"]).parent / f"downstream_lesion_L{L}.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
