"""E6 residual-stream probes: is the instruction represented, and is the mode decidable, before
generation?

The exemplar-close arms refuted generation-time copying: the model keeps producing the
demonstrated format with the exemplars unattendable. The surviving account is that the
exemplars install the answer-format mode upstream, during prefill. Two probes test it, from one
capture pass over the final pre-generation position's hidden states at every layer:

- **Probe 1 (instruction presence).** Same transcripts with the format system prompt vs a
  token-length-matched neutral one. A linear probe trained at depth 0 and tested at every mmlu
  depth asks whether the instruction stays decodable while compliance sits at 0.000. Flat AUC
  completes the three-level account: attention flat, representation intact, behavior flipped.
- **Probe 2 (mode visibility).** Within gsm8k's mixed cells (depths 12 and 15, compliance
  0.825/0.600), predict from the pre-generation state whether the reply will comply. Same depth,
  same filler kind, same fill — only the outcome differs. Decodability before the first token
  is emitted means the mode is set in the residual state, and the layer profile localizes it.

Capture only; analysis in ``analyze_format_probes.py``. Transcripts rebuild bit-identically from
the seed (verified: the spans re-runs matched the committed ladders at max |Δ| = 0.000).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_format_probes.py --preflight
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import (
    MEDICAL_SUBJECTS,
    generate_with_entropy,
    load_filler_pool,
    load_gsm8k_filler_pool,
    render_prompt,
)

from src.probes.context_fatigue.context_assembly import OverflowGuard
from src.probes.context_fatigue.ddxplus_cases import (
    format_case_question,
    load_evidence_db,
    load_probe_pool,
)
from src.probes.context_fatigue.instruction_checks import CLINICAL_FORMAT_SYSTEM

# Matched-length neutral twin: same role, no format constraint. Trimmed to the format prompt's
# exact token count at runtime so Probe 1 cannot read length.
NEUTRAL_SYSTEM_SEED = (
    "You are a doctor. For each patient, reply in whatever way seems most natural and helpful "
    "to you, describing your thinking about the case and its likely diagnosis in your own "
    "words, at whatever length and in whatever order you prefer for the patient described."
)

MMLU_DEPTHS = [0, 3, 7, 14, 21, 28, 35, 42]
GSM8K_DEPTHS = [0, 2, 4, 6, 9, 12, 15]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=256)  # the guard charges what erosion runs did
    p.add_argument("--headroom", type=int, default=16)
    p.add_argument("--max-filler-tokens", type=int, default=90)
    p.add_argument("--n-probes", type=int, default=40)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--filler-max-new", type=int, default=200)
    p.add_argument("--out-dir", default="results/context_fatigue/e6_format_probes")
    p.add_argument("--device", default="cuda")
    p.add_argument("--preflight", action="store_true")
    return p.parse_args()


def neutral_system(tokenizer) -> str:
    """The neutral twin, trimmed to exactly the format prompt's token count."""
    target = len(tokenizer.encode(CLINICAL_FORMAT_SYSTEM))
    ids = tokenizer.encode(NEUTRAL_SYSTEM_SEED)
    if len(ids) < target:
        raise ValueError(f"neutral seed too short: {len(ids)} < {target} tokens")
    text = tokenizer.decode(ids[:target])
    assert len(tokenizer.encode(text)) == target, "trim did not preserve token count"
    return text


def rebuild_snapshots(model, tokenizer, args, filler_kind, depths, is_chat=True):
    """The erosion driver's accumulation, reproduced on its exact rng stream."""
    if filler_kind == "mmlu":
        filler_pool = load_filler_pool(tokenizer, args.max_filler_tokens, MEDICAL_SUBJECTS)
    else:
        filler_pool = load_gsm8k_filler_pool((max(depths) + 2) * 4, args.seed)
    probe_pool = load_probe_pool(load_evidence_db(), args.n_options, args.seed)
    rng = random.Random(args.seed)
    filler = rng.sample(filler_pool, min(max(depths) + 2, len(filler_pool)))
    probes = rng.sample(probe_pool, args.n_probes)

    base = [{"role": "system", "content": CLINICAL_FORMAT_SYSTEM}]
    snapshots, conv = {}, list(base)
    for i, item in enumerate(filler):
        if i in depths:
            snapshots[i] = list(conv)
        if len(snapshots) == len(depths):
            break
        conv = conv + [{"role": "user", "content": item["text"]}]
        budget = 8 if filler_kind == "mmlu" else args.filler_max_new
        resp, _, _, _ = generate_with_entropy(
            model, tokenizer, render_prompt(tokenizer, conv, is_chat),
            args.device, budget, args.max_ctx)
        answer = (resp or "A").strip()
        conv = conv + [{"role": "assistant",
                        "content": answer[:8] if filler_kind == "mmlu" else answer}]
    for d in depths:
        snapshots.setdefault(d, list(conv))
    return snapshots, probes


def final_position_states(model, tokenizer, text, device) -> np.ndarray:
    """[n_layers+1, hidden] hidden states at the final prompt position."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device).eval()

    neutral = neutral_system(tokenizer)
    guard = OverflowGuard(count_tokens=lambda t: len(tokenizer.encode(t)),
                          max_ctx=args.max_ctx, max_new=args.max_new, headroom=args.headroom)

    n_probes = 2 if args.preflight else args.n_probes
    plan = [("mmlu", MMLU_DEPTHS[:2] if args.preflight else MMLU_DEPTHS, True),
            ("gsm8k", GSM8K_DEPTHS[:2] if args.preflight else GSM8K_DEPTHS, False)]

    manifest = {"model": args.model, "neutral_system": neutral, "rows": []}
    for filler_kind, depths, with_neutral_twin in plan:
        snapshots, probes = rebuild_snapshots(model, tokenizer, args, filler_kind, depths)
        probes = probes[:n_probes]
        for depth in depths:
            prefix = snapshots[depth]
            states, rows = [], []
            for pi, probe in enumerate(probes):
                question = format_case_question(probe["options"], args.n_options,
                                                answer_cue=False)
                user = {"role": "user", "content": probe["vignette"] + question}
                variants = [("format", prefix)]
                if with_neutral_twin:
                    twin = [{"role": "system", "content": neutral}] + prefix[1:]
                    variants.append(("neutral", twin))
                for variant, pre in variants:
                    text = render_prompt(tokenizer, pre + [user], is_chat=True)
                    if not guard.fits(text, used=0, index=pi):
                        continue
                    states.append(final_position_states(model, tokenizer, text, device))
                    rows.append({"filler": filler_kind, "depth": depth, "probe": pi,
                                 "variant": variant})
            np.savez_compressed(out_dir / f"{filler_kind}_d{depth}.npz",
                                states=np.stack(states),
                                rows=json.dumps(rows))
            manifest["rows"] += rows
            print(f"  {filler_kind} depth {depth:2d}: {len(rows)} states captured", flush=True)
            torch.cuda.empty_cache()

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Saved to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
