"""E6 mode-vector steering: is the answering style a residual direction you can install and erase?

The exemplar-close arms left one account standing — the exemplars install a format mode in the
residual stream during prefill, self-sustaining thereafter. If that mode is a direction, it
should steer both ways:

- **install**: add the vector to a clean, zero-filler context at one layer (all positions,
  decode included). Bare-letter replies with no exemplar in context = the direction suffices.
- **erase**: project the direction out at depth 42 with all 42 exemplars attendable.
  ANSWER:/SUPPORTING: returning = the direction is necessary — a third independent restoration
  (after upclamp and refresh), each implicating a different link.

Vector sources: ``meandiff`` (mean depth-42 state − mean depth-0 state at the layer, from
``run_format_probes.py``'s capture; confounds mode with context length, which the random-vector
controls and the erase arm's specificity address) or ``file`` (any (hidden,) .npy — e.g. the
Probe-2 weight direction, fit at matched depth/fill).

Layer convention: ``--layer`` takes the capture stack index (0 = embeddings, i = output of
decoder layer i−1), matching ``probe_results.json``; the hook attaches to decoder layer
``stack_index − 1``.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/context_fatigue/run_format_steering.py \
        --layer 24 --preflight
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import generate_with_entropy, render_prompt
from run_format_probes import GSM8K_DEPTHS, MMLU_DEPTHS, rebuild_snapshots

from src.probes.context_fatigue.context_assembly import OverflowGuard
from src.probes.context_fatigue.ddxplus_cases import format_case_question
from src.probes.context_fatigue.instruction_checks import check_clinical_format
from src.probes.safety.steering_hook import AdditionSteeringHook, DirectionProjectionHook


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--capture-dir", default="results/context_fatigue/e6_format_probes")
    p.add_argument("--layer", type=int, required=True,
                   help="capture stack index (embeddings = 0); hook goes on decoder layer-1")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--vector", choices=["meandiff", "file"], default="meandiff")
    p.add_argument("--vector-file", default=None)
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--headroom", type=int, default=16)
    p.add_argument("--max-filler-tokens", type=int, default=90)
    p.add_argument("--n-probes", type=int, default=40)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--filler-max-new", type=int, default=200)
    p.add_argument("--out-dir", default="results/context_fatigue/e6_mode_steering")
    p.add_argument("--device", default="cuda")
    p.add_argument("--round2", action="store_true",
                   help="refined arms after round 1's instrument failures: function-vector-style "
                        "install (final position only, alpha 1 and 3) and multi-layer erase "
                        "(project each layer's own mean-diff across --erase-stack-layers), each "
                        "with a norm-matched random control")
    p.add_argument("--erase-stack-layers", type=int, nargs="+",
                   default=list(range(14, 25)),
                   help="stack indices for round 2's multi-layer projection (Probe-2's "
                        "decodability band by default)")
    p.add_argument("--erase-only", action="store_true",
                   help="run only the erase and erase_rand arms — for testing an externally "
                        "supplied direction (e.g. Probe 2's) without the install arms")
    p.add_argument("--erase-context", choices=["mmlu", "gsm8k"], default="mmlu",
                   help="which arm's accumulated transcript to erase in; gsm8k erases at its "
                        "deepest depth in the distribution the probe direction was fit on")
    p.add_argument("--round3", action="store_true",
                   help="decode-time install: the vector rides the final prefill position and "
                        "every decode step, never the context — present for the whole "
                        "generation. Arms at alpha 1 and 3 plus a norm-matched random control.")
    p.add_argument("--preflight", action="store_true")
    return p.parse_args()


def load_states(capture_dir: Path, depth: int, layer: int) -> np.ndarray:
    z = np.load(capture_dir / f"mmlu_d{depth}.npz")
    rows = pd.DataFrame(json.loads(str(z["rows"])))
    keep = (rows.variant == "format").values
    return z["states"][keep][:, layer]


def meandiff_vector(capture_dir: Path, stack_layer: int) -> torch.Tensor:
    deep = load_states(capture_dir, max(MMLU_DEPTHS), stack_layer).mean(axis=0)
    clean = load_states(capture_dir, 0, stack_layer).mean(axis=0)
    return torch.from_numpy(deep - clean).float()


def build_vector(args) -> torch.Tensor:
    if args.vector == "file":
        return torch.from_numpy(np.load(args.vector_file)).float()
    return meandiff_vector(Path(args.capture_dir), args.layer)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hook_layer = args.layer - 1
    if hook_layer < 0:
        raise ValueError("--layer 0 is the embedding row; steering needs a decoder layer")

    v = build_vector(args)
    rng = torch.Generator().manual_seed(args.seed)
    rand = torch.randn(v.shape, generator=rng)
    rand = rand / rand.norm() * v.norm()
    print(f"vector norm {v.norm():.2f} at stack layer {args.layer} "
          f"(hook on decoder layer {hook_layer})", flush=True)

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()

    filler_kind = args.erase_context if args.erase_only else "mmlu"
    depth_list = GSM8K_DEPTHS if filler_kind == "gsm8k" else MMLU_DEPTHS
    snapshots, probes = rebuild_snapshots(model, tokenizer, args, filler_kind, depth_list)
    n_probes = 2 if args.preflight else args.n_probes
    probes = probes[:n_probes]
    guard = OverflowGuard(count_tokens=lambda t: len(tokenizer.encode(t)),
                          max_ctx=args.max_ctx, max_new=args.max_new, headroom=args.headroom)

    if args.erase_only:
        deep = max(depth_list)
        arms = [
            ("erase", deep, lambda m: DirectionProjectionHook(m, {hook_layer: v})),
            ("erase_rand", deep, lambda m: DirectionProjectionHook(m, {hook_layer: rand})),
        ]
    elif args.round3:
        # Round 2's lesson: the control must be norm-matched to the arm it controls for — the
        # alpha-3 random control was so large it broke generation outright, leaving alpha-1
        # uncontrolled. Every alpha here carries its own matched random arm.
        arms = [
            ("install_decode_a1", 0, lambda m: AdditionSteeringHook(
                m, {hook_layer: v}, decode_time=True)),
            ("install_decode_rand_a1", 0, lambda m: AdditionSteeringHook(
                m, {hook_layer: rand}, decode_time=True)),
            ("install_decode_a3", 0, lambda m: AdditionSteeringHook(
                m, {hook_layer: 3.0 * v}, decode_time=True)),
        ]
    elif args.round2:
        cap = Path(args.capture_dir)
        diffs = {sl - 1: meandiff_vector(cap, sl) for sl in args.erase_stack_layers}
        gen = torch.Generator().manual_seed(args.seed + 1)
        rand_diffs = {}
        for li, dv in diffs.items():
            r = torch.randn(dv.shape, generator=gen)
            rand_diffs[li] = r / r.norm() * dv.norm()
        arms = [
            ("install_last_a1", 0, lambda m: AdditionSteeringHook(
                m, {hook_layer: v}, last_token=True)),
            ("install_last_a3", 0, lambda m: AdditionSteeringHook(
                m, {hook_layer: 3.0 * v}, last_token=True)),
            ("install_last_rand", 0, lambda m: AdditionSteeringHook(
                m, {hook_layer: 3.0 * rand}, last_token=True)),
            ("erase_multi", max(MMLU_DEPTHS), lambda m: DirectionProjectionHook(m, diffs)),
            ("erase_multi_rand", max(MMLU_DEPTHS),
             lambda m: DirectionProjectionHook(m, rand_diffs)),
        ]
    else:
        arms = [
            ("install", 0, lambda m: AdditionSteeringHook(m, {hook_layer: args.alpha * v})),
            ("install_rand", 0, lambda m: AdditionSteeringHook(
                m, {hook_layer: args.alpha * rand})),
            ("erase", max(MMLU_DEPTHS), lambda m: DirectionProjectionHook(m, {hook_layer: v})),
            ("erase_rand", max(MMLU_DEPTHS),
             lambda m: DirectionProjectionHook(m, {hook_layer: rand})),
        ]
    records = []
    for arm, depth, make_hook in arms:
        prefix = snapshots[depth]
        hook = make_hook(model)
        try:
            for pi, probe in enumerate(probes):
                question = format_case_question(probe["options"], args.n_options,
                                                answer_cue=False)
                turns = prefix + [{"role": "user", "content": probe["vignette"] + question}]
                text = render_prompt(tokenizer, turns, is_chat=True)
                if not guard.fits(text, used=0, index=pi):
                    continue
                resp, ctx_len, entropy, _ = generate_with_entropy(
                    model, tokenizer, text, args.device, args.max_new, args.max_ctx)
                graded = check_clinical_format(resp or "", probe["vignette"],
                                               options=probe["options"][:args.n_options])
                records.append({
                    "arm": arm, "depth": depth, "probe": pi,
                    "ctx_tokens": ctx_len, "fill": round(ctx_len / args.max_ctx, 4),
                    "gold": probe["gold"], "pred": graded["answer"],
                    "correct": bool(graded["answer"] == probe["gold"]),
                    "parsed": graded["answer"] is not None,
                    "response_chars": len(resp or ""), "mean_entropy": entropy,
                    **{k: graded[k] for k in ("has_answer", "has_supporting", "n_symptoms",
                                              "grounded_fraction", "fully_compliant")},
                    "response": resp or "",
                })
        finally:
            hook.remove()
        df = pd.DataFrame(records)
        cur = df[df.arm == arm]
        print(f"  {arm:>12s} (depth {depth:2d}): compliant={cur.fully_compliant.mean():.3f}  "
              f"acc={cur.correct.mean():.3f}  chars={cur.response_chars.mean():.0f}",
              flush=True)
        df.to_csv(out_dir / "turns.csv", index=False)
        torch.cuda.empty_cache()

    (out_dir / "config.json").write_text(json.dumps({
        "model": args.model, "layer_stack_index": args.layer, "hook_layer": hook_layer,
        "alpha": args.alpha, "vector": args.vector, "vector_file": args.vector_file,
        "vector_norm": float(v.norm()), "n_probes": n_probes, "seed": args.seed}, indent=1))
    print(f"Saved to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
