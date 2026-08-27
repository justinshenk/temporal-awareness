"""E5 — does compliance need the system prompt's attention mass?

The paper's instruction-adherence null says a checkable canary is obeyed at ceiling regardless of
context fill. The all-layer attention sweep shows why that is worth pressing: pooled over 1,024
heads, attention on the 29-token system span falls 0.2351 -> 0.0375 between cold start and 90%
fill. Per token the span stays over-attended, so the model is not deciding the instruction matters
less -- but E1c/E1f established that on a *fixed-size* span it is absolute attention mass that
drives behaviour, and the system prompt is fixed-size.

So this takes E2a's design and points it at the system span: on cold-start contexts, clamp the
system prompt's post-softmax share down through the range accumulation reaches, and measure
whether the canaries survive.

    uv run python scripts/context_fatigue/run_system_clamp.py --preflight
    uv run python scripts/context_fatigue/run_system_clamp.py --n-items 120

Brief: `tasks/e5_system_clamp_brief.md`.
"""

import argparse
import gc
import json
import random
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import (
    extract_mcq_answer,
    generate_with_entropy,
    render_prompt,
)

from src.probes.context_fatigue.attention_clamp import (
    SpanAttentionClamp,
    locate_token_span,
    measure_span_share,
    solve_span_scale,
)
from src.probes.context_fatigue.context_assembly import OverflowGuard
from src.probes.context_fatigue.ddxplus_cases import (
    format_case_question,
    format_case_vignette,
    load_evidence_db,
    load_probe_pool,
)
from src.probes.context_fatigue.instruction_checks import (
    INSTRUCTIONS,
    bundled_system_text,
    check_all,
)

BASE_SYSTEM = ("You are a doctor. Read each patient profile and pick the single most likely "
               "diagnosis. Reply with just the letter.")
# Indexed on the all-layer mean share, matching how the fill sweep measured it: pooled over 32
# layers the system span goes 0.2351 (cold) -> 0.0375 (90% fill). At layer 24 alone the same
# quantity reads 0.2891 -> 0.0413, and across layers the full-context value spans 0.0084-0.0631 —
# which is why the readout is pooled rather than pinned to a layer picked post-hoc.
# Ladder derived from THIS setup's own profile (--profile, all-layer readout): the system span
# holds 0.166 with no prior cases, 0.081 with one, and accumulation alone drives it to 0.021 by
# eight. One prior case is the floor for this design -- the demonstrated/undemonstrated arms need
# an assistant turn to differ in -- so natural sits near 0.081 and the ladder walks from just
# below it down to the level eight accumulated cases reach on their own.
LEVELS = [0.06, 0.045, 0.0375, 0.03, 0.021]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--levels", type=float, nargs="+", default=LEVELS)
    p.add_argument("--n-items", type=int, default=120)
    p.add_argument("--profile", action="store_true",
                   help="measure the system span's natural share as prior cases accumulate, and "
                        "exit. The clamp ladder must be derived from THIS setup: importing a "
                        "share measured under a different system prompt and context puts arms "
                        "above natural, which the preflight caught.")
    p.add_argument("--profile-cases", type=int, nargs="+", default=[0, 1, 2, 4, 6, 8],
                   help="prior-case counts to profile the natural system share at")
    p.add_argument("--arms", nargs="+", default=["demonstrated", "undemonstrated"],
                   help="whether the prior assistant turn exhibits the canaries. Undemonstrated "
                        "is the arm that matters: with a compliant example in context the model "
                        "can copy the pattern and never read the system prompt, so a clamp null "
                        "there would be uninterpretable.")
    p.add_argument("--cold-start-cases", type=int, default=1,
                   help="prior answered cases before the probe; kept small so the system span "
                        "sits at its natural cold-start share")
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--reference-layers", type=int, nargs="+", default=None,
                   help="layers the share is read from (default: every layer). The clamp biases "
                        "all layers regardless, so a single-layer readout is a post-hoc choice; "
                        "pass 24 to reproduce the earlier experiments' indexing.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/context_fatigue/e5_system_clamp")
    p.add_argument("--device", default="cuda")
    p.add_argument("--preflight", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = list(INSTRUCTIONS.values())
    system_text = bundled_system_text(BASE_SYSTEM, specs)

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # eager: the clamp needs an explicit additive mask, and sdpa optimizes a purely causal one
    # away. Cold-start contexts are short, so the materialized [1, H, N, N] is cheap here.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager").eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    is_chat = tokenizer.chat_template is not None

    reference_layers = (args.reference_layers if args.reference_layers is not None
                        else list(range(len(model.model.layers))))
    evidence_db = load_evidence_db()
    probe_pool = load_probe_pool(evidence_db, args.n_options, args.seed)
    print(f"  probe pool {len(probe_pool)} DDXPlus cases", flush=True)

    levels = args.levels[:2] if args.preflight else args.levels
    n_items = 1 if args.preflight else args.n_items

    guard = OverflowGuard(count_tokens=lambda t: len(tokenizer.encode(t)),
                          max_ctx=args.max_ctx, max_new=args.max_new, headroom=16)
    rng = random.Random(args.seed)
    records, skipped, above_natural = [], 0, 0
    turns_path = out_dir / "turns.csv"

    def build(probe, n_prior, arm, seeded_rng):
        """Cold-start conversation. ``arm`` decides whether the canaries are demonstrated.

        In the ``demonstrated`` arm the prior assistant turn exhibits the canaries, so the model
        can satisfy them by copying the pattern in context without consulting the system prompt at
        all. That is the in-context-learning crutch this project exists to remove, and it would
        make a clamp null uninterpretable. ``undemonstrated`` answers with the bare letter, so the
        system prompt is the only place the instruction appears.
        """
        conv = [{"role": "system", "content": system_text}]
        for prior in seeded_rng.sample(probe_pool, n_prior):
            answer = (f"◆ {prior['gold']}\n⟦OK⟧" if arm == "demonstrated" else prior["gold"])
            conv += [{"role": "user", "content": prior["vignette"] + format_case_question(
                prior["options"], args.n_options)},
                {"role": "assistant", "content": answer}]
        conv += [{"role": "user", "content": probe["vignette"] + format_case_question(
            probe["options"], args.n_options)}]
        return render_prompt(tokenizer, conv, is_chat)

    if args.profile:
        print("\nnatural system-span share vs accumulated prior cases (this setup):")
        for n_prior in args.profile_cases:
            shares = []
            for idx in range(min(12, len(probe_pool))):
                text = build(probe_pool[idx], n_prior, "undemonstrated",
                             random.Random(args.seed + idx))
                ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
                shares.append(measure_span_share(
                    model, ids, locate_token_span(tokenizer, text, system_text),
                    reference_layers))
            print(f"  {n_prior:2d} prior cases: share {sum(shares)/len(shares):.4f}", flush=True)
        return

    for idx in range(n_items):
        probe = probe_pool[idx % len(probe_pool)]
        # Every item runs in BOTH arms off the same prior-case draw, so the arms differ only
        # in whether the canaries are exhibited and the contrast stays paired over items.
        arm_seed = args.seed + 7919 * idx
        for arm in args.arms:
            text = build(probe, args.cold_start_cases, arm, random.Random(arm_seed))
            if not guard.fits(text, used=0, index=idx):
                skipped += 1
                continue
            ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
            span = locate_token_span(tokenizer, text, system_text)

            natural = measure_span_share(model, ids, span, reference_layers)

            def record(level, target, achieved, scale, response):
                pred = extract_mcq_answer(response) if response else None
                obeyed = check_all(response or "", specs)
                records.append({
                    "item": idx, "arm": arm, "level": level, "target_share": target,
                    "achieved_share": achieved, "scale": scale,
                    "bias_nats": None if scale is None else float(torch.log(torch.tensor(scale))),
                    "ctx_tokens": int(ids.shape[1]),
                    "fill": round(int(ids.shape[1]) / args.max_ctx, 4),
                    "gold": probe["gold"], "pred": pred, "correct": bool(pred == probe["gold"]),
                    "parsed": pred is not None,
                    "n_obeyed": sum(obeyed.values()),
                    **{f"obey_{k}": v for k, v in obeyed.items()},
                    "response": (response or "")[:200],
                })

            resp, _, _, _ = generate_with_entropy(
                model, tokenizer, text, args.device, args.max_new, args.max_ctx)
            record("natural", None, natural, 1.0, resp)

            for target in levels:
                if target >= natural:
                    # Clamping *upward* is a different experiment; silently doing it would put an
                    # arm above the natural share and label it a reduction.
                    above_natural += 1
                    continue
                scale, achieved = solve_span_scale(
                    model, ids, span=span, target_share=target,
                    reference_layer=reference_layers, tol=1e-3)
                with SpanAttentionClamp(model, span=span, scale=scale):
                    resp, _, _, _ = generate_with_entropy(
                        model, tokenizer, text, args.device, args.max_new, args.max_ctx)
                record(f"{target:.4f}", target, achieved, scale, resp)

            pd.DataFrame(records).to_csv(turns_path, index=False)
            if (idx + 1) % 10 == 0:
                df = pd.DataFrame(records)
                print(f"  {idx+1}/{n_items} items | "
                      + "  ".join(f"{lv}:obey={g['n_obeyed'].mean():.2f}"
                                  for lv, g in df.groupby("level")), flush=True)
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    by_level = {}
    for (arm, level), g in df.groupby(["arm", "level"]):
        by_level[f"{arm}/{level}"] = {
            "n": int(len(g)),
            "achieved_share": float(g["achieved_share"].mean()),
            "bias_nats": None if g["bias_nats"].isna().all() else float(g["bias_nats"].mean()),
            "accuracy": float(g["correct"].mean()),
            "parse_rate": float(g["parsed"].mean()),
            "mean_canaries_obeyed": float(g["n_obeyed"].mean()),
            **{f"obey_{s.name}": float(g[f"obey_{s.name}"].mean()) for s in specs},
        }
    summary = {"model": args.model, "seed": args.seed, "reference_layers": reference_layers,
               "system_span_text": system_text, "levels": levels,
               "n_items": n_items, "overflow_skips": skipped,
               "levels_skipped_above_natural": above_natural, "by_level": by_level}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print(f"E5 SYSTEM-MASS CLAMP — {args.model}")
    print("=" * 72)
    print(f"items {n_items} | overflow skips {skipped} | levels skipped (>= natural) "
          f"{above_natural}")
    for level, s in sorted(by_level.items(), key=lambda kv: -kv[1]["achieved_share"]):
        nats = "     -" if s["bias_nats"] is None else f"{s['bias_nats']:+6.2f}"
        print(f"  {level:>8s}  share={s['achieved_share']:.4f}  nats={nats}  "
              f"acc={s['accuracy']:.3f}  parsed={s['parse_rate']:.3f}  "
              f"canaries={s['mean_canaries_obeyed']:.2f}/3  "
              + " ".join(f"{sp.name[:6]}={s['obey_' + sp.name]:.2f}" for sp in specs))
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
