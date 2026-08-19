"""E1 — within-harness distance sweep: does dilution bite when the evidence is *not* local?

The paper's null is that accumulating context costs no accuracy when every answer is locally
self-contained. The first reviewer objection is that the task is simply too easy to degrade. E1
answers it inside our own harness by varying one thing — **where the answer-bearing evidence
lives** — with items, model, fill and metric held fixed.

Design
------
Each session accumulates short MMLU filler turns. At fixed depths the transcript is snapshotted,
and for every snapshot each DDXPlus probe case is asked once per arm:

    local | back_2 | back_5 | back_10 | back_20

The probe's vignette is placed that many user turns before its question; the question text is
byte-identical in every arm, carries an explicit referent back to the patient, and the filler is
*the same* across arms at a given snapshot — so the arms differ in evidence position and nothing
else.

Why the filler is MMLU rather than more DDXPlus cases
-----------------------------------------------------
A full DDXPlus case averages 309 tokens and OLMo-2's window is 4096, so only ~13 turns fit:
``back_20`` would not exist and ``back_10`` would only occur above ~85% fill, welding distance to
fill. §6 asks for a joint fit in which distance is significant and fill is not, which collinear
predictors cannot deliver. Short filler (~74 tokens) puts 21 turns at ~43% fill and lets depth and
fill move independently. The *probe* is still a full DDXPlus case.

Every fill-dependent arm runs the overflow guard, which skips rather than truncates and logs what
it skipped — truncating a long item near the window edge manufactures exactly the late-window dip
this paper attributes to accumulation.
"""

import argparse
import gc
import json
import random
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import (
    RowAppender,
    extract_mcq_answer,
    generate_with_entropy,
    per_head_rows,
    render_prompt,
)

from src.probes.context_fatigue.attention_capture import SelectiveAttentionCapture
from src.probes.context_fatigue.attention_clamp import locate_token_span, span_share
from src.probes.context_fatigue.context_assembly import OverflowGuard, assemble_transcript
from src.probes.context_fatigue.ddxplus_cases import (
    format_case_question,
    format_case_vignette,
    load_evidence_db,
    load_probe_pool,
)

MMLU_LABELS = ["A", "B", "C", "D"]
INTRO = ("You are answering a mixed stream of questions. Some are multiple-choice knowledge "
         "questions; some describe a patient. Answer each one as it arrives.")
REFERENT = "For the patient described earlier"
ARMS = {"local": 0, "back_2": 2, "back_5": 5, "back_10": 10, "back_20": 20}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=32)  # 8 truncates a preamble before the letter
    p.add_argument("--headroom", type=int, default=16)
    p.add_argument("--max-filler-tokens", type=int, default=90)
    p.add_argument("--depths", type=int, nargs="+", default=[21, 28, 35, 42],
                   help="user-turn counts at which to snapshot the filler")
    p.add_argument("--probes-per-cell", type=int, default=8)
    p.add_argument("--n-sessions", type=int, default=6)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/context_fatigue/e1_distance_sweep")
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference-layer", type=int, default=24)
    p.add_argument("--measure-attention", action="store_true",
                   help="record attention mass on the evidence span at --reference-layer")
    p.add_argument("--per-head", action="store_true",
                   help="also write heads.csv: one row per probe x arm x head x layer. Implies "
                        "--attention-only, since the per-head shares come off the same forward.")
    p.add_argument("--head-layers", type=int, nargs="+", default=None,
                   help="layers to record per-head shares at (default: every layer). Extra layers "
                        "are free -- they are read off the same forward as the reference layer.")
    p.add_argument("--attention-only", action="store_true",
                   help="skip generation; measure attention only (fast)")
    p.add_argument("--preflight", action="store_true",
                   help="run one cell end-to-end, write it, and exit")
    return p.parse_args()


def format_mmlu(question, choices):
    return (question + "\n"
            + "".join(f"{MMLU_LABELS[i]}) {o}\n" for i, o in enumerate(choices))
            + "\nReply with only the letter (A, B, C, or D).")


def load_filler_pool(tokenizer, max_tokens):
    ds = load_dataset("cais/mmlu", "all", split="test")
    pool = []
    for row in ds:
        text = format_mmlu(row["question"], row["choices"])
        if len(tokenizer.encode(text)) <= max_tokens:
            pool.append({"text": text, "gold": MMLU_LABELS[row["answer"]],
                         "subject": row["subject"]})
    return pool


def main():
    args = parse_args()
    if args.per_head:
        args.attention_only = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    is_chat = tokenizer.chat_template is not None

    def n_tokens(conv):
        return len(tokenizer.encode(render_prompt(tokenizer, conv, is_chat)))

    evidence_db = load_evidence_db()
    filler_pool = load_filler_pool(tokenizer, args.max_filler_tokens)
    probe_pool = load_probe_pool(evidence_db, args.n_options, args.seed)
    print(f"  filler pool {len(filler_pool)} MMLU items <= {args.max_filler_tokens} tok | "
          f"probe pool {len(probe_pool)} DDXPlus cases", flush=True)

    depths = args.depths[:1] if args.preflight else args.depths
    sessions = 1 if args.preflight else args.n_sessions
    per_cell = 1 if args.preflight else args.probes_per_cell

    guard = OverflowGuard(count_tokens=lambda t: len(tokenizer.encode(t)),
                          max_ctx=args.max_ctx, max_new=args.max_new, headroom=args.headroom)
    records = []
    n_probe_attempts = 0
    turns_path = out_dir / "turns.csv"
    capture_layers = sorted({args.reference_layer, *(args.head_layers if args.head_layers
                                                     else range(len(model.model.layers)))})
    heads = RowAppender(out_dir / "heads.csv") if args.per_head else None

    for session in range(sessions):
        rng = random.Random(args.seed + 1000 * session)
        filler = rng.sample(filler_pool, min(max(depths) + 5, len(filler_pool)))
        probes = rng.sample(probe_pool, min(per_cell * len(depths), len(probe_pool)))

        conv = [{"role": "user", "content": INTRO},
                {"role": "assistant", "content": "Understood."}]
        snapshots = {}
        for item in filler:
            conv = conv + [{"role": "user", "content": item["text"]}]
            resp, _, _, _ = generate_with_entropy(
                model, tokenizer, render_prompt(tokenizer, conv, is_chat),
                args.device, args.max_new, args.max_ctx)
            conv = conv + [{"role": "assistant", "content": (resp or "A").strip()[:8]}]
            n_user = sum(1 for t in conv if t["role"] == "user") - 1  # exclude the intro turn
            if n_user in depths and n_user not in snapshots:
                snapshots[n_user] = list(conv)
                print(f"  [s{session}] snapshot at {n_user} filler turns "
                      f"({n_tokens(conv)} tok)", flush=True)
            if len(snapshots) == len(depths):
                break

        for depth_idx, depth in enumerate(depths):
            base = snapshots.get(depth)
            if base is None:
                continue
            base_tokens = n_tokens(base)
            cell_probes = probes[depth_idx * per_cell:(depth_idx + 1) * per_cell]
            for probe in cell_probes:
                n_probe_attempts += 1
                question = format_case_question(probe["options"], args.n_options,
                                                referent=REFERENT)
                if not guard.fits(probe["vignette"] + question, used=base_tokens,
                                  index=n_probe_attempts):
                    continue
                for arm, distance in ARMS.items():
                    try:
                        built = assemble_transcript(base, evidence=probe["vignette"],
                                                    question=question, distance=distance)
                    except ValueError:  # transcript too shallow for this arm
                        continue
                    rendered = render_prompt(tokenizer, built.turns, is_chat)

                    attn_cols = {}
                    if args.measure_attention or args.attention_only:
                        # The dilution question E1 could not answer on its own: does attention on
                        # the *displaced evidence* fall with distance, and does that track the
                        # accuracy drop? Without this the ladder isolates position, not mass.
                        ids = tokenizer(rendered, return_tensors="pt").input_ids.to(args.device)
                        capture = SelectiveAttentionCapture(model, capture_layers)
                        capture.enabled = True
                        with torch.no_grad():
                            model(ids)
                        capture.remove()
                        attn = capture.captured[args.reference_layer]
                        ev_span = locate_token_span(tokenizer, rendered, probe["vignette"])
                        q_span = locate_token_span(tokenizer, rendered, question)
                        attn_cols = {
                            "evidence_share": span_share(attn, ev_span),
                            "question_share": span_share(attn, q_span),
                            "evidence_tokens": ev_span[1] - ev_span[0],
                        }
                        if heads is not None:
                            for li in capture_layers:
                                heads.extend(per_head_rows(
                                    capture.captured[li],
                                    {"evidence": ev_span, "question": q_span},
                                    probe=n_probe_attempts, arm=arm, distance=distance,
                                    session=session, filler_turns=depth, layer=li,
                                    pathology=probe["pathology"]))

                    if args.attention_only:
                        resp, ctx_len, entropy, pred = None, int(ids.shape[1]), None, None
                    else:
                        resp, ctx_len, entropy, _ = generate_with_entropy(
                            model, tokenizer, rendered, args.device, args.max_new, args.max_ctx)
                        pred = extract_mcq_answer(resp) if resp else None
                    records.append({**attn_cols,
                        "session": session, "arm": arm, "distance": distance,
                        "filler_turns": depth, "context_fill": round(ctx_len / args.max_ctx, 4),
                        "ctx_tokens": ctx_len, "pathology": probe["pathology"],
                        "gold": probe["gold"], "pred": pred,
                        "correct": bool(pred == probe["gold"]), "mean_entropy": entropy,
                        "parsed": pred is not None, "response": (resp or "")[:200],
                    })
                    torch.cuda.empty_cache()
                # per-cell write: a killed session keeps every completed cell
                pd.DataFrame(records).to_csv(turns_path, index=False)

        done = len(records)
        acc = pd.DataFrame(records)["correct"].mean() if done else float("nan")
        print(f"  [s{session}] {done} rows so far, running acc={acc:.3f}", flush=True)

    if heads is not None:
        heads.flush()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    summary = {
        "model": args.model,
        "max_ctx": args.max_ctx,
        "arms": ARMS,
        "depths": depths,
        "n_sessions": sessions,
        "n_rows": len(df),
        "overflow": guard.report(n_seen=n_probe_attempts),
        "by_arm": (df.groupby("arm")
                     .agg(n=("correct", "size"), accuracy=("correct", "mean"),
                          mean_fill=("context_fill", "mean"))
                     .reset_index().to_dict("records")) if len(df) else [],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*66}\nE1 DISTANCE SWEEP — {args.model}\n{'='*66}")
    print(f"overflow guard: skipped {summary['overflow']['n_skipped']}/{n_probe_attempts} probes "
          f"({summary['overflow']['skip_rate']:.1%})")
    for row in summary["by_arm"]:
        print(f"  {row['arm']:9s} n={row['n']:4d}  acc={row['accuracy']:.3f}  "
              f"mean_fill={row['mean_fill']:.2f}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
