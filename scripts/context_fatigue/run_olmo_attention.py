"""Context rot in OLMo-2: attention dynamics vs per-case performance.

As DDXPlus cases accumulate in one conversation, this captures — for every case
— where the last token's attention lands (system prompt / early cases / recent
cases / current query), its entropy, and its peak position, at several layers;
and pairs each case with whether the model answered it correctly.

The question: does context rot show up as a *measurable attention shift* (system-
prompt erosion, recency bias, current-query neglect), and does that shift track
*per-case* failure — i.e. on the cases where attention drifts most, is the model
more likely to get it wrong?

Attention is replicated exactly for OLMo-2: q_norm(q_proj(x)) → reshape → RoPE,
computed only for the last query token at target layers (cheap), matching the
module's own computation. OLMo-2 7B has no GQA (32 Q = 32 KV heads).
"""

import argparse
import ast
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb

from _cf_common import (
    OPTION_LABELS,
    extract_mcq_answer,
    format_case_mcq,
    load_evidence_db,
)

SYSTEM_PROMPT = "You are a doctor. Read each patient profile and pick the single most likely diagnosis. Reply with just the letter."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--fill-target", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--n-sessions", type=int, default=1,
                   help="Repeat the accumulation N times with different case orders and "
                        "pool all cases — needed for per-case performance power (each "
                        "4096-ctx session only holds ~11 cases).")
    p.add_argument("--layers", default="0,8,16,24,31")
    p.add_argument("--out-dir", default="results/olmo_attention")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ── selective OLMo-2 attention capture (last-token, target layers) ──────

class Olmo2AttentionCapture:
    def __init__(self, model, target_layers):
        self.captured = {}
        self.enabled = False
        self.hooks = []
        for li in target_layers:
            attn = model.model.layers[li].self_attn
            self.hooks.append(attn.register_forward_pre_hook(self._mk(li), with_kwargs=True))

    def _mk(self, li):
        def hook(module, args, kwargs):
            if not self.enabled:
                return
            hs = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", None)
            if hs is None or pe is None or hs.shape[1] <= 1:
                return
            with torch.no_grad():
                b, seq, _ = hs.shape
                hd = module.head_dim
                q = module.q_norm(module.q_proj(hs)).view(b, seq, -1, hd).transpose(1, 2)
                k = module.k_norm(module.k_proj(hs)).view(b, seq, -1, hd).transpose(1, 2)
                cos, sin = pe
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                q_last = q[:, :, -1:, :]                                  # [b, H, 1, hd]
                scores = torch.matmul(q_last, k.transpose(-2, -1)) * (hd ** -0.5)
                w = torch.softmax(scores.float(), dim=-1)                 # [b, H, 1, seq]
                self.captured[li] = w[0, :, 0, :].cpu()                   # [H, seq]
        return hook

    def clear(self):
        self.captured = {}

    def remove(self):
        for h in self.hooks:
            h.remove()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_layers = [int(x) for x in args.layers.split(",")]

    evidence_db = load_evidence_db()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    capture = Olmo2AttentionCapture(model, target_layers)

    def n_tokens(conv, gen_prompt):
        return len(tokenizer.encode(tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=gen_prompt)))

    print("Loading DDXPlus...")
    ds = load_dataset("aai530-group6/ddxplus", split="test")
    valid = [i for i in range(len(ds))
             if ds[i]["PATHOLOGY"] in
             [d[0] for d in ast.literal_eval(ds[i]["DIFFERENTIAL_DIAGNOSIS"])[:args.n_options]]]

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    system_tokens = n_tokens(conversation, False)
    print(f"Probing layers {target_layers} | system region = {system_tokens} tokens | "
          f"{args.n_sessions} session(s)\n")

    records = []
    for session in range(args.n_sessions):
        # Fresh case order and option order per session so pooled cases are independent.
        srng = random.Random(args.seed + 1000 * session)
        order = list(valid)
        srng.shuffle(order)
        if args.max_cases:
            order = order[:args.max_cases]
        opt_rng = random.Random(args.seed + 1 + session)

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        sess_correct = []
        for case_num, idx in enumerate(order):
            ctx_now = n_tokens(conversation, True)
            if ctx_now / args.max_ctx > args.fill_target:
                break

            row = ds[idx]
            pathology = row["PATHOLOGY"]
            ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
            names = [d[0] for d in ddx[:args.n_options]]
            shuffled = [n for _, n in sorted(enumerate(names), key=lambda x: opt_rng.random())]
            gold_letter = OPTION_LABELS[shuffled.index(pathology)]
            case_text = format_case_mcq(row["AGE"], row["SEX"], row["INITIAL_EVIDENCE"],
                                        row["EVIDENCES"], evidence_db, shuffled, args.n_options)

            current_query_start = n_tokens(conversation, False)
            mid_point = (system_tokens + current_query_start) // 2

            conversation.append({"role": "user", "content": case_text})
            full = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(full, return_tensors="pt", truncation=True, max_length=args.max_ctx).input_ids.to(args.device)
            seq_len = ids.shape[1]

            capture.clear()
            capture.enabled = True
            with torch.no_grad():
                model(ids, use_cache=False)
            capture.enabled = False

            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
            pred = extract_mcq_answer(resp)
            correct = pred == gold_letter if pred else False
            sess_correct.append(correct)

            for li, attn in capture.captured.items():
                for h in range(attn.shape[0]):
                    a = attn[h]
                    ap = a[a > 1e-10]
                    ent = -(ap * torch.log(ap)).sum().item()
                    records.append({
                        "session": session, "case": case_num,
                        "context_fill": round(ctx_now / args.max_ctx, 4),
                        "context_tokens": ctx_now, "seq_len": seq_len, "layer": li, "head": h,
                        "correct": correct, "pathology": pathology,
                        "attention_entropy": ent,
                        "frac_system": a[:system_tokens].sum().item(),
                        "frac_early_cases": a[system_tokens:mid_point].sum().item(),
                        "frac_recent_cases": a[mid_point:current_query_start].sum().item(),
                        "frac_current_query": a[current_query_start:].sum().item(),
                        "peak_relative": a.argmax().item() / max(seq_len - 1, 1),
                    })

            conversation.append({"role": "assistant", "content": resp})
            torch.cuda.empty_cache()
            gc.collect()

        print(f"  session {session+1}/{args.n_sessions}: {len(sess_correct)} cases, "
              f"acc={np.mean(sess_correct):.2f}", flush=True)
        # Crash-safe: dump pooled records after every session.
        pd.DataFrame(records).to_csv(out_dir / "attention_stats.csv", index=False)

    capture.remove()
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "attention_stats.csv", index=False)

    analyze(df, target_layers, args.model, out_dir)


METRICS = ["frac_system", "frac_early_cases", "frac_recent_cases", "frac_current_query", "attention_entropy"]


def analyze(df, target_layers, model_name, out_dir):
    # Per-(session,case), per-layer aggregate (mean over heads) for correlation analysis.
    keys = ["layer", "session", "case"] if "session" in df.columns else ["layer", "case"]
    per_case = (df.groupby(keys)
                  .agg({**{m: "mean" for m in METRICS},
                        "context_fill": "first", "correct": "first"})
                  .reset_index())

    print(f"\n{'='*78}\nCONTEXT ROT — attention dynamics by context fill ({model_name})\n{'='*78}")
    for li in target_layers:
        ld = df[df["layer"] == li]
        if ld.empty:
            continue
        print(f"\nLayer {li}:  {'fill':>9s} {'sys':>7s} {'early':>7s} {'recent':>7s} {'current':>8s} {'attn_ent':>9s}")
        for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
            s = ld[(ld["context_fill"] >= lo) & (ld["context_fill"] < hi)]
            if s.empty:
                continue
            print(f"           {lo:>4.0%}-{hi:<4.0%}{s['frac_system'].mean():7.3f} "
                  f"{s['frac_early_cases'].mean():7.3f} {s['frac_recent_cases'].mean():7.3f} "
                  f"{s['frac_current_query'].mean():8.3f} {s['attention_entropy'].mean():9.3f}")

    # Per-case performance link: how attention relates to context fill and to correctness.
    print(f"\n{'='*78}\nATTENTION ↔ PERFORMANCE (per-case, mean over heads)\n{'='*78}")
    print(f"{'layer':>6s} | corr-with-context_fill        | correct vs incorrect (Δ = wrong−right)")
    print(f"{'':>6s} | sys     recent  current  ent     | n(✓/✗)   current(✓→✗)     entropy(✓→✗)")
    summary = []
    for li in target_layers:
        pc = per_case[per_case["layer"] == li]
        if len(pc) < 4:
            continue

        def corr(col):
            if pc[col].std() < 1e-9 or pc["context_fill"].std() < 1e-9:
                return float("nan")
            return float(np.corrcoef(pc[col], pc["context_fill"])[0, 1])

        c_sys, c_rec = corr("frac_system"), corr("frac_recent_cases")
        c_cur, c_ent = corr("frac_current_query"), corr("attention_entropy")
        right, wrong = pc[pc["correct"]], pc[~pc["correct"]]
        cur_r = right["frac_current_query"].mean() if len(right) else float("nan")
        cur_w = wrong["frac_current_query"].mean() if len(wrong) else float("nan")
        ent_r = right["attention_entropy"].mean() if len(right) else float("nan")
        ent_w = wrong["attention_entropy"].mean() if len(wrong) else float("nan")
        print(f"{li:6d} | {c_sys:+5.2f}  {c_rec:+5.2f}  {c_cur:+5.2f}  {c_ent:+5.2f}  | "
              f"{len(right):d}/{len(wrong):d}     {cur_r:.3f}→{cur_w:.3f}    {ent_r:.2f}→{ent_w:.2f}")
        summary.append({
            "layer": li, "model": model_name,
            "corr_system_fill": c_sys, "corr_recent_fill": c_rec,
            "corr_current_fill": c_cur, "corr_entropy_fill": c_ent,
            "n_correct": int(len(right)), "n_incorrect": int(len(wrong)),
            "current_correct": cur_r, "current_incorrect": cur_w,
            "entropy_correct": ent_r, "entropy_incorrect": ent_w,
        })

    pd.DataFrame(summary).to_csv(out_dir / "attention_performance.csv", index=False)

    # Fill-controlled performance link: within each fill quartile, is current-query
    # attention lower on the cases the model gets WRONG? (removes the fill confound)
    print(f"\n{'='*78}\nFILL-CONTROLLED: current-query attention, correct vs incorrect (layer 24)\n{'='*78}")
    pc24 = per_case[per_case["layer"] == 24]
    if len(pc24) >= 8:
        print(f"{'fill bin':>12s} {'n(✓/✗)':>10s} {'current ✓':>10s} {'current ✗':>10s} {'Δ(✗−✓)':>9s}")
        for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
            b = pc24[(pc24["context_fill"] >= lo) & (pc24["context_fill"] < hi)]
            r, w = b[b["correct"]], b[~b["correct"]]
            if len(r) and len(w):
                cr, cw = r["frac_current_query"].mean(), w["frac_current_query"].mean()
                print(f"  {lo:>4.0%}-{hi:<4.0%} {len(r):>3d}/{len(w):<3d}   {cr:>10.3f} {cw:>10.3f} {cw-cr:>+9.3f}")
            elif len(b):
                print(f"  {lo:>4.0%}-{hi:<4.0%} {len(r):>3d}/{len(w):<3d}   (need both classes in bin)")

    print(f"\nInterpretation: negative corr-system-fill = system-prompt erosion; "
          f"positive corr-recent-fill = recency bias; negative corr-current-fill = current-\n"
          f"query neglect; positive corr-entropy-fill = attention diffuses. The fill-controlled "
          f"table isolates whether attention drift predicts errors *within* a context level.")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
