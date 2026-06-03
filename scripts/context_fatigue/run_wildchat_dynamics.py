"""Entropy/attention depth dynamics on organic dialogue (WildChat-1M).

Replicates the DDXPlus context-rot probes (own-confidence entropy collapse; attention
reallocation away from the current turn toward recent turns) on real, heterogeneous,
topic-shifting conversations — the clean test of whether those signals are genuine
context-depth effects or artifacts of one task repeating (ICL comfort).

For each filtered conversation we walk every user->assistant boundary in the *real*
history (teacher-forced; no generation drift) and, with one ``generate`` call per
boundary, capture:
  - own-confidence entropy: mean next-token Shannon entropy over a short greedy probe
    of the model's OWN continuation (tokens discarded), matching the DDXPlus metric;
  - last-query-token attention split into first / middle / recent / current buckets.

Run from the repo root (WildChat-1M streams without a token):

    uv run python -m scripts.context_fatigue.run_wildchat_dynamics \
        --model Qwen/Qwen2.5-7B-Instruct --min-turns 10 --n-convs 150

    # smoke
    uv run python -m scripts.context_fatigue.run_wildchat_dynamics \
        --min-turns 6 --n-convs 3 --max-ctx 8192 --max-boundaries 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.context_fatigue.attention_capture import (
    SelectiveAttentionCapture,
    attention_distribution_entropy,
)
from src.probes.context_fatigue.instruction_checks import pearson
from src.probes.context_fatigue.wildchat_data import (
    assistant_boundary_indices,
    attention_segments,
    count_user_turns,
    normalize_conversation,
    passes_language_turn_filter,
    segment_fractions,
)
from src.probes.context_fatigue.wildchat_homogeneity import (
    consecutive_homogeneity,
    homogeneity_score,
    user_messages,
)

FIRST_TOKENS = 8  # opening-token (attention-sink) bucket width


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=32768)
    p.add_argument("--min-turns", type=int, default=10)
    p.add_argument("--n-convs", type=int, default=150)
    p.add_argument("--max-boundaries", type=int, default=20,
                   help="cap probed turns per conversation (runtime bound)")
    p.add_argument("--probe-k", type=int, default=8,
                   help="own-confidence entropy: tokens of greedy probe per boundary")
    p.add_argument("--layers", default="0,7,14,21,27")
    p.add_argument("--language", default="English")
    p.add_argument("--scan-limit", type=int, default=30000,
                   help="max WildChat rows to scan while collecting --n-convs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/context_fatigue/wildchat_dynamics")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def collect_conversations(tokenizer, args):
    """Stream WildChat, keep English convs with >= min_turns whose full render fits ctx."""
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    kept, scanned = [], 0
    for row in ds:
        scanned += 1
        if scanned > args.scan_limit:
            break
        if not passes_language_turn_filter(row, args.min_turns, args.language):
            continue
        msgs = normalize_conversation(row["conversation"])
        if count_user_turns(msgs) < args.min_turns:
            continue
        if len(assistant_boundary_indices(msgs)) < args.min_turns:
            continue
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        n_tok = len(tokenizer.encode(full))
        if n_tok > args.max_ctx:
            continue
        users = user_messages(msgs)
        meta = {"homogeneity": homogeneity_score(users),
                "consecutive_homogeneity": consecutive_homogeneity(users),
                "n_user_turns": len(users), "total_tokens": n_tok}
        kept.append((row["conversation_hash"], msgs, meta))
        if len(kept) >= args.n_convs:
            break
    print(f"Collected {len(kept)} conversations (scanned {scanned} rows).")
    return kept


def probe_boundary(model, tokenizer, capture, msgs, i, target_layers, args):
    """One user->assistant boundary: returns (turn_record, [attention_records])."""
    def ntok(ms, gen):
        if not ms:
            return 0
        return len(tokenizer.encode(
            tokenizer.apply_chat_template(ms, tokenize=False, add_generation_prompt=gen)))

    text = tokenizer.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
    seq_len = ids.shape[1]
    if seq_len + args.probe_k + 1 > args.max_ctx:
        return None, []

    current_start = ntok(msgs[:i - 1], False)            # context before current user turn
    recent_start = ntok(msgs[:i - 3], False) if i >= 3 else min(FIRST_TOKENS, current_start)
    first_end = min(FIRST_TOKENS, current_start)
    segments = attention_segments(seq_len, first_end, recent_start, current_start)

    capture.clear()
    capture.enabled = True
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.probe_k, do_sample=False,
                             return_dict_in_generate=True, output_scores=True,
                             pad_token_id=tokenizer.eos_token_id)
    capture.enabled = False

    ents = []
    for score in out.scores:
        probs = torch.softmax(score.float(), dim=-1)
        ents.append(float(-(probs * torch.log_softmax(score.float(), dim=-1)).sum()))
    probe_entropy = float(np.mean(ents)) if ents else 0.0

    turn_rec = {"context_tokens": seq_len,
                "context_fill": round(seq_len / args.max_ctx, 4),
                "probe_entropy": probe_entropy}
    attn_recs, head_recs = [], []
    for li in target_layers:
        if li not in capture.captured:
            continue
        per_head = capture.captured[li]  # (n_heads, seq_len)
        head_mean = per_head.mean(0)
        fr = segment_fractions(head_mean.tolist(), segments)
        attn_recs.append({"layer": li,
                          "attn_entropy": attention_distribution_entropy(head_mean),
                          **fr})
        for h in range(per_head.shape[0]):
            vec = per_head[h]
            head_recs.append({
                "layer": li, "head": h,
                "frac_current": segment_fractions(vec.tolist(), segments)["frac_current"],
                "attn_entropy": attention_distribution_entropy(vec)})
    return turn_rec, attn_recs, head_recs


def corr_table(df, value_cols, depth_col):
    """Pooled corr of each value column with a depth axis, per layer if present."""
    rows = []
    layers = sorted(df["layer"].unique()) if "layer" in df.columns else [None]
    for li in layers:
        sub = df if li is None else df[df["layer"] == li]
        rec = {"layer": int(li) if li is not None else None, "n": len(sub)}
        for c in value_cols:
            rec[f"corr_{c}_{depth_col}"] = pearson(sub[depth_col].tolist(), sub[c].tolist())
        rows.append(rec)
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_layers = [int(x) for x in args.layers.split(",")]

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    capture = SelectiveAttentionCapture(model, target_layers)

    convs = collect_conversations(tokenizer, args)

    turn_rows, attn_rows, head_rows, conv_rows = [], [], [], []
    for ci, (chash, msgs, meta) in enumerate(convs):
        conv_rows.append({"conv": chash, **meta})
        boundaries = assistant_boundary_indices(msgs)[: args.max_boundaries]
        for depth, i in enumerate(boundaries):
            turn_rec, attn_recs, head_recs = probe_boundary(
                model, tokenizer, capture, msgs, i, target_layers, args)
            if turn_rec is None:
                break  # too long from here on
            base = {"conv": chash, "depth": depth}
            turn_rows.append({**base, "homogeneity": meta["homogeneity"], **turn_rec})
            ctx = {"context_tokens": turn_rec["context_tokens"],
                   "context_fill": turn_rec["context_fill"]}
            for ar in attn_recs:
                attn_rows.append({**base, **ctx, **ar})
            for hr in head_recs:
                head_rows.append({**base, **ctx, **hr})
        torch.cuda.empty_cache()
        if (ci + 1) % 25 == 0:
            print(f"  {ci + 1}/{len(convs)} conversations probed")

    turn_df = pd.DataFrame(turn_rows)
    attn_df = pd.DataFrame(attn_rows)
    pd.DataFrame(conv_rows).to_csv(out_dir / "conversations.csv", index=False)
    pd.DataFrame(head_rows).to_parquet(out_dir / "attention_heads.parquet", index=False)
    turn_df.to_csv(out_dir / "turns.csv", index=False)
    attn_df.to_csv(out_dir / "attention.csv", index=False)

    fracs = ["frac_first", "frac_middle", "frac_recent", "frac_current", "attn_entropy"]
    summary = {
        "model": args.model, "min_turns": args.min_turns, "n_convs": len(convs),
        "n_boundaries": len(turn_df), "max_ctx": args.max_ctx, "probe_k": args.probe_k,
        "entropy_vs_depth": corr_table(turn_df, ["probe_entropy"], "depth"),
        "entropy_vs_tokens": corr_table(turn_df, ["probe_entropy"], "context_tokens"),
        "attention_vs_depth": corr_table(attn_df, fracs, "depth"),
    }
    # depth-binned entropy (early vs late) to compare against the DDXPlus 3-4x collapse
    if len(turn_df):
        d = turn_df["depth"]
        early = turn_df[d <= d.quantile(0.2)]["probe_entropy"].mean()
        late = turn_df[d >= d.quantile(0.8)]["probe_entropy"].mean()
        summary["entropy_early_late"] = {
            "early_mean": float(early), "late_mean": float(late),
            "ratio_early_over_late": float(early / late) if late else None}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/  ({len(turn_df)} boundaries, {len(convs)} convs)")
    if "entropy_early_late" in summary:
        el = summary["entropy_early_late"]
        print(f"Own-confidence entropy: early={el['early_mean']:.3f} late={el['late_mean']:.3f} "
              f"ratio={el['ratio_early_over_late']}")
    print("entropy corr vs depth (pooled):",
          f"{summary['entropy_vs_depth'][0]['corr_probe_entropy_depth']:+.3f}")
    print("attention frac corr vs depth, per layer:")
    for r in summary["attention_vs_depth"]:
        print(f"  L{r['layer']:2d} current={r['corr_frac_current_depth']:+.3f} "
              f"recent={r['corr_frac_recent_depth']:+.3f} first={r['corr_frac_first_depth']:+.3f}")


if __name__ == "__main__":
    main()
