"""
DDXPlus Attention Analysis — Llama 3.1 8B

Same methodology as the Qwen attention script, but using Llama-specific
RoPE import. Hooks into target layers to capture post-RoPE Q@K for the
last token only (memory efficient).
"""

import json, re, ast, argparse, random, gc
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


SYSTEM_PROMPT = "You are a doctor."
OPTION_LABELS = ["A", "B", "C", "D", "E"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max-ctx", type=int, default=32768)
    p.add_argument("--max-new", type=int, default=50)
    p.add_argument("--fill-target", type=float, default=0.92)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--layers", default="0,7,15,31",
                   help="Layer indices to probe (Llama 8B has 32 layers)")
    p.add_argument("--out-dir", default="results/ddxplus_attention_llama")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ── evidence helpers ────────────────────────────────────────────────────

def load_evidence_db(path):
    with open(path) as f:
        raw = json.load(f)
    db = {}
    for code, info in raw.items():
        vm = {}
        for vk, vv in info.get("value_meaning", {}).items():
            vm[vk] = vv.get("en", str(vv)) if isinstance(vv, dict) else str(vv)
        db[code] = {
            "question": info.get("question_en", ""),
            "is_antecedent": info.get("is_antecedent", False),
            "data_type": info.get("data_type", "B"),
            "value_meanings": vm,
        }
    return db


def decode_evidence(ev_str, evidence_db):
    evs = ast.literal_eval(ev_str)
    symptoms, antecedents = [], []
    grouped = {}
    for ev in evs:
        if "@" in ev:
            base, val = ev.split("@", 1)
            grouped.setdefault(base.strip().rstrip("_"), []).append(val.strip())
        else:
            grouped[ev] = []
    for code, values in grouped.items():
        if code not in evidence_db:
            continue
        info = evidence_db[code]
        stmt = info["question"].replace("Do you have ", "Has ").replace("Are you ", "Is ").rstrip("?.")
        if info["data_type"] == "B":
            text = f"Yes — {stmt}"
        elif info["data_type"] == "M" and values:
            dec = [info["value_meanings"].get(v, v) for v in values if info["value_meanings"].get(v, v) != "NA"]
            text = f"{stmt}: {', '.join(dec)}" if dec else f"Yes — {stmt}"
        elif info["data_type"] == "C" and values:
            text = f"{stmt}: {', '.join(values)}"
        else:
            text = f"Yes — {stmt}"
        (antecedents if info["is_antecedent"] else symptoms).append(text)
    return symptoms, antecedents


def format_case_mcq(age, sex, initial_ev, evidence_str, evidence_db, options):
    sex_full = "Male" if sex == "M" else "Female"
    chief = evidence_db.get(initial_ev, {}).get("question", initial_ev).replace("Do you have ", "").replace("?", "").strip()
    symptoms, antecedents = decode_evidence(evidence_str, evidence_db)
    lines = [f"Patient: {age}-year-old {sex_full}", f"Chief complaint: {chief}"]
    if symptoms:
        lines.append("Symptoms:")
        lines.extend(f"  - {s}" for s in symptoms)
    if antecedents:
        lines.append("History:")
        lines.extend(f"  - {a}" for a in antecedents)
    lines.append("\nMost likely diagnosis:")
    lines.extend(f"{OPTION_LABELS[i]}) {opt}" for i, opt in enumerate(options[:5]))
    lines.append("\nAnswer:")
    return "\n".join(lines)


def extract_mcq_answer(text):
    text = text.strip().upper()
    if text and text[0] in "ABCDE":
        return text[0]
    m = re.search(r'\b([ABCDE])\b', text)
    return m.group(1) if m else None


# ── selective attention capture ─────────────────────────────────────────

class SelectiveAttentionCapture:
    def __init__(self, model, target_layers):
        self.target_layers = set(target_layers)
        self.captured = {}
        self.hooks = []
        self.enabled = False

        for li in target_layers:
            attn = model.model.layers[li].self_attn
            hook = attn.register_forward_pre_hook(self._make_hook(li), with_kwargs=True)
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, args, kwargs):
            if not self.enabled:
                return
            hidden_states = kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                return
            if hidden_states.shape[1] <= 1:
                return

            with torch.no_grad():
                q = module.q_proj(hidden_states)
                k = module.k_proj(hidden_states)

                batch, seq_len, _ = hidden_states.shape
                head_dim = module.head_dim
                n_q_heads = module.config.num_attention_heads
                n_kv_heads = module.config.num_key_value_heads

                q = q.view(batch, seq_len, n_q_heads, head_dim).transpose(1, 2)
                k = k.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)

                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                n_rep = n_q_heads // n_kv_heads
                if n_rep > 1:
                    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                        batch, n_q_heads, seq_len, head_dim)

                q_last = q[:, :, -1:, :]
                scaling = head_dim ** -0.5
                attn_scores = torch.matmul(q_last, k.transpose(-2, -1)) * scaling
                attn_weights = torch.softmax(attn_scores.float(), dim=-1)

                self.captured[layer_idx] = attn_weights[0, :, 0, :].cpu()
                del q, k, q_last, attn_scores, attn_weights

        return hook_fn

    def clear(self):
        self.captured = {}

    def remove(self):
        for h in self.hooks:
            h.remove()


# ── main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_layers = [int(x) for x in args.layers.split(",")]

    evidence_db = load_evidence_db("release_evidences.json")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    print(f"Architecture: {n_layers} layers, {n_heads} heads, {n_kv} KV heads")

    capture = SelectiveAttentionCapture(model, target_layers)

    def count_tokens(text):
        return len(tokenizer.encode(text))

    print("Loading DDXPlus test set...")
    ds = load_dataset("aai530-group6/ddxplus", split="test")
    rng = random.Random(args.seed)
    valid_indices = []
    for i in range(len(ds)):
        ddx = ast.literal_eval(ds[i]["DIFFERENTIAL_DIAGNOSIS"])
        if ds[i]["PATHOLOGY"] in [d[0] for d in ddx[:5]]:
            valid_indices.append(i)
    rng.shuffle(valid_indices)
    if args.max_cases:
        valid_indices = valid_indices[:args.max_cases]

    print(f"Probing layers: {target_layers}")
    print(f"\n{'='*70}")
    print(f"DDXPlus Attention — {args.model}")
    print(f"{'='*70}\n")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    all_records = []

    system_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False)
    system_tokens = count_tokens(system_text)

    for case_num, idx in enumerate(valid_indices):
        conv_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        ctx_now = count_tokens(conv_text)
        if ctx_now / args.max_ctx > args.fill_target:
            print(f"\n>>> Context {ctx_now/args.max_ctx:.0%} full — stopping.")
            break

        row = ds[idx]
        pathology = row["PATHOLOGY"]
        ddx = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        option_names = [d[0] for d in ddx[:5]]
        shuffled = list(enumerate(option_names))
        rng.shuffle(shuffled)
        shuffled_names = [n for _, n in shuffled]
        gold_pos = next(i for i, n in enumerate(shuffled_names) if n == pathology)
        gold_letter = OPTION_LABELS[gold_pos]

        case_text = format_case_mcq(
            row["AGE"], row["SEX"], row["INITIAL_EVIDENCE"],
            row["EVIDENCES"], evidence_db, shuffled_names)

        pre_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False)
        current_query_start = count_tokens(pre_text)

        conversation.append({"role": "user", "content": case_text})

        full_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(full_text, return_tensors="pt").to(args.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        mid_point = (system_tokens + current_query_start) // 2

        capture.clear()
        capture.enabled = True
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        capture.enabled = False
        del outputs

        with torch.no_grad():
            gen_out = model.generate(
                input_ids, max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        new_tokens = gen_out[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred = extract_mcq_answer(response)
        is_correct = pred == gold_letter if pred else False

        for layer_idx, last_attn in capture.captured.items():
            n_h = last_attn.shape[0]
            for h in range(n_h):
                a = last_attn[h]
                a_pos = a[a > 1e-10]
                entropy = -(a_pos * torch.log(a_pos)).sum().item()

                frac_system = a[:system_tokens].sum().item()
                frac_early = a[system_tokens:mid_point].sum().item()
                frac_recent = a[mid_point:current_query_start].sum().item()
                frac_current = a[current_query_start:].sum().item()
                peak_pos = a.argmax().item()

                all_records.append({
                    "case": case_num,
                    "context_fill": round(ctx_now / args.max_ctx, 4),
                    "context_tokens": ctx_now,
                    "seq_len": seq_len,
                    "layer": layer_idx,
                    "head": h,
                    "correct": is_correct,
                    "pathology": pathology,
                    "attention_entropy": entropy,
                    "frac_system": frac_system,
                    "frac_early_cases": frac_early,
                    "frac_recent_cases": frac_recent,
                    "frac_current_query": frac_current,
                    "peak_position": peak_pos,
                    "peak_relative": peak_pos / max(seq_len - 1, 1),
                })

        print(f"  C{case_num+1:3d} ctx={ctx_now:6,} ({ctx_now/args.max_ctx:5.1%}) | "
              f"pred={pred} gold={gold_letter} {'✓' if is_correct else '✗'}")
        for li in sorted(capture.captured.keys()):
            la = capture.captured[li]
            n_h = la.shape[0]
            ents = [-(la[h][la[h]>1e-10] * torch.log(la[h][la[h]>1e-10])).sum().item() for h in range(n_h)]
            curs = [la[h][current_query_start:].sum().item() for h in range(n_h)]
            print(f"    L{li:2d}: attn_ent={np.mean(ents):.3f} current={np.mean(curs):.3f}")

        conversation.append({"role": "assistant", "content": response})
        torch.cuda.empty_cache()
        gc.collect()

    capture.remove()

    # ── save & summarize ────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "attention_stats.csv", index=False)
    df.to_parquet(out_dir / "attention_stats.parquet", index=False)

    print(f"\n{'='*70}")
    print("Attention Summary by Layer × Context Fill")
    print(f"{'='*70}")

    for li in target_layers:
        ldf = df[df["layer"] == li]
        if ldf.empty:
            continue
        print(f"\nLayer {li}:")
        print(f"  {'Ctx Fill':12s} {'Attn Ent':>10s} {'Sys':>8s} {'Early':>8s} {'Recent':>8s} {'Current':>8s}")
        for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
            sub = ldf[(ldf["context_fill"] >= lo) & (ldf["context_fill"] < hi)]
            if sub.empty:
                continue
            print(f"  {lo:.0%}-{hi:.0%}       "
                  f"{sub['attention_entropy'].mean():10.3f} "
                  f"{sub['frac_system'].mean():8.3f} "
                  f"{sub['frac_early_cases'].mean():8.3f} "
                  f"{sub['frac_recent_cases'].mean():8.3f} "
                  f"{sub['frac_current_query'].mean():8.3f}")

    print(f"\nTotal: {df['case'].nunique()} cases, {len(df)} records")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
