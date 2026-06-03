"""
knowing_saying_gap_pipeline.py
===============================
Complete end-to-end pipeline for the Knowing-Saying Gap research.

WHAT IT DOES (6 stages, run any subset):
  Stage 1: Collect activations + surface signals (entropy, logprob, vocab gap)
  Stage 2: Train layer-wise probes, find best layer, save weights
  Stage 3: Behavioral analysis (hedging, verbalization, overconfidence)
  Stage 4: Intervention experiment (baseline, reprompt, replace, branch+pick)
  Stage 5: Sensitivity ablation (sweep K and threshold)
  Stage 6: Generate all paper figures

KEY DESIGN DECISIONS:
  - ALL-TOKEN probing: activations extracted at EVERY token position in the
    prompt, then aggregated (mean, max, last). This gives richer signal than
    last-token-only and allows the persistence analysis.
  - MODEL-AGNOSTIC: works with any HuggingFace causal LM. Pass --model.
  - SINGLE FILE: one script, one dataset, one output directory.
  - CHECKPOINTS: saves every N traces so crashed runs can resume.

USAGE:
    # Full pipeline (stages 1-6)
    python knowing_saying_gap_pipeline.py \
        --dataset contrastive_math_dataset.json \
        --model Qwen/Qwen2.5-3B \
        --stages 1,2,3,4,5,6 \
        --n_probe 100 --n_intervene 300 --n_sensitivity 140 \
        --output_dir results/

    # Just probe training + figures (if you already have activations)
    python knowing_saying_gap_pipeline.py \
        --stages 2,6 --output_dir results/

    # Just intervention (requires probe weights from stage 2)
    python knowing_saying_gap_pipeline.py \
        --stages 4,6 --output_dir results/ --n_intervene 300

VAST.AI SETUP:
    pip install transformers accelerate bitsandbytes scikit-learn \
        matplotlib scipy pandas numpy seaborn
    scp -P PORT knowing_saying_gap_pipeline.py root@ssh.vast.ai:/root/
    scp -P PORT contrastive_math_dataset.json  root@ssh.vast.ai:/root/
"""

import json, os, re, sys, random, time, warnings, argparse, gc
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PROBE_CV_FOLDS  = 5
MAX_NEW_TOKENS  = 200
MAX_INPUT_LEN   = 768
BRANCH_K_VALUES = [1, 2, 4, 8]
TAU_MODES       = ["conservative", "youden", "loose"]
CHECKPOINT_EVERY = 20

HEDGE_PATTERNS = [
    r"i(?:'m| am) not (?:sure|certain|confident)",
    r"i(?:'m| am) unsure",
    r"i(?:'m| am) uncertain",
    r"not (?:entirely|completely|fully|totally) sure",
    r"(?:may|might|could) be (?:wrong|incorrect|mistaken|off)",
    r"(?:possibly|perhaps|probably) (?:wrong|incorrect|mistaken)",
    r"(?:hard|difficult) to (?:say|tell|know)",
    r"can'?t (?:be )?(?:sure|certain|confident)",
    r"(?:this|the answer) (?:may|might|could) (?:be )?(?:wrong|incorrect|off|inaccurate)",
    r"(?:double.?check|verify|confirm)",
    r"(?:not|without) (?:full|complete|high|much) confidence",
    r"(?:low|limited|reduced) confidence",
    r"(?:my|this) (?:estimate|answer|calculation) (?:may|might|could) be",
    r"(?:approximate|rough|ballpark)",
    r"if i.{0,15}(?:reading|understanding|interpreting).{0,20}correctly",
    r"assuming (?:the prior|the previous|my earlier|this is correct)",
    r"(?:the prior|previous) (?:step|answer|result) (?:may|might|could) .{0,20}(?:error|mistake|issue)",
]

OVERCONF_PATTERNS = [
    r"i(?:'m| am)\s+(?:absolutely|completely|totally|fully|entirely|highly|very)?\s*(?:sure|certain|confident|positive)",
    r"100\s*%\s*(?:sure|certain|confident|positive)",
    r"(?:absolutely|definitely|certainly|undoubtedly)\s+(?:correct|right|the answer|true)",
    r"(?:no|without|beyond)\s+(?:any\s+)?doubt",
    r"\bclear(?:ly)?\b",
    r"\bobviously\b",
    r"the answer\s+is\s+(?:simply|just|clearly|obviously)",
]

ABSURDITY_PATTERNS = [
    r"-\s*\$?\d", r"\bnegative\b", r"minus\s+\d",
    r"\bnot\s+possible\b", r"\bimpossible\b", r"\bcannot\s+be\b",
    r"doesn'?t\s+make\s+sense", r"\binvalid\b",
]

VERBALIZED_SYSTEM_PROMPT = (
    "You are solving a multi-step math problem. After each answer, "
    "write your confidence on its own line in the exact format:\n"
    "Confidence: N/10\n"
    "where N is 0 (completely unsure) to 10 (certain)."
)

ISOLATED_SYSTEM_PROMPT = "Solve this math problem. Give only the numeric answer."


# ═══════════════════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="Knowing-Saying Gap: unified pipeline")
    p.add_argument("--dataset",    default="contrastive_math_dataset.json")
    p.add_argument("--model",      default="Qwen/Qwen2.5-3B",
                   help="Any HuggingFace causal LM name or path")
    p.add_argument("--stages",     default="1,2,3,4,6",
                   help="Comma-separated stage numbers to run (1-6)")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--n_probe",    type=int, default=100,
                   help="Base problems for stages 1-3 (probe + behavioral)")
    p.add_argument("--n_intervene",type=int, default=300,
                   help="Base problems for stage 4 (intervention)")
    p.add_argument("--n_sensitivity", type=int, default=140,
                   help="Base problems for stage 5 (sensitivity)")
    p.add_argument("--no_4bit",    action="store_true")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--resume",     action="store_true",
                   help="Resume from last checkpoint if available")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# TEXT ANALYSIS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def detect_hedging(text):
    tl = text.lower()
    matched = [p for p in HEDGE_PATTERNS if re.search(p, tl)]
    return {"hedged": len(matched) > 0, "hedge_count": len(matched),
            "hedge_score": min(1.0, len(matched) / 2.0)}

def detect_overconfidence(text):
    tl = text.lower()
    matched = [p for p in OVERCONF_PATTERNS if re.search(p, tl)]
    return {"overconfident": len(matched) > 0, "overconf_count": len(matched)}

def is_absurd(text):
    tl = text.lower()
    return any(re.search(p, tl) for p in ABSURDITY_PATTERNS)

def check_correct(generated, expected):
    nums = re.findall(r"[-+]?\d*\.?\d+", str(expected))
    if not nums:
        return None
    try:
        target = float(nums[-1])
    except ValueError:
        return None
    markers = [r"(?:final\s+answer|the\s+answer|answer\s+is|=\s*)\s*\$?([-+]?\d*\.?\d+)"]
    candidates = []
    for m in markers:
        candidates.extend(re.findall(m, generated.lower()))
    if not candidates:
        all_nums = re.findall(r"[-+]?\d*\.?\d+", generated)
        if all_nums:
            candidates = [all_nums[-1]]
    for c in reversed(candidates):
        try:
            v = float(c)
            if abs(v - target) < 1e-3:
                return True
            if target != 0 and abs(v - target) / abs(target) < 0.01:
                return True
        except ValueError:
            pass
    return False

def extract_final_num(text):
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return nums[-1] if nums else None

def extract_verbalized_conf(text):
    tl = text.lower()
    patterns = [
        (r"confidence[:\s]+(\d+(?:\.\d+)?)\s*/\s*10\b", 10),
        (r"confidence[:\s]+(\d+(?:\.\d+)?)\s*/\s*100\b", 100),
        (r"(\d+(?:\.\d+)?)\s+out\s+of\s+10\b", 10),
        (r"i(?:'m| am)\s+(\d+(?:\.\d+)?)\s*%\s*(?:sure|certain|confident)", 100),
        (r"(\d+(?:\.\d+)?)\s*%\s*(?:sure|certain|confident)", 100),
        (r"confidence[:\s]+(\d+(?:\.\d+)?)\b", 10),
    ]
    for pat, scale in patterns:
        m = re.search(pat, tl)
        if m:
            try:
                v = float(m.group(1))
                return min(1.0, max(0.0, v / (100.0 if (scale == 100 or v > 10) else 10.0)))
            except ValueError:
                pass
    if re.search(r"i(?:'m| am) (?:very |absolutely |completely )?(?:sure|certain|confident)", tl):
        return 0.95
    if re.search(r"i(?:'m| am) (?:not sure|unsure|uncertain)", tl):
        return 0.15
    return None

def logprob_uncertainty(confs):
    if not confs:
        return 0.0
    return float(np.mean([-np.log(max(c, 1e-10)) for c in confs]))


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING (works with any HuggingFace causal LM)
# ═══════════════════════════════════════════════════════════════════════════

def load_model(name, no_4bit=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"  Loading {name}...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if no_4bit:
        mdl = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True)
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        mdl = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=bnb, device_map="auto",
            trust_remote_code=True)
    mdl.eval()
    n_layers = mdl.config.num_hidden_layers
    d_model  = mdl.config.hidden_size
    print(f"  Loaded: {n_layers} layers, d_model={d_model}")
    return tok, mdl, n_layers, d_model


def get_device(model):
    return next(model.parameters()).device


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION WITH ALL-TOKEN ACTIVATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def generate_with_signals(model, tokenizer, prompt, n_layers,
                          max_new=MAX_NEW_TOKENS, max_input=MAX_INPUT_LEN,
                          extract_acts=True):
    """
    Generate text and extract ALL signals:
      - Per-token logprob, entropy, vocab gap during generation
      - Residual stream activations at ALL prompt token positions (not just last)
        for every layer. Shape: (n_layers+1, seq_len, d_model)

    Returns dict with:
      text, confidences, entropies, vocab_gaps,
      acts_all_tokens (n_layers+1, prompt_len, d) if extract_acts else None,
      acts_last_token (n_layers+1, d) — the classic last-token activation,
      acts_mean_token (n_layers+1, d) — mean over all prompt tokens,
      prompt_len
    """
    import torch

    enc = tokenize_for_model(tokenizer, prompt, max_input)
    enc = {k: v.to(get_device(model)) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    S = input_ids.shape[1]
    device = input_ids.device

    acts_all = None
    acts_last = None
    acts_mean = None

    # Initial forward over the prompt. Captures KV cache so the generation loop
    # only has to feed one new token per step. Also optionally extracts hidden
    # states at every prompt position for probe training.
    with torch.no_grad():
        out = model(**enc, use_cache=True, output_hidden_states=extract_acts)
        past_kv = out.past_key_values
        logits = out.logits[0, -1, :]

        if extract_acts:
            hs = torch.stack(out.hidden_states, dim=0)  # (n_layers+1, 1, seq, d)
            acts_all  = hs[:, 0, :, :].float().cpu().numpy()
            acts_last = hs[:, 0, S-1, :].float().cpu().numpy()
            acts_mean = hs[:, 0, :, :].float().mean(dim=1).cpu().numpy()
            del hs
        del out
    torch.cuda.empty_cache()

    # Autoregressive generation: single-token forwards reusing the KV cache.
    confs, ents, vgaps, gen_ids = [], [], [], []
    with torch.no_grad():
        for _ in range(max_new):
            probs = torch.softmax(logits, dim=-1)
            top2 = torch.topk(probs, k=2)
            confs.append(top2.values[0].item())
            ents.append(-(probs * torch.log(probs + 1e-10)).sum().item())
            vgaps.append(top2.values[0].item() - top2.values[1].item())
            nxt_id = int(logits.argmax(dim=-1).item())
            gen_ids.append(nxt_id)
            if len(gen_ids) > 5 and "\n" in tokenizer.decode([nxt_id]):
                break
            nxt = torch.tensor([[nxt_id]], device=device)
            out = model(input_ids=nxt, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1, :]
            del out
    del past_kv
    torch.cuda.empty_cache()

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "text": text,
        "confidences": confs,
        "entropies": ents,
        "vocab_gaps": vgaps,
        "acts_all_tokens": acts_all,    # (n_layers+1, prompt_len, d)
        "acts_last_token": acts_last,   # (n_layers+1, d)
        "acts_mean_token": acts_mean,   # (n_layers+1, d)
        "prompt_len": S,
    }


def probe_score_from_acts(acts_vec, probe_W, probe_b, probe_mu, probe_sc):
    """Score a single activation vector with a pre-fitted probe."""
    x = (acts_vec - probe_mu) / (probe_sc + 1e-10)
    return float(1.0 / (1.0 + np.exp(-(np.dot(x, probe_W) + probe_b))))


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS (chat-template aware)
# ═══════════════════════════════════════════════════════════════════════════

def _has_chat_template(tokenizer):
    return getattr(tokenizer, "chat_template", None) is not None

def format_messages(messages, tokenizer, add_generation_prompt=True):
    """Render a messages list to a prompt string. Uses the tokenizer's chat
    template when available; falls back to a plain few-shot concatenation."""
    if _has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(m["content"] + "\n\n")
        else:
            parts.append(m["content"] + "\n")
    text = "".join(parts)
    if add_generation_prompt and messages and messages[-1]["role"] == "user":
        text = text + "Answer:"
    return text

def tokenize_for_model(tokenizer, prompt, max_input=MAX_INPUT_LEN):
    """Tokenize a (possibly chat-templated) prompt, avoiding double-BOS when
    the chat template already injects special tokens."""
    return tokenizer(prompt, return_tensors="pt",
                     max_length=max_input, truncation=True,
                     add_special_tokens=not _has_chat_template(tokenizer))

def _resolve_response(hop, overrides, i):
    if overrides and i in overrides:
        return overrides[i]
    if hop["is_injected"] and hop.get("injected_response"):
        return hop["injected_response"]
    return hop["correct_response"]

def build_hop_messages(trace, hop_idx, overrides=None):
    messages = []
    for i, hop in enumerate(trace["hops"]):
        if i < hop_idx:
            resp = _resolve_response(hop, overrides, i)
            messages.append({"role": "user", "content": f"Step {i+1}: {hop['prompt']}"})
            messages.append({"role": "assistant", "content": f"Answer: {resp}"})
        else:
            messages.append({"role": "user", "content": f"Step {i+1}: {hop['prompt']}"})
            break
    return messages

def build_final_messages(trace):
    messages = []
    for i, hop in enumerate(trace["hops"]):
        resp = _resolve_response(hop, None, i)
        if i < len(trace["hops"]) - 1:
            messages.append({"role": "user", "content": f"Step {i+1}: {hop['prompt']}"})
            messages.append({"role": "assistant", "content": f"Answer: {resp}"})
        else:
            messages.append({"role": "user", "content": f"Step {i+1}: {hop['prompt']}"})
    return messages

def build_verbalized_messages(trace, hop_idx):
    messages = [{"role": "system", "content": VERBALIZED_SYSTEM_PROMPT}]
    for i, hop in enumerate(trace["hops"]):
        if i < hop_idx:
            resp = _resolve_response(hop, None, i)
            messages.append({"role": "user", "content": f"Step {i+1}: {hop['prompt']}"})
            messages.append({"role": "assistant",
                             "content": f"Answer: {resp}\nConfidence: 10/10"})
        else:
            messages.append({"role": "user", "content": f"Step {i+1}: {hop['prompt']}"})
            break
    return messages

def build_isolated_messages(hop):
    return [
        {"role": "system", "content": ISOLATED_SYSTEM_PROMPT},
        {"role": "user", "content": hop["prompt"]},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: COLLECT ACTIVATIONS + SIGNALS
# ═══════════════════════════════════════════════════════════════════════════

def stage1_collect(args, dataset, by_base, model, tokenizer, n_layers):
    """Collect activations and surface signals for probe training."""
    print("\n" + "="*65)
    print("  STAGE 1: Collect activations + surface signals")
    print("="*65)

    out = Path(args.output_dir)
    all_ids = list(by_base.keys())
    sampled = all_ids[:args.n_probe]

    records = []
    # Store activations: lists of (n_layers+1, d) arrays
    act_last_list = []
    act_mean_list = []
    act_labels    = []
    t0 = time.time()

    for idx, base_id in enumerate(sampled):
        variants = by_base[base_id]
        clean = next((t for t in variants if t["variant"] == "clean"), None)
        error = next((t for t in variants if t["variant"] == "error_at_1"), None)
        if not clean or not error:
            continue

        elapsed = time.time() - t0
        eta = (elapsed / max(idx, 1)) * (len(sampled) - idx)
        print(f"  [{idx+1:3d}/{len(sampled)}] {base_id} "
              f"({clean['hop_depth']}h) {elapsed/60:.0f}m ETA {eta/60:.0f}m", flush=True)

        correct_answer = clean["correct_final_answer"]

        # Three conditions: clean, error_standard, error_verbalized
        conditions = [
            ("clean",            clean, format_messages(build_hop_messages(clean, 1), tokenizer),        0),
            ("error_standard",   error, format_messages(build_hop_messages(error, 1), tokenizer),        1),
            ("error_verbalized", error, format_messages(build_verbalized_messages(error, 1), tokenizer), 1),
        ]

        # Also get final answers for the full chain
        final_clean_prompt = format_messages(build_final_messages(clean), tokenizer)
        final_error_prompt = format_messages(build_final_messages(error), tokenizer)
        res_fc = generate_with_signals(model, tokenizer, final_clean_prompt,
                                        n_layers, extract_acts=False)
        res_fe = generate_with_signals(model, tokenizer, final_error_prompt,
                                        n_layers, extract_acts=False)
        final_correct_clean = check_correct(res_fc["text"], correct_answer)
        final_correct_error = check_correct(res_fe["text"], correct_answer)

        for cond_name, trace, prompt, label in conditions:
            res = generate_with_signals(model, tokenizer, prompt, n_layers,
                                         extract_acts=True)
            hedge = detect_hedging(res["text"])
            overc = detect_overconfidence(res["text"])
            verb  = extract_verbalized_conf(res["text"]) if cond_name == "error_verbalized" else None
            ne = min(20, len(res["confidences"]))

            rec = {
                "base_id":       base_id,
                "condition":     cond_name,
                "label":         label,
                "hop_depth":     trace["hop_depth"],
                "error_type":    trace.get("injected_error_type"),
                # Surface signals
                "peak_entropy":       float(np.max(res["entropies"])) if res["entropies"] else 0.0,
                "mean_entropy":       float(np.mean(res["entropies"])) if res["entropies"] else 0.0,
                "early_entropy":      float(np.mean(res["entropies"][:ne])) if res["entropies"] else 0.0,
                "min_confidence":     float(np.min(res["confidences"])) if res["confidences"] else 1.0,
                "mean_vocab_gap":     float(np.mean(res["vocab_gaps"])) if res["vocab_gaps"] else 1.0,
                "logprob_uncertainty":logprob_uncertainty(res["confidences"]),
                # Behavioral
                "hedged":             hedge["hedged"],
                "hedge_count":        hedge["hedge_count"],
                "overconfident":      overc["overconfident"],
                "overconf_count":     overc["overconf_count"],
                "verbalized_conf":    verb,
                "is_absurd":          is_absurd(res["text"]),
                # Downstream
                "final_correct_clean":final_correct_clean,
                "final_correct_error":final_correct_error,
                "final_degraded":     (final_correct_clean is True and final_correct_error is False),
                "prompt_len":         res["prompt_len"],
                "n_tokens_gen":       len(res["confidences"]),
                "generated_text":     res["text"],
            }
            records.append(rec)

            # Store activations (both last-token and mean-over-all-tokens)
            act_last_list.append(res["acts_last_token"])    # (n_layers+1, d)
            act_mean_list.append(res["acts_mean_token"])    # (n_layers+1, d)
            act_labels.append(label)

        # Checkpoint
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            _save_stage1(out, records, act_last_list, act_mean_list, act_labels)
            print(f"    [checkpoint at {idx+1}]")

    _save_stage1(out, records, act_last_list, act_mean_list, act_labels)
    print(f"  Stage 1 complete: {len(records)} records, {len(act_labels)} activation sets")
    return records

def _save_stage1(out, records, act_last, act_mean, labels):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "stage1_records.csv", index=False)
    np.save(out / "act_last.npy", np.stack(act_last, axis=0))
    np.save(out / "act_mean.npy", np.stack(act_mean, axis=0))
    np.save(out / "act_labels.npy", np.array(labels))


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: TRAIN LAYER-WISE PROBES
# ═══════════════════════════════════════════════════════════════════════════

def stage2_probes(args):
    """Fit logistic probes at every layer, two aggregation modes (last vs mean)."""
    from cuml.linear_model import LogisticRegression
    from cuml.preprocessing import StandardScaler
    from cuml.pipeline import Pipeline

    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_score,
        cross_val_predict
    )

    from sklearn.metrics import roc_auc_score, roc_curve

    print("\n" + "="*65)
    print("  STAGE 2: Train layer-wise probes")
    print("="*65)

    out = Path(args.output_dir)
    act_last   = np.load(out / "act_last.npy")     # (N, n_layers+1, d)
    act_mean   = np.load(out / "act_mean.npy")     # (N, n_layers+1, d)
    labels     = np.load(out / "act_labels.npy")    # (N,)
    n_layers_p1 = act_last.shape[1]
    print(f"  Data: {len(labels)} samples, {n_layers_p1} layers (incl embedding)")

    kf = StratifiedKFold(n_splits=PROBE_CV_FOLDS, shuffle=True, random_state=args.seed)
    results = {"layer": [], "mode": [], "auroc_mean": [], "auroc_std": [],
               "acc_mean": [], "acc_std": []}

    best_auroc = 0
    best_layer = 0
    best_mode  = "last"

    for mode, act_arr in [("last", act_last), ("mean", act_mean)]:
        print(f"\n  --- {mode}-token aggregation ---")
        for li in range(n_layers_p1):
            pipe = Pipeline([("s", StandardScaler()),
                             ("c", LogisticRegression(max_iter=2000, C=1.0))])
            aurocs = cross_val_score(pipe, act_arr[:, li, :], labels,
                                      cv=kf, scoring="roc_auc")
            accs   = cross_val_score(pipe, act_arr[:, li, :], labels,
                                      cv=kf, scoring="accuracy")
            results["layer"].append(li)
            results["mode"].append(mode)
            results["auroc_mean"].append(aurocs.mean())
            results["auroc_std"].append(aurocs.std())
            results["acc_mean"].append(accs.mean())
            results["acc_std"].append(accs.std())

            if li % 5 == 0:
                print(f"  L{li:2d} ({mode}): AUROC={aurocs.mean():.4f}±{aurocs.std():.4f}")

            if aurocs.mean() > best_auroc:
                best_auroc = aurocs.mean()
                best_layer = li
                best_mode  = mode

    probe_df = pd.DataFrame(results)
    probe_df.to_csv(out / "layer_probe_results.csv", index=False)
    print(f"\n  Best: L{best_layer} ({best_mode}) AUROC={best_auroc:.4f}")

    # Fit final probe at best layer + best mode
    act_best = act_last if best_mode == "last" else act_mean
    scaler = StandardScaler()
    X = scaler.fit_transform(act_best[:, best_layer, :])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
    clf.fit(X, labels)

    # Get out-of-fold scores for threshold calibration
    pipe_final = Pipeline([("s", StandardScaler()),
                           ("c", LogisticRegression(max_iter=2000, C=1.0,
                                                     random_state=args.seed))])
    oof_scores = cross_val_predict(pipe_final, act_best[:, best_layer, :], labels,
                                     cv=kf, method="predict_proba")[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, oof_scores)
    youden_j = tpr - fpr
    youden_idx = int(np.argmax(youden_j))
    youden_tau = float(thresholds[youden_idx])
    cons_tau   = float(np.percentile(oof_scores[labels == 0], 99))

    # Save probe weights
    np.savez(out / "probe_weights.npz",
             W=clf.coef_[0], b=clf.intercept_[0],
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             layer=best_layer, mode=best_mode)

    threshold_data = {
        "threshold": youden_tau,
        "youden_J": float(youden_j[youden_idx]),
        "tpr_at_threshold": float(tpr[youden_idx]),
        "fpr_at_threshold": float(fpr[youden_idx]),
        "conservative_threshold": cons_tau,
        "cv_auroc": best_auroc,
        "layer": best_layer,
        "mode": best_mode,
    }
    with open(out / "probe_threshold.json", "w") as f:
        json.dump(threshold_data, f, indent=2)

    print(f"  Threshold (youden): {youden_tau:.4f}  TPR={tpr[youden_idx]:.3f}  FPR={fpr[youden_idx]:.3f}")
    print(f"  Threshold (conservative): {cons_tau:.4f}")
    print(f"  Saved: probe_weights.npz, probe_threshold.json, layer_probe_results.csv")

    return probe_df, threshold_data


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: BEHAVIORAL ANALYSIS (prints summary, no new model runs needed)
# ═══════════════════════════════════════════════════════════════════════════

def stage3_behavioral(args):
    """Analyse the stage1 records for H3 (failure prediction) and H4 (verbalization)."""
    from sklearn.metrics import roc_auc_score
    print("\n" + "="*65)
    print("  STAGE 3: Behavioral analysis (H3 + H4)")
    print("="*65)

    out = Path(args.output_dir)
    df = pd.read_csv(out / "stage1_records.csv")

    # Load probe scores if available (attach to df)
    probe_path = out / "probe_weights.npz"
    if probe_path.exists():
        pw = np.load(probe_path)
        layer = int(pw["layer"])
        mode  = str(pw.get("mode", "last"))
        act_file = "act_last.npy" if mode == "last" else "act_mean.npy"
        acts = np.load(out / act_file)
        scores = []
        for i in range(len(acts)):
            scores.append(probe_score_from_acts(
                acts[i, layer, :], pw["W"], pw["b"], pw["scaler_mean"], pw["scaler_scale"]))
        df["probe_score"] = scores
    else:
        print("  WARNING: No probe weights found, skipping probe score attachment")

    # H3: Detection vs failure prediction
    print("\n  H3: Detection vs failure prediction")
    y_label = df["label"].values

    signals = []
    for sig, name in [("probe_score", "Activation Probe"),
                       ("logprob_uncertainty", "Logprob Uncertainty"),
                       ("peak_entropy", "Peak Entropy"),
                       ("mean_vocab_gap", "Vocab Gap (mean)")]:
        if sig not in df.columns:
            continue
        vals = df[sig].values
        try:
            auroc_det = roc_auc_score(y_label, vals)
        except:
            auroc_det = 0.5

        # Failure prediction: among error traces, does signal predict final wrong?
        err = df[df["condition"] == "error_standard"].copy()
        err["final_wrong"] = (~err["final_correct_error"].fillna(True).astype(bool)).astype(int)
        if err["final_wrong"].nunique() > 1 and sig in err.columns:
            try:
                auroc_fail = roc_auc_score(err["final_wrong"], err[sig])
            except:
                auroc_fail = 0.5
        else:
            auroc_fail = 0.5

        signals.append({"signal": sig, "name": name,
                         "auroc_detection": auroc_det,
                         "auroc_failure": auroc_fail})
        print(f"    {name:30s}: detect={auroc_det:.4f}  failure={auroc_fail:.4f}")

    pd.DataFrame(signals).to_csv(out / "h3_comparison.csv", index=False)

    # H4: Verbalization gap
    print("\n  H4: Verbalization gap")
    verb = df[df["condition"] == "error_verbalized"]
    err_std = df[df["condition"] == "error_standard"]

    hedge_rate_std = err_std["hedged"].fillna(False).mean()
    hedge_rate_verb = verb["hedged"].fillna(False).mean()
    overconf_rate = err_std["overconfident"].fillna(False).mean()

    parsed = verb["verbalized_conf"].dropna()
    compliance = len(parsed) / len(verb) if len(verb) > 0 else 0

    print(f"    Hedging rate (error_standard): {hedge_rate_std:.1%}")
    print(f"    Hedging rate (error_verb):     {hedge_rate_verb:.1%}")
    print(f"    Overconfidence rate:           {overconf_rate:.1%}")
    print(f"    Verbalized conf compliance:    {compliance:.0%} ({len(parsed)}/{len(verb)})")
    if len(parsed) > 0:
        vals = parsed.values
        binary = ((vals <= 0.05) | (vals >= 0.95)).sum()
        print(f"    Binary collapse (0/10 or 10/10): {binary}/{len(vals)} ({binary/len(vals):.0%})")
        print(f"    Unique values: {sorted(set(round(v,2) for v in vals))}")

        # Absurd crossover
        abs_idx = verb[verb["is_absurd"] == True].index
        pla_idx = verb[verb["is_absurd"] == False].index
        abs_parsed = verb.loc[abs_idx, "verbalized_conf"].dropna()
        pla_parsed = verb.loc[pla_idx, "verbalized_conf"].dropna()
        if len(abs_parsed) > 0:
            abs_low = (abs_parsed < 0.5).mean()
            print(f"    Absurd outputs → conf < 50%: {abs_low:.0%} (n={len(abs_parsed)})")
        if len(pla_parsed) > 0:
            pla_low = (pla_parsed < 0.5).mean()
            print(f"    Plausible outputs → conf < 50%: {pla_low:.0%} (n={len(pla_parsed)})")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: INTERVENTION EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

class RuntimeProbe:
    """Loads saved probe weights for real-time scoring during generation."""
    def __init__(self, weights_path, threshold_path):
        pw = np.load(weights_path)
        self.W     = pw["W"]
        self.b     = float(pw["b"])
        self.mu    = pw["scaler_mean"]
        self.sc    = pw["scaler_scale"]
        self.layer = int(pw["layer"])
        self.mode  = str(pw.get("mode", "last"))

        with open(threshold_path) as f:
            t = json.load(f)
        self.tau      = float(t["threshold"])
        self.tau_cons = float(t["conservative_threshold"])
        self.tau_loose= self.tau + (1 - self.tau) * 0.3

    def score(self, acts_vec):
        return probe_score_from_acts(acts_vec, self.W, self.b, self.mu, self.sc)

    def fires(self, s, mode="youden"):
        thresholds = {"conservative": self.tau_cons, "youden": self.tau,
                      "loose": self.tau_loose}
        return s >= thresholds[mode]


def get_probe_input(model, tokenizer, prompt, probe):
    """Get the activation vector the probe needs (respecting probe.mode)."""
    import torch
    enc = tokenize_for_model(tokenizer, prompt)
    enc = {k: v.to(get_device(model)) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)
        if probe.mode == "mean":
            acts = hs[probe.layer, 0, :, :].float().mean(dim=0).cpu().numpy()
        else:
            acts = hs[probe.layer, 0, -1, :].float().cpu().numpy()
        del out, hs
    del enc
    torch.cuda.empty_cache()
    gc.collect()
    return acts


def run_intervention_chain(trace, model, tokenizer, probe, condition,
                            fire_mode="youden", branch_k=4):
    """Run one trace through one condition, return result dict."""
    import torch
    hop_depth = len(trace["hops"])
    probe_scores = []
    overrides = {}
    n_interv = 0

    for hop_idx in range(hop_depth):
        messages = build_hop_messages(trace, hop_idx, overrides)
        prompt = format_messages(messages, tokenizer)
        acts = get_probe_input(model, tokenizer, prompt, probe)
        ps = probe.score(acts)
        probe_scores.append(ps)
        fired = probe.fires(ps, fire_mode)

        if condition == "baseline" or not fired or n_interv >= 3:
            text = _greedy(model, tokenizer, prompt)
        elif condition == "reprompt":
            nudge_msgs = build_hop_messages(trace, hop_idx, overrides)
            nudge_msgs[-1]["content"] += " [Some steps may contain errors. Recheck.]"
            text = _greedy(model, tokenizer, format_messages(nudge_msgs, tokenizer))
            n_interv += 1
        elif condition == "replace_prior" and hop_idx > 0:
            iso = format_messages(build_isolated_messages(trace["hops"][hop_idx - 1]), tokenizer)
            fresh = _greedy(model, tokenizer, iso)
            overrides[hop_idx - 1] = extract_final_num(fresh) or fresh.strip()
            n_interv += 1
            messages2 = build_hop_messages(trace, hop_idx, overrides)
            prompt2 = format_messages(messages2, tokenizer)
            acts2 = get_probe_input(model, tokenizer, prompt2, probe)
            probe_scores[-1] = probe.score(acts2)
            text = _greedy(model, tokenizer, prompt2)
        elif condition == "branch_and_pick":
            branches = []
            temps = [0.7, 0.85, 1.0, 1.15][:branch_k]
            for temp in temps:
                bt = _sample(model, tokenizer, prompt, temp)
                bp_msgs = messages + [{"role": "assistant", "content": bt}]
                bp_prompt = format_messages(bp_msgs, tokenizer, add_generation_prompt=False)
                bp_acts = get_probe_input(model, tokenizer, bp_prompt, probe)
                branches.append({"text": bt, "score": probe.score(bp_acts)})
            best = min(branches, key=lambda b: b["score"])
            text = best["text"]
            probe_scores[-1] = best["score"]
            n_interv += 1
        else:
            text = _greedy(model, tokenizer, prompt)

    return {
        "final_text": text,
        "n_interv": n_interv,
        "max_probe": float(np.max(probe_scores)),
        "mean_probe": float(np.mean(probe_scores)),
        "fired_any": any(probe.fires(s, fire_mode) for s in probe_scores),
    }


def _greedy(model, tokenizer, prompt):
    import torch
    enc = tokenize_for_model(tokenizer, prompt)
    enc = {k: v.to(get_device(model)) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(ids[0, in_len:], skip_special_tokens=True)
    del enc, ids
    torch.cuda.empty_cache()
    gc.collect()
    return text.split("\n")[0].strip() if "\n" in text else text.strip()


def _sample(model, tokenizer, prompt, temperature=0.9):
    import torch
    enc = tokenize_for_model(tokenizer, prompt)
    enc = {k: v.to(get_device(model)) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                              temperature=temperature, top_p=0.95,
                              pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(ids[0, in_len:], skip_special_tokens=True)
    del enc, ids
    torch.cuda.empty_cache()
    gc.collect()
    return text.split("\n")[0].strip() if "\n" in text else text.strip()


def stage4_intervention(args, dataset, by_base, model, tokenizer, n_layers):
    print("\n" + "="*65)
    print("  STAGE 4: Intervention experiment")
    print("="*65)

    out = Path(args.output_dir)
    probe = RuntimeProbe(str(out / "probe_weights.npz"),
                          str(out / "probe_threshold.json"))
    print(f"  Probe: L{probe.layer} ({probe.mode}), tau={probe.tau:.4f}")

    all_ids = list(by_base.keys())
    sampled = all_ids[:args.n_intervene]
    records = []
    t0 = time.time()

    for idx, base_id in enumerate(sampled):
        variants = by_base[base_id]
        clean = next((t for t in variants if t["variant"] == "clean"), None)
        error = next((t for t in variants if t["variant"] == "error_at_1"), None)
        if not clean or not error:
            continue

        elapsed = time.time() - t0
        eta = (elapsed / max(idx, 1)) * (len(sampled) - idx)
        print(f"  [{idx+1:3d}/{len(sampled)}] {clean['hop_depth']}h "
              f"{elapsed/60:.0f}m ETA {eta/60:.0f}m", flush=True)

        correct = clean["correct_final_answer"]
        for cond in ["baseline", "reprompt", "replace_prior", "branch_and_pick"]:
            try:
                r = run_intervention_chain(error, model, tokenizer, probe, cond)
                records.append({
                    "base_id": base_id, "condition": cond,
                    "hop_depth": clean["hop_depth"],
                    "error_type": error.get("injected_error_type"),
                    "final_correct": check_correct(r["final_text"], correct),
                    "final_absurd": is_absurd(r["final_text"]),
                    "max_probe_score": r["max_probe"],
                    "mean_probe_score": r["mean_probe"],
                    "fired_at_least_once": r["fired_any"],
                    "n_interventions": r["n_interv"],
                })
            except Exception as e:
                print(f"    ! {cond}: {e}")
                # On CUDA error, try to recover
                import torch
                if "CUDA" in str(e) or "cuda" in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(2)  # let GPU cool

        # Aggressive cleanup after each trace
        import torch
        torch.cuda.empty_cache()
        gc.collect()

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(out / "intervention_records.csv", index=False)
            print(f"    [checkpoint at {idx+1}]")

    pd.DataFrame(records).to_csv(out / "intervention_records.csv", index=False)
    _print_intervention_summary(pd.DataFrame(records))
    return records


def _print_intervention_summary(df):
    from scipy.stats import binom
    print("\n  INTERVENTION RESULTS:")
    pivot = df.pivot_table(index="base_id", columns="condition",
                            values="final_correct", aggfunc="first").dropna()
    bc = pivot["baseline"].astype(bool)
    for cond in ["baseline", "reprompt", "replace_prior", "branch_and_pick"]:
        if cond not in pivot.columns:
            continue
        s = df[df["condition"] == cond]
        n = s["final_correct"].notna().sum()
        c = s["final_correct"].fillna(False).astype(bool).sum()
        extra = ""
        if cond != "baseline":
            cc = pivot[cond].astype(bool)
            rescued = (~bc & cc).sum()
            broken = (bc & ~cc).sum()
            extra = f"  rescued={rescued} broken={broken} net={rescued-broken:+d}"
        print(f"    {cond:<18}: {c:3d}/{n:3d} = {c/n:.1%}{extra}")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5: SENSITIVITY ABLATION
# ═══════════════════════════════════════════════════════════════════════════

def stage5_sensitivity(args, dataset, by_base, model, tokenizer, n_layers):
    print("\n" + "="*65)
    print("  STAGE 5: Sensitivity ablation (K x threshold)")
    print("="*65)

    out = Path(args.output_dir)
    probe = RuntimeProbe(str(out / "probe_weights.npz"),
                          str(out / "probe_threshold.json"))

    all_ids = list(by_base.keys())
    sampled = all_ids[:args.n_sensitivity]
    records = []
    t0 = time.time()

    for idx, base_id in enumerate(sampled):
        variants = by_base[base_id]
        clean = next((t for t in variants if t["variant"] == "clean"), None)
        error = next((t for t in variants if t["variant"] == "error_at_1"), None)
        if not clean or not error:
            continue

        elapsed = time.time() - t0
        eta = (elapsed / max(idx, 1)) * (len(sampled) - idx)
        print(f"  [{idx+1:3d}/{len(sampled)}] {elapsed/60:.0f}m ETA {eta/60:.0f}m", flush=True)

        correct = clean["correct_final_answer"]
        # Baseline (once)
        r_base = run_intervention_chain(error, model, tokenizer, probe, "baseline")
        records.append({
            "base_id": base_id, "condition": "baseline", "K": 0,
            "tau_mode": "n/a", "hop_depth": clean["hop_depth"],
            "error_type": error.get("injected_error_type"),
            "final_correct": check_correct(r_base["final_text"], correct),
            "max_probe": r_base["max_probe"], "mean_probe": r_base["mean_probe"],
        })

        for K in BRANCH_K_VALUES:
            for tau_mode in TAU_MODES:
                try:
                    r = run_intervention_chain(error, model, tokenizer, probe,
                                               "branch_and_pick",
                                               fire_mode=tau_mode, branch_k=K)
                    records.append({
                        "base_id": base_id, "condition": "branch_and_pick",
                        "K": K, "tau_mode": tau_mode,
                        "hop_depth": clean["hop_depth"],
                        "error_type": error.get("injected_error_type"),
                        "final_correct": check_correct(r["final_text"], correct),
                        "max_probe": r["max_probe"], "mean_probe": r["mean_probe"],
                    })
                except Exception as e:
                    print(f"    ! K={K} tau={tau_mode}: {e}")

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(out / "sensitivity.csv", index=False)

    pd.DataFrame(records).to_csv(out / "sensitivity.csv", index=False)
    _print_sensitivity(pd.DataFrame(records))
    return records


def _print_sensitivity(df):
    print("\n  SENSITIVITY RESULTS:")
    base = df[df["condition"] == "baseline"].set_index("base_id")["final_correct"]
    bp = df[df["condition"] == "branch_and_pick"]
    for K in sorted(bp["K"].unique()):
        for tau in TAU_MODES:
            sub = bp[(bp["K"] == K) & (bp["tau_mode"] == tau)]
            if len(sub) == 0:
                continue
            n = sub["final_correct"].notna().sum()
            c = sub["final_correct"].fillna(False).astype(bool).sum()
            shared = sub.set_index("base_id")["final_correct"].dropna()
            bc_shared = base.reindex(shared.index).fillna(False).astype(bool)
            cc_shared = shared.astype(bool)
            rescued = (~bc_shared & cc_shared).sum()
            broken = (bc_shared & ~cc_shared).sum()
            print(f"    K={K} tau={tau:<14}: {c:3d}/{n:3d}={c/n:.0%} "
                  f"rescued={rescued} broken={broken}")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6: PAPER FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def stage6_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import matplotlib.patches as mpatches
    from scipy.stats import gaussian_kde
    from sklearn.metrics import roc_auc_score, roc_curve

    print("\n" + "="*65)
    print("  STAGE 6: Paper figures")
    print("="*65)

    out = Path(args.output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "pdf.fonttype": 42,
    })

    INK = "#1a1a1a"; PAPER = "#fefdf9"; ACCENT = "#c14a1d"; GOLD = "#b8893a"
    EMERALD = "#2a7a4a"; CRIMSON = "#a02a2a"; AZURE = "#3a5f8a"
    AMETHYST = "#6a4a7a"; ASH = "#8a8580"

    # ── FIG 1: Layer profile ──────────────────────────────────────────
    probe_df_path = out / "layer_probe_results.csv"
    if probe_df_path.exists():
        pdf = pd.read_csv(probe_df_path)
        for mode in pdf["mode"].unique():
            sub = pdf[pdf["mode"] == mode].sort_values("layer")
            layers = sub["layer"].values
            aurocs = sub["auroc_mean"].values

            fig, ax = plt.subplots(figsize=(7, 3))
            fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
            ax.plot(layers, aurocs, color=INK, lw=1.2, zorder=2)
            ax.scatter(layers, aurocs, s=10, color=INK, zorder=3)

            best_idx = int(np.argmax(aurocs))
            ax.scatter([layers[best_idx]], [aurocs[best_idx]], s=120, color=ACCENT,
                       zorder=5, edgecolors=PAPER, linewidths=1.5, marker="D")
            ax.annotate(f"L{layers[best_idx]}: {aurocs[best_idx]:.3f}",
                         xy=(layers[best_idx], aurocs[best_idx]),
                         xytext=(layers[best_idx]+2, aurocs[best_idx]-0.03),
                         fontsize=8.5, color=ACCENT, fontweight="bold",
                         arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.5))

            if len(aurocs) > 1 and aurocs[0] < 0.6 and aurocs[1] > 0.85:
                ax.scatter([0], [aurocs[0]], s=80, color=CRIMSON, zorder=5)
                ax.annotate(f"L0: chance", xy=(0, aurocs[0]),
                             xytext=(2, aurocs[0]+0.05), fontsize=8, color=CRIMSON,
                             arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=0.5))
                ax.scatter([1], [aurocs[1]], s=80, color=GOLD, zorder=5)
                jump = aurocs[1] - aurocs[0]
                ax.annotate(f"L0->L1\n+{jump:.2f}", xy=(1, aurocs[1]),
                             xytext=(3, aurocs[1]-0.12), fontsize=8, color=GOLD,
                             arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.5))

            ax.axhline(0.5, color=ASH, ls=":", lw=0.6)
            ax.axhline(0.9, color=EMERALD, ls=":", lw=0.6, alpha=0.6)
            ax.set_xlabel("Layer index"); ax.set_ylabel("Probe AUROC (5-fold CV)")
            ax.set_ylim(0.44, 1.05)
            plt.tight_layout()
            p = fig_dir / f"fig_layer_profile_{mode}.pdf"
            fig.savefig(p, bbox_inches="tight", facecolor=PAPER)
            plt.close(fig)
            print(f"    -> {p}")

    # ── FIG 2: Detection vs failure collapse ──────────────────────────
    h3_path = out / "h3_comparison.csv"
    if h3_path.exists():
        h3 = pd.read_csv(h3_path)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)

        colors = {"Activation Probe": ACCENT, "Logprob Uncertainty": AMETHYST,
                  "Peak Entropy": AZURE, "Vocab Gap (mean)": ASH}
        x_left, x_right = 0.30, 0.75
        for _, row in h3.iterrows():
            c = colors.get(row["name"], ASH)
            det, fail = row["auroc_detection"], row["auroc_failure"]
            ax.plot([x_left, x_right], [det, fail], color=c, lw=2)
            ax.scatter([x_left], [det], s=80, color=c, zorder=4, edgecolors=PAPER, lw=1.5)
            ax.scatter([x_right], [fail], s=80, color=c, zorder=4, edgecolors=PAPER, lw=1.5, marker="s")
            ax.text(x_left - 0.02, det, f"{det:.3f}", ha="right", va="center",
                    fontsize=8, color=c, fontweight="bold")
            ax.text(x_right + 0.02, fail, f"{fail:.3f}", ha="left", va="center",
                    fontsize=8, color=c, fontweight="bold")
            ax.text(x_left - 0.12, det, row["name"], ha="right", va="center",
                    fontsize=8, color=c, fontstyle="italic")

        ax.axhline(0.5, color=ASH, ls=":", lw=0.5)
        ax.text(0.95, 0.51, "chance", ha="right", fontsize=7, color=ASH)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(0.35, 1.05)
        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(["Detection task", "Failure prediction task"])
        ax.set_ylabel("AUROC")
        plt.tight_layout()
        p = fig_dir / "fig_detection_vs_failure.pdf"
        fig.savefig(p, bbox_inches="tight", facecolor=PAPER)
        plt.close(fig)
        print(f"    -> {p}")

    # ── FIG 3: Intervention outcomes ──────────────────────────────────
    interv_path = out / "intervention_records.csv"
    if interv_path.exists():
        df = pd.read_csv(interv_path)
        pivot = df.pivot_table(index="base_id", columns="condition",
                                values="final_correct", aggfunc="first").dropna()
        bc = pivot["baseline"].astype(bool)

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 4))
        fig.patch.set_facecolor(PAPER)
        fig.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.16, wspace=0.18)

        cond_meta = [("reprompt", "Reprompt", AZURE),
                     ("replace_prior", "Replace prior", EMERALD),
                     ("branch_and_pick", "Branch + pick", AMETHYST)]

        H = 7.0
        for ax, (cond, label, color) in zip(axes, cond_meta):
            ax.set_facecolor(PAPER); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
            if cond not in pivot.columns:
                continue
            cc = pivot[cond].astype(bool)
            total = len(pivot)
            rescued = (~bc & cc).sum()
            broken = (bc & ~cc).sum()
            acc = cc.sum() / total
            delta = acc - bc.sum() / total
            sign = "+" if delta >= 0 else ""

            h_correct = H * (bc.sum() / total)
            h_wrong = H * ((~bc).sum() / total)
            h_cc = H * (cc.sum() / total)

            LX, RX, LW, RW, pad = 1.0, 6.6, 0.5, 0.5, 0.4
            y_ct = 9.0; y_cb = y_ct - h_correct
            y_wt = y_cb - pad; y_wb = y_wt - h_wrong
            y_cct = 9.0; y_ccb = y_cct - h_cc
            y_cwt = y_ccb - pad

            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((LX, y_cb), LW, h_correct, facecolor=EMERALD, alpha=0.7))
            ax.add_patch(Rectangle((LX, y_wb), LW, h_wrong, facecolor=CRIMSON, alpha=0.4))
            ax.add_patch(Rectangle((RX, y_ccb), RW, h_cc, facecolor=EMERALD, alpha=0.7))
            ax.add_patch(Rectangle((RX, y_cwt - (H - h_cc - pad)), RW, H - h_cc - pad, facecolor=CRIMSON, alpha=0.4))

            ax.text(5.0, 9.85, label, ha="center", fontsize=10.5, fontweight="bold", color=color)
            ax.text(5.0, 9.45, f"{acc*100:.1f}%  ({sign}{delta*100:.1f}pp)",
                    ha="center", fontsize=8.5, color=color, fontstyle="italic")
            ax.text(5.0, 0.7, f"rescued: {rescued}    broken: {broken}",
                    ha="center", fontsize=8.5, color=INK)
            ax.text(5.0, 0.2, f"net: {rescued-broken:+d}",
                    ha="center", fontsize=9, color=color, fontweight="bold")

        plt.tight_layout()
        p = fig_dir / "fig_intervention.pdf"
        fig.savefig(p, bbox_inches="tight", facecolor=PAPER)
        plt.close(fig)
        print(f"    -> {p}")

    # ── FIG 4: Persistence ────────────────────────────────────────────
    if interv_path.exists():
        df = pd.read_csv(interv_path)
        base = df[df["condition"] == "baseline"].copy()
        if len(base) > 10 and "mean_probe_score" in base.columns:
            y = (~base["final_correct"].astype(bool)).astype(int)
            fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
            fig.patch.set_facecolor(PAPER)

            correct = base[base["final_correct"] == True]
            wrong = base[base["final_correct"] == False]

            ax = axes[0]; ax.set_facecolor(PAPER)
            for sub, c, lbl in [(correct, EMERALD, "correct"), (wrong, CRIMSON, "wrong")]:
                vals = sub["mean_probe_score"].dropna().values
                if len(vals) >= 3:
                    x = np.linspace(max(0, vals.min()-0.1), min(1, vals.max()+0.1), 300)
                    kde = gaussian_kde(vals, bw_method=0.30)(x)
                    ax.fill_between(x, 0, kde, color=c, alpha=0.18, label=f"final {lbl} (n={len(vals)})")
                    ax.plot(x, kde, color=c, lw=1.5)

            try:
                auroc_mean = roc_auc_score(y, base["mean_probe_score"])
                ax.text(0.96, 0.96, f"AUROC = {auroc_mean:.3f}", transform=ax.transAxes,
                        va="top", ha="right", fontsize=8.5, fontweight="bold")
            except:
                pass
            ax.set_xlabel("Mean probe score"); ax.set_ylabel("Density")
            ax.set_title("(a) Mean: a calibrated signal", fontsize=9.5)
            ax.legend(frameon=False, fontsize=7.5)

            ax = axes[1]; ax.set_facecolor(PAPER)
            ax.scatter(correct["max_probe_score"], correct["mean_probe_score"],
                       c=EMERALD, s=14, alpha=0.55, label=f"correct (n={len(correct)})")
            ax.scatter(wrong["max_probe_score"], wrong["mean_probe_score"],
                       c=CRIMSON, s=14, alpha=0.55, label=f"wrong (n={len(wrong)})")
            ax.set_xlabel("Max probe score"); ax.set_ylabel("Mean probe score")
            ax.set_title("(b) Persistence beats peak", fontsize=9.5)
            ax.legend(frameon=False, fontsize=7.5, loc="lower right")

            plt.tight_layout()
            p = fig_dir / "fig_persistence.pdf"
            fig.savefig(p, bbox_inches="tight", facecolor=PAPER)
            plt.close(fig)
            print(f"    -> {p}")

    # ── FIG 5: Sensitivity heatmap ────────────────────────────────────
    sens_path = out / "sensitivity.csv"
    if sens_path.exists():
        sdf = pd.read_csv(sens_path)
        bp = sdf[sdf["condition"] == "branch_and_pick"]
        if len(bp) > 0:
            pivot_s = bp.groupby(["K", "tau_mode"])["final_correct"].mean().unstack()
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
            import matplotlib.cm as cm
            cmap = cm.get_cmap("YlOrRd")
            for i, K in enumerate(sorted(pivot_s.index)):
                for j, tau in enumerate(pivot_s.columns):
                    val = pivot_s.loc[K, tau]
                    if pd.notna(val):
                        ax.add_patch(plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8,
                                                    facecolor=cmap(val), edgecolor=ASH, lw=0.5))
                        ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                                fontsize=9, fontweight="bold",
                                color="white" if val > 0.3 else INK)
            ax.set_xticks(range(len(pivot_s.columns)))
            ax.set_xticklabels(pivot_s.columns, fontsize=8)
            ax.set_yticks(range(len(pivot_s.index)))
            ax.set_yticklabels([f"K={k}" for k in pivot_s.index], fontsize=8)
            ax.set_xlim(-0.5, len(pivot_s.columns)-0.5)
            ax.set_ylim(-0.5, len(pivot_s.index)-0.5)
            ax.set_xlabel("Threshold mode"); ax.set_ylabel("Branch count K")
            ax.set_title("Branch+pick accuracy by K x threshold", fontsize=9.5)
            plt.tight_layout()
            p = fig_dir / "fig_sensitivity.pdf"
            fig.savefig(p, bbox_inches="tight", facecolor=PAPER)
            plt.close(fig)
            print(f"    -> {p}")

    print(f"  All figures saved to {fig_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    stages = set(int(x) for x in args.stages.split(","))
    random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nKnowing-Saying Gap Pipeline")
    print(f"  Model: {args.model}")
    print(f"  Stages: {sorted(stages)}")
    print(f"  Output: {args.output_dir}")

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    by_base = defaultdict(list)
    for t in dataset["traces"]:
        by_base[t["base_problem_id"]].append(t)
    all_ids = list(by_base.keys())
    random.shuffle(all_ids)
    # Rebuild by_base with shuffled order
    by_base_ordered = defaultdict(list)
    for bid in all_ids:
        by_base_ordered[bid] = by_base[bid]
    by_base = by_base_ordered
    print(f"  Dataset: {len(dataset['traces'])} traces, {len(all_ids)} base problems")

    # Load model only if needed
    model = tokenizer = n_layers = None
    if stages & {1, 4, 5}:
        tokenizer, model, n_layers, d_model = load_model(args.model, args.no_4bit)

    # Run stages
    if 1 in stages:
        stage1_collect(args, dataset, by_base, model, tokenizer, n_layers)
    if 2 in stages:
        stage2_probes(args)
    if 3 in stages:
        stage3_behavioral(args)
    if 4 in stages:
        stage4_intervention(args, dataset, by_base, model, tokenizer, n_layers)
    if 5 in stages:
        stage5_sensitivity(args, dataset, by_base, model, tokenizer, n_layers)
    if 6 in stages:
        stage6_figures(args)

    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Output directory: {args.output_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()