"""Kirby MCQ-27 discount-factor sweep across Qwen3 4B variants.

Reproduces and extends runiteking1/temporal-awareness/notebooks/discount_factor_*.py
under one harness:

  - Three models: Qwen3-4B-Instruct-2507, Qwen3-4B, Qwen3-4B-Thinking-2507.
  - Eight conditions per model: two direct (default + heroin personas) and six
    calibrated few-shot configs (default/heroin/cross/neutral x default/heroin shots).
  - Per-(model, condition) JSON cache under out/discount/cache/.
  - Five summary plots written to out/discount/.

Usage:
    uv run python scripts/discount/run_discount_factor.py
    uv run python scripts/discount/run_discount_factor.py --plot-only
    uv run python scripts/discount/run_discount_factor.py --models Qwen/Qwen3-4B-Instruct-2507
    uv run python scripts/discount/run_discount_factor.py --force
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from src.inference.backends import ModelBackend
from src.inference.model_runner import ModelRunner

try:
    from google.genai import types as google_genai_types
except ImportError:
    google_genai_types = None

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "out" / "discount" / "cache"
PLOT_DIR = REPO_ROOT / "out" / "discount"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


# -------------------- Kirby MCQ-27 --------------------

QUESTIONS_TXT = """1. Would you prefer $54 today, or $55 in 117 days?
2. Would you prefer $55 today, or $75 in 61 days?
3. Would you prefer $19 today, or $25 in 53 days?
4. Would you prefer $31 today, or $85 in 7 days?
5. Would you prefer $14 today, or $25 in 19 days?
6. Would you prefer $47 today, or $50 in 160 days?
7. Would you prefer $15 today, or $35 in 13 days?
8. Would you prefer $25 today, or $60 in 14 days?
9. Would you prefer $78 today, or $80 in 162 days?
10. Would you prefer $40 today, or $55 in 62 days?
11. Would you prefer $11 today, or $30 in 7 days?
12. Would you prefer $67 today, or $75 in 119 days?
13. Would you prefer $34 today, or $35 in 186 days?
14. Would you prefer $27 today, or $50 in 21 days?
15. Would you prefer $69 today, or $85 in 91 days?
16. Would you prefer $49 today, or $60 in 89 days?
17. Would you prefer $80 today, or $85 in 157 days?
18. Would you prefer $24 today, or $35 in 29 days?
19. Would you prefer $33 today, or $80 in 14 days?
20. Would you prefer $28 today, or $30 in 179 days?
21. Would you prefer $34 today, or $50 in 30 days?
22. Would you prefer $25 today, or $30 in 80 days?
23. Would you prefer $41 today, or $75 in 20 days?
24. Would you prefer $54 today, or $60 in 111 days?
25. Would you prefer $54 today, or $80 in 30 days?
26. Would you prefer $22 today, or $25 in 136 days?
27. Would you prefer $20 today, or $55 in 7 days?"""

_Q_RE = re.compile(
    r"(\d+)\. Would you prefer \$(\d+) today, or \$(\d+) in (\d+) days\?"
)

# API-baseline answers carried over from runiteking1/discount_factor_llm.py.
HUMAN_RESPONSES = {
    "Gemini": ["now","later","later","later","later","now","later","later","now","later",
               "later","now","now","later","later","later","now","later","later","now",
               "later","now","later","now","later","now","later"],
    "Claude": ["now","later","later","later","later","now","later","later","now","later",
               "later","now","now","later","later","later","now","later","later","now",
               "later","now","later","now","later","now","later"],
}

HUMAN_K_REF = {"controls": 0.013, "heroin_patients": 0.025}
HUMAN_K_BY_MAG = {"small": 0.012, "medium": 0.013, "large": 0.016}


def sigfigs(x, n=2):
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def parse_trials():
    rows = []
    for line in QUESTIONS_TXT.strip().splitlines():
        m = _Q_RE.match(line.strip())
        if m:
            order, sir, ldr, delay = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            rows.append(
                dict(order=order, sir=sir, ldr=ldr, delay=delay,
                     k_indiff=sigfigs((ldr / sir - 1) / delay))
            )
    return pd.DataFrame(rows)


def magnitude(ldr):
    if ldr <= 35:
        return "small"
    if ldr <= 60:
        return "medium"
    return "large"


def estimate_k(df):
    """Kirby (1999) maximum-consistency k estimator."""
    sorted_ks = sorted(df["k_indiff"].unique())
    candidates = [sorted_ks[0]]
    for i in range(len(sorted_ks) - 1):
        candidates.append(math.sqrt(sorted_ks[i] * sorted_ks[i + 1]))
    candidates.append(sorted_ks[-1])

    def count_consistent(k_val):
        n = 0
        for _, row in df.iterrows():
            if row["k_indiff"] > k_val and row["chose_delayed"]:
                n += 1
            elif row["k_indiff"] < k_val and not row["chose_delayed"]:
                n += 1
            elif abs(row["k_indiff"] - k_val) < 1e-10:
                n += 1
        return n

    scored = [(k, count_consistent(k)) for k in candidates]
    max_n = max(s[1] for s in scored)
    best = [s[0] for s in scored if s[1] == max_n]
    k = math.exp(sum(math.log(b) for b in best) / len(best))
    return dict(k=k, consistency=max_n / len(df),
                n_consistent=max_n, n_trials=len(df))


# -------------------- System prompts --------------------

SYSTEM_DEFAULT = (
    "You are a 35-year-old adult with a stable job and average finances. "
    "You are completing a psychology questionnaire about monetary preferences. "
    "For each question, give your genuine personal preference. "
    "Reply with exactly one word: now or later."
)

SYSTEM_HEROIN = (
    "You are a 36-year-old person who has been using heroin regularly for about 8 years. "
    "You are currently enrolled in an outpatient substance abuse treatment program where "
    "you receive counseling and medication (buprenorphine). You have a high school education. "
    "You are completing a psychology questionnaire about monetary preferences. "
    "For each question, give your genuine personal preference. "
    "Reply with exactly one word: now or later."
)

# -------------------- Few-shot configs (from discount_factor_best.py) --------------------

BASE_INSTRUCTION = (
    "You are completing a psychology questionnaire about monetary preferences. "
    "Both options are guaranteed real payments. "
    "People differ on these choices — there are no right or wrong answers. "
    "Reply with exactly one word: now or later."
)

# Persona x shot pattern. Default-shots flip later at high k_indiff (patient).
# Heroin-shots flip later only at very high k_indiff (impulsive).
PERSONAS = {
    "default":
        "You are a financially comfortable 45-year-old professional. ",
    "heroin":
        "You are a 36-year-old in a tough financial situation. ",
    "neutral":
        "You are an adult completing a survey. ",
}

DEFAULT_SHOTS = [
    ("Would you prefer $90 today, or $95 in 120 days?", "now"),
    ("Would you prefer $50 today, or $70 in 60 days?",  "now"),
    ("Would you prefer $24 today, or $35 in 29 days?",  "later"),
    ("Would you prefer $20 today, or $55 in 7 days?",   "later"),
]

HEROIN_SHOTS = [
    ("Would you prefer $90 today, or $95 in 120 days?", "now"),
    ("Would you prefer $50 today, or $70 in 60 days?",  "now"),
    ("Would you prefer $25 today, or $60 in 14 days?",  "later"),
    ("Would you prefer $20 today, or $55 in 7 days?",   "later"),
]

SHOT_VARIANTS = {"default": DEFAULT_SHOTS, "heroin": HEROIN_SHOTS}


def build_fewshot(persona_key, shot_key):
    msgs = [{"role": "system", "content": PERSONAS[persona_key] + BASE_INSTRUCTION}]
    for q, a in SHOT_VARIANTS[shot_key]:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    return msgs


# -------------------- Conditions --------------------

CONDITIONS = [
    {"name": "direct_default", "kind": "direct", "system": SYSTEM_DEFAULT},
    {"name": "direct_heroin",  "kind": "direct", "system": SYSTEM_HEROIN},
    {"name": "fewshot_default_x_default", "kind": "fewshot",
     "persona": "default", "shots": "default"},
    {"name": "fewshot_heroin_x_heroin",   "kind": "fewshot",
     "persona": "heroin",  "shots": "heroin"},
    {"name": "fewshot_default_x_heroin",  "kind": "fewshot",
     "persona": "default", "shots": "heroin"},
    {"name": "fewshot_heroin_x_default",  "kind": "fewshot",
     "persona": "heroin",  "shots": "default"},
    {"name": "fewshot_neutral_x_default", "kind": "fewshot",
     "persona": "neutral", "shots": "default"},
    {"name": "fewshot_neutral_x_heroin",  "kind": "fewshot",
     "persona": "neutral", "shots": "heroin"},
]


# -------------------- Models --------------------

MODELS = [
    # HuggingFace local models
    {"id": "Qwen/Qwen3-4B-Instruct-2507", "label": "4B-Instruct-2507", "kind": "hf", "thinking": False},
    {"id": "Qwen/Qwen3-4B",               "label": "4B (hybrid, no-think)", "kind": "hf", "thinking": False},
    {"id": "Qwen/Qwen3-4B::think",        "label": "4B (hybrid, think)",    "kind": "hf",
     "thinking": True, "hf_id": "Qwen/Qwen3-4B"},
    {"id": "Qwen/Qwen3-4B-Thinking-2507", "label": "4B-Thinking-2507", "kind": "hf", "thinking": True},
    # API models (representative subset from temporal_preference_paper coherence appendix)
    {"id": "claude-opus-4-7",            "label": "Claude Opus 4.7",   "kind": "anthropic"},
    {"id": "claude-sonnet-4-6",          "label": "Claude Sonnet 4.6", "kind": "anthropic"},
    {"id": "claude-haiku-4-5-20251001",  "label": "Claude Haiku 4.5",  "kind": "anthropic"},
    {"id": "gpt-5.4",                    "label": "GPT-5.4",           "kind": "openai"},
    {"id": "gemini-2.5-pro",             "label": "Gemini 2.5 Pro",    "kind": "gemini"},
]


# -------------------- Inference (delegates to src/inference) --------------------

KIND_TO_RUNNER_PREFIX = {
    "anthropic": "anthropic:",
    "openai":    "openai:",
    "gemini":    "gemini:",
    "hf":        "",
}


def make_runner(kind, model_id):
    """Instantiate a ModelRunner for any supported backend."""
    if kind == "hf":
        return ModelRunner(model_name=model_id, backend=ModelBackend.HUGGINGFACE)
    prefix = KIND_TO_RUNNER_PREFIX[kind]
    return ModelRunner(model_name=f"{prefix}{model_id}")


def hf_apply_chat_template(runner, messages, enable_thinking):
    """Render a multi-turn message list to the HF model's chat-template string,
    bypassing ModelRunner.apply_chat_template (which only handles single user)."""
    tokenizer = runner._tokenizer
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=enable_thinking,
    )


def hf_generate_chat(runner, messages, max_new_tokens, enable_thinking):
    formatted = hf_apply_chat_template(runner, messages, enable_thinking)
    return runner._backend.generate(
        formatted, max_new_tokens=max_new_tokens, temperature=0.0,
        intervention=None, past_kv_cache=None,
    )


def split_system(messages):
    sys_text = None
    rest = []
    for m in messages:
        if m["role"] == "system":
            sys_text = m["content"]
        else:
            rest.append(m)
    return sys_text, rest


def api_generate_chat(runner, messages, max_new_tokens):
    """Multi-turn API generation via the SDK client owned by ModelRunner's backend.

    The src/ backends only expose single-prompt generate(); for Kirby few-shot we
    need full chat history, so we go through `runner._backend._get_client()` (the
    same lazily-initialised SDK client the backend would use) and issue a native
    SDK call with the multi-turn message list.
    """
    backend = runner._backend
    kind = type(backend).__name__
    client = backend._get_client()
    model_id = backend._model

    if kind == "AnthropicBackend":
        sys_text, rest = split_system(messages)
        kwargs = {
            "model": model_id,
            "max_tokens": max_new_tokens,
            "messages": [{"role": m["role"], "content": m["content"]} for m in rest],
        }
        if sys_text:
            kwargs["system"] = sys_text
        if backend._supports_temperature():
            kwargs["temperature"] = 0.0
        resp = client.messages.create(**kwargs)
        return "".join(getattr(b, "text", "") for b in resp.content).strip()

    if kind == "OpenAIBackend":
        kwargs = {
            "model": model_id,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        }
        if backend._uses_completion_tokens():
            kwargs["max_completion_tokens"] = max_new_tokens
        else:
            kwargs["max_tokens"] = max_new_tokens
        if backend._supports_temperature():
            kwargs["temperature"] = 0.0
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    if kind == "GeminiBackend":
        if google_genai_types is None:
            raise RuntimeError("google-genai SDK types not importable")
        sys_text, rest = split_system(messages)
        contents = []
        for m in rest:
            role = "user" if m["role"] == "user" else "model"
            contents.append(google_genai_types.Content(
                role=role, parts=[google_genai_types.Part(text=m["content"])]
            ))
        # Gemini 2.5 Pro mandates thinking mode; the token budget must cover both
        # the silent reasoning and the visible answer, otherwise resp.text is "".
        gemini_max = max(max_new_tokens, 1024)
        config_kwargs = {
            "max_output_tokens": gemini_max,
            "temperature": 0.0,
        }
        if sys_text:
            config_kwargs["system_instruction"] = sys_text
        config = google_genai_types.GenerateContentConfig(**config_kwargs)
        resp = client.models.generate_content(
            model=model_id, contents=contents, config=config
        )
        return (resp.text or "").strip()

    raise ValueError(f"Unsupported backend type for API chat: {kind}")


def classify(reply):
    """Classify a reply as 'now' or 'delayed' (or raw if neither found)."""
    cleaned = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip().lower()
    if not cleaned:
        cleaned = reply.lower()
    cleaned = re.sub(r"\bnow or later\b", "___", cleaned)
    last_now = cleaned.rfind("now")
    last_later = cleaned.rfind("later")
    if last_now < 0 and last_later < 0:
        if "asap" in cleaned:
            return "now"
        return reply.strip().lower()[:20] or "unparsed"
    return "delayed" if last_later > last_now else "now"


def build_messages(condition, question_text):
    if condition["kind"] == "direct":
        return [
            {"role": "system", "content": condition["system"]},
            {"role": "user",   "content": question_text},
        ]
    fewshot = build_fewshot(condition["persona"], condition["shots"])
    return fewshot + [{"role": "user", "content": question_text}]


def run_condition_hf(runner, condition, trials_df, *, enable_thinking, max_new_tokens):
    responses, raw_replies = [], []
    for _, trial in trials_df.iterrows():
        q = f"Would you prefer ${int(trial.sir)} today, or ${int(trial.ldr)} in {int(trial.delay)} days?"
        msgs = build_messages(condition, q)
        reply = hf_generate_chat(runner, msgs, max_new_tokens, enable_thinking)
        ans = classify(reply)
        responses.append(ans)
        raw_replies.append(reply)
    return responses, raw_replies


def run_condition_api(runner, condition, trials_df, max_new_tokens):
    responses, raw_replies = [], []
    for _, trial in trials_df.iterrows():
        q = f"Would you prefer ${int(trial.sir)} today, or ${int(trial.ldr)} in {int(trial.delay)} days?"
        msgs = build_messages(condition, q)
        try:
            reply = api_generate_chat(runner, msgs, max_new_tokens)
        except Exception as e:
            reply = f"[ERROR: {type(e).__name__}: {e}]"
        ans = classify(reply)
        responses.append(ans)
        raw_replies.append(reply)
    return responses, raw_replies


# -------------------- Boundary search (per-trial LDR binary search) --------------------

def _normalize_for_boundary(ans):
    """Map classifier output to {'now', 'later', 'unparsed'} for boundary search."""
    if ans == "delayed":
        return "later"
    if ans == "now":
        return "now"
    return "unparsed"


def boundary_search_trial(ask_fn, sir, ldr, delay, *,
                          max_steps=20, max_multiplier=20):
    """Binary-search the LDR at which the model flips its choice.

    ask_fn(ldr_value) -> (norm_choice, raw_reply) where norm_choice is one of
    'now', 'later', 'unparsed'.

    Returns dict with keys: original_choice, boundary_ldr, boundary_k, flipped,
    search_log.
    """
    orig_choice, orig_reply = ask_fn(ldr)
    log = [(ldr, orig_choice, orig_reply)]

    if orig_choice == "now":
        # Flip target: bump LDR upward until model picks 'later'.
        lo, hi = float(ldr), float(sir * max_multiplier)
        target = "later"
        hi_choice, hi_reply = ask_fn(int(hi))
        log.append((int(hi), hi_choice, hi_reply))
        if hi_choice != target:
            return dict(original_choice=orig_choice, boundary_ldr=None,
                        boundary_k=None, flipped=False, search_log=log)
    elif orig_choice == "later":
        # Flip target: drop LDR until model picks 'now'.
        lo, hi = float(sir), float(ldr)
        target = "now"
        lo_choice, lo_reply = ask_fn(int(lo))
        log.append((int(lo), lo_choice, lo_reply))
        if lo_choice != target:
            # Even at LDR=SIR the model says 'later' (degenerate); record as
            # boundary at SIR with k=0.
            return dict(original_choice=orig_choice, boundary_ldr=float(sir),
                        boundary_k=0.0, flipped=False, search_log=log)
    else:
        # Unparseable original; can't search.
        return dict(original_choice=orig_choice, boundary_ldr=None,
                    boundary_k=None, flipped=False, search_log=log)

    for _ in range(max_steps):
        mid = round((lo + hi) / 2)
        if mid == lo or mid == hi:
            break
        choice, reply = ask_fn(int(mid))
        log.append((int(mid), choice, reply))
        if orig_choice == "now":
            if choice == "now":
                lo = mid
            else:
                hi = mid
        else:  # original was 'later'
            if choice == "later":
                hi = mid
            else:
                lo = mid

    boundary_ldr = round((lo + hi) / 2)
    boundary_k = ((boundary_ldr / sir - 1) / delay) if delay > 0 and boundary_ldr > sir else 0.0
    return dict(original_choice=orig_choice, boundary_ldr=int(boundary_ldr),
                boundary_k=float(boundary_k), flipped=True, search_log=log)


def _make_ask_hf(runner, condition, enable_thinking, max_new_tokens):
    def ask(ldr_value, sir, delay):
        q = f"Would you prefer ${int(sir)} today, or ${int(ldr_value)} in {int(delay)} days?"
        msgs = build_messages(condition, q)
        reply = hf_generate_chat(runner, msgs, max_new_tokens, enable_thinking)
        return _normalize_for_boundary(classify(reply)), reply
    return ask


def _make_ask_api(runner, condition, max_new_tokens):
    def ask(ldr_value, sir, delay):
        q = f"Would you prefer ${int(sir)} today, or ${int(ldr_value)} in {int(delay)} days?"
        msgs = build_messages(condition, q)
        try:
            reply = api_generate_chat(runner, msgs, max_new_tokens)
        except Exception as e:
            reply = f"[ERROR: {type(e).__name__}: {e}]"
        return _normalize_for_boundary(classify(reply)), reply
    return ask


def run_condition_boundary(ask_fn_factory, condition, trials_df, *,
                           max_search_steps=20, max_multiplier=20):
    """Run boundary search across all 27 trials. ask_fn_factory(ldr, sir, delay)."""
    results = []
    for _, trial in trials_df.iterrows():
        sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
        ask_local = lambda L, _sir=sir, _delay=delay: ask_fn_factory(L, _sir, _delay)
        out = boundary_search_trial(
            ask_local, sir, ldr, delay,
            max_steps=max_search_steps, max_multiplier=max_multiplier,
        )
        out.update({
            "question": int(trial.order), "sir": sir, "ldr_original": ldr,
            "delay": delay, "k_indiff": float(trial.k_indiff),
            "magnitude": magnitude(ldr),
        })
        results.append(out)
    return results


def boundary_summary(boundary_results):
    """Collapse per-trial boundary results into mean/median/max k + flip count."""
    flipped_ks = [r["boundary_k"] for r in boundary_results
                  if r.get("flipped") and r.get("boundary_k") is not None]
    n_flipped = len(flipped_ks)
    n_total = len(boundary_results)
    summary = {
        "n_flipped": n_flipped, "n_total": n_total,
        "boundaries_str": f"{n_flipped}/{n_total}",
    }
    if flipped_ks:
        ks_sorted = sorted(flipped_ks)
        summary["mean_k"] = sum(flipped_ks) / len(flipped_ks)
        summary["median_k"] = ks_sorted[len(ks_sorted) // 2]
        summary["max_k"] = max(flipped_ks)
    else:
        summary["mean_k"] = float("nan")
        summary["median_k"] = float("nan")
        summary["max_k"] = float("nan")
    return summary


# -------------------- Cache --------------------

def safe(s):
    return s.replace("/", "_")


def cache_path(model_id, condition_name):
    return CACHE_DIR / safe(model_id) / f"{condition_name}.json"


def save_cache(model_id, condition_name, payload):
    p = cache_path(model_id, condition_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))


def load_cache(model_id, condition_name):
    p = cache_path(model_id, condition_name)
    if not p.exists():
        return None
    return json.loads(p.read_text())


# -------------------- Aggregation --------------------

def build_response_df(trials_df, responses):
    df = trials_df.copy()
    df["response"] = responses
    df["chose_delayed"] = df["response"].apply(lambda r: r in ("later", "delayed"))
    df["magnitude"] = df["ldr"].apply(magnitude)
    return df


def summarize_results(trials_df, model_results):
    """model_results: list of dicts with model_id/label, condition, responses."""
    records = []
    for r in model_results:
        df = build_response_df(trials_df, r["responses"])
        full = estimate_k(df)
        record = {
            "model_id": r["model_id"],
            "model_label": r["model_label"],
            "condition": r["condition"],
            "responses": r["responses"],
            "k": full["k"],
            "consistency": full["consistency"],
            "n_consistent": full["n_consistent"],
            "n_trials": full["n_trials"],
        }
        for mag in ("small", "medium", "large"):
            sub = df[df["magnitude"] == mag]
            mres = estimate_k(sub) if len(sub) else {"k": float("nan"), "consistency": float("nan")}
            record[f"k_{mag}"] = mres["k"]
            record[f"consistency_{mag}"] = mres["consistency"]
        records.append(record)
    return pd.DataFrame(records)


def add_human_baselines(trials_df, summary_df):
    rows = []
    for name, answers in HUMAN_RESPONSES.items():
        df = trials_df.copy()
        df["response"] = [a.lower() for a in answers]
        df["chose_delayed"] = df["response"].apply(lambda r: r in ("later", "delayed"))
        df["magnitude"] = df["ldr"].apply(magnitude)
        full = estimate_k(df)
        record = {
            "model_id": f"api/{name}",
            "model_label": name,
            "condition": "api_default",
            "responses": [a.lower() for a in answers],
            "k": full["k"],
            "consistency": full["consistency"],
            "n_consistent": full["n_consistent"],
            "n_trials": full["n_trials"],
        }
        for mag in ("small", "medium", "large"):
            sub = df[df["magnitude"] == mag]
            mres = estimate_k(sub)
            record[f"k_{mag}"] = mres["k"]
            record[f"consistency_{mag}"] = mres["consistency"]
        rows.append(record)
    return pd.concat([summary_df, pd.DataFrame(rows)], ignore_index=True)


# -------------------- Plots --------------------

CONDITION_ORDER = [c["name"] for c in CONDITIONS]
CONDITION_LABEL = {
    "direct_default":            "direct\ndefault",
    "direct_heroin":             "direct\nheroin",
    "fewshot_default_x_default": "fs default\n× def-shots",
    "fewshot_heroin_x_heroin":   "fs heroin\n× hero-shots",
    "fewshot_default_x_heroin":  "fs default\n× hero-shots",
    "fewshot_heroin_x_default":  "fs heroin\n× def-shots",
    "fewshot_neutral_x_default": "fs neutral\n× def-shots",
    "fewshot_neutral_x_heroin":  "fs neutral\n× hero-shots",
}


def plot_k_summary(summary_df, out_path):
    fig, ax = plt.subplots(figsize=(15, 6))
    model_labels = sorted(summary_df["model_label"].unique().tolist())
    n_models = len(model_labels)
    n_cond = len(CONDITION_ORDER)
    width = 0.8 / max(n_models, 1)
    cmap = plt.get_cmap("tab20")
    x = np.arange(n_cond)

    for i, label in enumerate(model_labels):
        ks = []
        for cname in CONDITION_ORDER:
            row = summary_df[(summary_df["model_label"] == label) &
                             (summary_df["condition"] == cname)]
            ks.append(row["k"].values[0] if len(row) else np.nan)
        ax.bar(x + (i - (n_models - 1) / 2) * width, ks, width=width,
               label=label, color=cmap(i % 20), edgecolor="black", linewidth=0.4)

    ax.axhline(HUMAN_K_REF["controls"], color="green", ls="--", lw=1,
               label=f"human controls k={HUMAN_K_REF['controls']}")
    ax.axhline(HUMAN_K_REF["heroin_patients"], color="red", ls="--", lw=1,
               label=f"heroin patients k={HUMAN_K_REF['heroin_patients']}")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER], fontsize=8)
    ax.set_ylabel("Estimated discount rate k (log scale)")
    ax.set_title("Kirby MCQ-27: hyperbolic discount rate by model and condition")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0)
    fig.tight_layout(rect=[0, 0, 0.83, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_k_by_magnitude(summary_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    cmap = plt.get_cmap("tab20")
    model_labels = sorted(summary_df["model_label"].unique().tolist())
    n_models = len(model_labels)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(CONDITION_ORDER))

    handles_for_legend = None
    for ax, mag in zip(axes, ("small", "medium", "large")):
        for i, label in enumerate(model_labels):
            ks = []
            for cname in CONDITION_ORDER:
                row = summary_df[(summary_df["model_label"] == label) &
                                 (summary_df["condition"] == cname)]
                ks.append(row[f"k_{mag}"].values[0] if len(row) else np.nan)
            ax.bar(x + (i - (n_models - 1) / 2) * width, ks, width=width,
                   label=label, color=cmap(i % 20), edgecolor="black", linewidth=0.3)
        ax.axhline(HUMAN_K_BY_MAG[mag], color="green", ls="--", lw=1,
                   label=f"human ctrl k={HUMAN_K_BY_MAG[mag]}")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER],
                           fontsize=7, rotation=0)
        ax.set_title(f"{mag.capitalize()} reward (LDR)")
        ax.grid(axis="y", which="both", alpha=0.2)
        if handles_for_legend is None:
            handles_for_legend = ax.get_legend_handles_labels()
    axes[0].set_ylabel("k (log scale)")
    fig.legend(handles_for_legend[0], handles_for_legend[1],
               loc="center left", bbox_to_anchor=(0.86, 0.5),
               fontsize=8, borderaxespad=0.0)
    fig.suptitle("Per-magnitude discount rate by model x condition")
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_response_heatmap(trials_df, summary_df, out_path):
    sorted_trials = trials_df.sort_values("k_indiff").reset_index(drop=True)
    qcols = sorted_trials["order"].tolist()
    klabels = [f"Q{int(o)}\nk={k:.4f}"
               for o, k in zip(sorted_trials["order"], sorted_trials["k_indiff"])]

    rows = []
    row_labels = []
    for _, r in summary_df.iterrows():
        order_to_resp = dict(zip(trials_df["order"], r["responses"]))
        rows.append([1 if order_to_resp.get(o) in ("later", "delayed") else 0
                     for o in qcols])
        row_labels.append(f"{r['model_label']} | {r['condition']}")

    arr = np.array(rows)
    fig, ax = plt.subplots(figsize=(15, max(6, 0.32 * len(rows))))
    cmap = plt.cm.RdYlGn
    ax.imshow(arr, cmap=cmap, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(qcols)))
    ax.set_xticklabels(klabels, fontsize=6, rotation=90)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("Response heatmap (rows: model | condition; cols: questions sorted by k_indiff)\n"
                 "green = chose 'later' (patient), red = chose 'now' (impulsive)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap(0.0)),
        plt.Rectangle((0, 0), 1, 1, color=cmap(1.0)),
    ]
    ax.legend(handles, ["now", "later"], loc="upper left", fontsize=8,
              bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=[0, 0, 0.94, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration(trials_df, summary_df, out_path):
    """For each (model, condition): scatter k_indiff -> chose_delayed; overlay
    sigmoid implied by estimated k via Kirby max-consistency."""
    n = len(summary_df)
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    sorted_k = np.array(sorted(trials_df["k_indiff"].unique()))
    for i, (_, r) in enumerate(summary_df.iterrows()):
        ax = axes[i]
        order_to_resp = dict(zip(trials_df["order"], r["responses"]))
        ks_q = trials_df["k_indiff"].values
        chose_delayed = np.array(
            [1 if order_to_resp.get(o) in ("later", "delayed") else 0
             for o in trials_df["order"]]
        )
        ax.scatter(ks_q, chose_delayed + np.random.uniform(-0.04, 0.04, size=len(ks_q)),
                   s=18, alpha=0.7, color="steelblue")
        ax.axvline(r["k"], color="red", ls="--", lw=1, label=f"k={r['k']:.4f}")
        ax.set_xscale("log")
        ax.set_xlim(sorted_k.min() / 2, sorted_k.max() * 2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["now", "later"], fontsize=7)
        ax.set_title(f"{r['model_label']}\n{r['condition']}", fontsize=8)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, loc="center left")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.supxlabel("k_indiff (log)", fontsize=10)
    fig.suptitle("Per-question choices vs. trial k_indiff (red = estimated k boundary)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_consistency(summary_df, out_path):
    fig, ax = plt.subplots(figsize=(15, 5))
    cmap = plt.get_cmap("tab20")
    model_labels = sorted(summary_df["model_label"].unique().tolist())
    n_models = len(model_labels)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(CONDITION_ORDER))
    for i, label in enumerate(model_labels):
        cs = []
        for cname in CONDITION_ORDER:
            row = summary_df[(summary_df["model_label"] == label) &
                             (summary_df["condition"] == cname)]
            cs.append(row["consistency"].values[0] if len(row) else np.nan)
        ax.bar(x + (i - (n_models - 1) / 2) * width, cs, width=width,
               label=label, color=cmap(i % 20), edgecolor="black", linewidth=0.3)
    ax.axhline(0.96, color="green", ls="--", lw=1, label="human ctrl 96%")
    ax.axhline(0.94, color="red", ls="--", lw=1, label="heroin pts 94%")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER], fontsize=8)
    ax.set_ylabel("Maximum-consistency fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Choice consistency (Kirby maximum-consistency) by model x condition")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 0.83, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_csv(summary_df, out_path):
    keep = ["model_label", "model_id", "condition", "k", "consistency",
            "n_consistent", "n_trials", "k_small", "k_medium", "k_large",
            "consistency_small", "consistency_medium", "consistency_large"]
    summary_df[keep].to_csv(out_path, index=False)


# -------------------- Driver --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None,
                   help="Override default model list (HuggingFace ids).")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="Override condition list (names from CONDITIONS).")
    p.add_argument("--force", action="store_true", help="Ignore cache.")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip inference; render plots from cache.")
    p.add_argument("--boundary", action="store_true",
                   help="Run per-trial LDR boundary search; cache as <cond>_boundary.json")
    p.add_argument("--boundary-steps", type=int, default=20)
    p.add_argument("--boundary-max-mult", type=int, default=20)
    return p.parse_args()


def select_models(args):
    if args.models is None:
        return MODELS
    by_id = {m["id"]: m for m in MODELS}
    chosen = []
    for mid in args.models:
        if mid in by_id:
            chosen.append(by_id[mid])
        else:
            chosen.append({"id": mid, "label": mid.split("/")[-1], "thinking": False})
    return chosen


def select_conditions(args):
    if args.conditions is None:
        return CONDITIONS
    by_name = {c["name"]: c for c in CONDITIONS}
    return [by_name[n] for n in args.conditions]


def run_all(models, conditions, force, plot_only, *,
            boundary=False, boundary_steps=20, boundary_max_mult=20):
    trials_df = parse_trials()
    all_results = []
    boundary_records = []  # only used when boundary=True
    suffix = "_boundary" if boundary else ""

    for m in models:
        kind = m.get("kind", "hf")
        cached_all = True
        per_model_payloads = {}
        for cond in conditions:
            cname = cond["name"] + suffix
            cached = load_cache(m["id"], cname)
            if cached and not force:
                per_model_payloads[cname] = cached
            else:
                cached_all = False
                per_model_payloads[cname] = None

        if plot_only:
            for cname, payload in per_model_payloads.items():
                if payload is None:
                    print(f"[plot-only] missing cache: {m['label']} / {cname}")
                    continue
                if boundary:
                    boundary_records.append({
                        "model_id": m["id"], "model_label": m["label"],
                        "condition": cname[:-len(suffix)] if suffix else cname,
                        "trials": payload.get("trials", []),
                    })
                else:
                    all_results.append({
                        "model_id": m["id"], "model_label": m["label"],
                        "condition": cname, "responses": payload["responses"],
                    })
            continue

        if cached_all and not force:
            print(f"\n=== {m['label']} : all conditions cached, skipping ===")
            for cname, payload in per_model_payloads.items():
                if boundary:
                    boundary_records.append({
                        "model_id": m["id"], "model_label": m["label"],
                        "condition": cname[:-len(suffix)] if suffix else cname,
                        "trials": payload.get("trials", []),
                    })
                else:
                    all_results.append({
                        "model_id": m["id"], "model_label": m["label"],
                        "condition": cname, "responses": payload["responses"],
                    })
            continue

        if kind == "hf":
            hf_id = m.get("hf_id", m["id"])
            print(f"\n=== Loading {hf_id} via ModelRunner ===")
            t0 = time.time()
            runner = make_runner("hf", hf_id)
            print(f"Loaded in {time.time() - t0:.1f}s")
            max_new = 600 if m.get("thinking") else 4
            for cond in conditions:
                cname = cond["name"] + suffix
                payload = per_model_payloads[cname]
                if payload is None or force:
                    print(f"  -> running {cname} (thinking={m.get('thinking', False)}, max_new={max_new})")
                    t0 = time.time()
                    if boundary:
                        ask_factory = _make_ask_hf(
                            runner, cond, m.get("thinking", False), max_new,
                        )
                        trials = run_condition_boundary(
                            ask_factory, cond, trials_df,
                            max_search_steps=boundary_steps,
                            max_multiplier=boundary_max_mult,
                        )
                        dt = time.time() - t0
                        bsum = boundary_summary(trials)
                        payload = {
                            "model_id": m["id"], "model_label": m["label"],
                            "condition": cond["name"], "mode": "boundary",
                            "kind": kind, "hf_id": hf_id,
                            "thinking": m.get("thinking", False),
                            "trials": trials, "summary": bsum,
                            "elapsed_sec": dt,
                        }
                        save_cache(m["id"], cname, payload)
                        print(f"     done in {dt:.1f}s, flips={bsum['boundaries_str']}, "
                              f"mean_k={bsum['mean_k']:.4f}, median_k={bsum['median_k']:.4f}, "
                              f"max_k={bsum['max_k']:.4f}")
                    else:
                        responses, raw = run_condition_hf(
                            runner, cond, trials_df,
                            enable_thinking=m.get("thinking", False),
                            max_new_tokens=max_new,
                        )
                        dt = time.time() - t0
                        df = build_response_df(trials_df, responses)
                        k_info = estimate_k(df)
                        payload = {
                            "model_id": m["id"], "model_label": m["label"],
                            "condition": cond["name"],
                            "kind": kind, "hf_id": hf_id,
                            "thinking": m.get("thinking", False),
                            "max_new_tokens": max_new,
                            "responses": responses, "raw_replies": raw,
                            "k": k_info["k"], "consistency": k_info["consistency"],
                            "n_consistent": k_info["n_consistent"], "elapsed_sec": dt,
                        }
                        save_cache(m["id"], cname, payload)
                        print(f"     done in {dt:.1f}s, k={k_info['k']:.6f}, "
                              f"consistency={k_info['consistency']:.1%}")
                if boundary:
                    boundary_records.append({
                        "model_id": m["id"], "model_label": m["label"],
                        "condition": cond["name"], "trials": payload["trials"],
                    })
                else:
                    all_results.append({
                        "model_id": m["id"], "model_label": m["label"],
                        "condition": cond["name"], "responses": payload["responses"],
                    })
            del runner

        elif kind in ("anthropic", "openai", "gemini"):
            print(f"\n=== {m['label']} via {kind} API (ModelRunner) ===")
            runner = make_runner(kind, m["id"])
            max_new = 1024 if kind == "gemini" else 16
            for cond in conditions:
                cname = cond["name"] + suffix
                payload = per_model_payloads[cname]
                if payload is None or force:
                    print(f"  -> {cname}")
                    t0 = time.time()
                    if boundary:
                        ask_factory = _make_ask_api(runner, cond, max_new)
                        trials = run_condition_boundary(
                            ask_factory, cond, trials_df,
                            max_search_steps=boundary_steps,
                            max_multiplier=boundary_max_mult,
                        )
                        dt = time.time() - t0
                        bsum = boundary_summary(trials)
                        payload = {
                            "model_id": m["id"], "model_label": m["label"],
                            "condition": cond["name"], "mode": "boundary",
                            "kind": kind,
                            "trials": trials, "summary": bsum,
                            "elapsed_sec": dt,
                        }
                        save_cache(m["id"], cname, payload)
                        print(f"     done in {dt:.1f}s, flips={bsum['boundaries_str']}, "
                              f"mean_k={bsum['mean_k']:.4f}, median_k={bsum['median_k']:.4f}, "
                              f"max_k={bsum['max_k']:.4f}")
                    else:
                        responses, raw = run_condition_api(
                            runner, cond, trials_df, max_new_tokens=max_new,
                        )
                        dt = time.time() - t0
                        df = build_response_df(trials_df, responses)
                        k_info = estimate_k(df)
                        payload = {
                            "model_id": m["id"], "model_label": m["label"],
                            "condition": cond["name"], "kind": kind,
                            "max_new_tokens": max_new,
                            "responses": responses, "raw_replies": raw,
                            "k": k_info["k"], "consistency": k_info["consistency"],
                            "n_consistent": k_info["n_consistent"], "elapsed_sec": dt,
                        }
                        save_cache(m["id"], cname, payload)
                        print(f"     done in {dt:.1f}s, k={k_info['k']:.6f}, "
                              f"consistency={k_info['consistency']:.1%}")
                if boundary:
                    boundary_records.append({
                        "model_id": m["id"], "model_label": m["label"],
                        "condition": cond["name"], "trials": payload["trials"],
                    })
                else:
                    all_results.append({
                        "model_id": m["id"], "model_label": m["label"],
                        "condition": cond["name"], "responses": payload["responses"],
                    })
            del runner

        else:
            raise ValueError(f"Unknown model kind: {kind}")

    return trials_df, all_results, boundary_records


def write_boundary_csv(boundary_records, out_path):
    rows = []
    for r in boundary_records:
        bsum = boundary_summary(r["trials"])
        rows.append({
            "model_label": r["model_label"], "model_id": r["model_id"],
            "condition": r["condition"],
            "boundaries": bsum["boundaries_str"],
            "n_flipped": bsum["n_flipped"], "n_total": bsum["n_total"],
            "mean_k": bsum["mean_k"], "median_k": bsum["median_k"],
            "max_k": bsum["max_k"],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_boundary_summary(boundary_records, out_path):
    """Compact bar chart: mean / median / max boundary k per (model, condition)."""
    rows = []
    for r in boundary_records:
        bsum = boundary_summary(r["trials"])
        rows.append({
            "model_label": r["model_label"], "condition": r["condition"],
            "n_flipped": bsum["n_flipped"], "n_total": bsum["n_total"],
            "mean_k": bsum["mean_k"], "median_k": bsum["median_k"],
            "max_k": bsum["max_k"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return
    conds = sorted(df["condition"].unique().tolist())
    models = sorted(df["model_label"].unique().tolist())
    fig, axes = plt.subplots(1, len(conds), figsize=(7 * len(conds), 5),
                             sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    width = 0.25
    x = np.arange(len(models))
    for ax, cond in zip(axes[0], conds):
        sub = df[df["condition"] == cond].set_index("model_label").reindex(models)
        for j, stat in enumerate(("mean_k", "median_k", "max_k")):
            ax.bar(x + (j - 1) * width, sub[stat].values, width=width,
                   label=stat.replace("_", " "), color=cmap(j),
                   edgecolor="black", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        # Annotate flip counts above each model group.
        for j, (lbl, row) in enumerate(sub.iterrows()):
            if pd.notna(row["mean_k"]):
                ax.text(x[j], max(row["max_k"], row["mean_k"]) * 1.4,
                        f"{int(row['n_flipped'])}/{int(row['n_total'])}",
                        ha="center", va="bottom", fontsize=8)
        ax.axhline(HUMAN_K_REF["controls"], color="green", ls="--", lw=1)
        ax.axhline(HUMAN_K_REF["heroin_patients"], color="red", ls="--", lw=1)
        ax.set_title(cond)
        ax.grid(axis="y", which="both", alpha=0.25)
    axes[0, 0].set_ylabel("Boundary k (log scale)")
    axes[0, -1].legend(fontsize=8, loc="upper left",
                       bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.suptitle("Decision-boundary k: mean / median / max per (model, condition)\n"
                 "annotation = flip-count fraction (Boundaries / 27)")
    fig.tight_layout(rect=[0, 0, 0.93, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_boundary_table(boundary_records):
    if not boundary_records:
        return
    print("\n=== Boundary search summary ===")
    print(f"{'Model':<25s}  {'Condition':<24s}  {'Boundaries':>10s}  "
          f"{'Mean k':>8s}  {'Median k':>9s}  {'Max k':>8s}")
    print("-" * 92)
    for r in sorted(boundary_records, key=lambda x: (x["model_label"], x["condition"])):
        bsum = boundary_summary(r["trials"])
        print(f"{r['model_label']:<25s}  {r['condition']:<24s}  "
              f"{bsum['boundaries_str']:>10s}  {bsum['mean_k']:>8.4f}  "
              f"{bsum['median_k']:>9.4f}  {bsum['max_k']:>8.4f}")


def main():
    args = parse_args()
    models = select_models(args)
    conditions = select_conditions(args)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    trials_df, raw_results, boundary_records = run_all(
        models, conditions, args.force, args.plot_only,
        boundary=args.boundary,
        boundary_steps=args.boundary_steps,
        boundary_max_mult=args.boundary_max_mult,
    )

    if args.boundary:
        if not boundary_records:
            print("No boundary results; nothing to plot.")
            return
        print_boundary_table(boundary_records)
        write_boundary_csv(boundary_records, PLOT_DIR / "boundary_summary.csv")
        plot_boundary_summary(boundary_records, PLOT_DIR / "boundary_summary.png")
        print(f"\nBoundary outputs written under {PLOT_DIR}")
        return

    if not raw_results:
        print("No results; nothing to plot.")
        return

    summary = summarize_results(trials_df, raw_results)
    summary = add_human_baselines(trials_df, summary)
    write_summary_csv(summary, PLOT_DIR / "summary.csv")

    print("\n=== Summary (k by model x condition) ===")
    print(
        summary[["model_label", "condition", "k", "consistency"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    plot_k_summary(summary[summary["condition"] != "api_default"],
                   PLOT_DIR / "k_summary.png")
    plot_k_by_magnitude(summary[summary["condition"] != "api_default"],
                        PLOT_DIR / "k_by_magnitude.png")
    plot_response_heatmap(trials_df, summary, PLOT_DIR / "response_heatmap.png")
    plot_calibration(trials_df, summary, PLOT_DIR / "calibration.png")
    plot_consistency(summary[summary["condition"] != "api_default"],
                     PLOT_DIR / "consistency.png")
    print(f"\nPlots written under {PLOT_DIR}")


if __name__ == "__main__":
    main()
