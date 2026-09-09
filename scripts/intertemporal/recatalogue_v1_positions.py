"""Re-catalogue the v1 geometry captures to their true token positions.

The v1 capture ran the forward pass over chat_template(user: prompt + response),
while the position mapping was built from the trajectory
chat_template(user: prompt) + response. The activations are real; the labels
belong to the other ordering. This script rebuilds, per sample, the exact token
sequence the v1 forward pass saw, verifies the reconstruction against the
mapping's own stored decodes at every prompt position, and writes the true
token behind each saved activation position.

    python scripts/intertemporal/recatalogue_v1_positions.py <run_dir> <tokenizer_name>

Output: <run_dir>/recatalogue.json with one record per (sample, saved position):
the v1 label, the true token, its neighbors, and per-sample verification flags.
Samples whose prompt reconstruction does not reproduce the mapping's decodes
are catalogued as UNVERIFIED and excluded from the summary counts.
"""

import collections
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

run_dir = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(sys.argv[2])

def template(user_text):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True)

def ids_of(text):
    return tok(text, add_special_tokens=False)["input_ids"]

samples = sorted((run_dir / "data" / "samples").glob("sample_*"))
print(f"{len(samples)} samples, tokenizer {sys.argv[2]}")

records, n_ok, n_traj_mismatch, n_len_shift = [], 0, 0, 0
true_counts = collections.Counter()
for i, s in enumerate(samples):
    try:
        pm = json.load(open(s / "position_mapping.json"))
        ps = json.load(open(s / "preference_sample.json"))
    except Exception:
        continue
    prompt = ps["prompt_text"]
    resp = ps["choice"]["response_texts"][ps["choice_idx"]]

    # gate: the reconstructed trajectory must reproduce the mapping's decodes
    traj_prompt_ids = ids_of(template(prompt))
    stored = pm["positions"]
    n_check = min(len(traj_prompt_ids), len(stored))
    mismatches = [
        j for j in range(n_check)
        if tok.decode([traj_prompt_ids[j]]) != stored[j]["decoded_token"]
        and stored[j]["traj_section"] == "prompt"
    ]
    verified = not mismatches

    cache_ids = ids_of(template(prompt + resp))
    len_shift = len(cache_ids) - pm["full_len"]

    saved = sorted(
        p for name in ("chat_suffix", "chat_suffix_tail")
        for p in pm["named_positions"].get(name, []))
    for p in saved:
        v1_label = next(
            (q["decoded_token"] for q in stored if q["abs_pos"] == p), None)
        true_tok = tok.decode([cache_ids[p]]) if p < len(cache_ids) else None
        window = tok.decode(cache_ids[max(0, p - 2): p + 3])
        records.append({
            "sample": s.name, "abs_pos": p, "v1_label": v1_label,
            "true_token": true_tok, "window": window,
            "verified": verified, "len_shift": len_shift,
        })
        if verified and true_tok is not None:
            true_counts[(v1_label, true_tok)] += 1
    n_ok += verified
    n_traj_mismatch += not verified
    n_len_shift += len_shift != 0
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(samples)}")

out = run_dir / "recatalogue.json"
json.dump({
    "tokenizer": sys.argv[2],
    "n_samples": len(samples),
    "n_verified": n_ok,
    "n_traj_mismatch": n_traj_mismatch,
    "n_len_shift": n_len_shift,
    "true_token_counts": {
        f"{a!r} -> {b!r}": c for (a, b), c in true_counts.most_common()},
    "records": records,
}, open(out, "w"), indent=1)
print(f"\nverified {n_ok}/{len(samples)} samples "
      f"({n_traj_mismatch} trajectory mismatches, {n_len_shift} length shifts)")
print("top re-catalogued labels (v1 label -> true token):")
for (a, b), c in true_counts.most_common(12):
    print(f"  {a!r:>14} -> {b!r:<14} {c}")
print("wrote", out)
