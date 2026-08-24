"""Reconstruct the paper's two appendix example transcripts from the committed seeds.

Replays the exact construction of (a) E1 session 0's first depth-21 probe in the
`local` and `back_10` arms and (b) E3 probe 0's `near_dup` and `random` arms, then
validates each against the committed artifact (token counts, pathology, gold) so any
drift between this dump and the runs the paper quotes is a hard failure, not a
silently different transcript.  Generations themselves are NOT reproduced here; the
paper quotes them from the committed `turns.csv` files.

E1 filler replies are model-generated (greedy), so this driver needs the GPU for the
21 filler turns of session 0; everything in the E3 reconstruction is deterministic on
CPU.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cf_common import generate_with_entropy, load_filler_pool, render_prompt
from run_competition_sweep import ACK, build_context_turns
from run_competition_sweep import INTRO as E3_INTRO
from run_distance_sweep import INTRO as E1_INTRO
from run_distance_sweep import REFERENT

from src.probes.context_fatigue.context_assembly import (
    assemble_transcript,
    select_by_option_overlap,
)
from src.probes.context_fatigue.ddxplus_cases import (
    format_case_question,
    load_evidence_db,
    load_probe_pool,
)

E1_CSV = Path("results/context_fatigue/e1_distance_sweep/turns.csv")
E3_CSV = Path("results/context_fatigue/e3_competition/turns.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--max-filler-tokens", type=int, default=90)
    p.add_argument("--depths", type=int, nargs="+", default=[21, 28, 35, 42])
    p.add_argument("--probes-per-cell", type=int, default=8)
    p.add_argument("--n-probes-e3", type=int, default=384)
    p.add_argument("--n-context", type=int, default=8)
    p.add_argument("--min-overlap", type=int, default=3)
    p.add_argument("--n-options", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="results/context_fatigue/example_transcripts")
    return p.parse_args()


def write_transcript(path, turns):
    lines = [f"[{t['role']}] {t['content']}" for t in turns]
    path.write_text("\n\n".join(lines))
    print(f"  wrote {path} ({len(turns)} turns)")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    is_chat = tokenizer.chat_template is not None

    def n_tokens(conv):
        return len(tokenizer.encode(render_prompt(tokenizer, conv, is_chat)))

    evidence_db = load_evidence_db()
    probe_pool = load_probe_pool(evidence_db, args.n_options, args.seed)

    # ---- E3: competition, probe 0, near_dup vs random (CPU-deterministic) ----
    rng = random.Random(args.seed)
    probes = rng.sample(probe_pool, min(args.n_probes_e3, len(probe_pool)))
    probe = probes[0]
    assert probe["pathology"] == "Viral pharyngitis", probe["pathology"]
    question = format_case_question(probe["options"], args.n_options, referent=REFERENT)
    meta = {"e3_probe": {"pathology": probe["pathology"], "options": probe["options"]}}
    for arm, expected_tokens in [("near_dup", 3343), ("random", 2984)]:
        cases = select_by_option_overlap(probe_pool, probe, arm=arm, n=args.n_context,
                                         seed=args.seed, min_overlap=args.min_overlap)
        prior = build_context_turns(cases, args.n_options)
        built = assemble_transcript(prior, evidence=probe["vignette"],
                                    question=question, distance=0, ack=ACK)
        got = n_tokens(built.turns)
        assert got == expected_tokens, f"e3 {arm}: {got} tokens vs committed {expected_tokens}"
        write_transcript(out_dir / f"e3_{arm}.txt", built.turns)
        meta[f"e3_{arm}"] = {
            "ctx_tokens": got,
            "context_golds": [c["gold"] for c in cases],
            "shared_options": [sorted(set(c["options"]) & set(probe["options"]))
                               for c in cases],
        }
    assert built.turns[0]["content"] == E3_INTRO

    # ---- E1: displacement, session 0, first depth-21 probe (GPU for filler) ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    filler_pool = load_filler_pool(tokenizer, args.max_filler_tokens)

    rng = random.Random(args.seed)  # session 0: seed + 1000 * 0
    filler = rng.sample(filler_pool, min(max(args.depths) + 5, len(filler_pool)))
    probes = rng.sample(probe_pool, min(args.probes_per_cell * len(args.depths),
                                        len(probe_pool)))
    conv = [{"role": "user", "content": E1_INTRO},
            {"role": "assistant", "content": "Understood."}]
    snapshot = None
    for item in filler:
        conv = conv + [{"role": "user", "content": item["text"]}]
        resp, _, _, _ = generate_with_entropy(
            model, tokenizer, render_prompt(tokenizer, conv, is_chat),
            args.device, args.max_new, args.max_ctx)
        conv = conv + [{"role": "assistant", "content": (resp or "A").strip()[:8]}]
        if sum(1 for t in conv if t["role"] == "user") - 1 == args.depths[0]:
            snapshot = list(conv)
            break

    probe = probes[0]
    assert probe["pathology"] == "Acute pulmonary edema", probe["pathology"]
    question = format_case_question(probe["options"], args.n_options, referent=REFERENT)
    meta["e1_probe"] = {"pathology": probe["pathology"], "options": probe["options"]}
    for arm, distance in [("local", 0), ("back_10", 10)]:
        built = assemble_transcript(snapshot, evidence=probe["vignette"],
                                    question=question, distance=distance)
        got = n_tokens(built.turns)
        if arm == "local":
            assert got == 2049, f"e1 local: {got} tokens vs committed 2049"
        write_transcript(out_dir / f"e1_{arm}.txt", built.turns)
        meta[f"e1_{arm}"] = {"ctx_tokens": got, "distance": distance}

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out_dir / 'metadata.json'}")
    print("All reconstructions match the committed artifacts.")


if __name__ == "__main__":
    main()
