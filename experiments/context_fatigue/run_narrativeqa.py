"""
NarrativeQA Adversarial Context-Tracking

For each story:
  1. Present story summary
  2. Ask factual questions (including questions that will be affected by later mods)
  3. Introduce modifications (relationship, cause, outcome, detail changes)
  4. Quiz on modified facts
  5. Follow-up questions requiring reasoning about consequences
  6. Re-ask affected QA questions (randomized position) — tests whether model
     updates its answer after a correction
  7. Cross-reference: "summarize all corrections"
  8. Accumulate in a single conversation until context fills

Uses pre-generated, story-specific modifications from narrativeqa_modifications.json
instead of generic regex swaps.

Key metrics:
  - modification_quiz_correct: did model answer the new quiz correctly?
  - reask_updated: did model change its answer to match the modification?
  - reask_reverted: did model stick with the original (wrong) answer?

Outputs a single merged dataset (CSV + Parquet).
"""

import json, re, random, gc, argparse
from pathlib import Path

import torch, numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=500)
    p.add_argument("--max-stories", type=int, default=None,
                   help="Limit number of stories (for testing)")
    p.add_argument("--data",
                   default="../runpod_data/RQ3/experiment/data/narrativeqa_subset.json")
    p.add_argument("--modifications",
                   default="../runpod_data/RQ3/experiment/data/narrativeqa_modifications.json")
    p.add_argument("--out-dir", default="results_narrativeqa")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


SYSTEM_PROMPT = (
    "You are a reading comprehension assistant. Answer concisely based on the conversation. "
    "When given a CORRECTION, reply only with \"Noted.\" — do not repeat it."
)

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "probably", "might", "could be",
    "i think", "i believe", "it seems", "not sure", "not certain",
    "likely", "unlikely", "somewhat", "arguably", "i guess",
    "it appears", "in my opinion", "hard to say", "difficult to determine",
    "leaning", "tends to", "on the other hand", "however",
    "mixed", "nuanced", "complex", "ambiguous", "borderline",
]

REFUSAL_PATTERNS = [
    r"i('m| am) (not able|unable|cannot)",
    r"i can('t|not)",
    r"as an ai",
    r"i('m| am) sorry",
    r"i apologize",
    r"i don('t| do not) (have|provide|make)",
    r"not (appropriate|possible) for me",
]


def detect_hedging(t):
    return [w for w in HEDGE_WORDS if w in t.lower()]


def detect_refusal(t):
    return [p for p in REFUSAL_PATTERNS if re.search(p, t.lower())]


def check_answer(response, expected_answers):
    """Check if response contains any of the expected answers."""
    if not expected_answers:
        return None
    resp_lower = response.lower()
    return any(str(ans).lower() in resp_lower for ans in expected_answers)


def tokenize_simple(text):
    """Lowercase word tokenization for similarity computation."""
    return set(re.findall(r'[a-z]+', text.lower()))


def jaccard_similarity(set_a, set_b):
    """Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def build_turn_sequence(story, mods_entry, rng):
    """Build the full turn sequence for a story with pre-generated modifications.

    Turn flow:
      1. Present story summary
      2. Ask 2-3 factual questions (including affected_qa questions BEFORE mods)
      3. For each modification (randomized order):
         a. Issue the correction
         b. Ask the quiz question
         c. Ask the followup question
      4. Re-ask affected QA questions (scattered among followups/after mods)
      5. Cross-reference: summarize all corrections
    """
    summary = story["summary"]
    qa_pairs = story["qa_pairs"]
    modifications = mods_entry["modifications"]

    turns = []

    # ── Phase 1: Present story ──────────────────────────────────────────
    trunc_summary = summary[:800] if len(summary) > 800 else summary
    turns.append({
        "content": f"Here is a story summary:\n\n{trunc_summary}",
        "turn_type": "story_presentation",
        "expected_answers": None,
        "tracks_modification": False,
        "modification_type": None,
        "is_reask": False,
        "reask_expects_updated": False,
    })

    # ── Phase 2: Pre-modification factual questions ─────────────────────
    # First, ask the affected_qa questions BEFORE any modifications
    # so we have a baseline answer to compare against
    affected_questions_asked = []
    for mod in modifications:
        aq = mod.get("affected_qa", {})
        q = aq.get("question", "")
        if q and q not in affected_questions_asked:
            affected_questions_asked.append(q)
            orig_ans = aq.get("original_answer", "")
            turns.append({
                "content": q,
                "turn_type": "factual_question_pre",
                "expected_answers": [orig_ans] if orig_ans else None,
                "tracks_modification": False,
                "modification_type": None,
                "is_reask": False,
                "reask_expects_updated": False,
            })

    # Also ask 1-2 unrelated factual questions for baseline
    unrelated_qs = [
        qa for qa in qa_pairs
        if qa["question"] not in affected_questions_asked
    ]
    n_extra = min(2, len(unrelated_qs))
    if n_extra > 0:
        chosen = rng.sample(unrelated_qs, n_extra)
        for qa in chosen:
            turns.append({
                "content": qa["question"],
                "turn_type": "factual_question",
                "expected_answers": qa["answers"],
                "tracks_modification": False,
                "modification_type": None,
                "is_reask": False,
                "reask_expects_updated": False,
            })

    # ── Phase 3: Modifications + quizzes ────────────────────────────────
    # Shuffle modification order so position isn't predictable
    mod_indices = list(range(len(modifications)))
    rng.shuffle(mod_indices)

    # Build modification turns and collect re-ask turns separately
    mod_turns = []
    reask_turns = []

    for mi in mod_indices:
        mod = modifications[mi]

        # Correction
        mod_turns.append({
            "content": mod["instruction"],
            "turn_type": "modification",
            "expected_answers": None,
            "tracks_modification": False,
            "modification_type": mod["type"],
            "is_reask": False,
            "reask_expects_updated": False,
        })

        # Quiz on the modification
        mod_turns.append({
            "content": mod["quiz"],
            "turn_type": "modification_quiz",
            "expected_answers": [mod["expected_answer"]],
            "tracks_modification": True,
            "modification_type": mod["type"],
            "is_reask": False,
            "reask_expects_updated": False,
        })

        # Followup requiring reasoning
        mod_turns.append({
            "content": mod["followup"],
            "turn_type": "followup",
            "expected_answers": [mod.get("followup_expected", "")] if mod.get("followup_expected") else None,
            "tracks_modification": True,
            "modification_type": mod["type"],
            "is_reask": False,
            "reask_expects_updated": False,
        })

        # Prepare re-ask of affected QA (to be inserted later)
        aq = mod.get("affected_qa", {})
        if aq.get("question") and aq.get("new_answer"):
            reask_turns.append({
                "content": aq["question"],
                "turn_type": "reask_post_modification",
                "expected_answers": [aq["new_answer"]],
                "tracks_modification": True,
                "modification_type": mod["type"],
                "is_reask": True,
                "reask_expects_updated": True,
                "original_answer": aq.get("original_answer", ""),
                "new_answer": aq["new_answer"],
            })

    # ── Phase 4: Interleave re-asks among modification turns ────────────
    # Don't re-ask immediately after the corresponding modification.
    # Instead, scatter them: insert each re-ask 2-4 turns after its mod.
    combined = list(mod_turns)
    rng.shuffle(reask_turns)

    # Insert re-asks at spread-out positions in the second half of mod_turns
    if reask_turns and combined:
        # Place re-asks in the latter portion of the sequence
        min_pos = max(len(combined) // 2, 3)
        positions = sorted(
            rng.sample(
                range(min_pos, len(combined) + len(reask_turns)),
                min(len(reask_turns), len(combined)),
            )
        )
        for offset, rt in enumerate(reask_turns):
            if offset < len(positions):
                insert_at = min(positions[offset] + offset, len(combined))
                combined.insert(insert_at, rt)
            else:
                combined.append(rt)

    turns.extend(combined)

    # ── Phase 5: Cross-reference ────────────────────────────────────────
    mod_summaries = [mod["modified_fact"] for mod in modifications]
    turns.append({
        "content": "Summarize all the corrections I've made to this story so far.",
        "turn_type": "cross_reference",
        "expected_answers": mod_summaries,
        "tracks_modification": True,
        "modification_type": "all",
        "is_reask": False,
        "reask_expects_updated": False,
    })

    return turns


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device,
        attn_implementation="sdpa",
    )
    model.eval()
    model = torch.compile(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Ready.\n")

    def count_tokens(conv):
        return len(tokenizer.encode(
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True)))

    def generate(conv):
        input_text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(
            input_text, return_tensors="pt").input_ids.to(args.device)
        ctx_len = input_ids.shape[1]
        eff_max = min(args.max_new, args.max_ctx - ctx_len - 1)
        if eff_max < 10:
            return None, ctx_len, [], []

        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=eff_max, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.eos_token_id)

        new_tokens = out.sequences[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        entropies, logprobs = [], []
        for si, score in enumerate(out.scores):
            if si >= len(new_tokens):
                break
            logp = torch.log_softmax(score.float(), dim=-1)
            probs = torch.softmax(score.float(), dim=-1)
            entropies.append(-(probs * logp).sum(dim=-1).item())
            logprobs.append(logp[0, new_tokens[si].item()].item())

        return response, ctx_len, entropies, logprobs

    # Load stories and modifications
    with open(args.data) as f:
        stories = json.load(f)
    with open(args.modifications) as f:
        all_mods = json.load(f)

    # Index modifications by story_id
    mods_by_id = {m["story_id"]: m for m in all_mods}

    # Filter to stories that have modifications
    stories_with_mods = [
        s for s in stories
        if s["id"] in mods_by_id
        and len(s["summary"]) > 300
        and len(s["qa_pairs"]) >= 5
    ]

    rng = random.Random(args.seed)
    rng.shuffle(stories_with_mods)

    if args.max_stories:
        stories_with_mods = stories_with_mods[:args.max_stories]

    print(f"{'='*70}")
    print(f"NarrativeQA ADVERSARIAL — {args.model}")
    print(f"Context: {args.max_ctx:,} | Stories available: {len(stories_with_mods)}")
    print(f"Modifications: pre-generated, story-specific")
    print(f"{'='*70}\n")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    all_turns = []
    global_turn = 0
    stories_completed = 0
    prev_response_tokens = set()  # for tracking answer distinctness

    for story in stories_with_mods:
        mods_entry = mods_by_id[story["id"]]
        turn_sequence = build_turn_sequence(story, mods_entry, rng)
        story_broke = False

        for turn_info in turn_sequence:
            conversation.append({"role": "user", "content": turn_info["content"]})

            ctx_now = count_tokens(conversation)
            # Reserve ~5% of context for recall phase at the end
            effective_limit = int(args.max_ctx * 0.87)
            if ctx_now + args.max_new > effective_limit:
                print(f"\n>>> Stopping stories at turn {global_turn}, "
                      f"ctx={ctx_now} ({ctx_now/args.max_ctx:.0%}), "
                      f"reserving space for recall phase")
                conversation.pop()
                story_broke = True
                break

            response, ctx_len, entropies, logprobs = generate(conversation)
            if response is None:
                print(f"  >>> Context exhausted at turn {global_turn}")
                conversation.pop()
                story_broke = True
                break

            mean_ent = float(np.mean(entropies)) if entropies else 0
            mean_lp = float(np.mean(logprobs)) if logprobs else 0
            hedges = detect_hedging(response)
            refusals = detect_refusal(response)
            fill = ctx_len / args.max_ctx

            # Check answer correctness
            answer_correct = check_answer(
                response, turn_info.get("expected_answers"))

            # For re-ask turns, also check if they reverted to original
            reask_reverted = None
            if turn_info.get("is_reask") and turn_info.get("original_answer"):
                reask_reverted = check_answer(
                    response, [turn_info["original_answer"]])

            # Response similarity to previous response
            curr_tokens = tokenize_simple(response)
            sim_to_prev = jaccard_similarity(curr_tokens, prev_response_tokens)
            prev_response_tokens = curr_tokens

            turn_data = {
                "global_turn": global_turn,
                "story_id": story["id"],
                "story_num": stories_completed,
                "turn_type": turn_info["turn_type"],
                "tracks_modification": turn_info.get("tracks_modification", False),
                "modification_type": turn_info.get("modification_type"),
                "is_reask": turn_info.get("is_reask", False),
                "answer_correct": answer_correct,
                "reask_reverted": reask_reverted,
                "context_tokens": ctx_len,
                "context_fill": round(fill, 4),
                "mean_entropy": mean_ent,
                "max_entropy": float(np.max(entropies)) if entropies else 0,
                "min_entropy": float(np.min(entropies)) if entropies else 0,
                "median_entropy": float(np.median(entropies)) if entropies else 0,
                "std_entropy": float(np.std(entropies)) if entropies else 0,
                "mean_logprob": mean_lp,
                "max_logprob": float(np.max(logprobs)) if logprobs else 0,
                "min_logprob": float(np.min(logprobs)) if logprobs else 0,
                "median_logprob": float(np.median(logprobs)) if logprobs else 0,
                "std_logprob": float(np.std(logprobs)) if logprobs else 0,
                "perplexity": float(np.exp(-mean_lp)) if mean_lp else 0,
                "similarity_to_prev": round(sim_to_prev, 4),
                "response_length": len(response),
                "hedging_detected": len(hedges) > 0,
                "num_hedge_words": len(hedges),
                "hedge_words": hedges,
                "refusal_detected": len(refusals) > 0,
                "num_generated_tokens": len(entropies),
                "response_preview": response[:200],
                "user_input": turn_info["content"][:200],
            }
            all_turns.append(turn_data)
            global_turn += 1

            # Print progress
            flags = []
            if answer_correct is True:
                flags.append("OK")
            elif answer_correct is False:
                flags.append("WRONG")
            if turn_info.get("is_reask"):
                if reask_reverted:
                    flags.append("REVERTED")
                elif answer_correct:
                    flags.append("UPDATED")
            if hedges:
                flags.append(f"{len(hedges)}hw")
            flag_str = f" [{','.join(flags)}]" if flags else ""

            print(f"T{global_turn-1:>3} {turn_info['turn_type']:25s} "
                  f"ctx={ctx_len:,} ({fill:.0%}) | ent={mean_ent:.3f} "
                  f"lp={mean_lp:.3f} sim={sim_to_prev:.2f}"
                  f"{flag_str}")
            print(f"    \"{response[:80]}...\"")

            conversation.append({"role": "assistant", "content": response})
            torch.cuda.empty_cache()

        if story_broke:
            break

        stories_completed += 1
        conversation.append({"role": "user", "content": "--- NEW STORY ---"})
        conversation.append({"role": "assistant", "content": "Ready for the next story."})
        print(f"  [Story {stories_completed} complete]\n")

    # ── Recall phase: ask model to summarize each story ─────────────────
    print(f"\n{'='*70}")
    print(f"RECALL PHASE — asking model to summarize each of {stories_completed} stories")
    print(f"{'='*70}\n")

    completed_stories = stories_with_mods[:stories_completed]
    for recall_idx, story in enumerate(completed_stories):
        # Get a short identifier from the story summary
        summary_snippet = story["summary"][:80].strip()

        conversation.append({
            "role": "user",
            "content": f"Recall story {recall_idx + 1} that started with: \"{summary_snippet}...\" "
                       f"— summarize what happened in that story in 2-3 sentences."
        })

        ctx_now = count_tokens(conversation)
        if ctx_now + args.max_new > args.max_ctx:
            print(f"  >>> Context limit during recall at story {recall_idx + 1}")
            conversation.pop()
            break

        response, ctx_len, entropies, logprobs = generate(conversation)
        if response is None:
            conversation.pop()
            break

        mean_ent = float(np.mean(entropies)) if entropies else 0
        mean_lp = float(np.mean(logprobs)) if logprobs else 0
        hedges = detect_hedging(response)
        fill = ctx_len / args.max_ctx

        curr_tokens = tokenize_simple(response)
        # Check if response contains key details from the original summary
        summary_words = tokenize_simple(story["summary"])
        recall_overlap = jaccard_similarity(curr_tokens, summary_words)
        sim_to_prev = jaccard_similarity(curr_tokens, prev_response_tokens)
        prev_response_tokens = curr_tokens

        turn_data = {
            "global_turn": global_turn,
            "story_id": story["id"],
            "story_num": recall_idx,
            "turn_type": "recall_summary",
            "tracks_modification": False,
            "modification_type": None,
            "is_reask": False,
            "answer_correct": None,
            "reask_reverted": None,
            "context_tokens": ctx_len,
            "context_fill": round(fill, 4),
            "mean_entropy": mean_ent,
            "max_entropy": float(np.max(entropies)) if entropies else 0,
            "min_entropy": float(np.min(entropies)) if entropies else 0,
            "median_entropy": float(np.median(entropies)) if entropies else 0,
            "std_entropy": float(np.std(entropies)) if entropies else 0,
            "mean_logprob": mean_lp,
            "max_logprob": float(np.max(logprobs)) if logprobs else 0,
            "min_logprob": float(np.min(logprobs)) if logprobs else 0,
            "median_logprob": float(np.median(logprobs)) if logprobs else 0,
            "std_logprob": float(np.std(logprobs)) if logprobs else 0,
            "perplexity": float(np.exp(-mean_lp)) if mean_lp else 0,
            "similarity_to_prev": round(sim_to_prev, 4),
            "response_length": len(response),
            "hedging_detected": len(hedges) > 0,
            "num_hedge_words": len(hedges),
            "hedge_words": hedges,
            "refusal_detected": len(detect_refusal(response)) > 0,
            "num_generated_tokens": len(entropies),
            "response_preview": response[:200],
            "user_input": f"Recall story {recall_idx + 1}",
            "recall_overlap": round(recall_overlap, 4),
        }
        all_turns.append(turn_data)
        global_turn += 1

        print(f"RECALL S{recall_idx:>2} ({fill:.0%}) | ent={mean_ent:.3f} "
              f"overlap={recall_overlap:.2f} len={len(response)}")
        print(f"    \"{response[:100]}...\"")

        conversation.append({"role": "assistant", "content": response})
        torch.cuda.empty_cache()

    # ── Build dataset ───────────────────────────────────────────────────
    df = pd.DataFrame(all_turns)
    df["hedge_words"] = df["hedge_words"].apply(lambda x: "|".join(x))
    df.insert(0, "model", args.model)
    df.insert(1, "max_context_tokens", args.max_ctx)
    df.insert(2, "experiment", "narrativeqa_adversarial")

    df.to_csv(out_dir / "narrativeqa_adversarial.csv", index=False)
    df.to_parquet(out_dir / "narrativeqa_adversarial.parquet", index=False)

    with open(out_dir / "conversation.jsonl", "w") as f:
        for msg in conversation:
            f.write(json.dumps(msg) + "\n")

    # ── Summary statistics ──────────────────────────────────────────────
    # Modification quiz accuracy
    quiz_turns = [t for t in all_turns if t["turn_type"] == "modification_quiz"]
    quiz_correct = sum(1 for t in quiz_turns if t["answer_correct"])

    # Re-ask accuracy (did model update its answer?)
    reask_turns = [t for t in all_turns if t["is_reask"]]
    reask_updated = sum(1 for t in reask_turns if t["answer_correct"])
    reask_reverted_n = sum(1 for t in reask_turns if t.get("reask_reverted"))

    # Pre-modification factual accuracy
    pre_turns = [t for t in all_turns if t["turn_type"] == "factual_question_pre"]
    pre_correct = sum(1 for t in pre_turns if t["answer_correct"])

    summary = {
        "model": args.model,
        "max_context": args.max_ctx,
        "stories_completed": stories_completed,
        "total_turns": len(all_turns),
        "modification_quiz": {
            "total": len(quiz_turns),
            "correct": quiz_correct,
            "accuracy": round(quiz_correct / len(quiz_turns), 4) if quiz_turns else None,
        },
        "reask_tracking": {
            "total": len(reask_turns),
            "updated_correctly": reask_updated,
            "reverted_to_original": reask_reverted_n,
            "update_rate": round(reask_updated / len(reask_turns), 4) if reask_turns else None,
            "revert_rate": round(reask_reverted_n / len(reask_turns), 4) if reask_turns else None,
        },
        "pre_modification_factual": {
            "total": len(pre_turns),
            "correct": pre_correct,
            "accuracy": round(pre_correct / len(pre_turns), 4) if pre_turns else None,
        },
        "hedging_rate": (sum(1 for t in all_turns if t["hedging_detected"])
                         / len(all_turns)) if all_turns else 0,
        "refusal_rate": (sum(1 for t in all_turns if t["refusal_detected"])
                         / len(all_turns)) if all_turns else 0,
        "mean_entropy": float(np.mean([t["mean_entropy"] for t in all_turns])),
        "mean_logprob": float(np.mean([t["mean_logprob"] for t in all_turns])),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — {stories_completed} stories, {len(all_turns)} turns")
    print(f"{'='*70}")
    if quiz_turns:
        print(f"Modification quiz:  {quiz_correct}/{len(quiz_turns)} "
              f"= {quiz_correct/len(quiz_turns):.1%}")
    if reask_turns:
        print(f"Re-ask updated:     {reask_updated}/{len(reask_turns)} "
              f"= {reask_updated/len(reask_turns):.1%}")
        print(f"Re-ask reverted:    {reask_reverted_n}/{len(reask_turns)} "
              f"= {reask_reverted_n/len(reask_turns):.1%}")
    if pre_turns:
        print(f"Pre-mod factual:    {pre_correct}/{len(pre_turns)} "
              f"= {pre_correct/len(pre_turns):.1%}")
    print(f"Hedging rate:       {summary['hedging_rate']:.1%}")
    print(f"Mean entropy:       {summary['mean_entropy']:.4f}")
    print(f"\nSaved to {out_dir}/")
    print(f"  narrativeqa_adversarial.csv  ({len(df)} rows x {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
