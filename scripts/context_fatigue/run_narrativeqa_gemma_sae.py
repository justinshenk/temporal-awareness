"""
NarrativeQA Adversarial + SAE Feature Tracking — Gemma 2 9B IT

Runs the NarrativeQA adversarial fact-correction experiment on Gemma,
while capturing SAE features at layer 20 for each turn.

Tracks: accuracy, entropy, hedging, modification tracking,
plus SAE feature activations to see which features change when
the model encounters contradictory information.
"""

import json, re, random, gc, argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--max-ctx", type=int, default=8192)
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--max-stories", type=int, default=None)
    p.add_argument("--modifications",
                   default="/root/temporal-awareness/data/adversarial/narrativeqa/narrativeqa_modifications.json")
    p.add_argument("--out-dir", default="results/narrativeqa_gemma_sae")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "probably", "might", "could be",
    "i think", "i believe", "it seems", "not sure", "not certain",
    "likely", "unlikely", "somewhat", "arguably", "i guess",
    "it appears", "in my opinion", "hard to say",
]


def detect_hedging(t):
    return [w for w in HEDGE_WORDS if w in t.lower()]


def check_answer(response, expected_answers):
    if not expected_answers:
        return None
    return any(str(ans).lower() in response.lower() for ans in expected_answers)


def build_stories_from_hf(mod_ids):
    """Load NarrativeQA from HF and group by story."""
    ds = load_dataset("deepmind/narrativeqa", split="test")
    stories = {}
    for row in ds:
        sid = row["document"]["id"]
        if sid not in mod_ids:
            continue
        if sid not in stories:
            stories[sid] = {
                "id": sid,
                "summary": row["document"]["summary"]["text"],
                "qa_pairs": [],
            }
        stories[sid]["qa_pairs"].append({
            "question": row["question"]["text"],
            "answers": [a["text"] for a in row["answers"]],
        })
    return list(stories.values())


def build_turn_sequence(story, mods_entry, rng):
    summary = story["summary"]
    modifications = mods_entry["modifications"]
    turns = []

    # Phase 1: Present story
    trunc = summary[:800] if len(summary) > 800 else summary
    turns.append({"content": f"Here is a story summary:\n\n{trunc}",
                  "turn_type": "story_presentation", "expected_answers": None,
                  "tracks_modification": False, "modification_type": None,
                  "is_reask": False})

    # Phase 2: Pre-modification factual questions
    asked = []
    for mod in modifications:
        aq = mod.get("affected_qa", {})
        q = aq.get("question", "")
        if q and q not in asked:
            asked.append(q)
            turns.append({"content": q, "turn_type": "factual_question_pre",
                          "expected_answers": [aq.get("original_answer", "")] if aq.get("original_answer") else None,
                          "tracks_modification": False, "modification_type": None,
                          "is_reask": False})

    # Phase 3: Modifications + quizzes
    mod_indices = list(range(len(modifications)))
    rng.shuffle(mod_indices)
    mod_turns = []
    reask_turns = []

    for mi in mod_indices:
        mod = modifications[mi]
        mod_turns.append({"content": mod["instruction"], "turn_type": "modification",
                          "expected_answers": None, "tracks_modification": False,
                          "modification_type": mod["type"], "is_reask": False})
        mod_turns.append({"content": mod["quiz"], "turn_type": "modification_quiz",
                          "expected_answers": [mod["expected_answer"]],
                          "tracks_modification": True, "modification_type": mod["type"],
                          "is_reask": False})
        mod_turns.append({"content": mod["followup"], "turn_type": "followup",
                          "expected_answers": [mod.get("followup_expected", "")] if mod.get("followup_expected") else None,
                          "tracks_modification": True, "modification_type": mod["type"],
                          "is_reask": False})

        aq = mod.get("affected_qa", {})
        if aq.get("question") and aq.get("new_answer"):
            reask_turns.append({"content": aq["question"], "turn_type": "reask_post_modification",
                                "expected_answers": [aq["new_answer"]],
                                "tracks_modification": True, "modification_type": mod["type"],
                                "is_reask": True, "original_answer": aq.get("original_answer", ""),
                                "new_answer": aq["new_answer"]})

    combined = list(mod_turns)
    rng.shuffle(reask_turns)
    if reask_turns and combined:
        min_pos = max(len(combined) // 2, 3)
        positions = sorted(rng.sample(
            range(min_pos, len(combined) + len(reask_turns)),
            min(len(reask_turns), len(combined))))
        for offset, rt in enumerate(reask_turns):
            if offset < len(positions):
                combined.insert(min(positions[offset] + offset, len(combined)), rt)
            else:
                combined.append(rt)

    turns.extend(combined)
    return turns


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load modifications
    with open(args.modifications) as f:
        all_mods = json.load(f)
    mods_by_id = {m["story_id"]: m for m in all_mods}

    # Load stories from HF
    print("Loading NarrativeQA stories from HuggingFace...")
    stories = build_stories_from_hf(set(mods_by_id.keys()))
    stories = [s for s in stories if len(s["summary"]) > 300 and len(s["qa_pairs"]) >= 3]

    rng = random.Random(args.seed)
    rng.shuffle(stories)
    if args.max_stories:
        stories = stories[:args.max_stories]
    print(f"Stories with modifications: {len(stories)}")

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    # Load SAE
    print("Loading IT SAE (layer 20)...")
    sae = SAE.from_pretrained(release="gemma-scope-9b-it-res",
                               sae_id="layer_20/width_131k/average_l0_81")
    sae = sae.to(args.device).eval()
    print(f"  SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Hook layer 20
    captured = {}
    def hook_fn(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["act"] = hs[0, -1, :].detach().float()
    hook = model.model.layers[20].register_forward_hook(hook_fn)

    def count_tokens(conv):
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(text))

    def generate(conv):
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(args.device)
        input_ids = inputs["input_ids"]
        ctx_len = input_ids.shape[1]
        eff_max = min(args.max_new, args.max_ctx - ctx_len - 1)
        if eff_max < 10:
            return None, ctx_len, [], []

        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=eff_max, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        new_tokens = out.sequences[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        entropies, logprobs = [], []
        for si, score in enumerate(out.scores):
            if si >= len(new_tokens): break
            logp = torch.log_softmax(score.float(), dim=-1)
            probs = torch.softmax(score.float(), dim=-1)
            entropies.append(-(probs * logp).sum(dim=-1).item())
            logprobs.append(logp[0, new_tokens[si].item()].item())

        return response, ctx_len, entropies, logprobs

    # ── Run ─────────────────────────────────────────────────────────
    # Gemma has no system role — use first user message
    conversation = [{"role": "user", "content": "You are a reading comprehension assistant. Answer concisely. When given a CORRECTION, reply only with 'Noted.'"},
                    {"role": "model", "content": "Understood. I'll answer concisely and acknowledge corrections with 'Noted.'"}]
    all_turns = []
    all_sae_feats = []
    global_turn = 0

    for story_idx, story in enumerate(stories):
        ctx_now = count_tokens(conversation)
        if ctx_now / args.max_ctx > 0.88:
            print(f"\n>>> Context {ctx_now/args.max_ctx:.0%} full — stopping.")
            break

        sid = story["id"]
        mods_entry = mods_by_id[sid]
        turns = build_turn_sequence(story, mods_entry, rng)

        print(f"\n{'─'*50}")
        print(f"STORY {story_idx+1} ({len(turns)} turns) | ctx={ctx_now/args.max_ctx:.0%}")

        for turn in turns:
            ctx_now = count_tokens(conversation)
            if ctx_now / args.max_ctx > 0.88:
                break

            conversation.append({"role": "user", "content": turn["content"]})
            response, ctx_len, entropies, logprobs = generate(conversation)
            if response is None:
                break

            # Get SAE features
            act = captured["act"].to(args.device)
            with torch.no_grad():
                sae_feat = sae.encode(act.unsqueeze(0))[0].cpu()
            all_sae_feats.append(sae_feat)

            mean_ent = float(np.mean(entropies)) if entropies else 0
            mean_lp = float(np.mean(logprobs)) if logprobs else 0
            hedges = detect_hedging(response)

            # Score
            correct = check_answer(response, turn["expected_answers"])

            # Reask tracking
            reask_updated = False
            reask_reverted = False
            if turn["is_reask"]:
                orig = turn.get("original_answer", "").lower()
                new = turn.get("new_answer", "").lower()
                resp_l = response.lower()
                if new and new in resp_l:
                    reask_updated = True
                elif orig and orig in resp_l:
                    reask_reverted = True

            turn_data = {
                "global_turn": global_turn,
                "story_idx": story_idx,
                "turn_type": turn["turn_type"],
                "context_tokens": ctx_len,
                "context_fill": round(ctx_len / args.max_ctx, 4),
                "mean_entropy": mean_ent,
                "mean_logprob": mean_lp,
                "hedging_detected": len(hedges) > 0,
                "num_hedge_words": len(hedges),
                "correct": correct,
                "tracks_modification": turn["tracks_modification"],
                "modification_type": turn.get("modification_type"),
                "is_reask": turn["is_reask"],
                "reask_updated": reask_updated,
                "reask_reverted": reask_reverted,
                "response_length": len(response),
                "num_sae_active": int((sae_feat > 0).sum().item()),
                "response": response[:300],
            }
            all_turns.append(turn_data)
            global_turn += 1

            conversation.append({"role": "model", "content": response})

            marker = ""
            if correct is True: marker = "✓"
            elif correct is False: marker = "✗"
            if turn["is_reask"]:
                marker += " UPD" if reask_updated else (" REV" if reask_reverted else " ???")

            print(f"  T{global_turn:3d} {turn['turn_type']:25s} ent={mean_ent:.3f} "
                  f"sae_active={int((sae_feat>0).sum()):3d} {marker}")

            torch.cuda.empty_cache()

    hook.remove()

    # ── Save & Analyze ──────────────────────────────────────────────
    df = pd.DataFrame(all_turns)
    df.to_csv(out_dir / "turns.csv", index=False)

    sae_tensor = torch.stack(all_sae_feats)
    torch.save({"sae_features": sae_tensor, "turns": all_turns}, out_dir / "sae_data.pt")

    n = len(df)
    print(f"\n{'='*70}")
    print(f"RESULTS — {n} turns across {df['story_idx'].nunique()} stories")
    print(f"{'='*70}")

    # Entropy by turn type
    print(f"\nEntropy by turn type:")
    for tt in df["turn_type"].unique():
        sub = df[df["turn_type"] == tt]
        print(f"  {tt:30s}: ent={sub['mean_entropy'].mean():.4f} n={len(sub)}")

    # Modification tracking
    quiz = df[df["turn_type"] == "modification_quiz"]
    if len(quiz) > 0:
        print(f"\nModification quiz accuracy: {quiz['correct'].mean()*100:.1f}% ({quiz['correct'].sum()}/{len(quiz)})")

    reask = df[df["is_reask"]]
    if len(reask) > 0:
        print(f"Re-ask update rate: {reask['reask_updated'].mean()*100:.1f}%")
        print(f"Re-ask revert rate: {reask['reask_reverted'].mean()*100:.1f}%")

    # SAE feature count by context fill
    print(f"\nSAE active features by context fill:")
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        sub = df[(df["context_fill"] >= lo) & (df["context_fill"] < hi)]
        if sub.empty: continue
        print(f"  {lo:.0%}-{hi:.0%}: mean_active={sub['num_sae_active'].mean():.0f} "
              f"ent={sub['mean_entropy'].mean():.4f} n={len(sub)}")

    # SAE drift: compare features for story_presentation vs late turns
    early_feats = sae_tensor[:min(3, len(sae_tensor))]
    late_feats = sae_tensor[max(0, len(sae_tensor)-3):]
    if len(early_feats) > 0 and len(late_feats) > 0:
        early_active = (early_feats > 0).float().mean(0)
        late_active = (late_feats > 0).float().mean(0)
        diff = (late_active - early_active).abs()
        topk = diff.topk(10)
        print(f"\nTop 10 SAE features that change most (early vs late):")
        for val, idx in zip(topk.values, topk.indices):
            e = early_active[idx].item()
            l = late_active[idx].item()
            d = "↑" if l > e else "↓"
            print(f"  F{idx.item():6d}: early={e:.2f} late={l:.2f} {d}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
