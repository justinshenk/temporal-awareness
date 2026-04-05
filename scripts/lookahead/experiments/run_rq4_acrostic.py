#!/usr/bin/env python3
"""
RQ4 ACROSTIC EXPERIMENT
========================
Tests planning in a domain with NO function names or parameter names.
No surface feature confounds — pure structural planning or nothing.

Design:
1. Prompt: "Write a sentence where the first letter of each word spells [WORD]:"
2. Behavioral: does the model actually produce correct acrostics?
3. Probe at different token positions: can we predict the target word?
4. Key test: probe AFTER the target word in the prompt, during generation
   - If model plans, it should maintain the word representation
   - If not, it processes one letter at a time

5 target words (balanced classes): BRAVE, SMART, LIGHT, DREAM, PEACE
Each generates different letter sequences — no overlap in first letters (B,S,L,D,P)

Also tests with PARTIAL prompts (just first N letters given) to see
if the model commits to a full word from partial info.
"""

import json, os, sys, hashlib, random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

sys.path.insert(0, os.getcwd())

from transformer_lens import HookedTransformer
from src.lookahead.utils.types import PlanningExample, TaskType
from src.lookahead.probing.activation_extraction import extract_activations_batch

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

# ================================================================
# ACROSTIC DATASETS
# ================================================================

# 5 target words, 20 prompt variants each = 100 examples
ACROSTIC_WORDS = ["BRAVE", "SMART", "LIGHT", "DREAM", "PEACE"]

def make_acrostic_prompts():
    """Generate varied acrostic prompts for each target word."""
    templates = [
        "Write a sentence where the first letter of each word spells {word}:\n",
        "Create an acrostic for the word {word}:\n",
        "Make a sentence with words starting with {letters}:\n",
        "Write {n} words where the first letters spell {word}:\n",
        "Form an acrostic using the word {word}:\n",
        "Compose a phrase where initial letters spell {word}:\n",
        "Write words beginning with the letters {letters}:\n",
        "Create a sentence spelling out {word} with first letters:\n",
        "Generate an acrostic poem for {word}:\n",
        "Write a {n}-word phrase where first letters are {letters}:\n",
        "Spell {word} using the first letter of each word:\n",
        "Make an acrostic sentence for {word}:\n",
        "Write words starting with {letters} to spell {word}:\n",
        "Create a phrase where each word starts with a letter from {word}:\n",
        "Form a sentence spelling {word} acrostically:\n",
        "Write an acrostic: {word}:\n",
        "Compose words with initial letters {letters}:\n",
        "Generate a sentence for the acrostic {word}:\n",
        "Write {n} words with first letters spelling {word}:\n",
        "Create an acrostic sentence: {word}:\n",
    ]
    
    examples = []
    for word in ACROSTIC_WORDS:
        letters = ", ".join(list(word))
        n = len(word)
        for i, template in enumerate(templates):
            prompt = template.format(word=word, letters=letters, n=n)
            examples.append(PlanningExample(
                task_type=TaskType.CODE_RETURN,  # reuse type
                prompt=prompt,
                target_value=word,
                target_token_positions=[],
                example_id=hashlib.md5(f"acrostic_{word}_{i}".encode()).hexdigest()[:12],
                metadata={"word": word, "template_idx": i, "prompt": prompt},
            ))
    return examples


def make_first_letter_only_prompts():
    """Control: prompts where only the first letter differs.
    Tests if probe just reads the first letter (B vs S vs L vs D vs P)."""
    examples = []
    for word in ACROSTIC_WORDS:
        first = word[0]
        for i in range(20):
            # Minimal prompt — just the first letter as a cue
            prompt = f"Write a word starting with {first}:\n"
            examples.append(PlanningExample(
                task_type=TaskType.CODE_RETURN,
                prompt=prompt,
                target_value=word,
                target_token_positions=[],
                example_id=hashlib.md5(f"first_{word}_{i}".encode()).hexdigest()[:12],
                metadata={"word": word, "first_letter": first},
            ))
    return examples


def make_partial_acrostic_prompts():
    """Prompt with only first 2-3 letters given. Can model predict full word?
    B, R → BRAVE (the only word starting BR in our set)
    S, M → SMART
    L, I → LIGHT
    D, R → DREAM
    P, E → PEACE
    """
    examples = []
    for word in ACROSTIC_WORDS:
        for n_letters in [2, 3]:
            partial = ", ".join(list(word[:n_letters]))
            for i in range(10):
                prompt = f"Write words starting with letters {partial}, ...:\n"
                examples.append(PlanningExample(
                    task_type=TaskType.CODE_RETURN,
                    prompt=prompt,
                    target_value=word,
                    target_token_positions=[],
                    example_id=hashlib.md5(f"partial_{word}_{n_letters}_{i}".encode()).hexdigest()[:12],
                    metadata={"word": word, "n_letters": n_letters, "partial": partial},
                ))
    return examples


def run_behavioral_acrostic(model, examples, max_new=100):
    """Test if model can actually produce correct acrostics."""
    results = []
    for ex in examples:
        word = ex.metadata["word"]
        tokens = model.to_tokens(ex.prompt, prepend_bos=True)
        with torch.no_grad():
            output = model.generate(tokens, max_new_tokens=max_new, temperature=0.0)
        generated = model.to_string(output[0, tokens.shape[1]:])
        
        # Check if first letters of generated words spell the target
        gen_words = generated.strip().split()
        gen_initials = "".join(w[0].upper() for w in gen_words if w and w[0].isalpha())[:len(word)]
        
        correct = gen_initials == word
        results.append({
            "word": word,
            "prompt": ex.prompt[:50],
            "generated": generated[:100],
            "initials": gen_initials,
            "correct": correct,
        })
    
    return results


def run_probing_acrostic(model, examples, targets, layers, pca_dim=128, n_boot=500):
    """Probe activations to predict target word."""
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[ex.target_value] for ex in examples])
    
    caches = extract_activations_batch(model, model.tokenizer, examples, layers=layers, device="cuda")
    
    results = {}
    for layer in layers:
        min_seq = min(len(c.token_ids) for c in caches)
        best_acc, best_pos = 0, 0
        
        for pos in range(min_seq):
            X = np.stack([caches[i].activations[layer][pos] for i in range(len(examples))])
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            if X_s.shape[1] > pca_dim:
                X_s = PCA(n_components=min(pca_dim, X_s.shape[0]-1), random_state=42).fit_transform(X_s)
            scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            if scores.mean() > best_acc:
                best_acc = scores.mean()
                best_pos = pos
        
        # Bootstrap CI
        X_best = np.stack([caches[i].activations[layer][best_pos] for i in range(len(examples))])
        scaler = StandardScaler()
        X_best_s = scaler.fit_transform(X_best)
        if X_best_s.shape[1] > pca_dim:
            X_best_s = PCA(n_components=min(pca_dim, X_best_s.shape[0]-1), random_state=42).fit_transform(X_best_s)
        
        rng = np.random.RandomState(42)
        boot_accs = []
        for _ in range(n_boot):
            idx = rng.choice(len(X_best_s), len(X_best_s), replace=True)
            oob = list(set(range(len(X_best_s))) - set(idx))
            if len(oob) < 5 or len(np.unique(labels[idx])) < len(targets):
                continue
            p = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            p.fit(X_best_s[idx], labels[idx])
            boot_accs.append(p.score(X_best_s[oob], labels[oob]))
        
        ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5]) if boot_accs else (0, 0)
        
        results[f"layer_{layer}"] = {
            "probe": float(best_acc), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "best_pos": int(best_pos),
        }
        logger.info(f"  L{layer}: probe={best_acc:.3f} [{ci_lo:.3f},{ci_hi:.3f}] pos={best_pos}")
    
    return results, caches


def main():
    logger.info("=" * 70)
    logger.info("ACROSTIC PLANNING EXPERIMENT")
    logger.info("=" * 70)
    
    models = [
        ("gpt2-xl", torch.float32),
        ("pythia-2.8b", torch.float16),
        ("bigcode/santacoder", torch.float16),
        ("codellama/CodeLlama-7b-Python-hf", torch.float16),
        ("meta-llama/Llama-3.2-1B", torch.float16),
        ("meta-llama/Llama-3.2-1B-Instruct", torch.float16),
    ]
    
    all_results = {}
    
    for model_name, dtype in models:
        logger.info(f"\n{'='*70}")
        logger.info(f"MODEL: {model_name}")
        logger.info(f"{'='*70}")
        
        model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)
        model.eval()
        n_layers = model.cfg.n_layers
        layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))
        
        result = {"model": model_name, "n_layers": n_layers}
        
        # ============================================================
        # 1. BEHAVIORAL: Can the model do acrostics at all?
        # ============================================================
        logger.info("\n  === BEHAVIORAL: Can model produce acrostics? ===")
        acrostic_examples = make_acrostic_prompts()
        # Test on 5 examples per word (25 total) to save time
        beh_sample = []
        for word in ACROSTIC_WORDS:
            word_exs = [e for e in acrostic_examples if e.metadata["word"] == word][:5]
            beh_sample.extend(word_exs)
        
        beh_results = run_behavioral_acrostic(model, beh_sample)
        by_word = {}
        for r in beh_results:
            by_word.setdefault(r["word"], []).append(r["correct"])
        
        for word in ACROSTIC_WORDS:
            acc = np.mean(by_word.get(word, [False]))
            logger.info(f"    {word}: {acc:.1%}")
        
        overall_acc = np.mean([r["correct"] for r in beh_results])
        logger.info(f"    Overall: {overall_acc:.1%}")
        result["behavioral"] = {
            "overall": float(overall_acc),
            "by_word": {w: float(np.mean(v)) for w, v in by_word.items()},
            "examples": beh_results[:10],  # save first 10
        }
        
        # ============================================================
        # 2. PROBING: Full acrostic prompts (word is in prompt)
        # ============================================================
        logger.info("\n  === PROBING: Full acrostic prompts (100 examples) ===")
        logger.info("  (Target word IS in prompt — this should be trivially high)")
        probe_full, caches_full = run_probing_acrostic(
            model, acrostic_examples, ACROSTIC_WORDS, layers
        )
        result["probing_full"] = probe_full
        
        # ============================================================
        # 3. PROBING: First-letter-only prompts (harder baseline)
        # ============================================================
        logger.info("\n  === PROBING: First-letter-only (trivial — just reads B/S/L/D/P) ===")
        first_letter_examples = make_first_letter_only_prompts()
        probe_first, _ = run_probing_acrostic(
            model, first_letter_examples, ACROSTIC_WORDS, layers
        )
        result["probing_first_letter"] = probe_first
        
        # ============================================================
        # 4. PROBING: Partial acrostic (2-3 letters given)
        # ============================================================
        logger.info("\n  === PROBING: Partial acrostics (2-3 letters, can model predict full word?) ===")
        partial_examples = make_partial_acrostic_prompts()
        probe_partial, _ = run_probing_acrostic(
            model, partial_examples, ACROSTIC_WORDS, layers
        )
        result["probing_partial"] = probe_partial
        
        # ============================================================
        # 5. KEY COMPARISON: Full vs First-letter vs Partial
        # ============================================================
        logger.info("\n  === COMPARISON ===")
        logger.info(f"  {'Layer':<8} {'Full':>8} {'Partial':>10} {'1st Letter':>12} {'Chance':>8}")
        for layer in layers:
            lk = f"layer_{layer}"
            f = probe_full.get(lk, {}).get("probe", 0)
            p = probe_partial.get(lk, {}).get("probe", 0)
            fl = probe_first.get(lk, {}).get("probe", 0)
            logger.info(f"  L{layer:<6} {f:>8.3f} {p:>10.3f} {fl:>12.3f} {0.200:>8.3f}")
        
        all_results[model_name] = result
        
        del model, caches_full
        torch.cuda.empty_cache()
    
    # Save
    outfile = "results/lookahead/final/acrostic_results.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved to {outfile}")
    
    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
