#!/usr/bin/env python3
"""
Investigation: Why does ablating L6 improve IM classification?

This script investigates the surprising finding that ablating the long-term 
track (Layer 6) IMPROVED immediate classification from 88% to 100%.

Key questions:
1. Which 6 pairs fail at baseline?
2. Does L6 ablation specifically fix these failures?
3. What makes these pairs different?
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

# Import from activation_patching
import sys
sys.path.insert(0, str(Path(__file__).parent))
from activation_patching import (
    load_model_and_probe, load_dataset, 
    get_activations_with_cache, run_with_multi_patching
)


def investigate_baseline_failures(model, tokenizer, probe, pairs, probe_layer=8):
    """Identify which immediate pairs fail at baseline."""
    print("=" * 70)
    print("INVESTIGATION: Baseline IM Failures")
    print("=" * 70)
    
    failures = []
    successes = []
    
    for i, pair in enumerate(tqdm(pairs, desc="Checking baseline")):
        immediate_text = pair['question'] + pair['immediate']
        
        # Get baseline prediction
        result = get_activations_with_cache(model, tokenizer, immediate_text, [])
        act = result['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
        pred = probe.predict([act])[0]
        prob = probe.predict_proba([act])[0]
        
        info = {
            'pair_idx': i,
            'question': pair['question'][:80] + '...' if len(pair['question']) > 80 else pair['question'],
            'immediate': pair['immediate'][:60] + '...' if len(pair['immediate']) > 60 else pair['immediate'],
            'long_term': pair['long_term'][:60] + '...' if len(pair['long_term']) > 60 else pair['long_term'],
            'category': pair.get('category', 'unknown'),
            'pred': pred,
            'prob_longterm': prob[1],
            'prob_immediate': prob[0]
        }
        
        if pred == 0:  # Correctly predicts immediate
            successes.append(info)
        else:  # Incorrectly predicts long-term
            failures.append(info)
    
    print(f"\nBaseline: {len(successes)}/{len(pairs)} correct ({100*len(successes)/len(pairs):.1f}%)")
    print(f"Failures: {len(failures)} pairs")
    
    if failures:
        print("\n" + "-" * 70)
        print("FAILED PAIRS (classified as long-term when should be immediate):")
        print("-" * 70)
        for f in failures:
            print(f"\nPair {f['pair_idx']} [{f['category']}]:")
            print(f"  Q: {f['question']}")
            print(f"  IM: {f['immediate']}")
            print(f"  LT: {f['long_term']}")
            print(f"  Prob(LT): {f['prob_longterm']:.3f}, Prob(IM): {f['prob_immediate']:.3f}")
    
    return failures, successes


def test_l6_ablation_effect(model, tokenizer, probe, pairs, failures, probe_layer=8):
    """Test if L6 ablation specifically fixes the baseline failures."""
    print("\n" + "=" * 70)
    print("INVESTIGATION: L6 Ablation Effect")
    print("=" * 70)
    
    failure_indices = {f['pair_idx'] for f in failures}
    
    fixed_failures = []
    broken_successes = []
    
    L6 = 6  # Long-term track layer
    
    for i, pair in enumerate(tqdm(pairs, desc="Testing L6 ablation")):
        immediate_text = pair['question'] + pair['immediate']
        
        # Get baseline prediction
        baseline_result = get_activations_with_cache(model, tokenizer, immediate_text, [])
        baseline_act = baseline_result['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
        baseline_pred = probe.predict([baseline_act])[0]
        baseline_prob = probe.predict_proba([baseline_act])[0]
        
        # Get L6 ablated prediction
        interventions = [{'layer': L6, 'component': 'residual', 'method': 'ablation'}]
        ablated_outputs = run_with_multi_patching(model, tokenizer, immediate_text, interventions)
        ablated_act = ablated_outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
        ablated_pred = probe.predict([ablated_act])[0]
        ablated_prob = probe.predict_proba([ablated_act])[0]
        
        was_failure = i in failure_indices
        
        if was_failure and ablated_pred == 0:
            fixed_failures.append({
                'pair_idx': i,
                'baseline_prob_lt': baseline_prob[1],
                'ablated_prob_lt': ablated_prob[1],
                'prob_change': baseline_prob[1] - ablated_prob[1]
            })
        elif not was_failure and ablated_pred == 1:
            broken_successes.append({
                'pair_idx': i,
                'baseline_prob_lt': baseline_prob[1],
                'ablated_prob_lt': ablated_prob[1],
                'prob_change': ablated_prob[1] - baseline_prob[1]
            })
    
    print(f"\nResults:")
    print(f"  Failures fixed by L6 ablation: {len(fixed_failures)}/{len(failures)}")
    print(f"  Successes broken by L6 ablation: {len(broken_successes)}/{len(pairs) - len(failures)}")
    
    if fixed_failures:
        print("\n  Fixed failures:")
        for f in fixed_failures:
            print(f"    Pair {f['pair_idx']}: P(LT) {f['baseline_prob_lt']:.3f} → {f['ablated_prob_lt']:.3f} "
                  f"(Δ = -{f['prob_change']:.3f})")
    
    if broken_successes:
        print("\n  Broken successes:")
        for b in broken_successes:
            print(f"    Pair {b['pair_idx']}: P(LT) {b['baseline_prob_lt']:.3f} → {b['ablated_prob_lt']:.3f} "
                  f"(Δ = +{b['prob_change']:.3f})")
    
    return fixed_failures, broken_successes


def analyze_probability_shifts(model, tokenizer, probe, pairs, probe_layer=8):
    """Analyze how L6 ablation affects probabilities across all pairs."""
    print("\n" + "=" * 70)
    print("INVESTIGATION: Probability Shift Analysis")
    print("=" * 70)
    
    L6 = 6
    
    im_shifts = []  # Shift in P(LT) for immediate inputs
    lt_shifts = []  # Shift in P(LT) for long-term inputs
    
    for pair in tqdm(pairs, desc="Analyzing probability shifts"):
        immediate_text = pair['question'] + pair['immediate']
        longterm_text = pair['question'] + pair['long_term']
        
        # Immediate inputs
        baseline_im = get_activations_with_cache(model, tokenizer, immediate_text, [])
        baseline_im_act = baseline_im['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
        baseline_im_prob = probe.predict_proba([baseline_im_act])[0][1]
        
        interventions = [{'layer': L6, 'component': 'residual', 'method': 'ablation'}]
        ablated_im = run_with_multi_patching(model, tokenizer, immediate_text, interventions)
        ablated_im_act = ablated_im.hidden_states[probe_layer][0, -1, :].cpu().numpy()
        ablated_im_prob = probe.predict_proba([ablated_im_act])[0][1]
        
        im_shifts.append(ablated_im_prob - baseline_im_prob)
        
        # Long-term inputs
        baseline_lt = get_activations_with_cache(model, tokenizer, longterm_text, [])
        baseline_lt_act = baseline_lt['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
        baseline_lt_prob = probe.predict_proba([baseline_lt_act])[0][1]
        
        ablated_lt = run_with_multi_patching(model, tokenizer, longterm_text, interventions)
        ablated_lt_act = ablated_lt.hidden_states[probe_layer][0, -1, :].cpu().numpy()
        ablated_lt_prob = probe.predict_proba([ablated_lt_act])[0][1]
        
        lt_shifts.append(ablated_lt_prob - baseline_lt_prob)
    
    im_shifts = np.array(im_shifts)
    lt_shifts = np.array(lt_shifts)
    
    print(f"\nP(LT) shifts after L6 ablation:")
    print(f"  Immediate inputs: mean={im_shifts.mean():.3f}, std={im_shifts.std():.3f}")
    print(f"    Range: [{im_shifts.min():.3f}, {im_shifts.max():.3f}]")
    print(f"    Decreased for {(im_shifts < 0).sum()}/{len(im_shifts)} pairs")
    print(f"\n  Long-term inputs: mean={lt_shifts.mean():.3f}, std={lt_shifts.std():.3f}")
    print(f"    Range: [{lt_shifts.min():.3f}, {lt_shifts.max():.3f}]")
    print(f"    Decreased for {(lt_shifts < 0).sum()}/{len(lt_shifts)} pairs")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    if im_shifts.mean() < -0.05:
        print("  → L6 ablation REDUCES P(LT) for immediate inputs")
        print("    This suggests L6 adds 'long-term bias' that hurts IM classification")
    if lt_shifts.mean() < -0.05:
        print("  → L6 ablation also REDUCES P(LT) for long-term inputs")
        print("    This confirms L6 is necessary for LT encoding")
    
    return im_shifts, lt_shifts


def analyze_failure_characteristics(failures, pairs):
    """Analyze what makes the failed pairs different."""
    print("\n" + "=" * 70)
    print("INVESTIGATION: Failure Characteristics")
    print("=" * 70)
    
    if not failures:
        print("No failures to analyze!")
        return
    
    failure_indices = {f['pair_idx'] for f in failures}
    
    # Check category distribution
    failure_categories = [pairs[f['pair_idx']].get('category', 'unknown') for f in failures]
    all_categories = [p.get('category', 'unknown') for p in pairs]
    
    print("\nCategory distribution:")
    from collections import Counter
    all_cat_counts = Counter(all_categories)
    fail_cat_counts = Counter(failure_categories)
    
    for cat in sorted(all_cat_counts.keys()):
        total = all_cat_counts[cat]
        failed = fail_cat_counts.get(cat, 0)
        print(f"  {cat}: {failed}/{total} failed ({100*failed/total:.0f}%)")
    
    # Check text length characteristics
    failure_im_lens = [len(pairs[f['pair_idx']]['immediate']) for f in failures]
    failure_lt_lens = [len(pairs[f['pair_idx']]['long_term']) for f in failures]
    success_im_lens = [len(p['immediate']) for i, p in enumerate(pairs) if i not in failure_indices]
    success_lt_lens = [len(p['long_term']) for i, p in enumerate(pairs) if i not in failure_indices]
    
    print(f"\nText length analysis:")
    print(f"  Failed IM answers: avg {np.mean(failure_im_lens):.1f} chars")
    print(f"  Success IM answers: avg {np.mean(success_im_lens):.1f} chars")
    print(f"  Failed LT answers: avg {np.mean(failure_lt_lens):.1f} chars")
    print(f"  Success LT answers: avg {np.mean(success_lt_lens):.1f} chars")
    
    # Check probability characteristics
    print(f"\nProbability characteristics of failures:")
    for f in failures:
        print(f"  Pair {f['pair_idx']}: P(LT)={f['prob_longterm']:.3f} (confident={f['prob_longterm'] > 0.7})")


def main():
    print("Loading model and probe...")
    model, tokenizer, probe = load_model_and_probe(probe_layer=8)
    
    pairs = load_dataset()
    print(f"Loaded {len(pairs)} pairs")
    
    # Investigation 1: Identify baseline failures
    failures, successes = investigate_baseline_failures(model, tokenizer, probe, pairs)
    
    # Investigation 2: Test L6 ablation effect
    fixed, broken = test_l6_ablation_effect(model, tokenizer, probe, pairs, failures)
    
    # Investigation 3: Analyze probability shifts
    im_shifts, lt_shifts = analyze_probability_shifts(model, tokenizer, probe, pairs)
    
    # Investigation 4: Analyze failure characteristics
    analyze_failure_characteristics(failures, pairs)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline IM accuracy: {len(successes)}/{len(pairs)} ({100*len(successes)/len(pairs):.1f}%)")
    print(f"Failures fixed by L6 ablation: {len(fixed)}/{len(failures)}")
    print(f"Successes broken by L6 ablation: {len(broken)}/{len(successes)}")
    print(f"Net effect: +{len(fixed) - len(broken)} correct classifications")
    print(f"\nMean P(LT) shift for IM inputs: {im_shifts.mean():.3f}")
    print(f"Mean P(LT) shift for LT inputs: {lt_shifts.mean():.3f}")
    
    # Save results
    output_dir = Path('results/causal_encoding')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'baseline_accuracy': len(successes) / len(pairs),
        'n_failures': len(failures),
        'failures_fixed': len(fixed),
        'successes_broken': len(broken),
        'net_effect': len(fixed) - len(broken),
        'mean_im_prob_shift': float(im_shifts.mean()),
        'mean_lt_prob_shift': float(lt_shifts.mean()),
        'failure_pairs': [f['pair_idx'] for f in failures]
    }
    
    with open(output_dir / 'interference_investigation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'interference_investigation.json'}")


if __name__ == '__main__':
    main()
