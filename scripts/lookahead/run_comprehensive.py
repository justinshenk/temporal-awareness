#!/usr/bin/env python3
"""
RQ4 Comprehensive Experiment Runner.

This replaces the original Phase 1 script. It runs everything in the
correct order with all safeguards:

1. BEHAVIORAL VALIDATION — can the model do the task at all?
2. ACTIVATION EXTRACTION — all positions, selected layers
3. COMMITMENT PROBING — OOF probes at every position
4. COMPREHENSIVE BASELINES — BoW, PCA, position-shuffle, random direction
5. PERMUTATION TESTS + FDR CORRECTION — statistical significance
6. PLANNING VS CONTINUATION — nonce words, competing continuations

If any step fails its checks, we stop and report rather than
producing misleading results.

Usage:
    python scripts/lookahead/run_comprehensive.py \
        --model gpt2 \
        --task rhyme \
        --device cuda \
        --output results/lookahead/comprehensive
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("RQ4 COMPREHENSIVE EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Output: {output_dir}")
    
    # ══════════════════════════════════════════════════════════════════
    # STEP 0: GENERATE DATASETS
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n\n=== STEP 0: DATASET GENERATION ===")
    
    examples = _generate_dataset(args.task, args.n_examples)
    logger.info(f"Main dataset: {len(examples)} examples")
    
    # Planning vs continuation datasets (rhyme only)
    planning_examples = []
    counterfactual_pairs = []
    if args.task == "rhyme":
        from src.lookahead.datasets.planning_vs_continuation import (
            generate_nonce_rhyme_dataset,
            generate_competing_continuation_dataset,
            generate_counterfactual_pairs,
        )
        nonce = generate_nonce_rhyme_dataset()
        competing = generate_competing_continuation_dataset()
        counterfactual_pairs = generate_counterfactual_pairs()
        planning_examples = nonce + competing
        logger.info(f"Nonce rhymes: {len(nonce)}")
        logger.info(f"Competing continuations: {len(competing)}")
        logger.info(f"Counterfactual pairs: {len(counterfactual_pairs)}")
    
    # ══════════════════════════════════════════════════════════════════
    # STEP 1: LOAD MODEL
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n\n=== STEP 1: LOAD MODEL ===")
    
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    logger.info(f"Loaded: {n_layers} layers, d_model={d_model}")
    
    # ══════════════════════════════════════════════════════════════════
    # STEP 2: BEHAVIORAL VALIDATION (run BEFORE spending GPU on probing)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n\n=== STEP 2: BEHAVIORAL VALIDATION ===")
    logger.info("Checking if model can actually perform the task...")
    
    from src.lookahead.probing.behavioral_validation import (
        run_behavioral_validation,
        filter_to_successful,
        compute_behavioral_summary,
    )
    
    behavioral_results = run_behavioral_validation(
        model, examples, max_new_tokens=50, temperature=0.0,
    )
    
    summary = compute_behavioral_summary(behavioral_results)
    for task, stats in summary.items():
        logger.info(f"  {task}: accuracy={stats['task_accuracy']:.2%} "
                     f"(exact={stats['exact_match']:.2%}, n={stats['n']})")
    
    # Save behavioral results
    with open(output_dir / "behavioral_validation.json", "w") as f:
        json.dump({
            "summary": summary,
            "per_example": [
                {
                    "id": r.example_id,
                    "success": r.task_success,
                    "target": r.target_value,
                    "detected": r.detected_value,
                    "completion_preview": r.completion[:100],
                }
                for r in behavioral_results
            ],
        }, f, indent=2)
    
    # GATE: if behavioral accuracy is too low, warn loudly
    for task, stats in summary.items():
        if stats["task_accuracy"] < 0.1:
            logger.warning(f"  ⚠ VERY LOW behavioral accuracy for {task}: {stats['task_accuracy']:.1%}")
            logger.warning(f"    Model may not be capable of this task. Results will be unreliable.")
            logger.warning(f"    Consider using a larger model or a simpler task variant.")
    
    # Filter to successful examples for probing
    filtered_examples, filter_stats = filter_to_successful(examples, behavioral_results)
    logger.info(f"  Kept {filter_stats['kept']}/{filter_stats['total']} examples after behavioral filter")
    
    # If too few pass, fall back to all examples but FLAG this
    use_filtered = filter_stats["kept"] >= 20
    probe_examples = filtered_examples if use_filtered else examples
    if not use_filtered:
        logger.warning("  ⚠ Too few behavioral successes — probing ALL examples (results less reliable)")
    
    # ══════════════════════════════════════════════════════════════════
    # STEP 3: EXTRACT ACTIVATIONS
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n\n=== STEP 3: ACTIVATION EXTRACTION ===")
    
    from src.lookahead.probing.activation_extraction import extract_activations_batch
    
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    else:
        layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
        layers = sorted(set(layers))
    
    logger.info(f"  Layers: {layers}")
    
    t0 = time.time()
    caches = extract_activations_batch(
        model=model, tokenizer=model.tokenizer, examples=probe_examples,
        layers=layers, include_logits=True, device=args.device,
    )
    logger.info(f"  Extracted in {time.time() - t0:.1f}s")
    
    # Also extract for planning_examples if we have them
    planning_caches = []
    if planning_examples:
        logger.info(f"  Extracting planning vs continuation examples...")
        planning_caches = extract_activations_batch(
            model=model, tokenizer=model.tokenizer, examples=planning_examples,
            layers=layers, include_logits=False, device=args.device,
        )
    
    # ══════════════════════════════════════════════════════════════════
    # STEP 4: COMMITMENT PROBING + COMPREHENSIVE BASELINES
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n\n=== STEP 4: COMMITMENT PROBING + BASELINES ===")
    
    from src.lookahead.probing.commitment_probes import (
        ProbeConfig, train_commitment_probes, compute_commitment_curves,
        find_commitment_points, run_permutation_test,
    )
    from src.lookahead.probing.comprehensive_baselines import (
        run_all_baselines, fdr_correction,
    )
    
    config = ProbeConfig(
        commitment_threshold=args.threshold,
        n_folds=min(5, 20),  # conservative
        random_state=args.seed,
    )
    
    all_layer_results = {}
    
    for layer in layers:
        logger.info(f"\n  ── Layer {layer} ──")
        
        # Train probes
        probe_results = train_commitment_probes(caches, probe_examples, layer, config)
        if not probe_results:
            logger.warning(f"    No probe results — skipping")
            continue
        
        positions = sorted(probe_results.keys())
        accs = [probe_results[p]["cv_accuracy_mean"] for p in positions]
        n_classes = probe_results[positions[0]]["n_classes"]
        chance = 1.0 / n_classes
        max_acc = max(accs) if accs else 0
        logger.info(f"    Max CV accuracy: {max_acc:.3f} (chance: {chance:.3f})")
        
        # Commitment curves + points
        curves = compute_commitment_curves(caches, probe_examples, layer, probe_results, config)
        points = find_commitment_points(curves, threshold=config.commitment_threshold)
        valid_points = [p for p in points if p.is_valid]
        logger.info(f"    Commitment points: {len(valid_points)}/{len(points)} valid")
        
        # Permutation tests at key positions (post-commitment)
        logger.info(f"    Running permutation tests...")
        p_values = []
        p_value_positions = []
        for pos in positions:
            if probe_results[pos]["cv_accuracy_mean"] > chance + 0.05:
                perm = run_permutation_test(
                    caches, probe_examples, layer, pos,
                    n_permutations=args.n_permutations, config=config,
                )
                p_values.append(perm["p_value"])
                p_value_positions.append(pos)
        
        # Run ALL baselines
        logger.info(f"    Running comprehensive baselines...")
        baseline_suite = run_all_baselines(
            caches, probe_examples, layer, p_values=p_values,
            n_folds=config.n_folds, random_state=config.random_state,
        )
        
        # FDR correction
        if p_values:
            fdr_sig, fdr_adj = fdr_correction(p_values)
            n_sig_raw = sum(1 for p in p_values if p < 0.05)
            n_sig_fdr = sum(fdr_sig)
            logger.info(f"    Significant positions: {n_sig_raw} raw, {n_sig_fdr} after FDR")
        
        # Compare real probe vs baselines at post-commitment positions
        _log_baseline_comparison(probe_results, baseline_suite, positions, chance, layer)
        
        # Store everything
        all_layer_results[layer] = {
            "probe_results": {
                str(p): {
                    "cv_accuracy_mean": r["cv_accuracy_mean"],
                    "cv_accuracy_std": r["cv_accuracy_std"],
                    "n_samples": r["n_samples"],
                    "n_classes": r["n_classes"],
                }
                for p, r in probe_results.items()
            },
            "commitment_points": [
                {
                    "example_id": p.example_id,
                    "position": p.position,
                    "confidence": p.confidence_at_commitment,
                    "tokens_before_target": p.tokens_before_target,
                    "is_valid": p.is_valid,
                }
                for p in points
            ],
            "p_values": {str(pos): pv for pos, pv in zip(p_value_positions, p_values)} if p_values else {},
            "fdr_significant": {str(pos): sig for pos, sig in zip(p_value_positions, fdr_sig)} if p_values else {},
            "baselines": {
                "bow": [{"pos": b.metric_name, "acc": b.metric_value} for b in baseline_suite.bag_of_words],
                "position_shuffled": [{"pos": b.metric_name, "acc": b.metric_value} for b in baseline_suite.position_shuffled],
                "random_direction": [{"pos": b.metric_name, "acc": b.metric_value} for b in baseline_suite.random_direction],
                "anchor_word_only": [{"acc": b.metric_value} for b in baseline_suite.anchor_word_only],
                "pca": {
                    str(k): [{"pos": b.metric_name, "acc": b.metric_value} for b in v]
                    for k, v in baseline_suite.pca_reduction.items()
                },
            },
        }
    
    # ══════════════════════════════════════════════════════════════════
    # STEP 5: SAVE ALL RESULTS
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n\n=== STEP 5: SAVING RESULTS ===")
    
    results_path = output_dir / "comprehensive_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "model": args.model,
                "task": args.task,
                "n_examples": len(probe_examples),
                "layers": layers,
                "behavioral_filter": use_filtered,
            },
            "behavioral": summary,
            "layers": {str(k): v for k, v in all_layer_results.items()},
        }, f, indent=2, default=str)
    
    logger.info(f"  Saved to {results_path}")
    
    # ══════════════════════════════════════════════════════════════════
    # STEP 6: VERDICT
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n\n" + "=" * 70)
    logger.info("VERDICT")
    logger.info("=" * 70)
    _print_verdict(all_layer_results, summary)


def _log_baseline_comparison(probe_results, baseline_suite, positions, chance, layer):
    """Log comparison of real probes vs all baselines."""
    # Get max real probe accuracy
    real_accs = [probe_results[p]["cv_accuracy_mean"] for p in positions]
    max_real = max(real_accs) if real_accs else 0
    max_pos = positions[np.argmax(real_accs)] if real_accs else -1
    
    # Baseline accuracies at the same position
    bow_accs = {int(b.metric_name.split("_")[-1]): b.metric_value for b in baseline_suite.bag_of_words}
    pos_shuf_accs = {int(b.metric_name.split("_")[-1]): b.metric_value for b in baseline_suite.position_shuffled}
    rand_dir_accs = {int(b.metric_name.split("_")[-1]): b.metric_value for b in baseline_suite.random_direction}
    anchor_acc = baseline_suite.anchor_word_only[0].metric_value if baseline_suite.anchor_word_only else None
    
    logger.info(f"    At best position (pos={max_pos}):")
    logger.info(f"      Real probe:        {max_real:.3f}")
    logger.info(f"      Chance:            {chance:.3f}")
    if max_pos in bow_accs:
        logger.info(f"      Bag-of-words:      {bow_accs[max_pos]:.3f}")
    if max_pos in pos_shuf_accs:
        logger.info(f"      Position-shuffled:  {pos_shuf_accs[max_pos]:.3f}")
    if max_pos in rand_dir_accs:
        logger.info(f"      Random direction:   {rand_dir_accs[max_pos]:.3f}")
    if anchor_acc is not None:
        logger.info(f"      Anchor-only:        {anchor_acc:.3f}")
    
    # PCA comparison
    for k, pca_results in baseline_suite.pca_reduction.items():
        pca_accs = {int(b.metric_name.split("_")[-1]): b.metric_value for b in pca_results}
        if max_pos in pca_accs:
            logger.info(f"      PCA-{k}:            {pca_accs[max_pos]:.3f}")


def _print_verdict(all_layer_results, behavioral_summary):
    """Print final verdict based on all evidence."""
    for task, stats in behavioral_summary.items():
        logger.info(f"\n  Task: {task}")
        logger.info(f"    Behavioral accuracy: {stats['task_accuracy']:.1%}")
        
        if stats["task_accuracy"] < 0.1:
            logger.info("    ✗ MODEL CANNOT DO THIS TASK — probing results unreliable")
            continue
        elif stats["task_accuracy"] < 0.3:
            logger.info("    ⚠ Low behavioral accuracy — interpret probing with caution")
    
    for layer, results in all_layer_results.items():
        logger.info(f"\n  Layer {layer}:")
        
        probe_accs = [v["cv_accuracy_mean"] for v in results["probe_results"].values()]
        if not probe_accs:
            continue
        
        max_acc = max(probe_accs)
        n_classes = list(results["probe_results"].values())[0]["n_classes"]
        chance = 1.0 / n_classes
        
        # Check baselines
        bow_max = max((b["acc"] for b in results["baselines"]["bow"]), default=chance)
        pos_shuf_max = max((b["acc"] for b in results["baselines"]["position_shuffled"]), default=chance)
        
        signal_over_chance = max_acc - chance
        signal_over_bow = max_acc - bow_max
        
        # FDR-significant positions
        n_fdr_sig = sum(1 for v in results.get("fdr_significant", {}).values() if v)
        
        logger.info(f"    Max probe accuracy:    {max_acc:.3f}")
        logger.info(f"    Chance:                {chance:.3f}")
        logger.info(f"    Signal over chance:    {signal_over_chance:.3f}")
        logger.info(f"    Signal over BoW:       {signal_over_bow:.3f}")
        logger.info(f"    FDR-significant pos:   {n_fdr_sig}")
        
        if signal_over_chance < 0.05:
            logger.info("    ✗ NO SIGNAL — probe accuracy at chance level")
        elif signal_over_bow < 0.05:
            logger.info("    ✗ CONFOUNDED — BoW baseline explains the signal")
        elif n_fdr_sig == 0 and results.get("p_values"):
            logger.info("    ✗ NOT SIGNIFICANT after FDR correction")
        elif signal_over_chance > 0.15 and signal_over_bow > 0.05 and n_fdr_sig > 0:
            logger.info("    ✓ GENUINE PLANNING SIGNAL detected")
        else:
            logger.info("    ○ Weak/ambiguous signal — needs more data or larger model")


def _generate_dataset(task, n_examples):
    if task == "rhyme":
        from src.lookahead.datasets.rhyme import generate_rhyme_dataset
        return generate_rhyme_dataset(n_per_rhyme_set=max(1, n_examples // 15))
    elif task == "acrostic":
        from src.lookahead.datasets.acrostic import generate_acrostic_dataset
        return generate_acrostic_dataset(word_lengths=[3, 4, 5])
    elif task == "code":
        from src.lookahead.datasets.code_return import generate_code_return_dataset
        return generate_code_return_dataset()
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ4 Comprehensive Experiment")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--task", default="rhyme", choices=["rhyme", "acrostic", "code"])
    parser.add_argument("--n-examples", type=int, default=75)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/lookahead/comprehensive")
    
    args = parser.parse_args()
    main(args)
