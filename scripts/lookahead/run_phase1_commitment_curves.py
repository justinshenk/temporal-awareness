#!/usr/bin/env python3
"""
RQ4 Phase 1: Generate planning datasets and compute commitment curves.

This is the first experiment to run. It will tell us whether the signal
exists at all. If commitment curves show clear sigmoidal shape (confidence
jumps from ~chance to ~90% over a few tokens), we proceed to causal
patching. If curves are flat or noisy, we need to redesign tasks.

Usage:
    python scripts/lookahead/run_phase1_commitment_curves.py \
        --model gpt2 \
        --task rhyme \
        --output results/lookahead/phase1

Expected runtime: ~10-20 min on CPU for GPT-2 with 50 examples.
GPU recommended for 7B+ models.
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
    logger.info("RQ4 PHASE 1: COMMITMENT CURVE ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    # ── Step 1: Generate dataset ──────────────────────────────────────────
    logger.info("Step 1: Generating planning dataset...")
    
    examples = _generate_dataset(args.task, args.n_examples)
    logger.info(f"  Generated {len(examples)} examples")
    
    # Save dataset
    from src.lookahead.datasets.rhyme import save_dataset
    dataset_path = output_dir / f"{args.task}_dataset.json"
    save_dataset(examples, dataset_path)
    logger.info(f"  Saved to {dataset_path}")
    
    # ── Step 2: Load model ────────────────────────────────────────────────
    logger.info(f"\nStep 2: Loading model {args.model}...")
    
    model = _load_model(args.model, args.device)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    logger.info(f"  Layers: {n_layers}, d_model: {d_model}")
    logger.info(f"  Device: {next(model.parameters()).device}")
    
    # ── Step 3: Extract activations ───────────────────────────────────────
    logger.info(f"\nStep 3: Extracting activations at all positions...")
    
    from src.lookahead.probing.activation_extraction import extract_activations_batch
    
    # Choose layers to probe: early, middle, late
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    else:
        # Default: sample across full depth
        layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
        layers = sorted(set(layers))
    
    logger.info(f"  Probing layers: {layers}")
    
    t0 = time.time()
    caches = extract_activations_batch(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        layers=layers,
        include_logits=args.include_logits,
        device=args.device,
    )
    t_extract = time.time() - t0
    logger.info(f"  Extracted in {t_extract:.1f}s")
    
    # ── Step 4: Train commitment probes ───────────────────────────────────
    logger.info(f"\nStep 4: Training commitment probes...")
    
    from src.lookahead.probing.commitment_probes import (
        ProbeConfig,
        train_commitment_probes,
        compute_commitment_curves,
        find_commitment_points,
        run_shuffled_label_baseline,
    )
    
    config = ProbeConfig(
        commitment_threshold=args.threshold,
        n_shuffle_iterations=args.n_shuffle,
        random_state=args.seed,
    )
    
    all_results = {}
    all_curves = {}
    all_points = {}
    all_baselines = {}
    
    for layer in layers:
        logger.info(f"\n  Layer {layer}:")
        
        # Train probes
        probe_results = train_commitment_probes(caches, examples, layer, config)
        
        if not probe_results:
            logger.warning(f"    No probe results — skipping layer {layer}")
            continue
        
        # Log accuracy progression
        positions = sorted(probe_results.keys())
        accs = [probe_results[p]["cv_accuracy_mean"] for p in positions]
        max_acc = max(accs) if accs else 0
        max_pos = positions[np.argmax(accs)] if accs else -1
        logger.info(f"    Max CV accuracy: {max_acc:.3f} at position {max_pos}")
        logger.info(f"    Chance level: {1.0 / probe_results[positions[0]]['n_classes']:.3f}")
        
        # Compute commitment curves
        curves = compute_commitment_curves(caches, examples, layer, probe_results, config)
        points = find_commitment_points(curves, threshold=config.commitment_threshold)
        
        valid_points = [p for p in points if p.is_valid]
        logger.info(f"    Commitment points found: {len(valid_points)}/{len(points)}")
        
        if valid_points:
            mean_tokens_before = np.mean([p.tokens_before_target for p in valid_points])
            logger.info(f"    Mean tokens before target: {mean_tokens_before:.1f}")
        
        # Run shuffled-label baseline
        logger.info(f"    Running shuffled-label baseline ({config.n_shuffle_iterations} iterations)...")
        baselines = run_shuffled_label_baseline(caches, examples, layer, config)
        
        if baselines:
            baseline_mean = np.mean([b.metric_value for b in baselines])
            logger.info(f"    Shuffled baseline mean accuracy: {baseline_mean:.3f}")
        
        # Store results
        all_results[layer] = {
            pos: {
                "cv_accuracy_mean": r["cv_accuracy_mean"],
                "cv_accuracy_std": r["cv_accuracy_std"],
                "train_accuracy": r["train_accuracy"],
                "auc": r["auc"],
                "n_samples": r["n_samples"],
                "n_classes": r["n_classes"],
            }
            for pos, r in probe_results.items()
        }
        all_curves[layer] = curves
        all_points[layer] = points
        all_baselines[layer] = baselines
    
    # ── Step 5: Save results ──────────────────────────────────────────────
    logger.info(f"\n\nStep 5: Saving results...")
    
    # Save probe results (JSON-safe)
    results_path = output_dir / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {str(k): v for k, v in all_results.items()},
            f, indent=2, default=str,
        )
    logger.info(f"  Probe results: {results_path}")
    
    # Save commitment curves (numpy)
    for layer, curves in all_curves.items():
        curves_path = output_dir / f"commitment_curves_layer{layer}.npz"
        curve_data = {}
        for i, curve in enumerate(curves):
            curve_data[f"positions_{i}"] = curve.positions
            curve_data[f"confidences_{i}"] = curve.confidences
            curve_data[f"target_position_{i}"] = np.array([curve.target_position])
        np.savez(curves_path, **curve_data, n_curves=np.array([len(curves)]))
    logger.info(f"  Commitment curves saved")
    
    # Save commitment points
    points_path = output_dir / "commitment_points.json"
    all_points_json = {}
    for layer, points in all_points.items():
        all_points_json[str(layer)] = [
            {
                "example_id": p.example_id,
                "position": p.position,
                "confidence": p.confidence_at_commitment,
                "tokens_before_target": p.tokens_before_target,
                "is_valid": p.is_valid,
            }
            for p in points
        ]
    with open(points_path, "w") as f:
        json.dump(all_points_json, f, indent=2)
    logger.info(f"  Commitment points: {points_path}")
    
    # Save baselines
    baselines_path = output_dir / "shuffled_baselines.json"
    baselines_json = {}
    for layer, baselines in all_baselines.items():
        baselines_json[str(layer)] = [
            {
                "metric_name": b.metric_name,
                "metric_value": b.metric_value,
                "ci_low": b.confidence_interval[0],
                "ci_high": b.confidence_interval[1],
            }
            for b in baselines
        ]
    with open(baselines_path, "w") as f:
        json.dump(baselines_json, f, indent=2)
    logger.info(f"  Baselines: {baselines_path}")
    
    # ── Step 6: Summary ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    for layer in layers:
        if layer not in all_results:
            continue
        
        results = all_results[layer]
        positions = sorted(results.keys())
        accs = [results[p]["cv_accuracy_mean"] for p in positions]
        
        if not accs:
            continue
        
        max_acc = max(accs)
        chance = 1.0 / results[positions[0]]["n_classes"]
        
        # Get baseline for comparison
        baseline_accs = [b.metric_value for b in all_baselines.get(layer, [])]
        baseline_mean = np.mean(baseline_accs) if baseline_accs else chance
        
        # Verdict
        signal_above_baseline = max_acc - baseline_mean
        
        logger.info(f"\n  Layer {layer}:")
        logger.info(f"    Max probe accuracy:       {max_acc:.3f}")
        logger.info(f"    Chance level:             {chance:.3f}")
        logger.info(f"    Shuffled baseline:        {baseline_mean:.3f}")
        logger.info(f"    Signal above baseline:    {signal_above_baseline:.3f}")
        
        if signal_above_baseline > 0.15:
            logger.info(f"    ✓ STRONG planning signal detected")
        elif signal_above_baseline > 0.05:
            logger.info(f"    ○ Weak planning signal — may need more data")
        else:
            logger.info(f"    ✗ No reliable planning signal at this layer")
        
        # Commitment timing
        valid_points = [p for p in all_points.get(layer, []) if p.is_valid]
        if valid_points:
            tokens_before = [p.tokens_before_target for p in valid_points]
            logger.info(f"    Commitment timing: {np.mean(tokens_before):.1f} ± {np.std(tokens_before):.1f} tokens before target")
    
    logger.info("\n" + "=" * 70)
    logger.info("Next: If strong signal found, proceed to Phase 2 (causal patching)")
    logger.info("      Run: scripts/lookahead/run_phase2_causal_patching.py")
    logger.info("=" * 70)


def _generate_dataset(task: str, n_examples: int):
    """Generate dataset for the specified task."""
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
        raise ValueError(f"Unknown task: {task}. Choose from: rhyme, acrostic, code")


def _load_model(model_name: str, device: str):
    """Load model via TransformerLens."""
    from transformer_lens import HookedTransformer
    
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
    )
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ4 Phase 1: Commitment curve analysis"
    )
    parser.add_argument("--model", default="gpt2", help="Model name (TransformerLens compatible)")
    parser.add_argument("--task", default="rhyme", choices=["rhyme", "acrostic", "code"],
                        help="Planning task type")
    parser.add_argument("--n-examples", type=int, default=50, help="Number of examples to generate")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices (default: auto)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Commitment confidence threshold")
    parser.add_argument("--n-shuffle", type=int, default=50, help="Shuffled baseline iterations")
    parser.add_argument("--include-logits", action="store_true", help="Also cache per-position logits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="results/lookahead/phase1", help="Output directory")
    
    args = parser.parse_args()
    main(args)
