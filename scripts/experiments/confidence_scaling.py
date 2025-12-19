#!/usr/bin/env python3
"""
Experiment: Temporal horizon vs output confidence scaling

Measures whether LLM output confidence decreases with temporal distance.

Method:
1. Ask questions at varying time horizons
2. Generate with temperature=0 (greedy decoding)
3. Extract token logits, compute confidence per token
4. Plot: X = time horizon, Y = aggregate confidence

Idea by Nicola Sabbadini.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


# Time horizons to test (roughly log-spaced)
HORIZONS = [
    ("next hour", 1/24),           # ~0.04 days
    ("tomorrow", 1),               # 1 day
    ("next week", 7),              # 7 days
    ("next month", 30),            # 30 days
    ("next year", 365),            # 365 days
    ("in 10 years", 3650),         # 3650 days
    ("in 100 years", 36500),       # 36500 days
]

# Question templates
QUESTION_TEMPLATES = [
    "What will the weather be like {horizon}?",
    "What will I be doing {horizon}?",
    "What will happen in the world {horizon}?",
    "What technology will exist {horizon}?",
    "What will be different about life {horizon}?",
]


def generate_with_confidence(model, tokenizer, prompt, max_new_tokens=50):
    """
    Generate text with temperature=0 and extract token-level confidence.

    Returns:
        dict with:
            - text: generated text
            - token_confidences: list of confidence scores per token
            - mean_confidence: average confidence
            - tokens: list of generated tokens
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    input_length = inputs['input_ids'].shape[1]

    # Generate with output scores
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy (temperature=0)
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract generated tokens (excluding prompt)
    generated_ids = outputs.sequences[0, input_length:]

    # Calculate confidence for each generated token
    confidences = []
    tokens = []

    for i, (score, token_id) in enumerate(zip(outputs.scores, generated_ids)):
        # score shape: (1, vocab_size)
        probs = F.softmax(score[0], dim=-1)
        confidence = probs[token_id].item()
        confidences.append(confidence)
        tokens.append(tokenizer.decode([token_id]))

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        'text': generated_text,
        'token_confidences': confidences,
        'mean_confidence': np.mean(confidences) if confidences else 0,
        'min_confidence': np.min(confidences) if confidences else 0,
        'max_confidence': np.max(confidences) if confidences else 0,
        'std_confidence': np.std(confidences) if confidences else 0,
        'tokens': tokens,
        'n_tokens': len(tokens),
    }


def run_experiment(model, tokenizer, output_dir='results/confidence_scaling'):
    """Run the full confidence scaling experiment."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    print("="*70)
    print("CONFIDENCE vs TEMPORAL HORIZON EXPERIMENT")
    print("="*70)
    print()

    for template in tqdm(QUESTION_TEMPLATES, desc="Question templates"):
        print(f"\nTemplate: {template}")
        print("-"*70)

        for horizon_name, horizon_days in HORIZONS:
            prompt = template.format(horizon=horizon_name)

            result = generate_with_confidence(model, tokenizer, prompt)

            results.append({
                'template': template,
                'horizon_name': horizon_name,
                'horizon_days': horizon_days,
                'prompt': prompt,
                'response': result['text'],
                'mean_confidence': result['mean_confidence'],
                'min_confidence': result['min_confidence'],
                'max_confidence': result['max_confidence'],
                'std_confidence': result['std_confidence'],
                'n_tokens': result['n_tokens'],
                'token_confidences': result['token_confidences'],
            })

            print(f"  {horizon_name:15s} | conf: {result['mean_confidence']:.3f} "
                  f"(±{result['std_confidence']:.3f}) | {result['n_tokens']} tokens")

    return results


def analyze_results(results, output_dir='results/confidence_scaling'):
    """Analyze and visualize results."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Aggregate by horizon
    horizon_stats = {}
    for r in results:
        h = r['horizon_name']
        if h not in horizon_stats:
            horizon_stats[h] = {
                'days': r['horizon_days'],
                'confidences': [],
            }
        horizon_stats[h]['confidences'].append(r['mean_confidence'])

    # Calculate means and stds
    horizons_ordered = [h[0] for h in HORIZONS]
    days = [horizon_stats[h]['days'] for h in horizons_ordered]
    means = [np.mean(horizon_stats[h]['confidences']) for h in horizons_ordered]
    stds = [np.std(horizon_stats[h]['confidences']) for h in horizons_ordered]

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print()
    print(f"{'Horizon':<15} {'Days':>10} {'Mean Conf':>12} {'Std':>10}")
    print("-"*50)
    for h, d, m, s in zip(horizons_ordered, days, means, stds):
        print(f"{h:<15} {d:>10.0f} {m:>12.4f} {s:>10.4f}")

    # Statistical analysis
    from scipy import stats

    # Spearman correlation (rank-based, good for monotonic relationships)
    all_days = [r['horizon_days'] for r in results]
    all_confs = [r['mean_confidence'] for r in results]

    spearman_r, spearman_p = stats.spearmanr(all_days, all_confs)
    pearson_r, pearson_p = stats.pearsonr(np.log10(all_days), all_confs)

    print()
    print("="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    print(f"Spearman correlation (days vs confidence): r={spearman_r:.3f}, p={spearman_p:.4f}")
    print(f"Pearson correlation (log(days) vs confidence): r={pearson_r:.3f}, p={pearson_p:.4f}")

    if spearman_p < 0.05:
        if spearman_r < 0:
            print("\n✓ SIGNIFICANT NEGATIVE CORRELATION")
            print("  Confidence decreases with temporal distance!")
        else:
            print("\n⚠ SIGNIFICANT POSITIVE CORRELATION")
            print("  Confidence increases with temporal distance (unexpected)")
    else:
        print("\n○ NO SIGNIFICANT CORRELATION")
        print("  Model confidence does not scale with temporal horizon")

    # Plot 1: Confidence vs Horizon (log scale)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.errorbar(days, means, yerr=stds, fmt='o-', capsize=5, capthick=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_xlabel('Time Horizon (days)', fontsize=12)
    ax1.set_ylabel('Mean Output Confidence', fontsize=12)
    ax1.set_title('Output Confidence vs Temporal Horizon', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add horizon labels
    for d, m, h in zip(days, means, horizons_ordered):
        ax1.annotate(h, (d, m), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8)

    # Plot 2: All data points (scatter)
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(QUESTION_TEMPLATES)))

    for i, template in enumerate(QUESTION_TEMPLATES):
        template_results = [r for r in results if r['template'] == template]
        x = [r['horizon_days'] for r in template_results]
        y = [r['mean_confidence'] for r in template_results]
        label = template[:30] + '...' if len(template) > 30 else template
        ax2.scatter(x, y, c=[colors[i]], label=label, s=60, alpha=0.7)

    ax2.set_xscale('log')
    ax2.set_xlabel('Time Horizon (days)', fontsize=12)
    ax2.set_ylabel('Mean Output Confidence', fontsize=12)
    ax2.set_title('Confidence by Question Type', fontsize=14)
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = Path(output_dir) / f'confidence_vs_horizon_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {plot_path}")

    # Save results as JSON
    results_path = Path(output_dir) / f'confidence_scaling_results_{timestamp}.json'

    # Remove non-serializable items
    results_for_json = []
    for r in results:
        r_copy = r.copy()
        r_copy['token_confidences'] = [float(c) for c in r['token_confidences']]
        results_for_json.append(r_copy)

    with open(results_path, 'w') as f:
        json.dump({
            'results': results_for_json,
            'summary': {
                'horizons': horizons_ordered,
                'days': days,
                'means': means,
                'stds': stds,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
            }
        }, f, indent=2)

    print(f"✓ Results saved to {results_path}")

    return {
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'means': means,
        'stds': stds,
    }


def main():
    print("="*70)
    print("TEMPORAL HORIZON vs OUTPUT CONFIDENCE")
    print("="*70)
    print()
    print("Hypothesis: Models are less confident about distant futures")
    print()

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    print("✓ Model loaded")
    print()

    # Run experiment
    results = run_experiment(model, tokenizer)

    # Analyze
    stats = analyze_results(results)

    print()
    print("="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
