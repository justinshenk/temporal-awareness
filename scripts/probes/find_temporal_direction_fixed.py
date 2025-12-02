#!/usr/bin/env python3
"""
Find the Single Linear Direction for Temporal Scope - FIXED VERSION

Fixes from critical review:
1. Proper activation extraction (full forward pass)
2. Train/test split on same distribution
3. Absolute accuracy metrics (not percentage of above-chance)
4. Statistical significance testing (permutation tests)
5. Random direction baseline
6. Token position analysis (final vs punctuation)
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def load_datasets():
    """Load explicit and implicit datasets."""
    with open('research/datasets/temporal_scope_caa.json') as f:
        explicit_data = json.load(f)
    explicit_pairs = explicit_data['pairs'] if 'pairs' in explicit_data else explicit_data

    with open('research/datasets/temporal_scope_implicit.json') as f:
        implicit_data = json.load(f)
    implicit_pairs = implicit_data['pairs']

    return explicit_pairs, implicit_pairs


def extract_activation_proper(model, tokenizer, text, layer, position='final'):
    """
    FIXED: Extract activation using full forward pass.

    Args:
        position: 'final' or 'punctuation' (for summarization motif analysis)
    """
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(inputs['input_ids'], output_hidden_states=True)

    # outputs.hidden_states[layer] is [batch, seq_len, hidden_dim]
    hidden_state = outputs.hidden_states[layer][0]  # [seq_len, hidden_dim]

    if position == 'final':
        activation = hidden_state[-1, :].cpu().numpy()
    elif position == 'punctuation':
        # Find last punctuation token
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        punct_indices = [i for i, t in enumerate(tokens) if t in [',', '.', ':', ';', '?', '!']]
        if punct_indices:
            activation = hidden_state[punct_indices[-1], :].cpu().numpy()
        else:
            activation = hidden_state[-1, :].cpu().numpy()  # fallback to final
    else:
        raise ValueError(f"Unknown position: {position}")

    return activation


def find_temporal_direction_mean_diff(model, tokenizer, pairs, layer, position='final'):
    """
    FIXED: Extract temporal direction using proper forward pass.
    """
    immediate_acts = []
    longterm_acts = []

    for pair in tqdm(pairs, desc=f"Layer {layer}, pos={position}"):
        question = pair['question']
        option_keys = [k for k in pair.keys() if k not in ['question', 'category']]
        if len(option_keys) < 2:
            continue

        immediate_text = question + pair[option_keys[0]]
        longterm_text = question + pair[option_keys[1]]

        immediate_acts.append(extract_activation_proper(model, tokenizer, immediate_text, layer, position))
        longterm_acts.append(extract_activation_proper(model, tokenizer, longterm_text, layer, position))

    immediate_mean = np.mean(immediate_acts, axis=0)
    longterm_mean = np.mean(longterm_acts, axis=0)

    direction = longterm_mean - immediate_mean
    direction = direction / np.linalg.norm(direction)

    return direction, np.array(immediate_acts), np.array(longterm_acts)


def mass_mean_ablation(activations, direction):
    """Remove the component along direction from activations."""
    projections = np.dot(activations, direction)
    ablated = activations - np.outer(projections, direction)
    return ablated


def test_ablation_effect(model, tokenizer, probe, direction, test_pairs, layer, position='final'):
    """
    FIXED: Test ablation with proper metrics.

    Returns:
        baseline_acc: Accuracy without ablation
        ablated_acc: Accuracy with ablation
        accuracy_drop_abs: Absolute accuracy drop (percentage points)
        p_value: Significance via permutation test
    """
    X_baseline = []
    X_ablated = []
    y_true = []

    for pair in tqdm(test_pairs, desc=f"Testing ablation (Layer {layer})"):
        question = pair['question']
        option_keys = [k for k in pair.keys() if k not in ['question', 'category']]

        for i, key in enumerate(option_keys[:2]):
            text = question + pair[key]
            activation = extract_activation_proper(model, tokenizer, text, layer, position)

            X_baseline.append(activation)
            X_ablated.append(mass_mean_ablation([activation], direction)[0])
            y_true.append(i)

    X_baseline = np.array(X_baseline)
    X_ablated = np.array(X_ablated)
    y_true = np.array(y_true)

    baseline_acc = probe.score(X_baseline, y_true)
    ablated_acc = probe.score(X_ablated, y_true)
    accuracy_drop_abs = baseline_acc - ablated_acc

    # Permutation test for significance
    p_value = permutation_test(probe, X_baseline, y_true, accuracy_drop_abs, n_permutations=1000)

    return baseline_acc, ablated_acc, accuracy_drop_abs, p_value


def permutation_test(probe, X, y, observed_drop, n_permutations=1000):
    """
    Test if ablation effect is significant via permutation test.

    Null hypothesis: Ablation has no effect (drop = 0)
    """
    null_drops = []

    for _ in range(n_permutations):
        # Random direction
        random_dir = np.random.randn(X.shape[1])
        random_dir = random_dir / np.linalg.norm(random_dir)

        # Ablate with random direction
        X_random_ablated = mass_mean_ablation(X, random_dir)

        # Measure drop
        baseline_acc = probe.score(X, y)
        random_ablated_acc = probe.score(X_random_ablated, y)
        null_drops.append(baseline_acc - random_ablated_acc)

    # p-value: proportion of null drops >= observed drop
    p_value = np.mean(np.array(null_drops) >= observed_drop)

    return p_value


def random_direction_baseline(model, tokenizer, probe, test_pairs, layer, position='final', n_random=100):
    """
    FIXED: Test ablation with random directions as baseline.

    Returns mean and std of accuracy drops for random directions.
    """
    random_drops = []

    # Extract test activations once
    X = []
    y = []
    for pair in test_pairs:
        question = pair['question']
        option_keys = [k for k in pair.keys() if k not in ['question', 'category']]
        for i, key in enumerate(option_keys[:2]):
            text = question + pair[key]
            X.append(extract_activation_proper(model, tokenizer, text, layer, position))
            y.append(i)

    X = np.array(X)
    y = np.array(y)
    baseline_acc = probe.score(X, y)

    # Test random directions
    for _ in tqdm(range(n_random), desc="Random baseline"):
        random_dir = np.random.randn(X.shape[1])
        random_dir = random_dir / np.linalg.norm(random_dir)

        X_ablated = mass_mean_ablation(X, random_dir)
        ablated_acc = probe.score(X_ablated, y)
        random_drops.append(baseline_acc - ablated_acc)

    return np.mean(random_drops), np.std(random_drops), random_drops


def analyze_direction_properties(direction, immediate_acts, longterm_acts):
    """Analyze properties of the temporal direction."""
    immediate_proj = np.dot(immediate_acts, direction)
    longterm_proj = np.dot(longterm_acts, direction)

    # Cohen's d
    pooled_std = np.sqrt((immediate_proj.var() + longterm_proj.var()) / 2)
    cohen_d = (longterm_proj.mean() - immediate_proj.mean()) / pooled_std

    # Mann-Whitney U test for significance
    u_stat, u_p = stats.mannwhitneyu(immediate_proj, longterm_proj, alternative='two-sided')

    stats_dict = {
        'immediate_mean': float(immediate_proj.mean()),
        'immediate_std': float(immediate_proj.std()),
        'longterm_mean': float(longterm_proj.mean()),
        'longterm_std': float(longterm_proj.std()),
        'separation': float(longterm_proj.mean() - immediate_proj.mean()),
        'cohen_d': float(cohen_d),
        'mann_whitney_u': float(u_stat),
        'mann_whitney_p': float(u_p)
    }

    return stats_dict, immediate_proj, longterm_proj


def visualize_results(stats, immediate_proj, longterm_proj, layer, position,
                      baseline_acc, ablated_acc, accuracy_drop, p_value,
                      random_mean, random_std, output_file):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of projections
    axes[0, 0].hist(immediate_proj, bins=30, alpha=0.7, label='Immediate', color='red')
    axes[0, 0].hist(longterm_proj, bins=30, alpha=0.7, label='Long-term', color='blue')
    axes[0, 0].axvline(stats['immediate_mean'], color='red', linestyle='--', linewidth=2)
    axes[0, 0].axvline(stats['longterm_mean'], color='blue', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Projection onto temporal direction')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Layer {layer} ({position}): Projections\nCohen\'s d={stats["cohen_d"]:.3f}, p={stats["mann_whitney_p"]:.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Box plot
    axes[0, 1].boxplot([immediate_proj, longterm_proj], tick_labels=['Immediate', 'Long-term'])
    axes[0, 1].set_ylabel('Projection')
    axes[0, 1].set_title(f'Separation: {stats["separation"]:.3f}')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Ablation effect
    categories = ['Baseline', 'Ablated']
    accuracies = [baseline_acc * 100, ablated_acc * 100]
    colors = ['green', 'red']
    bars = axes[1, 0].bar(categories, accuracies, color=colors, alpha=0.7)
    axes[1, 0].axhline(50, color='black', linestyle='--', label='Chance')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title(f'Ablation Effect\nDrop: {accuracy_drop*100:.1f} pp, p={p_value:.4f}')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.1f}%', ha='center', va='bottom')

    # 4. Comparison to random baseline
    axes[1, 1].axhline(accuracy_drop * 100, color='red', linewidth=2,
                      label=f'Temporal direction: {accuracy_drop*100:.1f} pp')
    axes[1, 1].axhline(random_mean * 100, color='gray', linestyle='--', linewidth=2,
                      label=f'Random mean: {random_mean*100:.1f} pp')
    axes[1, 1].fill_between([0, 1],
                           (random_mean - 2*random_std) * 100,
                           (random_mean + 2*random_std) * 100,
                           color='gray', alpha=0.3, label='Random ±2σ')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([min(-5, (random_mean - 3*random_std) * 100),
                         max(15, accuracy_drop * 100 * 1.2)])
    axes[1, 1].set_ylabel('Accuracy drop (pp)')
    axes[1, 1].set_title('Temporal vs Random Directions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_xticks([])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def test_all_layers(model, tokenizer, train_pairs, test_pairs, probes, positions=['final']):
    """Test all layers with train/test split and multiple token positions."""
    results = []

    Path('research/results/temporal_directions_fixed').mkdir(parents=True, exist_ok=True)

    for layer in tqdm(range(len(model.transformer.h)), desc="Testing layers"):
        if layer not in probes:
            continue

        for position in positions:
            print(f"\n{'='*70}")
            print(f"LAYER {layer} - POSITION: {position}")
            print(f"{'='*70}\n")

            # Extract direction from training set
            direction, immediate_acts, longterm_acts = find_temporal_direction_mean_diff(
                model, tokenizer, train_pairs, layer, position
            )

            # Save direction
            direction_file = f'research/results/temporal_directions_fixed/layer_{layer}_{position}_direction.npy'
            np.save(direction_file, direction)

            # Analyze properties
            stats_dict, immediate_proj, longterm_proj = analyze_direction_properties(
                direction, immediate_acts, longterm_acts
            )

            print(f"Direction statistics:")
            print(f"  Separation: {stats_dict['separation']:.3f}")
            print(f"  Cohen's d: {stats_dict['cohen_d']:.3f}")
            print(f"  Mann-Whitney p: {stats_dict['mann_whitney_p']:.4f}")

            # Test ablation on held-out test set
            probe = probes[layer]
            baseline_acc, ablated_acc, accuracy_drop, p_value = test_ablation_effect(
                model, tokenizer, probe, direction, test_pairs, layer, position
            )

            print(f"\nAblation results:")
            print(f"  Baseline accuracy: {baseline_acc:.1%}")
            print(f"  Ablated accuracy: {ablated_acc:.1%}")
            print(f"  Accuracy drop: {accuracy_drop*100:.1f} pp")
            print(f"  Significance (permutation): p={p_value:.4f}")

            # Random direction baseline
            random_mean, random_std, random_drops = random_direction_baseline(
                model, tokenizer, probe, test_pairs, layer, position, n_random=100
            )

            print(f"\nRandom baseline:")
            print(f"  Mean drop: {random_mean*100:.1f} pp")
            print(f"  Std: {random_std*100:.1f} pp")
            print(f"  Temporal vs Random: {(accuracy_drop - random_mean) / random_std:.2f} σ")

            # Visualize
            viz_file = f'research/results/temporal_directions_fixed/layer_{layer}_{position}_viz.png'
            visualize_results(stats_dict, immediate_proj, longterm_proj, layer, position,
                            baseline_acc, ablated_acc, accuracy_drop, p_value,
                            random_mean, random_std, viz_file)
            print(f"Saved visualization to {viz_file}")

            results.append({
                'layer': layer,
                'position': position,
                'baseline_acc': float(baseline_acc),
                'ablated_acc': float(ablated_acc),
                'accuracy_drop_pp': float(accuracy_drop * 100),
                'p_value': float(p_value),
                'random_mean_pp': float(random_mean * 100),
                'random_std_pp': float(random_std * 100),
                'z_score': float((accuracy_drop - random_mean) / random_std) if random_std > 0 else 0,
                'separation': stats_dict['separation'],
                'cohen_d': stats_dict['cohen_d'],
                'mann_whitney_p': stats_dict['mann_whitney_p']
            })

    return results


def summarize_results(results):
    """Print comprehensive summary."""
    print("\n" + "="*70)
    print("SUMMARY: TEMPORAL DIRECTION ANALYSIS (FIXED)")
    print("="*70)
    print()

    print(f"{'Layer':<7} {'Pos':<8} {'Base%':<8} {'Abl%':<8} {'Drop(pp)':<10} {'p-val':<8} {'vs Rand':<10} {'Cohen d':<10}")
    print("-"*95)

    for r in results:
        significant = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
        print(f"{r['layer']:<7} {r['position']:<8} {r['baseline_acc']:<8.1%} {r['ablated_acc']:<8.1%} "
              f"{r['accuracy_drop_pp']:<10.1f} {r['p_value']:<8.4f}{significant:<3} "
              f"{r['z_score']:<10.2f} {r['cohen_d']:<10.3f}")

    print()

    # Best layer
    significant_results = [r for r in results if r['p_value'] < 0.05]
    if significant_results:
        best = max(significant_results, key=lambda x: x['accuracy_drop_pp'])
        print(f"Best layer: {best['layer']} ({best['position']})")
        print(f"  Accuracy drop: {best['accuracy_drop_pp']:.1f} pp")
        print(f"  Significance: p={best['p_value']:.4f}")
        print(f"  vs Random: {best['z_score']:.2f} σ above random baseline")
        print(f"  Cohen's d: {best['cohen_d']:.3f}")
    else:
        print("No significant results found (all p >= 0.05)")

    print()
    print("Comparison to Tigges et al. (2024):")
    print("  Their result: 76% loss of above-chance accuracy")
    if significant_results:
        print(f"  Our result: {best['accuracy_drop_pp']:.1f} pp absolute drop")


def main():
    print("="*70)
    print("TEMPORAL DIRECTION ANALYSIS - FIXED VERSION")
    print("="*70)
    print()
    print("Improvements:")
    print("  1. Proper activation extraction (full forward pass)")
    print("  2. Train/test split (80/20)")
    print("  3. Absolute accuracy metrics (percentage points)")
    print("  4. Statistical significance (permutation tests)")
    print("  5. Random direction baseline")
    print("  6. Token position analysis (final + punctuation)")
    print()

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print("Loading datasets...")
    explicit_pairs, implicit_pairs = load_datasets()

    # Train/test split on explicit data (same distribution)
    np.random.seed(42)
    indices = np.random.permutation(len(explicit_pairs))
    train_size = int(0.8 * len(explicit_pairs))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_pairs = [explicit_pairs[i] for i in train_indices]
    test_pairs = [explicit_pairs[i] for i in test_indices]

    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Test: {len(test_pairs)} pairs")

    # Load probes
    print("Loading probes...")
    probes = {}
    probe_files = Path('research/probes').glob('temporal_caa_layer_*_probe.pkl')
    for probe_file in probe_files:
        layer = int(probe_file.stem.split('_')[-2])
        with open(probe_file, 'rb') as f:
            probes[layer] = pickle.load(f)
    print(f"  Loaded {len(probes)} probes")

    # Test all layers with multiple token positions
    results = test_all_layers(model, tokenizer, train_pairs, test_pairs, probes,
                              positions=['final', 'punctuation'])

    # Summarize
    summarize_results(results)

    # Save results
    results_file = 'research/results/temporal_directions_fixed/ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")


if __name__ == "__main__":
    main()
