#!/usr/bin/env python3
"""
Adversarial Probe Training: Using Edge Cases to Improve Robustness

Research Questions:
1. Can probes trained on adversarial examples (negations, paraphrases, implicit markers)
   better detect temporal scale in edge cases?
2. Do adversarial-trained probes maintain performance on standard cases?
3. What's the optimal mix of standard vs adversarial training data?
4. Can we extract more robust steering vectors from adversarial-trained probes?

Approach:
- Create augmented dataset with adversarial examples
- Train probes on: (a) standard only, (b) adversarial only, (c) mixed
- Compare performance on held-out standard and adversarial test sets
- Extract new CAA vectors from adversarial-aware activations
- Test if adversarial steering vectors are more robust

Expected Benefits:
- Probes learn to handle negations ("not short-term" → long-term)
- Probes rely less on lexical markers, more on semantic meaning
- Steering vectors become robust to paraphrasing
- Better generalization to creative temporal expressions
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Temporal scale mapping (for labeling)
TEMPORAL_SCALES = {
    'minutes': 0,     # Immediate class
    'hours': 0,
    'days': 0,
    'weeks': 0,
    'months': 1,      # Long-term class (transition)
    'quarters': 1,
    'years': 1,
    'decades': 1,
    'generations': 1,
    'centuries': 1,
}


def create_adversarial_training_dataset():
    """
    Create comprehensive training dataset with adversarial examples.

    Categories:
    1. Standard: Original CAA-format prompts
    2. Negations: "Not short-term" (label: long), "Not long-term" (label: short)
    3. Paraphrases: Multiple phrasings of same temporal concept
    4. Implicit: No explicit time words, only semantic markers
    5. Contradictions: For filtering/uncertainty detection
    """

    dataset = []

    # 1. NEGATIONS - teach probes to handle "not X"
    negation_pairs = [
        # Not immediate → long-term
        {
            'prompt': "This is not an immediate priority, but rather a",
            'scale': 'years',
            'label': 1,
            'type': 'negation',
            'marker': '¬immediate → long'
        },
        {
            'prompt': "We should not focus on short-term gains here; instead think about",
            'scale': 'decades',
            'label': 1,
            'type': 'negation',
            'marker': '¬short → long'
        },
        {
            'prompt': "Not a quick fix, but a",
            'scale': 'years',
            'label': 1,
            'type': 'negation',
            'marker': '¬quick → long'
        },
        {
            'prompt': "This isn't something for the next few weeks; we're planning for",
            'scale': 'decades',
            'label': 1,
            'type': 'negation',
            'marker': '¬weeks → long'
        },

        # Not long-term → immediate
        {
            'prompt': "This is not a long-term strategy; we need action",
            'scale': 'hours',
            'label': 0,
            'type': 'negation',
            'marker': '¬long → immediate'
        },
        {
            'prompt': "Don't think decades ahead here; focus on what we can do",
            'scale': 'days',
            'label': 0,
            'type': 'negation',
            'marker': '¬decades → immediate'
        },
        {
            'prompt': "Not for future generations, but for",
            'scale': 'hours',
            'label': 0,
            'type': 'negation',
            'marker': '¬generations → immediate'
        },
        {
            'prompt': "We shouldn't be strategic here; we need tactical",
            'scale': 'days',
            'label': 0,
            'type': 'negation',
            'marker': '¬strategic → immediate'
        },
    ]

    # 2. PARAPHRASES - diverse expressions of same temporal concept
    paraphrase_groups = [
        # Immediate (minutes/hours)
        {
            'scale': 'hours',
            'label': 0,
            'type': 'paraphrase',
            'variants': [
                "What we can do right now this instant",
                "Immediate action in the very near term",
                "Before the day is over",
                "Within the hour",
                "As soon as possible",
                "Without delay",
                "Urgently and immediately",
            ]
        },

        # Short-term (days/weeks)
        {
            'scale': 'days',
            'label': 0,
            'type': 'paraphrase',
            'variants': [
                "Over the next several days",
                "By the end of this week",
                "In the coming few days",
                "Before next week arrives",
                "Within a week's time",
                "This week's priorities",
            ]
        },

        # Medium-term (months/quarters)
        {
            'scale': 'quarters',
            'label': 1,
            'type': 'paraphrase',
            'variants': [
                "Over the coming months",
                "By year-end",
                "Through the next few quarters",
                "This fiscal year",
                "In the next 6-9 months",
                "Before the year concludes",
            ]
        },

        # Long-term (years/decades)
        {
            'scale': 'years',
            'label': 1,
            'type': 'paraphrase',
            'variants': [
                "Over the next several years",
                "In the 2030s",
                "By the end of this decade",
                "Over a multi-year horizon",
                "Through the coming years",
                "In the medium to long run",
            ]
        },

        # Very long-term (generations/centuries)
        {
            'scale': 'generations',
            'label': 1,
            'type': 'paraphrase',
            'variants': [
                "For our children and grandchildren",
                "For posterity",
                "For the ages",
                "For those who come after us",
                "For the long-term future of humanity",
                "For generations yet unborn",
                "For the next chapter of human history",
            ]
        },
    ]

    # 3. IMPLICIT MARKERS - no explicit time words
    implicit_examples = [
        # Immediate implicit markers
        {
            'prompt': "The urgent crisis demands our attention",
            'scale': 'hours',
            'label': 0,
            'type': 'implicit',
            'marker': 'urgent/crisis → immediate'
        },
        {
            'prompt': "We need a rapid response with quick wins",
            'scale': 'days',
            'label': 0,
            'type': 'implicit',
            'marker': 'rapid/quick → immediate'
        },
        {
            'prompt': "This is a tactical, operational matter requiring",
            'scale': 'days',
            'label': 0,
            'type': 'implicit',
            'marker': 'tactical/operational → immediate'
        },
        {
            'prompt': "An emergency intervention is required for",
            'scale': 'hours',
            'label': 0,
            'type': 'implicit',
            'marker': 'emergency → immediate'
        },
        {
            'prompt': "Time-sensitive and pressing considerations demand",
            'scale': 'hours',
            'label': 0,
            'type': 'implicit',
            'marker': 'time-sensitive/pressing → immediate'
        },

        # Long-term implicit markers
        {
            'prompt': "Our sustainable and enduring approach focuses on",
            'scale': 'decades',
            'label': 1,
            'type': 'implicit',
            'marker': 'sustainable/enduring → long-term'
        },
        {
            'prompt': "The strategic vision for transformational change requires",
            'scale': 'years',
            'label': 1,
            'type': 'implicit',
            'marker': 'strategic/transformational → long-term'
        },
        {
            'prompt': "Building a lasting legacy means",
            'scale': 'generations',
            'label': 1,
            'type': 'implicit',
            'marker': 'legacy → long-term'
        },
        {
            'prompt': "Our foundational and fundamental work on",
            'scale': 'decades',
            'label': 1,
            'type': 'implicit',
            'marker': 'foundational/fundamental → long-term'
        },
        {
            'prompt': "The visionary and pioneering efforts toward",
            'scale': 'generations',
            'label': 1,
            'type': 'implicit',
            'marker': 'visionary/pioneering → long-term'
        },
    ]

    # 4. LEXICAL VARIATIONS - unusual/creative temporal expressions
    lexical_variations = [
        # Immediate
        {
            'prompt': "Before the sun sets today, we must",
            'scale': 'hours',
            'label': 0,
            'type': 'lexical_variation'
        },
        {
            'prompt': "Between now and your next coffee break",
            'scale': 'minutes',
            'label': 0,
            'type': 'lexical_variation'
        },
        {
            'prompt': "Before you finish reading this sentence",
            'scale': 'minutes',
            'label': 0,
            'type': 'lexical_variation'
        },

        # Long-term
        {
            'prompt': "When our great-great-grandchildren look back",
            'scale': 'centuries',
            'label': 1,
            'type': 'lexical_variation'
        },
        {
            'prompt': "In the history books of the future",
            'scale': 'generations',
            'label': 1,
            'type': 'lexical_variation'
        },
        {
            'prompt': "Through the lifetimes of those yet to be born",
            'scale': 'generations',
            'label': 1,
            'type': 'lexical_variation'
        },
    ]

    # Compile dataset
    dataset.extend(negation_pairs)

    for group in paraphrase_groups:
        for variant in group['variants']:
            dataset.append({
                'prompt': variant,
                'scale': group['scale'],
                'label': group['label'],
                'type': group['type']
            })

    dataset.extend(implicit_examples)
    dataset.extend(lexical_variations)

    return dataset


def load_standard_caa_dataset(dataset_path='research/datasets/temporal_scope_caa.json'):
    """Load original CAA dataset for comparison."""
    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data['pairs'] if 'pairs' in data else data

    # Convert to same format as adversarial dataset
    standard_dataset = []

    for pair in pairs:
        question = pair['question']
        option_keys = [k for k in pair.keys() if k not in ['question', 'category']]

        if len(option_keys) != 2:
            continue

        # Assume first is immediate, second is long-term
        immediate_key = option_keys[0]
        long_term_key = option_keys[1]

        # Immediate example
        standard_dataset.append({
            'prompt': question + "\n\nChoices:\n" + pair[immediate_key],
            'scale': 'immediate',
            'label': 0,
            'type': 'standard_caa'
        })

        # Long-term example
        standard_dataset.append({
            'prompt': question + "\n\nChoices:\n" + pair[long_term_key],
            'scale': 'long_term',
            'label': 1,
            'type': 'standard_caa'
        })

    return standard_dataset


def extract_activations(model, tokenizer, prompt):
    """Extract activations from all layers."""
    inputs = tokenizer(prompt, return_tensors='pt')

    activations = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            activations[layer_idx] = output[0][0, -1, :].detach().cpu().numpy()
        return hook

    hooks = []
    for i, layer in enumerate(model.transformer.h):
        hook = layer.register_forward_hook(hook_fn(i))
        hooks.append(hook)

    with torch.no_grad():
        model(**inputs)

    for hook in hooks:
        hook.remove()

    return activations


def create_training_datasets(model, tokenizer, standard_data, adversarial_data):
    """
    Extract activations for all training data.

    Returns:
        standard_X, standard_y: Standard CAA data
        adversarial_X, adversarial_y: Adversarial data
        mixed_X, mixed_y: Combined data
    """

    print("Extracting activations for training datasets...")
    print(f"  Standard examples: {len(standard_data)}")
    print(f"  Adversarial examples: {len(adversarial_data)}")
    print()

    n_layers = len(model.transformer.h)

    # Standard dataset
    print("Processing standard CAA dataset...")
    standard_X = {i: [] for i in range(n_layers)}
    standard_y = []

    for item in tqdm(standard_data, desc="Standard"):
        activations = extract_activations(model, tokenizer, item['prompt'])
        for layer in range(n_layers):
            standard_X[layer].append(activations[layer])
        standard_y.append(item['label'])

    for layer in range(n_layers):
        standard_X[layer] = np.array(standard_X[layer])
    standard_y = np.array(standard_y)

    # Adversarial dataset
    print("\nProcessing adversarial dataset...")
    adversarial_X = {i: [] for i in range(n_layers)}
    adversarial_y = []

    for item in tqdm(adversarial_data, desc="Adversarial"):
        activations = extract_activations(model, tokenizer, item['prompt'])
        for layer in range(n_layers):
            adversarial_X[layer].append(activations[layer])
        adversarial_y.append(item['label'])

    for layer in range(n_layers):
        adversarial_X[layer] = np.array(adversarial_X[layer])
    adversarial_y = np.array(adversarial_y)

    # Mixed dataset
    mixed_X = {}
    for layer in range(n_layers):
        mixed_X[layer] = np.vstack([standard_X[layer], adversarial_X[layer]])
    mixed_y = np.concatenate([standard_y, adversarial_y])

    print(f"\n✓ Dataset sizes:")
    print(f"  Standard: {len(standard_y)} samples")
    print(f"  Adversarial: {len(adversarial_y)} samples")
    print(f"  Mixed: {len(mixed_y)} samples")
    print()

    return (standard_X, standard_y), (adversarial_X, adversarial_y), (mixed_X, mixed_y)


def train_probe_variants(standard_data, adversarial_data, mixed_data, output_dir='research/probes'):
    """
    Train three variants of probes:
    1. Standard-only (baseline)
    2. Adversarial-only (edge case specialist)
    3. Mixed (robust generalist)

    Compare performance on held-out standard and adversarial test sets.
    """

    print("="*70)
    print("TRAINING PROBE VARIANTS")
    print("="*70)
    print()

    standard_X, standard_y = standard_data
    adversarial_X, adversarial_y = adversarial_data
    mixed_X, mixed_y = mixed_data

    n_layers = len(standard_X)

    # Split datasets
    print("Splitting train/test sets...")

    # Standard split (80/20)
    standard_X_train = {}
    standard_X_test = {}
    for layer in range(n_layers):
        X_train, X_test, y_train, y_test = train_test_split(
            standard_X[layer], standard_y, test_size=0.2, random_state=42, stratify=standard_y
        )
        standard_X_train[layer] = X_train
        standard_X_test[layer] = X_test
    _, _, standard_y_train, standard_y_test = train_test_split(
        standard_X[0], standard_y, test_size=0.2, random_state=42, stratify=standard_y
    )

    # Adversarial split (80/20)
    adversarial_X_train = {}
    adversarial_X_test = {}
    for layer in range(n_layers):
        X_train, X_test, y_train, y_test = train_test_split(
            adversarial_X[layer], adversarial_y, test_size=0.2, random_state=42, stratify=adversarial_y
        )
        adversarial_X_train[layer] = X_train
        adversarial_X_test[layer] = X_test
    _, _, adversarial_y_train, adversarial_y_test = train_test_split(
        adversarial_X[0], adversarial_y, test_size=0.2, random_state=42, stratify=adversarial_y
    )

    # Mixed split (80/20)
    mixed_X_train = {}
    mixed_X_test = {}
    for layer in range(n_layers):
        X_train, X_test, y_train, y_test = train_test_split(
            mixed_X[layer], mixed_y, test_size=0.2, random_state=42, stratify=mixed_y
        )
        mixed_X_train[layer] = X_train
        mixed_X_test[layer] = X_test
    _, _, mixed_y_train, mixed_y_test = train_test_split(
        mixed_X[0], mixed_y, test_size=0.2, random_state=42, stratify=mixed_y
    )

    print(f"  Standard train: {len(standard_y_train)}, test: {len(standard_y_test)}")
    print(f"  Adversarial train: {len(adversarial_y_train)}, test: {len(adversarial_y_test)}")
    print(f"  Mixed train: {len(mixed_y_train)}, test: {len(mixed_y_test)}")
    print()

    # Train probes for each variant
    results = []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for layer in range(n_layers):
        print(f"Layer {layer}/{n_layers-1}")
        print("-"*70)

        # 1. Standard-only probe
        probe_standard = LogisticRegression(max_iter=1000, random_state=42)
        probe_standard.fit(standard_X_train[layer], standard_y_train)

        # 2. Adversarial-only probe
        probe_adversarial = LogisticRegression(max_iter=1000, random_state=42)
        probe_adversarial.fit(adversarial_X_train[layer], adversarial_y_train)

        # 3. Mixed probe
        probe_mixed = LogisticRegression(max_iter=1000, random_state=42)
        probe_mixed.fit(mixed_X_train[layer], mixed_y_train)

        # Evaluate on BOTH test sets
        # Standard probe
        std_on_std = probe_standard.score(standard_X_test[layer], standard_y_test)
        std_on_adv = probe_standard.score(adversarial_X_test[layer], adversarial_y_test)

        # Adversarial probe
        adv_on_std = probe_adversarial.score(standard_X_test[layer], standard_y_test)
        adv_on_adv = probe_adversarial.score(adversarial_X_test[layer], adversarial_y_test)

        # Mixed probe
        mix_on_std = probe_mixed.score(standard_X_test[layer], standard_y_test)
        mix_on_adv = probe_mixed.score(adversarial_X_test[layer], adversarial_y_test)

        print(f"  Standard probe: std={std_on_std:.3f}, adv={std_on_adv:.3f}")
        print(f"  Adversarial probe: std={adv_on_std:.3f}, adv={adv_on_adv:.3f}")
        print(f"  Mixed probe: std={mix_on_std:.3f}, adv={mix_on_adv:.3f}")
        print()

        # Save probes
        with open(f'{output_dir}/standard_layer_{layer}_probe.pkl', 'wb') as f:
            pickle.dump(probe_standard, f)
        with open(f'{output_dir}/adversarial_layer_{layer}_probe.pkl', 'wb') as f:
            pickle.dump(probe_adversarial, f)
        with open(f'{output_dir}/mixed_layer_{layer}_probe.pkl', 'wb') as f:
            pickle.dump(probe_mixed, f)

        results.append({
            'layer': layer,
            'standard_on_standard': std_on_std,
            'standard_on_adversarial': std_on_adv,
            'adversarial_on_standard': adv_on_std,
            'adversarial_on_adversarial': adv_on_adv,
            'mixed_on_standard': mix_on_std,
            'mixed_on_adversarial': mix_on_adv,
        })

    df = pd.DataFrame(results)

    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print()
    print(df.to_string(index=False))
    print()

    # Find best configurations
    best_standard = df.loc[df['standard_on_standard'].idxmax()]
    best_adversarial = df.loc[df['adversarial_on_adversarial'].idxmax()]
    best_mixed_avg = df.assign(
        avg=(df['mixed_on_standard'] + df['mixed_on_adversarial']) / 2
    ).loc[lambda x: x['avg'].idxmax()]

    print("="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)
    print(f"\nBest standard probe (on standard test): Layer {int(best_standard['layer'])}")
    print(f"  Accuracy: {best_standard['standard_on_standard']:.3f}")

    print(f"\nBest adversarial probe (on adversarial test): Layer {int(best_adversarial['layer'])}")
    print(f"  Accuracy: {best_adversarial['adversarial_on_adversarial']:.3f}")

    print(f"\nBest mixed probe (average): Layer {int(best_mixed_avg['layer'])}")
    print(f"  Standard: {best_mixed_avg['mixed_on_standard']:.3f}")
    print(f"  Adversarial: {best_mixed_avg['mixed_on_adversarial']:.3f}")
    print(f"  Average: {(best_mixed_avg['mixed_on_standard'] + best_mixed_avg['mixed_on_adversarial']) / 2:.3f}")
    print()

    # Save results
    results_file = Path(output_dir).parent / 'results' / 'adversarial_probe_comparison.csv'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_file, index=False)
    print(f"✓ Results saved to {results_file}")

    return df


def main():
    print("="*70)
    print("ADVERSARIAL PROBE TRAINING EXPERIMENT")
    print("="*70)
    print()

    # Configuration
    model_name = 'gpt2'
    standard_dataset_path = 'research/datasets/temporal_scope_caa.json'
    output_dir = 'research/probes'

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded\n")

    # Create datasets
    print("Creating datasets...")
    standard_data = load_standard_caa_dataset(standard_dataset_path)
    adversarial_data = create_adversarial_training_dataset()
    print(f"  ✓ Standard dataset: {len(standard_data)} examples")
    print(f"  ✓ Adversarial dataset: {len(adversarial_data)} examples")
    print()

    # Extract activations
    (standard_X, standard_y), (adversarial_X, adversarial_y), (mixed_X, mixed_y) = \
        create_training_datasets(model, tokenizer, standard_data, adversarial_data)

    # Train probe variants
    results_df = train_probe_variants(
        (standard_X, standard_y),
        (adversarial_X, adversarial_y),
        (mixed_X, mixed_y),
        output_dir
    )

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Analyze which probe variant generalizes best")
    print("  2. Extract adversarial-robust steering vectors from mixed probes")
    print("  3. Test if adversarial steering handles negations/paraphrases better")
    print()


if __name__ == "__main__":
    main()
