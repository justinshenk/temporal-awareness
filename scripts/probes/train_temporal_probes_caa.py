#!/usr/bin/env python3
"""
Train linear probes to detect temporal scale from model activations.

This script validates that GPT-2 encodes temporal information by training
logistic regression probes on activations from CAA-format temporal prompts.

Workflow:
1. Load CAA temporal dataset (50 immediate vs long-term pairs)
2. Extract activations from model for each prompt
3. Train linear probe per layer to classify immediate vs long-term
4. Report accuracy and identify which layers encode temporal info

Success criteria:
- Accuracy > 70%: Model clearly encodes temporal information
- Accuracy 55-70%: Weak but detectable temporal signal
- Accuracy < 55%: No temporal information encoded
"""

import torch
import numpy as np
import json
import pickle
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


def load_caa_dataset(dataset_path):
    """Load CAA-format temporal dataset."""
    with open(dataset_path) as f:
        data = json.load(f)

    if 'pairs' in data:
        pairs = data['pairs']
        metadata = data['metadata']
    else:
        pairs = data
        metadata = None

    return pairs, metadata


def extract_activations(model, tokenizer, prompt):
    """
    Extract activations from all layers for a given prompt.

    Returns: dict mapping layer_idx -> activation vector (hidden_dim,)
    """
    inputs = tokenizer(prompt, return_tensors='pt')

    activations = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            # output[0] is hidden_states: (batch, seq_len, hidden_dim)
            # Take last token activation
            activations[layer_idx] = output[0][0, -1, :].detach().cpu().numpy()
        return hook

    # Register hooks for all layers
    hooks = []
    for i, layer in enumerate(model.transformer.h):
        hook = layer.register_forward_hook(hook_fn(i))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations


def create_probe_dataset(model, tokenizer, pairs):
    """
    Extract activations for all prompts and create probe training dataset.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        pairs: List of CAA-format prompt pairs

    Returns:
        X_by_layer: Dict mapping layer_idx -> numpy array (n_samples, hidden_dim)
        y: Labels (0=immediate, 1=long_term)
    """
    print(f"Extracting activations from {len(pairs)} prompt pairs...")
    print(f"This will create {len(pairs) * 2} samples (immediate + long-term)\n")

    n_layers = len(model.transformer.h)

    # Initialize storage
    activations_by_layer = {i: [] for i in range(n_layers)}
    labels = []

    for pair in tqdm(pairs, desc="Processing pairs"):
        question = pair['question']

        # Detect option keys (immediate/long_term or option_a/option_b)
        option_keys = [k for k in pair.keys() if k not in ['question', 'category']]

        if len(option_keys) != 2:
            raise ValueError(f"Expected 2 options, got {option_keys}")

        # Assume first key is immediate, second is long-term
        # (or alphabetically option_a is immediate, option_b is long-term)
        immediate_key = option_keys[0]
        long_term_key = option_keys[1]

        # Create full prompts
        immediate_prompt = question + "\n\nChoices:\n" + pair[immediate_key]
        long_term_prompt = question + "\n\nChoices:\n" + pair[long_term_key]

        # Extract activations
        immediate_acts = extract_activations(model, tokenizer, immediate_prompt)
        long_term_acts = extract_activations(model, tokenizer, long_term_prompt)

        # Store activations and labels
        for layer in range(n_layers):
            activations_by_layer[layer].append(immediate_acts[layer])
            activations_by_layer[layer].append(long_term_acts[layer])

        labels.extend([0, 1])  # 0=immediate, 1=long_term

    # Convert to numpy arrays
    X_by_layer = {}
    for layer in range(n_layers):
        X_by_layer[layer] = np.array(activations_by_layer[layer])

    y = np.array(labels)

    print(f"\n✓ Created dataset:")
    print(f"  Total samples: {len(y)}")
    print(f"  Immediate (0): {sum(y == 0)}")
    print(f"  Long-term (1): {sum(y == 1)}")
    print(f"  Balance: {sum(y == 0)/len(y):.1%} / {sum(y == 1)/len(y):.1%}")
    print(f"  Features per layer: {X_by_layer[0].shape[1]}\n")

    return X_by_layer, y


def train_probes(X_by_layer, y, output_dir='research/probes'):
    """
    Train a linear probe for each layer.

    Returns:
        results_df: DataFrame with accuracy per layer
    """
    print("="*70)
    print("TRAINING TEMPORAL PROBES")
    print("="*70)
    print()

    results = []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_layers = len(X_by_layer)

    for layer in range(n_layers):
        print(f"Layer {layer}/{n_layers-1}")
        print("-"*70)

        X = X_by_layer[layer]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train probe with cross-validation
        probe = LogisticRegression(max_iter=1000, random_state=42)

        # 5-fold CV on training set
        cv_scores = cross_val_score(
            probe, X_train, y_train, cv=5, scoring='accuracy'
        )

        # Train on full training set
        probe.fit(X_train, y_train)

        # Evaluate on test set
        test_acc = probe.score(X_test, y_test)

        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        print(f"  Test Accuracy: {test_acc:.3f}")

        # Save probe
        probe_file = Path(output_dir) / f'temporal_caa_layer_{layer}_probe.pkl'
        with open(probe_file, 'wb') as f:
            pickle.dump(probe, f)

        print(f"  ✓ Saved to {probe_file}")
        print()

        results.append({
            'layer': layer,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'test_accuracy': test_acc,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_features': X.shape[1]
        })

    results_df = pd.DataFrame(results)

    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    print()

    # Find best layer
    best_layer = results_df.loc[results_df['test_accuracy'].idxmax()]
    print("="*70)
    print("BEST LAYER")
    print("="*70)
    print(f"  Layer: {int(best_layer['layer'])}")
    print(f"  Test Accuracy: {best_layer['test_accuracy']:.3f}")
    print(f"  CV Accuracy: {best_layer['cv_accuracy_mean']:.3f} (+/- {best_layer['cv_accuracy_std']:.3f})")
    print()

    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70)

    best_acc = best_layer['test_accuracy']

    if best_acc >= 0.70:
        print("  ✓ STRONG SIGNAL (accuracy ≥ 70%)")
        print("  GPT-2 clearly encodes temporal information!")
        print("  The model has learned to represent immediate vs long-term thinking.")
    elif best_acc >= 0.55:
        print("  ○ WEAK SIGNAL (accuracy 55-70%)")
        print("  Temporal information is present but not strongly encoded.")
        print("  Consider:")
        print("    - More diverse prompts")
        print("    - Larger dataset")
        print("    - Different prompt format")
    else:
        print("  ✗ NO SIGNAL (accuracy < 55%)")
        print("  Model does not encode temporal information in activations.")
        print("  Steering may not work as expected.")

    print("="*70)
    print()

    # Save results
    results_file = Path(output_dir).parent / 'results' / 'temporal_probe_results_caa.csv'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"✓ Results saved to {results_file}\n")

    return results_df


def detailed_evaluation(model, tokenizer, pairs, probe_path, layer):
    """
    Detailed evaluation of a specific probe with confusion matrix.
    """
    print("="*70)
    print(f"DETAILED EVALUATION - Layer {layer}")
    print("="*70)
    print()

    # Load probe
    with open(probe_path, 'rb') as f:
        probe = pickle.load(f)

    # Extract activations
    X_by_layer, y = create_probe_dataset(model, tokenizer, pairs)
    X = X_by_layer[layer]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Predictions
    y_pred = probe.predict(X_test)

    print("Classification Report:")
    print("-"*70)
    print(classification_report(
        y_test, y_pred,
        target_names=['Immediate (0)', 'Long-term (1)']
    ))

    print("\nConfusion Matrix:")
    print("-"*70)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"               Immediate  Long-term")
    print(f"Actual Immediate    {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"       Long-term    {cm[1,0]:3d}       {cm[1,1]:3d}")
    print()


def main():
    print("="*70)
    print("TEMPORAL PROBE TRAINING (CAA FORMAT)")
    print("="*70)
    print()

    # Configuration
    dataset_path = 'research/datasets/temporal_scope_caa.json'
    model_name = 'gpt2'
    output_dir = 'research/probes'

    print(f"Configuration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_dir}")
    print()

    # Load dataset
    print("Loading CAA dataset...")
    pairs, metadata = load_caa_dataset(dataset_path)

    if metadata:
        print(f"  Dimension: {metadata['dimension']}")
        print(f"  Style: {metadata['style']}")
        print(f"  Pairs: {metadata['n_pairs']}")
    else:
        print(f"  Pairs: {len(pairs)}")
    print()

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()
    print("✓ Model loaded")
    print()

    # Extract activations and create dataset
    X_by_layer, y = create_probe_dataset(model, tokenizer, pairs)

    # Train probes
    results_df = train_probes(X_by_layer, y, output_dir)

    # Detailed evaluation on best layer
    best_layer = results_df.loc[results_df['test_accuracy'].idxmax(), 'layer']
    probe_path = Path(output_dir) / f'temporal_caa_layer_{int(best_layer)}_probe.pkl'

    detailed_evaluation(model, tokenizer, pairs, probe_path, int(best_layer))

    print("="*70)
    print("COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Run validation with steering vectors:")
    print("     python research/tools/validate_temporal_steering_with_probes.py")
    print()
    print("  2. Check if steering changes probe predictions:")
    print("     Negative strength → probe predicts immediate (0)")
    print("     Positive strength → probe predicts long-term (1)")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train temporal probes on CAA-format dataset"
    )
    parser.add_argument(
        '--dataset',
        default='research/datasets/temporal_scope_caa.json',
        help='Path to CAA dataset'
    )
    parser.add_argument(
        '--model',
        default='gpt2',
        help='Model name'
    )
    parser.add_argument(
        '--output',
        default='research/probes',
        help='Output directory for probes'
    )

    args = parser.parse_args()

    main()
