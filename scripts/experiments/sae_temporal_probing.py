#!/usr/bin/env python3
"""
SAE Probing for Temporal Scope Detection

Based on methodology from "How do LLMs learn facts?" (Neel Nanda et al.)
https://arxiv.org/pdf/2502.16681

Uses Gemma-2-2B with Gemma Scope SAEs to train probes on SAE latent space
for classifying immediate vs long-term temporal scope.
"""

import torch
import numpy as np
import json
import re
import random
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, log_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# We'll use transformer_lens + sae_lens for SAE access
from transformer_lens import HookedTransformer
from sae_lens import SAE

DEVICE = "cpu" # GPU requires a huge amount of memory which I don't have
print(f"Using device: {DEVICE}")

# Configuration
MODEL_NAME = "gemma-2-2b"  # Smaller Gemma model for faster iteration
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"  # Gemma Scope SAEs (canonical version)

# Number of top SAE latents to use for probing; 32 is on the lower end, but we have a small training
# set so setting 64 might just grab the vectors corresponding to the words
TOP_K_LATENTS = 64

def load_temporal_dataset(dataset_path, verbose=False):
    """Load temporal scope pairs dataset."""
    if verbose:
        print(f"\n{'='*70}")
        print("LOADING DATASET")
        print(f"{'='*70}")
        print(f"Path: {dataset_path}")

    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data['pairs']
    metadata = data.get('metadata', {})

    if verbose:
        print(f"Loaded {len(pairs)} pairs")
        print(f"Description: {metadata.get('description', 'N/A')}")

    prompts = []
    labels = []

    for pair in pairs:
        prompts.append(pair['immediate'])
        labels.append(0)  # 0 = immediate
        prompts.append(pair['long_term'])
        labels.append(1)  # 1 = long_term

    if verbose:
        print(f"Total samples: {len(prompts)}")
        print(f"  Immediate (0): {sum(l == 0 for l in labels)}")
        print(f"  Long-term (1): {sum(l == 1 for l in labels)}")
        print(f"  Example: {prompts[0]} {labels[0]}")
        print(f"  Example: {prompts[1]} {labels[1]}")
        print(f"  Example: {prompts[2]} {labels[2]}")

    return prompts, np.array(labels), metadata


def load_temporal_test_dataset(dataset_path, seed=42, verbose=False):
    """
    Load temporal_scope_clean.json as a test dataset.

    For each pair, randomly chooses either immediate or long_term answer,
    strips the (A)/(B) prefix, and creates a single query (question + answer).

    Args:
        dataset_path: Path to temporal_scope_clean.json
        seed: Random seed for reproducibility

    Returns:
        prompts: List of full queries (question + stripped answer)
        labels: numpy array of labels (0=immediate, 1=long_term)
        metadata: Dataset metadata dict
    """
    random.seed(seed)

    if verbose:
        print(f"\n{'='*70}")
        print("LOADING TEST DATASET")
        print(f"{'='*70}")
        print(f"Path: {dataset_path}")

    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data['pairs']
    metadata = data.get('metadata', {})

    if verbose:
        print(f"Loaded {len(pairs)} pairs")
        print(f"Description: {metadata.get('description', 'N/A')}")

    prompts = []
    labels = []

    for pair in pairs:
        question = pair['question']

        # Randomly choose immediate or long_term
        is_long_term = random.choice([0, 1])

        if is_long_term:
            answer = pair['long_term']
            labels.append(1)
        else:
            answer = pair['immediate']
            labels.append(0)

        # Strip (A) or (B) prefix - handles " (A) " or " (B) " format
        answer_stripped = re.sub(r'^\s*\([AB]\)\s*', '', answer)

        # Create full query
        full_query = question + " " + answer_stripped
        prompts.append(full_query)

    if verbose:
        print(f"Total samples: {len(prompts)}")
        print(f"  Immediate (0): {sum(l == 0 for l in labels)}")
        print(f"  Long-term (1): {sum(l == 1 for l in labels)}")
        print(f"  Example: {prompts[0]}")
        print(f"    Label: {labels[0]}")

    return prompts, np.array(labels), metadata


def extract_activations_with_sae(model, sae, prompts, layer, batch_size=16, verbose=True):
    """
    Extract residual stream activations and pass through SAE encoder.

    Returns:
        raw_activations: (n_prompts, d_model) - raw residual stream activations
        sae_latents: (n_prompts, d_sae) - SAE encoded latent space
    """

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXTRACTING ACTIVATIONS (Layer {layer})")
        print(f"{'='*70}")
        print(f"Processing {len(prompts)} prompts in batches of {batch_size}...")

    hook_name = f"blocks.{layer}.hook_resid_post"

    all_raw_acts = []
    all_sae_latents = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches", disable=not verbose):
        batch_prompts = prompts[i:i + batch_size]

        with torch.no_grad():
            temp, cache = model.run_with_cache(
                batch_prompts,
                names_filter=[hook_name],
                stop_at_layer=layer + 1,  # Only compute up to needed layer (saves memory)
            )

        # Get last token activations: (batch, seq_len, d_model) -> (batch, d_model)
        acts = cache[hook_name]
        assert len(acts.shape) == 3, "Why is this shape not dimension 3?"
        last_token_acts = acts[:, -1, :]  # (batch, d_model) Methodology of Nanda paper

        # Pass through SAE encoder
        sae_out = sae.encode(last_token_acts)  # (batch, d_sae)

        all_raw_acts.append(last_token_acts.detach().float().cpu().numpy())
        all_sae_latents.append(sae_out.detach().float().cpu().numpy())

    raw_activations = np.concatenate(all_raw_acts, axis=0)
    sae_latents = np.concatenate(all_sae_latents, axis=0)

    if verbose:
        print(f"\nExtracted activations:")
        print(f"  Raw shape: {raw_activations.shape}")
        print(f"  SAE latents shape: {sae_latents.shape}")
        print(f"  Mean SAE sparsity: {(sae_latents != 0).mean():.4f}")

    return raw_activations, sae_latents


def select_top_k_latents(sae_latents, labels, k=32, verbose=True):
    """
    Select top k SAE latents with highest absolute difference between classes.

    Implements Equation (1) from the paper:
    I = arg top_k |mean(Z[T1]) - mean(Z[T0])|

    where T1 = prompts with target=1 (long_term)
          T0 = prompts with target=0 (immediate)

    Returns:
        top_k_indices: Indices of most discriminative latents
        signed_diff: Signed difference (mean_T1 - mean_T0) for all latents.
                     Positive = fires more for long_term, negative = fires more for immediate.
                     Used for steering direction.
    """
    T0_mask = labels == 0  # immediate
    T1_mask = labels == 1  # long_term

    mean_T0 = sae_latents[T0_mask].mean(axis=0)  # (d_sae,)
    mean_T1 = sae_latents[T1_mask].mean(axis=0)  # (d_sae,)

    signed_diff = mean_T1 - mean_T0  # (d_sae,) - preserves direction info
    abs_diff = np.abs(signed_diff)   # for selection only

    top_k_indices = np.argsort(abs_diff)[-k:][::-1]  # descending order


    # Stats on selected latents
    if verbose:
        selected_diffs = abs_diff[top_k_indices]
        print(f"\nSelected latent statistics:")
        print(f"  Top k: {top_k_indices}")
        print(f"  Max abs diff: {selected_diffs.max():.4f}")
        print(f"  Min abs diff: {selected_diffs.min():.4f}")
        print(f"  Mean abs diff: {selected_diffs.mean():.4f}")

    return top_k_indices, signed_diff


def steer_with_sae(model, sae, prompt, layer, top_k_indices, signed_diff, alpha=1.0, direction="long_term"):
    """
    Steer model predictions using SAE decoder directions.

    Uses the discriminative SAE latents identified by probing to create a steering
    vector, then adds it to the residual stream during inference.

    Args:
        model: HookedTransformer instance
        sae: SAE instance with decoder weights (W_dec)
        prompt: Input prompt string
        layer: Layer to apply steering (same layer used for probing)
        top_k_indices: Indices of discriminative SAE latents (from select_top_k_latents)
        signed_diff: Signed difference (mean_T1 - mean_T0) from select_top_k_latents.
                     Positive values = latent fires more for long_term.
        alpha: Steering strength (default 1.0). Higher = stronger effect.
        direction: "long_term" (positive steering) or "immediate" (negative steering)

    Returns:
        logits: Model output logits with steering applied
    """
    # Get decoder directions for discriminative latents
    # W_dec shape: (d_sae, d_model), so indexing gives (k, d_model)
    steering_directions = sae.W_dec[top_k_indices].detach()  # (k, d_model)

    # Use SIGNED weights - this preserves direction information
    # Positive signed_diff means latent fires more for long_term
    # Negative signed_diff means latent fires more for immediate
    weights = signed_diff[top_k_indices]  # keep signs!
    weights = torch.tensor(weights, dtype=steering_directions.dtype, device=steering_directions.device)

    # Compute steering vector as weighted sum of decoder directions
    # This vector points toward "long_term" due to signed weights
    steering_vector = (weights[:, None] * steering_directions).sum(dim=0)  # (d_model,)

    # Flip direction for "long_term"
    # (signed_diff is mean_longterm - mean_immediate, so the raw vector points toward immediate)
    if direction == "long_term":
        steering_vector = -steering_vector

    # Create hook function that adds steering to last token
    hook_name = f"blocks.{layer}.hook_resid_post"

    def steering_hook(activations, hook):
        # activations shape: (batch, seq_len, d_model)
        # Add steering vector to last token position only
        activations[:, -1, :] += alpha * steering_vector
        return activations

    # Run model with steering hook
    with torch.no_grad():
        logits = model.run_with_hooks(
            prompt,
            fwd_hooks=[(hook_name, steering_hook)]
        )

    return logits


def train_sae_probe(sae_latents, labels, top_k_indices, test_size=0.2, verbose=True):
    """
    Train a linear probe on selected SAE latents.

    From the paper: "We then train a probe p_SAE to map from Z[:, I] -> t"

    Verified
    """
    if verbose:
        print(f"\n{'='*70}")
        print("TRAINING SAE PROBE")
        print(f"{'='*70}")

    X = sae_latents[:, top_k_indices]  # (n_samples, k)
    y = labels

    if verbose:
        print(f"Feature matrix shape: {X.shape}")
        print(f"Using {len(top_k_indices)} SAE latents as features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if verbose:
        print(f"\nTrain set: {len(y_train)} samples")
        print(f"Test set: {len(y_test)} samples")

    # Note here, we don't use l1, it seems odd to require more l1?
    # probe = LogisticRegression(penalty='elasticnet', l1_ratio=1, solver='saga', max_iter=5000, random_state=42)
    probe = LogisticRegression(max_iter=5000, random_state=42)

    if verbose:
        print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(probe, X_train, y_train, cv=5, scoring='accuracy')
    if verbose:
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        print("\nTraining final probe...")
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # Compute log loss (cross-entropy)
    train_loss = log_loss(y_train, probe.predict_proba(X_train))
    test_loss = log_loss(y_test, probe.predict_proba(X_test))
    if verbose:
        print(f"\nTrain Log Loss: {train_loss:.4f}")
        print(f"Test Log Loss:  {test_loss:.4f}")
        print(f"Test Accuracy:  {test_acc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Immediate', 'Long-term']))

    return probe, {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'test_accuracy': test_acc,
        'train_log_loss': train_loss,
        'test_log_loss': test_loss,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_features': len(top_k_indices)
    }


def train_activation_probe(raw_activations, labels, test_size=0.2, verbose=True):
    """Train a baseline probe directly on raw activations (without SAE).

    Verified; only first one of 6...
    """
    if verbose:
        print(f"\n{'='*70}")
        print("TRAINING ACTIVATION PROBE (Baseline)")
        print(f"{'='*70}")

    X = raw_activations
    y = labels

    if verbose:
        print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    probe = LogisticRegression(max_iter=5000, random_state=42)

    if verbose:
        print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(probe, X_train, y_train, cv=5, scoring='accuracy')
    if verbose:
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # Compute log loss (cross-entropy)
    train_loss = log_loss(y_train, probe.predict_proba(X_train))
    test_loss = log_loss(y_test, probe.predict_proba(X_test))

    if verbose:
        print(f"\nTrain Log Loss: {train_loss:.4f}")
        print(f"Test Log Loss:  {test_loss:.4f}")
        print(f"Test Accuracy:  {test_acc:.3f}")

    return probe, {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'test_accuracy': test_acc,
        'train_log_loss': train_loss,
        'test_log_loss': test_loss,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_features': X.shape[1]
    }


def run_multi_layer_experiment(model_name, sae_release, prompts, labels, layers, k=50):
    """Run SAE probing experiment across multiple layers.

    Verified
    """
    print(f"\n{'='*70}")
    print("MULTI-LAYER SAE PROBING EXPERIMENT")
    print(f"{'='*70}")
    print(f"Layers to probe: {layers}")
    print(f"Top-k latents: {k}")

    results = []

    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=DEVICE)
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hidden dim: {model.cfg.d_model}")
    print(f"  Vocab size: {model.cfg.d_vocab}")

    # Load additional tests
    test_dataset_path = Path(__file__).parent.parent.parent / "data" / "raw" / "temporal_scope_clean.json"
    test_prompts, test_labels, _ = load_temporal_test_dataset(
        test_dataset_path,
        seed=42
    )

    for layer in layers:
        print(f"\n{'#'*70}")
        print(f"# LAYER {layer}")
        print(f"{'#'*70}")

        # Load SAE for this layer
        sae_id = f"layer_{layer}/width_16k/canonical"
        try:
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=DEVICE
            )
        except Exception as e:
            print(f"Failed to load SAE for layer {layer}: {e}")
            continue

        # Extract activations
        raw_acts, sae_latents = extract_activations_with_sae(
            model, sae, prompts, layer
        )

        # Select top-k latents
        top_k_indices, signed_diff = select_top_k_latents(sae_latents, labels, k=k)

        # Train SAE probe
        sae_probe, sae_metrics = train_sae_probe(sae_latents, labels, top_k_indices)

        # Train activation probe (baseline)
        act_probe, act_metrics = train_activation_probe(raw_acts, labels)

        # Now test on test_prompts
        test_raw_acts, test_sae_latents = extract_activations_with_sae(
            model, sae, test_prompts, layer
        )

        sae_test_preds = sae_probe.predict(test_sae_latents[:, top_k_indices])
        sae_test_acc = np.mean(sae_test_preds == test_labels)
        print(f"\n  SAE test accuracy: {sae_test_acc:.3f} ({np.sum(sae_test_preds == test_labels)}/{len(test_labels)})")

        act_test_preds = act_probe.predict(test_raw_acts)
        act_test_acc = np.mean(act_test_preds == test_labels)
        print(f"  ACT test accuracy: {act_test_acc:.3f} ({np.sum(act_test_preds == test_labels)}/{len(test_labels)})")

        results.append({
            'layer': layer,
            'sae_probe': sae_metrics,
            'activation_probe': act_metrics,
            'top_k_indices': top_k_indices.tolist(),
            'signed_diff_top10': signed_diff[top_k_indices[:10]].tolist(),
            'sae_test_accuracy': float(sae_test_acc),
            'act_test_accuracy': float(act_test_acc),
        })

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results, model


def plot_results(results, output_path):
    """Plot SAE probe vs activation probe accuracy across layers."""
    layers = [r['layer'] for r in results]
    sae_accs = [r['sae_probe']['test_accuracy'] for r in results]
    act_accs = [r['activation_probe']['test_accuracy'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax.bar(x - width/2, sae_accs, width, label='SAE Probe', color='steelblue')
    bars2 = ax.bar(x + width/2, act_accs, width, label='Activation Probe', color='coral')

    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Temporal Scope Classification: SAE Probe vs Activation Probe')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.set_ylim(0, 1)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")
    plt.close()


def main():
    print("="*70)
    print("SAE TEMPORAL PROBING EXPERIMENT")
    print("Using methodology from Neel Nanda et al.")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    dataset_path = Path(__file__).parent.parent.parent / "data" / "raw" / "temporal_scope_pairs_minimal.json"
    output_dir = Path(__file__).parent.parent.parent / "results" / "sae_probing_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Middle layers tend to encode semantic information best
    layers_to_probe = [6, 13, 20, 24]  # Middle-ish layers for Gemma-2-2b (26 layers total) # Checked

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  SAE Release: {SAE_RELEASE}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output: {output_dir}")
    print(f"  Layers: {layers_to_probe}")
    print(f"  Top-k latents: {TOP_K_LATENTS}")
    print(f"  Device: {DEVICE}")

    # Load dataset
    prompts, labels, metadata = load_temporal_dataset(dataset_path)

    # Run experiment
    results, model = run_multi_layer_experiment(
        MODEL_NAME, SAE_RELEASE, prompts, labels, layers_to_probe, k=TOP_K_LATENTS
    )

    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    print("\nLayer | SAE Probe | Act Probe | Difference")
    print("-" * 50)
    for r in results:
        sae_acc = r['sae_probe']['test_accuracy']
        act_acc = r['activation_probe']['test_accuracy']
        diff = sae_acc - act_acc
        print(f"  {r['layer']:3d} |    {sae_acc:.3f}  |   {act_acc:.3f}   | {diff:+.3f}")

    best_sae = max(results, key=lambda x: x['sae_probe']['test_accuracy'])
    best_act = max(results, key=lambda x: x['activation_probe']['test_accuracy'])

    print(f"\nBest SAE probe: Layer {best_sae['layer']} ({best_sae['sae_probe']['test_accuracy']:.3f})")
    print(f"Best Act probe: Layer {best_act['layer']} ({best_act['activation_probe']['test_accuracy']:.3f})")


    # Save results
    results_file = output_dir / "sae_probing_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'model': MODEL_NAME,
                'sae_release': SAE_RELEASE,
                'layers': layers_to_probe,
                'top_k_latents': TOP_K_LATENTS,
                'n_prompts': len(prompts),
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Plot
    plot_path = output_dir / "sae_probing_comparison.png"
    plot_results(results, plot_path)

    # =========================================================================
    # STEERING DEMO: Compare unsteered vs steered outputs
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEERING DEMO")
    print(f"{'='*70}")

    # Use the best layer's results
    best_layer = best_sae['layer']
    top_k_indices = np.array(best_sae['top_k_indices'])

    # Reload SAE for steering (need full signed_diff, not just top10)
    sae_id = f"layer_{best_layer}/width_16k/canonical"
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=DEVICE)

    # Recompute signed_diff for the best layer (we only saved top10 in results)
    _, sae_latents = extract_activations_with_sae(model, sae, prompts, best_layer, verbose=False)
    _, signed_diff = select_top_k_latents(sae_latents, labels, k=TOP_K_LATENTS, verbose=False)

    # Test prompts designed to elicit "now" vs "later" completions
    demo_prompts = [
        "I want to achieve success",
        "I need to get this done",
        "I should start working on this",
        "I want to be happy",
        "I will do it",
    ]

    # Get token IDs for temporal words
    temporal_tokens = {
        " now": model.tokenizer.encode(" now", add_special_tokens=False)[0],
        " later": model.tokenizer.encode(" later", add_special_tokens=False)[0],
        " soon": model.tokenizer.encode(" soon", add_special_tokens=False)[0],
        " today": model.tokenizer.encode(" today", add_special_tokens=False)[0],
        " tomorrow": model.tokenizer.encode(" tomorrow", add_special_tokens=False)[0],
        " eventually": model.tokenizer.encode(" eventually", add_special_tokens=False)[0],
    }

    print(f"\nUsing layer {best_layer} with top {TOP_K_LATENTS} discriminative latents")
    print(f"Comparing P('now') vs P('later') under different steering conditions")
    print(f"\nTemporal token IDs: {temporal_tokens}")

    # Test multiple alpha values
    alphas = [0, 10, 50, 100]

    for demo_prompt in demo_prompts:
        print(f"\n{'-'*60}")
        print(f"Prompt: \"{demo_prompt}\"")
        print(f"{'-'*60}")
        print(f"{'Alpha':<8} | {'Direction':<12} | {'P(now)':<12} | {'P(later)':<12} | {'now/later':<10}")
        print("-" * 70)

        for alpha in alphas:
            if alpha == 0:
                # Unsteered
                with torch.no_grad():
                    logits = model(demo_prompt)
                probs = torch.softmax(logits[0, -1], dim=-1)

                p_now = probs[temporal_tokens[" now"]].item()
                p_later = probs[temporal_tokens[" later"]].item()
                ratio = p_now / p_later if p_later > 0 else float('inf')

                print(f"{alpha:<8} | {'unsteered':<12} | {p_now:<12.6f} | {p_later:<12.6f} | {ratio:<10.2f}")
            else:
                # Long-term steered
                logits_lt = steer_with_sae(
                    model, sae, demo_prompt, best_layer,
                    top_k_indices, signed_diff, alpha=alpha, direction="long_term"
                )
                probs_lt = torch.softmax(logits_lt[0, -1], dim=-1)

                p_now_lt = probs_lt[temporal_tokens[" now"]].item()
                p_later_lt = probs_lt[temporal_tokens[" later"]].item()
                ratio_lt = p_now_lt / p_later_lt if p_later_lt > 0 else float('inf')

                print(f"{alpha:<8} | {'long_term':<12} | {p_now_lt:<12.6f} | {p_later_lt:<12.6f} | {ratio_lt:<10.2f}")

                # Immediate steered
                logits_im = steer_with_sae(
                    model, sae, demo_prompt, best_layer,
                    top_k_indices, signed_diff, alpha=alpha, direction="immediate"
                )
                probs_im = torch.softmax(logits_im[0, -1], dim=-1)

                p_now_im = probs_im[temporal_tokens[" now"]].item()
                p_later_im = probs_im[temporal_tokens[" later"]].item()
                ratio_im = p_now_im / p_later_im if p_later_im > 0 else float('inf')

                print(f"{alpha:<8} | {'immediate':<12} | {p_now_im:<12.6f} | {p_later_im:<12.6f} | {ratio_im:<10.2f}")

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    # SAE-lens must be high enough version; pytoml might be bugging out
    main()