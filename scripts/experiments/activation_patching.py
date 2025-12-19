#!/usr/bin/env python3
"""
Activation Patching for Causal Circuit Analysis of Temporal Scope

This experiment identifies which layers/components causally encode temporal scope
by patching activations between immediate and long-term inputs.

Method:
1. Run model on immediate input, cache activations
2. Run model on long-term input, but patch in immediate activations at specific layers
3. Measure if probe predictions flip (causal effect)

References:
- Meng et al. "Locating and Editing Factual Associations" (ROME)
- Anthropic's Causal Tracing work
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatchingResult:
    """Results from patching a single layer/component."""
    layer: int
    component: str  # 'residual', 'attn', 'mlp'
    flip_rate: float  # % of predictions that flipped
    n_samples: int
    mean_prob_change: float  # average change in probe probability


class ActivationCache:
    """Cache activations during forward pass."""
    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def save_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.cache[name] = output[0].detach().clone()
            else:
                self.cache[name] = output.detach().clone()
        return hook

    def clear(self):
        self.cache.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def load_model_and_probe():
    """Load GPT-2 and trained temporal probe."""
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()

    # Load best probe (Layer 8)
    probe_path = Path('results/checkpoints/temporal_caa_layer_8_probe.pkl')
    if not probe_path.exists():
        # Try alternate locations
        for alt_path in [
            'results/probes/temporal_caa_layer_8_probe.pkl',
            'research/probes/temporal_caa_layer_8_probe.pkl'
        ]:
            if Path(alt_path).exists():
                probe_path = Path(alt_path)
                break

    if probe_path.exists():
        print(f"Loading probe from {probe_path}...")
        with open(probe_path, 'rb') as f:
            probe_data = pickle.load(f)
        probe = probe_data['probe'] if isinstance(probe_data, dict) else probe_data
    else:
        print("No trained probe found, training a quick one...")
        probe = train_quick_probe(model, tokenizer)

    return model, tokenizer, probe


def train_quick_probe(model, tokenizer, layer=8):
    """Train a quick probe if none exists."""
    with open('data/raw/temporal_scope_caa.json') as f:
        data = json.load(f)
    pairs = data['pairs']

    X, y = [], []
    for pair in tqdm(pairs[:30], desc="Extracting activations for probe"):
        for label, key in [(0, 'immediate'), (1, 'long_term')]:
            text = pair['question'] + pair[key]
            inputs = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(inputs['input_ids'], output_hidden_states=True)
            act = outputs.hidden_states[layer][0, -1, :].cpu().numpy()
            X.append(act)
            y.append(label)

    probe = LogisticRegression(max_iter=1000)
    probe.fit(np.array(X), np.array(y))
    print(f"Quick probe accuracy: {probe.score(np.array(X), np.array(y)):.1%}")
    return probe


def load_dataset():
    """Load temporal scope dataset."""
    with open('data/raw/temporal_scope_caa.json') as f:
        data = json.load(f)
    return data['pairs']


def get_activations_with_cache(model, tokenizer, text, layers_to_cache: List[int]):
    """Run forward pass and cache activations at specified layers."""
    cache = ActivationCache()

    # Register hooks for each layer
    for layer_idx in layers_to_cache:
        # Cache residual stream (output of transformer block)
        block = model.transformer.h[layer_idx]
        hook = block.register_forward_hook(cache.save_hook(f'layer_{layer_idx}_residual'))
        cache.hooks.append(hook)

        # Cache attention output
        attn_hook = block.attn.register_forward_hook(cache.save_hook(f'layer_{layer_idx}_attn'))
        cache.hooks.append(attn_hook)

        # Cache MLP output
        mlp_hook = block.mlp.register_forward_hook(cache.save_hook(f'layer_{layer_idx}_mlp'))
        cache.hooks.append(mlp_hook)

    # Forward pass
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs['input_ids'], output_hidden_states=True)

    # Get hidden states for probe
    hidden_states = {i: outputs.hidden_states[i] for i in range(len(outputs.hidden_states))}

    result = {
        'cache': dict(cache.cache),
        'hidden_states': hidden_states,
        'input_ids': inputs['input_ids']
    }

    cache.clear()
    return result


def run_with_patching(model, tokenizer, text, patch_cache: Dict[str, torch.Tensor],
                      layer: int, component: str):
    """
    Run forward pass with patched activations.

    Only patches the FINAL token position to avoid shape mismatches
    and because the probe only uses final token activations.
    """
    patch_key = f'layer_{layer}_{component}'

    if patch_key not in patch_cache:
        raise ValueError(f"No cached activation for {patch_key}")

    cached_activation = patch_cache[patch_key]
    # Get the final token activation from cache
    cached_final = cached_activation[:, -1:, :]  # [1, 1, hidden_dim]

    def patch_hook(module, input, output):
        # Only patch the final token position
        if isinstance(output, tuple):
            out_tensor = output[0]
            # Replace only the last position with cached value
            patched = out_tensor.clone()
            patched[:, -1:, :] = cached_final
            return (patched,) + output[1:]
        else:
            patched = output.clone()
            patched[:, -1:, :] = cached_final
            return patched

    # Register patching hook
    block = model.transformer.h[layer]
    if component == 'residual':
        hook = block.register_forward_hook(patch_hook)
    elif component == 'attn':
        hook = block.attn.register_forward_hook(patch_hook)
    elif component == 'mlp':
        hook = block.mlp.register_forward_hook(patch_hook)
    else:
        raise ValueError(f"Unknown component: {component}")

    # Forward pass with patching
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs['input_ids'], output_hidden_states=True)

    hook.remove()
    return outputs


def run_patching_experiment(model, tokenizer, probe, pairs: List[dict],
                            layers: List[int], components: List[str],
                            probe_layer: int = 8) -> List[PatchingResult]:
    """
    Run activation patching experiment.

    For each pair:
    1. Get activations for immediate input
    2. Run long-term input with patched immediate activations
    3. Check if probe prediction flips from long-term to immediate
    """
    results = []

    for layer in tqdm(layers, desc="Layers"):
        for component in components:
            flip_count = 0
            prob_changes = []
            n_valid = 0

            for pair in pairs:
                immediate_text = pair['question'] + pair['immediate']
                longterm_text = pair['question'] + pair['long_term']

                # Get cached activations from immediate input
                immediate_result = get_activations_with_cache(
                    model, tokenizer, immediate_text, [layer]
                )

                # Get baseline prediction for long-term input (should be class 1)
                longterm_baseline = get_activations_with_cache(
                    model, tokenizer, longterm_text, [layer]
                )
                baseline_act = longterm_baseline['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
                baseline_prob = probe.predict_proba([baseline_act])[0, 1]  # P(long-term)
                baseline_pred = probe.predict([baseline_act])[0]

                # Skip if baseline is already wrong
                if baseline_pred != 1:
                    continue
                n_valid += 1

                # Run with patching
                patched_outputs = run_with_patching(
                    model, tokenizer, longterm_text,
                    immediate_result['cache'], layer, component
                )

                # Get patched prediction
                patched_act = patched_outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
                patched_prob = probe.predict_proba([patched_act])[0, 1]
                patched_pred = probe.predict([patched_act])[0]

                # Check if prediction flipped
                if patched_pred == 0:  # Flipped to immediate
                    flip_count += 1

                prob_changes.append(baseline_prob - patched_prob)

            if n_valid > 0:
                results.append(PatchingResult(
                    layer=layer,
                    component=component,
                    flip_rate=flip_count / n_valid,
                    n_samples=n_valid,
                    mean_prob_change=np.mean(prob_changes)
                ))

    return results


def plot_results(results: List[PatchingResult], output_path: str):
    """Plot patching results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Organize by component
    components = list(set(r.component for r in results))
    colors = {'residual': '#3498db', 'attn': '#e74c3c', 'mlp': '#2ecc71'}

    # Plot 1: Flip rate by layer
    ax1 = axes[0]
    for comp in components:
        comp_results = [r for r in results if r.component == comp]
        layers = [r.layer for r in comp_results]
        flip_rates = [r.flip_rate * 100 for r in comp_results]
        ax1.plot(layers, flip_rates, 'o-', label=comp.upper(), color=colors.get(comp, 'gray'), linewidth=2, markersize=8)

    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Flip Rate (%)', fontsize=12)
    ax1.set_title('Causal Effect: Patching Immediate → Long-term', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (strong effect)')

    # Plot 2: Probability change by layer
    ax2 = axes[1]
    for comp in components:
        comp_results = [r for r in results if r.component == comp]
        layers = [r.layer for r in comp_results]
        prob_changes = [r.mean_prob_change * 100 for r in comp_results]
        ax2.bar([l + components.index(comp) * 0.25 for l in layers], prob_changes,
                width=0.25, label=comp.upper(), color=colors.get(comp, 'gray'), alpha=0.8)

    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Mean Probability Change (%)', fontsize=12)
    ax2.set_title('Probe Probability Shift from Patching', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def print_results_table(results: List[PatchingResult]):
    """Print results as a formatted table."""
    print("\n" + "=" * 70)
    print("ACTIVATION PATCHING RESULTS")
    print("=" * 70)
    print(f"{'Layer':<8} {'Component':<12} {'Flip Rate':<12} {'Prob Change':<14} {'N':<6}")
    print("-" * 70)

    # Sort by layer then component
    sorted_results = sorted(results, key=lambda r: (r.layer, r.component))

    for r in sorted_results:
        flip_str = f"{r.flip_rate*100:.1f}%"
        prob_str = f"{r.mean_prob_change*100:+.1f}%"
        marker = "  ← significant" if r.flip_rate > 0.3 else ""
        print(f"{r.layer:<8} {r.component:<12} {flip_str:<12} {prob_str:<14} {r.n_samples:<6}{marker}")

    print("=" * 70)

    # Summary
    high_effect = [r for r in results if r.flip_rate > 0.3]
    if high_effect:
        print("\nHigh causal effect (>30% flip rate):")
        for r in sorted(high_effect, key=lambda x: -x.flip_rate):
            print(f"  Layer {r.layer} {r.component}: {r.flip_rate*100:.1f}% flip rate")


def save_results(results: List[PatchingResult], output_path: str):
    """Save results to JSON."""
    data = {
        'experiment': 'activation_patching',
        'description': 'Causal circuit analysis via activation patching',
        'results': [
            {
                'layer': r.layer,
                'component': r.component,
                'flip_rate': r.flip_rate,
                'mean_prob_change': r.mean_prob_change,
                'n_samples': r.n_samples
            }
            for r in results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {output_path}")


def main():
    print("=" * 70)
    print("ACTIVATION PATCHING EXPERIMENT")
    print("Identifying causal components for temporal scope encoding")
    print("=" * 70 + "\n")

    # Load model and probe
    model, tokenizer, probe = load_model_and_probe()

    # Load dataset
    pairs = load_dataset()
    print(f"Loaded {len(pairs)} pairs")

    # Use subset for faster experimentation
    test_pairs = pairs[:25]
    print(f"Using {len(test_pairs)} pairs for patching experiment\n")

    # Run experiment
    layers = list(range(12))  # All GPT-2 layers
    components = ['residual', 'attn', 'mlp']

    print("Running activation patching...")
    results = run_patching_experiment(
        model, tokenizer, probe, test_pairs,
        layers=layers, components=components, probe_layer=8
    )

    # Print results
    print_results_table(results)

    # Save results
    output_dir = Path('results/activation_patching')
    output_dir.mkdir(parents=True, exist_ok=True)

    save_results(results, output_dir / 'patching_results.json')
    plot_results(results, output_dir / 'patching_results.png')

    print("\nExperiment complete!")
    return results


if __name__ == '__main__':
    main()
