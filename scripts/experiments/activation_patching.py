#!/usr/bin/env python3
"""
Activation Patching for Causal Circuit Analysis of Temporal Scope

This experiment identifies which layers/components causally encode temporal scope
by patching activations between immediate and long-term inputs.

Method:
1. Run model on immediate input, cache activations
2. Run model on long-term input, but patch in immediate activations at specific layers
3. Measure if probe predictions flip (causal effect)

Methods (--improved flag enables all three):
- replacement: Swap target activation with source activation
- random: Replace with Gaussian noise (baseline for non-specific disruption)
- addition: Add (source - target) difference to target activation

References:
- Meng et al. "Locating and Editing Factual Associations" (ROME)
- Wang et al. "Interpretability in the Wild" (activation patching methodology)
- Anthropic's Causal Tracing work
"""

import argparse
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Activation patching for causal circuit analysis of temporal scope'
    )
    parser.add_argument(
        '--improved', action='store_true',
        help='Run all three methods: replacement, random baseline, and activation addition'
    )
    parser.add_argument(
        '--n-pairs', type=int, default=25,
        help='Number of pairs to use for experiment (default: 25)'
    )
    return parser.parse_args()


@dataclass
class PatchingResult:
    """Results from patching a single layer/component."""
    layer: int
    component: str  # 'residual', 'attn', 'mlp'
    method: str  # 'replacement', 'random', 'addition'
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
                      layer: int, component: str, method: str = 'replacement',
                      target_cache: Optional[Dict[str, torch.Tensor]] = None):
    """
    Run forward pass with patched activations.

    Args:
        method: 'replacement' | 'random' | 'addition'
            - replacement: Replace with source (immediate) activation
            - random: Replace with Gaussian noise matching source statistics
            - addition: Add (source - target) difference
        target_cache: Required for 'addition' method
    """
    patch_key = f'layer_{layer}_{component}'

    if patch_key not in patch_cache:
        raise ValueError(f"No cached activation for {patch_key}")

    source_activation = patch_cache[patch_key]
    source_final = source_activation[:, -1:, :]  # [1, 1, hidden_dim]

    # Get target activation for addition method
    target_final = None
    if method == 'addition':
        if target_cache is None or patch_key not in target_cache:
            raise ValueError(f"Addition method requires target_cache with {patch_key}")
        target_final = target_cache[patch_key][:, -1:, :]

    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            out_tensor = output[0]
            patched = out_tensor.clone()
        else:
            out_tensor = output
            patched = output.clone()

        # Apply patching method at final token position
        if method == 'replacement':
            patched[:, -1:, :] = source_final
        elif method == 'random':
            noise = torch.randn_like(source_final)
            patched[:, -1:, :] = noise * source_final.std() + source_final.mean()
        elif method == 'addition':
            diff = source_final - target_final
            patched[:, -1:, :] = out_tensor[:, -1:, :] + diff

        if isinstance(output, tuple):
            return (patched,) + output[1:]
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
                            methods: List[str] = None,
                            probe_layer: int = 8) -> List[PatchingResult]:
    """
    Run activation patching experiment.

    Args:
        methods: List of methods to run. Default is ['replacement'].
                 Use ['replacement', 'random', 'addition'] for full comparison.
    """
    if methods is None:
        methods = ['replacement']

    results = []
    total_iterations = len(layers) * len(components) * len(methods)

    with tqdm(total=total_iterations, desc="Patching") as pbar:
        for layer in layers:
            for component in components:
                for method in methods:
                    flip_count = 0
                    prob_changes = []
                    n_valid = 0

                    for pair in pairs:
                        immediate_text = pair['question'] + pair['immediate']
                        longterm_text = pair['question'] + pair['long_term']

                        # Cache activations from immediate (source) input
                        immediate_result = get_activations_with_cache(
                            model, tokenizer, immediate_text, [layer]
                        )

                        # Cache activations and baseline prediction for long-term (target)
                        longterm_result = get_activations_with_cache(
                            model, tokenizer, longterm_text, [layer]
                        )
                        baseline_act = longterm_result['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
                        baseline_prob = probe.predict_proba([baseline_act])[0, 1]
                        baseline_pred = probe.predict([baseline_act])[0]

                        # Skip if baseline prediction is already wrong
                        if baseline_pred != 1:
                            continue
                        n_valid += 1

                        # Run with patching
                        patched_outputs = run_with_patching(
                            model, tokenizer, longterm_text,
                            immediate_result['cache'], layer, component,
                            method=method,
                            target_cache=longterm_result['cache']
                        )

                        # Get patched prediction
                        patched_act = patched_outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
                        patched_prob = probe.predict_proba([patched_act])[0, 1]
                        patched_pred = probe.predict([patched_act])[0]

                        if patched_pred == 0:  # Flipped to immediate
                            flip_count += 1

                        prob_changes.append(baseline_prob - patched_prob)

                    if n_valid > 0:
                        results.append(PatchingResult(
                            layer=layer,
                            component=component,
                            method=method,
                            flip_rate=flip_count / n_valid,
                            n_samples=n_valid,
                            mean_prob_change=np.mean(prob_changes)
                        ))

                    pbar.update(1)
                    pbar.set_postfix(layer=layer, comp=component[:3], method=method[:3])

    return results


def plot_results(results: List[PatchingResult], output_path: str):
    """Plot patching results for single-method runs."""
    # Filter to replacement method only for backwards compatibility
    filtered = [r for r in results if r.method == 'replacement']
    if not filtered:
        filtered = results

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    components = list(set(r.component for r in filtered))
    colors = {'residual': '#3498db', 'attn': '#e74c3c', 'mlp': '#2ecc71'}

    ax1 = axes[0]
    for comp in components:
        comp_results = [r for r in filtered if r.component == comp]
        layers = [r.layer for r in comp_results]
        flip_rates = [r.flip_rate * 100 for r in comp_results]
        ax1.plot(layers, flip_rates, 'o-', label=comp.upper(),
                 color=colors.get(comp, 'gray'), linewidth=2, markersize=8)

    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Flip Rate (%)', fontsize=12)
    ax1.set_title('Causal Effect: Patching Immediate → Long-term', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    ax2 = axes[1]
    for comp in components:
        comp_results = [r for r in filtered if r.component == comp]
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
    print(f"Saved: {output_path}")
    plt.close()


def plot_method_comparison(results: List[PatchingResult], output_path: str):
    """Plot method comparison for residual stream (matching Issue #26 format)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    method_config = {
        'replacement': {'color': '#3498db', 'marker': 'o', 'label': 'Replacement'},
        'random': {'color': '#e74c3c', 'marker': 's', 'label': 'Random Baseline'},
        'addition': {'color': '#2ecc71', 'marker': '^', 'label': 'Activation Addition'}
    }

    for method, config in method_config.items():
        method_results = [r for r in results if r.method == method and r.component == 'residual']
        if not method_results:
            continue
        method_results.sort(key=lambda x: x.layer)
        layers = [r.layer for r in method_results]
        flip_rates = [r.flip_rate * 100 for r in method_results]
        ax.plot(layers, flip_rates, f'{config["marker"]}-',
                label=config['label'], color=config['color'],
                linewidth=2, markersize=8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Flip Rate (%)', fontsize=12)
    ax.set_title('Activation Patching: Method Comparison', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(12))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_results_table(results: List[PatchingResult], improved: bool = False):
    """Print results as a formatted table."""
    methods = list(set(r.method for r in results))

    if improved and len(methods) > 1:
        print_method_comparison_table(results)
    else:
        print_single_method_table(results)


def print_single_method_table(results: List[PatchingResult]):
    """Print results table for single-method runs."""
    print("\n" + "=" * 70)
    print("ACTIVATION PATCHING RESULTS")
    print("=" * 70)
    print(f"{'Layer':<8} {'Component':<12} {'Flip Rate':<12} {'Prob Change':<14} {'N':<6}")
    print("-" * 70)

    sorted_results = sorted(results, key=lambda r: (r.layer, r.component))
    for r in sorted_results:
        flip_str = f"{r.flip_rate*100:.1f}%"
        prob_str = f"{r.mean_prob_change*100:+.1f}%"
        marker = "  ←" if r.flip_rate > 0.3 else ""
        print(f"{r.layer:<8} {r.component:<12} {flip_str:<12} {prob_str:<14} {r.n_samples:<6}{marker}")

    print("=" * 70)


def print_method_comparison_table(results: List[PatchingResult]):
    """Print method comparison table with net effects (residual stream only)."""
    print("\n" + "=" * 70)
    print("ACTIVATION PATCHING: METHOD COMPARISON (Residual Stream)")
    print("=" * 70)
    print(f"{'Layer':<8} {'Replace':<12} {'Random':<12} {'Addition':<12} {'Net Effect':<12}")
    print("-" * 70)

    # Build lookup dictionaries
    replace_by_layer = {r.layer: r for r in results
                        if r.method == 'replacement' and r.component == 'residual'}
    random_by_layer = {r.layer: r for r in results
                       if r.method == 'random' and r.component == 'residual'}
    addition_by_layer = {r.layer: r for r in results
                         if r.method == 'addition' and r.component == 'residual'}

    layers = sorted(set(replace_by_layer.keys()))
    peak_layer, peak_net = None, -100

    for layer in layers:
        rep = replace_by_layer.get(layer)
        rnd = random_by_layer.get(layer)
        add = addition_by_layer.get(layer)

        rep_str = f"{rep.flip_rate*100:.0f}%" if rep else "N/A"
        rnd_str = f"{rnd.flip_rate*100:.0f}%" if rnd else "N/A"
        add_str = f"{add.flip_rate*100:.0f}%" if add else "N/A"

        if rep and rnd:
            net = (rep.flip_rate - rnd.flip_rate) * 100
            net_str = f"{net:.0f}%"
            if net > peak_net:
                peak_net, peak_layer = net, layer
        else:
            net_str = "N/A"

        marker = " ← peak" if layer == peak_layer else ""
        print(f"{layer:<8} {rep_str:<12} {rnd_str:<12} {add_str:<12} {net_str:<12}{marker}")

    print("=" * 70)

    if peak_layer is not None:
        print(f"\nPeak causal effect: Layer {peak_layer} with {peak_net:.0f}% net temporal effect")
        print(f"  Replacement: {replace_by_layer[peak_layer].flip_rate*100:.0f}%")
        print(f"  Random baseline: {random_by_layer[peak_layer].flip_rate*100:.0f}%")


def save_results(results: List[PatchingResult], output_path: str, improved: bool = False):
    """Save results to JSON."""
    methods = list(set(r.method for r in results))

    data = {
        'experiment': 'activation_patching',
        'description': 'Causal circuit analysis via activation patching',
        'methods': methods,
        'results': [
            {
                'layer': r.layer,
                'component': r.component,
                'method': r.method,
                'flip_rate': r.flip_rate,
                'mean_prob_change': r.mean_prob_change,
                'n_samples': r.n_samples
            }
            for r in results
        ]
    }

    # Add net effects summary for improved mode
    if improved and len(methods) > 1:
        replace_by_layer = {r.layer: r for r in results
                           if r.method == 'replacement' and r.component == 'residual'}
        random_by_layer = {r.layer: r for r in results
                          if r.method == 'random' and r.component == 'residual'}

        net_effects = {}
        for layer in replace_by_layer:
            if layer in random_by_layer:
                net_effects[str(layer)] = {
                    'replacement': replace_by_layer[layer].flip_rate,
                    'random': random_by_layer[layer].flip_rate,
                    'net_effect': replace_by_layer[layer].flip_rate - random_by_layer[layer].flip_rate
                }
        data['net_effects_residual'] = net_effects

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    print("=" * 70)
    print("ACTIVATION PATCHING EXPERIMENT")
    if args.improved:
        print("Mode: Extended (replacement + random + addition)")
    else:
        print("Mode: Standard (replacement only)")
    print("=" * 70 + "\n")

    # Load model and probe
    model, tokenizer, probe = load_model_and_probe()

    # Load dataset
    pairs = load_dataset()
    print(f"Loaded {len(pairs)} pairs")

    test_pairs = pairs[:args.n_pairs]
    print(f"Using {len(test_pairs)} pairs for experiment\n")

    # Configure methods
    if args.improved:
        methods = ['replacement', 'random', 'addition']
        print(f"Methods: {', '.join(methods)}")
    else:
        methods = ['replacement']

    # Run experiment
    layers = list(range(12))
    components = ['residual', 'attn', 'mlp']

    print("Running activation patching...")
    results = run_patching_experiment(
        model, tokenizer, probe, test_pairs,
        layers=layers, components=components,
        methods=methods, probe_layer=8
    )

    # Print results
    print_results_table(results, improved=args.improved)

    # Save results
    output_dir = Path('results/activation_patching')
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.improved:
        save_results(results, output_dir / 'patching_results_extended.json', improved=True)
        plot_results(results, output_dir / 'patching_results.png')
        plot_method_comparison(results, output_dir / 'patching_comparison.png')
    else:
        save_results(results, output_dir / 'patching_results.json', improved=False)
        plot_results(results, output_dir / 'patching_results.png')

    print("\nExperiment complete!")
    return results


if __name__ == '__main__':
    main()
