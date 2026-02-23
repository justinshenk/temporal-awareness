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

Ablation (--ablation flag):
- ablation: Zero out activation to test causal necessity

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
    parser.add_argument(
        '--ablation', action='store_true',
        help='Run ablation (necessity) experiment testing both temporal directions'
    )
    parser.add_argument(
        '--probe-layer', type=int, default=8,
        help='Layer to use for probe classification (default: 8)'
    )
    parser.add_argument(
        '--behavioral', action='store_true',
        help='Use behavioral measure (token probabilities) instead of probe'
    )
    parser.add_argument(
        '--blocking', action='store_true',
        help='Run causal blocking experiments testing track interaction'
    )
    parser.add_argument(
        '--dose-response', action='store_true',
        help='Run dose-response analysis with graded interventions (Phase 3)'
    )
    parser.add_argument(
        '--alpha-steps', type=int, default=7,
        help='Number of alpha values to test for dose-response (default: 7)'
    )
    parser.add_argument(
        '--dose-layers', type=str, default='1,6,7',
        help='Comma-separated list of layers to test for dose-response (default: 1,6,7)'
    )
    parser.add_argument(
        '--cross-track', action='store_true',
        help='Include cross-track dose-response analysis (Experiment 3)'
    )
    parser.add_argument(
        '--cross-pair', action='store_true',
        help='Run cross-pair interchange experiment (Phase 4)'
    )
    parser.add_argument(
        '--cross-pair-layers', type=str, default='1,6,7',
        help='Comma-separated list of layers to test for cross-pair (default: 1,6,7)'
    )
    parser.add_argument(
        '--cross-pair-sample', type=int, default=None,
        help='Sample N pairs for cross-pair matrix (default: use all pairs)'
    )
    parser.add_argument(
        '--average-direction', action='store_true',
        help='Also test average direction patching in cross-pair experiment'
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


@dataclass
class AblationResult:
    """Results from ablation experiment testing both directions."""
    layer: int
    component: str
    longterm_disruption_rate: float  # Long-term flipped to immediate
    immediate_disruption_rate: float  # Immediate flipped to long-term
    mean_disruption_rate: float  # Average of both
    n_samples_lt: int
    n_samples_im: int


@dataclass
class BlockingResult:
    """Results from causal blocking experiment."""
    experiment: str  # e.g., 'ablate_L1_test_LT', 'double_ablation', 'conflicting_patch'
    ablated_layers: List[int]
    target_direction: str  # 'immediate' or 'longterm'
    accuracy: float  # Classification accuracy after intervention
    baseline_accuracy: float  # Without intervention
    accuracy_drop: float  # baseline - accuracy
    n_samples: int
    details: Optional[Dict] = None


@dataclass
class DoseResponseResult:
    """Results from dose-response experiment with graded interventions."""
    layer: int
    component: str
    method: str  # 'ablation' or 'replacement'
    direction: str  # 'LT_to_IM' or 'IM_to_LT' or 'patching'
    alpha_values: List[float]
    flip_rates: List[float]  # Flip rate at each alpha
    prob_changes: List[float]  # Mean probability change at each alpha
    threshold_alpha: Optional[float]  # Alpha where flip rate crosses 50%
    n_samples: int


@dataclass
class CrossPairResult:
    """Results from cross-pair interchange experiment."""
    source_pair_idx: int
    target_pair_idx: int
    layer: int
    component: str
    flipped: bool
    prob_change: float
    source_category: Optional[str] = None
    target_category: Optional[str] = None


@dataclass
class CrossPairSummary:
    """Summary statistics for cross-pair interchange experiment."""
    layer: int
    component: str
    same_pair_flip_rate: float
    cross_pair_flip_rate: float
    universality_score: float  # cross_pair / same_pair (1.0 = fully universal)
    n_same_pair: int
    n_cross_pair: int
    within_category_flip_rate: Optional[float] = None
    cross_category_flip_rate: Optional[float] = None


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


def load_model_and_probe(probe_layer: int = 8):
    """Load GPT-2 and trained temporal probe at specified layer."""
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()

    # Load probe for specified layer
    probe_path = Path(f'results/checkpoints/temporal_caa_layer_{probe_layer}_probe.pkl')
    if not probe_path.exists():
        # Try alternate locations
        for alt_path in [
            f'results/probes/temporal_caa_layer_{probe_layer}_probe.pkl',
            f'research/probes/temporal_caa_layer_{probe_layer}_probe.pkl'
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
        print(f"No trained probe found for layer {probe_layer}, training a quick one...")
        probe = train_quick_probe(model, tokenizer, layer=probe_layer)

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


# Temporal tokens for behavioral measure
IMMEDIATE_TOKENS = ['now', 'today', 'immediately', 'soon', 'current', 'present', 
                    'urgently', 'right', 'quick', 'fast', 'instant', 'moment']
LONGTERM_TOKENS = ['future', 'eventually', 'years', 'decade', 'long-term', 'planning',
                   'later', 'sustainable', 'lasting', 'permanent', 'strategic', 'legacy']


def get_temporal_token_ids(tokenizer):
    """Get token IDs for immediate and long-term temporal words."""
    immediate_ids = []
    longterm_ids = []
    
    for word in IMMEDIATE_TOKENS:
        # Try different capitalizations and prefixes
        for variant in [word, word.capitalize(), ' ' + word, ' ' + word.capitalize()]:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            immediate_ids.extend(tokens)
    
    for word in LONGTERM_TOKENS:
        for variant in [word, word.capitalize(), ' ' + word, ' ' + word.capitalize()]:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            longterm_ids.extend(tokens)
    
    return list(set(immediate_ids)), list(set(longterm_ids))


def measure_temporal_probability(logits, immediate_ids: List[int], longterm_ids: List[int]):
    """
    Measure total probability mass on immediate vs long-term tokens.
    
    Returns:
        dict with immediate_prob, longterm_prob, and ratio
    """
    probs = torch.softmax(logits, dim=-1)
    
    immediate_prob = probs[immediate_ids].sum().item()
    longterm_prob = probs[longterm_ids].sum().item()
    
    # Log ratio (positive = more long-term, negative = more immediate)
    if immediate_prob > 0 and longterm_prob > 0:
        log_ratio = np.log(longterm_prob / immediate_prob)
    else:
        log_ratio = 0.0
    
    return {
        'immediate_prob': immediate_prob,
        'longterm_prob': longterm_prob,
        'log_ratio': log_ratio
    }


@dataclass
class BehavioralAblationResult:
    """Results from behavioral ablation experiment."""
    layer: int
    component: str
    # For long-term inputs: did ablation shift probability toward immediate?
    lt_prob_shift_to_immediate: float  # Average decrease in LT prob / increase in IM prob
    # For immediate inputs: did ablation shift probability toward long-term?
    im_prob_shift_to_longterm: float  # Average decrease in IM prob / increase in LT prob
    n_samples_lt: int
    n_samples_im: int


def run_ablation_behavioral(model, tokenizer, pairs: List[dict],
                            layers: List[int], components: List[str]) -> List[BehavioralAblationResult]:
    """
    Run ablation experiment using behavioral measure (token probabilities).
    
    Instead of using a probe, measures actual model output probability shifts
    for temporal tokens.
    """
    results = []
    total_iterations = len(layers) * len(components)
    
    # Get temporal token IDs
    immediate_ids, longterm_ids = get_temporal_token_ids(tokenizer)
    print(f"Tracking {len(immediate_ids)} immediate token variants, {len(longterm_ids)} long-term token variants")
    
    with tqdm(total=total_iterations, desc="Behavioral Ablation") as pbar:
        for layer in layers:
            for component in components:
                lt_shifts = []  # Shifts for long-term inputs
                im_shifts = []  # Shifts for immediate inputs
                
                for pair in pairs:
                    immediate_text = pair['question'] + pair['immediate']
                    longterm_text = pair['question'] + pair['long_term']
                    
                    # Test on LONG-TERM inputs
                    # Baseline: measure prob distribution on long-term input
                    inputs_lt = tokenizer(longterm_text, return_tensors='pt')
                    with torch.no_grad():
                        baseline_outputs_lt = model(inputs_lt['input_ids'])
                    baseline_probs_lt = measure_temporal_probability(
                        baseline_outputs_lt.logits[0, -1, :], immediate_ids, longterm_ids
                    )
                    
                    # Get cache for ablation
                    lt_cache_result = get_activations_with_cache(model, tokenizer, longterm_text, [layer])
                    
                    # Run with ablation
                    ablated_outputs_lt = run_with_patching(
                        model, tokenizer, longterm_text,
                        lt_cache_result['cache'], layer, component,
                        method='ablation'
                    )
                    ablated_probs_lt = measure_temporal_probability(
                        ablated_outputs_lt.logits[0, -1, :], immediate_ids, longterm_ids
                    )
                    
                    # Calculate shift: decrease in log_ratio means shift toward immediate
                    lt_shift = baseline_probs_lt['log_ratio'] - ablated_probs_lt['log_ratio']
                    lt_shifts.append(lt_shift)
                    
                    # Test on IMMEDIATE inputs
                    inputs_im = tokenizer(immediate_text, return_tensors='pt')
                    with torch.no_grad():
                        baseline_outputs_im = model(inputs_im['input_ids'])
                    baseline_probs_im = measure_temporal_probability(
                        baseline_outputs_im.logits[0, -1, :], immediate_ids, longterm_ids
                    )
                    
                    im_cache_result = get_activations_with_cache(model, tokenizer, immediate_text, [layer])
                    
                    ablated_outputs_im = run_with_patching(
                        model, tokenizer, immediate_text,
                        im_cache_result['cache'], layer, component,
                        method='ablation'
                    )
                    ablated_probs_im = measure_temporal_probability(
                        ablated_outputs_im.logits[0, -1, :], immediate_ids, longterm_ids
                    )
                    
                    # Calculate shift: increase in log_ratio means shift toward long-term
                    im_shift = ablated_probs_im['log_ratio'] - baseline_probs_im['log_ratio']
                    im_shifts.append(im_shift)
                
                results.append(BehavioralAblationResult(
                    layer=layer,
                    component=component,
                    lt_prob_shift_to_immediate=np.mean(lt_shifts) if lt_shifts else 0,
                    im_prob_shift_to_longterm=np.mean(im_shifts) if im_shifts else 0,
                    n_samples_lt=len(lt_shifts),
                    n_samples_im=len(im_shifts)
                ))
                
                pbar.update(1)
                pbar.set_postfix(layer=layer, comp=component[:3])
    
    return results


def print_behavioral_results(results: List[BehavioralAblationResult]):
    """Print behavioral ablation results table."""
    print("\n" + "=" * 80)
    print("BEHAVIORAL ABLATION RESULTS - Residual Stream")
    print("(Positive = ablation shifted probability in expected direction)")
    print("=" * 80)
    print(f"{'Layer':<8} {'LT→IM shift':<15} {'IM→LT shift':<15} {'Asymmetry':<15}")
    print("-" * 80)
    
    residual_results = [r for r in results if r.component == 'residual']
    residual_results.sort(key=lambda x: x.layer)
    
    for r in residual_results:
        lt_str = f"{r.lt_prob_shift_to_immediate:.3f}"
        im_str = f"{r.im_prob_shift_to_longterm:.3f}"
        # Asymmetry: difference between the two shifts
        asymmetry = r.lt_prob_shift_to_immediate - r.im_prob_shift_to_longterm
        asym_str = f"{asymmetry:.3f}"
        
        # Mark layers with strong asymmetry
        marker = ""
        if r.lt_prob_shift_to_immediate > 0.1 and r.im_prob_shift_to_longterm < 0.05:
            marker = " ← LT-specific"
        elif r.im_prob_shift_to_longterm > 0.1 and r.lt_prob_shift_to_immediate < 0.05:
            marker = " ← IM-specific"
        
        print(f"{r.layer:<8} {lt_str:<15} {im_str:<15} {asym_str:<15}{marker}")
    
    print("=" * 80)


def save_behavioral_results(results: List[BehavioralAblationResult], output_path: str):
    """Save behavioral results to JSON."""
    data = {
        'experiment': 'behavioral_ablation',
        'description': 'Ablation using token probability shifts instead of probe',
        'results': [
            {
                'layer': r.layer,
                'component': r.component,
                'lt_prob_shift_to_immediate': r.lt_prob_shift_to_immediate,
                'im_prob_shift_to_longterm': r.im_prob_shift_to_longterm,
                'n_samples_lt': r.n_samples_lt,
                'n_samples_im': r.n_samples_im
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


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
                      target_cache: Optional[Dict[str, torch.Tensor]] = None,
                      alpha: float = 1.0):
    """
    Run forward pass with patched activations.

    Args:
        method: 'replacement' | 'random' | 'addition' | 'ablation'
            - replacement: Replace with source (immediate) activation
            - random: Replace with Gaussian noise matching source statistics
            - addition: Add (source - target) difference
            - ablation: Zero out activation (for necessity testing)
        target_cache: Required for 'addition' method
        alpha: Intervention strength (0.0 = no intervention, 1.0 = full intervention)
               For dose-response analysis, use values between 0 and 1.
               - ablation: patched = (1 - alpha) * original + alpha * 0
               - replacement: patched = (1 - alpha) * original + alpha * source
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

        # Apply patching method at final token position with alpha scaling
        if method == 'replacement':
            # Interpolate: (1-alpha)*original + alpha*source
            patched[:, -1:, :] = (1 - alpha) * out_tensor[:, -1:, :] + alpha * source_final
        elif method == 'random':
            noise = torch.randn_like(source_final)
            random_target = noise * source_final.std() + source_final.mean()
            patched[:, -1:, :] = (1 - alpha) * out_tensor[:, -1:, :] + alpha * random_target
        elif method == 'addition':
            diff = source_final - target_final
            patched[:, -1:, :] = out_tensor[:, -1:, :] + alpha * diff
        elif method == 'ablation':
            # Interpolate toward zero: (1-alpha)*original + alpha*0 = (1-alpha)*original
            patched[:, -1:, :] = (1 - alpha) * out_tensor[:, -1:, :]

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


def run_with_multi_patching(model, tokenizer, text, 
                            interventions: List[Dict],
                            patch_caches: Optional[Dict[str, Dict]] = None):
    """
    Run forward pass with multiple interventions (ablations or patches) at different layers.
    
    Args:
        interventions: List of dicts, each with:
            - 'layer': int
            - 'component': str ('residual', 'attn', 'mlp')
            - 'method': str ('ablation', 'replacement')
            - 'source_key': str (optional, key in patch_caches for replacement)
        patch_caches: Dict mapping source keys to activation caches
    
    Returns:
        Model outputs with all interventions applied
    """
    hooks = []
    
    for intervention in interventions:
        layer = intervention['layer']
        component = intervention['component']
        method = intervention['method']
        
        # Get source activation if doing replacement
        source_final = None
        if method == 'replacement' and patch_caches:
            source_key = intervention.get('source_key', 'default')
            if source_key in patch_caches:
                patch_key = f'layer_{layer}_{component}'
                if patch_key in patch_caches[source_key]:
                    source_final = patch_caches[source_key][patch_key][:, -1:, :]
        
        def make_patch_hook(method, source_final):
            def patch_hook(module, input, output):
                if isinstance(output, tuple):
                    out_tensor = output[0]
                    patched = out_tensor.clone()
                else:
                    out_tensor = output
                    patched = output.clone()
                
                if method == 'ablation':
                    patched[:, -1:, :] = 0
                elif method == 'replacement' and source_final is not None:
                    patched[:, -1:, :] = source_final
                
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched
            return patch_hook
        
        # Register hook
        block = model.transformer.h[layer]
        if component == 'residual':
            hook = block.register_forward_hook(make_patch_hook(method, source_final))
        elif component == 'attn':
            hook = block.attn.register_forward_hook(make_patch_hook(method, source_final))
        elif component == 'mlp':
            hook = block.mlp.register_forward_hook(make_patch_hook(method, source_final))
        hooks.append(hook)
    
    # Forward pass with all hooks
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs['input_ids'], output_hidden_states=True)
    
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    return outputs


def run_blocking_experiments(model, tokenizer, probe, pairs: List[dict],
                             probe_layer: int = 8) -> List[BlockingResult]:
    """
    Run causal blocking experiments to test track interaction.
    
    Experiments:
    1. Ablate L1 (immediate track), test LT classification
    2. Ablate L6 (long-term track), test IM classification  
    3. Double ablation (L1 + L6)
    4. Conflicting patches
    """
    results = []
    
    # Define track layers based on Phase 1 findings
    IMMEDIATE_TRACK_LAYER = 1  # Strong immediate encoding
    LONGTERM_TRACK_LAYER = 6   # Strong long-term encoding
    
    print("Running causal blocking experiments...")
    print(f"  Immediate track layer: {IMMEDIATE_TRACK_LAYER}")
    print(f"  Long-term track layer: {LONGTERM_TRACK_LAYER}")
    print()
    
    # First, establish baselines
    print("Establishing baselines...")
    lt_correct_baseline = 0
    im_correct_baseline = 0
    n_lt = 0
    n_im = 0
    
    for pair in tqdm(pairs, desc="Baseline"):
        immediate_text = pair['question'] + pair['immediate']
        longterm_text = pair['question'] + pair['long_term']
        
        # Test long-term baseline
        lt_result = get_activations_with_cache(model, tokenizer, longterm_text, [])
        lt_act = lt_result['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
        lt_pred = probe.predict([lt_act])[0]
        if lt_pred == 1:  # Correctly predicts long-term
            lt_correct_baseline += 1
        n_lt += 1
        
        # Test immediate baseline
        im_result = get_activations_with_cache(model, tokenizer, immediate_text, [])
        im_act = im_result['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
        im_pred = probe.predict([im_act])[0]
        if im_pred == 0:  # Correctly predicts immediate
            im_correct_baseline += 1
        n_im += 1
    
    lt_baseline_acc = lt_correct_baseline / n_lt
    im_baseline_acc = im_correct_baseline / n_im
    print(f"  LT baseline accuracy: {lt_baseline_acc:.1%}")
    print(f"  IM baseline accuracy: {im_baseline_acc:.1%}")
    print()
    
    # Experiment 1: Ablate L1 (immediate track), test LT classification
    print("Experiment 1: Ablate immediate track (L1), test long-term...")
    lt_correct = 0
    for pair in tqdm(pairs, desc="Exp1"):
        longterm_text = pair['question'] + pair['long_term']
        
        interventions = [{'layer': IMMEDIATE_TRACK_LAYER, 'component': 'residual', 'method': 'ablation'}]
        outputs = run_with_multi_patching(model, tokenizer, longterm_text, interventions)
        
        act = outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
        pred = probe.predict([act])[0]
        if pred == 1:  # Still correctly predicts long-term
            lt_correct += 1
    
    exp1_acc = lt_correct / n_lt
    results.append(BlockingResult(
        experiment='ablate_immediate_track_test_LT',
        ablated_layers=[IMMEDIATE_TRACK_LAYER],
        target_direction='longterm',
        accuracy=exp1_acc,
        baseline_accuracy=lt_baseline_acc,
        accuracy_drop=lt_baseline_acc - exp1_acc,
        n_samples=n_lt,
        details={'immediate_track_layer': IMMEDIATE_TRACK_LAYER}
    ))
    print(f"  LT accuracy after L1 ablation: {exp1_acc:.1%} (drop: {lt_baseline_acc - exp1_acc:.1%})")
    
    # Experiment 2: Ablate L6 (long-term track), test IM classification
    print("\nExperiment 2: Ablate long-term track (L6), test immediate...")
    im_correct = 0
    for pair in tqdm(pairs, desc="Exp2"):
        immediate_text = pair['question'] + pair['immediate']
        
        interventions = [{'layer': LONGTERM_TRACK_LAYER, 'component': 'residual', 'method': 'ablation'}]
        outputs = run_with_multi_patching(model, tokenizer, immediate_text, interventions)
        
        act = outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
        pred = probe.predict([act])[0]
        if pred == 0:  # Still correctly predicts immediate
            im_correct += 1
    
    exp2_acc = im_correct / n_im
    results.append(BlockingResult(
        experiment='ablate_longterm_track_test_IM',
        ablated_layers=[LONGTERM_TRACK_LAYER],
        target_direction='immediate',
        accuracy=exp2_acc,
        baseline_accuracy=im_baseline_acc,
        accuracy_drop=im_baseline_acc - exp2_acc,
        n_samples=n_im,
        details={'longterm_track_layer': LONGTERM_TRACK_LAYER}
    ))
    print(f"  IM accuracy after L6 ablation: {exp2_acc:.1%} (drop: {im_baseline_acc - exp2_acc:.1%})")
    
    # Experiment 3: Double ablation (both tracks)
    print("\nExperiment 3: Double ablation (L1 + L6)...")
    lt_correct_double = 0
    im_correct_double = 0
    
    for pair in tqdm(pairs, desc="Exp3"):
        immediate_text = pair['question'] + pair['immediate']
        longterm_text = pair['question'] + pair['long_term']
        
        interventions = [
            {'layer': IMMEDIATE_TRACK_LAYER, 'component': 'residual', 'method': 'ablation'},
            {'layer': LONGTERM_TRACK_LAYER, 'component': 'residual', 'method': 'ablation'}
        ]
        
        # Test LT with double ablation
        outputs_lt = run_with_multi_patching(model, tokenizer, longterm_text, interventions)
        act_lt = outputs_lt.hidden_states[probe_layer][0, -1, :].cpu().numpy()
        if probe.predict([act_lt])[0] == 1:
            lt_correct_double += 1
        
        # Test IM with double ablation
        outputs_im = run_with_multi_patching(model, tokenizer, immediate_text, interventions)
        act_im = outputs_im.hidden_states[probe_layer][0, -1, :].cpu().numpy()
        if probe.predict([act_im])[0] == 0:
            im_correct_double += 1
    
    exp3_lt_acc = lt_correct_double / n_lt
    exp3_im_acc = im_correct_double / n_im
    results.append(BlockingResult(
        experiment='double_ablation_test_LT',
        ablated_layers=[IMMEDIATE_TRACK_LAYER, LONGTERM_TRACK_LAYER],
        target_direction='longterm',
        accuracy=exp3_lt_acc,
        baseline_accuracy=lt_baseline_acc,
        accuracy_drop=lt_baseline_acc - exp3_lt_acc,
        n_samples=n_lt
    ))
    results.append(BlockingResult(
        experiment='double_ablation_test_IM',
        ablated_layers=[IMMEDIATE_TRACK_LAYER, LONGTERM_TRACK_LAYER],
        target_direction='immediate',
        accuracy=exp3_im_acc,
        baseline_accuracy=im_baseline_acc,
        accuracy_drop=im_baseline_acc - exp3_im_acc,
        n_samples=n_im
    ))
    print(f"  LT accuracy after double ablation: {exp3_lt_acc:.1%} (drop: {lt_baseline_acc - exp3_lt_acc:.1%})")
    print(f"  IM accuracy after double ablation: {exp3_im_acc:.1%} (drop: {im_baseline_acc - exp3_im_acc:.1%})")
    
    # Experiment 4: Conflicting patches
    # Patch L1 with IMMEDIATE activation, L6 with LONG-TERM activation on same input
    print("\nExperiment 4: Conflicting patches...")
    conflict_predicts_lt = 0
    conflict_predicts_im = 0
    
    for pair in tqdm(pairs, desc="Exp4"):
        immediate_text = pair['question'] + pair['immediate']
        longterm_text = pair['question'] + pair['long_term']
        
        # Cache activations from both directions
        im_cache = get_activations_with_cache(model, tokenizer, immediate_text, 
                                               [IMMEDIATE_TRACK_LAYER, LONGTERM_TRACK_LAYER])
        lt_cache = get_activations_with_cache(model, tokenizer, longterm_text,
                                               [IMMEDIATE_TRACK_LAYER, LONGTERM_TRACK_LAYER])
        
        # Apply conflicting patches to a NEUTRAL input (use immediate text as base)
        # Patch L1 with IMMEDIATE (reinforce), L6 with LONG-TERM (conflict)
        patch_caches = {'immediate': im_cache['cache'], 'longterm': lt_cache['cache']}
        interventions = [
            {'layer': IMMEDIATE_TRACK_LAYER, 'component': 'residual', 
             'method': 'replacement', 'source_key': 'immediate'},
            {'layer': LONGTERM_TRACK_LAYER, 'component': 'residual',
             'method': 'replacement', 'source_key': 'longterm'}
        ]
        
        outputs = run_with_multi_patching(model, tokenizer, immediate_text, 
                                          interventions, patch_caches)
        act = outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
        pred = probe.predict([act])[0]
        
        if pred == 1:
            conflict_predicts_lt += 1
        else:
            conflict_predicts_im += 1
    
    # Calculate which track "wins"
    total_conflict = conflict_predicts_lt + conflict_predicts_im
    lt_win_rate = conflict_predicts_lt / total_conflict if total_conflict > 0 else 0
    
    results.append(BlockingResult(
        experiment='conflicting_patches',
        ablated_layers=[IMMEDIATE_TRACK_LAYER, LONGTERM_TRACK_LAYER],  # Not ablation, but intervention layers
        target_direction='conflict',
        accuracy=lt_win_rate,  # Rate at which LT patch "wins"
        baseline_accuracy=0.5,  # Random baseline
        accuracy_drop=0.5 - lt_win_rate,  # How far from random
        n_samples=total_conflict,
        details={
            'lt_wins': conflict_predicts_lt,
            'im_wins': conflict_predicts_im,
            'interpretation': 'LT track dominates' if lt_win_rate > 0.6 else 
                            ('IM track dominates' if lt_win_rate < 0.4 else 'Mixed/balanced')
        }
    ))
    print(f"  Conflicting patches: LT wins {conflict_predicts_lt}/{total_conflict} ({lt_win_rate:.1%})")
    print(f"  Interpretation: {results[-1].details['interpretation']}")
    
    return results


def print_blocking_results(results: List[BlockingResult]):
    """Print blocking experiment results table."""
    print("\n" + "=" * 90)
    print("CAUSAL BLOCKING RESULTS")
    print("=" * 90)
    print(f"{'Experiment':<35} {'Target':<10} {'Baseline':<10} {'After':<10} {'Drop':<10} {'N':<6}")
    print("-" * 90)
    
    for r in results:
        exp_name = r.experiment[:33] + '..' if len(r.experiment) > 35 else r.experiment
        print(f"{exp_name:<35} {r.target_direction:<10} {r.baseline_accuracy:.1%}      "
              f"{r.accuracy:.1%}      {r.accuracy_drop:.1%}      {r.n_samples:<6}")
    
    print("=" * 90)
    
    # Interpretation
    print("\nINTERPRETATION:")
    
    # Find key results
    exp1 = next((r for r in results if r.experiment == 'ablate_immediate_track_test_LT'), None)
    exp2 = next((r for r in results if r.experiment == 'ablate_longterm_track_test_IM'), None)
    exp4 = next((r for r in results if r.experiment == 'conflicting_patches'), None)
    
    if exp1 and exp2:
        if exp1.accuracy_drop < 0.1 and exp2.accuracy_drop < 0.1:
            print("  → INDEPENDENT TRACKS: Each track can function without the other")
        elif exp1.accuracy_drop > 0.2 and exp2.accuracy_drop < 0.1:
            print("  → LT TRACK DEPENDS ON IM TRACK: Long-term encoding needs immediate track")
        elif exp1.accuracy_drop < 0.1 and exp2.accuracy_drop > 0.2:
            print("  → IM TRACK DEPENDS ON LT TRACK: Immediate encoding needs long-term track")
        else:
            print("  → BIDIRECTIONAL DEPENDENCY: Tracks interact with each other")
    
    if exp4 and exp4.details:
        print(f"  → DOMINANCE: {exp4.details['interpretation']}")


def save_blocking_results(results: List[BlockingResult], output_path: str):
    """Save blocking results to JSON."""
    data = {
        'experiment': 'causal_blocking',
        'description': 'Testing whether temporal encoding tracks interact or are independent',
        'results': [
            {
                'experiment': r.experiment,
                'ablated_layers': r.ablated_layers,
                'target_direction': r.target_direction,
                'accuracy': r.accuracy,
                'baseline_accuracy': r.baseline_accuracy,
                'accuracy_drop': r.accuracy_drop,
                'n_samples': r.n_samples,
                'details': r.details
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


def get_alpha_values(n_steps: int = 7) -> List[float]:
    """Generate alpha values for dose-response analysis."""
    if n_steps == 7:
        # Standard set from the plan
        return [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    else:
        # Generate evenly spaced values
        return [round(i / (n_steps - 1), 2) for i in range(n_steps)]


def find_threshold_alpha(alpha_values: List[float], flip_rates: List[float], 
                         threshold: float = 0.5) -> Optional[float]:
    """
    Find the alpha value where flip rate crosses the threshold (default 50%).
    Uses linear interpolation between points.
    
    Returns:
        Alpha value at threshold crossing, or None if no crossing found.
    """
    for i in range(len(flip_rates) - 1):
        # Check if threshold is between current and next flip rate
        if (flip_rates[i] <= threshold <= flip_rates[i + 1] or 
            flip_rates[i] >= threshold >= flip_rates[i + 1]):
            # Linear interpolation
            if flip_rates[i + 1] != flip_rates[i]:
                t = (threshold - flip_rates[i]) / (flip_rates[i + 1] - flip_rates[i])
                threshold_alpha = alpha_values[i] + t * (alpha_values[i + 1] - alpha_values[i])
                return round(threshold_alpha, 3)
    return None


def run_dose_response_experiment(model, tokenizer, probe, pairs: List[dict],
                                  layers: List[int], component: str = 'residual',
                                  method: str = 'ablation', alpha_steps: int = 7,
                                  probe_layer: int = 8) -> List[DoseResponseResult]:
    """
    Run dose-response analysis with graded interventions.
    
    Tests both directions for ablation:
    - LT_to_IM: Does graded ablation on LT inputs cause flip to IM?
    - IM_to_LT: Does graded ablation on IM inputs cause flip to LT?
    
    For patching (replacement), tests:
    - Patching LT with IM activation at various strengths
    
    Args:
        layers: Layers to test (e.g., [1, 6, 7] for key track layers)
        component: Which component to intervene on ('residual', 'attn', 'mlp')
        method: 'ablation' or 'replacement'
        alpha_steps: Number of alpha values to test
        probe_layer: Layer to use for probe classification
    
    Returns:
        List of DoseResponseResult for each layer and direction
    """
    results = []
    alpha_values = get_alpha_values(alpha_steps)
    
    print(f"\nDose-Response Analysis: {method.upper()}")
    print(f"Alpha values: {alpha_values}")
    print(f"Layers: {layers}")
    print(f"Component: {component}")
    print()
    
    for layer in tqdm(layers, desc=f"Layers"):
        # For ablation, test both directions
        if method == 'ablation':
            directions = ['LT_to_IM', 'IM_to_LT']
        else:
            # For patching, only test LT->IM direction (patch LT with IM activation)
            directions = ['patching']
        
        for direction in directions:
            flip_rates_by_alpha = []
            prob_changes_by_alpha = []
            
            for alpha in tqdm(alpha_values, desc=f"L{layer} {direction}", leave=False):
                flip_count = 0
                prob_changes = []
                n_valid = 0
                
                for pair in pairs:
                    immediate_text = pair['question'] + pair['immediate']
                    longterm_text = pair['question'] + pair['long_term']
                    
                    if direction == 'LT_to_IM' or direction == 'patching':
                        # Test on long-term inputs
                        test_text = longterm_text
                        expected_pred = 1  # Baseline should predict long-term
                        flipped_pred = 0   # Flip to immediate
                        source_text = immediate_text  # For patching
                    else:  # IM_to_LT
                        # Test on immediate inputs
                        test_text = immediate_text
                        expected_pred = 0  # Baseline should predict immediate
                        flipped_pred = 1   # Flip to long-term
                        source_text = longterm_text  # For patching
                    
                    # Cache activations
                    test_result = get_activations_with_cache(
                        model, tokenizer, test_text, [layer]
                    )
                    baseline_act = test_result['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
                    baseline_pred = probe.predict([baseline_act])[0]
                    baseline_prob = probe.predict_proba([baseline_act])[0, 1]  # P(long-term)
                    
                    # Skip if baseline prediction is wrong
                    if baseline_pred != expected_pred:
                        continue
                    n_valid += 1
                    
                    # For patching method, cache source activations
                    if method == 'replacement':
                        source_result = get_activations_with_cache(
                            model, tokenizer, source_text, [layer]
                        )
                        patch_cache = source_result['cache']
                    else:
                        patch_cache = test_result['cache']
                    
                    # Run with intervention at this alpha
                    patched_outputs = run_with_patching(
                        model, tokenizer, test_text,
                        patch_cache, layer, component,
                        method=method, alpha=alpha
                    )
                    
                    patched_act = patched_outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
                    patched_pred = probe.predict([patched_act])[0]
                    patched_prob = probe.predict_proba([patched_act])[0, 1]
                    
                    if patched_pred == flipped_pred:
                        flip_count += 1
                    
                    # Track probability change
                    if direction == 'LT_to_IM' or direction == 'patching':
                        # For LT inputs, positive prob change = decrease in P(LT)
                        prob_changes.append(baseline_prob - patched_prob)
                    else:
                        # For IM inputs, positive prob change = increase in P(LT)
                        prob_changes.append(patched_prob - baseline_prob)
                
                flip_rate = flip_count / n_valid if n_valid > 0 else 0
                mean_prob_change = np.mean(prob_changes) if prob_changes else 0
                
                flip_rates_by_alpha.append(flip_rate)
                prob_changes_by_alpha.append(mean_prob_change)
            
            # Find threshold alpha
            threshold_alpha = find_threshold_alpha(alpha_values, flip_rates_by_alpha)
            
            results.append(DoseResponseResult(
                layer=layer,
                component=component,
                method=method,
                direction=direction,
                alpha_values=alpha_values,
                flip_rates=flip_rates_by_alpha,
                prob_changes=prob_changes_by_alpha,
                threshold_alpha=threshold_alpha,
                n_samples=n_valid
            ))
            
            # Print summary for this layer/direction
            print(f"  L{layer} {direction}: threshold={threshold_alpha}, "
                  f"max_flip={max(flip_rates_by_alpha):.1%}, n={n_valid}")
    
    return results


def run_cross_track_dose_response(model, tokenizer, probe, pairs: List[dict],
                                   alpha_steps: int = 7, probe_layer: int = 8) -> Dict:
    """
    Phase 3 Experiment 3: Cross-track dose-response analysis.
    
    Tests if partially ablating one track affects the other track's threshold.
    
    Setup:
    - With L1 at 50% ablation, what is L6's threshold?
    - With L6 at 50% ablation, what is L1's threshold?
    
    This reveals whether the tracks interact in a graded way.
    """
    alpha_values = get_alpha_values(alpha_steps)
    
    IMMEDIATE_TRACK = 1  # L1
    LONGTERM_TRACK = 6   # L6
    
    print("\nCross-Track Dose-Response Analysis")
    print("=" * 60)
    
    results = {
        'L6_threshold_with_L1_partial': {},
        'L1_threshold_with_L6_partial': {},
        'alpha_values': alpha_values
    }
    
    # Test several partial ablation levels for the "background" track
    partial_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Experiment: L1 at various partial ablation, test L6's dose-response
    print("\n1. L1 partial ablation → L6 dose-response")
    for l1_alpha in tqdm(partial_alphas, desc="L1 partial levels"):
        l6_flip_rates = []
        
        for l6_alpha in alpha_values:
            flip_count = 0
            n_valid = 0
            
            for pair in pairs:
                longterm_text = pair['question'] + pair['long_term']
                
                # Baseline
                test_result = get_activations_with_cache(
                    model, tokenizer, longterm_text, [IMMEDIATE_TRACK, LONGTERM_TRACK]
                )
                baseline_act = test_result['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
                baseline_pred = probe.predict([baseline_act])[0]
                
                if baseline_pred != 1:  # Skip if baseline wrong
                    continue
                n_valid += 1
                
                # Apply both interventions using multi-patching
                interventions = [
                    {'layer': IMMEDIATE_TRACK, 'component': 'residual', 
                     'method': 'ablation', 'alpha': l1_alpha},
                    {'layer': LONGTERM_TRACK, 'component': 'residual', 
                     'method': 'ablation', 'alpha': l6_alpha}
                ]
                
                outputs = run_with_multi_patching_graded(
                    model, tokenizer, longterm_text, 
                    interventions, test_result['cache']
                )
                
                patched_act = outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
                patched_pred = probe.predict([patched_act])[0]
                
                if patched_pred == 0:  # Flipped to immediate
                    flip_count += 1
            
            flip_rate = flip_count / n_valid if n_valid > 0 else 0
            l6_flip_rates.append(flip_rate)
        
        threshold = find_threshold_alpha(alpha_values, l6_flip_rates)
        results['L6_threshold_with_L1_partial'][f'L1_alpha_{l1_alpha}'] = {
            'flip_rates': l6_flip_rates,
            'threshold': threshold
        }
        print(f"  L1@{l1_alpha:.0%} → L6 threshold: {threshold}")
    
    # Experiment: L6 at various partial ablation, test L1's dose-response
    print("\n2. L6 partial ablation → L1 dose-response")
    for l6_alpha in tqdm(partial_alphas, desc="L6 partial levels"):
        l1_flip_rates = []
        
        for l1_alpha in alpha_values:
            flip_count = 0
            n_valid = 0
            
            for pair in pairs:
                immediate_text = pair['question'] + pair['immediate']
                
                # Baseline
                test_result = get_activations_with_cache(
                    model, tokenizer, immediate_text, [IMMEDIATE_TRACK, LONGTERM_TRACK]
                )
                baseline_act = test_result['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
                baseline_pred = probe.predict([baseline_act])[0]
                
                if baseline_pred != 0:  # Skip if baseline wrong
                    continue
                n_valid += 1
                
                # Apply both interventions
                interventions = [
                    {'layer': IMMEDIATE_TRACK, 'component': 'residual', 
                     'method': 'ablation', 'alpha': l1_alpha},
                    {'layer': LONGTERM_TRACK, 'component': 'residual', 
                     'method': 'ablation', 'alpha': l6_alpha}
                ]
                
                outputs = run_with_multi_patching_graded(
                    model, tokenizer, immediate_text, 
                    interventions, test_result['cache']
                )
                
                patched_act = outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
                patched_pred = probe.predict([patched_act])[0]
                
                if patched_pred == 1:  # Flipped to long-term
                    flip_count += 1
            
            flip_rate = flip_count / n_valid if n_valid > 0 else 0
            l1_flip_rates.append(flip_rate)
        
        threshold = find_threshold_alpha(alpha_values, l1_flip_rates)
        results['L1_threshold_with_L6_partial'][f'L6_alpha_{l6_alpha}'] = {
            'flip_rates': l1_flip_rates,
            'threshold': threshold
        }
        print(f"  L6@{l6_alpha:.0%} → L1 threshold: {threshold}")
    
    return results


def run_with_multi_patching_graded(model, tokenizer, text, 
                                    interventions: List[Dict],
                                    cache: Dict[str, torch.Tensor]):
    """
    Run forward pass with multiple graded interventions at different layers.
    
    Args:
        interventions: List of dicts, each with:
            - 'layer': int
            - 'component': str ('residual', 'attn', 'mlp')
            - 'method': str ('ablation', 'replacement')
            - 'alpha': float (intervention strength)
        cache: Activation cache for the input
    
    Returns:
        Model outputs with all interventions applied
    """
    hooks = []
    
    for intervention in interventions:
        layer = intervention['layer']
        component = intervention['component']
        method = intervention['method']
        alpha = intervention.get('alpha', 1.0)
        
        def make_patch_hook(method, alpha):
            def patch_hook(module, input, output):
                if isinstance(output, tuple):
                    out_tensor = output[0]
                    patched = out_tensor.clone()
                else:
                    out_tensor = output
                    patched = output.clone()
                
                if method == 'ablation':
                    # Graded ablation: (1-alpha)*original
                    patched[:, -1:, :] = (1 - alpha) * out_tensor[:, -1:, :]
                
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched
            return patch_hook
        
        # Register hook
        block = model.transformer.h[layer]
        if component == 'residual':
            hook = block.register_forward_hook(make_patch_hook(method, alpha))
        elif component == 'attn':
            hook = block.attn.register_forward_hook(make_patch_hook(method, alpha))
        elif component == 'mlp':
            hook = block.mlp.register_forward_hook(make_patch_hook(method, alpha))
        hooks.append(hook)
    
    # Forward pass with all hooks
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs['input_ids'], output_hidden_states=True)
    
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    return outputs


# =============================================================================
# PHASE 4: CROSS-PAIR INTERCHANGE EXPERIMENTS
# =============================================================================

# Semantic categories for pairs (based on typical temporal_scope_caa.json structure)
PAIR_CATEGORIES = {
    'health': ['exercise', 'diet', 'sleep', 'medical', 'fitness', 'wellness'],
    'finance': ['investment', 'savings', 'spending', 'career', 'money', 'financial'],
    'relationships': ['family', 'friends', 'dating', 'social', 'marriage', 'partner'],
    'personal': ['habits', 'goals', 'decisions', 'learning', 'growth', 'skills'],
    'work': ['job', 'career', 'business', 'project', 'team', 'management'],
    'lifestyle': ['travel', 'home', 'food', 'entertainment', 'hobby', 'leisure']
}


def categorize_pair(pair: dict) -> str:
    """
    Attempt to categorize a pair based on its question and answers.
    Returns category name or 'other' if no match.
    """
    # Combine question and answers into searchable text
    text = (pair.get('question', '') + ' ' + 
            pair.get('immediate', '') + ' ' + 
            pair.get('long_term', '')).lower()
    
    # Check if pair has a category field
    if 'category' in pair:
        return pair['category']
    
    # Try to match against category keywords
    for category, keywords in PAIR_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                return category
    
    return 'other'


def run_cross_pair_experiment(model, tokenizer, probe, pairs: List[dict],
                               layers: List[int], component: str = 'residual',
                               probe_layer: int = 8,
                               sample_size: Optional[int] = None) -> Tuple[List[CrossPairResult], List[CrossPairSummary]]:
    """
    Run cross-pair interchange experiment (Phase 4).
    
    Tests whether patching activations from Pair A's immediate response
    into Pair B's long-term response causes B to flip to immediate.
    
    If cross-pair patching works as well as same-pair patching,
    the temporal direction is universal.
    
    Args:
        layers: Layers to test (e.g., [1, 6, 7] for key track layers)
        component: Which component to patch
        probe_layer: Layer to use for probe classification
        sample_size: If provided, randomly sample this many pairs
    
    Returns:
        Tuple of (detailed_results, summary_per_layer)
    """
    import random
    
    # Optionally sample pairs
    if sample_size and sample_size < len(pairs):
        test_pairs = random.sample(pairs, sample_size)
        print(f"Sampled {sample_size} pairs from {len(pairs)}")
    else:
        test_pairs = pairs
    
    n_pairs = len(test_pairs)
    
    # Categorize pairs
    pair_categories = [categorize_pair(p) for p in test_pairs]
    print(f"Categories found: {set(pair_categories)}")
    
    print(f"\nCross-Pair Interchange Experiment")
    print(f"Testing {n_pairs} pairs × {n_pairs} pairs = {n_pairs * n_pairs} combinations per layer")
    print(f"Layers: {layers}")
    print()
    
    all_results = []
    summaries = []
    
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        
        # First, cache all activations
        print("Caching activations for all pairs...")
        pair_caches = []
        for pair in tqdm(test_pairs, desc="Caching"):
            immediate_text = pair['question'] + pair['immediate']
            longterm_text = pair['question'] + pair['long_term']
            
            im_cache = get_activations_with_cache(model, tokenizer, immediate_text, [layer])
            lt_cache = get_activations_with_cache(model, tokenizer, longterm_text, [layer])
            
            pair_caches.append({
                'immediate_cache': im_cache['cache'],
                'longterm_cache': lt_cache['cache'],
                'immediate_hidden': im_cache['hidden_states'],
                'longterm_hidden': lt_cache['hidden_states'],
                'longterm_text': longterm_text,
                'immediate_text': immediate_text
            })
        
        # Run cross-pair patching
        same_pair_flips = 0
        same_pair_total = 0
        cross_pair_flips = 0
        cross_pair_total = 0
        within_category_flips = 0
        within_category_total = 0
        cross_category_flips = 0
        cross_category_total = 0
        
        layer_results = []
        
        print("Running cross-pair patching...")
        for source_idx in tqdm(range(n_pairs), desc="Source pairs"):
            source_cache = pair_caches[source_idx]['immediate_cache']
            source_category = pair_categories[source_idx]
            
            for target_idx in range(n_pairs):
                target_cache = pair_caches[target_idx]
                target_text = target_cache['longterm_text']
                target_category = pair_categories[target_idx]
                
                # Get baseline prediction for target
                baseline_act = target_cache['longterm_hidden'][probe_layer][0, -1, :].cpu().numpy()
                baseline_pred = probe.predict([baseline_act])[0]
                baseline_prob = probe.predict_proba([baseline_act])[0, 1]  # P(long-term)
                
                # Skip if baseline is already wrong (should predict long-term)
                if baseline_pred != 1:
                    continue
                
                # Patch source's immediate into target's long-term
                patched_outputs = run_with_patching(
                    model, tokenizer, target_text,
                    source_cache, layer, component,
                    method='replacement'
                )
                
                patched_act = patched_outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
                patched_pred = probe.predict([patched_act])[0]
                patched_prob = probe.predict_proba([patched_act])[0, 1]
                
                flipped = (patched_pred == 0)  # Flipped to immediate
                prob_change = baseline_prob - patched_prob
                
                # Record result
                result = CrossPairResult(
                    source_pair_idx=source_idx,
                    target_pair_idx=target_idx,
                    layer=layer,
                    component=component,
                    flipped=flipped,
                    prob_change=prob_change,
                    source_category=source_category,
                    target_category=target_category
                )
                layer_results.append(result)
                all_results.append(result)
                
                # Update statistics
                if source_idx == target_idx:
                    # Same-pair
                    same_pair_total += 1
                    if flipped:
                        same_pair_flips += 1
                else:
                    # Cross-pair
                    cross_pair_total += 1
                    if flipped:
                        cross_pair_flips += 1
                    
                    # Category analysis
                    if source_category == target_category and source_category != 'other':
                        within_category_total += 1
                        if flipped:
                            within_category_flips += 1
                    elif source_category != 'other' and target_category != 'other':
                        cross_category_total += 1
                        if flipped:
                            cross_category_flips += 1
        
        # Calculate rates
        same_pair_rate = same_pair_flips / same_pair_total if same_pair_total > 0 else 0
        cross_pair_rate = cross_pair_flips / cross_pair_total if cross_pair_total > 0 else 0
        universality_score = cross_pair_rate / same_pair_rate if same_pair_rate > 0 else 0
        
        within_cat_rate = within_category_flips / within_category_total if within_category_total > 0 else None
        cross_cat_rate = cross_category_flips / cross_category_total if cross_category_total > 0 else None
        
        summary = CrossPairSummary(
            layer=layer,
            component=component,
            same_pair_flip_rate=same_pair_rate,
            cross_pair_flip_rate=cross_pair_rate,
            universality_score=universality_score,
            n_same_pair=same_pair_total,
            n_cross_pair=cross_pair_total,
            within_category_flip_rate=within_cat_rate,
            cross_category_flip_rate=cross_cat_rate
        )
        summaries.append(summary)
        
        # Print layer summary
        print(f"  Same-pair flip rate: {same_pair_rate:.1%} (n={same_pair_total})")
        print(f"  Cross-pair flip rate: {cross_pair_rate:.1%} (n={cross_pair_total})")
        print(f"  Universality score: {universality_score:.2f}")
        if within_cat_rate is not None:
            print(f"  Within-category: {within_cat_rate:.1%} (n={within_category_total})")
            print(f"  Cross-category: {cross_cat_rate:.1%} (n={cross_category_total})")
    
    return all_results, summaries


def compute_average_direction(model, tokenizer, pairs: List[dict],
                               layer: int, component: str = 'residual') -> torch.Tensor:
    """
    Compute the average temporal direction across all pairs.
    
    Direction = mean(immediate_activations) - mean(longterm_activations)
    
    This can be used to test if a universal direction works for steering.
    """
    immediate_activations = []
    longterm_activations = []
    
    for pair in tqdm(pairs, desc=f"Computing avg direction L{layer}"):
        immediate_text = pair['question'] + pair['immediate']
        longterm_text = pair['question'] + pair['long_term']
        
        im_cache = get_activations_with_cache(model, tokenizer, immediate_text, [layer])
        lt_cache = get_activations_with_cache(model, tokenizer, longterm_text, [layer])
        
        patch_key = f'layer_{layer}_{component}'
        
        # Get final token activations
        im_act = im_cache['cache'][patch_key][:, -1:, :]  # [1, 1, hidden_dim]
        lt_act = lt_cache['cache'][patch_key][:, -1:, :]
        
        immediate_activations.append(im_act)
        longterm_activations.append(lt_act)
    
    # Stack and compute means
    im_stack = torch.cat(immediate_activations, dim=0)  # [n_pairs, 1, hidden_dim]
    lt_stack = torch.cat(longterm_activations, dim=0)
    
    avg_immediate = im_stack.mean(dim=0)  # [1, hidden_dim]
    avg_longterm = lt_stack.mean(dim=0)
    
    # Direction: immediate - longterm
    avg_direction = avg_immediate - avg_longterm
    
    return avg_direction


def run_average_direction_experiment(model, tokenizer, probe, pairs: List[dict],
                                      layers: List[int], component: str = 'residual',
                                      probe_layer: int = 8) -> Dict:
    """
    Test if the average temporal direction works for steering.
    
    Instead of patching with a specific pair's activations,
    add the average (immediate - longterm) direction to long-term inputs.
    """
    results = {}
    
    for layer in layers:
        print(f"\n--- Average Direction Experiment: Layer {layer} ---")
        
        # Compute average direction
        avg_direction = compute_average_direction(model, tokenizer, pairs, layer, component)
        
        # Test on each pair's long-term input
        flips = 0
        total = 0
        prob_changes = []
        
        for pair in tqdm(pairs, desc=f"Testing avg direction L{layer}"):
            longterm_text = pair['question'] + pair['long_term']
            
            # Get baseline
            lt_cache = get_activations_with_cache(model, tokenizer, longterm_text, [layer])
            baseline_act = lt_cache['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
            baseline_pred = probe.predict([baseline_act])[0]
            baseline_prob = probe.predict_proba([baseline_act])[0, 1]
            
            if baseline_pred != 1:  # Skip wrong baselines
                continue
            total += 1
            
            # Create a "fake" cache that adds the average direction
            patch_key = f'layer_{layer}_{component}'
            original_act = lt_cache['cache'][patch_key][:, -1:, :]
            steered_act = original_act + avg_direction
            
            # Create modified cache
            modified_cache = {patch_key: lt_cache['cache'][patch_key].clone()}
            modified_cache[patch_key][:, -1:, :] = steered_act
            
            # Run with modified activation
            patched_outputs = run_with_patching(
                model, tokenizer, longterm_text,
                modified_cache, layer, component,
                method='replacement'  # This will use the steered activation
            )
            
            patched_act = patched_outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
            patched_pred = probe.predict([patched_act])[0]
            patched_prob = probe.predict_proba([patched_act])[0, 1]
            
            if patched_pred == 0:
                flips += 1
            prob_changes.append(baseline_prob - patched_prob)
        
        flip_rate = flips / total if total > 0 else 0
        mean_prob_change = np.mean(prob_changes) if prob_changes else 0
        
        results[layer] = {
            'flip_rate': flip_rate,
            'mean_prob_change': mean_prob_change,
            'n_samples': total
        }
        
        print(f"  Average direction flip rate: {flip_rate:.1%} (n={total})")
        print(f"  Mean probability change: {mean_prob_change:.3f}")
    
    return results


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


def run_ablation_experiment(model, tokenizer, probe, pairs: List[dict],
                            layers: List[int], components: List[str],
                            probe_layer: int = 8) -> List[AblationResult]:
    """
    Run ablation experiment testing BOTH directions for causal necessity.

    Tests:
    1. Long-term inputs with ablation: Does zeroing cause LT to flip to IM?
    2. Immediate inputs with ablation: Does zeroing cause IM to flip to LT?

    If a layer is necessary for temporal encoding, ablation should disrupt
    classification in both directions.

    Returns:
        List of AblationResult containing disruption rates for each layer/component.
    """
    results = []
    total_iterations = len(layers) * len(components)

    with tqdm(total=total_iterations, desc="Ablation") as pbar:
        for layer in layers:
            for component in components:
                # Track disruption in both directions
                lt_flip_count = 0  # Long-term flipped to immediate
                im_flip_count = 0  # Immediate flipped to long-term
                n_valid_lt = 0
                n_valid_im = 0

                for pair in pairs:
                    immediate_text = pair['question'] + pair['immediate']
                    longterm_text = pair['question'] + pair['long_term']

                    # Test 1: Ablation on LONG-TERM inputs
                    longterm_result = get_activations_with_cache(
                        model, tokenizer, longterm_text, [layer]
                    )
                    lt_baseline_act = longterm_result['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
                    lt_baseline_pred = probe.predict([lt_baseline_act])[0]

                    # Only test if baseline correctly predicts long-term
                    if lt_baseline_pred == 1:
                        n_valid_lt += 1
                        # Run with ablation
                        ablated_outputs = run_with_patching(
                            model, tokenizer, longterm_text,
                            longterm_result['cache'], layer, component,
                            method='ablation'
                        )
                        ablated_act = ablated_outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
                        ablated_pred = probe.predict([ablated_act])[0]
                        if ablated_pred == 0:  # Flipped to immediate
                            lt_flip_count += 1

                    # Test 2: Ablation on IMMEDIATE inputs
                    immediate_result = get_activations_with_cache(
                        model, tokenizer, immediate_text, [layer]
                    )
                    im_baseline_act = immediate_result['hidden_states'][probe_layer][0, -1, :].cpu().numpy()
                    im_baseline_pred = probe.predict([im_baseline_act])[0]

                    # Only test if baseline correctly predicts immediate
                    if im_baseline_pred == 0:
                        n_valid_im += 1
                        # Run with ablation
                        ablated_outputs = run_with_patching(
                            model, tokenizer, immediate_text,
                            immediate_result['cache'], layer, component,
                            method='ablation'
                        )
                        ablated_act = ablated_outputs.hidden_states[probe_layer][0, -1, :].cpu().numpy()
                        ablated_pred = probe.predict([ablated_act])[0]
                        if ablated_pred == 1:  # Flipped to long-term
                            im_flip_count += 1

                # Calculate rates
                lt_rate = lt_flip_count / n_valid_lt if n_valid_lt > 0 else 0
                im_rate = im_flip_count / n_valid_im if n_valid_im > 0 else 0
                mean_rate = (lt_rate + im_rate) / 2 if (n_valid_lt > 0 and n_valid_im > 0) else 0

                results.append(AblationResult(
                    layer=layer,
                    component=component,
                    longterm_disruption_rate=lt_rate,
                    immediate_disruption_rate=im_rate,
                    mean_disruption_rate=mean_rate,
                    n_samples_lt=n_valid_lt,
                    n_samples_im=n_valid_im
                ))

                pbar.update(1)
                pbar.set_postfix(layer=layer, comp=component[:3],
                                lt=f"{lt_rate:.0%}", im=f"{im_rate:.0%}")

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


def plot_ablation_results(ablation_results: List[AblationResult],
                          random_results: Optional[List[PatchingResult]],
                          output_path: str):
    """
    Plot ablation results comparing both directions and random baseline.

    Shows:
    - Long-term disruption rate (LT → IM when ablated)
    - Immediate disruption rate (IM → LT when ablated)
    - Random baseline (for comparison, if provided)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to residual stream only
    residual_results = [r for r in ablation_results if r.component == 'residual']
    residual_results.sort(key=lambda x: x.layer)

    layers = [r.layer for r in residual_results]
    lt_rates = [r.longterm_disruption_rate * 100 for r in residual_results]
    im_rates = [r.immediate_disruption_rate * 100 for r in residual_results]
    mean_rates = [r.mean_disruption_rate * 100 for r in residual_results]

    # Plot ablation results
    ax.plot(layers, lt_rates, 'o-', label='Ablation: LT→IM',
            color='#9b59b6', linewidth=2, markersize=8)
    ax.plot(layers, im_rates, 's-', label='Ablation: IM→LT',
            color='#f39c12', linewidth=2, markersize=8)
    ax.plot(layers, mean_rates, '^-', label='Ablation: Mean',
            color='#1abc9c', linewidth=2, markersize=8)

    # Plot random baseline if provided
    if random_results:
        random_residual = [r for r in random_results
                          if r.method == 'random' and r.component == 'residual']
        random_residual.sort(key=lambda x: x.layer)
        if random_residual:
            random_layers = [r.layer for r in random_residual]
            random_rates = [r.flip_rate * 100 for r in random_residual]
            ax.plot(random_layers, random_rates, 'x--', label='Random Baseline',
                    color='#e74c3c', linewidth=2, markersize=8, alpha=0.7)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Disruption Rate (%)', fontsize=12)
    ax.set_title('Ablation Study: Causal Necessity Testing', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(12))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_dose_response(results: List[DoseResponseResult], output_path: str):
    """
    Plot dose-response curves showing flip rate vs alpha.
    
    Creates a multi-panel figure:
    - Left: Ablation dose-response for key layers (comparing LT→IM and IM→LT)
    - Right: Threshold comparison across layers
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color schemes for layers
    layer_colors = {
        1: '#e74c3c',   # Red for L1 (immediate track)
        6: '#3498db',   # Blue for L6 (long-term track)
        7: '#2ecc71',   # Green for L7 (peak sufficiency)
    }
    
    direction_styles = {
        'LT_to_IM': {'linestyle': '-', 'marker': 'o'},
        'IM_to_LT': {'linestyle': '--', 'marker': 's'},
        'patching': {'linestyle': ':', 'marker': '^'},
    }
    
    ax1 = axes[0]
    
    # Plot dose-response curves
    for r in results:
        if r.layer not in layer_colors:
            continue
        
        color = layer_colors[r.layer]
        style = direction_styles.get(r.direction, {'linestyle': '-', 'marker': 'o'})
        
        label = f"L{r.layer} {r.direction}"
        ax1.plot(r.alpha_values, [fr * 100 for fr in r.flip_rates],
                 linestyle=style['linestyle'], marker=style['marker'],
                 color=color, linewidth=2, markersize=8, label=label)
        
        # Mark threshold if found
        if r.threshold_alpha is not None:
            ax1.axvline(x=r.threshold_alpha, color=color, linestyle=':', alpha=0.5)
            ax1.scatter([r.threshold_alpha], [50], color=color, s=100, zorder=5,
                       marker='*', edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Alpha (Intervention Strength)', fontsize=12)
    ax1.set_ylabel('Flip Rate (%)', fontsize=12)
    ax1.set_title('Dose-Response: Flip Rate vs Intervention Strength', fontsize=14)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)
    ax1.set_xlim(-0.05, 1.05)
    
    # Right panel: Threshold comparison
    ax2 = axes[1]
    
    # Group results by layer and show thresholds
    layers = sorted(set(r.layer for r in results))
    x_positions = list(range(len(layers)))
    
    lt_thresholds = []
    im_thresholds = []
    
    for layer in layers:
        lt_result = next((r for r in results if r.layer == layer and r.direction == 'LT_to_IM'), None)
        im_result = next((r for r in results if r.layer == layer and r.direction == 'IM_to_LT'), None)
        
        lt_thresholds.append(lt_result.threshold_alpha if lt_result and lt_result.threshold_alpha else None)
        im_thresholds.append(im_result.threshold_alpha if im_result and im_result.threshold_alpha else None)
    
    bar_width = 0.35
    
    # Plot bars for thresholds
    lt_vals = [t if t is not None else 0 for t in lt_thresholds]
    im_vals = [t if t is not None else 0 for t in im_thresholds]
    
    bars1 = ax2.bar([x - bar_width/2 for x in x_positions], lt_vals, bar_width, 
                    label='LT→IM threshold', color='#9b59b6', alpha=0.8)
    bars2 = ax2.bar([x + bar_width/2 for x in x_positions], im_vals, bar_width,
                    label='IM→LT threshold', color='#f39c12', alpha=0.8)
    
    # Add "N/A" labels for missing thresholds
    for i, (lt, im) in enumerate(zip(lt_thresholds, im_thresholds)):
        if lt is None:
            ax2.text(i - bar_width/2, 0.05, 'N/A', ha='center', fontsize=9, color='gray')
        if im is None:
            ax2.text(i + bar_width/2, 0.05, 'N/A', ha='center', fontsize=9, color='gray')
    
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Threshold Alpha (50% flip)', fontsize=12)
    ax2.set_title('Threshold Comparison by Layer', fontsize=14)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'L{l}' for l in layers])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_cross_track_dose_response(cross_track_results: Dict, output_path: str):
    """
    Plot cross-track dose-response results showing how partial ablation
    of one track affects the other track's threshold.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    alpha_values = cross_track_results['alpha_values']
    
    # Left panel: L6 threshold with varying L1 partial ablation
    ax1 = axes[0]
    l6_data = cross_track_results['L6_threshold_with_L1_partial']
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(l6_data)))
    for i, (key, data) in enumerate(sorted(l6_data.items())):
        l1_alpha = float(key.split('_')[-1])
        label = f"L1 @ {l1_alpha:.0%}"
        ax1.plot(alpha_values, [fr * 100 for fr in data['flip_rates']],
                 'o-', color=colors[i], linewidth=2, markersize=6, label=label)
    
    ax1.set_xlabel('L6 Alpha (Ablation Strength)', fontsize=12)
    ax1.set_ylabel('LT→IM Flip Rate (%)', fontsize=12)
    ax1.set_title('L6 Dose-Response with L1 Partial Ablation', fontsize=14)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax1.legend(title='L1 Status', loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)
    
    # Right panel: L1 threshold with varying L6 partial ablation
    ax2 = axes[1]
    l1_data = cross_track_results['L1_threshold_with_L6_partial']
    
    for i, (key, data) in enumerate(sorted(l1_data.items())):
        l6_alpha = float(key.split('_')[-1])
        label = f"L6 @ {l6_alpha:.0%}"
        ax2.plot(alpha_values, [fr * 100 for fr in data['flip_rates']],
                 's-', color=colors[i], linewidth=2, markersize=6, label=label)
    
    ax2.set_xlabel('L1 Alpha (Ablation Strength)', fontsize=12)
    ax2.set_ylabel('IM→LT Flip Rate (%)', fontsize=12)
    ax2.set_title('L1 Dose-Response with L6 Partial Ablation', fontsize=14)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(title='L6 Status', loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_dose_response_table(results: List[DoseResponseResult]):
    """Print dose-response results as a formatted table."""
    print("\n" + "=" * 90)
    print("DOSE-RESPONSE ANALYSIS RESULTS")
    print("=" * 90)
    
    # Group by method
    for method in ['ablation', 'replacement']:
        method_results = [r for r in results if r.method == method]
        if not method_results:
            continue
            
        print(f"\n{method.upper()} METHOD:")
        print("-" * 90)
        print(f"{'Layer':<8} {'Direction':<12} {'Threshold α':<14} {'Max Flip':<12} {'Curve Type':<20} {'N':<6}")
        print("-" * 90)
        
        for r in sorted(method_results, key=lambda x: (x.layer, x.direction)):
            threshold_str = f"{r.threshold_alpha:.3f}" if r.threshold_alpha else "N/A"
            max_flip = max(r.flip_rates) * 100
            
            # Determine curve type (sharp vs gradual)
            if r.threshold_alpha is not None:
                # Check steepness around threshold
                idx = min(range(len(r.alpha_values)), 
                         key=lambda i: abs(r.alpha_values[i] - r.threshold_alpha))
                if idx > 0 and idx < len(r.flip_rates) - 1:
                    slope = (r.flip_rates[idx+1] - r.flip_rates[idx-1]) / \
                           (r.alpha_values[idx+1] - r.alpha_values[idx-1])
                    if slope > 2:  # >200% change per unit alpha
                        curve_type = "SHARP (discrete)"
                    elif slope > 0.5:
                        curve_type = "MODERATE"
                    else:
                        curve_type = "GRADUAL (continuous)"
                else:
                    curve_type = "N/A"
            else:
                if max_flip > 80:
                    curve_type = "SATURATED (high)"
                elif max_flip < 20:
                    curve_type = "RESISTANT (low)"
                else:
                    curve_type = "PARTIAL"
            
            print(f"{r.layer:<8} {r.direction:<12} {threshold_str:<14} {max_flip:.0f}%{'':<7} "
                  f"{curve_type:<20} {r.n_samples:<6}")
    
    print("=" * 90)
    
    # Summary interpretation
    print("\nINTERPRETATION:")
    
    # Find key results
    l1_lt = next((r for r in results if r.layer == 1 and r.direction == 'LT_to_IM'), None)
    l6_lt = next((r for r in results if r.layer == 6 and r.direction == 'LT_to_IM'), None)
    l1_im = next((r for r in results if r.layer == 1 and r.direction == 'IM_to_LT'), None)
    l6_im = next((r for r in results if r.layer == 6 and r.direction == 'IM_to_LT'), None)
    
    if l1_lt and l6_lt:
        if l1_lt.threshold_alpha and l6_lt.threshold_alpha:
            if abs(l1_lt.threshold_alpha - l6_lt.threshold_alpha) < 0.15:
                print("  → SIMILAR THRESHOLDS: Both tracks have comparable sensitivity")
            elif l1_lt.threshold_alpha < l6_lt.threshold_alpha:
                print("  → L1 MORE FRAGILE: Immediate track is more easily disrupted")
            else:
                print("  → L6 MORE FRAGILE: Long-term track is more easily disrupted")
    
    if l1_im and l6_im:
        l1_max = max(l1_im.flip_rates) if l1_im else 0
        l6_max = max(l6_im.flip_rates) if l6_im else 0
        if l1_max > l6_max + 0.2:
            print("  → L1 more susceptible to causing IM→LT flips")
        elif l6_max > l1_max + 0.2:
            print("  → L6 more susceptible to causing IM→LT flips")


def save_dose_response_results(results: List[DoseResponseResult], 
                                cross_track_results: Optional[Dict],
                                output_path: str):
    """Save dose-response results to JSON."""
    data = {
        'experiment': 'dose_response',
        'description': 'Graded intervention analysis to find causal thresholds',
        'results': [
            {
                'layer': r.layer,
                'component': r.component,
                'method': r.method,
                'direction': r.direction,
                'alpha_values': r.alpha_values,
                'flip_rates': r.flip_rates,
                'prob_changes': r.prob_changes,
                'threshold_alpha': r.threshold_alpha,
                'n_samples': r.n_samples
            }
            for r in results
        ]
    }
    
    if cross_track_results:
        data['cross_track'] = cross_track_results
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


# =============================================================================
# PHASE 4: CROSS-PAIR VISUALIZATION AND OUTPUT FUNCTIONS
# =============================================================================

def plot_cross_pair_results(summaries: List[CrossPairSummary], 
                            results: List[CrossPairResult],
                            output_path: str):
    """
    Plot cross-pair interchange results.
    
    Creates a multi-panel figure:
    - Left: Same-pair vs cross-pair flip rates by layer
    - Right: Universality score by layer
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    layers = [s.layer for s in summaries]
    same_pair_rates = [s.same_pair_flip_rate * 100 for s in summaries]
    cross_pair_rates = [s.cross_pair_flip_rate * 100 for s in summaries]
    universality_scores = [s.universality_score for s in summaries]
    
    # Left panel: Same-pair vs Cross-pair
    ax1 = axes[0]
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, same_pair_rates, width, label='Same-pair', color='#3498db')
    bars2 = ax1.bar(x + width/2, cross_pair_rates, width, label='Cross-pair', color='#e74c3c')
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Flip Rate (%)', fontsize=12)
    ax1.set_title('Same-Pair vs Cross-Pair Flip Rates', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L{l}' for l in layers])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 105)
    
    # Add value labels
    for bar, val in zip(bars1, same_pair_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.0f}%', ha='center', fontsize=9)
    for bar, val in zip(bars2, cross_pair_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.0f}%', ha='center', fontsize=9)
    
    # Right panel: Universality score
    ax2 = axes[1]
    colors = ['#2ecc71' if u >= 0.9 else '#f1c40f' if u >= 0.7 else '#e74c3c' for u in universality_scores]
    bars = ax2.bar(x, universality_scores, color=colors)
    
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Universality Score', fontsize=12)
    ax2.set_title('Universality Score by Layer\n(Cross-pair / Same-pair)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{l}' for l in layers])
    ax2.axhline(y=0.9, color='#2ecc71', linestyle='--', alpha=0.7, label='Highly universal (0.9)')
    ax2.axhline(y=0.7, color='#f1c40f', linestyle='--', alpha=0.7, label='Mostly universal (0.7)')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.2)
    
    # Add value labels
    for bar, val in zip(bars, universality_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_cross_pair_heatmap(results: List[CrossPairResult], layer: int, 
                            n_pairs: int, output_path: str):
    """
    Plot heatmap of cross-pair flip success (source pair × target pair).
    """
    # Filter results for this layer
    layer_results = [r for r in results if r.layer == layer]
    
    # Create flip matrix
    flip_matrix = np.zeros((n_pairs, n_pairs))
    count_matrix = np.zeros((n_pairs, n_pairs))
    
    for r in layer_results:
        if r.source_pair_idx < n_pairs and r.target_pair_idx < n_pairs:
            count_matrix[r.source_pair_idx, r.target_pair_idx] = 1
            if r.flipped:
                flip_matrix[r.source_pair_idx, r.target_pair_idx] = 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a diverging colormap
    im = ax.imshow(flip_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xlabel('Target Pair Index', fontsize=12)
    ax.set_ylabel('Source Pair Index', fontsize=12)
    ax.set_title(f'Cross-Pair Interchange Success Matrix (Layer {layer})\n'
                 f'Green = Flipped to Immediate, Red = No Flip', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Flipped', fontsize=11)
    
    # Highlight diagonal (same-pair)
    for i in range(min(n_pairs, flip_matrix.shape[0])):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, 
                                    edgecolor='blue', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_category_comparison(summaries: List[CrossPairSummary], output_path: str):
    """
    Plot within-category vs cross-category flip rates.
    """
    # Filter to summaries with category data
    with_categories = [s for s in summaries 
                       if s.within_category_flip_rate is not None 
                       and s.cross_category_flip_rate is not None]
    
    if not with_categories:
        print("No category data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = [s.layer for s in with_categories]
    within_rates = [s.within_category_flip_rate * 100 for s in with_categories]
    cross_rates = [s.cross_category_flip_rate * 100 for s in with_categories]
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, within_rates, width, label='Within-category', color='#9b59b6')
    bars2 = ax.bar(x + width/2, cross_rates, width, label='Cross-category', color='#1abc9c')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Flip Rate (%)', fontsize=12)
    ax.set_title('Semantic Category Effect on Cross-Pair Interchange', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_cross_pair_table(summaries: List[CrossPairSummary], 
                           avg_direction_results: Optional[Dict] = None):
    """Print cross-pair interchange results as a formatted table."""
    print("\n" + "=" * 100)
    print("CROSS-PAIR INTERCHANGE RESULTS (Phase 4)")
    print("=" * 100)
    print(f"{'Layer':<8} {'Same-Pair':<12} {'Cross-Pair':<12} {'Universality':<14} "
          f"{'Within-Cat':<12} {'Cross-Cat':<12} {'Interpretation':<20}")
    print("-" * 100)
    
    for s in sorted(summaries, key=lambda x: x.layer):
        same_str = f"{s.same_pair_flip_rate*100:.1f}%"
        cross_str = f"{s.cross_pair_flip_rate*100:.1f}%"
        univ_str = f"{s.universality_score:.2f}"
        
        within_str = f"{s.within_category_flip_rate*100:.1f}%" if s.within_category_flip_rate else "N/A"
        cross_cat_str = f"{s.cross_category_flip_rate*100:.1f}%" if s.cross_category_flip_rate else "N/A"
        
        # Interpretation
        if s.universality_score >= 0.9:
            interp = "UNIVERSAL"
        elif s.universality_score >= 0.7:
            interp = "Mostly universal"
        elif s.universality_score >= 0.5:
            interp = "Mixed"
        else:
            interp = "Example-specific"
        
        print(f"{s.layer:<8} {same_str:<12} {cross_str:<12} {univ_str:<14} "
              f"{within_str:<12} {cross_cat_str:<12} {interp:<20}")
    
    print("=" * 100)
    
    # Print average direction results if available
    if avg_direction_results:
        print("\nAVERAGE DIRECTION EXPERIMENT:")
        print("-" * 60)
        print(f"{'Layer':<8} {'Flip Rate':<12} {'vs Same-Pair':<15} {'Interpretation':<20}")
        print("-" * 60)
        
        for layer, data in sorted(avg_direction_results.items()):
            flip_str = f"{data['flip_rate']*100:.1f}%"
            
            # Find same-pair rate for comparison
            same_pair = next((s.same_pair_flip_rate for s in summaries if s.layer == layer), None)
            if same_pair:
                ratio = data['flip_rate'] / same_pair if same_pair > 0 else 0
                ratio_str = f"{ratio:.2f}x"
            else:
                ratio_str = "N/A"
            
            # Interpretation
            if data['flip_rate'] >= 0.8:
                interp = "Works well"
            elif data['flip_rate'] >= 0.5:
                interp = "Partial effect"
            else:
                interp = "Weak effect"
            
            print(f"{layer:<8} {flip_str:<12} {ratio_str:<15} {interp:<20}")
        
        print("-" * 60)
    
    # Summary interpretation
    print("\nINTERPRETATION:")
    best = max(summaries, key=lambda s: s.universality_score)
    worst = min(summaries, key=lambda s: s.universality_score)
    
    avg_univ = np.mean([s.universality_score for s in summaries])
    
    if avg_univ >= 0.9:
        print("  → HIGHLY UNIVERSAL: Temporal direction generalizes across all examples")
        print("  → Steering vectors are viable for any input")
    elif avg_univ >= 0.7:
        print("  → MOSTLY UNIVERSAL: Temporal direction mostly generalizes")
        print("  → Steering vectors should work for most inputs")
    elif avg_univ >= 0.5:
        print("  → MIXED: Some universality, but significant example-specific variation")
        print("  → Steering may require context-specific vectors")
    else:
        print("  → EXAMPLE-SPECIFIC: Temporal encoding varies significantly by example")
        print("  → Universal steering vectors may not work well")
    
    print(f"\n  Best layer: L{best.layer} (universality={best.universality_score:.2f})")
    print(f"  Worst layer: L{worst.layer} (universality={worst.universality_score:.2f})")


def save_cross_pair_results(results: List[CrossPairResult],
                            summaries: List[CrossPairSummary],
                            avg_direction_results: Optional[Dict],
                            output_path: str):
    """Save cross-pair results to JSON."""
    data = {
        'experiment': 'cross_pair_interchange',
        'description': 'Testing universality of temporal direction across examples',
        'summaries': [
            {
                'layer': s.layer,
                'component': s.component,
                'same_pair_flip_rate': float(s.same_pair_flip_rate),
                'cross_pair_flip_rate': float(s.cross_pair_flip_rate),
                'universality_score': float(s.universality_score),
                'n_same_pair': int(s.n_same_pair),
                'n_cross_pair': int(s.n_cross_pair),
                'within_category_flip_rate': float(s.within_category_flip_rate) if s.within_category_flip_rate is not None else None,
                'cross_category_flip_rate': float(s.cross_category_flip_rate) if s.cross_category_flip_rate is not None else None
            }
            for s in summaries
        ],
        'detailed_results': [
            {
                'source_pair_idx': int(r.source_pair_idx),
                'target_pair_idx': int(r.target_pair_idx),
                'layer': int(r.layer),
                'component': r.component,
                'flipped': bool(r.flipped),
                'prob_change': float(r.prob_change),
                'source_category': r.source_category,
                'target_category': r.target_category
            }
            for r in results
        ]
    }
    
    if avg_direction_results:
        data['average_direction'] = avg_direction_results
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


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


def print_ablation_table(ablation_results: List[AblationResult],
                         random_results: Optional[List[PatchingResult]] = None):
    """Print ablation results table showing necessity scores (residual stream only)."""
    print("\n" + "=" * 80)
    print("ABLATION (NECESSITY) RESULTS - Residual Stream")
    print("=" * 80)
    print(f"{'Layer':<8} {'LT→IM':<12} {'IM→LT':<12} {'Mean':<12} {'Random':<12} {'Necessity':<12}")
    print("-" * 80)

    # Filter to residual stream
    residual_results = [r for r in ablation_results if r.component == 'residual']
    residual_results.sort(key=lambda x: x.layer)

    # Build random baseline lookup if provided
    random_by_layer = {}
    if random_results:
        random_by_layer = {r.layer: r for r in random_results
                          if r.method == 'random' and r.component == 'residual'}

    peak_layer, peak_necessity = None, -100

    for r in residual_results:
        lt_str = f"{r.longterm_disruption_rate*100:.0f}%"
        im_str = f"{r.immediate_disruption_rate*100:.0f}%"
        mean_str = f"{r.mean_disruption_rate*100:.0f}%"

        rnd = random_by_layer.get(r.layer)
        if rnd:
            rnd_str = f"{rnd.flip_rate*100:.0f}%"
            necessity = (r.mean_disruption_rate - rnd.flip_rate) * 100
            necessity_str = f"{necessity:.0f}%"
            if necessity > peak_necessity:
                peak_necessity, peak_layer = necessity, r.layer
        else:
            rnd_str = "N/A"
            necessity_str = "N/A"

        marker = " ← peak" if r.layer == peak_layer else ""
        print(f"{r.layer:<8} {lt_str:<12} {im_str:<12} {mean_str:<12} {rnd_str:<12} {necessity_str:<12}{marker}")

    print("=" * 80)

    # Summary interpretation
    if peak_layer is not None:
        peak_result = next(r for r in residual_results if r.layer == peak_layer)
        print(f"\nPeak necessity effect: Layer {peak_layer}")
        print(f"  Ablation mean disruption: {peak_result.mean_disruption_rate*100:.0f}%")
        if peak_layer in random_by_layer:
            print(f"  Random baseline: {random_by_layer[peak_layer].flip_rate*100:.0f}%")
            print(f"  Necessity score: {peak_necessity:.0f}%")

        # Interpretation
        if peak_necessity > 30:
            print(f"\n  INTERPRETATION: Layer {peak_layer} is NECESSARY for temporal encoding.")
            print(f"  Zeroing this layer destroys temporal information beyond random disruption.")
        elif peak_necessity > 10:
            print(f"\n  INTERPRETATION: Layer {peak_layer} shows moderate necessity.")
        else:
            print(f"\n  INTERPRETATION: No strong necessity signal detected.")
            print(f"  Temporal info may route through multiple pathways.")


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


def save_ablation_results(ablation_results: List[AblationResult],
                          output_path: str,
                          random_results: Optional[List[PatchingResult]] = None):
    """Save ablation results to JSON."""
    # Build random baseline lookup
    random_by_layer = {}
    if random_results:
        random_by_layer = {r.layer: r.flip_rate for r in random_results
                          if r.method == 'random' and r.component == 'residual'}

    # Calculate necessity scores
    necessity_scores = {}
    for r in ablation_results:
        if r.component == 'residual':
            random_baseline = random_by_layer.get(r.layer, None)
            necessity = None
            if random_baseline is not None:
                necessity = r.mean_disruption_rate - random_baseline
            necessity_scores[str(r.layer)] = {
                'mean_disruption': r.mean_disruption_rate,
                'random_baseline': random_baseline,
                'necessity': necessity
            }

    data = {
        'experiment': 'ablation_necessity',
        'description': 'Causal necessity test via activation zeroing',
        'results': [
            {
                'layer': r.layer,
                'component': r.component,
                'longterm_disruption_rate': r.longterm_disruption_rate,
                'immediate_disruption_rate': r.immediate_disruption_rate,
                'mean_disruption_rate': r.mean_disruption_rate,
                'n_samples_lt': r.n_samples_lt,
                'n_samples_im': r.n_samples_im
            }
            for r in ablation_results
        ],
        'necessity_scores_residual': necessity_scores
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()
    
    # Parse dose-layers argument
    dose_layers = [int(l.strip()) for l in args.dose_layers.split(',')]
    
    # Parse cross-pair-layers argument
    cross_pair_layers = [int(l.strip()) for l in args.cross_pair_layers.split(',')]

    print("=" * 70)
    print("ACTIVATION PATCHING EXPERIMENT")
    if getattr(args, 'cross_pair', False):
        print("Mode: Cross-Pair Interchange (Phase 4)")
        print(f"  Layers: {cross_pair_layers}")
        if args.cross_pair_sample:
            print(f"  Sample size: {args.cross_pair_sample}")
        if getattr(args, 'average_direction', False):
            print("  Average direction test: ENABLED")
    elif getattr(args, 'dose_response', False):
        print("Mode: Dose-Response Analysis (Phase 3)")
        print(f"  Alpha steps: {args.alpha_steps}")
        print(f"  Layers: {dose_layers}")
        if getattr(args, 'cross_track', False):
            print("  Cross-track analysis: ENABLED")
    elif args.blocking:
        print("Mode: Causal Blocking (track interaction test)")
    elif args.behavioral:
        print("Mode: Behavioral (token probability measure)")
    elif args.ablation:
        print("Mode: Ablation (causal necessity test)")
    elif args.improved:
        print("Mode: Extended (replacement + random + addition)")
    else:
        print("Mode: Standard (replacement only)")
    print(f"Probe layer: {args.probe_layer}")
    print("=" * 70 + "\n")

    # Load model and probe (probe not needed for behavioral mode)
    if args.behavioral:
        print("Loading GPT-2...")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model.eval()
        probe = None
    else:
        model, tokenizer, probe = load_model_and_probe(probe_layer=args.probe_layer)

    # Load dataset
    pairs = load_dataset()
    print(f"Loaded {len(pairs)} pairs")

    test_pairs = pairs[:args.n_pairs]
    print(f"Using {len(test_pairs)} pairs for experiment\n")

    # Setup output directory
    output_dir = Path('results/causal_encoding')
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = list(range(12))
    components = ['residual', 'attn', 'mlp']

    # Store results for potential cross-reference
    patching_results = None
    ablation_results = None
    behavioral_results = None
    
    # Suffix for output files when using non-default probe layer
    probe_suffix = f'_probe{args.probe_layer}' if args.probe_layer != 8 else ''

    # Run behavioral ablation if requested
    if args.behavioral:
        print("Running behavioral ablation experiment...")
        behavioral_results = run_ablation_behavioral(
            model, tokenizer, test_pairs,
            layers=layers, components=components
        )
        
        # Print and save behavioral results
        print_behavioral_results(behavioral_results)
        save_behavioral_results(behavioral_results, 
                                output_dir / f'behavioral_ablation_results.json')
        
        print("\nExperiment complete!")
        return {'behavioral': behavioral_results}

    # Run cross-pair interchange experiment if requested (Phase 4)
    if getattr(args, 'cross_pair', False):
        print("=" * 70)
        print("PHASE 4: CROSS-PAIR INTERCHANGE ANALYSIS")
        print("=" * 70)
        
        # Run main cross-pair experiment
        cross_pair_results, cross_pair_summaries = run_cross_pair_experiment(
            model, tokenizer, probe, test_pairs,
            layers=cross_pair_layers, component='residual',
            probe_layer=args.probe_layer,
            sample_size=args.cross_pair_sample
        )
        
        # Run average direction experiment if requested
        avg_direction_results = None
        if getattr(args, 'average_direction', False):
            print("\n--- Average Direction Experiment ---")
            avg_direction_results = run_average_direction_experiment(
                model, tokenizer, probe, test_pairs,
                layers=cross_pair_layers, component='residual',
                probe_layer=args.probe_layer
            )
        
        # Print and save results
        print_cross_pair_table(cross_pair_summaries, avg_direction_results)
        
        # Generate visualizations
        plot_cross_pair_results(cross_pair_summaries, cross_pair_results,
                               output_dir / f'cross_pair_results{probe_suffix}.png')
        
        # Generate heatmaps for each layer
        n_pairs = len(test_pairs) if not args.cross_pair_sample else args.cross_pair_sample
        for layer in cross_pair_layers:
            plot_cross_pair_heatmap(cross_pair_results, layer, n_pairs,
                                   output_dir / f'cross_pair_heatmap_L{layer}{probe_suffix}.png')
        
        # Category comparison if data available
        if any(s.within_category_flip_rate is not None for s in cross_pair_summaries):
            plot_category_comparison(cross_pair_summaries,
                                    output_dir / f'cross_pair_category{probe_suffix}.png')
        
        # Save JSON results
        save_cross_pair_results(cross_pair_results, cross_pair_summaries,
                               avg_direction_results,
                               output_dir / f'cross_pair_results{probe_suffix}.json')
        
        print("\n" + "=" * 70)
        print("PHASE 4 COMPLETE")
        print("=" * 70)
        
        # Print summary
        avg_univ = np.mean([s.universality_score for s in cross_pair_summaries])
        print(f"\nOVERALL UNIVERSALITY: {avg_univ:.2f}")
        
        if avg_univ >= 0.9:
            print("CONCLUSION: Temporal direction is HIGHLY UNIVERSAL")
            print("  → Steering vectors will work across examples")
        elif avg_univ >= 0.7:
            print("CONCLUSION: Temporal direction is MOSTLY UNIVERSAL")
            print("  → Steering vectors should work for most inputs")
        elif avg_univ >= 0.5:
            print("CONCLUSION: MIXED universality")
            print("  → May need context-specific steering")
        else:
            print("CONCLUSION: EXAMPLE-SPECIFIC encoding")
            print("  → Universal steering vectors may not work well")
        
        return {
            'cross_pair_results': cross_pair_results,
            'cross_pair_summaries': cross_pair_summaries,
            'average_direction': avg_direction_results
        }

    # Run causal blocking experiments if requested
    if args.blocking:
        blocking_results = run_blocking_experiments(
            model, tokenizer, probe, test_pairs,
            probe_layer=args.probe_layer
        )
        
        # Print and save blocking results
        print_blocking_results(blocking_results)
        save_blocking_results(blocking_results,
                              output_dir / f'blocking_results{probe_suffix}.json')
        
        print("\nExperiment complete!")
        return {'blocking': blocking_results}

    # Run dose-response analysis if requested (Phase 3)
    if getattr(args, 'dose_response', False):
        print("=" * 70)
        print("PHASE 3: DOSE-RESPONSE ANALYSIS")
        print("=" * 70)
        
        # Experiment 1 & 2: Ablation and Patching Dose-Response
        print("\n--- Experiment 1: Ablation Dose-Response (per track) ---")
        ablation_dose_results = run_dose_response_experiment(
            model, tokenizer, probe, test_pairs,
            layers=dose_layers, component='residual',
            method='ablation', alpha_steps=args.alpha_steps,
            probe_layer=args.probe_layer
        )
        
        print("\n--- Experiment 2: Patching Dose-Response ---")
        patching_dose_results = run_dose_response_experiment(
            model, tokenizer, probe, test_pairs,
            layers=dose_layers, component='residual',
            method='replacement', alpha_steps=args.alpha_steps,
            probe_layer=args.probe_layer
        )
        
        # Combine results
        dose_response_results = ablation_dose_results + patching_dose_results
        
        # Experiment 3: Cross-track dose-response (optional)
        cross_track_results = None
        if getattr(args, 'cross_track', False):
            print("\n--- Experiment 3: Cross-Track Dose-Response ---")
            cross_track_results = run_cross_track_dose_response(
                model, tokenizer, probe, test_pairs,
                alpha_steps=args.alpha_steps, probe_layer=args.probe_layer
            )
        
        # Print and save results
        print_dose_response_table(dose_response_results)
        
        # Generate visualizations
        plot_dose_response(dose_response_results,
                          output_dir / f'dose_response{probe_suffix}.png')
        
        if cross_track_results:
            plot_cross_track_dose_response(cross_track_results,
                                           output_dir / f'cross_track_dose_response{probe_suffix}.png')
        
        # Save JSON results
        save_dose_response_results(dose_response_results, cross_track_results,
                                   output_dir / f'dose_response_results{probe_suffix}.json')
        
        print("\n" + "=" * 70)
        print("PHASE 3 COMPLETE")
        print("=" * 70)
        
        # Print summary interpretation
        print("\nSUMMARY:")
        
        # Find key thresholds
        l1_ablation_lt = next((r for r in dose_response_results 
                               if r.layer == 1 and r.method == 'ablation' and r.direction == 'LT_to_IM'), None)
        l6_ablation_lt = next((r for r in dose_response_results 
                               if r.layer == 6 and r.method == 'ablation' and r.direction == 'LT_to_IM'), None)
        l1_ablation_im = next((r for r in dose_response_results 
                               if r.layer == 1 and r.method == 'ablation' and r.direction == 'IM_to_LT'), None)
        l6_ablation_im = next((r for r in dose_response_results 
                               if r.layer == 6 and r.method == 'ablation' and r.direction == 'IM_to_LT'), None)
        
        if l1_ablation_lt:
            print(f"  L1 (Immediate Track) - LT→IM threshold: {l1_ablation_lt.threshold_alpha or 'N/A'}")
        if l6_ablation_lt:
            print(f"  L6 (Long-term Track) - LT→IM threshold: {l6_ablation_lt.threshold_alpha or 'N/A'}")
        if l1_ablation_im:
            print(f"  L1 (Immediate Track) - IM→LT threshold: {l1_ablation_im.threshold_alpha or 'N/A'}")
        if l6_ablation_im:
            print(f"  L6 (Long-term Track) - IM→LT threshold: {l6_ablation_im.threshold_alpha or 'N/A'}")
        
        # Interpretation based on findings from Phase 2
        print("\n  Key Observations:")
        if l6_ablation_im and max(l6_ablation_im.flip_rates) < 0.2:
            print("  - L6 ablation has LOW IM→LT effect (consistent with 'LT amplifier' model)")
            print("  - L6 doesn't disrupt IM encoding; it only adds LT bias")
        
        if l6_ablation_lt and l6_ablation_lt.threshold_alpha and l6_ablation_lt.threshold_alpha < 0.5:
            print(f"  - L6 is FRAGILE for LT encoding (threshold < 0.5)")
        
        return {
            'dose_response': dose_response_results,
            'cross_track': cross_track_results
        }

    # Run ablation experiment if requested
    if args.ablation:
        print("Running ablation experiment (testing both directions)...")
        ablation_results = run_ablation_experiment(
            model, tokenizer, probe, test_pairs,
            layers=layers, components=components,
            probe_layer=args.probe_layer
        )

        # Get random baseline for comparison (run quick patching if not already done)
        random_results = None
        if args.improved:
            # Will run full patching below, use those results
            pass
        else:
            # Run just random baseline for comparison
            print("\nRunning random baseline for comparison...")
            random_results = run_patching_experiment(
                model, tokenizer, probe, test_pairs,
                layers=layers, components=['residual'],
                methods=['random'], probe_layer=args.probe_layer
            )

        # Print and save ablation results
        print_ablation_table(ablation_results, random_results)
        plot_ablation_results(ablation_results, random_results,
                              output_dir / f'ablation_results{probe_suffix}.png')
        save_ablation_results(ablation_results, output_dir / f'ablation_results{probe_suffix}.json',
                              random_results)

    # Run standard patching experiment if not ablation-only
    if not args.ablation or args.improved:
        # Configure methods
        if args.improved:
            methods = ['replacement', 'random', 'addition']
            print(f"Methods: {', '.join(methods)}")
        else:
            methods = ['replacement']

    print("Running activation patching...")
        patching_results = run_patching_experiment(
        model, tokenizer, probe, test_pairs,
            layers=layers, components=components,
            methods=methods, probe_layer=args.probe_layer
    )

    # Print results
        print_results_table(patching_results, improved=args.improved)

    # Save results
        if args.improved:
            save_results(patching_results, output_dir / f'patching_results_extended{probe_suffix}.json', improved=True)
            plot_results(patching_results, output_dir / f'patching_results{probe_suffix}.png')
            plot_method_comparison(patching_results, output_dir / f'patching_comparison{probe_suffix}.png')

            # If also running ablation, update with full random baseline
            if args.ablation and ablation_results:
                print("\nUpdating ablation comparison with full patching results...")
                print_ablation_table(ablation_results, patching_results)
                plot_ablation_results(ablation_results, patching_results,
                                      output_dir / f'ablation_results{probe_suffix}.png')
                save_ablation_results(ablation_results, output_dir / f'ablation_results{probe_suffix}.json',
                                      patching_results)
        else:
            save_results(patching_results, output_dir / f'patching_results{probe_suffix}.json', improved=False)
            plot_results(patching_results, output_dir / f'patching_results{probe_suffix}.png')

    print("\nExperiment complete!")
    return {'patching': patching_results, 'ablation': ablation_results}


if __name__ == '__main__':
    main()
