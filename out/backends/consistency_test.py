#!/usr/bin/env python3
"""Backend consistency test - verifies outputs match across backends."""

import json
import sys
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference import ModelRunner
from src.inference.backends import ModelBackend

# Backends to test (excluding NNSIGHT - too slow)
BACKENDS = [
    ModelBackend.HUGGINGFACE,
    ModelBackend.PYVENE,
    ModelBackend.TRANSFORMERLENS,
    ModelBackend.MLX,
]

# Test prompts
TEST_PROMPTS = [
    "What is 2+2? Answer:",
    "The capital of France is",
]

# Test full vocab KL divergence
def compare_full_vocab_distributions(model_name: str) -> dict:
    """Compare full vocabulary probability distributions across backends."""
    print(f"\n{'='*60}")
    print(f"Full Vocabulary Distribution Comparison")
    print('='*60)

    prompt = "The answer is"
    results = {}
    all_probs = {}

    for backend in BACKENDS:
        print(f"  {backend.name}...", end=" ", flush=True)
        try:
            runner = ModelRunner(model_name, backend=backend)
            logits, _ = runner.run_with_cache(prompt)
            # Get full vocab probs at last position
            last_logits = logits[0, -1, :].float()
            probs = torch.softmax(last_logits, dim=-1)
            log_probs = torch.log_softmax(last_logits, dim=-1)

            results[backend.name] = {
                "shape": probs.shape[0],
                "entropy": float(-(probs * log_probs).sum()),
                "max_prob": float(probs.max()),
                "top_token": int(probs.argmax()),
            }
            all_probs[backend.name] = probs.cpu()
            print(f"OK - vocab={probs.shape[0]}, entropy={results[backend.name]['entropy']:.4f}")

            del runner
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"ERROR: {e}")
            results[backend.name] = {"error": str(e)}

    # Compute KL divergences
    if len(all_probs) >= 2:
        ref_backend = list(all_probs.keys())[0]
        ref_probs = all_probs[ref_backend]

        print(f"\n  KL Divergence from {ref_backend}:")
        for backend, probs in all_probs.items():
            if backend != ref_backend:
                # KL(ref || other) = sum(ref * log(ref/other))
                # Add small epsilon to avoid log(0)
                eps = 1e-10
                kl = float((ref_probs * (torch.log(ref_probs + eps) - torch.log(probs + eps))).sum())
                # Also compute reverse KL
                kl_rev = float((probs * (torch.log(probs + eps) - torch.log(ref_probs + eps))).sum())
                # And cosine similarity of probability vectors
                cosine = float(torch.nn.functional.cosine_similarity(
                    ref_probs.unsqueeze(0), probs.unsqueeze(0)
                ))
                print(f"    {backend}: KL={kl:.6f}, KL_rev={kl_rev:.6f}, cosine={cosine:.6f}")
                results[backend]["kl_from_ref"] = kl
                results[backend]["cosine_sim"] = cosine

    return results

# Models to test
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]

OUT_DIR = Path("out/backends")


@dataclass
class BackendResult:
    """Results from a single backend run."""
    backend: str
    # Generation
    generated_text: str = ""
    generated_tokens: list = field(default_factory=list)
    # Logits at each position
    logits_shape: tuple = ()
    logits_mean: float = 0.0
    logits_std: float = 0.0
    logits_first_pos: list = field(default_factory=list)  # First few logits at pos 0
    logits_last_pos: list = field(default_factory=list)   # First few logits at last pos
    # Full logits tensor for comparison
    full_logits: torch.Tensor = None
    # Top token predictions
    top_tokens_last: list = field(default_factory=list)   # Top 5 tokens at last position
    top_probs_last: list = field(default_factory=list)    # Their probabilities
    # Activation cache (for supported backends)
    cache_keys: list = field(default_factory=list)
    num_cached: int = 0
    activation_stats: dict = field(default_factory=dict)  # key -> {mean, std, shape}
    # Full activation cache for comparison
    full_cache: dict = field(default_factory=dict)
    # Error info
    error: str = ""


def compare_tensors(name: str, tensors: dict[str, torch.Tensor], tolerance: float = 1e-4) -> dict:
    """Compare tensors from different backends."""
    if len(tensors) < 2:
        return {"status": "SKIP", "reason": "Not enough backends"}

    backends = list(tensors.keys())
    reference = tensors[backends[0]]

    results = {
        "reference": backends[0],
        "shape_match": True,
        "value_match": True,
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "details": {}
    }

    for backend in backends[1:]:
        tensor = tensors[backend]

        # Check shape
        if tensor.shape != reference.shape:
            results["shape_match"] = False
            results["details"][backend] = {
                "shape_match": False,
                "ref_shape": list(reference.shape),
                "this_shape": list(tensor.shape)
            }
            continue

        # Check values
        diff = torch.abs(tensor.float() - reference.float())
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        results["details"][backend] = {
            "shape_match": True,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "value_match": max_diff < tolerance
        }

        results["max_diff"] = max(results["max_diff"], max_diff)
        results["mean_diff"] = max(results["mean_diff"], mean_diff)
        if max_diff >= tolerance:
            results["value_match"] = False

    return results


def run_backend(model_name: str, backend: ModelBackend, prompt: str) -> BackendResult:
    """Run a single backend and collect results."""
    result = BackendResult(backend=backend.name)

    try:
        runner = ModelRunner(model_name, backend=backend)

        # Get logits and cache (MLX returns empty cache)
        logits, cache = runner.run_with_cache(prompt)

        # Store logits info
        result.logits_shape = tuple(logits.shape)
        result.logits_mean = float(logits.float().mean())
        result.logits_std = float(logits.float().std())
        result.full_logits = logits.float().cpu()  # Store full logits for comparison

        # Store first few logits at first and last position
        result.logits_first_pos = logits[0, 0, :10].float().tolist()
        result.logits_last_pos = logits[0, -1, :10].float().tolist()

        # Get top tokens at last position
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        result.top_tokens_last = top_indices.tolist()
        result.top_probs_last = top_probs.tolist()

        # Store cache info
        if cache:
            result.num_cached = len(cache)
            result.cache_keys = list(cache.keys())[:10]  # First 10 keys
            # Store full cache for activation comparison
            result.full_cache = {k: v.float().cpu() for k, v in cache.items() if isinstance(v, torch.Tensor)}

            # Get stats for a few activations
            for name, tensor in list(cache.items())[:5]:
                if isinstance(tensor, torch.Tensor):
                    result.activation_stats[name] = {
                        "shape": list(tensor.shape),
                        "mean": float(tensor.float().mean()),
                        "std": float(tensor.float().std()),
                    }

        # Generate text
        generated = runner.generate(prompt, max_new_tokens=10, temperature=0.0)
        result.generated_text = generated

        # Tokenize to get token IDs
        tokens = runner.tokenizer.encode(generated, add_special_tokens=False)
        prompt_tokens = runner.tokenizer.encode(prompt, add_special_tokens=False)
        result.generated_tokens = tokens[len(prompt_tokens):]

        # Clean up
        del runner
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()

    return result


def test_consistency_for_prompt(model_name: str, prompt: str) -> dict:
    """Test consistency across all backends for a single prompt."""
    print(f"\n{'='*60}")
    print(f"Testing: {prompt[:50]}...")
    print('='*60)

    results = {}
    all_logits = {}

    for backend in BACKENDS:
        print(f"  {backend.name}...", end=" ", flush=True)
        result = run_backend(model_name, backend, prompt)
        results[backend.name] = result

        if result.error:
            print(f"ERROR: {result.error[:50]}")
        else:
            print(f"OK - logits {result.logits_shape}, generated '{result.generated_text[len(prompt):len(prompt)+20]}...'")

    return results


def compare_activations_at_random_positions(results: dict[str, BackendResult], n_samples: int = 30) -> dict:
    """Compare activation values at random layer/positions across backends."""
    import random

    # Find common activation keys across backends that have caches
    backends_with_cache = {k: v for k, v in results.items() if v.full_cache}
    if len(backends_with_cache) < 2:
        return {"status": "SKIP", "reason": "Not enough backends with caches"}

    # Find common keys (resid_post is most comparable)
    all_keys = [set(v.full_cache.keys()) for v in backends_with_cache.values()]
    common_keys = set.intersection(*all_keys) if all_keys else set()

    # Filter to resid_post keys for comparison (most comparable across backends)
    resid_keys = [k for k in common_keys if "resid_post" in k]
    if not resid_keys:
        resid_keys = list(common_keys)[:10]  # Fallback to any common keys

    if not resid_keys:
        return {"status": "SKIP", "reason": "No common activation keys"}

    # Sample random positions
    ref_backend = list(backends_with_cache.keys())[0]
    ref_cache = backends_with_cache[ref_backend].full_cache

    samples = []
    random.seed(42)  # Reproducible

    for _ in range(n_samples):
        key = random.choice(resid_keys)
        tensor = ref_cache[key]
        # tensor shape is [batch, seq_len, d_model]
        if len(tensor.shape) == 3:
            pos = random.randint(0, tensor.shape[1] - 1)
            dim = random.randint(0, tensor.shape[2] - 1)
            samples.append((key, pos, dim))

    # Compare values at sampled positions
    comparisons = []
    for key, pos, dim in samples:
        values = {}
        for backend, result in backends_with_cache.items():
            if key in result.full_cache:
                tensor = result.full_cache[key]
                if len(tensor.shape) == 3 and pos < tensor.shape[1] and dim < tensor.shape[2]:
                    values[backend] = float(tensor[0, pos, dim])

        if len(values) >= 2:
            ref_val = list(values.values())[0]
            max_diff = max(abs(v - ref_val) for v in values.values())
            comparisons.append({
                "key": key,
                "pos": pos,
                "dim": dim,
                "values": values,
                "max_diff": max_diff,
            })

    # Summarize
    if comparisons:
        max_diffs = [c["max_diff"] for c in comparisons]
        return {
            "status": "OK",
            "n_samples": len(comparisons),
            "max_diff": max(max_diffs),
            "mean_diff": sum(max_diffs) / len(max_diffs),
            "all_close": all(d < 0.01 for d in max_diffs),
            "samples": comparisons[:5],  # First 5 for inspection
        }

    return {"status": "SKIP", "reason": "No valid samples"}


def compare_results(results: dict[str, BackendResult]) -> dict:
    """Compare results across backends."""
    comparison = {
        "generation": {},
        "logits": {},
        "probabilities": {},
        "top_predictions": {},
        "activations": {},
    }

    # Compare activations at random positions
    comparison["activations"] = compare_activations_at_random_positions(results)

    # Compare generated text
    texts = {k: v.generated_text for k, v in results.items() if not v.error}
    tokens = {k: v.generated_tokens for k, v in results.items() if not v.error}

    if len(texts) >= 2:
        ref_backend = list(texts.keys())[0]
        ref_text = texts[ref_backend]
        ref_tokens = tokens[ref_backend]

        comparison["generation"]["reference"] = ref_backend
        comparison["generation"]["ref_text"] = ref_text
        comparison["generation"]["ref_tokens"] = ref_tokens
        comparison["generation"]["matches"] = {}

        for backend, text in texts.items():
            if backend != ref_backend:
                comparison["generation"]["matches"][backend] = {
                    "text_match": text == ref_text,
                    "token_match": tokens[backend] == ref_tokens,
                    "text": text,
                    "tokens": tokens[backend],
                }

    # Compare logits statistics
    comparison["logits"]["shapes"] = {k: v.logits_shape for k, v in results.items() if not v.error}
    comparison["logits"]["means"] = {k: v.logits_mean for k, v in results.items() if not v.error}
    comparison["logits"]["stds"] = {k: v.logits_std for k, v in results.items() if not v.error}
    comparison["logits"]["first_pos"] = {k: v.logits_first_pos for k, v in results.items() if not v.error}
    comparison["logits"]["last_pos"] = {k: v.logits_last_pos for k, v in results.items() if not v.error}

    # Compare full logits tensors
    full_logits = {k: v.full_logits for k, v in results.items() if not v.error and v.full_logits is not None}
    if len(full_logits) >= 2:
        ref_backend = list(full_logits.keys())[0]
        ref_logits = full_logits[ref_backend]
        comparison["logits"]["full_comparison"] = {"reference": ref_backend}

        for backend, logits in full_logits.items():
            if backend != ref_backend:
                diff = (logits - ref_logits).abs()
                comparison["logits"]["full_comparison"][backend] = {
                    "max_diff": float(diff.max()),
                    "mean_diff": float(diff.mean()),
                    "std_diff": float(diff.std()),
                }

    # Compare probability distributions (softmax of logits)
    # This is what actually matters for generation - raw logits may differ by constant
    last_pos_logits = {}
    for k, v in results.items():
        if not v.error and v.logits_last_pos:
            last_pos_logits[k] = torch.tensor(v.logits_last_pos)

    if len(last_pos_logits) >= 2:
        probs = {k: torch.softmax(v, dim=-1) for k, v in last_pos_logits.items()}
        comparison["probabilities"]["softmax_values"] = {k: v.tolist() for k, v in probs.items()}

        # Compute correlation and max diff between softmax values
        ref_backend = list(probs.keys())[0]
        ref_probs = probs[ref_backend]
        comparison["probabilities"]["reference"] = ref_backend
        comparison["probabilities"]["correlations"] = {}
        comparison["probabilities"]["max_diffs"] = {}

        for backend, p in probs.items():
            if backend != ref_backend:
                # Correlation
                corr = float(torch.corrcoef(torch.stack([ref_probs, p]))[0, 1])
                comparison["probabilities"]["correlations"][backend] = corr
                # Max difference
                max_diff = float((ref_probs - p).abs().max())
                comparison["probabilities"]["max_diffs"][backend] = max_diff

    # Compare top token predictions
    comparison["top_predictions"]["top_tokens"] = {k: v.top_tokens_last for k, v in results.items() if not v.error}
    comparison["top_predictions"]["top_probs"] = {k: v.top_probs_last for k, v in results.items() if not v.error}

    # Check if top prediction matches across backends
    if comparison["top_predictions"]["top_tokens"]:
        ref_backend = list(comparison["top_predictions"]["top_tokens"].keys())[0]
        ref_top = comparison["top_predictions"]["top_tokens"][ref_backend][0] if comparison["top_predictions"]["top_tokens"][ref_backend] else None

        comparison["top_predictions"]["top1_match"] = all(
            toks[0] == ref_top
            for toks in comparison["top_predictions"]["top_tokens"].values()
            if toks
        )

        # Check if top-5 match
        ref_top5 = set(comparison["top_predictions"]["top_tokens"][ref_backend][:5])
        comparison["top_predictions"]["top5_match"] = all(
            set(toks[:5]) == ref_top5
            for toks in comparison["top_predictions"]["top_tokens"].values()
            if toks
        )

    return comparison


def print_comparison(prompt: str, comparison: dict, results: dict[str, BackendResult]):
    """Print a formatted comparison."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {prompt[:60]}...")
    print('='*80)

    # Generation comparison
    print("\n--- Generation ---")
    gen = comparison.get("generation", {})
    if "ref_text" in gen:
        ref = gen["reference"]
        print(f"  Reference ({ref}): {gen['ref_text']}")
        for backend, match_info in gen.get("matches", {}).items():
            status = "MATCH" if match_info["text_match"] else "DIFF"
            print(f"  {backend}: [{status}] {match_info['text']}")

    # Logits comparison
    print("\n--- Logits (raw values - may differ by constant) ---")
    logits = comparison.get("logits", {})
    print(f"  Shapes: {logits.get('shapes', {})}")
    print(f"  Means: ", end="")
    for backend, mean in logits.get("means", {}).items():
        print(f"{backend}={mean:.4f}  ", end="")
    print()

    # Probability comparison (this is what matters)
    print("\n--- Probabilities (softmax - should match) ---")
    probs_data = comparison.get("probabilities", {})
    ref = probs_data.get("reference", "N/A")
    print(f"  Reference: {ref}")
    if "correlations" in probs_data:
        print("  Probability correlation vs reference:")
        for backend, corr in probs_data.get("correlations", {}).items():
            max_diff = probs_data.get("max_diffs", {}).get(backend, 0)
            status = "EXACT" if max_diff < 0.001 else "CLOSE" if corr > 0.999 else "DIFF"
            print(f"    {backend}: corr={corr:.6f}, max_diff={max_diff:.6f} [{status}]")

    # Top predictions
    print("\n--- Top Predictions ---")
    preds = comparison.get("top_predictions", {})
    top1_match = preds.get("top1_match", False)
    top5_match = preds.get("top5_match", False)
    print(f"  Top-1 Match: {top1_match}")
    print(f"  Top-5 Match: {top5_match}")
    print("  Top-5 tokens per backend:")
    for backend, tokens in preds.get("top_tokens", {}).items():
        probs = preds.get("top_probs", {}).get(backend, [])
        print(f"    {backend}: {tokens[:5]} (probs: {[f'{p:.3f}' for p in probs[:5]]})")

    # Cache comparison
    print("\n--- Activation Cache ---")
    for backend, result in results.items():
        if not result.error:
            print(f"  {backend}: {result.num_cached} activations")
            if result.activation_stats:
                for name, stats in list(result.activation_stats.items())[:2]:
                    print(f"    {name}: shape={stats['shape']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Activation value comparison at random positions
    print("\n--- Activation Values (30 random layer/pos/dim) ---")
    act_comp = comparison.get("activations", {})
    if act_comp.get("status") == "OK":
        print(f"  Samples: {act_comp['n_samples']}")
        print(f"  Max diff: {act_comp['max_diff']:.6f}")
        print(f"  Mean diff: {act_comp['mean_diff']:.6f}")
        print(f"  All close (<0.01): {act_comp['all_close']}")
        if act_comp.get("samples"):
            print("  Sample values:")
            for s in act_comp["samples"][:3]:
                print(f"    {s['key']}[{s['pos']},{s['dim']}]: {s['values']} (diff={s['max_diff']:.6f})")
    else:
        print(f"  {act_comp.get('status', 'N/A')}: {act_comp.get('reason', '')}")

    # Full logits comparison
    print("\n--- Full Logits Comparison ---")
    logits_comp = comparison.get("logits", {}).get("full_comparison", {})
    if logits_comp:
        ref = logits_comp.get("reference", "N/A")
        print(f"  Reference: {ref}")
        for backend, stats in logits_comp.items():
            if backend != "reference":
                print(f"  {backend}: max_diff={stats['max_diff']:.6f}, mean_diff={stats['mean_diff']:.6f}")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Parse args
    models_to_test = MODELS
    if len(sys.argv) > 1:
        if sys.argv[1] == "--model":
            models_to_test = [sys.argv[2]]

    all_results = {}

    for model in models_to_test:
        model_short = model.split("/")[-1]
        print(f"\n{'#'*80}")
        print(f"# TESTING MODEL: {model}")
        print('#'*80)

        all_results[model] = {}

        for prompt in TEST_PROMPTS:
            results = test_consistency_for_prompt(model, prompt)
            comparison = compare_results(results)
            print_comparison(prompt, comparison, results)

            # Store results (convert dataclasses to dicts)
            all_results[model][prompt] = {
                "results": {k: v.__dict__ for k, v in results.items()},
                "comparison": comparison,
            }

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)

    for model, model_results in all_results.items():
        model_short = model.split("/")[-1]
        print(f"\n{model_short}:")

        for prompt, data in model_results.items():
            comparison = data["comparison"]
            gen_match = all(
                m["text_match"]
                for m in comparison.get("generation", {}).get("matches", {}).values()
            )
            top1_match = comparison.get("top_predictions", {}).get("top1_match", False)
            top5_match = comparison.get("top_predictions", {}).get("top5_match", False)

            # Check probability correlation
            probs_data = comparison.get("probabilities", {})
            correlations = probs_data.get("correlations", {})
            all_corr_high = all(c > 0.999 for c in correlations.values()) if correlations else True

            # PASS if top5 tokens match and probabilities correlate highly
            status = "PASS" if (top5_match and all_corr_high) else "DIFF"
            print(f"  [{status}] {prompt[:40]}... (gen={gen_match}, top1={top1_match}, top5={top5_match}, corr={all_corr_high})")

    # Save results
    output_file = OUT_DIR / "consistency_results.json"
    output_file.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
