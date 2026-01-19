#!/usr/bin/env python3
"""
Probe Validation Script for GCP

Run this on GCP with GPU to validate that temporal probes detect semantic
features (not just lexical shortcuts).

Usage:
    # On GCP VM with GPU
    python scripts/validate_probes_gcp.py --gpu

    # Quick test (CPU, subset)
    python scripts/validate_probes_gcp.py --quick

Validation Protocol:
1. Train probes on EXPLICIT dataset (contains temporal keywords)
2. Test probes on IMPLICIT dataset (semantic only, no keywords)
3. If accuracy holds → probes detect semantic temporal reasoning
4. If accuracy drops → probes detect lexical shortcuts (bad!)

Success Criteria:
- Implicit accuracy > 70%: Semantic encoding confirmed
- Implicit accuracy 55-70%: Weak semantic signal
- Implicit accuracy < 55%: Probes are detecting keywords, not semantics
"""

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"
CHECKPOINTS_DIR = ROOT / "results" / "checkpoints"
RESULTS_DIR = ROOT / "results"


def get_reproducibility_info():
    """Collect all information needed to reproduce this experiment."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": 42,
    }

    # Git info
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
        ).strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip())
    except Exception:
        info["git_commit"] = "unknown"
        info["git_branch"] = "unknown"
        info["git_dirty"] = None

    # Package versions
    info["package_versions"] = {
        "torch": torch.__version__,
        "transformers": None,
        "sklearn": None,
        "numpy": np.__version__,
    }
    try:
        import transformers
        info["package_versions"]["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import sklearn
        info["package_versions"]["sklearn"] = sklearn.__version__
    except Exception:
        pass

    # Model info
    info["model"] = {
        "name": "gpt2",
        "source": "huggingface",
    }

    return info


def hash_file(path):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_dataset_info():
    """Get dataset metadata and hashes for reproducibility."""
    explicit_path = DATA_DIR / "temporal_scope_caa.json"
    implicit_path = DATA_DIR / "temporal_scope_implicit.json"

    info = {}
    if explicit_path.exists():
        info["explicit_dataset"] = {
            "path": str(explicit_path.relative_to(ROOT)),
            "sha256": hash_file(explicit_path),
        }
    if implicit_path.exists():
        info["implicit_dataset"] = {
            "path": str(implicit_path.relative_to(ROOT)),
            "sha256": hash_file(implicit_path),
        }
    return info


def load_dataset(path):
    """Load CAA-format dataset."""
    with open(path) as f:
        data = json.load(f)
    return data.get("pairs", data), data.get("metadata", {})


def extract_activations(model, tokenizer, prompt, device="cpu", use_mean=False):
    """Extract last-token activations from all layers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    activations = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            if use_mean:
                activations[layer_idx] = output[0][0, :, :].mean(dim=0).detach().cpu().numpy()
            else:
                activations[layer_idx] = output[0][0, -1, :].detach().cpu().numpy()
        return hook

    hooks = []
    for i, layer in enumerate(model.transformer.h):
        hooks.append(layer.register_forward_hook(hook_fn(i)))

    with torch.no_grad():
        model(**inputs)

    for hook in hooks:
        hook.remove()

    return activations


def create_dataset(model, tokenizer, pairs, device="cpu", use_mean=False):
    """Create activation dataset from prompt pairs."""
    n_layers = len(model.transformer.h)
    activations_by_layer = {i: [] for i in range(n_layers)}
    labels = []

    for pair in tqdm(pairs, desc="Extracting activations"):
        question = pair["question"]

        # Get option keys
        option_keys = [k for k in pair.keys() if k not in ["question", "category"]]
        immediate_key = option_keys[0]
        long_term_key = option_keys[1]

        # Create prompts
        immediate_prompt = f"{question}\n\nChoices:\n{pair[immediate_key]}"
        long_term_prompt = f"{question}\n\nChoices:\n{pair[long_term_key]}"

        # Extract activations
        imm_acts = extract_activations(model, tokenizer, immediate_prompt, device, use_mean)
        lt_acts = extract_activations(model, tokenizer, long_term_prompt, device, use_mean)

        for layer in range(n_layers):
            activations_by_layer[layer].append(imm_acts[layer])
            activations_by_layer[layer].append(lt_acts[layer])

        labels.extend([0, 1])  # 0=immediate, 1=long_term

    X_by_layer = {l: np.array(acts) for l, acts in activations_by_layer.items()}
    y = np.array(labels)

    return X_by_layer, y


def train_probes_on_explicit(X_by_layer, y):
    """Train probes on explicit (keyword-rich) data."""
    probes = {}
    results = []

    for layer in range(len(X_by_layer)):
        X = X_by_layer[layer]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train probe
        probe = LogisticRegression(max_iter=1000, random_state=42)
        cv_scores = cross_val_score(probe, X_train, y_train, cv=5, scoring="accuracy")
        probe.fit(X_train, y_train)

        test_acc = probe.score(X_test, y_test)

        probes[layer] = probe
        results.append({
            "layer": layer,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_acc": test_acc
        })

        print(f"  Layer {layer:2d}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}, Test={test_acc:.3f}")

    return probes, results


def evaluate_on_implicit(probes, X_implicit_by_layer, y_implicit):
    """Evaluate probes trained on explicit data against implicit data."""
    results = []

    for layer, probe in probes.items():
        X = X_implicit_by_layer[layer]
        y_pred = probe.predict(X)
        acc = accuracy_score(y_implicit, y_pred)

        results.append({
            "layer": layer,
            "implicit_acc": acc
        })

        print(f"  Layer {layer:2d}: Implicit accuracy = {acc:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate temporal probes on GCP")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--quick", action="store_true", help="Quick test (subset)")
    parser.add_argument("--use-mean", action="store_true", help="Use activation means instead of last token")
    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.gpu and not torch.cuda.is_available():
        print("WARNING: --gpu specified but CUDA not available, using CPU")

    # Load datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)

    explicit_pairs, explicit_meta = load_dataset(DATA_DIR / "temporal_scope_caa.json")
    implicit_pairs, implicit_meta = load_dataset(DATA_DIR / "temporal_scope_implicit.json")

    print(f"Explicit dataset: {len(explicit_pairs)} pairs")
    print(f"Implicit dataset: {len(implicit_pairs)} pairs")

    if args.quick:
        explicit_pairs = explicit_pairs[:10]
        implicit_pairs = implicit_pairs[:10]
        print(f"Quick mode: using {len(explicit_pairs)} pairs each")

    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    print("GPT-2 loaded")

    # Create datasets
    print("\n" + "="*70)
    print("EXTRACTING ACTIVATIONS - EXPLICIT DATA")
    print("="*70)

    start = time.time()
    X_explicit, y_explicit = create_dataset(model, tokenizer, explicit_pairs, device, args.use_mean)
    explicit_time = time.time() - start
    print(f"Extracted in {explicit_time:.1f}s")

    print("\n" + "="*70)
    print("EXTRACTING ACTIVATIONS - IMPLICIT DATA")
    print("="*70)

    start = time.time()
    X_implicit, y_implicit = create_dataset(model, tokenizer, implicit_pairs, device, args.use_mean)
    implicit_time = time.time() - start
    print(f"Extracted in {implicit_time:.1f}s")

    # Train probes on explicit data
    print("\n" + "="*70)
    print("TRAINING PROBES ON EXPLICIT (KEYWORD) DATA")
    print("="*70)

    probes, explicit_results = train_probes_on_explicit(X_explicit, y_explicit)

    # Evaluate on implicit data
    print("\n" + "="*70)
    print("EVALUATING ON IMPLICIT (SEMANTIC) DATA")
    print("="*70)

    implicit_results = evaluate_on_implicit(probes, X_implicit, y_implicit)

    # Combine results
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    print(f"\n{'Layer':<6} {'Explicit':<12} {'Implicit':<12} {'Gap':<10} {'Status'}")
    print("-"*50)

    for exp, imp in zip(explicit_results, implicit_results):
        layer = exp["layer"]
        exp_acc = exp["test_acc"]
        imp_acc = imp["implicit_acc"]
        gap = exp_acc - imp_acc

        if imp_acc >= 0.70:
            status = "✓ SEMANTIC"
        elif imp_acc >= 0.55:
            status = "○ WEAK"
        else:
            status = "✗ LEXICAL"

        print(f"{layer:<6} {exp_acc:<12.3f} {imp_acc:<12.3f} {gap:<10.3f} {status}")

    # Find best semantic layer
    best_implicit = max(implicit_results, key=lambda x: x["implicit_acc"])
    best_layer = best_implicit["layer"]
    best_acc = best_implicit["implicit_acc"]

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBest semantic layer: {best_layer}")
    print(f"Implicit accuracy: {best_acc:.1%}")

    if best_acc >= 0.70:
        print("\n✓ PROBES VALIDATED: Detecting semantic temporal features")
        print("  Safe to proceed with SPD experiments")
    elif best_acc >= 0.55:
        print("\n○ WEAK VALIDATION: Some semantic signal detected")
        print("  Proceed with caution - results may be noisy")
    else:
        print("\n✗ VALIDATION FAILED: Probes likely detect keywords, not semantics")
        print("  DO NOT proceed with SPD until probes are improved")
        print("  Suggestions:")
        print("    - Train on implicit data directly")
        print("    - Use paraphrase augmentation")
        print("    - Expand dataset diversity")

    # Collect reproducibility info
    repro_info = get_reproducibility_info()
    dataset_info = get_dataset_info()

    # Save results
    output_file = RESULTS_DIR / "probe_validation_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "experiment": "probe_semantic_validation",
            "description": "Validate that probes detect semantic temporal features, not keywords",
            **repro_info,
        },
        "datasets": dataset_info,
        "config": {
            "device": device,
            "n_explicit_pairs": len(explicit_pairs),
            "n_implicit_pairs": len(implicit_pairs),
            "train_test_split": 0.2,
            "cv_folds": 5,
            "probe_type": "LogisticRegression",
            "probe_max_iter": 1000,
        },
        "explicit_results": explicit_results,
        "implicit_results": implicit_results,
        "summary": {
            "best_semantic_layer": best_layer,
            "best_implicit_accuracy": best_acc,
            "validation_passed": best_acc >= 0.70,
        },
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
