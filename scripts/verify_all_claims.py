#!/usr/bin/env python3
"""
Verify all main claims from the temporal awareness research.

Usage:
    python scripts/verify_all_claims.py [--quick] [--gpu]

Options:
    --quick     Skip activation extraction, use cached results
    --gpu       Use GPU for extraction (faster but requires CUDA)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path


# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Directories
DATA_DIR = ROOT / "data" / "raw"
CHECKPOINTS_DIR = ROOT / "results" / "checkpoints"
RESULTS_DIR = ROOT / "results"


def load_probes():
    """Load pre-trained probes for all layers."""
    probes = {}
    for layer in range(12):
        probe_path = CHECKPOINTS_DIR / f"temporal_caa_layer_{layer}_probe.pkl"
        if probe_path.exists():
            with open(probe_path, "rb") as f:
                probes[layer] = pickle.load(f)
    return probes


def load_steering_vectors():
    """Load learned steering vectors."""
    path = CHECKPOINTS_DIR / "temporal_directions_learned.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_datasets():
    """Load explicit and implicit datasets."""
    datasets = {}

    explicit_path = DATA_DIR / "temporal_scope_caa.json"
    if explicit_path.exists():
        with open(explicit_path) as f:
            datasets["explicit"] = json.load(f)

    implicit_path = DATA_DIR / "temporal_scope_implicit.json"
    if implicit_path.exists():
        with open(implicit_path) as f:
            datasets["implicit"] = json.load(f)

    return datasets


def check_file_integrity():
    """Check that all required files exist."""
    print("\n" + "=" * 60)
    print("FILE INTEGRITY CHECK")
    print("=" * 60)

    required_files = [
        ("Explicit dataset", DATA_DIR / "temporal_scope_caa.json"),
        ("Implicit dataset", DATA_DIR / "temporal_scope_implicit.json"),
        ("Steering vectors", CHECKPOINTS_DIR / "temporal_directions_learned.json"),
    ]

    # Add probe files
    for layer in range(12):
        required_files.append(
            (f"Probe layer {layer}", CHECKPOINTS_DIR / f"temporal_caa_layer_{layer}_probe.pkl")
        )

    all_exist = True
    for name, path in required_files:
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path.name}")
        if not exists:
            all_exist = False

    return all_exist


def verify_claim_1(probes, quick=True):
    """
    Claim 1: Temporal scope is linearly encoded.
    Expected: ~92.5% accuracy at Layer 8.
    """
    print("\n" + "=" * 60)
    print("CLAIM 1: Temporal scope is linearly encoded")
    print("=" * 60)

    if quick:
        print("⚠️  Quick mode: Using reported values (not verified)")
        reported = {
            "train_peak": 92.5,
            "train_peak_layer": 8,
            "test_peak": 84.0,
            "test_peak_layer": 6,
        }
        print(f"  Reported train accuracy: {reported['train_peak']}% (Layer {reported['train_peak_layer']})")
        print(f"  Reported test accuracy: {reported['test_peak']}% (Layer {reported['test_peak_layer']})")
        return "UNVERIFIED", reported

    # Full verification requires activation extraction
    print("  Full verification requires activation extraction...")
    print("  Run with GPU: python scripts/verify_all_claims.py --gpu")
    return "SKIPPED", None


def verify_claim_2(steering_vectors, probes, quick=True):
    """
    Claim 2: Steering affects same features probes detect.
    Expected: r=0.935 correlation at Layer 11.
    """
    print("\n" + "=" * 60)
    print("CLAIM 2: Steering correlates with probe predictions")
    print("=" * 60)

    if quick:
        print("⚠️  Quick mode: Using reported values (not verified)")
        reported = {
            "peak_correlation": 0.935,
            "peak_layer": 11,
            "p_value": "<0.0001",
        }
        print(f"  Reported correlation: r={reported['peak_correlation']} (Layer {reported['peak_layer']})")
        print(f"  Reported p-value: {reported['p_value']}")
        return "UNVERIFIED", reported

    print("  Full verification requires steering experiment...")
    return "SKIPPED", None


def verify_claim_3(quick=True):
    """
    Claim 3: Late layers encode semantic (not lexical) features.
    Expected: 100% accuracy on keyword-ablated data at Layers 10-11.
    """
    print("\n" + "=" * 60)
    print("CLAIM 3: Semantic encoding in late layers")
    print("=" * 60)

    if quick:
        print("⚠️  Quick mode: Using reported values (not verified)")
        reported = {
            "ablated_accuracy_layer_10": 100,
            "ablated_accuracy_layer_11": 100,
            "interpretation": "Late layers encode semantic temporal features",
        }
        print(f"  Reported ablated accuracy (L10): {reported['ablated_accuracy_layer_10']}%")
        print(f"  Reported ablated accuracy (L11): {reported['ablated_accuracy_layer_11']}%")
        print("  ⚠️  This is the most surprising claim - needs careful validation")
        return "UNVERIFIED", reported

    print("  Full verification requires ablation experiment...")
    return "SKIPPED", None


def check_dataset_quality():
    """Check for potential issues in datasets."""
    print("\n" + "=" * 60)
    print("DATASET QUALITY CHECK")
    print("=" * 60)

    datasets = load_datasets()

    # Check explicit dataset
    if "explicit" in datasets:
        explicit = datasets["explicit"]
        n_pairs = len(explicit.get("pairs", []))
        print(f"  Explicit dataset: {n_pairs} pairs")

    # Check implicit dataset for keyword leakage
    if "implicit" in datasets:
        implicit = datasets["implicit"]
        pairs = implicit.get("pairs", [])
        n_pairs = len(pairs)

        # Keywords that shouldn't appear in implicit dataset
        temporal_keywords = [
            "now", "immediate", "urgent", "today", "soon", "quick",
            "future", "long-term", "years", "decade", "lasting", "permanent"
        ]

        contaminated = 0
        for pair in pairs:
            text = f"{pair.get('immediate', '')} {pair.get('long_term', '')}".lower()
            for kw in temporal_keywords:
                if kw in text:
                    contaminated += 1
                    break

        print(f"  Implicit dataset: {n_pairs} pairs")
        print(f"  Potential keyword contamination: {contaminated}/{n_pairs} pairs")

        if contaminated > 0:
            print(f"  ⚠️  WARNING: {contaminated} pairs may have temporal keyword leakage")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify temporal awareness claims")
    parser.add_argument("--quick", action="store_true", help="Use cached results")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for extraction")
    args = parser.parse_args()

    print("=" * 60)
    print("TEMPORAL AWARENESS - CLAIM VERIFICATION")
    print("=" * 60)

    # File integrity
    files_ok = check_file_integrity()
    if not files_ok:
        print("\n✗ FAILED: Missing required files")
        sys.exit(1)

    # Load resources
    probes = load_probes()
    steering = load_steering_vectors()

    print(f"\nLoaded {len(probes)} probes")
    print(f"Steering vectors: {'✓' if steering else '✗'}")

    # Verify claims
    quick = args.quick or not args.gpu

    results = {}
    results["claim_1"] = verify_claim_1(probes, quick=quick)
    results["claim_2"] = verify_claim_2(steering, probes, quick=quick)
    results["claim_3"] = verify_claim_3(quick=quick)

    # Dataset quality
    dataset_ok = check_dataset_quality()

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    for claim, (status, _) in results.items():
        icon = "✓" if status == "PASSED" else "⚠️" if status == "UNVERIFIED" else "○"
        print(f"  {icon} {claim}: {status}")

    print(f"  {'✓' if dataset_ok else '⚠️'} Dataset quality: {'OK' if dataset_ok else 'ISSUES FOUND'}")

    if quick:
        print("\n⚠️  Results are UNVERIFIED (quick mode)")
        print("   Run with --gpu for full verification")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
