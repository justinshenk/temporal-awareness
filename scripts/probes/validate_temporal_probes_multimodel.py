#!/usr/bin/env python3
"""
Validate temporal probes on the implicit dataset for multiple models.

This script reuses:
- explicit probe checkpoints from `research/probes`
- explicit training metrics from `research/results/*_temporal_probe_results_caa.csv`

For each model it:
1. Loads the already-trained explicit-data probes
2. Extracts activations on the implicit dataset
3. Evaluates every layer's probe on implicit data
4. Writes a notebook-compatible validation JSON per model

It also writes a small comparison JSON that summarizes all successful runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import torch
from transformers import __version__ as transformers_version

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from scripts.probes.train_temporal_probes_caa_multimodel import (
    SUPPORTED_MODELS,
    create_probe_dataset,
    get_default_attention_implementation,
    load_caa_dataset,
    load_model_and_tokenizer,
    make_model_tag,
    resolve_model_name,
)

print(f"Using ROOT directory: {ROOT}")

DATA_DIR = ROOT / "data" / "raw" / "temporal_scope_AB_randomized"
PROBES_DIR = ROOT / "research" / "probes"
EXPLICIT_RESULTS_DIR = ROOT / "research" / "results"
VALIDATION_RESULTS_DIR = ROOT / "results" / "probe_validation"


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_reproducibility_info(model_name: str, model_tag: str, device: str) -> dict:
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": 42,
        "model": {
            "name": model_name,
            "tag": model_tag,
            "source": "huggingface",
        },
        "package_versions": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "compute": {
            "device": device,
            "cuda_available": torch.cuda.is_available(),
        },
    }

    if torch.cuda.is_available():
        info["compute"]["gpu_name"] = torch.cuda.get_device_name(0)
        info["compute"]["gpu_count"] = torch.cuda.device_count()

    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        info["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
            ).strip()
        )
    except Exception:
        info["git_commit"] = "unknown"
        info["git_branch"] = "unknown"
        info["git_dirty"] = None

    return info


def get_dataset_info() -> dict:
    explicit_path = DATA_DIR / "temporal_scope_caa.json"
    implicit_path = DATA_DIR / "temporal_scope_implicit.json"
    return {
        "explicit_dataset": {
            "path": str(explicit_path.relative_to(ROOT)),
            "sha256": hash_file(explicit_path),
        },
        "implicit_dataset": {
            "path": str(implicit_path.relative_to(ROOT)),
            "sha256": hash_file(implicit_path),
        },
    }


def load_explicit_results(model_tag: str) -> list[dict]:
    results_path = EXPLICIT_RESULTS_DIR / f"{model_tag}_temporal_probe_results_caa.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing explicit-results CSV for {model_tag}: {results_path}"
        )

    df = pd.read_csv(results_path).sort_values("layer")
    return [
        {
            "layer": int(row.layer),
            "cv_mean": float(row.cv_accuracy_mean),
            "cv_std": float(row.cv_accuracy_std),
            "test_acc": float(row.test_accuracy),
        }
        for row in df.itertuples(index=False)
    ]


def load_probes(model_tag: str) -> dict[int, object]:
    probes = {}
    pattern = f"temporal_caa_layer_{model_tag}_*_probe.pkl"
    for probe_path in sorted(PROBES_DIR.glob(pattern)):
        layer = int(probe_path.stem.split("_")[-2])
        with open(probe_path, "rb") as f:
            probes[layer] = pickle.load(f)

    if not probes:
        raise FileNotFoundError(
            f"No saved probes found for {model_tag} in {PROBES_DIR}"
        )

    return probes


def evaluate_on_implicit(probes: dict[int, object], X_implicit_by_layer, y_implicit) -> list[dict]:
    results = []
    for layer in sorted(probes):
        probe = probes[layer]
        X = X_implicit_by_layer[layer]
        implicit_acc = float(probe.score(X, y_implicit))
        results.append(
            {
                "layer": layer,
                "implicit_acc": implicit_acc,
            }
        )
        print(f"  Layer {layer:2d}: implicit accuracy = {implicit_acc:.3f}")

    return results


def summarize_results(explicit_results: list[dict], implicit_results: list[dict]) -> dict:
    explicit_by_layer = {row["layer"]: row for row in explicit_results}
    implicit_by_layer = {row["layer"]: row for row in implicit_results}

    merged = []
    for layer in sorted(explicit_by_layer):
        explicit_acc = explicit_by_layer[layer]["test_acc"]
        implicit_acc = implicit_by_layer[layer]["implicit_acc"]
        merged.append(
            {
                "layer": layer,
                "explicit_acc": explicit_acc,
                "implicit_acc": implicit_acc,
                "gap": explicit_acc - implicit_acc,
            }
        )

    best = max(merged, key=lambda row: row["implicit_acc"])
    semantic_layers = [row["layer"] for row in merged if row["implicit_acc"] >= 0.70]
    weak_layers = [row["layer"] for row in merged if 0.55 <= row["implicit_acc"] < 0.70]
    lexical_layers = [row["layer"] for row in merged if row["implicit_acc"] < 0.55]

    return {
        "best_semantic_layer": best["layer"],
        "best_implicit_accuracy": best["implicit_acc"],
        "validation_passed": best["implicit_acc"] >= 0.70,
        "semantic_layers": semantic_layers,
        "weak_layers": weak_layers,
        "lexical_layers": lexical_layers,
        "mean_explicit_accuracy": float(np.mean([row["explicit_acc"] for row in merged])),
        "mean_implicit_accuracy": float(np.mean([row["implicit_acc"] for row in merged])),
        "mean_generalization_gap": float(np.mean([row["gap"] for row in merged])),
    }


def validate_single_model(args, model_alias: str) -> dict:
    model_name = resolve_model_name(model_alias)
    model_tag = make_model_tag(model_name)
    attn_implementation = (
        args.attn_implementation
        if args.attn_implementation != "auto"
        else get_default_attention_implementation(model_name)
    )

    explicit_path = DATA_DIR / "temporal_scope_caa.json"
    implicit_path = DATA_DIR / "temporal_scope_implicit.json"

    explicit_pairs, _ = load_caa_dataset(str(explicit_path))
    implicit_pairs, _ = load_caa_dataset(str(implicit_path))

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    print(f"Model tag: {model_tag}")
    print(f"Attention implementation: {attn_implementation or 'model default'}")

    explicit_results = load_explicit_results(model_tag)
    probes = load_probes(model_tag)

    print(f"Loaded {len(probes)} saved probes from {PROBES_DIR}")
    print(f"Loaded explicit metrics from {EXPLICIT_RESULTS_DIR / f'{model_tag}_temporal_probe_results_caa.csv'}")

    print("\nLoading model for implicit evaluation...")
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=attn_implementation,
        local_files_only=args.local_files_only,
    )

    print("\nExtracting implicit activations...")
    X_implicit_by_layer, y_implicit = create_probe_dataset(
        model=model,
        tokenizer=tokenizer,
        pairs=implicit_pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("\nEvaluating saved probes on implicit dataset...")
    implicit_results = evaluate_on_implicit(probes, X_implicit_by_layer, y_implicit)
    summary = summarize_results(explicit_results, implicit_results)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {
        "metadata": {
            "experiment": "probe_semantic_validation",
            "description": "Validate that probes detect semantic temporal features, not keywords",
            **get_reproducibility_info(model_name, model_tag, device),
        },
        "datasets": get_dataset_info(),
        "config": {
            "device": device,
            "n_explicit_pairs": len(explicit_pairs),
            "n_implicit_pairs": len(implicit_pairs),
            "train_test_split": 0.2,
            "cv_folds": 5,
            "probe_type": "LogisticRegression",
            "probe_max_iter": 1000,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "attn_implementation": attn_implementation or "model_default",
            "local_files_only": args.local_files_only,
            "probe_source_dir": str(PROBES_DIR.relative_to(ROOT)),
            "explicit_results_csv": str(
                (EXPLICIT_RESULTS_DIR / f"{model_tag}_temporal_probe_results_caa.csv").relative_to(ROOT)
            ),
        },
        "explicit_results": explicit_results,
        "implicit_results": implicit_results,
        "summary": summary,
    }

    output_path = VALIDATION_RESULTS_DIR / f"{model_tag}_probe_validation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved validation results to {output_path}")

    return {
        "model_name": model_name,
        "model_tag": model_tag,
        "output_path": str(output_path.relative_to(ROOT)),
        "summary": summary,
    }


def write_comparison_file(successes: list[dict], failures: list[dict], args) -> Path:
    comparison = {
        "metadata": {
            "experiment": "probe_semantic_validation_comparison",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requested_models": args.models,
        },
        "successful_models": successes,
        "failed_models": failures,
    }

    output_path = VALIDATION_RESULTS_DIR / "model_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Validate saved temporal probes on the implicit dataset for multiple models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3-4b", "phi-3-mini-4k-instruct", "llama-3.2-3b"],
        help=(
            "Model aliases or Hugging Face ids. "
            "Supported aliases: " + ", ".join(sorted(SUPPORTED_MODELS))
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for implicit activation extraction",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Optional tokenizer max length for validation prompts",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders if needed",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Attention backend to request when loading models",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from the local Hugging Face cache",
    )
    args = parser.parse_args()

    successes = []
    failures = []

    for model_alias in args.models:
        try:
            successes.append(validate_single_model(args, model_alias))
        except Exception as error:
            failure = {
                "model": model_alias,
                "error": str(error),
            }
            failures.append(failure)
            print("\n" + "!" * 70)
            print(f"FAILED: {model_alias}")
            print(error)
            print("!" * 70)
            if len(args.models) == 1:
                raise

    comparison_path = write_comparison_file(successes, failures, args)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Successful models: {len(successes)}")
    print(f"Failed models: {len(failures)}")
    print(f"Comparison summary: {comparison_path}")


if __name__ == "__main__":
    main()
