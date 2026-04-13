#!/usr/bin/env python3
"""
Validate implicit-trained temporal probes on the explicit AB-randomized CAA
dataset for multiple models.

This script supports the same probe methods as the trainer:
- lr: LogisticRegression on residual-stream hidden states
- dmm: difference-of-means direction on residual-stream hidden states
- attn: LogisticRegression on attention-pattern summary features
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

from scripts.probes.train_temporal_probes_caa_multimodel import (  # noqa: E402
    PROBE_DISPLAY_NAMES,
    PROBE_METHODS,
    SUPPORTED_MODELS,
    create_probe_dataset,
    get_default_attention_implementation,
    load_caa_dataset,
    load_model_and_tokenizer,
    make_model_tag,
    method_probe_dir,
    normalize_probe_method,
    predict_probe_artifact,
    resolve_model_name,
    results_csv_path,
)

print(f"Using ROOT directory: {ROOT}")

DATA_DIR = ROOT / "data" / "raw" / "temporal_scope_AB_randomized"
PROBES_DIR = ROOT / "research" / "probes"
VALIDATION_RESULTS_DIR = ROOT / "results" / "probe_validation"


def method_validation_dir(probe_method: str) -> Path:
    return VALIDATION_RESULTS_DIR / probe_method


def validation_json_path(probe_method: str, model_tag: str) -> Path:
    return method_validation_dir(probe_method) / f"{model_tag}_probe_validation_{probe_method}.json"


def comparison_json_path(probe_method: str) -> Path:
    return method_validation_dir(probe_method) / f"model_comparison_{probe_method}.json"


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


def load_implicit_training_results(probe_method: str, model_tag: str) -> list[dict]:
    path = results_csv_path(PROBES_DIR, probe_method, model_tag)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing implicit training-results CSV for {model_tag} / {probe_method}: {path}"
        )

    df = pd.read_csv(path).sort_values("layer")
    return [
        {
            "layer": int(row.layer),
            "probe_method": probe_method,
            "cv_mean": float(row.cv_accuracy_mean),
            "cv_std": float(row.cv_accuracy_std),
            "implicit_acc": float(row.test_accuracy),
            "n_features": int(row.n_features),
        }
        for row in df.itertuples(index=False)
    ]


def load_probe_artifacts(probe_method: str, model_tag: str) -> dict[int, dict]:
    probe_dir = method_probe_dir(PROBES_DIR, probe_method, model_tag)
    pattern = f"temporal_probe_{probe_method}_{model_tag}_layer_*.pkl"
    artifacts = {}
    for path in sorted(probe_dir.glob(pattern)):
        if path.stem.endswith("_scaler"):
            continue
        layer = int(path.stem.rsplit("_layer_", 1)[1])
        with open(path, "rb") as f:
            artifacts[layer] = pickle.load(f)

    if not artifacts:
        raise FileNotFoundError(
            f"No saved {probe_method} probes found for {model_tag} in {probe_dir}"
        )

    return artifacts


def evaluate_on_explicit(
    artifacts: dict[int, dict],
    X_explicit_by_layer,
    y_explicit,
) -> list[dict]:
    results = []
    for layer in sorted(artifacts):
        artifact = artifacts[layer]
        y_pred = predict_probe_artifact(artifact, X_explicit_by_layer[layer])
        explicit_acc = float(np.mean(y_pred == y_explicit))
        results.append(
            {
                "layer": layer,
                "probe_method": artifact["method"],
                "explicit_acc": explicit_acc,
                "test_acc": explicit_acc,
            }
        )
        print(f"  Layer {layer:2d}: explicit accuracy = {explicit_acc:.3f}")

    return results


def summarize_results(implicit_results: list[dict], explicit_results: list[dict]) -> dict:
    explicit_by_layer = {row["layer"]: row for row in explicit_results}
    implicit_by_layer = {row["layer"]: row for row in implicit_results}

    merged = []
    for layer in sorted(explicit_by_layer):
        explicit_acc = explicit_by_layer[layer]["explicit_acc"]
        implicit_acc = implicit_by_layer[layer]["implicit_acc"]
        merged.append(
            {
                "layer": layer,
                "explicit_acc": explicit_acc,
                "implicit_acc": implicit_acc,
                "gap": explicit_acc - implicit_acc,
            }
        )

    best = max(merged, key=lambda row: row["explicit_acc"])
    semantic_layers = [row["layer"] for row in merged if row["explicit_acc"] >= 0.70]
    weak_layers = [row["layer"] for row in merged if 0.55 <= row["explicit_acc"] < 0.70]
    lexical_layers = [row["layer"] for row in merged if row["explicit_acc"] < 0.55]

    return {
        "best_semantic_layer": best["layer"],
        "best_explicit_accuracy": best["explicit_acc"],
        "best_implicit_accuracy": best["implicit_acc"],
        "validation_passed": best["explicit_acc"] >= 0.70,
        "semantic_layers": semantic_layers,
        "weak_layers": weak_layers,
        "lexical_layers": lexical_layers,
        "mean_explicit_accuracy": float(np.mean([row["explicit_acc"] for row in merged])),
        "mean_implicit_accuracy": float(np.mean([row["implicit_acc"] for row in merged])),
        "mean_generalization_gap": float(np.mean([row["gap"] for row in merged])),
    }


def validate_single_model(args, model_alias: str) -> dict:
    probe_method = normalize_probe_method(args.probe_method)
    model_name = resolve_model_name(model_alias)
    model_tag = make_model_tag(model_name)
    attn_implementation = (
        args.attn_implementation
        if args.attn_implementation != "auto"
        else get_default_attention_implementation(model_name)
    )
    if probe_method == "attn" and attn_implementation is None:
        attn_implementation = "eager"

    explicit_path = DATA_DIR / "temporal_scope_caa.json"
    implicit_path = DATA_DIR / "temporal_scope_implicit.json"

    explicit_pairs, _ = load_caa_dataset(str(explicit_path))
    implicit_pairs, _ = load_caa_dataset(str(implicit_path))

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    print(f"Probe method: {probe_method} ({PROBE_DISPLAY_NAMES[probe_method]})")
    print(f"Model tag: {model_tag}")
    print(f"Attention implementation: {attn_implementation or 'model default'}")
    print(f"Device map: {args.device_map}")

    implicit_results = load_implicit_training_results(probe_method, model_tag)
    artifacts = load_probe_artifacts(probe_method, model_tag)

    print(f"Loaded {len(artifacts)} saved probes from {method_probe_dir(PROBES_DIR, probe_method, model_tag)}")
    print(f"Loaded implicit training metrics from {results_csv_path(PROBES_DIR, probe_method, model_tag)}")

    print("\nLoading model for explicit evaluation...")
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=attn_implementation,
        local_files_only=args.local_files_only,
        device_map=args.device_map,
    )

    print("\nExtracting explicit features...")
    X_explicit_by_layer, y_explicit = create_probe_dataset(
        model=model,
        tokenizer=tokenizer,
        pairs=explicit_pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        probe_method=probe_method,
    )

    print("\nEvaluating saved probes on explicit dataset...")
    explicit_results = evaluate_on_explicit(artifacts, X_explicit_by_layer, y_explicit)
    summary = summarize_results(implicit_results, explicit_results)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {
        "metadata": {
            "experiment": "probe_semantic_validation",
            "description": "Validate implicit-trained probes on explicit temporal examples",
            **get_reproducibility_info(model_name, model_tag, device),
        },
        "datasets": get_dataset_info(),
        "config": {
            "device": device,
            "probe_method": probe_method,
            "probe_type": PROBE_DISPLAY_NAMES[probe_method],
            "n_explicit_pairs": len(explicit_pairs),
            "n_implicit_pairs": len(implicit_pairs),
            "train_test_split": 0.2,
            "cv_folds": 5,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "attn_implementation": attn_implementation or "model_default",
            "device_map": args.device_map,
            "local_files_only": args.local_files_only,
            "probe_source_dir": str(method_probe_dir(PROBES_DIR, probe_method, model_tag).relative_to(ROOT)),
            "implicit_training_results_csv": str(
                results_csv_path(PROBES_DIR, probe_method, model_tag).relative_to(ROOT)
            ),
        },
        "explicit_results": explicit_results,
        "implicit_results": implicit_results,
        "summary": summary,
    }

    output_path = validation_json_path(probe_method, model_tag)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved validation results to {output_path}")

    return {
        "model_name": model_name,
        "model_tag": model_tag,
        "probe_method": probe_method,
        "output_path": str(output_path.relative_to(ROOT)),
        "summary": summary,
    }


def write_comparison_file(successes: list[dict], failures: list[dict], args) -> Path:
    probe_method = normalize_probe_method(args.probe_method)
    comparison = {
        "metadata": {
            "experiment": "probe_semantic_validation_comparison",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "probe_method": probe_method,
            "requested_models": args.models,
        },
        "successful_models": successes,
        "failed_models": failures,
    }

    output_path = comparison_json_path(probe_method)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Validate implicit-trained temporal probes on the explicit dataset for multiple models"
    )
    parser.add_argument(
        "--probe-method",
        default="lr",
        choices=PROBE_METHODS,
        help="Probe method to validate",
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
        help="Batch size for explicit feature extraction",
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
    parser.add_argument(
        "--device-map",
        default="single",
        choices=["single", "auto"],
        help=(
            "Device placement for model loading. 'single' keeps the whole model on cuda:0 "
            "when CUDA is available; 'auto' allows Accelerate to shard across visible devices."
        ),
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
                "probe_method": args.probe_method,
                "error": str(error),
            }
            failures.append(failure)
            print("\n" + "!" * 70)
            print(f"FAILED: {model_alias} / {args.probe_method}")
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
