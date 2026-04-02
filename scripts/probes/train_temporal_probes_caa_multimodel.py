#!/usr/bin/env python3
"""
Train temporal probes on the CAA dataset for multiple Hugging Face causal LMs.

This script mirrors the GPT-2-only workflow in `train_temporal_probes_caa.py`
but uses model-agnostic hidden-state extraction so it also works for:

- Qwen/Qwen3-4B
- microsoft/Phi-3-mini-4k-instruct
- meta-llama/Llama-3.2-3B

Saved artifacts follow the same naming pattern as the GPT-2 script, with a
filesystem-safe model tag in the model-name slot:

- research/probes/temporal_caa_layer_{model_tag}_{layer}_probe.pkl
- research/results/{model_tag}_temporal_probe_results_caa.csv
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub.errors import GatedRepoError
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.set_device(0)

SUPPORTED_MODELS = {
    "gpt2": "gpt2",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
}

def load_caa_dataset(dataset_path: str):
    """Load the CAA-format temporal dataset."""
    with open(dataset_path) as f:
        data = json.load(f)

    if "pairs" in data:
        return data["pairs"], data.get("metadata")

    return data, None


def resolve_model_name(model_name: str) -> str:
    """Resolve shorthand aliases to Hugging Face model ids."""
    return SUPPORTED_MODELS.get(model_name, model_name)


def make_model_tag(model_name: str) -> str:
    """Create a path-safe tag while preserving the original naming layout."""
    return model_name.replace("/", "__")


def get_default_attention_implementation(model_name: str) -> str | None:
    """
    Choose a conservative attention backend.

    Some newer decoder models can route into SDPA kernels that are unavailable on
    certain GPU / driver combinations. Defaulting them to eager attention is
    slower but much more robust for probe extraction.
    """
    lowered = model_name.lower()
    if any(token in lowered for token in ("qwen", "phi-3", "phi3", "llama", "mistral")):
        return "eager"
    return None


def resolve_model_source(model_name: str, local_files_only: bool) -> str:
    """
    Resolve a model id to the actual path used by Transformers.

    Loading directly from the cached snapshot avoids some tokenizer code paths
    that still try to hit the network even when `local_files_only=True`.
    """
    if not local_files_only:
        return model_name

    return snapshot_download(model_name, local_files_only=True)


def build_prompt(question: str, answer: str) -> str:
    return question + "\n\nChoices:\n" + answer


def get_pair_options(pair: dict) -> tuple[str, str]:
    """Return immediate and long-term option keys for a CAA pair."""
    if "immediate" in pair and "long_term" in pair:
        return "immediate", "long_term"

    option_keys = [k for k in pair.keys() if k not in {"question", "category"}]
    if len(option_keys) != 2:
        raise ValueError(f"Expected exactly 2 answer options, got {option_keys}")

    option_keys = sorted(option_keys)
    return option_keys[0], option_keys[1]


def prepare_prompts_and_labels(pairs: list[dict]) -> tuple[list[str], np.ndarray]:
    """Expand CAA pairs into prompt strings and binary labels."""
    prompts = []
    labels = []

    for pair in pairs:
        immediate_key, long_term_key = get_pair_options(pair)
        question = pair["question"]

        prompts.append(build_prompt(question, pair[immediate_key]))
        labels.append(0)

        prompts.append(build_prompt(question, pair[long_term_key]))
        labels.append(1)

    return prompts, np.array(labels)


def load_model_and_tokenizer(
    model_name: str,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    local_files_only: bool = False,
):
    """Load a causal LM plus tokenizer with sensible defaults for batching."""
    model_source = resolve_model_source(model_name, local_files_only=local_files_only)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": "auto",
        "offload_folder":"offload",
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.eval()

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def format_model_load_error(model_name: str, error: Exception) -> str:
    """Convert common HF loading failures into concise actionable messages."""
    if isinstance(error, GatedRepoError):
        return (
            f"Access to '{model_name}' is gated on Hugging Face.\n"
            "Request access to the repo and authenticate on this machine, then rerun.\n"
            "Example: `huggingface-cli login` or set `HF_TOKEN` in the environment."
        )

    message = str(error)
    if "gated repo" in message.lower() or "401 client error" in message.lower():
        return (
            f"Access to '{model_name}' is gated on Hugging Face.\n"
            "Request access to the repo and authenticate on this machine, then rerun.\n"
            "Example: `huggingface-cli login` or set `HF_TOKEN` in the environment."
        )

    return message


def extract_hidden_state_dataset(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 4,
    max_length: int | None = None,
):
    """
    Extract last-token hidden states for every decoder layer.

    Returns:
        X_by_layer: Dict[layer_idx, np.ndarray] with shape (n_samples, hidden_dim)
    """
    device = model.get_input_embeddings().weight.device
    n_layers = model.config.num_hidden_layers
    activations_by_layer = {layer: [] for layer in range(n_layers)}

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting activations"):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states[1:]
        attention_mask = inputs["attention_mask"]
        last_token_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)

        for layer_idx, layer_hidden in enumerate(hidden_states):
            last_token_hidden = layer_hidden[batch_indices, last_token_positions, :]
            activations_by_layer[layer_idx].append(last_token_hidden.float().cpu().numpy())

    return {
        layer: np.concatenate(layer_chunks, axis=0)
        for layer, layer_chunks in activations_by_layer.items()
    }


def create_probe_dataset(
    model,
    tokenizer,
    pairs: list[dict],
    batch_size: int = 4,
    max_length: int | None = None,
):
    """Create the probe dataset from CAA pairs."""
    prompts, y = prepare_prompts_and_labels(pairs)

    print(f"Extracting activations from {len(pairs)} prompt pairs...")
    print(f"This will create {len(prompts)} samples (immediate + long-term)\n")

    X_by_layer = extract_hidden_state_dataset(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("\n✓ Created dataset:")
    print(f"  Total samples: {len(y)}")
    print(f"  Immediate (0): {sum(y == 0)}")
    print(f"  Long-term (1): {sum(y == 1)}")
    print(f"  Balance: {sum(y == 0) / len(y):.1%} / {sum(y == 1) / len(y):.1%}")
    print(f"  Layers: {len(X_by_layer)}")
    print(f"  Features per layer: {X_by_layer[0].shape[1]}\n")

    return X_by_layer, y


def train_probes(X_by_layer, y, output_dir="research/probes", model_tag="gpt2"):
    """Train a linear probe for each layer and save results."""
    print("=" * 70)
    print("TRAINING TEMPORAL PROBES")
    print("=" * 70)
    print()

    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for layer, _ in enumerate(X_by_layer):
        X = X_by_layer[layer]
        print(f"Layer {layer}/{len(X_by_layer) - 1}")
        print("-" * 70)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        probe = LogisticRegression(max_iter=1000, random_state=42)
        cv_scores = cross_val_score(probe, X_train, y_train, cv=5, scoring="accuracy")
        probe.fit(X_train, y_train)
        test_acc = probe.score(X_test, y_test)

        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        print(f"  Test Accuracy: {test_acc:.3f}")

        probe_file = output_path / f"temporal_caa_layer_{model_tag}_{layer}_probe.pkl"
        with open(probe_file, "wb") as f:
            pickle.dump(probe, f)

        print(f"  ✓ Saved to {probe_file}")
        print()

        results.append(
            {
                "layer": layer,
                "cv_accuracy_mean": cv_scores.mean(),
                "cv_accuracy_std": cv_scores.std(),
                "test_accuracy": test_acc,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_features": X.shape[1],
            }
        )

    results_df = pd.DataFrame(results)

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print()

    best_layer = results_df.loc[results_df["test_accuracy"].idxmax()]
    print("=" * 70)
    print("BEST LAYER")
    print("=" * 70)
    print(f"  Layer: {int(best_layer['layer'])}")
    print(f"  Test Accuracy: {best_layer['test_accuracy']:.3f}")
    print(
        f"  CV Accuracy: {best_layer['cv_accuracy_mean']:.3f} "
        f"(+/- {best_layer['cv_accuracy_std']:.3f})"
    )
    print()

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    best_acc = best_layer["test_accuracy"]
    if best_acc >= 0.70:
        print("  ✓ STRONG SIGNAL (accuracy ≥ 70%)")
        print("  Model clearly encodes temporal information!")
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

    print("=" * 70)
    print()

    results_file = output_path.parent / "results" / f"{model_tag}_temporal_probe_results_caa.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"✓ Results saved to {results_file}\n")

    return results_df


def detailed_evaluation(
    model,
    tokenizer,
    pairs,
    probe_path,
    layer,
    batch_size: int = 4,
    max_length: int | None = None,
):
    """Run a detailed evaluation of a specific probe with confusion matrix."""
    print("=" * 70)
    print(f"DETAILED EVALUATION - Layer {layer}")
    print("=" * 70)
    print()

    with open(probe_path, "rb") as f:
        probe = pickle.load(f)

    X_by_layer, y = create_probe_dataset(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        batch_size=batch_size,
        max_length=max_length,
    )
    X = X_by_layer[layer]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_pred = probe.predict(X_test)

    print("Classification Report:")
    print("-" * 70)
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Immediate (0)", "Long-term (1)"],
        )
    )

    print("\nConfusion Matrix:")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred)
    print("                 Predicted")
    print("               Immediate  Long-term")
    print(f"Actual Immediate    {cm[0, 0]:3d}       {cm[0, 1]:3d}")
    print(f"       Long-term    {cm[1, 0]:3d}       {cm[1, 1]:3d}")
    print()


def main(args):
    print("=" * 70)
    print("TEMPORAL PROBE TRAINING (CAA FORMAT, MULTI-MODEL)")
    print("=" * 70)
    print()

    dataset_path = args.dataset
    resolved_model_name = resolve_model_name(args.model)
    model_tag = make_model_tag(resolved_model_name)
    attn_implementation = (
        args.attn_implementation
        if args.attn_implementation != "auto"
        else get_default_attention_implementation(resolved_model_name)
    )
    output_dir = args.output

    print("Configuration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Model: {resolved_model_name}")
    print(f"  Model tag: {model_tag}")
    print(f"  Attention implementation: {attn_implementation or 'model default'}")
    print(f"  Output: {output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max length: {args.max_length}")
    print()

    print("Loading CAA dataset...")
    pairs, metadata = load_caa_dataset(dataset_path)

    if metadata:
        print(f"  Dimension: {metadata.get('dimension', 'N/A')}")
        print(f"  Style: {metadata.get('style', 'N/A')}")
        print(f"  Pairs: {metadata.get('n_pairs', len(pairs))}")
    else:
        print(f"  Pairs: {len(pairs)}")
    print()

    print(f"Loading {resolved_model_name}...")
    try:
        model, tokenizer = load_model_and_tokenizer(
            resolved_model_name,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=attn_implementation,
            local_files_only=args.local_files_only,
        )
    except Exception as error:
        raise RuntimeError(format_model_load_error(resolved_model_name, error)) from error
    print("✓ Model loaded")
    print()

    X_by_layer, y = create_probe_dataset(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    results_df = train_probes(
        X_by_layer=X_by_layer,
        y=y,
        output_dir=output_dir,
        model_tag=model_tag,
    )

    best_layer = int(results_df.loc[results_df["test_accuracy"].idxmax(), "layer"])
    probe_path = Path(output_dir) / f"temporal_caa_layer_{model_tag}_{best_layer}_probe.pkl"

    detailed_evaluation(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        probe_path=probe_path,
        layer=best_layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train temporal probes on the CAA dataset for multiple causal LMs"
    )
    parser.add_argument(
        "--dataset",
        default="data/raw/temporal_scope_AB_randomized/temporal_scope_caa.json",
        help="Path to the CAA dataset",
    )
    parser.add_argument(
        "--model",
        default="qwen3-4b",
        help=(
            "Model alias or Hugging Face model id. "
            "Supported aliases: " + ", ".join(sorted(SUPPORTED_MODELS))
        ),
    )
    parser.add_argument(
        "--output",
        default="research/probes",
        help="Output directory for saved probes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for hidden-state extraction",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional tokenizer max length",
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
        help=(
            "Attention backend to request from Transformers. "
            "'auto' defaults Qwen / Phi-3 / Llama to eager attention for stability."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from the local Hugging Face cache",
    )

    main(parser.parse_args())
