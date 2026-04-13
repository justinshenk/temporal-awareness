#!/usr/bin/env python3
"""
Train temporal probes on the implicit AB-randomized CAA dataset for multiple
Hugging Face causal LMs.

Supported probe methods:
- lr: LogisticRegression on residual-stream hidden states
- dmm: difference-of-means direction on residual-stream hidden states
- attn: LogisticRegression on attention-pattern summary features

Artifacts are written under method/model-specific directories, for example:
- research/probes/lr/Qwen__Qwen3-4B/temporal_probe_lr_Qwen__Qwen3-4B_layer_12.pkl
- research/probes/dmm/Qwen__Qwen3-4B/temporal_probe_dmm_Qwen__Qwen3-4B_layer_12.pkl
- research/probes/attn/Qwen__Qwen3-4B/temporal_probe_attn_Qwen__Qwen3-4B_layer_12.pkl
- research/results/lr/Qwen__Qwen3-4B_temporal_probe_lr_implicit_train.csv
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
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
PROBE_METHODS = ("lr", "dmm", "attn")
PROBE_DISPLAY_NAMES = {
    "lr": "LogisticRegression(hidden-state)",
    "dmm": "DifferenceOfMeans(hidden-state)",
    "attn": "LogisticRegression(attention-summary)",
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


def normalize_probe_method(probe_method: str) -> str:
    method = probe_method.lower()
    if method not in PROBE_METHODS:
        raise ValueError(f"Unsupported probe method {probe_method!r}; choose from {PROBE_METHODS}")
    return method


def method_probe_dir(output_dir: str | Path, probe_method: str, model_tag: str) -> Path:
    return Path(output_dir) / probe_method / model_tag


def method_results_dir(output_dir: str | Path, probe_method: str) -> Path:
    return Path(output_dir).parent / "results" / probe_method


def probe_artifact_path(output_dir: str | Path, probe_method: str, model_tag: str, layer: int) -> Path:
    return method_probe_dir(output_dir, probe_method, model_tag) / (
        f"temporal_probe_{probe_method}_{model_tag}_layer_{layer}.pkl"
    )


def scaler_artifact_path(output_dir: str | Path, probe_method: str, model_tag: str, layer: int) -> Path:
    return method_probe_dir(output_dir, probe_method, model_tag) / (
        f"temporal_probe_{probe_method}_{model_tag}_layer_{layer}_scaler.pkl"
    )


def results_csv_path(output_dir: str | Path, probe_method: str, model_tag: str) -> Path:
    return method_results_dir(output_dir, probe_method) / (
        f"{model_tag}_temporal_probe_{probe_method}_implicit_train.csv"
    )


def get_default_attention_implementation(model_name: str) -> str | None:
    """
    Choose a conservative attention backend.

    Attention probes require returned attention tensors; eager attention is the
    most reliable backend for that across current HF decoder models.
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
    """Expand CAA pairs into all immediate prompts followed by all long-term prompts."""
    immediate_prompts = []
    long_term_prompts = []

    for pair in pairs:
        immediate_key, long_term_key = get_pair_options(pair)
        question = pair["question"]

        immediate_prompts.append(build_prompt(question, pair[immediate_key]))
        long_term_prompts.append(build_prompt(question, pair[long_term_key]))

    prompts = immediate_prompts + long_term_prompts
    labels = np.array([0] * len(immediate_prompts) + [1] * len(long_term_prompts))

    return prompts, labels


def load_model_and_tokenizer(
    model_name: str,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    local_files_only: bool = False,
    device_map: str = "single",
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
        "torch_dtype": "auto",
        "offload_folder": "offload",
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if device_map == "auto":
        model_kwargs["device_map"] = "auto"
    elif device_map == "single" and torch.cuda.is_available():
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    elif device_map != "single":
        raise ValueError(f"Unsupported device_map mode: {device_map}")

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


def get_last_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the final attended token index for each sequence, padding-side agnostic."""
    return attention_mask.size(1) - 1 - torch.flip(attention_mask, dims=[1]).argmax(dim=1)


def print_last_token_sample(tokenizer, input_ids: torch.Tensor, positions: torch.Tensor) -> None:
    token_id = input_ids[0, positions[0]].item()
    token_text = tokenizer.decode([token_id])
    print(
        "  Last extracted token sample: "
        f"{token_text!r} (id={token_id}, position={positions[0].item()})"
    )


def extract_hidden_state_dataset(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 4,
    max_length: int | None = None,
):
    """
    Extract last-content-token hidden states for every decoder layer.

    Returns:
        X_by_layer: Dict[layer_idx, np.ndarray] with shape (n_samples, hidden_dim)
    """
    device = model.get_input_embeddings().weight.device
    n_layers = model.config.num_hidden_layers
    activations_by_layer = {layer: [] for layer in range(n_layers)}

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting hidden states"):
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
        last_token_positions = get_last_token_positions(attention_mask)
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)

        if start == 0:
            print_last_token_sample(tokenizer, inputs["input_ids"], last_token_positions)

        for layer_idx, layer_hidden in enumerate(hidden_states):
            last_token_hidden = layer_hidden[batch_indices, last_token_positions, :]
            activations_by_layer[layer_idx].append(last_token_hidden.float().cpu().numpy())

    return {
        layer: np.concatenate(layer_chunks, axis=0)
        for layer, layer_chunks in activations_by_layer.items()
    }


def summarize_attention_for_last_token(
    layer_attention: torch.Tensor,
    attention_mask: torch.Tensor,
    last_token_positions: torch.Tensor,
) -> np.ndarray:
    """Build fixed-width per-head features from last-token attention patterns."""
    features = []
    eps = 1e-12
    batch_size, n_heads = layer_attention.shape[:2]

    for batch_idx in range(batch_size):
        last_pos = int(last_token_positions[batch_idx].item())
        valid_length = int(attention_mask[batch_idx].sum().item())
        valid_positions = attention_mask[batch_idx].bool()
        weights = layer_attention[batch_idx, :, last_pos, :]
        weights = weights[:, valid_positions].float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(eps)
        weights = weights / denom

        seq_len = weights.shape[1]
        first_half_end = max(seq_len // 2, 1)
        second_half_start = seq_len // 2
        last_5_start = max(seq_len - 5, 0)
        last_10_start = max(seq_len - 10, 0)

        entropy = -(weights * (weights + eps).log()).sum(dim=1)
        max_weight = weights.max(dim=1).values
        last_weight = weights[:, -1]
        last_5_mean = weights[:, last_5_start:].mean(dim=1)
        last_10_mean = weights[:, last_10_start:].mean(dim=1)
        first_half_mean = weights[:, :first_half_end].mean(dim=1)
        second_half_mean = weights[:, second_half_start:].mean(dim=1)

        per_head = torch.stack(
            [
                entropy,
                max_weight,
                last_weight,
                last_5_mean,
                last_10_mean,
                first_half_mean,
                second_half_mean,
            ],
            dim=1,
        )
        features.append(per_head.reshape(n_heads * 7).cpu().numpy())

    return np.stack(features, axis=0)


def extract_attention_feature_dataset(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 2,
    max_length: int | None = None,
):
    """
    Extract fixed-width attention-summary features for every decoder layer.

    Each layer feature vector contains 7 summary statistics per attention head
    for the final attended token's attention distribution.
    """
    device = model.get_input_embeddings().weight.device
    n_layers = model.config.num_hidden_layers
    features_by_layer = {layer: [] for layer in range(n_layers)}

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting attention features"):
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
                output_attentions=True,
                use_cache=False,
            )

        if outputs.attentions is None or len(outputs.attentions) == 0:
            raise RuntimeError(
                "Model did not return attention tensors. "
                "Run AttnProbe with --attn-implementation eager."
            )

        attention_mask = inputs["attention_mask"]
        last_token_positions = get_last_token_positions(attention_mask)
        if start == 0:
            print_last_token_sample(tokenizer, inputs["input_ids"], last_token_positions)

        for layer_idx, layer_attention in enumerate(outputs.attentions):
            layer_features = summarize_attention_for_last_token(
                layer_attention=layer_attention,
                attention_mask=attention_mask,
                last_token_positions=last_token_positions,
            )
            features_by_layer[layer_idx].append(layer_features)

    return {
        layer: np.concatenate(layer_chunks, axis=0)
        for layer, layer_chunks in features_by_layer.items()
    }


def create_probe_dataset(
    model,
    tokenizer,
    pairs: list[dict],
    batch_size: int = 4,
    max_length: int | None = None,
    probe_method: str = "lr",
):
    """Create the probe dataset from CAA pairs for the requested probe method."""
    probe_method = normalize_probe_method(probe_method)
    prompts, y = prepare_prompts_and_labels(pairs)

    print(f"Extracting features from {len(pairs)} prompt pairs...")
    print(f"This will create {len(prompts)} samples (immediate + long-term)")
    print(f"Feature source: {PROBE_DISPLAY_NAMES[probe_method]}\n")

    if probe_method == "attn":
        X_by_layer = extract_attention_feature_dataset(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=batch_size,
            max_length=max_length,
        )
    else:
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


def validate_label_order(y: np.ndarray) -> int:
    if len(y) % 2 != 0:
        raise ValueError(f"Expected an even number of labels, got {len(y)}")

    n_pairs = len(y) // 2
    expected_y = np.array([0] * n_pairs + [1] * n_pairs)
    if not np.array_equal(y, expected_y):
        raise ValueError(
            "Expected labels ordered as all immediate examples followed by all long-term examples"
        )
    return n_pairs


def pair_indices_to_row_indices(pair_indices: np.ndarray, n_pairs: int) -> np.ndarray:
    return np.concatenate([pair_indices, pair_indices + n_pairs])


def make_pair_level_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """Split all-immediate/all-long-term activations without separating a pair."""
    n_pairs = validate_label_order(y)
    pair_idx = np.arange(n_pairs)
    train_pairs, test_pairs = train_test_split(
        pair_idx,
        test_size=test_size,
        random_state=42,
    )

    train_idx = pair_indices_to_row_indices(train_pairs, n_pairs)
    test_idx = pair_indices_to_row_indices(test_pairs, n_pairs)
    y_train = np.array([0] * len(train_pairs) + [1] * len(train_pairs))
    y_test = np.array([0] * len(test_pairs) + [1] * len(test_pairs))
    train_groups = np.concatenate([train_pairs, train_pairs])

    return X[train_idx], X[test_idx], y_train, y_test, train_groups


def fit_lr_artifact(X_train: np.ndarray, y_train: np.ndarray, probe_method: str) -> dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    probe = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    probe.fit(X_train_scaled, y_train)
    return {
        "method": probe_method,
        "probe_type": PROBE_DISPLAY_NAMES[probe_method],
        "scaler": scaler,
        "probe": probe,
        "probe_c": 0.1,
        "probe_max_iter": 1000,
    }


def fit_dmm_artifact(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    mean_immediate = X_train_scaled[y_train == 0].mean(axis=0)
    mean_long_term = X_train_scaled[y_train == 1].mean(axis=0)
    direction = mean_long_term - mean_immediate
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("DMM direction has zero norm; cannot train probe")

    immediate_projection = mean_immediate @ direction
    long_term_projection = mean_long_term @ direction
    threshold = float((immediate_projection + long_term_projection) / 2.0)

    return {
        "method": "dmm",
        "probe_type": PROBE_DISPLAY_NAMES["dmm"],
        "scaler": scaler,
        "mean_immediate": mean_immediate,
        "mean_long_term": mean_long_term,
        "direction": direction,
        "direction_norm": norm,
        "threshold": threshold,
    }


def fit_probe_artifact(probe_method: str, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    if probe_method in {"lr", "attn"}:
        return fit_lr_artifact(X_train, y_train, probe_method)
    if probe_method == "dmm":
        return fit_dmm_artifact(X_train, y_train)
    raise ValueError(f"Unsupported probe method: {probe_method}")


def predict_probe_artifact(artifact: dict, X: np.ndarray) -> np.ndarray:
    method = artifact["method"]
    X_scaled = artifact["scaler"].transform(X)

    if method in {"lr", "attn"}:
        return artifact["probe"].predict(X_scaled)

    if method == "dmm":
        scores = X_scaled @ artifact["direction"]
        return (scores >= artifact["threshold"]).astype(int)

    raise ValueError(f"Unsupported probe artifact method: {method}")


def score_probe_artifact(artifact: dict, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(predict_probe_artifact(artifact, X) == y))


def pair_level_cv_scores(
    probe_method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
) -> np.ndarray:
    n_cv_splits = min(5, len(np.unique(train_groups)))
    if n_cv_splits < 2:
        return np.array([np.nan])

    cv = StratifiedGroupKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train, groups=train_groups):
        artifact = fit_probe_artifact(probe_method, X_train[train_idx], y_train[train_idx])
        scores.append(score_probe_artifact(artifact, X_train[val_idx], y_train[val_idx]))

    return np.array(scores, dtype=float)


def save_probe_artifacts(
    artifact: dict,
    output_dir: str | Path,
    probe_method: str,
    model_tag: str,
    layer: int,
) -> tuple[Path, Path]:
    probe_dir = method_probe_dir(output_dir, probe_method, model_tag)
    probe_dir.mkdir(parents=True, exist_ok=True)

    probe_file = probe_artifact_path(output_dir, probe_method, model_tag, layer)
    with open(probe_file, "wb") as f:
        pickle.dump(artifact, f)

    scaler_file = scaler_artifact_path(output_dir, probe_method, model_tag, layer)
    with open(scaler_file, "wb") as f:
        pickle.dump(artifact["scaler"], f)

    return probe_file, scaler_file


def train_probes(
    X_by_layer,
    y,
    output_dir="research/probes",
    model_tag="gpt2",
    probe_method="lr",
):
    """Train one probe per layer and save method-specific results."""
    probe_method = normalize_probe_method(probe_method)

    print("=" * 70)
    print(f"TRAINING TEMPORAL PROBES: {PROBE_DISPLAY_NAMES[probe_method]}")
    print("=" * 70)
    print()

    results = []

    for layer, _ in enumerate(X_by_layer):
        X = X_by_layer[layer]
        print(f"Layer {layer}/{len(X_by_layer) - 1}")
        print("-" * 70)

        X_train, X_test, y_train, y_test, train_groups = make_pair_level_split(X, y)
        cv_scores = pair_level_cv_scores(probe_method, X_train, y_train, train_groups)
        artifact = fit_probe_artifact(probe_method, X_train, y_train)
        test_acc = score_probe_artifact(artifact, X_test, y_test)

        cv_mean = float(np.nanmean(cv_scores))
        cv_std = float(np.nanstd(cv_scores))

        print(f"  CV Accuracy: {cv_mean:.3f} (+/- {cv_std:.3f})")
        print(f"  Test Accuracy: {test_acc:.3f}")

        probe_file, scaler_file = save_probe_artifacts(
            artifact=artifact,
            output_dir=output_dir,
            probe_method=probe_method,
            model_tag=model_tag,
            layer=layer,
        )

        print(f"  ✓ Saved probe to {probe_file}")
        print(f"  ✓ Saved scaler to {scaler_file}")
        print()

        results.append(
            {
                "layer": layer,
                "probe_method": probe_method,
                "probe_type": artifact["probe_type"],
                "cv_accuracy_mean": cv_mean,
                "cv_accuracy_std": cv_std,
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
        print("  Model clearly encodes temporal information on the implicit training split.")
    elif best_acc >= 0.55:
        print("  ○ WEAK SIGNAL (accuracy 55-70%)")
        print("  Temporal information is present but not strongly encoded.")
    else:
        print("  ✗ NO SIGNAL (accuracy < 55%)")
        print("  Model does not encode temporal information in this probe feature space.")

    print("=" * 70)
    print()

    results_file = results_csv_path(output_dir, probe_method, model_tag)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"✓ Results saved to {results_file}\n")

    return results_df


def detailed_evaluation(
    X_by_layer,
    y,
    probe_path,
    layer,
):
    """Run a detailed evaluation of a specific probe with confusion matrix."""
    print("=" * 70)
    print(f"DETAILED EVALUATION - Layer {layer}")
    print("=" * 70)
    print()

    with open(probe_path, "rb") as f:
        artifact = pickle.load(f)

    X = X_by_layer[layer]
    _, X_test, _, y_test, _ = make_pair_level_split(X, y)
    y_pred = predict_probe_artifact(artifact, X_test)

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
    probe_method = normalize_probe_method(args.probe_method)
    resolved_model_name = resolve_model_name(args.model)
    model_tag = make_model_tag(resolved_model_name)
    attn_implementation = (
        args.attn_implementation
        if args.attn_implementation != "auto"
        else get_default_attention_implementation(resolved_model_name)
    )
    if probe_method == "attn" and attn_implementation is None:
        attn_implementation = "eager"
    output_dir = args.output

    print("Configuration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Probe method: {probe_method} ({PROBE_DISPLAY_NAMES[probe_method]})")
    print(f"  Model: {resolved_model_name}")
    print(f"  Model tag: {model_tag}")
    print(f"  Attention implementation: {attn_implementation or 'model default'}")
    print(f"  Device map: {args.device_map}")
    print(f"  Output: {method_probe_dir(output_dir, probe_method, model_tag)}")
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
            device_map=args.device_map,
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
        probe_method=probe_method,
    )

    results_df = train_probes(
        X_by_layer=X_by_layer,
        y=y,
        output_dir=output_dir,
        model_tag=model_tag,
        probe_method=probe_method,
    )

    best_layer = int(results_df.loc[results_df["test_accuracy"].idxmax(), "layer"])
    probe_path = probe_artifact_path(output_dir, probe_method, model_tag, best_layer)

    detailed_evaluation(
        X_by_layer=X_by_layer,
        y=y,
        probe_path=probe_path,
        layer=best_layer,
    )

    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train temporal probes on the implicit CAA dataset for multiple causal LMs"
    )
    parser.add_argument(
        "--dataset",
        default="data/raw/temporal_scope_AB_randomized/temporal_scope_implicit.json",
        help="Path to the implicit CAA-format training dataset",
    )
    parser.add_argument(
        "--probe-method",
        default="lr",
        choices=PROBE_METHODS,
        help="Probe method to train",
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
        help="Root output directory for saved probes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for feature extraction",
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
    parser.add_argument(
        "--device-map",
        default="single",
        choices=["single", "auto"],
        help=(
            "Device placement for model loading. 'single' keeps the whole model on cuda:0 "
            "when CUDA is available; 'auto' allows Accelerate to shard across visible devices."
        ),
    )

    main(parser.parse_args())
