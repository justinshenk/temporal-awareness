"""
Run classification on explicit temporal-scope statements, filter by logit margin,
then patch confident long examples from short examples and vice versa.
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "git+https://github.com/SD-interp/mech-interp-toolkit.git",
    ],
    check=True,
)

load_dotenv()
CONFIG_PATH = Path(__file__).parent / "config"
torch.set_grad_enabled(False)

LayerComponent = tuple[int, str]


def tensor_to_float(tensor: torch.Tensor) -> float:
    """Convert a scalar tensor to a Python float."""
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.to(torch.float32)
    return float(cpu_tensor.item())


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_path(config_path: Path) -> Path:
    """Resolve config paths relative to the local config directory by default."""
    if config_path.is_absolute():
        return config_path
    if config_path.exists():
        return config_path
    return CONFIG_PATH / config_path


def load_labeled_statements(input_file: Path) -> list[dict[str, str]]:
    """Load statement records with question text and labels."""
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    statements: list[dict[str, str]] = []
    for record in data.get("data", []):
        if not isinstance(record, dict):
            raise RuntimeError("Expected each classification record to be a dict")

        question = record.get("question", "")
        label = record.get("label", "")
        if not question or not label:
            raise ValueError("Each record must include non-empty 'question' and 'label'")

        statements.append(
            {
                "id": str(record.get("id", "")),
                "question": str(question),
                "label": str(label),
            }
        )

    return statements


def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split items into equally sized chunks."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def sanitize_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def select_patch_layer_components(model: Any) -> list[LayerComponent]:
    """Patch all z and mlp_hidden activations across the full model depth."""
    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(getattr(model, "config", None), "n_layer", None)
    if num_layers is None:
        raise AttributeError("Could not determine model layer count from model.config")

    return [
        (layer, component)
        for layer in range(int(num_layers))
        for component in ("z", "mlp_hidden")
    ]


def slice_batch_dict(batch_dict: Any, start: int, end: int) -> Any:
    """Slice a tokenized batch along the batch dimension."""
    sliced = {}
    for key, value in batch_dict.items():
        sliced[key] = value[start:end]

    try:
        return batch_dict.__class__(sliced)
    except TypeError:
        return sliced


def predicted_label(
    short_logit: float,
    long_logit: float,
    short_label: str,
    long_label: str,
) -> str:
    """Return the model's predicted label from the option logits."""
    return long_label if long_logit >= short_logit else short_label


def correct_label_logit_diff(
    short_logit: float,
    long_logit: float,
    label: str,
    short_label: str,
    long_label: str,
) -> float:
    """Return the correct-option logit minus the incorrect-option logit."""
    if label == short_label:
        return short_logit - long_logit
    if label == long_label:
        return long_logit - short_logit
    raise ValueError(f"Unexpected label {label!r}")


def collect_base_responses(
    *,
    records: list[dict[str, str]],
    batch_size: int,
    tokenizer: Any,
    model: Any,
    token_id_short: int,
    token_id_long: int,
    short_label: str,
    long_label: str,
    min_logit_diff: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the base model on every statement and partition confident examples."""
    from mech_interp_toolkit.activation_utils import get_activations

    base_layer_components = [(0, "attn")]
    base_responses: list[dict[str, Any]] = []
    confident_short: list[dict[str, Any]] = []
    confident_long: list[dict[str, Any]] = []

    for batch_records in tqdm(chunk_list(records, batch_size), desc="Base responses"):
        batch_prompts = [record["question"] for record in batch_records]
        batch_dict = tokenizer(batch_prompts)
        _, logits = get_activations(
            model,
            batch_dict,
            base_layer_components,
            return_logits=True,
        )

        for batch_idx, record in enumerate(batch_records):
            short_logit = tensor_to_float(logits[batch_idx, -1, token_id_short])  # type: ignore[index]
            long_logit = tensor_to_float(logits[batch_idx, -1, token_id_long])  # type: ignore[index]
            response = predicted_label(short_logit, long_logit, short_label, long_label)
            logit_diff = correct_label_logit_diff(
                short_logit,
                long_logit,
                record["label"],
                short_label,
                long_label,
            )
            keep = logit_diff >= min_logit_diff

            response_record: dict[str, Any] = {
                "id": record["id"],
                "question": record["question"],
                "label": record["label"],
                "response": response,
                "short_logit": short_logit,
                "long_logit": long_logit,
                "correct_logit_diff": logit_diff,
                "kept": keep,
            }
            base_responses.append(response_record)

            if not keep:
                continue

            if record["label"] == short_label:
                confident_short.append(response_record)
            elif record["label"] == long_label:
                confident_long.append(response_record)
            else:
                raise ValueError(f"Unexpected label {record['label']!r}")

    return base_responses, confident_short, confident_long


def collect_directional_patched_responses(
    *,
    target_records: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    token_id_short: int,
    token_id_long: int,
    short_label: str,
    long_label: str,
    num_random_sources: int,
    seed: int,
    direction_name: str,
) -> list[dict[str, Any]]:
    """Patch each confident target example with activations from random source examples."""
    from mech_interp_toolkit.activation_utils import get_activations, patch_activations

    if len(source_records) < num_random_sources:
        raise ValueError(
            f"Need at least {num_random_sources} source examples for {direction_name}, "
            f"but only found {len(source_records)}"
        )

    patch_layer_components = select_patch_layer_components(model)
    rng = np.random.default_rng(seed)
    patched_responses: list[dict[str, Any]] = []

    for target_record in tqdm(target_records, desc=f"Patched responses ({direction_name})"):
        source_indices = rng.choice(
            len(source_records),
            size=num_random_sources,
            replace=False,
        )
        sampled_source_records = [source_records[int(index)] for index in source_indices]
        source_prompts = [record["question"] for record in sampled_source_records]
        target_prompts = [target_record["question"]] * num_random_sources

        combined_batch = tokenizer(source_prompts + target_prompts)
        source_batch = slice_batch_dict(combined_batch, 0, num_random_sources)
        target_batch = slice_batch_dict(
            combined_batch,
            num_random_sources,
            2 * num_random_sources,
        )

        source_acts, _ = get_activations(
            model,
            source_batch,
            patch_layer_components,
            return_logits=True,
        )

        _, patch_logits = patch_activations(
            model,
            target_batch,
            patch_layer_components,
            source_acts,
            return_logits=True,
        )

        for patch_idx, source_record in enumerate(sampled_source_records):
            short_logit = tensor_to_float(patch_logits[patch_idx, -1, token_id_short])  # type: ignore[index]
            long_logit = tensor_to_float(patch_logits[patch_idx, -1, token_id_long])  # type: ignore[index]
            response = predicted_label(short_logit, long_logit, short_label, long_label)
            logit_diff = correct_label_logit_diff(
                short_logit,
                long_logit,
                target_record["label"],
                short_label,
                long_label,
            )

            patched_responses.append(
                {
                    "target_id": target_record["id"],
                    "target_question": target_record["question"],
                    "target_label": target_record["label"],
                    "target_base_response": target_record["response"],
                    "target_base_correct_logit_diff": target_record[
                        "correct_logit_diff"
                    ],
                    "source_id": source_record["id"],
                    "source_question": source_record["question"],
                    "source_label": source_record["label"],
                    "source_base_response": source_record["response"],
                    "source_base_correct_logit_diff": source_record[
                        "correct_logit_diff"
                    ],
                    "patch_direction": direction_name,
                    "patch_index": patch_idx,
                    "response": response,
                    "short_logit": short_logit,
                    "long_logit": long_logit,
                    "correct_logit_diff": logit_diff,
                }
            )

    return patched_responses


def run_bidirectional_classification_activation_patch(
    config_path: Path,
    *,
    min_logit_diff: float = 1.0,
    num_random_sources: int = 5,
    model: Any = None,
    tokenizer: Any = None,
) -> tuple[Any, Any, Path]:
    """Run the bidirectional classification activation patching experiment.

    This entrypoint is intended for Python scripts and notebooks, mirroring the
    structure of ``run_cross_task_generalization``.
    """
    config = load_config(resolve_config_path(config_path))

    model_name: str = config["setup"]["model"]
    seed: int = config["setup"]["seed"]
    batch_size: int = config["setup"]["batch_size"]
    dtype = config["setup"].get("dtype", None)

    data_loc = Path(config["paths"]["data_loc"])
    save_loc = Path(config["paths"]["save_loc"])

    data_file: str = config["input"]["data_file"]
    option_keys: list[str] = config["input"]["option_keys"]
    prompt_suffix: str = config["input"]["prompt_suffix"]
    system_prompt: str = config["parameters"]["system_prompt"]
    output_filename: str = config.get("output", {}).get(
        "filename",
        Path(data_file).stem,
    )

    short_label, long_label = option_keys
    input_file_path = data_loc / data_file
    save_loc.mkdir(parents=True, exist_ok=True)

    from mech_interp_toolkit.utils import load_model_tokenizer_config, set_global_seed

    set_global_seed(seed)

    if model is None or tokenizer is None:
        model, tokenizer, _ = load_model_tokenizer_config(
            model_name=model_name,
            suffix=prompt_suffix,
            system_prompt=system_prompt,
            attn_type="eager",
            dtype=dtype,
        )

    token_id_short = tokenizer.tokenizer.encode(
        short_label,
        add_special_tokens=False,
    )[0]
    token_id_long = tokenizer.tokenizer.encode(
        long_label,
        add_special_tokens=False,
    )[0]

    records = load_labeled_statements(input_file_path)
    base_responses, confident_short, confident_long = collect_base_responses(
        records=records,
        batch_size=batch_size,
        tokenizer=tokenizer,
        model=model,
        token_id_short=token_id_short,
        token_id_long=token_id_long,
        short_label=short_label,
        long_label=long_label,
        min_logit_diff=min_logit_diff,
    )

    patched_long_from_short = collect_directional_patched_responses(
        target_records=confident_long,
        source_records=confident_short,
        tokenizer=tokenizer,
        model=model,
        token_id_short=token_id_short,
        token_id_long=token_id_long,
        short_label=short_label,
        long_label=long_label,
        num_random_sources=num_random_sources,
        seed=seed,
        direction_name=f"{short_label}_to_{long_label}",
    )
    patched_short_from_long = collect_directional_patched_responses(
        target_records=confident_short,
        source_records=confident_long,
        tokenizer=tokenizer,
        model=model,
        token_id_short=token_id_short,
        token_id_long=token_id_long,
        short_label=short_label,
        long_label=long_label,
        num_random_sources=num_random_sources,
        seed=seed + 1,
        direction_name=f"{long_label}_to_{short_label}",
    )

    output_payload = {
        "metadata": {
            "model": model_name,
            "data_file": data_file,
            "system_prompt": system_prompt,
            "option_keys": option_keys,
            "min_logit_diff": min_logit_diff,
            "num_random_sources": num_random_sources,
            "seed": seed,
            "n_base_examples": len(base_responses),
            "n_confident_short": len(confident_short),
            "n_confident_long": len(confident_long),
            "n_patched_long_from_short": len(patched_long_from_short),
            "n_patched_short_from_long": len(patched_short_from_long),
            "patch_components": ["z", "mlp_hidden"],
            "config": config,
        },
        "base_responses": base_responses,
        "patched_responses": {
            f"{short_label}_to_{long_label}": patched_long_from_short,
            f"{long_label}_to_{short_label}": patched_short_from_long,
        },
    }

    output_path = save_loc / (
        f"{sanitize_filename(output_filename)}__bidirectional_activation_patch.json"
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    return model, tokenizer, output_path


def run_classification_short_to_long_activation_patch(
    config_path: Path,
    *,
    min_logit_diff: float = 1.0,
    num_random_sources: int = 5,
    model: Any = None,
    tokenizer: Any = None,
) -> tuple[Any, Any, Path]:
    """Backward-compatible wrapper for notebook and CLI usage."""
    return run_bidirectional_classification_activation_patch(
        config_path,
        min_logit_diff=min_logit_diff,
        num_random_sources=num_random_sources,
        model=model,
        tokenizer=tokenizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bidirectional activation patching for confident classification examples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--min-logit-diff",
        type=float,
        default=1.0,
        help="Discard examples whose correct-label logit diff is below this value.",
    )
    parser.add_argument(
        "--num-random-sources",
        type=int,
        default=5,
        help="Number of random source activations to patch into each target example.",
    )

    args = parser.parse_args()
    run_bidirectional_classification_activation_patch(
        args.config,
        min_logit_diff=args.min_logit_diff,
        num_random_sources=args.num_random_sources,
    )


if __name__ == "__main__":
    main()
