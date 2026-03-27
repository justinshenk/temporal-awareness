"""
Run EAP Integrated Gradients on clean vs corrupted prompts and save scores to NPZ.
"""

import argparse
import json
import os
import pickle
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

LayerComponent = tuple[int, str]
LayerComponentNode = tuple[LayerComponent, int]

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
NODES_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "data"
    / "selected_nodes"
    / "final_200_QnA.pkl"
)
torch.set_grad_enabled(False)
HF_REPO_ID = os.getenv("HF_REPO_ID", "Temporal_Awareness_EAP_IG")
SUPPORTED_QUADRATURES = {
    "gauss-chebyshev",
    "gauss-legendre",
    "riemann-midpoint",
}
QUADRATURE_ALIASES = {
    "midpoint": "riemann-midpoint",
}


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array, normalizing unsupported dtypes."""
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.to(torch.float32)
    return cpu_tensor.numpy()


def load_prompts(
    input_file: Path,
) -> list[str]:
    """Load prompts from ``input_file`` and return a list of clean prompts.

    Args:
        input_file: Path to a JSON file containing a ``pairs`` list. Each item
            may be either a prompt string or a dict with a ``question`` field.

    Returns:
        List of prompt strings extracted from the input file.
    """
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clean_prompts = []
    pairs = data.get("data", [])

    # Accept either raw prompt strings or dict records containing a question field.
    for pair in pairs:
        if isinstance(pair, str):
            prompt = pair
        elif isinstance(pair, dict):
            prompt = "{}".format(
                pair.get("question", ""),
            )
        else:
            raise RuntimeError("Incorrect type for pairs")

        clean_prompts.append(prompt)
    return clean_prompts


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary with configuration values
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_path(config_path: Path) -> Path:
    """Resolve config paths relative to the local config directory by default."""
    if config_path.is_absolute():
        return config_path
    if config_path.exists():
        return config_path
    return CONFIG_PATH / config_path


def str_to_layer_component(s: str):
    comp, layer = s.split("/")
    return int(layer), comp


def sanitize_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def scale_selected_nodes(
    activations: Any,
    layer_component_nodes: list[LayerComponentNode],
    scale_factor: float = 2.0,
) -> None:
    """Scale the requested nodes in-place."""
    activations.split_heads()

    for (layer, component), node in layer_component_nodes:
        if component == "z":
            activations[(layer, component)][:, :, node, :] *= scale_factor
        elif component == "mlp_hidden":
            activations[(layer, component)][:, :, node] *= scale_factor
        else:
            raise ValueError(f"Unsupported component for patching: {component}")

    activations.merge_heads()


def run_cross_task_generalization(
    config_path: Path,
    node_classes: list[str],
    *,
    nodes_path: Path = NODES_PATH,
    model=None,
    tokenizer=None,
) -> tuple[Any, Any]:
    """Run cross-task generalization from Python or notebooks."""
    config = load_config(resolve_config_path(config_path))

    node_lookup = pickle.load(nodes_path.open("rb"))
    nodes = [node_lookup[x] for x in node_classes]
    nodes = [item for sublist in nodes for item in sublist]
    nodes = [(str_to_layer_component(x[0]), x[1]) for x in nodes]
    nodes = list(set(nodes))

    model_name: str = config["setup"]["model"]
    seed: int = config["setup"]["seed"]
    batch_size: int = config["setup"]["batch_size"]
    dtype = config["setup"].get("dtype", None)

    data_loc: Path = Path(config["paths"]["data_loc"])
    save_loc: Path = Path(config["paths"]["save_loc"])

    data_file: str = config["input"]["data_file"]
    option_keys: list[str] = config["input"]["option_keys"]
    prompt_suffix: str = config["input"]["prompt_suffix"]

    system_prompt: str = config["parameters"]["system_prompt"]

    input_file_path = data_loc / data_file
    save_loc.mkdir(parents=True, exist_ok=True)

    from mech_interp_toolkit.activation_utils import get_activations, patch_activations
    from mech_interp_toolkit.utils import (
        load_model_tokenizer_config,
        set_global_seed,
    )

    set_global_seed(seed)

    if model is None or tokenizer is None:
        model, tokenizer, _ = load_model_tokenizer_config(
            model_name=model_name,
            suffix=prompt_suffix,
            system_prompt=system_prompt,
            attn_type="eager",
            dtype=dtype,
        )

    system_prompt_length = (
        len(tokenizer.tokenizer.encode(system_prompt, add_special_tokens=False)) + 1
    )  # For <|im_end|>

    token_id_short = tokenizer.tokenizer.encode(
        option_keys[0], add_special_tokens=False
    )[0]
    token_id_long = tokenizer.tokenizer.encode(
        option_keys[1], add_special_tokens=False
    )[0]

    def chunk_list(prompt_list: list[str]) -> list[list[str]]:
        """Split a list into chunks of size batch_size."""
        return [
            prompt_list[i : i + batch_size]
            for i in range(0, len(prompt_list), batch_size)
        ]

    all_clean_prompts = load_prompts(
        input_file_path,
    )
    all_clean_prompts = chunk_list(all_clean_prompts)

    # Pre-tokenize all batches once per order to avoid redundant work across num_steps iterations
    tokenized_clean = [tokenizer(b) for b in all_clean_prompts]
    layer_components = [x[0] for x in nodes]
    scale_factors = [0.5, 1.5]
    base_metrics_short: list[np.ndarray] = []
    base_metrics_long: list[np.ndarray] = []
    patch_metrics_by_scale = {
        scale_factor: {"short": [], "long": []} for scale_factor in scale_factors
    }

    for batch_dict in tqdm(tokenized_clean, desc="Batches"):
        base_acts, base_logits = get_activations(
            model,
            batch_dict,
            layer_components,
            return_logits=True,
        )
        base_metric = (
            base_logits[:, -1, token_id_short],  # type: ignore
            base_logits[:, -1, token_id_long],  # type: ignore
        )  # type: ignore[index]
        base_metrics_short.append(tensor_to_numpy(base_metric[0]))
        base_metrics_long.append(tensor_to_numpy(base_metric[1]))

        for scale_factor in scale_factors:
            patch_acts = base_acts.clone()
            scale_selected_nodes(patch_acts, nodes, scale_factor=scale_factor)

            _, patch_logits = patch_activations(
                model,
                batch_dict,
                [(0, "attn")],
                patch_acts,
                return_logits=True,
            )
            patch_metric = (
                patch_logits[:, -1, token_id_short],  # type: ignore
                patch_logits[:, -1, token_id_long],  # type: ignore
            )  # type: ignore[index]
            patch_metrics_by_scale[scale_factor]["short"].append(
                tensor_to_numpy(patch_metric[0])
            )
            patch_metrics_by_scale[scale_factor]["long"].append(
                tensor_to_numpy(patch_metric[1])
            )

    output_arrays: dict[str, np.ndarray] = {
        "base_metric_short": np.concatenate(base_metrics_short, axis=0),
        "base_metric_long": np.concatenate(base_metrics_long, axis=0),
        "metadata__config_json": np.array(json.dumps(config, sort_keys=True)),
        "metadata__data_file": np.array(data_file),
        "metadata__node_classes_json": np.array(json.dumps(node_classes)),
        "metadata__node_count": np.array(len(nodes)),
        "metadata__option_keys_json": np.array(json.dumps(option_keys)),
        "metadata__scale_factors_json": np.array(json.dumps(scale_factors)),
    }

    for scale_factor, patch_metrics in patch_metrics_by_scale.items():
        scale_suffix = str(scale_factor).replace(".", "_")
        output_arrays[f"patch_metric_short_scale_{scale_suffix}"] = np.concatenate(
            patch_metrics["short"], axis=0
        )
        output_arrays[f"patch_metric_long_scale_{scale_suffix}"] = np.concatenate(
            patch_metrics["long"], axis=0
        )

    data_stem = sanitize_filename(Path(data_file).stem)
    node_class_stem = sanitize_filename("__".join(node_classes))
    output_file = (
        save_loc / f"cross_task_generalization__{data_stem}__{node_class_stem}.npz"
    )
    np.savez_compressed(output_file, **output_arrays)

    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run!")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file (e.g., step_numbers.yaml)",
    )

    parser.add_argument(
        "--node-class",
        type=str,
        nargs="+",
        required=True,
        help="class of nodes to patch",
    )

    args = parser.parse_args()
    run_cross_task_generalization(args.config, args.node_class)


if __name__ == "__main__":
    main()
