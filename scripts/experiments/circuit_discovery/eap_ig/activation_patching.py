"""Patch selected nodes and save one plot per QnA node group."""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from dotenv import load_dotenv

load_dotenv()
CONFIG_PATH = Path(__file__).parent / "config"
DEFAULT_QNA_NODES_PATH = Path("data/selected_nodes/QnA_sufficient(200).pkl")
torch.set_grad_enabled(False)

LayerComponent = tuple[int, str]
LayerComponentNode = tuple[LayerComponent, int]


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array, normalizing unsupported dtypes."""
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.to(torch.float32)
    return cpu_tensor.numpy()


def load_and_merge_pairs(
    input_file: Path,
    template: str,
    option_keys: list[str],
    text_order: list[str],
) -> tuple[list[str], list[str]]:
    """Load pairs from ``input_file`` and return both clean and swapped prompts."""
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clean_prompts: list[str] = []
    swapped_prompts: list[str] = []
    option_a, option_b = option_keys
    pairs = data.get("pairs", [])

    # The regex is robust for cases where one option might be a substring of another.
    for pair in pairs:
        if isinstance(pair, str):
            prompt = pair
        elif isinstance(pair, dict):
            prompt = template.format(
                pair.get(text_order[0], ""),
                pair.get(text_order[1], ""),
                pair.get(text_order[2], ""),
            )
        else:
            raise RuntimeError("Incorrect type for pairs")

        prompt = prompt.replace("(A)", option_a)
        prompt = prompt.replace("(B)", option_b)

        clean_prompts.append(prompt)

        swapped_prompt = re.sub(
            f"{re.escape(option_a)}|{re.escape(option_b)}",
            lambda m: option_b if m.group(0) == option_a else option_a,
            prompt,
        )
        swapped_prompts.append(swapped_prompt)

    return clean_prompts, swapped_prompts


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_alnum(value: str) -> str:
    """Extract the first alphanumeric character from a string."""
    for char in value:
        if char.isalnum():
            return char
    raise ValueError(f"malformed option string {value}")


def chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    """Split a list into equally sized chunks."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def str_to_layer_component_node(node_spec: str) -> LayerComponentNode:
    """Parse a node spec like ``z/12_4`` into structured identifiers."""
    component, remainder = node_spec.split("/")
    layer, node = remainder.rsplit("_", 1)
    return (int(layer), component), int(node)


def unique_layer_components(
    layer_component_nodes: list[LayerComponentNode],
) -> list[LayerComponent]:
    """Preserve order while dropping duplicate layer/component requests."""
    return list(
        dict.fromkeys(layer_component for layer_component, _ in layer_component_nodes)
    )


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


def sanitize_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "plot"


def save_scatter_plot(
    base_metric: tuple[torch.Tensor, torch.Tensor],
    patch_metric: tuple[torch.Tensor, torch.Tensor],
    node_group_name: str,
    save_path: Path,
) -> None:
    """Save a single base-vs-patched scatter plot for one node group."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        tensor_to_numpy(base_metric[0]),
        tensor_to_numpy(base_metric[1]),
        label="Base",
        alpha=0.8,
    )
    ax.scatter(
        tensor_to_numpy(patch_metric[0]),
        tensor_to_numpy(patch_metric[1]),
        label="Patch",
        alpha=0.8,
    )
    ax.set_title(node_group_name)
    ax.set_xlabel("logit_A")
    ax.set_ylabel("logit_B")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def load_qna_nodes(nodes_path: Path) -> dict[str, list[str]]:
    """Load the selected node groups from pickle."""
    with nodes_path.open("rb") as f:
        qna_nodes = pickle.load(f)

    if not isinstance(qna_nodes, dict):
        raise TypeError(
            f"Expected {nodes_path} to contain a dict, got {type(qna_nodes)!r}"
        )

    return qna_nodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch selected nodes and save one plot per QnA node group."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file relative to the local config directory.",
    )
    parser.add_argument(
        "--nodes-path",
        type=Path,
        default=DEFAULT_QNA_NODES_PATH,
        help="Path to a pickle file containing qna node groups.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for where plots should be written.",
    )
    args = parser.parse_args()

    config = load_config(CONFIG_PATH / args.config)

    model_name: str = config["setup"]["model"]
    seed: int = config["setup"]["seed"]
    batch_size: int = config["setup"]["batch_size"]
    dtype = config["setup"].get("dtype")

    data_loc = Path(config["paths"]["data_loc"])
    save_loc = Path(config["paths"]["save_loc"])
    output_dir = args.output_dir or save_loc / "simple_patch"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file: str = config["input"]["data_file"]
    template: str = config["input"]["template"]
    option_keys: list[str] = config["input"]["option_keys"]
    prompt_suffix: str = config["input"]["prompt_suffix"]
    system_prompt: str = config["parameters"]["system_prompt"]

    input_file_path = data_loc / data_file
    qna_nodes = load_qna_nodes(args.nodes_path)

    try:
        from mech_interp_toolkit.activation_utils import (
            get_activations,
            patch_activations,
        )
        from mech_interp_toolkit.utils import (
            load_model_tokenizer_config,
            set_global_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "mech_interp_toolkit is required. Install it in the active environment "
            "before running this script."
        ) from exc

    set_global_seed(seed)

    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        suffix=prompt_suffix,
        system_prompt=system_prompt,
        attn_type="eager",
        dtype=dtype,
    )

    all_clean_prompts, _ = load_and_merge_pairs(
        input_file_path,
        template=template,
        option_keys=option_keys,
        text_order=["question", "immediate", "long_term"],
    )
    chunked_clean_prompts = chunk_list(all_clean_prompts, batch_size)
    input_dict = tokenizer(chunked_clean_prompts[0])

    token_a = tokenizer.tokenizer.encode(
        extract_alnum(option_keys[0]),
        add_special_tokens=False,
    )[0]
    token_b = tokenizer.tokenizer.encode(
        extract_alnum(option_keys[1]),
        add_special_tokens=False,
    )[0]

    for node_group_name, node_specs in qna_nodes.items():
        if not node_specs:
            continue

        layer_component_nodes = [
            str_to_layer_component_node(node_spec) for node_spec in node_specs
        ]
        layer_components = unique_layer_components(layer_component_nodes)

        base_acts, base_logits = get_activations(
            model,
            input_dict,
            layer_components,
            return_logits=True,
        )
        base_metric = (base_logits[:, -1, token_a], base_logits[:, -1, token_b])  # type: ignore[index]

        scale_selected_nodes(base_acts, layer_component_nodes)

        _, patch_logits = patch_activations(
            model,
            input_dict,
            layer_components,
            base_acts,
            return_logits=True,
        )
        patch_metric = (
            patch_logits[:, -1, token_a],  # type: ignore
            patch_logits[:, -1, token_b],  # type: ignore
        )  # type: ignore[index]

        save_scatter_plot(
            base_metric=base_metric,
            patch_metric=patch_metric,
            node_group_name=node_group_name,
            save_path=output_dir / f"{sanitize_filename(node_group_name)}.png",
        )


if __name__ == "__main__":
    main()
