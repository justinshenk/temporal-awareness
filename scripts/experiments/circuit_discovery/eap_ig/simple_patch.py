"""
Run EAP Integrated Gradients on clean vs corrupted prompts and save scores to NPZ.
"""

import argparse
import json
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from dotenv import load_dotenv

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
    """Load pairs from ``input_file`` and return both clean and swapped prompts.

    Args:
        input_file: Path to JSON file containing question pairs
        template: Template string for formatting prompts
        option_keys: List of option keys to use
        text_order: List of keys specifying the order in which to extract fields
            from each pair dict (e.g. ``["question", "immediate", "long_term"]``)

    Returns:
        Tuple of (clean_prompts, swapped_prompts)
    """
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clean_prompts = []
    swapped_prompts = []
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


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary with configuration values
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_alnum(s: str) -> str:
    """Extract the first alphanumeric character from a string.

    Args:
        s: Input string to search

    Returns:
        First alphanumeric character found

    Raises:
        ValueError: If no alphanumeric character is found
    """
    for c in s:
        if c.isalnum():
            return c
    raise ValueError(f"malformed option string {s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EAP Integrated Gradients on clean vs corrupted prompts"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file (e.g., step_numbers.yaml)",
    )

    args = parser.parse_args()

    config = load_config(CONFIG_PATH / args.config)

    model_name: str = config["setup"]["model"]
    seed: int = config["setup"]["seed"]
    batch_size: int = config["setup"]["batch_size"]

    dtype = config["setup"].get("dtype", None)

    data_loc: Path = Path(config["paths"]["data_loc"])
    save_loc: Path = Path(config["paths"]["save_loc"])

    data_file: str = config["input"]["data_file"]
    template = config["input"]["template"]
    option_keys: list[str] = config["input"]["option_keys"]
    prompt_suffix: str = config["input"]["prompt_suffix"]

    system_prompt: str = config["parameters"]["system_prompt"]

    input_file_path = data_loc / data_file
    save_loc.mkdir(parents=True, exist_ok=True)

    with open(
        "/content/temporal-awareness/data/selected_nodes/QnA_sufficient(200).pkl", "rb"
    ) as f:
        qna_nodes = pickle.load(f)

    from mech_interp_toolkit.activation_utils import (
        get_activations,
        patch_activations,
    )
    from mech_interp_toolkit.utils import (
        load_model_tokenizer_config,
        set_global_seed,
    )

    set_global_seed(seed)

    # Suffix prompts the model to complete with option character
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

    def chunk_list(prompt_list: list[str]) -> list[list[str]]:
        """Split a list into chunks of size batch_size."""
        return [
            prompt_list[i : i + batch_size]
            for i in range(0, len(prompt_list), batch_size)
        ]

    def str_to_layer_components(s):
        comp, rem = s.split("/")
        layer, node = rem.rsplit("_", 1)
        return (int(layer), comp), int(node)

    chunked_clean_prompts = chunk_list(all_clean_prompts)
    token_a = tokenizer.tokenizer.encode(
        extract_alnum(option_keys[0]), add_special_tokens=False
    )[0]
    token_b = tokenizer.tokenizer.encode(
        extract_alnum(option_keys[1]), add_special_tokens=False
    )[0]
    ## cache acts

    input_dict = tokenizer(chunked_clean_prompts[0])

    layer_components_nodes = [
        str_to_layer_components(x) for x in qna_nodes["ST_n_suffcient"]
    ]

    layer_components = [x[0] for x in layer_components_nodes]

    base_acts, base_logits = get_activations(
        model,
        input_dict,
        layer_components,
        return_logits=True,
    )

    base_acts.split_heads()

    base_metric = (base_logits[:, -1, token_a], base_logits[:, -1, token_b])  # type: ignore

    for (layer, component), node in layer_components_nodes:
        if component == "z":
            base_acts[(layer, component)][:, :, node, :] = (
                base_acts[(layer, component)][:, :, node, :] * 2
            )

        if component == "mlp_hidden":
            base_acts[(layer, component)][:, :, node] = (
                base_acts[(layer, component)][:, :, node] * 2
            )

    base_acts.merge_heads()

    _, patch_logits = patch_activations(
        model,
        input_dict,
        layer_components,
        base_acts,
        return_logits=True,
    )

    patch_metric = (patch_logits[:, -1, token_a], patch_logits[:, -1, token_b])  # type: ignore

    plt.scatter(
        base_metric[0].cpu().float(), base_metric[1].cpu().float(), label="Base"
    )
    plt.xlabel("logit_A")
    plt.ylabel("logit_B")
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.scatter(
        patch_metric[0].cpu().float(), patch_metric[1].cpu().float(), label="Patch"
    )
    plt.legend()
    plt.savefig("/content/temporal-awareness/fig.png")
    plt.show()


if __name__ == "__main__":
    main()
