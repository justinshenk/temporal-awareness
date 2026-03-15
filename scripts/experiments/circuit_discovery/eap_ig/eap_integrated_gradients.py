"""
Run EAP Integrated Gradients on clean vs corrupted prompts and save scores to NPZ.
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi
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
HF_REPO_ID = "Temporal_Awareness_EAP_IG"


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
    layer_components = config["setup"].get("layer_components", None)
    granularity = config["setup"].get("granularity", "coarse")

    dtype = config["setup"].get("dtype", None)

    data_loc: Path = Path(config["paths"]["data_loc"])
    save_loc: Path = Path(config["paths"]["save_loc"])

    data_file: str = config["input"]["data_file"]
    template = config["input"]["template"]
    option_keys: list[str] = config["input"]["option_keys"]
    prompt_suffix: str = config["input"]["prompt_suffix"]

    filename: str = config["output"]["filename"]
    hf_repo_id: str = HF_REPO_ID
    hf_repo_type: str = config["output"].get("hf_repo_type", "dataset")

    system_prompt: str = config["parameters"]["system_prompt"]
    metric_type: str = config["parameters"]["metric_type"]
    steps: list[int] = config["parameters"]["steps"]

    input_file_path = data_loc / data_file
    save_loc.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required for Hub uploads.")
    hf_api = HfApi(token=hf_token)
    hf_api.create_repo(repo_id=hf_repo_id, repo_type=hf_repo_type, exist_ok=True)

    def _upload_to_hf(local_file: Path, path_in_repo: str) -> None:
        file_size = local_file.stat().st_size
        print(
            f"[HF upload] Starting file={local_file.name} "
            f"local_path={local_file} repo_path={path_in_repo} "
            f"size_gb={file_size / (1024 ** 3):.2f}",
            flush=True,
        )
        hf_api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=path_in_repo,
            repo_id=hf_repo_id,
            repo_type=hf_repo_type,
            commit_message=f"Upload {path_in_repo}",
        )
        print(
            f"[HF upload] Completed file={local_file.name}",
            flush=True,
        )

    from mech_interp_toolkit.activation_dict import expand_mask
    from mech_interp_toolkit.gradient_based_attribution import (
        eap_integrated_gradients,
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

    n_layers = model.config.num_hidden_layers

    if layer_components is None:
        if granularity == "coarse":
            layer_components = [
                (layer, component)
                for layer in range(n_layers)
                for component in ("attn", "mlp")
            ]
        elif granularity == "fine":
            layer_components = [
                (layer, component)
                for layer in range(n_layers)
                for component in ("z", "mlp_hidden")
            ]
        else:
            raise ValueError(f"Invalid granularity: {granularity}")
    else:
        layer_components = [tuple(lc) for lc in layer_components]

    system_prompt_length = (
        len(tokenizer.tokenizer.encode(system_prompt, add_special_tokens=False)) + 1
    )  # For <|im_end|>

    token_a = tokenizer.tokenizer.encode(
        extract_alnum(option_keys[0]), add_special_tokens=False
    )[0]
    token_b = tokenizer.tokenizer.encode(
        extract_alnum(option_keys[1]), add_special_tokens=False
    )[0]

    metrics = {
        "logit_A": lambda logits: logits[:, -1, token_a].mean(),
        "logit_B": lambda logits: logits[:, -1, token_b].mean(),
    }

    def chunk_list(prompt_list: list[str]) -> list[list[str]]:
        """Split a list into chunks of size batch_size."""
        return [
            prompt_list[i : i + batch_size]
            for i in range(0, len(prompt_list), batch_size)
        ]

    # Load the data file once and generate both clean and swapped prompts.
    # "normal" order: short-term option first (option_keys as-is).
    # "swapped" order: long-term option first (option_keys reversed).
    all_clean_prompts, all_corrupted_prompts = load_and_merge_pairs(
        input_file_path,
        template=template,
        option_keys=option_keys,
        text_order=["question", "immediate", "long_term"],
    )

    # Reverse order is an artifact of data syntax
    all_corrupted_prompts_swapped, all_clean_prompts_swapped = load_and_merge_pairs(
        input_file_path,
        template=template,
        option_keys=option_keys,
        text_order=["question", "long_term", "immediate"],
    )

    option_orders = [
        (
            "short_first",
            chunk_list(all_clean_prompts),
            chunk_list(all_corrupted_prompts),
        ),
        (
            "long_first",
            chunk_list(all_clean_prompts_swapped),
            chunk_list(all_corrupted_prompts_swapped),
        ),
    ]

    for (
        order_label,
        chunked_clean_prompts,
        chunked_corrupted_prompts,
    ) in option_orders:
        # Pre-tokenize all batches once per order to avoid redundant work across num_steps iterations
        tokenized_clean = [tokenizer(b) for b in chunked_clean_prompts]
        tokenized_corrupted = [tokenizer(b) for b in chunked_corrupted_prompts]

        for metric_label, metric_fn in metrics.items():
            batch_outputs: list[dict[str, np.ndarray]] = []
            for _ in range(len(tokenized_clean)):
                batch_output: dict[str, np.ndarray] = {}
                batch_output["metadata__config_json"] = np.array(
                    json.dumps(config), dtype=np.str_
                )
                batch_output["metadata__option_order"] = np.array(
                    order_label, dtype=np.str_
                )
                batch_output["metadata__metric_type"] = np.array(
                    metric_type, dtype=np.str_
                )
                batch_outputs.append(batch_output)

            for num_steps in tqdm(
                steps, desc=f"[{order_label}/{metric_label}] Processing step counts"
            ):
                for i in tqdm(
                    range(len(tokenized_clean)),
                    desc=f"Batches (steps={num_steps})",
                    leave=False,
                ):
                    # Deep-copy tensors to prevent get_embeddings_dict from mutating the
                    # pre-tokenized dicts (it pops input_ids and injects inputs_embeds
                    # in-place, causing stale GPU tensors with live graphs to accumulate).
                    clean_inputs = {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in tokenized_clean[i].items()
                    }
                    corrupted_inputs = {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in tokenized_corrupted[i].items()
                    }

                    eap_ig_scores, (clean_logits, corrupted_logits) = (
                        eap_integrated_gradients(  # (batch, pos) | (batch, pos, n_head) | (batch, pos, neuron)
                            model,  # type: ignore
                            clean_inputs,
                            corrupted_inputs,
                            metric_fn,
                            layer_components,
                            num_steps,
                            include_block_outputs=True,
                        )
                    )

                    clean_logits_cpu = (
                        clean_logits[:, -1, [token_a, token_b]].detach().cpu()  # type: ignore[index]
                    )
                    corrupted_logits_cpu = (
                        corrupted_logits[:, -1, [token_a, token_b]].detach().cpu()  # type: ignore[index]
                    )
                    del clean_logits, corrupted_logits

                    eap_ig_scores.attention_mask = expand_mask(
                        eap_ig_scores.attention_mask, system_prompt_length
                    )
                    eap_ig_scores = eap_ig_scores.apply(
                        torch.nanmean, dim=1, mask_aware=True
                    )  # (batch,) | (batch, n_head) | (batch, neuron)
                    eap_ig_scores = eap_ig_scores.apply(lambda x: x.detach().cpu())

                    for key, value in eap_ig_scores.items():
                        batch_outputs[i][
                            f"step_{num_steps}__{key[1]}__{key[0]}"
                        ] = tensor_to_numpy(value)

                    batch_outputs[i][f"step_{num_steps}__clean_logits"] = (
                        clean_logits_cpu.float().numpy()
                    )
                    batch_outputs[i][f"step_{num_steps}__corrupted_logits"] = (
                        corrupted_logits_cpu.float().numpy()
                    )

                    # Delete temporary objects to free memory
                    del (
                        eap_ig_scores,
                        clean_inputs,
                        corrupted_inputs,
                        clean_logits_cpu,
                        corrupted_logits_cpu,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            for batch_idx, output_arrays in enumerate(batch_outputs):
                output_file = save_loc / (
                    f"{filename}_{order_label}_{metric_label}_batch_{batch_idx:05d}.npz"
                )
                output_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output_file, **output_arrays)
                output_file_abs = output_file.resolve()
                try:
                    path_in_repo = output_file_abs.relative_to(
                        Path.cwd().resolve()
                    ).as_posix()
                except ValueError:
                    path_in_repo = output_file.name
                _upload_to_hf(output_file_abs, path_in_repo)


if __name__ == "__main__":
    main()
