"""
Run EAP Integrated Gradients on clean vs corrupted prompts and save scores to JSON.
"""

import argparse
import gc
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

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
    if layer_components is not None:
        layer_components = [tuple(lc) for lc in layer_components]

    dtype = config["setup"].get("dtype", None)

    data_loc: Path = Path(config["paths"]["data_loc"])
    save_loc: Path = Path(config["paths"]["save_loc"])

    data_file: str = config["input"]["data_file"]
    template = config["input"]["template"]
    option_keys: list[str] = config["input"]["option_keys"]
    prompt_suffix: str = config["input"]["prompt_suffix"]

    filename: str = config["output"]["filename"]

    system_prompt: str = config["parameters"]["system_prompt"]
    metric_type: str = config["parameters"]["metric_type"]
    steps: list[int] = config["parameters"]["steps"]

    input_file_path = data_loc / data_file
    save_loc.mkdir(parents=True, exist_ok=True)

    from mech_interp_toolkit.activation_dict import concat_activations, expand_mask
    from mech_interp_toolkit.gradient_based_attribution import (
        eap_integrated_gradients,
    )
    from mech_interp_toolkit.utils import (
        load_model_tokenizer_config,
        set_global_seed,
    )

    set_global_seed(seed)
    torch.set_grad_enabled(False)

    # Suffix prompts the model to complete with option character
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

    for order_label, chunked_clean_prompts, chunked_corrupted_prompts in option_orders:
        # Pre-tokenize all batches once per order to avoid redundant work across num_steps iterations
        tokenized_clean = [tokenizer(b) for b in chunked_clean_prompts]
        tokenized_corrupted = [tokenizer(b) for b in chunked_corrupted_prompts]

        for metric_label, metric_fn in metrics.items():
            output_file = save_loc / f"{filename}_{order_label}_{metric_label}.json"
            output_dict = config.copy()
            output_dict["option_order"] = order_label
            output_dict["metric_type"] = metric_type
            output_dict["steps"] = {}

            for num_steps in tqdm(
                steps, desc=f"[{order_label}/{metric_label}] Processing step counts"
            ):
                scores_list = []
                all_clean_logits = []
                all_corrupted_logits = []
                for i in tqdm(
                    range(len(tokenized_clean)),
                    desc=f"Batches (steps={num_steps})",
                    leave=False,
                ):
                    model.zero_grad(set_to_none=True)
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

                    all_clean_logits.append(clean_logits_cpu)
                    all_corrupted_logits.append(corrupted_logits_cpu)
                    del clean_logits_cpu, corrupted_logits_cpu

                    eap_ig_scores.attention_mask = expand_mask(
                        eap_ig_scores.attention_mask, system_prompt_length
                    )
                    eap_ig_scores = eap_ig_scores.apply(
                        torch.nanmean, dim=1, mask_aware=True
                    )  # (batch,) | (batch, n_head) | (batch, neuron)

                    scores_list.append(eap_ig_scores)

                    # Delete temporary objects to free memory
                    del eap_ig_scores, clean_inputs, corrupted_inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                scores = concat_activations(scores_list)
                raw_scores_dict = {
                    f"{key[1]}/{key[0]}": value.tolist()
                    for key, value in scores.items()
                }

                clean_logits_list = torch.cat(all_clean_logits, dim=0).tolist()
                corrupted_logits_list = torch.cat(all_corrupted_logits, dim=0).tolist()

                output_dict["steps"][num_steps] = {
                    **raw_scores_dict,
                    "clean_logits": clean_logits_list,
                    "corrupted_logits": corrupted_logits_list,
                }

                # Free memory between num_steps iterations
                del scores, scores_list, all_clean_logits, all_corrupted_logits
                del raw_scores_dict, clean_logits_list, corrupted_logits_list
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Write complete results with all steps to file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(output_dict, indent=2))


if __name__ == "__main__":
    main()
