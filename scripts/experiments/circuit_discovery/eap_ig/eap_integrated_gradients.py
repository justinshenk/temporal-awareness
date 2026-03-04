"""
Run EAP Integrated Gradients on clean vs corrupted prompts and save scores to JSON.
"""

import argparse
import gc
import importlib.util
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import cast

import torch
import wandb
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")


load_dotenv()
CONFIG_PATH = Path(__file__).parent / "config"


def ensure_mech_interp_toolkit_installed() -> None:
    """Install mech_interp_toolkit if it is not already available."""
    if importlib.util.find_spec("mech_interp_toolkit") is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "mech_interp_toolkit"]
        )


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

        prompt.replace("(A)", option_a)
        prompt.replace("(B)", option_b)

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

    wandb_project: str = config["output"]["wandb_project"]
    filename: str = config["output"]["filename"]

    system_prompt: str = config["parameters"]["system_prompt"]
    metric_type: str = config["parameters"]["metric_type"]
    steps: list[int] = config["parameters"]["steps"]

    config_stem = args.config.stem

    input_file_path = data_loc / data_file
    save_loc.mkdir(parents=True, exist_ok=True)
    ensure_mech_interp_toolkit_installed()

    # Clear stdout and print success message
    sys.stdout.write("\033[2J\033[H")  # Clear screen and move cursor to top
    sys.stdout.flush()
    print("mech_interp_toolkit installed successfully")

    from mech_interp_toolkit.activation_utils import concat_activations, expand_mask
    from mech_interp_toolkit.gradient_based_attribution import (
        eap_integrated_gradients,
    )
    from mech_interp_toolkit.utils import (
        load_model_tokenizer_config,
        set_global_seed,
    )

    set_global_seed(seed)
    torch.set_grad_enabled(True)

    # Suffix prompts the model to complete with option character
    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        suffix=prompt_suffix,
        system_prompt=system_prompt,
        attn_type="sdpa",
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
        for metric_label, metric_fn in metrics.items():
            run_name = f"{config_stem}_{order_label}_{metric_label}"
            with wandb.init(project=wandb_project, name=run_name, config=config) as run:
                run_id = run.id
                output_file = (
                    save_loc
                    / f"{wandb_project}_{filename}_{order_label}_{metric_label}_{run_id}.json"
                )
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
                        range(len(chunked_clean_prompts)),
                        desc=f"Batches (steps={num_steps})",
                        leave=False,
                    ):
                        clean_inputs = tokenizer(chunked_clean_prompts[i])
                        corrupted_inputs = tokenizer(chunked_corrupted_prompts[i])

                        eap_ig_scores, (clean_logits, corrupted_logits) = (
                            eap_integrated_gradients(  # (batch, pos)
                                model,  # type: ignore
                                clean_inputs,
                                corrupted_inputs,
                                metric_fn,
                                num_steps,
                                include_block_outputs=True,
                            )
                        )

                        clean_logits = cast(torch.Tensor, clean_logits)
                        corrupted_logits = cast(torch.Tensor, corrupted_logits)

                        clean_logits = clean_logits[:, -1, [token_a, token_b]]
                        corrupted_logits = corrupted_logits[:, -1, [token_a, token_b]]

                        all_clean_logits.append(clean_logits.detach().cpu())
                        all_corrupted_logits.append(corrupted_logits.detach().cpu())

                        eap_ig_scores.attention_mask = expand_mask(
                            eap_ig_scores.attention_mask, system_prompt_length
                        )
                        eap_ig_scores = eap_ig_scores.apply(
                            torch.nanmean, dim=-1, mask_aware=True
                        )  # (batch,)

                        # Move scores to CPU to free GPU memory
                        eap_ig_scores = eap_ig_scores.apply(lambda x: x.detach().cpu())
                        scores_list.append(eap_ig_scores)

                        # Delete temporary objects to free memory
                        del eap_ig_scores
                        del clean_inputs
                        del corrupted_inputs
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    scores = concat_activations(scores_list)
                    raw_scores_dict = {
                        f"{key[1]}/{key[0]}": value.tolist()
                        for key, value in scores.items()
                    }

                    clean_logits_list = torch.cat(all_clean_logits, dim=0).tolist()
                    corrupted_logits_list = torch.cat(
                        all_corrupted_logits, dim=0
                    ).tolist()

                    output_dict["steps"][num_steps] = {
                        **raw_scores_dict,
                        "clean_logits": clean_logits_list,
                        "corrupted_logits": corrupted_logits_list,
                    }

                    run.log(
                        {
                            **raw_scores_dict,
                            "clean_logits": clean_logits_list,
                            "corrupted_logits": corrupted_logits_list,
                            "num_steps": num_steps,
                        }
                    )

                # Write complete results with all steps to file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(json.dumps(output_dict, indent=2))


if __name__ == "__main__":
    main()
