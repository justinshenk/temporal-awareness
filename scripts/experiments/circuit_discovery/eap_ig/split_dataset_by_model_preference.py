"""
Split a dataset into subsets based on model token preference (option A, option B, ambiguous).
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from mech_interp_toolkit.utils import load_model_tokenizer_config
from tqdm import tqdm

warnings.filterwarnings("ignore")

load_dotenv()
CONFIG_PATH = Path(__file__).parent / "config"


def load_clean_prompts(input_file: Path, template: str) -> list[str]:
    """Load pairs from ``input_file`` and return formatted clean prompts.

    Args:
        input_file: Path to JSON file containing question pairs
        template: Template string for formatting prompts

    Returns:
        List of formatted prompt strings
    """
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        template.format(
            pair.get("question", ""),
            pair.get("immediate", ""),
            pair.get("long_term", ""),
        )
        for pair in data.get("pairs", [])
    ]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_alnum(s: str) -> str:
    """Return the first alphanumeric character in *s*, or raise ValueError."""
    for c in s:
        if c.isalnum():
            return c
    raise ValueError(f"No alphanumeric character found in option string {s!r}")


def save_split(prompts: list[str], path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "pairs": prompts}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a dataset by model token preference"
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
    batch_size: int = config["setup"]["batch_size"]
    dtype = config["setup"].get("dtype", None)

    data_loc: Path = Path(config["paths"]["data_loc"])

    data_file: str = config["input"]["data_file"]
    template: str = config["input"]["template"]
    option_keys: list[str] = config["input"]["option_keys"]
    prompt_suffix: str = config["input"]["prompt_suffix"]
    option_a_horizon, option_b_horizon = config["input"]["horizon"]

    system_prompt: str = config["parameters"]["system_prompt"]

    input_file_path = data_loc / data_file
    save_loc = data_loc.parent / "stratified_dataset"

    torch.set_grad_enabled(False)

    # Suffix prompts the model to complete with option character
    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        suffix=prompt_suffix,
        system_prompt=system_prompt,
        attn_type="sdpa",
        dtype=dtype,
    )

    token_a = tokenizer.tokenizer.encode(
        extract_alnum(option_keys[0]), add_special_tokens=False
    )[0]
    token_b = tokenizer.tokenizer.encode(
        extract_alnum(option_keys[1]), add_special_tokens=False
    )[0]

    def extract_preference(logits: torch.Tensor) -> tuple[bool, bool]:
        logit_a_preferred = (logits[token_a] > logits[token_b]).item()
        _, topk = torch.topk(logits, k=7)
        clear_preference = torch.any((topk == token_a) | (topk == token_b)).item()
        return logit_a_preferred, clear_preference  # type: ignore

    all_clean_prompts = load_clean_prompts(input_file_path, template=template)

    chunks = [
        all_clean_prompts[i : i + batch_size]
        for i in range(0, len(all_clean_prompts), batch_size)
    ]

    a_preferred = []
    b_preferred = []
    ambiguous = []

    for i, chunk in enumerate(tqdm(chunks)):
        clean_inputs = tokenizer(chunk)
        batch_logits = model(**clean_inputs)  # type: ignore

        for j, all_pos_logits in enumerate(batch_logits):
            logits = all_pos_logits[-1, :]  # (d_vocab,)
            logit_a_preferred, clear_preference = extract_preference(logits)

            prompt = all_clean_prompts[i * batch_size + j]
            if clear_preference:
                (a_preferred if logit_a_preferred else b_preferred).append(prompt)
            else:
                ambiguous.append(prompt)

    base_metadata = {
        "model": model_name,
        "source_file": data_file,
        "n_total": len(all_clean_prompts),
        "n_a_preferred": len(a_preferred),
        "n_b_preferred": len(b_preferred),
        "n_ambiguous": len(ambiguous),
        "option_a_horizon": option_a_horizon,
        "option_b_horizon": option_b_horizon,
    }

    if a_preferred:
        save_split(
            a_preferred,
            save_loc / option_a_horizon / data_file,
            {**base_metadata, "split": option_a_horizon, "n_pairs": len(a_preferred)},
        )
    if b_preferred:
        save_split(
            b_preferred,
            save_loc / option_b_horizon / data_file,
            {**base_metadata, "split": option_b_horizon, "n_pairs": len(b_preferred)},
        )
    if ambiguous:
        save_split(
            ambiguous,
            save_loc / "ambiguous" / data_file,
            {**base_metadata, "split": "ambiguous", "n_pairs": len(ambiguous)},
        )


if __name__ == "__main__":
    main()
