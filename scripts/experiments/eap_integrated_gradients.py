"""
Run EAP Integrated Gradients on clean vs corrupted prompts and save scores to JSON.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "temporal_scope_caa.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "eap_integrated_gradients"
RESULTS_PREFIX = "test_"

OPTION_KEYS = ["A", "B"]
NUM_STEPS = 10
NUM_SAMPLES = 5  # Set to an integer to limit the number of samples, or None to use all
output_file = RESULTS_DIR / f"{RESULTS_PREFIX}eap_ig_scores.json"

# Template that stitches question, immediate, and long-term parts into one prompt.
TEMPLATE = "{}\n\n{}\n{}"


def ensure_mech_interp_toolkit_installed() -> None:
    """Install mech_interp_toolkit if it is not already available."""
    if importlib.util.find_spec("mech_interp_toolkit") is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "mech_interp_toolkit"]
        )


def load_and_merge_pairs(
    input_file: Path, swap: bool = False, num_samples: int | None = None
) -> list:
    """Load pairs from ``input_file`` and merge them into formatted prompts.

    Args:
        input_file: Path to JSON file containing question pairs
        swap: If True, swap option keys (A) and (B) in the prompts
        num_samples: Maximum number of samples to load, or None to load all

    Returns:
        List of formatted prompt strings
    """
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    option_a, option_b = OPTION_KEYS
    pairs = data.get("pairs", [])

    # Limit the number of pairs if num_samples is specified
    if num_samples is not None:
        pairs = pairs[:num_samples]

    for pair in pairs:
        prompt = TEMPLATE.format(
            pair.get("question", ""),
            pair.get("immediate", ""),
            pair.get("long_term", ""),
        )
        if swap:
            prompt = re.sub(
                f"{f'({option_a})'}|{f'({option_b})'}",
                lambda m: option_b if m.group(0) == option_a else option_a,
                prompt,
            )

        prompts.append(prompt)

    return prompts


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_mech_interp_toolkit_installed()

    from mech_interp_toolkit.gradient_based_attribution import (
        eap_integrated_gradients,
    )
    from mech_interp_toolkit.utils import (
        load_model_tokenizer_config,
        set_global_seed,
    )

    set_global_seed()
    torch.set_grad_enabled(True)

    system_prompt = "You are a strategic planning assistant that follows user instructions carefully"

    model, tokenizer, _ = load_model_tokenizer_config(
        MODEL_NAME,
        suffix="Answer: (",
        system_prompt=system_prompt,
        attn_type="sdpa",
    )

    clean_prompts = load_and_merge_pairs(
        INPUT_FILE, swap=False, num_samples=NUM_SAMPLES
    )
    corrupted_prompts = load_and_merge_pairs(
        INPUT_FILE, swap=True, num_samples=NUM_SAMPLES
    )

    clean_inputs = tokenizer(clean_prompts)
    corrupted_inputs = tokenizer(corrupted_prompts)

    token_a = tokenizer.tokenizer.encode(OPTION_KEYS[0], add_special_tokens=False)
    token_b = tokenizer.tokenizer.encode(OPTION_KEYS[1], add_special_tokens=False)

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        return (logits[:, -1, token_a] - logits[:, -1, token_b]).sum()

    eap_ig_scores = eap_integrated_gradients(
        model,
        clean_inputs,
        corrupted_inputs,
        metric_fn,
        NUM_STEPS,
    )

    scores_dict = {str(key): value.item() for key, value in eap_ig_scores.items()}
    output_file.write_text(json.dumps(scores_dict, indent=2))


if __name__ == "__main__":
    main()
