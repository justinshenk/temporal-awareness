"""Generate LLM completions for the feature-geometry prompt dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from mech_interp_toolkit.utils import load_model_tokenizer_config
from tqdm import tqdm

try:
    from .dataset_gen import PromptRecord, generate_task_dataset
except ImportError:
    from dataset_gen import PromptRecord, generate_task_dataset


DEFAULT_OUTPUT_PATH = Path(__file__).with_name("completions.jsonl")


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
) -> tuple[Any, Any]:
    """Load and return a HuggingFace causal LM and tokenizer."""
    model, tokenizer, _ = load_model_tokenizer_config(model_name)
    return model, tokenizer


def iter_batches(
    records: list[PromptRecord],
    batch_size: int,
) -> Iterator[list[PromptRecord]]:
    """Yield contiguous record batches."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def generate_full_strings(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate and decode full prompt-plus-completion strings."""
    inputs = tokenizer(prompts)

    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def write_completions(
    output_path: Path,
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    batch_size: int = 128,
    max_new_tokens: int = 256,
    temperature: float = 0.5,
    top_p: float = 0.95,
    randomize_template: bool = False,
) -> None:
    """Generate completions for the task dataset and write one JSONL record per prompt.

    The ``full_text`` field contains the entire decoded model output, including
    the original prompt and generated continuation.
    """
    records = generate_task_dataset(randomize_template=randomize_template)
    model, tokenizer = load_model_and_tokenizer(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        batches = iter_batches(records, batch_size)
        total_batches = (len(records) + batch_size - 1) // batch_size
        for batch in tqdm(batches, total=total_batches, desc="Generating completions"):
            prompts = [record["text"] for record in batch]
            full_strings = generate_full_strings(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            for record, full_string in zip(batch, full_strings, strict=True):
                prompt_metadata = {
                    key: value for key, value in record.items() if key != "text"
                }
                output_file.write(
                    json.dumps(
                        {
                            "prompt": record["text"],
                            "prompt_metadata": prompt_metadata,
                            "model_name": model_name,
                            "full_text": full_string,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HF model completions for feature-geometry prompts."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"JSONL output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Optional model identifier passed to load_model_and_tokenizer().",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--randomize-template",
        action="store_true",
        help="Randomly select one template per task-parameter sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Use 0 for greedy decoding.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_completions(
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        randomize_template=args.randomize_template,
    )


if __name__ == "__main__":
    main()
