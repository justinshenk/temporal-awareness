"""Activation extraction utilities for RQ1 (Temporal Horizon Detection).

Extracts residual stream activations at every layer from a TransformerLens-backed
ModelRunner for all prompts in the implicit temporal scope dataset.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm


def load_implicit_dataset(path: str | Path) -> tuple[list[dict], dict]:
    """Load the implicit temporal scope dataset.

    Args:
        path: Path to temporal_scope_implicit_backup_300.json

    Returns:
        (pairs, metadata) where pairs is a list of dicts with keys
        {question, immediate, long_term, category} and metadata is the
        dataset-level metadata dict.
    """
    with open(path) as f:
        data = json.load(f)
    return data["pairs"], data["metadata"]


def build_prompts(
    pairs: list[dict],
) -> tuple[list[str], list[str], list[str]]:
    """Build immediate and long-term prompts from contrastive pairs.

    Template: question + choice_text  (raw concatenation, no separator)

    Args:
        pairs: List of pair dicts with {question, immediate, long_term, category}.

    Returns:
        (immediate_prompts, long_term_prompts, categories)
    """
    immediate_prompts = []
    long_term_prompts = []
    categories = []

    for pair in pairs:
        question = pair["question"]
        immediate_prompts.append(pair["question"] + pair["immediate"])
        long_term_prompts.append(pair["question"] + pair["long_term"])
        categories.append(pair["category"])

    return immediate_prompts, long_term_prompts, categories


def extract_activations_for_prompts(
    runner,
    prompts: list[str],
    desc: str = "Extracting",
) -> torch.Tensor:
    """Extract residual stream activations at the last choice token for every layer.

    For chat-template models the formatted sequence ends with:
        ... <choice text> <|im_end|> \\n <|im_start|> assistant \\n
    The token at position -1 is therefore an assistant-turn suffix token, not the
    choice text itself.  We find the last <|im_end|> in the sequence (which closes
    the user turn) and step back one position to land on the final token of the
    choice text.  For tokenizers that do not have <|im_end|> we fall back to -1.

    Args:
        runner: ModelRunner with TransformerLens backend.
        prompts: List of prompt strings.
        desc: tqdm description string.

    Returns:
        Tensor of shape [n_prompts, n_layers, d_model].
    """
    n_layers = runner.n_layers
    all_acts = []

    # Resolve the <|im_end|> token ID once outside the loop.
    _im_end_id = runner._model.tokenizer.convert_tokens_to_ids("<|im_end|>")
    _has_im_end = isinstance(_im_end_id, int) and _im_end_id != runner._model.tokenizer.unk_token_id

    for prompt in tqdm(prompts, desc=desc):
        formatted = runner.apply_chat_template(prompt)
        input_ids = runner.encode(formatted)          # [1, seq_len]
        _, cache  = runner._backend.run_with_cache(
            input_ids,
            names_filter=lambda n: "hook_resid_post" in n,
        )

        if _has_im_end:
            ids_list   = input_ids[0].tolist()
            last_pos   = max(i for i, t in enumerate(ids_list) if t == _im_end_id)
            target_idx = last_pos - 1
            assert target_idx >= 0, (
                "target_idx < 0 — the chat template may have changed shape"
            )
        else:
            target_idx = -1

        layer_acts = torch.stack(
            [cache[f"blocks.{layer}.hook_resid_post"][0, target_idx, :] for layer in range(n_layers)],
            dim=0,
        )  # [n_layers, d_model]
        all_acts.append(layer_acts.cpu())

    return torch.stack(all_acts, dim=0)  # [n_prompts, n_layers, d_model]


def save_extraction_results(
    out_dir: Path,
    acts_immediate: torch.Tensor,
    acts_long_term: torch.Tensor,
    categories: list[str],
    pairs: list[dict],
    model_name: str,
    dataset_path: str | Path,
) -> None:
    """Save extraction outputs to disk.

    Writes:
        activations_immediate.pt   — shape [n_prompts, n_layers, d_model]
        activations_long_term.pt   — shape [n_prompts, n_layers, d_model]
        metadata.json              — model, dataset, shapes, timestamp
        pair_metadata.json         — per-pair questions and categories

    Args:
        out_dir: Output directory (must already exist).
        acts_immediate: Immediate-option activations tensor.
        acts_long_term: Long-term-option activations tensor.
        categories: Per-pair category labels.
        pairs: Original pair dicts.
        model_name: HuggingFace model identifier string.
        dataset_path: Path to the source dataset file.
    """
    out_dir = Path(out_dir)

    torch.save(acts_immediate, out_dir / "activations_immediate.pt")
    torch.save(acts_long_term, out_dir / "activations_long_term.pt")

    metadata = {
        "model_name": model_name,
        "dataset_path": str(dataset_path),
        "n_prompts": acts_immediate.shape[0],
        "n_layers": acts_immediate.shape[1],
        "d_model": acts_immediate.shape[2],
        "shape_immediate": list(acts_immediate.shape),
        "shape_long_term": list(acts_long_term.shape),
        "extracted_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    pair_metadata = [
        {"question": p["question"], "category": p["category"]} for p in pairs
    ]
    with open(out_dir / "pair_metadata.json", "w") as f:
        json.dump(pair_metadata, f, indent=2)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.inference import ModelRunner
    from src.inference.backends import ModelBackend
    from src.intertemporal.common.project_paths import get_experiment_dir

    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
    DATASET_PATH = Path("data/raw/temporal_scope/temporal_scope_implicit_backup_300.json")
    OUT_DIR = get_experiment_dir() / "activation_extraction"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs, metadata = load_implicit_dataset(DATASET_PATH)
    print(f"Loaded {len(pairs)} pairs")

    immediate_prompts, long_term_prompts, categories = build_prompts(pairs)

    runner = ModelRunner(MODEL_NAME, backend=ModelBackend.TRANSFORMERLENS, dtype=torch.float16)

    acts_immediate = extract_activations_for_prompts(runner, immediate_prompts, desc="Immediate")
    acts_long_term = extract_activations_for_prompts(runner, long_term_prompts, desc="Long-term")

    print(f"Immediate shape: {acts_immediate.shape}")
    print(f"Long-term shape: {acts_long_term.shape}")

    save_extraction_results(OUT_DIR, acts_immediate, acts_long_term, categories, pairs, MODEL_NAME, DATASET_PATH)
    print(f"Saved to: {OUT_DIR}")
