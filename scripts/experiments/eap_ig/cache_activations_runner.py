"""Activation-cache execution loop."""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

try:
    from .cache_activations_data import (
        get_sample_text,
        get_time_horizon_label,
        get_time_horizon_months,
        load_prompt_dataset,
    )
    from .cache_activations_extract import (
        extract_selected_activations,
        maybe_average_positions,
    )
    from .cache_activations_nodes import (
        get_unique_layer_components,
        load_selected_node_groups,
    )
    from .cache_activations_paths import chunk_list
    from .cache_activations_upload import upload_cache_folder
except ImportError:
    from cache_activations_data import (
        get_sample_text,
        get_time_horizon_label,
        get_time_horizon_months,
        load_prompt_dataset,
    )
    from cache_activations_extract import (
        extract_selected_activations,
        maybe_average_positions,
    )
    from cache_activations_nodes import (
        get_unique_layer_components,
        load_selected_node_groups,
    )
    from cache_activations_paths import chunk_list
    from cache_activations_upload import upload_cache_folder


def selected_nodes_include_attention_heads(selected_node_groups: dict) -> bool:
    """Return whether selected nodes include z attention-head entries."""
    return any(
        component == "z"
        for group_nodes in selected_node_groups.values()
        for (_, component), _ in group_nodes
    )


def cache_prompt_activations(
    *,
    dataset: str,
    model_name: str,
    nodes_path: Path,
    output_dir: Path,
    batch_size: int,
    dtype: str | None,
    device: str | None,
    attn_type: str,
    max_samples: int | None,
    save_to_hf: bool,
    hf_repo_id: str,
    hf_repo_type: str,
    average_positions: bool,
) -> None:
    """Load a model with mech_interp_toolkit and cache activations."""
    from mech_interp_toolkit.activation_utils import get_activations
    from mech_interp_toolkit.utils import load_model_tokenizer_config

    selected_node_groups = load_selected_node_groups(nodes_path)
    layer_components = get_unique_layer_components(selected_node_groups)

    prompt_dataset = load_prompt_dataset(dataset)
    samples = prompt_dataset.samples
    if max_samples is not None:
        samples = samples[:max_samples]

    prompts = [get_sample_text(sample) for sample in samples]
    metadata = [
        {
            "sample_index": idx,
            "time_horizon_months": get_time_horizon_months(sample),
            "time_horizon_label": get_time_horizon_label(sample),
        }
        for idx, sample in enumerate(samples)
    ]

    model, tokenizer, _ = load_model_tokenizer_config(
        model_name=model_name,
        device=device,
        dtype=dtype,
        attn_type=attn_type,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_batches = chunk_list(prompts, batch_size)
    metadata_batches = chunk_list(metadata, batch_size)

    for batch_idx, prompt_batch in enumerate(
        tqdm(prompt_batches, desc="Caching activation batches")
    ):
        tokenized_batch = tokenizer(prompt_batch)
        activations, logits = get_activations(
            model,
            tokenized_batch,
            layer_components,
            positions=None,
            return_logits=False,
            clone_tensors=True,
        )
        if average_positions and selected_nodes_include_attention_heads(
            selected_node_groups
        ):
            activations = activations.split_heads()
        activations = maybe_average_positions(activations, average_positions)

        cache_payload = {
            "model_name": model_name,
            "nodes_path": str(nodes_path),
            "layer_components": layer_components,
            "metadata": metadata_batches[batch_idx],
            "average_positions": average_positions,
            "activations": extract_selected_activations(
                activations,
                selected_node_groups,
            ),
        }
        if logits is not None:
            cache_payload["logits"] = logits.detach().cpu()

        output_file = output_dir / f"activations_batch_{batch_idx:05d}.pt"
        torch.save(cache_payload, output_file)

    output_dir_abs = output_dir.resolve()
    try:
        path_in_repo = output_dir_abs.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        path_in_repo = output_dir.name
    upload_cache_folder(
        save_to_hf=save_to_hf,
        local_dir=output_dir_abs,
        path_in_repo=path_in_repo,
        hf_repo_id=hf_repo_id,
        hf_repo_type=hf_repo_type,
    )
