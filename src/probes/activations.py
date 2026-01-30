"""Activation extraction for probe training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..common.token_positions import ResolvedPositionInfo
from ..profiler import P

if TYPE_CHECKING:
    from ..models import ModelRunner
    from ..data import PreferenceItem


@dataclass
class ExtractionResult:
    """Result of activation extraction."""
    X: dict[tuple[int, int], np.ndarray]  # (layer, pos_idx) -> activations
    position_info: ResolvedPositionInfo  # Token info for labels


def extract_activations(
    runner: "ModelRunner",
    samples: list["PreferenceItem"],
    layers: list[int],
    token_positions: list,
    batch_size: int = 8,
) -> ExtractionResult:
    """Extract activations for all samples at specified layers and positions.

    Args:
        runner: ModelRunner instance
        samples: List of PreferenceItem with prompt_text
        layers: Layer indices to extract from (negative indices count from end)
        token_positions: Position specs (int, keyword dict, or relative dict)
        batch_size: Number of samples to process per batch

    Returns:
        ExtractionResult with activations dict and position info for labels
    """
    from ..common.token_positions import resolve_positions_with_info

    # Resolve negative layer indices
    resolved_layers = []
    for l in layers:
        if l < 0:
            resolved_layers.append(runner.n_layers + l)
        else:
            resolved_layers.append(l)

    X = {(l, p): [] for l in resolved_layers for p in range(len(token_positions))}
    position_info = ResolvedPositionInfo()

    layer_set = set(resolved_layers)

    def names_filter(name: str) -> bool:
        return any(f"blocks.{l}.hook_resid_post" in name for l in layer_set)

    # Process in batches
    n_samples = len(samples)
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_samples = samples[batch_start:batch_end]
        print(f"  Extracting batch {batch_start + 1}-{batch_end}/{n_samples}...")

        # Tokenize all samples in batch
        batch_texts = [s.prompt_text + (s.response or "") for s in batch_samples]
        batch_tokens = []
        batch_token_strs = []
        batch_resolved = []

        with P("tokenize_batch"):
            for text in batch_texts:
                tokens = runner.tokenize(text)
                token_strs = [runner.tokenizer.decode([t]) for t in tokens[0].tolist()]
                batch_tokens.append(tokens)
                batch_token_strs.append(token_strs)

                resolved, sample_info = resolve_positions_with_info(token_positions, token_strs)
                batch_resolved.append(resolved)

                # Store position info from first sample for labels
                if batch_start == 0 and len(batch_resolved) == 1:
                    position_info = sample_info

        # Forward pass for each sample (batching across samples requires padding)
        with P("forward_batch"):
            for idx, (text, resolved) in enumerate(zip(batch_texts, batch_resolved)):
                with torch.no_grad():
                    _, cache = runner.run_with_cache(text, names_filter=names_filter)

                # Extract activations
                for layer in resolved_layers:
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    if hook_name in cache:
                        acts = cache[hook_name]
                        if isinstance(acts, torch.Tensor):
                            acts = acts[0].cpu().numpy()

                        for pos_idx, pos in enumerate(resolved):
                            pos_i = pos.index if pos.found and 0 <= pos.index < acts.shape[0] else -1
                            X[(layer, pos_idx)].append(acts[pos_i])

                # Clear cache to free memory
                del cache

    return ExtractionResult(
        X={k: np.array(v) for k, v in X.items()},
        position_info=position_info,
    )
