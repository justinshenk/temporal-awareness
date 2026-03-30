"""Position-based activation extraction for SAE analysis.

Extracts activations at specific token positions (source, dest, secondary_source)
across multiple components (resid_pre, resid_post, mlp_out, attn_out).

This replaces the previous sentence-level mean-pooling approach with precise
position-based extraction aligned with circuit analysis findings.
"""

import gc
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from ...inference.model_runner import ModelRunner
from ...common.device_utils import get_device, clear_gpu_memory

from .sae_positions import (
    COMPONENTS,
    POSITION_NAMES,
    ResolvedPositions,
    decode_tokens,
    get_hook_name,
    get_names_filter,
    resolve_positions,
)
from .text_processing import parse_llm_choice
from ..formatting.configs.default_prompt_format import DefaultPromptFormat


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PositionActivations:
    """Activations extracted at specific positions for a single sample.

    Attributes:
        positions: Resolved token positions
        activations: Dict mapping (layer, component, position_name) -> activation vector
    """

    positions: ResolvedPositions
    activations: dict[str, np.ndarray]  # key format: "L{layer}_{component}_P{pos_name}"

    def get(self, layer: int, component: str, position_name: str) -> np.ndarray | None:
        """Get activation for a specific (layer, component, position) tuple."""
        key = f"L{layer}_{component}_P{position_name}"
        return self.activations.get(key)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "positions": self.positions.to_dict(),
            "activations": {k: v.tolist() for k, v in self.activations.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PositionActivations":
        """Create from serialized dict."""
        positions = ResolvedPositions(**d["positions"])
        activations = {k: np.array(v) for k, v in d["activations"].items()}
        return cls(positions=positions, activations=activations)


# =============================================================================
# Activation Extraction
# =============================================================================


def _extract_position_activations(
    cache: dict,
    positions: ResolvedPositions,
    layers: list[int],
    components: list[str],
    position_names: list[str],
) -> dict[str, np.ndarray]:
    """Extract activations at specific positions from cache.

    Args:
        cache: TransformerLens cache from run_with_cache
        positions: Resolved token positions
        layers: List of layer indices to extract
        components: List of component types (resid_pre, resid_post, mlp_out, attn_out)
        position_names: List of position names to extract (source, dest, secondary_source)

    Returns:
        Dict mapping "L{layer}_{component}_P{pos_name}" -> activation array (d_model,)
    """
    result = {}

    for layer in layers:
        for component in components:
            hook_name = get_hook_name(component, layer)
            if hook_name not in cache:
                continue

            acts = cache[hook_name].detach().cpu().float()
            # Remove batch dimension if present
            if acts.dim() == 3:
                acts = acts.squeeze(0)

            seq_len = acts.shape[0]

            for pos_name in position_names:
                pos_idx = positions.get(pos_name)
                if pos_idx < 0 or pos_idx >= seq_len:
                    continue

                key = f"L{layer}_{component}_P{pos_name}"
                act = acts[pos_idx].numpy()

                if np.isfinite(act).all():
                    result[key] = act

    return result


def _count_prompt_tokens(formatted_text: str, prompt_text: str, tokenizer) -> int:
    """Estimate the number of tokens in the prompt portion.

    Args:
        formatted_text: Full text with chat template applied
        prompt_text: Original prompt text
        tokenizer: Tokenizer for encoding

    Returns:
        Estimated number of prompt tokens
    """
    # Encode full text and prompt
    full_encoding = tokenizer(formatted_text, add_special_tokens=False)
    prompt_encoding = tokenizer(prompt_text, add_special_tokens=False)

    # Prompt length is roughly the prompt tokens
    # This is an approximation since chat template adds tokens
    return len(prompt_encoding["input_ids"])


# =============================================================================
# Main Extraction Functions
# =============================================================================


def generate_and_extract(
    samples: list[dict],
    model_name: str,
    max_new_tokens: int,
    layers: list[int] | None = None,
    components: list[str] | None = None,
    position_names: list[str] | None = None,
) -> tuple[list[dict], list[PositionActivations]]:
    """Generate LLM responses and extract position-specific activations.

    Args:
        samples: List of sample dicts from generate_samples
        model_name: HuggingFace model name
        max_new_tokens: Max tokens to generate
        layers: Layers to extract (defaults to all)
        components: Components to extract (defaults to all)
        position_names: Positions to extract (defaults to all)

    Returns:
        (updated_samples, activations) where:
        - updated_samples: List of dicts with added response_text, llm_choice, positions
        - activations: List of PositionActivations per sample
    """
    device = get_device()
    print(f"Loading model: {model_name} on {device}")
    runner = ModelRunner(model_name=model_name, device=device)
    tokenizer = runner._tokenizer
    prompt_format = DefaultPromptFormat()

    # Default to all layers if not specified
    if layers is None:
        layers = list(range(runner.n_layers))
    if components is None:
        components = COMPONENTS
    if position_names is None:
        position_names = POSITION_NAMES

    # Build names filter for efficient caching
    names_filter = get_names_filter(components, layers)

    updated_samples = []
    activations = []

    print(
        f"Processing {len(samples)} samples "
        f"({len(layers)} layers x {len(components)} components x {len(position_names)} positions)..."
    )

    for sample in tqdm(samples, desc="Samples"):
        prompt_text = sample["prompt_text"]

        # Generate response
        try:
            response_text = runner.generate(
                prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )
        except Exception as e:
            print(f"  Generation failed for sample {sample.get('sample_idx', '?')}: {e}")
            response_text = ""

        choice = parse_llm_choice(
            response_text,
            sample["short_term_label"],
            sample["long_term_label"],
        )

        # Update sample with response info
        updated = dict(sample)
        updated["response_text"] = response_text
        updated["llm_choice"] = choice

        # Extract activations at specific positions
        sample_activations = None

        if response_text:
            full_text = prompt_text + response_text
            formatted_text = runner.apply_chat_template(full_text)
            tokens = decode_tokens(tokenizer, tokenizer.encode(formatted_text))

            # Estimate prompt token count
            prompt_len = _count_prompt_tokens(formatted_text, prompt_text, tokenizer)

            # Resolve positions
            positions = resolve_positions(tokens, prompt_len, prompt_format)
            updated["positions"] = positions.to_dict()

            try:
                _, cache = runner.run_with_cache(full_text, names_filter=names_filter)

                acts = _extract_position_activations(
                    cache, positions, layers, components, position_names
                )

                sample_activations = PositionActivations(
                    positions=positions, activations=acts
                )

                del cache
            except Exception as e:
                print(
                    f"  Extraction failed for sample {sample.get('sample_idx', '?')}: {e}"
                )

            gc.collect()
            clear_gpu_memory()

        updated_samples.append(updated)
        activations.append(sample_activations)

    # Report statistics
    n_with_acts = sum(1 for a in activations if a is not None)
    total_keys = sum(len(a.activations) for a in activations if a is not None)
    print(f"Processed {len(updated_samples)} samples, {n_with_acts} with activations")
    print(f"  Total activation keys: {total_keys}")

    del runner
    clear_gpu_memory()

    return updated_samples, activations


def extract_activations_only(
    samples: list[dict],
    model_name: str,
    layers: list[int],
    components: list[str],
    position_names: list[str],
) -> list[PositionActivations | None]:
    """Extract activations from already-generated samples (no generation).

    Use this when samples already have response_text and you just need to
    extract activations at different positions/layers/components.

    Args:
        samples: Samples with response_text already populated
        model_name: Model name for tokenization and activation extraction
        layers: Layers to extract
        components: Components to extract
        position_names: Positions to extract

    Returns:
        List of PositionActivations (or None for failed extractions)
    """
    device = get_device()
    print(f"Loading model: {model_name} on {device}")
    runner = ModelRunner(model_name=model_name, device=device)
    tokenizer = runner._tokenizer
    prompt_format = DefaultPromptFormat()

    names_filter = get_names_filter(components, layers)
    activations = []

    print(f"Extracting activations from {len(samples)} samples...")

    for sample in tqdm(samples, desc="Extracting"):
        prompt_text = sample["prompt_text"]
        response_text = sample.get("response_text", "")

        if not response_text:
            activations.append(None)
            continue

        full_text = prompt_text + response_text
        formatted_text = runner.apply_chat_template(full_text)
        tokens = decode_tokens(tokenizer, tokenizer.encode(formatted_text))
        prompt_len = _count_prompt_tokens(formatted_text, prompt_text, tokenizer)

        positions = resolve_positions(tokens, prompt_len, prompt_format)

        try:
            _, cache = runner.run_with_cache(full_text, names_filter=names_filter)

            acts = _extract_position_activations(
                cache, positions, layers, components, position_names
            )

            activations.append(PositionActivations(positions=positions, activations=acts))

            del cache
        except Exception as e:
            print(f"  Extraction failed: {e}")
            activations.append(None)

        gc.collect()
        clear_gpu_memory()

    del runner
    clear_gpu_memory()

    return activations


# =============================================================================
# Training Data Preparation
# =============================================================================


def form_training_data(
    activations: list[PositionActivations | None],
    layer: int,
    component: str,
    position_name: str,
) -> tuple[np.ndarray, list[int]]:
    """Form training data matrix for a specific (layer, component, position) tuple.

    Args:
        activations: List of PositionActivations from extraction
        layer: Layer index
        component: Component type
        position_name: Position name

    Returns:
        (X, indices) where X is (n_samples, d_model) and indices maps rows to sample indices
    """
    key = f"L{layer}_{component}_P{position_name}"

    vectors = []
    indices = []

    for i, act in enumerate(activations):
        if act is None:
            continue
        vec = act.activations.get(key)
        if vec is not None:
            vectors.append(vec)
            indices.append(i)

    if not vectors:
        raise ValueError(f"No activations found for {key}")

    X = np.stack(vectors, axis=0)
    return X, indices
