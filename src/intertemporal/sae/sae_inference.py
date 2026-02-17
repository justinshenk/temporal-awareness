"""Combined LLM generation + activation extraction in a single model load.

Loads the model once, generates responses, then runs run_with_cache on each
sample to extract sentence-level activations from all layers.

Returns data in the format expected by the pipeline:
- updated_samples: list of dicts with response_text, sentences, labels
- activations: list of {sentence_idx: {layer_key: ndarray}} per sample
"""

import gc

import numpy as np
from tqdm import tqdm

from ...inference.model_runner import ModelRunner

from .sae_activations import Sentence
from .text_processing import split_into_sentences, parse_llm_choice
from ...common.device_utils import get_device, clear_gpu_memory


def _build_char_to_token_map(text: str, tokenizer) -> dict[int, int]:
    """Build mapping from character position to token index."""
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding.get("offset_mapping", [])

    char_to_token = {}
    for token_idx, (start, end) in enumerate(offsets):
        for char_pos in range(start, end + 1):
            if char_pos not in char_to_token:
                char_to_token[char_pos] = token_idx
    return char_to_token


def _get_token_span_for_sentence(
    sentence: Sentence,
    full_text: str,
    char_to_token: dict[int, int],
    max_token_idx: int,
) -> tuple[int, int] | None:
    """Get (start_token, end_token) for a sentence, or None if not found."""
    text_pos = full_text.find(sentence.text)
    if text_pos < 0:
        return None

    token_start = char_to_token.get(text_pos)
    token_end = char_to_token.get(text_pos + len(sentence.text) - 1)

    if token_start is None or token_end is None or token_start >= token_end:
        return None

    # Clamp to valid range
    token_end = min(token_end, max_token_idx)
    return (token_start, token_end)


def _extract_all_layer_activations(
    cache: dict,
    full_text: str,
    sentences: list[Sentence],
    layers: list[int],
    char_to_token: dict[int, int],
) -> dict[int, dict[str, np.ndarray]]:
    """Extract mean-pooled activations for all sentences across all layers.

    Returns: {sentence_idx: {layer_key: activation_array}}
    """
    result: dict[int, dict[str, np.ndarray]] = {}

    # Pre-compute token spans for all sentences (layer-independent)
    # Get max token idx from first available layer
    max_token_idx = 0
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        if hook_name in cache:
            layer_acts = cache[hook_name]
            if layer_acts.dim() == 3:
                max_token_idx = layer_acts.shape[1] - 1
            else:
                max_token_idx = layer_acts.shape[0] - 1
            break

    sentence_spans = {}
    for sentence_idx, sentence in enumerate(sentences):
        span = _get_token_span_for_sentence(
            sentence, full_text, char_to_token, max_token_idx
        )
        if span is not None:
            sentence_spans[sentence_idx] = span

    if not sentence_spans:
        return result

    # Extract activations for each layer
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        if hook_name not in cache:
            continue

        layer_acts = cache[hook_name].detach().cpu().float()
        if layer_acts.dim() == 3:
            layer_acts = layer_acts.squeeze(0)

        layer_key = f"layer_{layer}"

        for sentence_idx, (token_start, token_end) in sentence_spans.items():
            segment = layer_acts[token_start : token_end + 1]
            if segment.shape[0] == 0:
                continue

            act = segment.mean(dim=0).numpy()
            if not np.isfinite(act).all():
                continue

            if sentence_idx not in result:
                result[sentence_idx] = {}
            result[sentence_idx][layer_key] = act

    return result


def generate_and_extract(
    samples: list[dict],
    model_name: str,
    max_new_tokens: int,
) -> tuple[list[dict], list[dict]]:
    """Generate LLM responses and extract sentence-level activations.

    Processes one sample at a time: generate response, then extract all layers
    in a single forward pass via run_with_cache.

    Args:
        samples: list of sample dicts from generate_samples
        model_name: HuggingFace model name
        max_new_tokens: max tokens to generate

    Returns:
        (updated_samples, activations) where:
        - updated_samples: list of dicts with added response_text, sentences, labels
        - activations: list of {sentence_idx: {layer_key: ndarray}} per sample
    """
    device = get_device()
    print(f"Loading model: {model_name} on {device}")
    runner = ModelRunner(model_name=model_name, device=device)
    tokenizer = runner.tokenizer
    layers = list(range(runner.n_layers))

    updated_samples = []
    activations = []

    print(f"Processing {len(samples)} samples ({len(layers)} layers each)...")

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
            print(f"  Generation failed for sample {sample['sample_idx']}: {e}")
            response_text = ""

        choice = parse_llm_choice(
            response_text,
            sample["short_term_label"],
            sample["long_term_label"],
        )
        sentences = split_into_sentences(prompt_text, response_text)

        updated = dict(sample)
        updated["response_text"] = response_text
        updated["llm_choice"] = choice
        updated["sentences"] = [s.to_dict() for s in sentences]
        updated_samples.append(updated)

        # Extract activations for all layers
        sample_activations = {}

        if sentences:
            full_text = prompt_text + response_text
            formatted_text = runner.apply_chat_template(full_text)
            char_to_token = _build_char_to_token_map(formatted_text, tokenizer)

            try:
                names_filter = lambda name: "hook_resid_post" in name
                _, cache = runner.run_with_cache(full_text, names_filter=names_filter)

                sample_activations = _extract_all_layer_activations(
                    cache, formatted_text, sentences, layers, char_to_token
                )

                del cache
            except Exception as e:
                print(f"  Extraction failed for sample {sample['sample_idx']}: {e}")

            gc.collect()
            clear_gpu_memory()

        activations.append(sample_activations)

    print(f"Processed {len(updated_samples)} samples")
    n_sentences = sum(len(a) for a in activations)
    print(f"  Total sentence activations: {n_sentences}")

    del runner
    clear_gpu_memory()

    return updated_samples, activations
