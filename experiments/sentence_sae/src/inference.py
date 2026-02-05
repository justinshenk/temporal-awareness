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

from src.models.model_runner import ModelRunner

from .activations import Sentence
from .text_processing import split_into_sentences, parse_llm_choice
from .utils import get_device, clear_gpu_memory


def _char_to_token_map(text: str, tokenizer) -> dict[int, int]:
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding.get("offset_mapping", [])
    char_to_token = {}
    for token_idx, (start, end) in enumerate(offsets):
        for char_pos in range(start, end + 1):
            if char_pos not in char_to_token:
                char_to_token[char_pos] = token_idx
    return char_to_token


# ── Sentence activation extraction from cache ───────────────────────────────


def _extract_sentence_activations_single_layer(
    cache: dict,
    full_text: str,
    sentences: list[Sentence],
    layer: int,
    tokenizer,
    char_to_token: dict[int, int],
) -> dict[int, np.ndarray]:
    """Extract per-sentence activations from a single-layer cache.

    Returns {sentence_idx: mean_pooled_activation}.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    if hook_name not in cache:
        return {}

    layer_acts = cache[hook_name].detach().cpu().float()
    if layer_acts.dim() == 3:
        layer_acts = layer_acts.squeeze(0)

    result: dict[int, np.ndarray] = {}
    for sentence_idx, sentence in enumerate(sentences):
        text_pos = full_text.find(sentence.text)
        if text_pos < 0:
            continue
        token_start = char_to_token.get(text_pos)
        token_end = char_to_token.get(text_pos + len(sentence.text) - 1)
        if token_start is None or token_end is None or token_start >= token_end:
            continue

        clamped_end = min(token_end, layer_acts.shape[0] - 1)
        segment = layer_acts[token_start : clamped_end + 1]
        if segment.shape[0] == 0:
            continue

        act = segment.mean(dim=0).numpy()
        if not np.isfinite(act).all():
            continue

        result[sentence_idx] = act

    return result


# ── Main combined function ──────────────────────────────────────────────────


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

    for idx, sample in enumerate(tqdm(samples, desc="Samples")):
        prompt_text = sample["prompt_text"]

        # Generate response via ModelRunner API
        try:
            response_text = runner.generate(
                prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )
        except Exception as e:
            print(f"  Generation failed for sample {sample['sample_id']}: {e}")
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

        # Extract activations — all layers in one forward pass
        sample_activations = {}

        if sentences:
            full_text = prompt_text + response_text
            # Build char_to_token on formatted text (matches run_with_cache)
            formatted_text = runner._apply_chat_template(full_text)
            char_to_token = _char_to_token_map(formatted_text, tokenizer)

            try:
                names_filter = lambda name: "hook_resid_post" in name
                _, cache = runner.run_with_cache(full_text, names_filter=names_filter)

                for layer in layers:
                    layer_acts = _extract_sentence_activations_single_layer(
                        cache,
                        formatted_text,
                        sentences,
                        layer,
                        tokenizer,
                        char_to_token,
                    )
                    for sentence_idx, act in layer_acts.items():
                        if sentence_idx not in sample_activations:
                            sample_activations[sentence_idx] = {}
                        sample_activations[sentence_idx][f"layer_{layer}"] = act

                del cache
            except Exception as e:
                print(f"  Extraction failed for sample {sample['sample_id']}: {e}")

            gc.collect()
            clear_gpu_memory()

        activations.append(sample_activations)

    print(f"Processed {len(updated_samples)} samples")
    n_sentences = sum(len(a) for a in activations)
    print(f"  Total sentence activations: {n_sentences}")

    del runner
    clear_gpu_memory()

    return updated_samples, activations
