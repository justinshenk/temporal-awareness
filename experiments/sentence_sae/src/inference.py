"""Combined LLM generation + activation extraction in a single model load.

Loads the model once, generates responses, then runs run_with_cache on each
sample to extract sentence-level activations from all layers.

Returns data in the format expected by the pipeline:
- updated_samples: list of dicts with response_text, sentences, labels
- activations: list of {sentence_idx: {layer_key: ndarray}} per sample
"""

import gc
import re

import numpy as np
from tqdm import tqdm

from src.models.query_runner import parse_choice

from .data import CHOICE_SHORT_TERM, CHOICE_LONG_TERM, CHOICE_UNKNOWN, Sentence
from .utils import get_device, clear_gpu_memory

# ── Sentence utilities ──────────────────────────────────────────────────────

MIN_SENTENCE_WORDS = 3
ACTIVATION_EPS = 1e-8


def _get_format():
    from src.formatting.configs import DefaultPromptFormat
    return DefaultPromptFormat()


# Lazy init to avoid circular imports
_PROMPT_MARKERS = None
_RESPONSE_CHOICE_PREFIX = None
_RESPONSE_REASONING_PREFIX = None


def _ensure_markers():
    global _PROMPT_MARKERS, _RESPONSE_CHOICE_PREFIX, _RESPONSE_REASONING_PREFIX
    if _PROMPT_MARKERS is not None:
        return
    fmt = _get_format()
    _PROMPT_MARKERS = fmt.get_prompt_section_markers()
    response_markers = fmt.get_response_markers()
    _RESPONSE_CHOICE_PREFIX = response_markers["choice_prefix"]
    _RESPONSE_REASONING_PREFIX = response_markers["reasoning_prefix"]


def _split_raw(text: str, min_words: int = MIN_SENTENCE_WORDS) -> list[str]:
    """Split text into sentences, protecting decimals."""
    if not text or not text.strip():
        return []
    protected = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', text)
    raw = re.split(r'(?<=[.!?;\n])\s+', protected)
    sentences = []
    for s in raw:
        s = s.replace('<DECIMAL>', '.').strip()
        if s and len(s.split()) >= min_words:
            sentences.append(s)
    return sentences


def _prompt_sections(prompt_text: str) -> list[tuple[str, str]]:
    """Split prompt into (section_name, text) pairs using format markers."""
    _ensure_markers()
    markers_sorted = sorted(
        _PROMPT_MARKERS.items(),
        key=lambda kv: prompt_text.find(kv[1]),
    )
    sections: list[tuple[str, str]] = []
    for i, (name, marker) in enumerate(markers_sorted):
        start = prompt_text.find(marker)
        if start < 0:
            continue
        if i + 1 < len(markers_sorted):
            next_start = prompt_text.find(markers_sorted[i + 1][1])
            if next_start > start:
                sections.append((name, prompt_text[start:next_start]))
                continue
        sections.append((name, prompt_text[start:]))
    return sections


def split_into_sentences(
    prompt_text: str,
    response_text: str,
    min_words: int = MIN_SENTENCE_WORDS,
) -> list[Sentence]:
    """Split prompt + response into classified Sentence objects.

    Uses DefaultPromptFormat markers to identify:
    - Prompt sections: situation, task, consider, action, format
    - Response sections: choice (around "I select:") and reasoning
      (around "My reasoning:")
    """
    _ensure_markers()
    sentences: list[Sentence] = []

    # ── Prompt sentences ──
    for section_name, section_text in _prompt_sections(prompt_text):
        for s in _split_raw(section_text, min_words):
            sentences.append(Sentence(text=s, source="prompt", section=section_name))

    # ── Response sentences ──
    if not response_text or not response_text.strip():
        return sentences

    # Strip chat-template artifacts from response
    clean_response = re.sub(r'<\|[^|]*\|>', '', response_text).strip()

    # Locate "I select:" and "My reasoning:" (last occurrences, matching
    # the convention in src.analysis.markers for response parsing).
    choice_pos = clean_response.lower().rfind(_RESPONSE_CHOICE_PREFIX.lower())
    reasoning_pos = clean_response.lower().rfind(_RESPONSE_REASONING_PREFIX.lower())

    if choice_pos < 0 and reasoning_pos < 0:
        # No markers found — treat everything as reasoning
        for s in _split_raw(clean_response, min_words):
            sentences.append(Sentence(text=s, source="response", section="reasoning"))
        return sentences

    # Determine boundaries
    if choice_pos >= 0 and reasoning_pos >= 0 and reasoning_pos > choice_pos:
        choice_text = clean_response[choice_pos:reasoning_pos]
        reasoning_text = clean_response[reasoning_pos:]
    elif choice_pos >= 0:
        choice_text = clean_response[choice_pos:]
        reasoning_text = ""
    else:
        choice_text = ""
        reasoning_text = clean_response[reasoning_pos:]

    for s in _split_raw(choice_text, min_words):
        sentences.append(Sentence(text=s, source="response", section="choice"))
    for s in _split_raw(reasoning_text, min_words):
        sentences.append(Sentence(text=s, source="response", section="reasoning"))

    return sentences


def _char_to_token_map(text: str, tokenizer) -> dict[int, int]:
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding.get("offset_mapping", [])
    char_to_token = {}
    for token_idx, (start, end) in enumerate(offsets):
        for char_pos in range(start, end + 1):
            if char_pos not in char_to_token:
                char_to_token[char_pos] = token_idx
    return char_to_token


def normalize_activations_raw(X: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Center and L2-normalise a raw activation matrix."""
    X_centered = X - mean
    norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
    return X_centered / (norms + ACTIVATION_EPS)


# ── Choice parsing ──────────────────────────────────────────────────────────

_CHOICE_MAP = {
    "short_term": CHOICE_SHORT_TERM,
    "long_term": CHOICE_LONG_TERM,
    "unknown": CHOICE_UNKNOWN,
}


def _parse_choice(response_text: str, short_label: str, long_label: str) -> int:
    _ensure_markers()
    result = parse_choice(response_text, short_label, long_label, _RESPONSE_CHOICE_PREFIX)
    return _CHOICE_MAP[result]


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
        segment = layer_acts[token_start:clamped_end + 1]
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
    from src.models.model_runner import ModelRunner
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

        choice = _parse_choice(
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
                        cache, formatted_text, sentences, layer, tokenizer, char_to_token,
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
