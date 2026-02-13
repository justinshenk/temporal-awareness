"""Data structures, constants, and activation normalization utilities."""

from dataclasses import dataclass, fields

import numpy as np

# ── Constants ───────────────────────────────────────────────────────────────

HORIZON_NONE = 0  # No time horizon specified
HORIZON_SHORT = 1  # <= 2 years
HORIZON_MEDIUM = 2  # 2-5 years
HORIZON_LONG = 3  # > 5 years

CHOICE_SHORT_TERM = 0
CHOICE_LONG_TERM = 1
CHOICE_UNKNOWN = -1


# ── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class Sentence:
    """A sentence with metadata about its origin in the prompt/response."""

    text: str
    source: str  # "prompt" or "response"
    section: str  # prompt: "situation","task","consider","action","format"
    # response: "choice","reasoning"

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "section": self.section}

    @classmethod
    def from_dict(cls, d: dict) -> "Sentence":
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    @staticmethod
    def get_sections() -> list[str]:
        return [
            "situation",
            "task",
            "consider",
            "action",
            "format",
            "choice",
            "reasoning",
        ]


# ── Sample conversion ───────────────────────────────────────────────────────


def _horizon_bucket(time_horizon) -> int:
    """Convert a TimeValue to a horizon bucket."""
    if time_horizon is None:
        return HORIZON_NONE
    months = time_horizon.to_months()
    if months <= 12:
        return HORIZON_SHORT
    elif months <= (12 * 5):
        return HORIZON_MEDIUM
    return HORIZON_LONG


# ── Pipeline analysis helpers ────────────────────────────────────────────────


def get_choice_time(sample):
    llm_choice = sample.get("llm_choice", -1)
    short_term_time_months = sample.get("short_term_time_months", -1)
    long_term_time_months = sample.get("long_term_time_months", -1)
    if llm_choice == 0:
        return short_term_time_months
    if llm_choice == 1:
        return long_term_time_months
    return -1


def get_sentences(
    samples: list[dict],
    activations: list[dict],
    section_activation_means: dict[int, dict[str, np.ndarray]],
) -> list[dict]:
    """Flatten samples + activations into a list of sentence dicts.

    Activations are centered by section mean per layer.

    Each dict has: sentence (Sentence metadata), sample_idx,
    time_horizon_bucket, llm_choice, activations ({layer_key: centered ndarray}).
    """
    result = []
    for sample_idxx, sample in enumerate(samples):
        raw_sentences = sample.get("sentences", [])
        if sample_idxx >= len(activations) or not activations[sample_idxx]:
            continue
        sample_acts = activations[sample_idxx]
        for sentence_idx, raw in enumerate(raw_sentences):
            if sentence_idx not in sample_acts:
                continue
            sentence = Sentence.from_dict(raw)
            centered_acts = {}
            for layer_key, act in sample_acts[sentence_idx].items():
                layer = int(layer_key.split("_")[1])
                if layer in section_activation_means:
                    centered_acts[layer_key] = center_activation(
                        act, sentence.section, section_activation_means[layer]
                    )
            result.append(
                {
                    "text": sentence.text,
                    "source": sentence.source,
                    "section": sentence.section,
                    "sample_idx": sample.get("sample_idx"),
                    "time_horizon_bucket": sample.get("time_horizon_bucket", -1),
                    "time_horizon_months": sample.get("time_horizon_months"),
                    "llm_choice": sample.get("llm_choice", -1),
                    "llm_choice_time_months": get_choice_time(sample),
                    "activations": centered_acts,
                }
            )
    return result


# ── Activation normalization ────────────────────────────────────────────────


def center_activation(
    act: np.ndarray, section: str, section_means: dict[str, np.ndarray]
) -> np.ndarray:
    """Subtract a section's mean activation from a single activation vector."""
    return act - section_means[section]


def calculate_activation_means_by_section(
    samples: list, activations: list, layers: list[int]
) -> dict[int, dict[str, np.ndarray]]:
    """Compute the mean activation vector per (layer, section) pair.

    Args:
        samples: list of sample dicts, each with a "sentences" key containing
            sentence dicts that have a "section" field.
        activations: parallel list where activations[i] is
            {sentence_idx: {layer_key: ndarray}}.
        layers: list of layer indices to compute means for.

    Returns:
        dict mapping layer -> section -> mean activation vector.
        Sections with no vectors get a zero vector.
    """
    sections = Sentence.get_sections()

    # Collect vectors grouped by (layer, section).
    vectors: dict[int, dict[str, list[np.ndarray]]] = {
        layer: {s: [] for s in sections} for layer in layers
    }

    for sample_idxx, sample in enumerate(samples):
        if sample_idxx >= len(activations):
            continue
        sample_acts = activations[sample_idxx]
        raw_sentences = sample.get("sentences", [])

        for sentence_idx in sorted(sample_acts.keys(), key=int):
            if sentence_idx >= len(raw_sentences):
                continue
            section = raw_sentences[sentence_idx].get("section")
            if section not in sections:
                continue
            sentence_acts = sample_acts[sentence_idx]
            for layer in layers:
                layer_key = f"layer_{layer}"
                if layer_key in sentence_acts:
                    vectors[layer][section].append(sentence_acts[layer_key])

    # Compute means.  Infer d_in from the first available vector.
    d_in = None
    for layer in layers:
        for s in sections:
            if vectors[layer][s]:
                d_in = vectors[layer][s][0].shape[0]
                break
        if d_in is not None:
            break

    if d_in is None:
        raise ValueError("No activation vectors found in any layer/section")

    result: dict[int, dict[str, np.ndarray]] = {}
    for layer in layers:
        result[layer] = {}
        for s in sections:
            vecs = vectors[layer][s]
            if vecs:
                result[layer][s] = np.stack(vecs).mean(axis=0)
            else:
                result[layer][s] = np.zeros(d_in)

    return result


def form_training_datasets(
    samples: list[dict],
    activations: list[dict],
    layer: int,
    section_activation_means: dict[int, dict[str, np.ndarray]],
    filter_sentence=None,
) -> np.ndarray:
    """Build training matrix for a specific layer from all samples/sentences.

    Flattens samples + activations via get_sentences (which centers by section),
    then extracts the requested layer's vectors.

    Returns stacked centered activation vectors.
    """
    sentences = get_sentences(samples, activations, section_activation_means)
    X, _ = get_normalized_vectors_for_sentences(layer, sentences, filter_sentence)
    print(f"    Layer {layer}: {X.shape[0]} sentence vectors, d={X.shape[1]}")
    return X


def get_normalized_vectors_for_sentences(
    layer: int,
    sentences: list[dict],
    filter_sentence=None,
) -> tuple[np.ndarray, list[dict]]:
    """Extract activation vectors for a layer from sentence dicts.

    Assumes activations are already centered by section means.
    Returns (X_norm, filtered_sentences).
    """
    layer_key = f"layer_{layer}"
    vectors = []
    kept = []
    for s in sentences:
        if filter_sentence is not None and not filter_sentence(Sentence.from_dict(s)):
            continue
        act = s["activations"].get(layer_key)
        if act is None:
            continue
        vectors.append(act)
        kept.append(s)

    if not vectors:
        raise ValueError(f"No activations found for layer {layer}")

    X_norm = np.stack(vectors, axis=0)
    return X_norm, kept
