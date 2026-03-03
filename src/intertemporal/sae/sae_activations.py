"""Data structures and activation processing for SAE pipeline."""

from dataclasses import dataclass, fields
import numpy as np


# =============================================================================
# Constants
# =============================================================================

HORIZON_NONE = 0
HORIZON_SHORT = 1  # <= 1 year
HORIZON_MEDIUM = 2  # 1-5 years
HORIZON_LONG = 3  # > 5 years

CHOICE_SHORT_TERM = 0
CHOICE_LONG_TERM = 1
CHOICE_UNKNOWN = -1


# =============================================================================
# Sentence
# =============================================================================


@dataclass
class Sentence:
    """A sentence with metadata about its origin."""

    text: str
    source: str  # "prompt" or "response"
    section: (
        str  # prompt: situation/task/consider/action/format; response: choice/reasoning
    )

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


# =============================================================================
# Helpers
# =============================================================================


def horizon_bucket(time_horizon) -> int:
    """Convert a TimeValue to a horizon bucket."""
    if time_horizon is None:
        return HORIZON_NONE
    months = time_horizon.to_months()
    if months <= 12:
        return HORIZON_SHORT
    if months <= 60:
        return HORIZON_MEDIUM
    return HORIZON_LONG


def get_choice_time(sample: dict) -> float:
    """Get the time (in months) of the chosen option."""
    choice = sample.get("llm_choice", -1)
    if choice == CHOICE_SHORT_TERM:
        return sample.get("short_term_time_months", -1)
    if choice == CHOICE_LONG_TERM:
        return sample.get("long_term_time_months", -1)
    return -1


# =============================================================================
# Sentence Extraction
# =============================================================================


def get_sentences(
    samples: list[dict],
    activations: list[dict],
    section_means: dict[int, dict[str, np.ndarray]],
) -> list[dict]:
    """Flatten samples + activations into sentence dicts with centered activations."""
    result = []

    for sample_idx, sample in enumerate(samples):
        if sample_idx >= len(activations) or not activations[sample_idx]:
            continue

        raw_sentences = sample.get("sentences", [])
        sample_acts = activations[sample_idx]

        for sent_idx, raw in enumerate(raw_sentences):
            if sent_idx not in sample_acts:
                continue

            sentence = Sentence.from_dict(raw)
            centered = {}
            for layer_key, act in sample_acts[sent_idx].items():
                layer = int(layer_key.split("_")[1])
                if layer in section_means:
                    centered[layer_key] = act - section_means[layer][sentence.section]

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
                    "formatting_id": sample.get("formatting_id"),
                    "matches_rational": sample.get("matches_rational"),
                    "matches_associated": sample.get("matches_associated"),
                    "activations": centered,
                }
            )

    return result


# =============================================================================
# Activation Processing
# =============================================================================


def get_normalized_vectors_for_sentences(
    layer: int,
    sentences: list[dict],
    filter_fn=None,
) -> tuple[np.ndarray, list[dict]]:
    """Extract activation vectors for a layer from sentence dicts."""
    layer_key = f"layer_{layer}"
    vectors = []
    kept = []

    for s in sentences:
        if filter_fn and not filter_fn(Sentence.from_dict(s)):
            continue
        act = s["activations"].get(layer_key)
        if act is None:
            continue
        vectors.append(act)
        kept.append(s)

    if not vectors:
        raise ValueError(f"No activations found for layer {layer}")

    return np.stack(vectors), kept


def form_training_datasets(
    samples: list[dict],
    activations: list[dict],
    layer: int,
    section_means: dict[int, dict[str, np.ndarray]],
    filter_fn=None,
) -> np.ndarray:
    """Build training matrix for a specific layer."""
    sentences = get_sentences(samples, activations, section_means)
    X, _ = get_normalized_vectors_for_sentences(layer, sentences, filter_fn)
    return X
