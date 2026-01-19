"""Query runner for preference datasets."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..common.io import load_json
from ..common.schema_utils import SchemaClass
from .model_runner import ModelRunner


@dataclass
class InternalsConfig(SchemaClass):
    """Configuration for capturing model internals."""

    activations: dict = field(default_factory=dict)
    token_positions: list = field(default_factory=list)


@dataclass
class QueryConfig(SchemaClass):
    """Query configuration."""

    models: list[str]
    datasets: list[str]
    internals: Optional[InternalsConfig] = None
    max_new_tokens: int = 256
    temperature: float = 0.0
    subsample: float = 1.0


@dataclass
class CapturedInternals:
    """Captured activations from a forward pass."""

    activations: dict  # name -> tensor
    activation_names: list[str]


@dataclass
class PreferenceItem(SchemaClass):
    """Single preference result."""

    sample_id: int
    time_horizon: Optional[dict]
    short_term_label: str
    long_term_label: str
    choice: str
    choice_prob: float
    alt_prob: float
    response: str
    internals: Optional[CapturedInternals] = None


@dataclass
class QueryOutput(SchemaClass):
    """Query output."""

    dataset_id: str
    model: str
    preferences: list[PreferenceItem]

    def print_summary(self) -> None:
        """Print summary of query results."""
        counts = {"short_term": 0, "long_term": 0, "unknown": 0}
        for p in self.preferences:
            counts[p.choice] += 1
        print(
            f"\nResults: short={counts['short_term']}, long={counts['long_term']}, unknown={counts['unknown']}"
        )


def parse_choice(
    response: str,
    short_label: str,
    long_label: str,
    choice_prefix: str,
) -> str:
    """
    Parse choice from model response.

    Looks for pattern: "<choice_prefix> <label>"
    Returns: "short_term", "long_term", or "unknown"
    """
    response_lower = response.lower().strip()
    prefix_lower = choice_prefix.lower()

    labels = [short_label, long_label]
    labels_stripped = [l.rstrip(".)") for l in labels]
    all_variants = set(l.lower() for l in labels + labels_stripped)
    labels_pattern = "|".join(
        re.escape(l) for l in sorted(all_variants, key=len, reverse=True)
    )

    pattern = rf"{re.escape(prefix_lower)}\s*({labels_pattern})"
    match = re.search(pattern, response_lower)

    if match:
        matched = match.group(1)
        if matched in (short_label.lower(), short_label.rstrip(".)").lower()):
            return "short_term"
        elif matched in (long_label.lower(), long_label.rstrip(".)").lower()):
            return "long_term"

    return "unknown"


class QueryRunner:
    """Query runner for preference datasets."""

    def __init__(self, config: QueryConfig, datasets_dir: Path):
        self.config = config
        self.datasets_dir = datasets_dir
        self._model: Optional[ModelRunner] = None
        self._model_name: Optional[str] = None

    @classmethod
    def load_config(cls, path: Path) -> QueryConfig:
        """Load query configuration from JSON file."""
        data = load_json(path)

        internals = None
        if "internals" in data:
            internals = InternalsConfig(
                activations=data["internals"],
                token_positions=data.get("token_positions", []),
            )

        return QueryConfig(
            models=data.get("models", []),
            datasets=data.get("datasets", []),
            internals=internals,
            max_new_tokens=data.get("max_new_tokens", 256),
            temperature=data.get("temperature", 0.0),
            subsample=data.get("subsample", 1.0),
        )

    def _load_model(self, name: str) -> ModelRunner:
        if self._model is not None and self._model_name == name:
            return self._model
        self._model = ModelRunner(model_name=name)
        self._model_name = name
        return self._model

    def load_dataset(self, dataset_id: str) -> dict:
        """Load dataset by ID."""
        matches = list(self.datasets_dir.glob(f"*_{dataset_id}.json"))
        if matches:
            return load_json(matches[0])
        raise FileNotFoundError(f"Dataset not found: {dataset_id}")

    def _get_activation_names(self) -> list[str]:
        """Get hook names for activation capture."""
        if not self.config.internals:
            return []

        names = []
        for act_type, spec in self.config.internals.activations.items():
            layers = spec.get("layers", [])
            for layer in layers:
                # if layer < 0:
                #     layer = n_layers + layer
                names.append(f"blocks.{layer}.hook_{act_type}")
        return names

    def query_dataset(self, dataset_id: str, model_name: str) -> QueryOutput:
        """Query a single dataset with a model. Returns results in memory."""
        dataset = self.load_dataset(dataset_id)
        samples = dataset.get("samples", [])

        choice_prefix = (
            dataset.get("config", {})
            .get("prompt_format", {})
            .get("const_keywords", {})
            .get("choice_prefix", "I select:")
        )

        if self.config.subsample < 1.0:
            n = max(1, int(len(samples) * self.config.subsample))
            samples = random.sample(samples, n)

        model_runner = self._load_model(model_name)
        activation_names = self._get_activation_names()
        preferences = []

        print(f"Querying LLM for {len(samples)} samples...")
        for i, sample in enumerate(samples):
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(samples)}")

            prompt = sample["prompt"]["text"]
            pair = sample["prompt"]["preference_pair"]
            short_label = pair["short_term"]["label"]
            long_label = pair["long_term"]["label"]

            # Step 1: Calculate kv cache for prompt and get prefill logits
            kv_cache = model_runner.init_kv_cache()
            prefill_logits, _ = model_runner.run_with_cache(
                prompt,
                names_filter=lambda name: name in activation_names,
                past_kv_cache=kv_cache,
            )
            kv_cache.freeze()

            # Step 2: Run generation using prefill logits and frozen cache
            response = model_runner.generate_from_cache(
                prefill_logits,
                kv_cache,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            choice = parse_choice(response, short_label, long_label, choice_prefix)

            # Step 3: Capture internals
            _, cache = model_runner.run_with_cache(
                prompt + response,
                names_filter=lambda name: name in activation_names,
                past_kv_cache=kv_cache,
            )
            activations = {}
            for name in activation_names:
                if name in cache:
                    activations[name] = cache[name][0].cpu()
            internals = CapturedInternals(
                activations=activations,
                activation_names=list(activations.keys()),
            )

            # Step 4: Capture choice probs
            probs = model_runner.get_label_probs(
                prompt, choice_prefix, (short_label, long_label), kv_cache
            )

            if choice == "short_term":
                choice_prob, alt_prob = probs[0], probs[1]
            elif choice == "long_term":
                choice_prob, alt_prob = probs[1], probs[0]
            else:
                choice_prob, alt_prob = (
                    max(probs[0], probs[1]),
                    min(probs[0], probs[1]),
                )

            preferences.append(
                PreferenceItem(
                    sample_id=sample["sample_id"],
                    time_horizon=sample["prompt"].get("time_horizon"),
                    short_term_label=short_label,
                    long_term_label=long_label,
                    choice=choice,
                    choice_prob=choice_prob,
                    alt_prob=alt_prob,
                    response=response,
                    internals=internals,
                )
            )

        return QueryOutput(
            dataset_id=dataset_id,
            model=model_name,
            preferences=preferences,
        )
