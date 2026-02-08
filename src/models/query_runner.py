"""Query runner for preference datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..common.io import load_json
from ..common.types import CapturedInternals, PreferenceSample
from .interventions import load_intervention_from_dict, Intervention
from .model_runner import ModelRunner
from ..preference import PreferenceDataset
from ..prompt import PromptDataset
from ..parsing import parse_choice


@dataclass
class ActivationSpec:
    """Specification for which activations to capture."""

    component: str  # e.g., "resid_pre", "resid_post", "attn_out", "mlp_out"
    layers: list[int] = field(default_factory=list)


@dataclass
class InternalsConfig:
    """Configuration for capturing model internals.

    If None is passed to QueryConfig.internals, ALL activations are captured.
    Use InternalsConfig.empty() to capture no activations.
    """

    activations: list[ActivationSpec] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "InternalsConfig":
        """Create an empty config that captures no activations."""
        return cls(activations=[])

    @classmethod
    def from_dict(cls, data: dict) -> "InternalsConfig":
        """Create config from dict format.

        Dict format: {"component_name": {"layers": [0, 1, 2]}, ...}
        Example: {"resid_post": {"layers": [8, 14]}}
        """
        specs = [
            ActivationSpec(component=comp, layers=spec.get("layers", []))
            for comp, spec in data.items()
        ]
        return cls(activations=specs)

    def is_empty(self) -> bool:
        """Return True if this config captures no activations."""
        return len(self.activations) == 0


@dataclass
class QueryConfig:
    """Query configuration."""

    internals: Optional[InternalsConfig] = None
    max_new_tokens: int = 256
    temperature: float = 0.0
    subsample: float = 1.0
    intervention: Optional[dict] = None  # Raw intervention config (loaded per-model)
    skip_generation: bool = (
        False  # If True, infer choice from probs only (~100x faster)
    )

    @classmethod
    def from_dict(cls, data: dict) -> "QueryConfig":
        """Create config from dict."""
        internals = None
        if data.get("internals"):
            internals = InternalsConfig.from_dict(data["internals"])

        return cls(
            internals=internals,
            max_new_tokens=data.get("max_new_tokens", 256),
            temperature=data.get("temperature", 0.0),
            subsample=data.get("subsample", 1.0),
            intervention=data.get("intervention"),
            skip_generation=data.get("skip_generation", False),
        )

    @classmethod
    def from_json(cls, path: "Path") -> "QueryConfig":
        """Load query config from JSON file."""
        data = load_json(path)
        return cls.from_dict(data)


class QueryRunner:
    """Query runner for preference datasets."""

    def __init__(self, config: QueryConfig):
        self.config = config
        self._model: Optional[ModelRunner] = None
        self._model_name: Optional[str] = None
        self._min_choice_prob = 0.5

    def _load_model(self, name: str) -> ModelRunner:
        if self._model is not None and self._model_name == name:
            return self._model
        self._model = ModelRunner(model_name=name)
        self._model_name = name
        self._intervention = None  # Reset intervention for new model
        return self._model

    def _load_intervention(self, model_runner: ModelRunner) -> Optional[Intervention]:
        """Load intervention config for the current model."""
        if self.config.intervention is None:
            return None

        return load_intervention_from_dict(self.config.intervention, model_runner)

    def _get_activation_names(self, model_runner: ModelRunner) -> list[str]:
        """Get hook names for activation capture.

        If internals is None, captures all activations at all layers.
        If internals.is_empty(), captures nothing.
        Otherwise, uses the specific config.
        """
        internals = self.config.internals

        # Empty config means no activations
        if internals is not None and internals.is_empty():
            return []

        # None means capture all activations
        if internals is None:
            n_layers = model_runner.n_layers
            components = ["resid_pre", "resid_post", "attn_out", "mlp_out"]
            return [
                f"blocks.{layer}.hook_{comp}"
                for layer in range(n_layers)
                for comp in components
            ]

        # Use specific config
        names = []
        for spec in internals.activations:
            for layer in spec.layers:
                names.append(f"blocks.{layer}.hook_{spec.component}")
        return names

    def query_dataset(
        self, prompt_dataset: PromptDataset, model_name: str
    ) -> PreferenceDataset:
        """Query a single dataset with a model. Returns results in memory."""

        model_runner = self._load_model(model_name)
        activation_names = self._get_activation_names(model_runner)
        intervention = self._load_intervention(model_runner)
        preferences = []

        samples = prompt_dataset.samples

        # Get choice_prefix from dataset config
        choice_prefix = (
            prompt_dataset.config.prompt_format_config.get_exact_prefix_before_choice()
        )

        if self.config.subsample < 1.0:
            n = max(1, int(len(samples) * self.config.subsample))
            samples = random.sample(samples, n)

        if intervention is not None:
            print(
                f"Using intervention: mode={intervention.mode} at layer {intervention.layer}"
            )

        print(f"Querying LLM for {len(samples)} samples...")
        for i, sample in enumerate(samples):
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(samples)}")

            sample_idx = sample.sample_idx
            prompt_text = sample.prompt.text
            pair = sample.prompt.preference_pair
            short_label = pair.short_term.label
            long_label = pair.long_term.label
            short_time = pair.short_term.time.to_years()
            long_time = pair.long_term.time.to_years()
            short_reward = pair.short_term.reward.value
            long_reward = pair.long_term.reward.value
            time_horizon = (
                sample.prompt.time_horizon.to_years()
                if sample.prompt.time_horizon
                else None
            )
            decoding_mismatch = False

            # Step 1: Get choice probabilities

            short_prob, long_prob = model_runner.get_label_probs(
                prompt_text, choice_prefix, (short_label, long_label)
            )
            short_response, long_response = model_runner.get_canonical_response_texts(
                choice_prefix, (short_label, long_label)
            )

            canonical_choice = "short_term" if short_prob > long_prob else "long_term"
            canonical_response = (
                short_response if short_prob > long_prob else long_response
            )

            choice_prob, alt_prob = (
                max(short_prob, long_prob),
                min(short_prob, long_prob),
            )

            if choice_prob < self._min_choice_prob:
                decoding_mismatch = True

            # Step 2: Generate response (or skip if skip_generation=True)
            if not self.config.skip_generation:
                generated_response = model_runner.generate(
                    prompt_text,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    intervention=intervention,
                )
                parsed_choice = parse_choice(
                    generated_response, short_label, long_label, choice_prefix
                )
                functional_response = generated_response
                if parsed_choice != canonical_choice:
                    decoding_mismatch = True
                if self._min_choice_prob < choice_prob:
                    choice = canonical_choice
                else:
                    choice = parsed_choice
            else:
                generated_response = ""
                functional_response = canonical_response
                choice = canonical_choice

            # Step 3: Capture internals (only if requested)
            internals = None
            if activation_names:
                _, cache = model_runner.run_with_cache(
                    prompt_text + functional_response,
                    names_filter=lambda name: name in activation_names,
                )
                activations = {}
                for name in activation_names:
                    if name in cache:
                        activations[name] = cache[name][0].cpu()
                internals = CapturedInternals(
                    activations=activations,
                    activation_names=list(activations.keys()),
                )

            preferences.append(
                PreferenceSample(
                    sample_idx=sample_idx,
                    choice=choice,
                    choice_prob=choice_prob,
                    alt_prob=alt_prob,
                    short_term_label=short_label,
                    long_term_label=long_label,
                    short_term_time=short_time,
                    long_term_time=long_time,
                    short_term_reward=short_reward,
                    long_term_reward=long_reward,
                    time_horizon=time_horizon,
                    prompt_text=prompt_text,
                    response_text=generated_response,
                    internals=internals,
                    internals_paths=None,
                    decoding_mismatch=decoding_mismatch,
                )
            )

        config_name = prompt_dataset.config.name

        return PreferenceDataset(
            prompt_dataset_id=prompt_dataset.dataset_id,
            model=model_name,
            preferences=preferences,
            prompt_dataset_name=config_name,
        )
