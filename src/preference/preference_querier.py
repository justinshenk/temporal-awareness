"""Preference querier for datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..common.io import load_json
from ..common.types import PreferenceSample, CapturedInternals
from ..models.interventions import load_intervention_from_dict, Intervention
from ..models.binary_choice_runner import (
    BinaryChoiceRunner,
    parse_choice_from_generated_response,
)
from .preference_dataset import PreferenceDataset
from ..prompt import PromptDataset


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


class PreferenceQuerier:
    """Preference querier for datasets."""

    def __init__(self, config: QueryConfig):
        self.config = config
        self._runner: Optional[BinaryChoiceRunner] = None
        self._min_choice_prob = 0.5

    def _load_model(self, name: str) -> BinaryChoiceRunner:
        if self._runner is not None and self._runner.model_name == name:
            return self._runner
        self._runner = BinaryChoiceRunner(model_name=name)
        self._intervention = None  # Reset intervention for new model
        return self._runner

    def _load_intervention(self, runner: BinaryChoiceRunner) -> Optional[Intervention]:
        """Load intervention config for the current model."""
        if self.config.intervention is None:
            return None

        return load_intervention_from_dict(self.config.intervention, runner)

    def _get_activation_names(self, runner: BinaryChoiceRunner) -> list[str]:
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
            return runner.get_all_names_for_internals()

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

        runner = self._load_model(model_name)

        samples = prompt_dataset.samples
        if self.config.subsample < 1.0:
            n = max(1, int(len(samples) * self.config.subsample))
            samples = random.sample(samples, n)

        activation_names = self._get_activation_names(runner)
        intervention = self._load_intervention(runner)

        if intervention:
            print(
                f"query_dataset: Using intervention: mode={intervention.mode} at layer {intervention.layer}"
            )
        if activation_names:
            print(f"query_dataset: Capturing activations/internals {activation_names}")

        # Get choice_prefix from dataset config
        choice_prefix = (
            prompt_dataset.config.prompt_format_config.get_exact_prefix_before_choice()
        )

        print(f"query_dataset: Querying LLM for {len(samples)} samples...")
        preferences = []
        for i, sample in enumerate(samples):
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(samples)}")

            sample_idx = sample.sample_idx
            time_horizon = (
                sample.prompt.time_horizon.to_years()
                if sample.prompt.time_horizon
                else None
            )
            prompt_text = sample.prompt.text

            pair = sample.prompt.preference_pair
            short_label = pair.short_term.label
            long_label = pair.long_term.label
            short_time = pair.short_term.time.to_years()
            long_time = pair.long_term.time.to_years()
            short_reward = pair.short_term.reward.value
            long_reward = pair.long_term.reward.value

            decoding_mismatch = False

            # Step 1: Query choice prob based on format
            choice = runner.choose(
                prompt_text, choice_prefix, (short_label, long_label)
            )

            # Step 2: Generate response (or skip if skip_generation=True)
            if not self.config.skip_generation:
                generated_response = runner.generate(
                    prompt_text,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    intervention=intervention,
                )
                generated_choice_idx = parse_choice_from_generated_response(
                    generated_response, short_label, long_label, choice_prefix
                )
                if generated_choice_idx != choice.choice_idx:
                    decoding_mismatch = True
                functional_response = generated_response
            else:
                generated_response = ""
                functional_response = choice.response_texts[choice.choice_idx]

            # Step 3: Capture internals (only if requested)
            captured_internals = None
            if activation_names:
                _, cache = runner.run_with_cache(
                    prompt_text + functional_response,
                    names_filter=lambda name: name in activation_names,
                )
                captured_internals = CapturedInternals.from_activation_names(
                    activation_names, cache
                )

            preferences.append(
                PreferenceSample(
                    # Choice
                    choice=choice.without_data(),
                    # Sample Info
                    sample_idx=sample_idx,
                    time_horizon=time_horizon,
                    prompt_text=prompt_text,
                    response_text=generated_response,
                    # Other Choice Info
                    short_term_label=short_label,
                    long_term_label=long_label,
                    short_term_time=short_time,
                    long_term_time=long_time,
                    short_term_reward=short_reward,
                    long_term_reward=long_reward,
                    # Extra Info
                    internals=captured_internals,
                    internals_paths=None,
                    decoding_mismatch=decoding_mismatch,
                )
            )

        return PreferenceDataset(
            prompt_dataset_id=prompt_dataset.dataset_id,
            model=model_name,
            preferences=preferences,
            prompt_dataset_name=prompt_dataset.config.name,
        )
