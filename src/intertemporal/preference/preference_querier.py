"""Preference querier for datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ...common.file_io import load_json
from ...inference import InternalsConfig, CapturedInternals
from ..common.preference_types import PreferenceSample
from ...inference.interventions import (
    Intervention,
    InterventionTarget,
    random_direction,
)
from ...binary_choice.binary_choice_runner import BinaryChoiceRunner
from ...binary_choice.choice_utils import verify_greedy_generation
from .preference_dataset import PreferenceDataset
from ..prompt import PromptDataset


@dataclass
class PreferenceQueryConfig:
    """Query configuration for preference experiments."""

    internals: Optional[InternalsConfig] = None
    max_new_tokens: int = 256
    temperature: float = 0.0
    subsample: float = 1.0
    intervention: Optional[dict] = None  # Raw intervention config (loaded per-model)
    skip_generation: bool = (
        False  # If True, infer choice from probs only (~100x faster)
    )

    @classmethod
    def from_dict(cls, data: dict) -> "PreferenceQueryConfig":
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
    def from_json(cls, path: "Path") -> "PreferenceQueryConfig":
        """Load query config from JSON file."""
        data = load_json(path)
        return cls.from_dict(data)


class PreferenceQuerier:
    """Preference querier for datasets."""

    def __init__(self, config: PreferenceQueryConfig):
        self.config = config
        self._runner: Optional[BinaryChoiceRunner] = None

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

        cfg = self.config.intervention

        # Handle "random" values
        values = cfg.get("values", 0)
        if values == "random":
            values = random_direction(runner.d_model)

        # Parse target
        target_data = cfg.get("target", "all")
        if target_data == "all" or target_data is None:
            target = InterventionTarget.all()
        elif isinstance(target_data, dict):
            target = InterventionTarget.at(
                positions=target_data.get("positions"),
                layers=target_data.get("layers"),
            )
        else:
            target = InterventionTarget.all()

        return Intervention(
            layer=cfg.get("layer", 0),
            mode=cfg.get("mode", "add"),
            values=values,
            target=target,
            component=cfg.get("component", "resid_post"),
            strength=cfg.get("strength", 1.0),
        )

    def _get_activation_names(self, runner: BinaryChoiceRunner) -> list[str]:
        """Get hook names for activation capture."""
        internals = self.config.internals

        # Empty config means no activations
        if internals is None:
            return []

        if internals.save_all:
            return runner.get_all_names_for_internals()

        # Use specific config
        return internals.get_names()

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
        choice_prefix = prompt_dataset.config.prompt_format_config.get_response_prefix_before_choice()

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
                    prefilling=runner.skip_thinking_prefix,
                )
                decoding_mismatch = verify_greedy_generation(
                    choice,
                    generated_response,
                    short_label,
                    long_label,
                    choice_prefix,
                    runner=runner,
                    prompt=prompt_text,
                )
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

            # Compute matches_rational and matches_associated
            choice_idx = choice.choice_idx
            expected_rational = sample.expected_rational_choice
            associated = sample.associated_choice

            matches_rational = (
                (choice_idx == expected_rational)
                if expected_rational is not None
                else None
            )
            matches_associated = (
                (choice_idx == associated) if associated is not None else None
            )

            preferences.append(
                PreferenceSample(
                    # Choice
                    choice=choice,
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
                    choice_prefix=choice_prefix,
                    # Extra Info
                    internals=captured_internals,
                    internals_paths=None,
                    decoding_mismatch=decoding_mismatch,
                    formatting_id=sample.formatting_id,
                    matches_rational=matches_rational,
                    matches_associated=matches_associated,
                )
            )

        return PreferenceDataset(
            prompt_dataset_id=prompt_dataset.dataset_id,
            model=model_name,
            preferences=preferences,
            prompt_dataset_name=prompt_dataset.config.name,
            prompt_format=prompt_dataset.config.prompt_format,
        )
