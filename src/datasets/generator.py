"""
Dataset generator class for intertemporal preference experiments.

Generates datasets from config files with support for:
- Grid and random sampling methods
- Linear and logarithmic stepping
- Separate dataset and formatting configs
- Optional formatting variations (labels, time units, number spelling)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

from ..common.io import load_json
from ..formatting.prompt_formats import find_prompt_format_config
from ..formatting.formatting_variation import FormattingVariation, apply_time_variation
from .schemas import (
    ContextConfig,
    DatasetConfig,
    OptionRangeConfig,
    StepType,
)
from ..common.types import (
    DatasetSample,
    IntertemporalOption,
    PreferencePair,
    Prompt,
    RewardValue,
    TimeValue,
)


class DatasetGenerator:
    """
    Generator for intertemporal preference datasets.

    Reads config and generates samples with varying time horizons and options.
    Supports grid and random sampling with linear/logarithmic stepping.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
    ):
        self.dataset_config = dataset_config

    @classmethod
    def load_dataset_config(cls, path: Path) -> DatasetConfig:
        """Load and parse dataset config from JSON file."""
        data = load_json(path)
        return cls.load_dataset_config_from_dict(data)

    @classmethod
    def load_dataset_config_from_dict(cls, data: dict) -> DatasetConfig:
        """
        Parse dataset config from dictionary.

        Sample dict (see schemas: DatasetConfig, ContextConfig, OptionRangeConfig):
        {
            "name": "cityhousing",
            "context": {
                "reward_unit": "housing units",
                "role": "the city administration",
                "situation": "Plan for housing in the city.",
                "action_in_question": "build",
                "reasoning_ask": "why choice was made",
                "domain": "housing",
                "labels": ["a)", "b)"],
                "method": "grid",
                "seed": 42
            },
            "options": {
                "short_term": {
                    "reward_range": [2000, 5000],
                    "time_range": [[3, "months"], [1, "years"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"]
                },
                "long_term": { ... }  // same structure as short_term
            },
            "time_horizons": [[5, "months"], [15, "years"]]
        }
        """
        # Parse context (includes labels, method, seed)
        ctx = data["context"]
        labels = ctx.get("labels", ["a)", "b)"])
        if isinstance(labels, list):
            labels = tuple(labels)

        context = ContextConfig(
            reward_unit=ctx["reward_unit"],
            role=ctx["role"],
            situation=ctx["situation"],
            task_in_question=ctx.get("task_in_question", "to decide between"),
            reasoning_ask=ctx.get("reasoning_ask", "why this choice was made"),
            domain=ctx.get("domain", ""),
            extra_situation=ctx.get("extra_situation", ""),
            labels=labels,
            method=ctx.get("method", "grid"),
        )

        prompt_format_name = data.get("prompt_format", "default_prompt_format")
        prompt_format = find_prompt_format_config(prompt_format_name)

        # Parse options
        options = {}
        for key in ("short_term", "long_term"):
            opt = data["options"][key]
            options[key] = OptionRangeConfig(
                reward_range=tuple(opt["reward_range"]),
                time_range=(
                    TimeValue.parse_time_value(opt["time_range"][0]),
                    TimeValue.parse_time_value(opt["time_range"][1]),
                ),
                reward_steps=(
                    opt["reward_steps"][0],
                    StepType(opt["reward_steps"][1]),
                ),
                time_steps=(
                    opt["time_steps"][0],
                    StepType(opt["time_steps"][1]),
                ),
            )

        # Parse time horizons (null/None = no time horizon constraint)
        time_horizons = [
            TimeValue.parse_time_value(t) if t is not None else None
            for t in data["time_horizons"]
        ]

        # Parse optional formatting variations flag
        add_formatting_variations = data.get("add_formatting_variations", False)

        return DatasetConfig(
            name=data["name"],
            context=context,
            prompt_format=prompt_format,
            options=options,
            time_horizons=time_horizons,
            add_formatting_variations=add_formatting_variations,
        )

    def generate_steps(
        self,
        min_val: float,
        max_val: float,
        num_intervals: int,
        step_type: StepType,
    ) -> list[float]:
        """
        Generate stepped values between min and max.

        Args:
            min_val: Minimum value
            max_val: Maximum value
            num_intervals: Number of intervals (0 = midpoint only, 1 = endpoints, 2 = 3 values, etc.)
            step_type: LINEAR or LOGARITHMIC

        Returns:
            List of values (num_intervals + 1 values, or 1 value if num_intervals=0)
        """
        if num_intervals == 0:
            # Return midpoint
            if step_type == StepType.LINEAR:
                return [(min_val + max_val) / 2]
            else:  # LOGARITHMIC
                if min_val <= 0:
                    raise ValueError("Logarithmic stepping requires positive values")
                return [math.exp((math.log(min_val) + math.log(max_val)) / 2)]

        num_values = num_intervals + 1

        if step_type == StepType.LINEAR:
            step = (max_val - min_val) / num_intervals
            return [min_val + i * step for i in range(num_values)]
        else:  # LOGARITHMIC
            if min_val <= 0:
                raise ValueError("Logarithmic stepping requires positive values")
            log_min = math.log(min_val)
            log_max = math.log(max_val)
            log_step = (log_max - log_min) / num_intervals
            return [math.exp(log_min + i * log_step) for i in range(num_values)]

    def generate_time_steps(
        self,
        min_time: TimeValue,
        max_time: TimeValue,
        num_intervals: int,
        step_type: StepType,
    ) -> list[TimeValue]:
        """
        Generate stepped time values.

        Args:
            min_time: Minimum time
            max_time: Maximum time
            num_intervals: Number of intervals (0 = midpoint only)
            step_type: LINEAR or LOGARITHMIC

        Returns:
            List of TimeValue objects
        """
        # Convert to common unit (months) for stepping
        min_months = min_time.to_months()
        max_months = max_time.to_months()

        month_values = self.generate_steps(
            min_months, max_months, num_intervals, step_type
        )

        # Convert back, using appropriate unit based on magnitude
        result = []
        for months in month_values:
            if months >= 12:
                # Use years for 12+ months
                years = months / 12
                result.append(TimeValue(value=round(years, 1), unit="years"))
            else:
                result.append(TimeValue(value=round(months, 1), unit="months"))

        return result

    def generate_option_grid(self, option_key: str) -> list[tuple[float, TimeValue]]:
        """
        Generate grid of (reward, time) combinations for an option.

        Args:
            option_key: "short_term" or "long_term"

        Returns:
            List of (reward_value, time_value) tuples
        """
        opt = self.dataset_config.options[option_key]

        # Generate reward steps
        rewards = self.generate_steps(
            opt.reward_range[0],
            opt.reward_range[1],
            opt.reward_steps[0],
            opt.reward_steps[1],
        )

        # Generate time steps
        times = self.generate_time_steps(
            opt.time_range[0],
            opt.time_range[1],
            opt.time_steps[0],
            opt.time_steps[1],
        )

        # Create grid
        grid = []
        for reward in rewards:
            for time in times:
                grid.append((reward, time))

        return grid

    def format_question(
        self,
        left_option: IntertemporalOption,
        right_option: IntertemporalOption,
        time_horizon: Optional[TimeValue],
        labels: tuple[str, str],
        left_time_str: Optional[str] = None,
        right_time_str: Optional[str] = None,
        horizon_time_str: Optional[str] = None,
    ) -> str:
        """
        Format prompt text using template and context.

        Uses prompt_format's keyword system:
        - const_keywords: constant values (choice_prefix, reasoning_prefix, etc.)
        - keywords: values from context config (situation, role, etc.)
        - var_keywords: sample-specific values (time_horizon, left_term_label, etc.)

        Args:
            left_option: Option displayed on left (first)
            right_option: Option displayed on right (second)
            time_horizon: Decision time horizon (None = no constraint)
            labels: (left_label, right_label) tuple
            left_time_str: Optional formatted time string for left option
            right_time_str: Optional formatted time string for right option
            horizon_time_str: Optional formatted time string for horizon

        Returns:
            Formatted prompt text
        """
        ctx = self.dataset_config.context
        pf = self.dataset_config.prompt_format
        prompt = pf.question_template

        # Use provided time strings or default to str(time)
        left_time = left_time_str if left_time_str else str(left_option.time)
        right_time = right_time_str if right_time_str else str(right_option.time)
        horizon_str = (
            horizon_time_str
            if horizon_time_str
            else (str(time_horizon) if time_horizon else "")
        )

        # Build var_keywords values dict
        var_values = {
            "time_horizon": horizon_str,
            "left_term_label": labels[0],
            "left_term_reward": f"{round(left_option.reward.value):,}",
            "left_term_time": left_time,
            "right_term_label": labels[1],
            "right_term_reward": f"{round(right_option.reward.value):,}",
            "right_term_time": right_time,
        }

        # Build keywords values dict from context
        keyword_values = {
            "situation": ctx.situation,
            "extra_situation": ctx.extra_situation,
            "role": ctx.role,
            "task_in_question": ctx.task_in_question,
            "reward_units": ctx.reward_unit,
            "reasoning_ask": ctx.reasoning_ask,
        }

        # Handle time_horizon_spec specially - expand it first if time_horizon exists
        const_keywords = dict(pf.const_keywords)
        if time_horizon is not None:
            time_horizon_spec = const_keywords.get("time_horizon_spec", "")
            time_horizon_spec = time_horizon_spec.replace("[time_horizon]", horizon_str)
            const_keywords["time_horizon_spec"] = time_horizon_spec
        else:
            const_keywords["time_horizon_spec"] = ""

        # Replace const_keywords
        for key, value in const_keywords.items():
            prompt = prompt.replace(f"[{key}]", value)

        # Replace keywords from context
        for key, value in keyword_values.items():
            prompt = prompt.replace(f"[{key}]", value)

        # Replace var_keywords
        for key, value in var_values.items():
            prompt = prompt.replace(f"[{key}]", value)

        # Validate no unreplaced placeholders remain
        self._validate_no_unreplaced_placeholders(prompt, "question_template")

        return prompt

    def _validate_no_unreplaced_placeholders(
        self, text: str, context: str = ""
    ) -> None:
        """
        Validate that no [PLACEHOLDER] patterns remain in text.

        Args:
            text: Text to check
            context: Description of where this text came from (for error messages)

        Raises:
            ValueError: If unreplaced placeholders are found
        """
        import re

        # Find [WORD] patterns that look like placeholders:
        # - Must contain underscore OR be longer than 2 chars
        # - This excludes labels like [A], [B], [1], [2] which are intentional
        all_brackets = re.findall(r"\[[A-Z][A-Z0-9_]*\]", text)
        placeholders = [
            p for p in all_brackets if "_" in p or len(p) > 4
        ]  # [XX] = 4 chars
        if placeholders:
            unique = sorted(set(placeholders))
            ctx = f" in {context}" if context else ""
            raise ValueError(
                f"Unreplaced placeholders found{ctx}: {', '.join(unique)}\n"
                f"Text snippet: {text[:200]}..."
            )

    def _get_formatting_variation(self) -> FormattingVariation:
        """Get a formatting variation (random if enabled, default otherwise)."""
        if self.dataset_config.add_formatting_variations:
            return FormattingVariation.random(allow_all=True)
        return FormattingVariation.default()

    def create_sample(
        self,
        sample_id: int,
        short_term_data: tuple[float, TimeValue],
        long_term_data: tuple[float, TimeValue],
        time_horizon: Optional[TimeValue],
    ) -> DatasetSample:
        """
        Create a dataset sample from option data.

        Randomly assigns short_term to left or right position.
        Applies formatting variations if enabled in config.

        Args:
            sample_id: Unique sample ID
            short_term_data: (reward, time) for short-term option
            long_term_data: (reward, time) for long-term option
            time_horizon: Decision time horizon (None = no constraint)

        Returns:
            DatasetSample instance
        """
        ctx = self.dataset_config.context

        # Get formatting variation (random if enabled, default otherwise)
        variation = self._get_formatting_variation()

        # Use variation labels if enabled, otherwise use config labels
        if self.dataset_config.add_formatting_variations:
            labels = variation.labels
        else:
            labels = ctx.labels

        # Randomly assign short_term to left (index 0) or right (index 1)
        # The variation.flip_order adds another layer of randomization
        short_on_left = random.choice([True, False])
        if variation.flip_order:
            short_on_left = not short_on_left

        if short_on_left:
            left_label, right_label = labels[0], labels[1]
            short_term_label, long_term_label = left_label, right_label
        else:
            left_label, right_label = labels[0], labels[1]
            short_term_label, long_term_label = right_label, left_label

        short_term = IntertemporalOption(
            label=short_term_label,
            time=short_term_data[1],
            reward=RewardValue(value=round(short_term_data[0]), unit=ctx.reward_unit),
        )

        long_term = IntertemporalOption(
            label=long_term_label,
            time=long_term_data[1],
            reward=RewardValue(value=round(long_term_data[0]), unit=ctx.reward_unit),
        )

        pair = PreferencePair(short_term=short_term, long_term=long_term)

        # Determine which option goes on left/right for formatting
        if short_on_left:
            left_option, right_option = short_term, long_term
        else:
            left_option, right_option = long_term, short_term

        # Apply time variations for prompt formatting
        _, left_time_str = apply_time_variation(left_option.time, variation)
        _, right_time_str = apply_time_variation(right_option.time, variation)

        # Apply time variation to horizon if present
        horizon_time_str = None
        if time_horizon is not None:
            _, horizon_time_str = apply_time_variation(time_horizon, variation)

        question_text = self.format_question(
            left_option,
            right_option,
            time_horizon,
            labels,
            left_time_str=left_time_str,
            right_time_str=right_time_str,
            horizon_time_str=horizon_time_str,
        )

        # Format response_template with labels and const_keywords
        pf = self.dataset_config.prompt_format
        response_format = self.dataset_config.prompt_format.response_template

        # Replace const_keywords
        for key, value in pf.const_keywords.items():
            response_format = response_format.replace(f"[{key}]", value)

        # Replace var_keywords for labels
        response_format = response_format.replace("[left_term_label]", labels[0])
        response_format = response_format.replace("[right_term_label]", labels[1])

        prompt_text = question_text + "\n" + response_format

        prompt = Prompt(
            preference_pair=pair,
            time_horizon=time_horizon,
            text=prompt_text,
        )

        return DatasetSample(
            sample_id=sample_id,
            prompt=prompt,
        )

    def generate_grid(self) -> list[DatasetSample]:
        """
        Generate samples using grid method.

        Creates all combinations of:
        - Short-term option grid
        - Long-term option grid
        - Time horizons

        Returns:
            List of DatasetSample objects
        """
        short_term_grid = self.generate_option_grid("short_term")
        long_term_grid = self.generate_option_grid("long_term")
        time_horizons = self.dataset_config.time_horizons

        samples = []
        sample_id = 0

        for time_horizon in time_horizons:
            for short_data in short_term_grid:
                for long_data in long_term_grid:
                    sample = self.create_sample(
                        sample_id, short_data, long_data, time_horizon
                    )
                    samples.append(sample)
                    sample_id += 1

        return samples

    def generate_random(self, num_samples: int = 100) -> list[DatasetSample]:
        """
        Generate samples using random sampling.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of DatasetSample objects
        """
        ctx = self.dataset_config.context
        time_horizons = self.dataset_config.time_horizons

        samples = []

        for sample_id in range(num_samples):
            # Random time horizon
            time_horizon = random.choice(time_horizons)

            # Random short-term option
            st_opt = self.dataset_config.options["short_term"]
            st_reward = random.uniform(*st_opt.reward_range)
            st_time_months = random.uniform(
                st_opt.time_range[0].to_months(),
                st_opt.time_range[1].to_months(),
            )
            if st_time_months >= 12:
                st_time = TimeValue(value=round(st_time_months / 12, 1), unit="years")
            else:
                st_time = TimeValue(value=round(st_time_months, 1), unit="months")

            # Random long-term option
            lt_opt = self.dataset_config.options["long_term"]
            lt_reward = random.uniform(*lt_opt.reward_range)
            lt_time_months = random.uniform(
                lt_opt.time_range[0].to_months(),
                lt_opt.time_range[1].to_months(),
            )
            if lt_time_months >= 12:
                lt_time = TimeValue(value=round(lt_time_months / 12, 1), unit="years")
            else:
                lt_time = TimeValue(value=round(lt_time_months, 1), unit="months")

            sample = self.create_sample(
                sample_id,
                (st_reward, st_time),
                (lt_reward, lt_time),
                time_horizon,
            )
            samples.append(sample)

        return samples

    def generate(self, num_random_samples: Optional[int] = None) -> list[DatasetSample]:
        """
        Generate dataset samples and metadata.

        Args:
            num_random_samples: Number of samples for random method (ignored for grid)

        Returns:
            Tuple of (samples, metadata)
        """
        if self.dataset_config.context.method == "grid":
            samples = self.generate_grid()
        else:  # random
            samples = self.generate_random(num_random_samples or 100)

        return samples
