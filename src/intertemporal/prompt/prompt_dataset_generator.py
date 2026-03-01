"""
Prompt dataset generator class for intertemporal preference experiments.

Generates prompt datasets from config files with support for:
- Grid and random sampling methods
- Linear and logarithmic stepping
- Separate dataset and formatting configs
- Optional formatting variations (labels, time units, number spelling)
"""

from __future__ import annotations

import math
import random
import re
from typing import Optional
from itertools import product
from ..formatting.formatting_variation import (
    FormattingVariation,
    apply_time_variation,
    get_formatting_id,
)
from .prompt_dataset_config import PromptDatasetConfig, StepType
from .prompt_dataset import PromptDataset
from ..common.preference_types import (
    PromptSample,
    IntertemporalOption,
    PreferencePair,
    Prompt,
    RewardValue,
    TimeValue,
)


class PromptDatasetGenerator:
    """
    Generator for intertemporal preference prompt datasets.

    Reads config and generates samples with varying time horizons and options.
    Supports grid and random sampling with linear/logarithmic stepping.
    """

    def __init__(
        self,
        dataset_config: PromptDatasetConfig,
    ):
        self.dataset_config = dataset_config

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
        - prompt_const_keywords: constant values (choice_prefix, reasoning_prefix, etc.)
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
        pf = self.dataset_config.prompt_format_config

        # Assemble question template (conditionally includes time-horizon spec)
        prompt = pf.question_template(time_horizon)

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

        # Replace prompt_const_keywords (markers like SITUATION:, TASK:, etc.)
        for key, value in pf.prompt_const_keywords.items():
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

    def get_default_formatting(self):
        default_var = FormattingVariation.default()
        default_var.labels = self.dataset_config.context.labels
        return default_var

    def _process_formatting_variation(
        self, variation: Optional[FormattingVariation]
    ) -> FormattingVariation:
        need_to_reformat = (
            self.dataset_config.add_formatting_variations
            and not self.dataset_config.do_variation_grid
        )
        if need_to_reformat:
            return FormattingVariation.random()
        if not variation:
            return self.get_default_formatting()
        return variation

    def create_sample(
        self,
        sample_idx: int,
        short_term_data: tuple[float, TimeValue],
        long_term_data: tuple[float, TimeValue],
        time_horizon: Optional[TimeValue],
        variation: Optional[FormattingVariation] = None,
    ) -> PromptSample:
        """
        Create a dataset sample from option data.

        Randomly assigns short_term to left or right position.
        Applies formatting variations if enabled in config.

        Args:
            sample_idx: Unique sample ID
            short_term_data: (reward, time) for short-term option
            long_term_data: (reward, time) for long-term option
            time_horizon: Decision time horizon (None = no constraint)

        Returns:
            PromptSample instance
        """
        ctx = self.dataset_config.context

        variation = self._process_formatting_variation(variation)
        labels = variation.labels

        short_on_left = True
        if variation.flip_order:
            short_on_left = random.choice([True, False])
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

        # Format response_template with labels and prompt_const_keywords
        pf = self.dataset_config.prompt_format_config
        response_format = pf.response_template

        # Replace prompt_const_keywords
        for key, value in pf.prompt_const_keywords.items():
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

        # Compute formatting_id based on which label is assigned to short_term
        formatting_id = get_formatting_id(short_term_label, long_term_label)

        return PromptSample(
            sample_idx=sample_idx,
            prompt=prompt,
            formatting_id=formatting_id,
        )

    def generate_formatting_variation_grid(self):
        if self.dataset_config.do_variation_grid:
            return FormattingVariation.get_grid()
        return [self.get_default_formatting()]

    def generate_grid(self):
        short_term_grid = self.generate_option_grid("short_term")
        long_term_grid = self.generate_option_grid("long_term")
        time_horizons_grid = self.dataset_config.time_horizons
        var_grid = self.generate_formatting_variation_grid()
        full_grid = product(
            short_term_grid, long_term_grid, time_horizons_grid, var_grid
        )
        return full_grid

    def generate_samples(self) -> list[PromptSample]:
        """
        Generate samples using grid method.

        Creates all combinations of:
        - Short-term option grid
        - Long-term option grid
        - Time horizons

        Returns:
            List of PromptSample objects
        """

        samples = []
        grid = self.generate_grid()
        for i, params in enumerate(grid):
            sample = self.create_sample(i, *params)
            samples.append(sample)

        return samples

    def generate(self) -> PromptDataset:
        """
        Generate prompt dataset with samples.

        Returns:
            PromptDataset with generated samples
        """
        samples = self.generate_samples()
        return PromptDataset(
            dataset_id=self.dataset_config.get_id(),
            config=self.dataset_config,
            samples=samples,
        )
