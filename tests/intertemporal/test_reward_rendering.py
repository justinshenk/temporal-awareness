"""Reward values must render faithfully, including fractional ones.

Rewards were rendered with `f"{round(value):,}"`. Python's round() is
banker's rounding, so a 0.5 reward printed as "0" — the health domain offers
"quality-adjusted life years" from 0.5, and 4,536 of its prompts offered a
worthless option. Every other domain has integer rewards >= 1, so the bug was
invisible until a fractional domain was run.

The integer path is load-bearing: the paper's investment prompts must render
byte-identically after the fix.
"""

from __future__ import annotations

import pytest

from src.intertemporal.prompt.prompt_dataset_generator import (
    format_reward,
    normalize_reward,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        # Logarithmic stepping noise must not reach the prompt.
        (30000.000000000007, 30000),
        (999.9999999999999, 1000),
        (1000.0000000000001, 1000),
        # Genuinely fractional values survive untouched.
        (0.5, 0.5),
        (2.3, 2.3),
        (1.4, 1.4),
    ],
)
def test_normalize_reward(value, expected):
    assert normalize_reward(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        # The regression: these must not collapse to an integer.
        (0.5, "0.5"),
        (1.5, "1.5"),
        (2.75, "2.75"),
        (0.25, "0.25"),
        # Whole numbers stay integers, with thousands separators.
        (1, "1"),
        (5, "5"),
        (1000, "1,000"),
        (100000, "100,000"),
        # Floats that are exactly whole render as integers, not "1000.0".
        (1000.0, "1,000"),
        (5.0, "5"),
        # Trailing zeros are not carried.
        (2.50, "2.5"),
    ],
)
def test_format_reward(value, expected):
    assert format_reward(value) == expected


def test_half_no_longer_rounds_to_zero():
    """The exact failure that put '0 quality-adjusted life years' in prompts."""
    assert round(0.5) == 0  # the old behaviour, for the record
    assert format_reward(0.5) == "0.5"


def test_large_fractional_keeps_separator_and_decimal():
    assert format_reward(1234.5) == "1,234.5"


# --- step rounding -----------------------------------------------------------
# The renderer alone is not enough: rewards are rounded when the grid is built,
# and that is where 0.5 actually became 0. `_round_time` already cascades to
# avoid "0 years"; reward rounding needs the same guarantee.


@pytest.fixture
def generator():
    import json

    from src.intertemporal.prompt.prompt_dataset_config import PromptDatasetConfig
    from src.intertemporal.prompt.prompt_dataset_generator import PromptDatasetGenerator

    with open("data/intertemporal/health/health_geometry.json") as fh:
        cfg = PromptDatasetConfig.from_dict(json.load(fh))
    return PromptDatasetGenerator(cfg)


@pytest.mark.parametrize("value", [0.5, 0.25, 0.1, 0.004, 0.9])
def test_positive_rewards_never_round_to_zero(generator, value):
    assert generator._round_reward(value) > 0, f"{value} collapsed to zero"


@pytest.mark.parametrize(
    "value,expected", [(1.4, 1), (2.6, 3), (5.0, 5), (40, 40), (0.0, 0)]
)
def test_ordinary_rounding_is_unchanged(generator, value, expected):
    assert generator._round_reward(value) == expected


@pytest.mark.parametrize(
    "domain", ["health", "wellbeing", "cityhousing", "investment", "charity"]
)
def test_no_domain_renders_a_worthless_or_absurd_reward(domain):
    """No prompt may offer a zero reward or a reward with absurd precision."""
    import json
    import re

    from src.intertemporal.prompt.prompt_dataset_config import PromptDatasetConfig
    from src.intertemporal.prompt.prompt_dataset_generator import PromptDatasetGenerator

    path = f"data/intertemporal/{domain}/{domain}_geometry.json"
    with open(path) as fh:
        cfg = PromptDatasetConfig.from_dict(json.load(fh))
    dataset = PromptDatasetGenerator(cfg).generate()
    samples = getattr(dataset, "samples", dataset)

    unit = cfg.context.reward_unit.split()[0]
    zero = re.compile(rf"(^|\s)0 {re.escape(unit)}")
    absurd = re.compile(r"\d\.\d{4,}")

    for sample in samples:
        text = sample.text if isinstance(sample.text, str) else "\n".join(sample.text)
        assert not zero.search(text), f"{domain}: zero reward in\n{text[:200]}"
        assert not absurd.search(text), f"{domain}: absurd precision in\n{text[:200]}"


def test_health_dataset_has_no_worthless_option(generator):
    """End to end: no generated health prompt may offer a zero reward."""
    dataset = generator.generate()
    samples = getattr(dataset, "samples", dataset)
    zero = [
        s
        for s in samples
        if min(
            s.prompt.preference_pair.short_term.reward.value,
            s.prompt.preference_pair.long_term.reward.value,
        )
        <= 0
    ]
    assert not zero, f"{len(zero)} of {len(samples)} prompts offer a zero reward"
