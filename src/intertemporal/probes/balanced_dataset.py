"""Balanced sampling utilities for probe training."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..preference import PreferenceDataset


def get_time_horizon_months(time_horizon: Optional[dict]) -> Optional[float]:
    """Convert time horizon dict to months."""
    if time_horizon is None:
        return None

    value = time_horizon.get("value", 0)
    unit = time_horizon.get("unit", "months")

    multiplier = {"days": 1 / 30, "weeks": 7 / 30, "months": 1, "years": 12}.get(unit, 1)
    return value * multiplier


def balance_by_choice(samples: list, random_seed: int = 42) -> list:
    """Balance samples by choice (short_term vs long_term)."""
    random.seed(random_seed)

    short = [s for s in samples if s.choice == "short_term"]
    long = [s for s in samples if s.choice == "long_term"]
    n = min(len(short), len(long))

    if n == 0:
        return []

    random.shuffle(short)
    random.shuffle(long)
    return short[:n] + long[:n]


def balance_by_time_horizon(
    samples: list,
    threshold_months: float = 12.0,
    random_seed: int = 42,
) -> list:
    """Balance samples by time horizon (<=threshold vs >threshold months)."""
    random.seed(random_seed)

    short, long = [], []
    for s in samples:
        months = get_time_horizon_months(s.time_horizon)
        if months is None:
            continue
        if months <= threshold_months:
            short.append(s)
        else:
            long.append(s)

    n = min(len(short), len(long))
    if n == 0:
        return []

    random.shuffle(short)
    random.shuffle(long)
    return short[:n] + long[:n]


def prepare_samples(
    pref_data: "PreferenceDataset",
    probe_type: str = "choice",
    balance_by: str = "choice",
    random_seed: int = 42,
) -> tuple[list, np.ndarray]:
    """Prepare balanced samples and labels for probe training.

    Uses ALL valid samples from the dataset after balancing.

    Args:
        pref_data: PreferenceDataset with preferences
        probe_type: "choice" (short_term vs long_term) or "time_horizon" (<=1yr vs >1yr)
        balance_by: "choice", "time_horizon", or "none"
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (samples, labels) where labels are 0/1 numpy array
    """
    random.seed(random_seed)
    samples = pref_data.filter_valid()

    if balance_by == "choice":
        samples = balance_by_choice(samples, random_seed)
    elif balance_by == "time_horizon":
        samples = balance_by_time_horizon(samples, random_seed=random_seed)

    if probe_type == "choice":
        labels = np.array([1 if s.choice == "long_term" else 0 for s in samples])
    else:  # time_horizon
        filtered, labels_list = [], []
        for s in samples:
            months = get_time_horizon_months(s.time_horizon)
            if months is not None:
                filtered.append(s)
                labels_list.append(1 if months > 12 else 0)
        samples, labels = filtered, np.array(labels_list)

    return samples, labels
