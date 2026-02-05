"""Scenario-based sample generation."""

import random
from pathlib import Path

from src.datasets.generator import DatasetGenerator

from .utils import SCENARIOS_DIR
from .activations import sample_to_dict


# ── Scenario loading & sample generation ────────────────────────────────────


def load_all_scenario_configs(
    scenarios_dir: Path = SCENARIOS_DIR,
) -> list[tuple[Path, object]]:
    """Load all scenario configs from the scenarios directory."""
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")

    scenario_files = list(scenarios_dir.glob("*.json"))
    if not scenario_files:
        raise FileNotFoundError(f"No scenario configs found in {scenarios_dir}")

    configs = []
    for path in sorted(scenario_files):
        try:
            dataset_config = DatasetGenerator.load_dataset_config(path)
            configs.append((path, dataset_config))
        except Exception as e:
            print(f"Warning: Failed to load scenario {path.name}: {e}")

    print(f"Loaded {len(configs)} scenario configs: {[c[1].name for c in configs]}")
    return configs


def generate_samples(n_samples: int, seed: int) -> list[dict]:
    """Generate samples from all diverse scenarios, returned as dicts.

    Distributes samples roughly equally across all available scenarios,
    then shuffles for mixing.
    """
    random.seed(seed)

    scenario_configs = load_all_scenario_configs()
    n_scenarios = len(scenario_configs)

    base_per_scenario = n_samples // n_scenarios
    extra = n_samples % n_scenarios

    all_samples = []
    sample_id = 0

    for i, (path, dataset_config) in enumerate(scenario_configs):
        n_for_scenario = base_per_scenario + (1 if i < extra else 0)
        if n_for_scenario == 0:
            continue

        generator = DatasetGenerator(dataset_config)
        scenario_samples = generator.generate_random(n_for_scenario)

        for sample in scenario_samples:
            sample.sample_id = sample_id
            all_samples.append(sample_to_dict(sample))
            sample_id += 1

        print(f"  {dataset_config.name}: {len(scenario_samples)} samples")

    random.shuffle(all_samples)
    return all_samples
