"""Scenario-based sample generation."""

from pathlib import Path

from src.prompt_datasets import PromptDatasetConfig, PromptDatasetGenerator

from .utils import SCENARIOS_DIR


# ── Scenario loading & sample generation ────────────────────────────────────


def load_all_scenario_configs(
    scenarios_dir: Path = SCENARIOS_DIR,
) -> list[tuple[Path, PromptDatasetConfig]]:
    """Load all scenario configs from the scenarios directory."""
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")

    scenario_files = list(scenarios_dir.glob("*.json"))
    if not scenario_files:
        raise FileNotFoundError(f"No scenario configs found in {scenarios_dir}")

    configs = []
    for path in sorted(scenario_files):
        try:
            dataset_config = PromptDatasetConfig.load_from_json(path)
            configs.append((path, dataset_config))
        except Exception as e:
            print(f"Warning: Failed to load scenario {path.name}: {e}")

    print(f"Loaded {len(configs)} scenario configs: {[c[1].name for c in configs]}")
    return configs


def generate_samples() -> list:
    all_samples = []
    scenario_configs = load_all_scenario_configs()
    for _, dataset_config in scenario_configs:
        generator = PromptDatasetGenerator(dataset_config)
        prompt_dataset = generator.generate()
        all_samples.extend(prompt_dataset.samples)
        print(f"  {dataset_config.name}: {len(prompt_dataset.samples)} samples")
    return all_samples
