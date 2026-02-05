"""Data structures and scenario-based sample generation."""

import random
from dataclasses import dataclass, fields
from pathlib import Path


from .utils import SCENARIOS_DIR

# ── Constants ───────────────────────────────────────────────────────────────

HORIZON_NONE = 0  # No time horizon specified
HORIZON_SHORT = 1  # <= 2 years
HORIZON_MEDIUM = 2  # 2-5 years
HORIZON_LONG = 3  # > 5 years

CHOICE_SHORT_TERM = 0
CHOICE_LONG_TERM = 1
CHOICE_UNKNOWN = -1


# ── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class Sentence:
    """A sentence with metadata about its origin in the prompt/response."""

    text: str
    source: str  # "prompt" or "response"
    section: str  # prompt: "situation","task","consider","action","format"
    # response: "choice","reasoning"

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "section": self.section}

    @classmethod
    def from_dict(cls, d: dict) -> "Sentence":
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


# ── Sample conversion ───────────────────────────────────────────────────────


def _horizon_bucket(time_horizon) -> int:
    """Convert a TimeValue to a horizon bucket."""
    if time_horizon is None:
        return HORIZON_NONE
    months = time_horizon.to_months()
    if months <= 12:
        return HORIZON_SHORT
    elif months <= (12 * 5):
        return HORIZON_MEDIUM
    return HORIZON_LONG


def sample_to_dict(sample) -> dict:
    """Convert a DatasetSample object to a JSON-serializable dict."""
    th = sample.prompt.time_horizon
    return {
        "sample_id": sample.sample_id,
        "prompt_text": sample.prompt.text,
        "time_horizon_bucket": _horizon_bucket(th),
        "time_horizon_months": th.to_months() if th else None,
        "short_term_label": sample.prompt.preference_pair.short_term.label,
        "long_term_label": sample.prompt.preference_pair.long_term.label,
        "short_term_time_months": sample.prompt.preference_pair.short_term.time.to_months(),
        "long_term_time_months": sample.prompt.preference_pair.long_term.time.to_months(),
    }


# ── Scenario loading & sample generation ────────────────────────────────────


def load_all_scenario_configs(
    scenarios_dir: Path = SCENARIOS_DIR,
) -> list[tuple[Path, object]]:
    """Load all scenario configs from the scenarios directory."""
    from src.datasets.generator import DatasetGenerator

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
    from src.datasets.generator import DatasetGenerator

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
