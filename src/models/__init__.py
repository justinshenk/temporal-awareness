"""Model inference and interventions module."""

from .model_runner import ModelRunner, ModelBackend, LabelProbsOutput
from .query_runner import (
    QueryRunner,
    QueryConfig,
    QueryOutput,
    PreferenceItem,
    InternalsConfig,
    CapturedInternals,
    parse_choice,
)
from .interventions import (
    Intervention,
    Target,
    create_intervention_hook,
    PatternMatcher,
)
from .intervention_utils import (
    steering,
    ablation,
    patch,
    scale,
    interpolate,
    compute_mean_activations,
    get_activations,
    random_direction,
)
from .intervention_loader import (
    load_intervention,
    load_intervention_from_dict,
    load_intervention_json,
    list_sample_interventions,
)

__all__ = [
    # Model runner
    "ModelRunner",
    "ModelBackend",
    "LabelProbsOutput",
    # Query runner
    "QueryRunner",
    "QueryConfig",
    "QueryOutput",
    "PreferenceItem",
    "InternalsConfig",
    "CapturedInternals",
    "parse_choice",
    # Interventions
    "Intervention",
    "Target",
    "create_intervention_hook",
    "PatternMatcher",
    # Intervention utils
    "steering",
    "ablation",
    "patch",
    "scale",
    "interpolate",
    "compute_mean_activations",
    "get_activations",
    "random_direction",
    # Intervention loader
    "load_intervention",
    "load_intervention_from_dict",
    "load_intervention_json",
    "list_sample_interventions",
]
