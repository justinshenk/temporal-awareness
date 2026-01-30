"""Experiment orchestration for intertemporal preference analysis.

This package provides unified functions for running the full analysis pipeline:
- Data generation
- Activation patching
- Attribution patching
- Contrastive analysis (steering vectors)
- Steering evaluation

IMPORTANT DESIGN PRINCIPLES:
1. NEVER use TransformerLens/NNsight/Pyvene APIs directly - ALWAYS use ModelRunner
2. All experiment functions must work with any backend (configurable via ModelRunner)
3. No magic numbers - use config parameters or named constants
4. Check for existing code before adding new functions (avoid duplication)
"""

from .config import ExperimentConfig, ExperimentResults
from .activation_patching import run_activation_patching
from .attribution_patching import run_attribution_patching
from .steering import compute_steering_vector, apply_steering
from .probe_training import run_probe_training
from .intertemporal import ExperimentArgs, run_experiment


__all__ = [
    "ExperimentConfig",
    "ExperimentResults",
    "run_activation_patching",
    "run_attribution_patching",
    "compute_steering_vector",
    "apply_steering",
    "run_probe_training",
    # Intertemporal experiment
    "ExperimentArgs",
    "run_experiment",
]
