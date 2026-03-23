"""MLP neuron analysis for intertemporal experiments.

Provides neuron-level analysis of MLP layers to understand:
1. Which neurons fire differently between short/long horizon prompts
2. Per-neuron logit contributions (via W_out projection)
3. Sparsity of the computation (few neurons vs distributed)
4. Max-activating prompts for top neurons
"""

from .mlp_analysis_results import (
    NeuronInfo,
    MLPNeuronLayerResult,
    MLPPairResult,
    MLPAggregatedResults,
)
from .mlp_analysis_run import run_mlp_analysis

__all__ = [
    "NeuronInfo",
    "MLPNeuronLayerResult",
    "MLPPairResult",
    "MLPAggregatedResults",
    "run_mlp_analysis",
]
