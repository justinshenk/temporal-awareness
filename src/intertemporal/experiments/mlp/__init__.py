"""MLP neuron analysis for intertemporal experiments.

Provides neuron-level analysis of MLP layers to understand:
1. Which neurons fire differently between short/long horizon prompts
2. Per-neuron logit contributions (via W_out projection)
3. Sparsity of the computation (few neurons vs distributed)
4. Max-activating prompts for top neurons
"""

from src.common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
