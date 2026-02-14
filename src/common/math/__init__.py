"""Math utilities for LLM analysis.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.

Module hierarchy:
- entropy_diversity/: Core theory (q_diversity, renyi_entropy, power_mean, etc.)
  - structure_aware: Structure-aware diversity (compliance, orientation, deviance)
- num_types: Type aliases (Num, Nums, is_tensor)
- math_primitives: Low-level helpers (argmin, normalize, etc.)
- aggregation: AggregationMethod, aggregate

Application-level metrics (wrap entropy_diversity with domain-specific signatures):
- trajectory_metrics: Sequence metrics (perplexity, alpha_inv_perplexity, etc.)
- branch_metrics: Distribution metrics (q_branch_diversity, vocab_entropy_from_logits)
- fork_metrics: Binary choice metrics (q_fork_diversity, log_odds, margin, etc.)

Usage:
    from src.common.math import AggregationMethod, aggregate
    from src.common.math import perplexity, q_diversity, shannon_entropy
    from src.common.math import orientation, deviance, core_entropy
    from src.common.analysis import XenoStructure, XenoSystem, XenoTrajectory
"""

from src.common.auto_export import auto_export

# auto_export imports all modules in this package and re-exports their public names
# Do NOT add `from .xxx import *` statements - let auto_export handle it
__all__ = auto_export(__file__, __name__, globals())
