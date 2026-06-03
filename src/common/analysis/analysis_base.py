"""Base class for distributional analysis with prob/odds expansion.

Provides DistributionalAnalysis which automatically converts logprob fields
to probability fields in to_dict() output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..base_schema import BaseSchema


@dataclass
class DistributionalAnalysis(BaseSchema):
    """Base class for analysis dataclasses that adds prob/odds fields to to_dict().

    Automatically converts (via _to_dict_hook, called by _canon):
    - *_logprob (float) -> *_prob via exp(logprob)
    - *_logprobs (list) -> *_probs via [exp(lp) for lp in logprobs]
    - log_odds -> odds via exp(log_odds)
    - *_log_odds -> *_odds via exp(log_odds)
    """

    def _to_dict_hook(self, d: dict) -> dict:
        """Hook called by _canon to add probability/odds fields.

        This is called automatically during serialization, ensuring nested
        analysis objects also get their prob/odds fields expanded.
        """
        return self._expand_logprob_fields(d)

    def _expand_logprob_fields(self, d: dict) -> dict:
        """Add probability fields for logprob fields."""
        additions = {}

        def _exp_safe(lp: float) -> float:
            """Safely compute exp(logprob), rounded to 4 decimal places."""
            if isinstance(lp, str):
                # Handle "Inf", "-Inf", "NaN" strings from _canon
                if lp == "Inf":
                    return float("inf")
                if lp == "-Inf" or lp == "NaN":
                    return 0.0
                return 0.0
            if math.isfinite(lp):
                return round(math.exp(lp), 4)
            return 0.0 if lp == float("-inf") else float("inf")

        for key, value in d.items():
            if value is None:
                continue

            # Handle exact "logprob" field -> "prob"
            if key == "logprob" and isinstance(value, (int, float, str)):
                additions["prob"] = _exp_safe(value)

            # Handle scalar *_logprob fields -> *_prob
            elif key.endswith("_logprob") and isinstance(value, (int, float, str)):
                prob_key = key.replace("_logprob", "_prob")
                additions[prob_key] = _exp_safe(value)

            # Handle *_logprobs sequences -> *_probs (e.g., next_token_logprobs)
            elif key.endswith("_logprobs") and isinstance(value, (list, tuple)):
                prob_key = key.replace("_logprobs", "_probs")
                additions[prob_key] = [_exp_safe(lp) for lp in value]

            # Handle sequence logprob fields (e.g., logprob_trajectory -> prob_trajectory)
            elif "logprob" in key and isinstance(value, list):
                prob_key = key.replace("logprob", "prob")
                additions[prob_key] = [_exp_safe(lp) for lp in value]

            # Handle log_odds -> odds
            elif key == "log_odds" and isinstance(value, (int, float, str)):
                if isinstance(value, str):
                    additions["odds"] = float("inf") if value == "Inf" else 0.0
                elif math.isfinite(value):
                    additions["odds"] = round(math.exp(value), 4)
                else:
                    additions["odds"] = float("inf") if value > 0 else 0.0

            # Handle *_log_odds -> *_odds
            elif key.endswith("_log_odds") and isinstance(value, (int, float, str)):
                odds_key = key.replace("_log_odds", "_odds")
                if isinstance(value, str):
                    additions[odds_key] = float("inf") if value == "Inf" else 0.0
                elif math.isfinite(value):
                    additions[odds_key] = round(math.exp(value), 4)
                else:
                    additions[odds_key] = float("inf") if value > 0 else 0.0

        d.update(additions)
        return d
