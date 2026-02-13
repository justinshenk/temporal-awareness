"""Binary choice runner for preference experiments.

Extends ModelRunner with specialized binary choice methods.
"""

from __future__ import annotations

from typing import Any, Optional, Union


from ..inference.model_runner import ModelRunner
from ..inference.interventions import Intervention
from ..inference import GeneratedTrajectory
from .choice_utils import encode_into_trajectory_ids
from ..common.choice import LabeledSimpleBinaryChoice
from ..common.profiler import profile


class BinaryChoiceRunner(ModelRunner):
    """High-level runner for binary choice preference experiments.

    Inherits all ModelRunner functionality and adds methods that run two
    forced-continuation trajectories (one per label), build a
    TokenTree, and return a BinaryChoice.
    """

    # ══════════════════════════════════════════════════════════════════════
    #  Single-prompt API
    # ══════════════════════════════════════════════════════════════════════

    @profile("step_load_model")
    def choose(
        self,
        prompt: str,
        choice_prefix: str,
        labels: tuple[str, str],
        *,
        with_cache: bool = False,
        intervention: Optional[Union[Intervention, list[Intervention]]] = None,
        names_filter: Optional[callable] = None,
        past_kv_cache: Any = None,
    ) -> LabeledSimpleBinaryChoice:
        """Run a binary choice experiment for a single prompt.

        Args:
            prompt:         The task / question text.
            choice_prefix:  Shared response prefix, e.g. "I choose:"
            labels:         Two candidate labels, e.g. ("<a>", "<b>")
            with_cache:     If True, capture activation caches.
            intervention:   Optional intervention(s) to apply.
            names_filter:   Hook filter for caching.
            past_kv_cache:  Optional pre-computed KV cache.

        Returns:
            LabeledSimpleBinaryChoice with the tree, decision, and metadata.
        """
        response_text_a = choice_prefix + labels[0]
        response_text_b = choice_prefix + labels[1]

        token_ids_a = encode_into_trajectory_ids(self, prompt, response_text_a)
        token_ids_b = encode_into_trajectory_ids(self, prompt, response_text_b)

        # ── Inference ────────────────────────────────────────────────────

        traj_a, traj_b = self._run_pair(
            token_ids_a,
            token_ids_b,
            intervention=intervention,
            with_cache=with_cache,
            names_filter=names_filter,
            past_kv_cache=past_kv_cache,
        )

        # ── Assemble result ──────────────────────────────────────────────

        return LabeledSimpleBinaryChoice.from_trajectories(
            traj_a,
            traj_b,
            labels=labels,
            response_texts=(response_text_a, response_text_b),
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Batch API
    # ══════════════════════════════════════════════════════════════════════

    @profile("batch_choose")
    def batch_choose(
        self,
        prompts: list[str],
        choice_prefix: str,
        labels: tuple[str, str],
    ) -> list[LabeledSimpleBinaryChoice]:
        """Run binary choice experiments for a batch of prompts.

        Uses a single batched forward pass per label (2 passes total).

        Args:
            prompts:        List of N task / question texts.
            choice_prefix:  Shared response prefix.
            labels:         Two candidate labels.

        Returns:
            List of N LabeledSimpleBinaryChoice results.
        """
        response_text_a = choice_prefix + labels[0]
        response_text_b = choice_prefix + labels[1]

        batch_ids_a = [
            encode_into_trajectory_ids(self, p, response_text_a) for p in prompts
        ]
        batch_ids_b = [
            encode_into_trajectory_ids(self, p, response_text_b) for p in prompts
        ]

        # ── Batched inference (one forward pass per label) ───────────────

        trajs_a = self.get_prob_trajectories_for_batch(batch_ids_a)
        trajs_b = self.get_prob_trajectories_for_batch(batch_ids_b)

        # ── Assemble results ─────────────────────────────────────────────

        results: list[LabeledSimpleBinaryChoice] = []
        for i in range(len(prompts)):
            choice = LabeledSimpleBinaryChoice.from_trajectories(
                trajs_a[i],
                trajs_b[i],
                labels=labels,
                response_texts=(response_text_a, response_text_b),
                internals=None,
            )
            results.append(choice)
        return results

    # ══════════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ══════════════════════════════════════════════════════════════════════

    @profile("_run_pair")
    def _run_pair(
        self,
        token_ids_a: list[int],
        token_ids_b: list[int],
        *,
        intervention: Optional[Union[Intervention, list[Intervention]]] = None,
        with_cache: bool = False,
        names_filter: Optional[callable] = None,
        past_kv_cache: Any = None,
    ) -> tuple[GeneratedTrajectory, GeneratedTrajectory]:
        """Run two trajectories through the appropriate ModelRunner method.

        Returns (traj_a, traj_b). If with_cache=True, returns
        GeneratedTrajectory instances with internals attached.
        """
        if intervention and with_cache:
            traj_a = self.generate_trajectory_with_intervention_and_cache(
                token_ids_a, intervention, names_filter
            )
            traj_b = self.generate_trajectory_with_intervention_and_cache(
                token_ids_b, intervention, names_filter
            )
            return traj_a, traj_b

        if intervention:
            traj_a = self.generate_trajectory_with_intervention(
                token_ids_a, intervention, names_filter
            )
            traj_b = self.generate_trajectory_with_intervention(
                token_ids_b, intervention, names_filter
            )
            return traj_a, traj_b

        if with_cache:
            traj_a = self.generate_trajectory_with_cache(
                token_ids_a, names_filter, past_kv_cache
            )
            traj_b = self.generate_trajectory_with_cache(
                token_ids_b, names_filter, past_kv_cache
            )
            return traj_a, traj_b

        # Default: plain forward pass, batched for efficiency
        trajs = self.get_prob_trajectories_for_batch([token_ids_a, token_ids_b])
        return trajs[0], trajs[1]
