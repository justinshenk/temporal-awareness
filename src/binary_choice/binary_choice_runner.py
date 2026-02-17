"""Binary choice runner for preference experiments.

Extends ModelRunner with specialized binary choice methods.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union


from ..inference.model_runner import ModelRunner
from ..inference.interventions import Intervention
from ..inference import GeneratedTrajectory
from .choice_utils import encode_into_trajectory_ids
from ..common.choice import LabeledSimpleBinaryChoice, GroupedBinaryChoice
from ..common.token_tree import TokenTree
from ..common.analysis.trajectory_branch_node_analysis import analyze_token_tree
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

    @profile("run_binary_choice")
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

        prompt = self.apply_chat_template(prompt)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        # Auto-prepend skip thinking prefix for reasoning models
        effective_prefix = self.skip_thinking_prefix + choice_prefix

        response_text_a = effective_prefix + labels[0]
        response_text_b = effective_prefix + labels[1]
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
            trunk=prompt_ids,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Batch API
    # ══════════════════════════════════════════════════════════════════════

    @profile("batch_choose")
    def batch_choose(
        self,
        prompts: Sequence[str],
        choice_prefixes: Sequence[str],
        labels: Sequence[tuple[str, str]],
    ) -> list[LabeledSimpleBinaryChoice]:
        """Run binary choice experiments for a batch of prompts with per-prompt prefixes/labels.

        Args:
            prompts:         Sequence of N task / question texts.
            choice_prefixes: Sequence of N response prefixes (one per prompt).
            labels:          Sequence of N label pairs (one per prompt).

        Returns:
            List of N LabeledSimpleBinaryChoice results.

        Note: With per-prompt labels, this requires 2*N forward passes (not 2).
        For uniform prefixes/labels, use batch_choose_uniform() for efficiency.
        """
        n = len(prompts)
        if len(choice_prefixes) != n:
            raise ValueError(
                f"choice_prefixes length ({len(choice_prefixes)}) must match "
                f"prompts length ({n})"
            )
        if len(labels) != n:
            raise ValueError(
                f"labels length ({len(labels)}) must match prompts length ({n})"
            )

        # Auto-prepend skip thinking prefix for reasoning models
        skip_prefix = self.skip_thinking_prefix

        # Build per-prompt response texts
        response_texts_a = [
            skip_prefix + choice_prefixes[i] + labels[i][0] for i in range(n)
        ]
        response_texts_b = [
            skip_prefix + choice_prefixes[i] + labels[i][1] for i in range(n)
        ]

        # Encode each prompt with its own response
        batch_ids_a = [
            encode_into_trajectory_ids(self, prompts[i], response_texts_a[i])
            for i in range(n)
        ]
        batch_ids_b = [
            encode_into_trajectory_ids(self, prompts[i], response_texts_b[i])
            for i in range(n)
        ]

        # ── Batched inference ────────────────────────────────────────────

        trajs_a = self.get_prob_trajectories_for_batch(batch_ids_a)
        trajs_b = self.get_prob_trajectories_for_batch(batch_ids_b)

        # ── Compute trunk (prompt_ids) per prompt ────────────────────────

        prompt_ids_list = [
            self.tokenizer.encode(self.apply_chat_template(p), add_special_tokens=True)
            for p in prompts
        ]

        # ── Assemble results ─────────────────────────────────────────────

        results: list[LabeledSimpleBinaryChoice] = []
        for i in range(n):
            choice = LabeledSimpleBinaryChoice.from_trajectories(
                trajs_a[i],
                trajs_b[i],
                labels=labels[i],
                response_texts=(response_texts_a[i], response_texts_b[i]),
                trunk=prompt_ids_list[i],
            )
            results.append(choice)
        return results

    def batch_choose_uniform(
        self,
        prompts: Sequence[str],
        choice_prefix: str,
        labels: tuple[str, str],
    ) -> list[LabeledSimpleBinaryChoice]:
        """Convenience wrapper with uniform prefix/labels.

        This is more efficient when all prompts share the same prefix and labels,
        as the response encoding can be reused.

        Args:
            prompts:       Sequence of N task / question texts.
            choice_prefix: Shared response prefix for all prompts.
            labels:        Shared label pair for all prompts.

        Returns:
            List of N LabeledSimpleBinaryChoice results.
        """
        n = len(prompts)
        return self.batch_choose(
            prompts,
            [choice_prefix] * n,
            [labels] * n,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Multi-label API
    # ══════════════════════════════════════════════════════════════════════

    @profile("multilabel_choose")
    def multilabel_choose(
        self,
        prompt: str,
        choice_prefix: str,
        labels: Sequence[tuple[str, str]],
        *,
        with_cache: bool = False,
        intervention: Optional[Union[Intervention, list[Intervention]]] = None,
        names_filter: Optional[callable] = None,
        past_kv_cache: Any = None,
    ) -> GroupedBinaryChoice:
        """Run a multi-label choice experiment for a single prompt.

        Creates 2*N trajectories (N label pairs), with groups:
        - (0,1) for pair 0
        - (2,3) for pair 1
        - etc.

        Fork arms: [(0,1), (2,3), ...] to create forks between each pair.

        Args:
            prompt:         The task / question text.
            choice_prefix:  Shared response prefix, e.g. "I choose:"
            labels:         N candidate label pairs, e.g. [("a)", "b)"), ("[i]", "[ii]")]
            with_cache:     If True, capture activation caches.
            intervention:   Optional intervention(s) to apply.
            names_filter:   Hook filter for caching.
            past_kv_cache:  Optional pre-computed KV cache.

        Returns:
            GroupedBinaryChoice with a single tree containing all trajectories.
        """
        if not labels:
            raise ValueError("labels must contain at least one label pair")

        prompt = self.apply_chat_template(prompt)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        # Auto-prepend skip thinking prefix for reasoning models
        effective_prefix = self.skip_thinking_prefix + choice_prefix

        # Build all response texts and token IDs
        all_token_ids = []
        label_pairs_tuple = tuple(labels)

        for label_a, label_b in labels:
            response_text_a = effective_prefix + label_a
            response_text_b = effective_prefix + label_b
            token_ids_a = encode_into_trajectory_ids(self, prompt, response_text_a)
            token_ids_b = encode_into_trajectory_ids(self, prompt, response_text_b)
            all_token_ids.append(token_ids_a)
            all_token_ids.append(token_ids_b)

        # ── Inference ────────────────────────────────────────────────────

        if intervention or with_cache:
            # Non-batched path for interventions/caching
            trajs = []
            for token_ids in all_token_ids:
                if intervention and with_cache:
                    traj = self.generate_trajectory_with_intervention_and_cache(
                        token_ids, intervention, names_filter
                    )
                elif intervention:
                    traj = self.generate_trajectory_with_intervention(
                        token_ids, intervention, names_filter
                    )
                else:
                    traj = self.generate_trajectory_with_cache(
                        token_ids, names_filter, past_kv_cache
                    )
                trajs.append(traj)
        else:
            # Batched inference
            trajs = self.get_prob_trajectories_for_batch(all_token_ids)

        # ── Build tree structure ────────────────────────────────────────

        n_pairs = len(labels)

        # Groups: each pair gets two consecutive groups
        # Pair 0: trajs 0,1 -> groups 0,1
        # Pair 1: trajs 2,3 -> groups 2,3
        # etc.
        groups_per_traj = []
        for i in range(n_pairs):
            groups_per_traj.append((2 * i,))  # traj 2*i in group 2*i
            groups_per_traj.append((2 * i + 1,))  # traj 2*i+1 in group 2*i+1

        # Fork arms: each pair creates a fork between its two groups
        fork_arms = [(2 * i, 2 * i + 1) for i in range(n_pairs)]

        # Build tree with all trajectories
        tree = TokenTree.from_trajectories(
            trajs,
            groups_per_traj=groups_per_traj,
            fork_arms=fork_arms,
            trunk=prompt_ids,
        )
        analyze_token_tree(tree)

        # ── Assemble result ──────────────────────────────────────────────

        return GroupedBinaryChoice(
            tree=tree,
            label_pairs=label_pairs_tuple,
        )

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
