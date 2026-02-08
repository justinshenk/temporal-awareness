"""Binary choice runner for preference experiments.

Provides high-level API for binary choice probability computation.
Wraps ModelRunner for low-level model operations.
"""

from __future__ import annotations

from typing import Any

import torch
from math import prod

from .model_runner import ModelRunner


class BinaryChoiceRunner:
    """High-level runner for binary choice preference experiments."""

    def __init__(self, model_runner: ModelRunner):
        self._runner = model_runner

    @property
    def tokenizer(self):
        return self._runner.tokenizer

    @property
    def device(self):
        return self._runner.device

    def get_label_probs(
        self,
        prompt: str | list[str],
        choice_prefix: str,
        labels: tuple[str, str],
        past_kv_cache: Any = None,
    ) -> tuple | list[tuple]:
        """Get probabilities for two label options."""
        if isinstance(prompt, str):
            return self._get_label_probs_single(
                prompt, choice_prefix, labels, past_kv_cache
            )
        return [
            self._get_label_probs_single(p, choice_prefix, labels, past_kv_cache)
            for p in prompt
        ]

    def get_label_next_token_prob_sequences(
        self,
        prompt: str,
        choice_prefix: str,
        labels: tuple[str, str],
        past_kv_cache: Any = None,
    ) -> tuple[list[float], list[float]]:
        """Get probabilities for two label options.

        Computes P(label | prefix) = product of P(tok_i | prefix + tok_0..i-1)
        for all tokens from the divergence point onwards.
        """

        # Tokenize labels IN CONTEXT to get correct token IDs
        # (tokenizers merge spaces with following chars contextually)
        ids1, ids2 = self._get_full_choice_context_token_ids(
            prompt, choice_prefix, labels
        )

        # Find first position where the two sequences diverge
        diverge_pos = self._get_diverge_pos(ids1, ids2)

        seq1 = self._get_next_token_prob_sequence(ids1, diverge_pos, past_kv_cache)
        seq2 = self._get_next_token_prob_sequence(ids2, diverge_pos, past_kv_cache)

        return (seq1, seq2)

    def get_canonical_response_texts(
        self, choice_prefix: str, labels: tuple[str, str]
    ) -> tuple[str, str]:
        return (choice_prefix + labels[0], choice_prefix + labels[1])

    # Where is this function used? It is sus.
    def get_divergent_token_ids(self, label1: str, label2: str) -> tuple[int, int]:
        """Get first divergent token IDs for two labels.

        For multi-token labels like OPTION_ONE/OPTION_TWO, finds where they diverge
        and returns the token IDs at that position.

        Args:
            label1: First label string
            label2: Second label string

        Returns:
            Tuple of (token_id_1, token_id_2) at the first divergent position
        """
        tokenizer = self.tokenizer
        ids1 = tokenizer.encode(label1, add_special_tokens=False)
        ids2 = tokenizer.encode(label2, add_special_tokens=False)

        diverge_pos = self._get_diverge_pos(ids1, ids2)
        tok1 = ids1[diverge_pos] if diverge_pos < len(ids1) else ids1[-1]
        tok2 = ids2[diverge_pos] if diverge_pos < len(ids2) else ids2[-1]
        return tok1, tok2

    ##################
    #### Internal ####
    ##################

    def _get_base_text(self, prompt: str, choice_prefix: str):
        return self._runner._apply_chat_template(prompt) + choice_prefix

    def _get_full_choice_context_token_ids(
        self, prompt: str, choice_prefix: str, labels: tuple[str, str]
    ) -> tuple[list[int], list[int]]:
        label1, label2 = labels
        base_text = self._get_base_text(prompt, choice_prefix)
        ids1 = self.tokenizer.encode(base_text + label1, add_special_tokens=False)
        ids2 = self.tokenizer.encode(base_text + label2, add_special_tokens=False)
        return (ids1, ids2)

    def _get_diverge_pos(self, ids1: list[int], ids2: list[int]):
        diverge_pos = 0
        for i in range(min(len(ids1), len(ids2))):
            if ids1[i] != ids2[i]:
                diverge_pos = i
                break
        else:
            diverge_pos = min(len(ids1), len(ids2))
        return diverge_pos

    def _get_label_probs_single(
        self,
        prompt: str,
        choice_prefix: str,
        labels: tuple[str, str],
        past_kv_cache: Any = None,
    ) -> tuple:
        """Get probabilities for two label options.

        Computes P(label | prefix) = product of P(tok_i | prefix + tok_0..i-1)
        for all tokens from the divergence point onwards.
        """
        seq1, seq2 = self.get_label_next_token_prob_sequences(
            prompt, choice_prefix, labels, past_kv_cache
        )
        seq_prob1 = prod(seq1)
        seq_prob2 = prod(seq2)
        return (seq_prob1, seq_prob2)

    def _get_next_token_prob_sequence(
        self,
        token_ids: list[int],
        start_pos: int,
        past_kv_cache: Any = None,
    ) -> list[float]:
        """Get sequence of next-token probabilities via single forward pass.

        For token_ids = [t0, t1, t2, t3] and start_pos = 1:
        Returns [P(t1|t0), P(t2|t0,t1), P(t3|t0,t1,t2)]

        Args:
            token_ids: Full token ID sequence
            start_pos: Position from which to compute probabilities
            past_kv_cache: Unused, kept for API compatibility

        Returns:
            List of probabilities for each token after start_pos
        """
        # Single forward pass with all tokens
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self._runner._backend.forward(input_ids)  # [1, seq_len, vocab_size]

        # logits[0, i, :] predicts token at position i+1
        # For position i in [start_pos-1, len-2], get P(token_ids[i+1])
        continuation_probs = []
        for i in range(start_pos - 1, len(token_ids) - 1):
            probs = torch.softmax(logits[0, i, :], dim=-1)
            next_tok = token_ids[i + 1]
            continuation_probs.append(probs[next_tok].item())

        return continuation_probs
