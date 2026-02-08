"""
Model runner for inference with intervention support.

Supports TransformerLens, NNsight, and Pyvene backends.

IMPORTANT DESIGN PRINCIPLES (do not violate):
1. NEVER use TransformerLens/NNsight/Pyvene backend APIs directly from outside this module
   - ALL code must use ModelRunner public API methods
2. All backends MUST implement identical behavior for all methods
   - Tests verify cross-backend consistency (see tests/test_model_runner.py)
3. NEVER use magic numbers - use named constants or parameters
4. Tests should NEVER be skipped (xfail is acceptable for known issues)
5. When adding features, add corresponding tests for ALL backends

Example:
    runner = ModelRunner("Qwen/Qwen2.5-7B-Instruct")
    output = runner.generate("What is 2+2?")

    # With intervention
    from src.models.interventions import steering
    intervention = steering(layer=26, direction=probe.direction, strength=100.0)
    output = runner.generate("What is 2+2?", intervention=intervention)
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from math import prod
from ..common.device import get_device
from .interventions import Intervention
from .backends import (
    ModelBackend,
    TransformerLensBackend,
    NNsightBackend,
    PyveneBackend,
)


class ModelRunner:
    """Model runner for inference with intervention support."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        backend: ModelBackend = ModelBackend.TRANSFORMERLENS,
    ):
        self.model_name = model_name
        self._is_chat_model = self._detect_chat_model(model_name)

        if device is None:
            device = get_device()
        self.device = device
        if dtype is None:
            dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
        self.dtype = dtype

        # IMPORTANT: self._model should never be used outside ModelRunner + Children + Backends
        self._model = None

        # IMPORTANT: self._backend should never be used outside ModelRunner + Children
        self._backend = backend
        if backend == ModelBackend.TRANSFORMERLENS:
            self._init_transformerlens()
        elif backend == ModelBackend.NNSIGHT:
            self._init_nnsight()
        elif backend == ModelBackend.PYVENE:
            self._init_pyvene()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        print(f"Model loaded: {model_name} (chat={self._is_chat_model})")
        print(f"  n_layers={self.n_layers}, d_model={self.d_model}\n")

    ############################
    ###### Low-Level API #######
    ##########################

    @property
    def tokenizer(self):
        return self._backend.get_tokenizer()

    @property
    def n_layers(self) -> int:
        return self._backend.get_n_layers()

    @property
    def d_model(self) -> int:
        return self._backend.get_d_model()

    def tokenize(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        """Tokenize text into tensor of token IDs.

        Args:
            text: Input text to tokenize
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Token IDs tensor of shape [1, seq_len]
        """
        # prepend_bos=False by default for consistent behavior across backends:
        # - TransformerLens: uses to_tokens(prepend_bos=X) directly
        # - NNsight/Pyvene: use HF tokenizer, only prepend if bos_token_id exists
        # Some models (e.g. Qwen) have bos_token_id=None, so prepend_bos=True
        # may behave differently across backends for those models.
        return self._backend.tokenize(text, prepend_bos=prepend_bos)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self._backend.decode(token_ids)

    def generate(
        self,
        prompt: str | list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        intervention: Optional[Intervention] = None,
        past_kv_cache: Any = None,
    ) -> str | list[str]:
        """Generate text, optionally with intervention."""
        if isinstance(prompt, str):
            return self._generate_single(
                prompt, max_new_tokens, temperature, intervention, past_kv_cache
            )

        return [
            self._generate_single(
                p, max_new_tokens, temperature, intervention, past_kv_cache
            )
            for p in prompt
        ]

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate using prefill logits and frozen kv_cache."""
        return self._backend.generate_from_cache(
            prefill_logits, frozen_kv_cache, max_new_tokens, temperature
        )

    def run_with_cache(
        self,
        prompt: str,
        names_filter: Optional[callable] = None,
        past_kv_cache: Any = None,
        prepend_bos: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass and return activation cache.

        Args:
            prompt: Input text
            names_filter: Function to filter which hooks to cache (e.g. lambda n: 'resid' in n)
            past_kv_cache: Optional past key-value cache for continuation
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Tuple of (logits, cache) where cache maps hook names to activation tensors
        """
        formatted = self._apply_chat_template(prompt)
        # prepend_bos ensures consistent seq_len across backends for activation comparison
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        return self._backend.run_with_cache(input_ids, names_filter, past_kv_cache)

    def run_with_cache_and_grad(
        self,
        prompt: str,
        names_filter: Optional[callable] = None,
        prepend_bos: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass with gradients enabled for attribution patching.

        Args:
            prompt: Input text
            names_filter: Function to filter which hooks to cache
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Tuple of (logits, cache) where cache values have requires_grad=True
        """
        formatted = self._apply_chat_template(prompt)
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        # Unlike run_with_cache, does NOT use torch.no_grad() - gradients flow through
        return self._backend.run_with_cache_and_grad(input_ids, names_filter)

    def forward_with_intervention(
        self,
        prompt: str,
        intervention: Union[Intervention, list[Intervention]],
        prepend_bos: bool = False,
    ) -> torch.Tensor:
        """Run forward pass with intervention(s) applied.

        Args:
            prompt: Input text
            intervention: Single Intervention or list of Interventions to apply
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size]
        """
        formatted = self._apply_chat_template(prompt)
        # Note: intervention.positions are relative to tokenized input
        # If prepend_bos=True, position 0 is BOS, position 1 is first real token
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        # Normalize to list for uniform handling
        interventions = (
            [intervention] if isinstance(intervention, Intervention) else intervention
        )
        return self._backend.forward_with_intervention(input_ids, interventions)

    def forward_with_intervention_and_cache(
        self,
        prompt: str,
        intervention: Union[Intervention, list[Intervention]],
        names_filter: Optional[callable] = None,
        prepend_bos: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with intervention AND capture activations with gradients.

        Combines forward_with_intervention + run_with_cache_and_grad.
        Used for attribution patching where we need:
        1. Apply interventions (e.g., interpolation)
        2. Capture activations
        3. Enable gradients for backprop

        Args:
            prompt: Input text
            intervention: Single Intervention or list of Interventions to apply
            names_filter: Optional function to filter which hooks to cache
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            (logits, cache) where cache values have requires_grad=True
        """
        formatted = self._apply_chat_template(prompt)
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        interventions = (
            [intervention] if isinstance(intervention, Intervention) else intervention
        )
        return self._backend.forward_with_intervention_and_cache(
            input_ids, interventions, names_filter
        )

    def init_kv_cache(self):
        return self._backend.init_kv_cache()

    #######################
    #### High-level API ###
    #######################

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

    def _init_transformerlens(self) -> None:
        from transformer_lens import HookedTransformer

        print(f"Loading {self.model_name} on {self.device} (TransformerLens)...")
        self._model = HookedTransformer.from_pretrained(
            self.model_name, device=self.device, dtype=self.dtype
        )
        self._model.eval()
        self._backend = TransformerLensBackend(self)

    def _init_nnsight(self) -> None:
        from nnsight import LanguageModel

        print(f"Loading {self.model_name} on {self.device} (nnsight)...")
        self._model = LanguageModel(
            self.model_name, device_map=self.device, torch_dtype=self.dtype
        )
        self._backend = NNsightBackend(self)

    def _init_pyvene(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name} on {self.device} (pyvene)...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._backend = PyveneBackend(self)

    def _detect_chat_model(self, model_name: str | None) -> bool:
        if not model_name:
            model_name = self.model_name
        name = model_name.lower()
        return any(x in name for x in ["instruct", "chat", "-it", "rlhf"])

    def _apply_chat_template(self, prompt: str) -> str:
        if not self._is_chat_model:
            return prompt
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        formatted = self._apply_chat_template(prompt)
        return self._backend.generate(
            formatted, max_new_tokens, temperature, intervention, past_kv_cache
        )

    def _get_base_text(self, prompt: str, choice_prefix: str):
        return self._apply_chat_template(prompt) + choice_prefix

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

        Uses vectorized softmax + gather instead of a per-position loop.

        Args:
            token_ids: Full token ID sequence
            start_pos: Position from which to compute probabilities
            past_kv_cache: Unused, kept for API compatibility

        Returns:
            List of probabilities for each token after start_pos
        """
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self._backend.forward(input_ids)  # [1, seq_len, vocab_size]

        # logits[0, i, :] predicts token at position i+1
        # So to get P(t_{start_pos}), ..., P(t_{end}), we need
        # logits at positions [start_pos-1, ..., len-2]
        pred_logits = logits[
            0, start_pos - 1 : len(token_ids) - 1, :
        ]  # [n_preds, vocab]

        # The ground-truth tokens we want probabilities for
        target_ids = torch.tensor(
            token_ids[start_pos:], device=self.device
        )  # [n_preds]

        # Softmax over vocab, then pick out the target token at each position
        probs = torch.softmax(pred_logits, dim=-1)  # [n_preds, vocab]
        target_probs = probs[torch.arange(len(target_ids)), target_ids]  # [n_preds]

        return target_probs.tolist()
