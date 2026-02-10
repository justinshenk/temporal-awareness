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
    #            API           #
    ############################

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

    # High-level API

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        intervention: Optional[Intervention] = None,
        past_kv_cache: Any = None,
    ) -> str:
        """Generate text, optionally with intervention."""
        formatted = self._apply_chat_template(prompt)
        return self._backend.generate(
            formatted, max_new_tokens, temperature, intervention, past_kv_cache
        )

    # Optimized inference APIs (for classes like BinaryChoiceRunner)

    def get_prob_trajectory(
        self,
        token_ids: list[int],  # [seq_len]
        start_pos: int = 0,
    ) -> list[float]:
        """Get sequence of next-token probabilities via single forward pass.

        For token_ids = [t0, t1, t2, t3] and start_pos = 1:
        Returns [P(t1|t0), P(t2|t0,t1), P(t3|t0,t1,t2)]

        Uses vectorized softmax + gather instead of a per-position loop.

        Args:
            token_ids: Full token ID sequence
            start_pos: Position from which to compute probabilities

        Returns:
            List of probabilities for each token after start_pos
        """
        return self.get_prob_trajectories_for_batch([token_ids], start_pos)[0]

    def get_prob_trajectories_for_batch(
        self,
        token_ids_batch: list[list[int]],  # [batch][seq_len_i]
        start_pos: int = 0,
    ) -> list[list[float]]:
        # TODO(claude): How do I fix this? Do MINIMAL LOCALIZED fix
        input_ids_batch = torch.tensor(
            token_ids_batch, device=self.device
        )  # [batch, max(seq_len_i), vocab_size]

        with torch.inference_mode():
            logits_batch = self._backend.forward(
                input_ids_batch
            )  # [batch, seq_len, vocab_size]
        return self._calculate_prob_trajectories_for_batch(
            token_ids_batch, logits_batch, start_pos
        )

    # Basic Interpretability APIs

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

    def run_with_intervention(
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
            Logits tensor of shape [1, seq_len, vocab_size]
        """
        formatted = self._apply_chat_template(prompt)
        # Note: intervention.positions are relative to tokenized input
        # If prepend_bos=True, position 0 is BOS, position 1 is first real token
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        # Normalize to list for uniform handling
        interventions = (
            [intervention] if isinstance(intervention, Intervention) else intervention
        )
        # TODO(claude): Update backend to match model runner naming
        return self._backend.forward_with_intervention(input_ids, interventions)

    # Complex Interpretability APIs

    def run_with_intervention_and_cache(
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
        # TODO(claude): Update backend to match model runner naming
        return self._backend.forward_with_intervention_and_cache(
            input_ids, interventions, names_filter
        )

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

    # Complex Interpretability APIs (for classes like BinaryChoiceRunner)

    def get_prob_trajectory_with_intervention(
        self,
        token_ids: list[int],  # [seq_len]
        start_pos: int = 0,
        intervention: Union[Intervention, list[Intervention]] | None = None,
        names_filter: Optional[callable] = None,
    ) -> list[float]:
        input_ids = torch.tensor(
            [token_ids], device=self.device
        )  # [1, seq_len, vocab_size]

        with torch.inference_mode():
            logits_batch = self._backend.forward_with_intervention(
                input_ids, interventions, names_filter
            )  # [1, seq_len, vocab_size]

        logits = logits_batch[0]  # [seq_len, vocab_size]

        return self._calculate_single_prob_trajectory(
            input_ids, logits, start_pos
        )  # [seq_len]

    def get_prob_trajectory_with_cache(
        self,
        token_ids: list[int],  # [seq_len]
        start_pos: int = 0,
        past_kv_cache: Any = None,
    ) -> list[float]:
        input_ids = torch.tensor(
            [token_ids], device=self.device
        )  # [1, seq_len, vocab_size]

        with torch.inference_mode():
            logits_batch, internals_cache = self._backend.run_with_cache(
                input_ids, names_filter, past_kv_cache
            )  # [1, seq_len, vocab_size]

        logits = logits_batch[0]  # [seq_len, vocab_size]

        return self._calculate_single_prob_trajectory(
            input_ids, logits, start_pos
        ), internals_cache

    def get_prob_trajectory_with_intervention_and_cache(
        self,
        token_ids: list[int],  # [seq_len]
        start_pos: int = 0,
        intervention: Union[Intervention, list[Intervention]] | None = None,
        names_filter: Optional[callable] = None,
    ) -> tuple[torch.Tensor, dict]:
        input_ids = torch.tensor(
            [token_ids], device=self.device
        )  # [1, seq_len, vocab_size]

        with torch.inference_mode():
            logits_batch, internals_cache = (
                self._backend.forward_with_intervention_and_cache(
                    input_ids, interventions, names_filter
                )
            )  # [1, seq_len, vocab_size]

        logits = logits_batch[0]  # [seq_len, vocab_size]

        return self._calculate_single_prob_trajectory(
            input_ids, logits, start_pos
        ), internals_cache

    # KV Cache APIs

    def init_kv_cache(self):
        return self._backend.init_kv_cache()

    def generate_from_kv_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate using prefill logits and frozen kv_cache."""
        # TODO(claude): Update backend to match model runner naming
        return self._backend.generate_from_cache(
            prefill_logits, frozen_kv_cache, max_new_tokens, temperature
        )

    def get_all_names_for_internals(self):
        n_layers = self.n_layers
        components = ["resid_pre", "resid_post", "attn_out", "mlp_out"]
        return [
            f"blocks.{layer}.hook_{comp}"
            for layer in range(self.n_layers)
            for comp in components
        ]

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

    def _calculate_single_prob_trajectory(
        self,
        token_ids: list[int],  # [seq_len]
        logits: nn.Tensor,  # [seq_len, vocab_size]
        start_pos: int = 0,
    ) -> list[float]:
        """Get sequence of next-token probabilities via single forward pass.

        For token_ids = [t0, t1, t2, t3] and start_pos = 1:
        Returns [P(t1|t0), P(t2|t0,t1), P(t3|t0,t1,t2)]

        Uses vectorized softmax + gather instead of a per-position loop.
        """

        # logits[i, :] predicts token at position i+1.
        # For start_pos >= 1, we need logits[start_pos-1 : len-1].
        # For start_pos == 0, the first token has no conditioning context,
        # so P(t0) := 1.0; remaining probs come from logits[0 : len-1].
        pred_start = max(start_pos - 1, 0)  # n_preds < seq_len
        pred_logits = logits[pred_start : len(token_ids) - 1, :]  # [n_preds, vocab]

        # The ground-truth tokens we want probabilities for
        target_ids = torch.tensor(
            token_ids[start_pos:], device=self.device
        )  # [n_preds]

        # Softmax over vocab, then pick out the target token at each position
        probs = torch.softmax(pred_logits, dim=-1)  # [n_preds, vocab]
        target_probs = probs[torch.arange(len(target_ids)), target_ids]  # [n_preds]

        # When start_pos == 0, the first token has no prior context so its
        # probability is defined as 1.0 and inserted manually below.
        # The remaining probabilities are computed from logits as usual.
        result = target_probs.tolist()  # len(result) == n_preds
        if start_pos == 0:
            result.insert(0, 1.0)  # len(result) == n_preds + 1 == seq_len
        return result  # len(result) == len(target_ids[start_pos:])

    def _calculate_prob_trajectories_for_batch(
        self,
        token_ids_batch: list[list[int]],  # [batch][seq_len_i]
        logits_batch: torch.Tensor,  # [batch, max_seq_len, vocab_size]
        start_pos: int = 0,
    ) -> list[list[float]]:
        """Get sequence of next-token probabilities for a batch of sequences.
        For token_ids = [t0, t1, t2, t3] and start_pos = 1:
        Returns [P(t1|t0), P(t2|t0,t1), P(t3|t0,t1,t2)] per sequence.
        Uses vectorized softmax + gather instead of a per-position loop.
        Sequences may vary in length; padding beyond each sequence's
        actual length in logits_batch is ignored.
        """
        prob_trajectories = []
        for i, token_ids in enumerate(token_ids_batch):
            logits = logits_batch[i]  # [max_seq_len, vocab_size]
            traj = self._calculate_single_prob_trajectory(token_ids, logits, start_pos)
            prob_trajectories.append(traj)

        return prob_trajectories  # [batch][varying lengths]
