"""Model runner for inference with intervention support."""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

from ..common.device_utils import get_device
from ..common.profiler import profile
from .interventions import Intervention
from .backends import (
    ModelBackend,
    TransformerLensBackend,
    NNsightBackend,
    PyveneBackend,
    HuggingFaceBackend,
    MLXBackend,
    get_recommended_backend_inference,
)
from .generated_trajectory import (
    GeneratedTrajectory,
    calculate_trajectories_for_batch,
)


class ModelRunner:
    """Model runner for inference with intervention support."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        backend: ModelBackend = get_recommended_backend_inference(),
    ):
        self.model_name = model_name

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
        elif backend == ModelBackend.HUGGINGFACE:
            self._init_huggingface()
        elif backend == ModelBackend.MLX:
            self._init_mlx()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Detect chat model after tokenizer is available
        self._is_chat_model = self._detect_chat_model(model_name)

        print(f"Model loaded: {backend} {model_name} (chat={self._is_chat_model})")
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
        return self._backend.tokenize(text, prepend_bos=prepend_bos)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self._backend.decode(token_ids)

    # High-level API

    @profile
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        intervention: Optional[Intervention] = None,
        past_kv_cache: Any = None,
        prefilling: str = "",
    ) -> str:
        """Generate text, optionally with intervention."""
        formatted = self.apply_chat_template(prompt) + prefilling
        return self._backend.generate(
            formatted, max_new_tokens, temperature, intervention, past_kv_cache
        )

    # Optimized inference APIs (for classes like BinaryChoiceRunner)

    @profile
    def generate_trajectory(
        self,
        token_ids: list[int],
    ) -> GeneratedTrajectory:
        """Get sequence of next-token probabilities via single forward pass.

        For token_ids = [t0, t1, t2, t3]:
        Returns trajectory with logprobs [P(t1|t0), P(t2|t0,t1), P(t3|t0,t1,t2)]

        Args:
            token_ids: Full token ID sequence

        Returns:
            GeneratedTrajectory with per-token logprobs/logits and full vocab tensor
        """
        return self.get_prob_trajectories_for_batch([token_ids])[0]

    @profile
    def get_prob_trajectories_for_batch(
        self,
        token_ids_batch: list[list[int]],
    ) -> list[GeneratedTrajectory]:
        max_len = max(len(ids) for ids in token_ids_batch)
        pad_token = self.tokenizer.pad_token_id or 0
        padded = [ids + [pad_token] * (max_len - len(ids)) for ids in token_ids_batch]
        input_ids_batch = torch.tensor(padded, device=self.device)

        ctx = (
            torch.inference_mode()
            if self._backend.supports_inference_mode
            else torch.no_grad()
        )
        with ctx:
            logits_batch = self._backend.forward(
                input_ids_batch
            )  # [batch, seq_len, vocab_size]
        return calculate_trajectories_for_batch(
            token_ids_batch, logits_batch, self.device
        )

    # Basic Interpretability APIs

    @profile
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
            names_filter: Function to filter which hooks to cache
            past_kv_cache: Optional past key-value cache for continuation
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Tuple of (logits, cache) where cache maps hook names to activation tensors
        """
        formatted = self.apply_chat_template(prompt)
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        return self._backend.run_with_cache(input_ids, names_filter, past_kv_cache)

    @profile
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
        formatted = self.apply_chat_template(prompt)
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        interventions = (
            [intervention] if isinstance(intervention, Intervention) else intervention
        )
        return self._backend.run_with_intervention(input_ids, interventions)

    # Complex Interpretability APIs

    @profile
    def run_with_intervention_and_cache(
        self,
        prompt: str,
        intervention: Union[Intervention, list[Intervention]],
        names_filter: Optional[callable] = None,
        prepend_bos: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with intervention AND capture activations with gradients."""
        formatted = self.apply_chat_template(prompt)
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        interventions = (
            [intervention] if isinstance(intervention, Intervention) else intervention
        )
        return self._backend.run_with_intervention_and_cache(
            input_ids, interventions, names_filter
        )

    @profile
    def run_with_cache_and_grad(
        self,
        prompt: str,
        names_filter: Optional[callable] = None,
        prepend_bos: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass with gradients enabled for attribution patching."""
        formatted = self.apply_chat_template(prompt)
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)
        return self._backend.run_with_cache_and_grad(input_ids, names_filter)

    # Complex Interpretability APIs (for classes like BinaryChoiceRunner)

    @profile
    def generate_trajectory_with_intervention(
        self,
        token_ids: list[int],
        intervention: Union[Intervention, list[Intervention]] | None = None,
        names_filter: Optional[callable] = None,
    ) -> GeneratedTrajectory:
        input_ids = torch.tensor([token_ids], device=self.device)

        interventions = (
            [intervention] if isinstance(intervention, Intervention) else intervention
        )

        ctx = (
            torch.inference_mode()
            if self._backend.supports_inference_mode
            else torch.no_grad()
        )
        with ctx:
            logits_batch = self._backend.run_with_intervention(
                input_ids, interventions
            )  # [1, seq_len, vocab_size]

        logits = logits_batch[0]  # [seq_len, vocab_size]
        return GeneratedTrajectory.from_inference(token_ids, logits, self.device)

    @profile
    def generate_trajectory_with_cache(
        self,
        token_ids: list[int],
        names_filter: Optional[callable] = None,
        past_kv_cache: Any = None,
    ) -> GeneratedTrajectory:
        input_ids = torch.tensor([token_ids], device=self.device)

        ctx = (
            torch.inference_mode()
            if self._backend.supports_inference_mode
            else torch.no_grad()
        )
        with ctx:
            logits_batch, internals_cache = self._backend.run_with_cache(
                input_ids, names_filter, past_kv_cache
            )  # [1, seq_len, vocab_size]

        logits = logits_batch[0]  # [seq_len, vocab_size]
        return GeneratedTrajectory.from_inference(
            token_ids, logits, self.device, internals=internals_cache
        )

    @profile
    def generate_trajectory_with_intervention_and_cache(
        self,
        token_ids: list[int],
        intervention: Union[Intervention, list[Intervention]] | None = None,
        names_filter: Optional[callable] = None,
    ) -> GeneratedTrajectory:
        input_ids = torch.tensor([token_ids], device=self.device)

        interventions = (
            [intervention] if isinstance(intervention, Intervention) else intervention
        )

        ctx = (
            torch.inference_mode()
            if self._backend.supports_inference_mode
            else torch.no_grad()
        )
        with ctx:
            logits_batch, internals_cache = (
                self._backend.run_with_intervention_and_cache(
                    input_ids, interventions, names_filter
                )
            )  # [1, seq_len, vocab_size]

        logits = logits_batch[0]  # [seq_len, vocab_size]
        return GeneratedTrajectory.from_inference(
            token_ids, logits, self.device, internals=internals_cache
        )

    # Basic Forward API

    @profile
    def forward(
        self,
        prompt: str,
        prepend_bos: bool = False,
    ) -> torch.Tensor:
        """Run forward pass and return logits.

        Args:
            prompt: Input text
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Logits tensor of shape [1, seq_len, vocab_size]
        """
        formatted = self.apply_chat_template(prompt)
        input_ids = self.tokenize(formatted, prepend_bos=prepend_bos)

        ctx = (
            torch.inference_mode()
            if self._backend.supports_inference_mode
            else torch.no_grad()
        )
        with ctx:
            return self._backend.forward(input_ids)

    # KV Cache APIs
    def init_kv_cache(self):
        return self._backend.init_kv_cache()

    @profile
    def generate_from_kv_cache(
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

    def get_all_names_for_internals(self) -> list:
        n_layers = self.n_layers
        components = ["resid_pre", "resid_post", "attn_out", "mlp_out"]
        return [
            f"blocks.{layer}.hook_{comp}"
            for layer in range(n_layers)
            for comp in components
        ]

    def apply_chat_template(self, prompt: str) -> str:
        if not self._is_chat_model:
            return prompt
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        print("apply_chat_template: tokenizer does not have apply_chat_template")
        return prompt

    ##################
    #### Internal ####
    ##################

    def _init_transformerlens(self) -> None:
        from transformer_lens import HookedTransformer

        print(f"Loading {self.model_name} on {self.device} (TransformerLens)...")
        # Use from_pretrained_no_processing to avoid weight centering/folding
        # that changes raw logit values (though softmax output is the same)
        self._model = HookedTransformer.from_pretrained_no_processing(
            self.model_name, device=self.device, dtype=self.dtype
        )
        self._model.eval()
        self._backend = TransformerLensBackend(self)

    def _init_nnsight(self) -> None:
        from nnsight import LanguageModel

        print(f"Loading {self.model_name} on {self.device} (nnsight)...")
        self._model = LanguageModel(
            self.model_name, device_map=self.device, dtype=self.dtype
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

    def _init_huggingface(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name} on {self.device} (HuggingFace)...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._backend = HuggingFaceBackend(self)

    def _init_mlx(self) -> None:
        from mlx_lm import load

        print(f"Loading {self.model_name} (MLX)...")
        self._model, self._tokenizer = load(self.model_name)
        self._backend = MLXBackend(self)

    def _detect_chat_model(self, model_name: str) -> bool:
        """Detect if model is a chat/instruct model.

        Detection strategy:
        1. Check if tokenizer has a chat_template (most reliable)
        2. Check for base model indicators in name (return False)
        3. Fall back to name heuristics for instruct indicators
        """
        # Primary method: check if tokenizer has a chat template
        tokenizer = self.tokenizer
        if tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
            if chat_template:
                # chat_template can be a string or dict (for multiple templates)
                if isinstance(chat_template, str) and chat_template.strip():
                    return True
                if isinstance(chat_template, dict) and chat_template:
                    return True

        # Secondary method: name-based heuristics
        if not model_name:
            model_name = self.model_name
        name = model_name.lower()

        # Explicit base model indicators (e.g., Qwen3-0.6B-Base)
        if any(x in name for x in ["-base", "_base"]):
            return False

        # Instruct/chat model indicators
        return any(x in name for x in ["instruct", "chat", "-it", "rlhf"])

    def _detect_reasoning_model(self) -> bool:
        """Detect if model supports thinking/reasoning mode.

        Detection strategy:
        1. Check if chat_template contains thinking-related tokens (most reliable)
        2. Fall back to name heuristics, excluding known non-reasoning variants
        """
        name = self.model_name.lower()

        # Explicit non-reasoning model indicators
        # Qwen3-*-Instruct-2507 variants are non-reasoning
        non_reasoning_indicators = ["-2507", "_2507"]
        if any(ind in name for ind in non_reasoning_indicators):
            return False

        # Primary method: check chat_template for thinking tokens
        tokenizer = self.tokenizer
        if tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
            if chat_template:
                template_str = (
                    chat_template
                    if isinstance(chat_template, str)
                    else str(chat_template)
                )
                # Check for thinking-related tokens in template
                thinking_indicators = [
                    "<think>",
                    "</think>",
                    "enable_thinking",
                    "<|thinking|>",
                    "<reasoning>",
                ]
                if any(indicator in template_str for indicator in thinking_indicators):
                    return True

        # Name-based heuristics for known reasoning models
        reasoning_models = ["qwen3", "deepseek-r1", "o1", "o3"]
        return any(model in name for model in reasoning_models)

    @property
    def is_reasoning_model(self) -> bool:
        """Whether this model supports thinking/reasoning mode."""
        if not hasattr(self, "_is_reasoning_model"):
            self._is_reasoning_model = self._detect_reasoning_model()
        return self._is_reasoning_model

    @property
    def skip_thinking_prefix(self) -> str:
        """Prefix to skip thinking mode for reasoning models.

        Returns empty string for non-reasoning models.
        """
        if self.is_reasoning_model:
            return "<think>\n</think>\n\n"
        return ""
