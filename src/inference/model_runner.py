"""Model runner for inference with intervention support."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..common.device_utils import get_device
from ..common.profiler import profile
from .interventions import Intervention, Interventions
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
    def _tokenizer(self):
        return self._backend.get_tokenizer()

    @property
    def bos_token_id(self) -> int | None:
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self._tokenizer.pad_token_id

    @property
    def bos_token(self) -> str | None:
        return self._tokenizer.bos_token

    @property
    def eos_token(self) -> str | None:
        return self._tokenizer.eos_token

    @property
    def n_layers(self) -> int:
        return self._backend.get_n_layers()

    @property
    def d_model(self) -> int:
        return self._backend.get_d_model()

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        """Encode text into tensor of token IDs.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add special tokens (default True)
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Token IDs tensor of shape [1, seq_len]
        """
        return self._backend.encode(
            text, add_special_tokens=add_special_tokens, prepend_bos=prepend_bos
        )

    def encode_ids(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> list[int]:
        """Encode text into list of token IDs.

        Convenience method that returns a list instead of tensor.
        """
        tensor = self.encode(
            text, add_special_tokens=add_special_tokens, prepend_bos=prepend_bos
        )
        # flatten() ensures we always get a 1D tensor, tolist() then returns a list
        return tensor.flatten().tolist()

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode tensor of token IDs to string."""
        return self._backend.decode(token_ids)

    def decode_ids(self, token_ids: list[int]) -> str:
        """Decode list of token IDs to string.

        Convenience method that accepts a list instead of tensor.
        """
        return self._backend.decode(torch.tensor(token_ids))

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

    @profile
    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GeneratedTrajectory:
        """Generate text autoregressively and return trajectory with logprobs.

        Uses backend's optimized generation with KV caching for maximum speed.

        Args:
            token_ids: Initial token IDs to start generation from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            GeneratedTrajectory containing all tokens (input + generated) with logprobs
        """
        all_token_ids, all_logprobs = self._backend.generate_trajectory(
            token_ids, max_new_tokens, temperature
        )
        return GeneratedTrajectory.from_logprobs(all_token_ids, all_logprobs)

    @profile
    def generate_trajectory_with_intervention(
        self,
        token_ids: list[int],
        intervention: Interventions,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GeneratedTrajectory:
        """Generate trajectory with intervention applied at each step.

        This is slower than generate_trajectory because interventions invalidate
        KV caching - each token requires a full forward pass through all positions.
        Use generate_trajectory when no intervention is needed.

        Args:
            token_ids: Initial token IDs to start generation from
            intervention: Intervention(s) to apply at each generation step
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            GeneratedTrajectory with full logits (not just logprobs)
        """
        interventions = (
            [intervention] if isinstance(intervention, Intervention) else intervention
        )

        all_token_ids = list(token_ids)
        all_logits: list[torch.Tensor] = []

        ctx = (
            torch.inference_mode()
            if self._backend.supports_inference_mode
            else torch.no_grad()
        )

        with ctx:
            for _ in range(max_new_tokens):
                input_ids = torch.tensor([all_token_ids], device=self.device)
                logits_batch = self._backend.run_with_intervention(
                    input_ids, interventions
                )

                if len(all_logits) == 0:
                    all_logits.extend(list(logits_batch[0]))
                else:
                    all_logits.append(logits_batch[0, -1, :])

                next_logits = logits_batch[0, -1, :]
                if temperature == 0.0:
                    next_token = next_logits.argmax().item()
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                all_token_ids.append(next_token)

                if next_token == self.eos_token_id:
                    break

        vocab_size = all_logits[0].shape[-1]
        dummy_logits = torch.zeros(
            vocab_size, device=self.device, dtype=all_logits[0].dtype
        )
        all_logits.append(dummy_logits)
        full_logits = torch.stack(all_logits, dim=0)

        return GeneratedTrajectory.from_inference(
            all_token_ids, full_logits, self.device
        )

    @profile
    def generate_trajectory_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        prefilling: str = "",
        intervention: Optional[Intervention] = None,
    ) -> GeneratedTrajectory:
        """Generate text from prompt and return trajectory with logprobs.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            prefilling: Optional text to prepend to model response
            intervention: Optional intervention to apply during generation

        Returns:
            GeneratedTrajectory containing all tokens with logprobs
        """
        formatted = self.apply_chat_template(prompt) + prefilling
        token_ids = self.encode_ids(formatted, add_special_tokens=True)
        if intervention is not None:
            return self.generate_trajectory_with_intervention(
                token_ids, intervention, max_new_tokens, temperature
            )
        return self.generate_trajectory(token_ids, max_new_tokens, temperature)

    # Optimized inference APIs (for classes like BinaryChoiceRunner)

    @profile
    def compute_trajectory(
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
        return self.compute_trajectories_batch([token_ids])[0]

    @profile
    def compute_trajectories_batch(
        self,
        token_ids_batch: list[list[int]],
    ) -> list[GeneratedTrajectory]:
        max_len = max(len(ids) for ids in token_ids_batch)
        pad_token = self._tokenizer.pad_token_id or 0
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
        input_ids = self.encode(formatted, prepend_bos=prepend_bos)
        return self._backend.run_with_cache(input_ids, names_filter, past_kv_cache)

    @profile
    def run_with_intervention(
        self,
        prompt: str,
        intervention: Interventions,
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
        input_ids = self.encode(formatted, prepend_bos=prepend_bos)
        interventions = (
            [intervention] if isinstance(intervention, Intervention) else intervention
        )
        return self._backend.run_with_intervention(input_ids, interventions)

    # Complex Interpretability APIs

    @profile
    def run_with_intervention_and_cache(
        self,
        prompt: str,
        intervention: Interventions,
        names_filter: Optional[callable] = None,
        prepend_bos: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with intervention AND capture activations with gradients."""
        formatted = self.apply_chat_template(prompt)
        input_ids = self.encode(formatted, prepend_bos=prepend_bos)
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
        input_ids = self.encode(formatted, prepend_bos=prepend_bos)
        return self._backend.run_with_cache_and_grad(input_ids, names_filter)

    # Complex Interpretability APIs (for classes like BinaryChoiceRunner)

    @profile
    def compute_trajectory_with_intervention(
        self,
        token_ids: list[int],
        intervention: Interventions | None = None,
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
    def compute_trajectory_with_cache(
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
    def compute_trajectory_with_intervention_and_cache(
        self,
        token_ids: list[int],
        intervention: Interventions | None = None,
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

    @profile
    def compute_trajectory_with_cache_and_grad(
        self,
        token_ids: list[int],
        names_filter: Optional[callable] = None,
    ) -> GeneratedTrajectory:
        """Generate trajectory with cache and gradients enabled.

        Similar to compute_trajectory_with_cache, but keeps gradients
        enabled for attribution patching. The returned trajectory's
        internals will have requires_grad=True.

        Args:
            token_ids: Full token ID sequence
            names_filter: Function to filter which hooks to cache

        Returns:
            GeneratedTrajectory with internals that support gradient computation
        """
        input_ids = torch.tensor([token_ids], device=self.device)

        # No inference_mode context - keep gradients enabled
        logits_batch, internals_cache = self._backend.run_with_cache_and_grad(
            input_ids, names_filter
        )  # [1, seq_len, vocab_size]

        logits = logits_batch[0]  # [seq_len, vocab_size]
        return GeneratedTrajectory.from_inference(
            token_ids, logits, self.device, internals=internals_cache
        )

    @profile
    def get_embeddings(self, token_ids: list[int]) -> torch.Tensor:
        """Get token embeddings from the model.

        Args:
            token_ids: List of token IDs

        Returns:
            Embeddings tensor [1, seq_len, d_model]
        """
        input_ids = torch.tensor([token_ids], device=self.device)
        return self._backend.get_embeddings(input_ids)

    @property
    def W_E(self) -> torch.Tensor:
        """Token embedding matrix W_E.

        Returns:
            Embedding matrix of shape [vocab_size, d_model]
        """
        return self._backend.get_W_E()

    @property
    def W_U(self) -> torch.Tensor:
        """Unembedding matrix W_U.

        Returns:
            Unembedding matrix of shape [d_model, vocab_size]
        """
        return self._backend.get_W_U()

    @property
    def b_U(self) -> torch.Tensor | None:
        """Unembedding bias b_U.

        Returns:
            Unembedding bias of shape [vocab_size], or None if no bias
        """
        return self._backend.get_b_U()

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
        input_ids = self.encode(formatted, prepend_bos=prepend_bos)

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
            # print(f"apply_chat_template: {self.model_name} is not chat model")
            return prompt
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            # print(f"apply_chat_template: True for {self.model_name}")
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        print(
            f"apply_chat_template: tokenizer does not have apply_chat_template for {self.model_name}"
        )
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
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._backend = PyveneBackend(self, tokenizer)

    def _init_huggingface(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name} on {self.device} (HuggingFace)...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)
        self._model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._backend = HuggingFaceBackend(self, tokenizer)

    def _init_mlx(self) -> None:
        from mlx_lm import load

        print(f"Loading {self.model_name} (MLX)...")
        self._model, tokenizer = load(self.model_name)
        self._backend = MLXBackend(self, tokenizer)

    def _detect_chat_model(self, model_name: str) -> bool:
        """Detect if model is a chat/instruct model based on name.

        Detection strategy (name-based only, tokenizer check was unreliable):
        1. Check for explicit base model indicators (return False)
        2. Special case Qwen3 (always chat/instruct, no base variant exists)
        3. Check for instruct/chat/etc. indicators in name
        """
        if not model_name:
            model_name = self.model_name
        name = model_name.lower()

        # Explicit base model indicators
        if any(x in name for x in ["-base", "_base"]):
            return False

        # Qwen3/Qwen3.5 models are instruct/reasoning by default (no base variant)
        if any(x in name for x in ["qwen3", "qwen-3", "qwen_3"]):
            return True

        # Explicit chat/instruct indicators
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
        tokenizer = self._tokenizer
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
        reasoning_models = ["qwen3", "qwen-3", "qwen_3", "deepseek-r1", "o1", "o3"]
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
