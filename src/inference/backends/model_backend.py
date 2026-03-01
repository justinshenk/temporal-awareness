"""Abstract base class for model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence

import torch

from ..interventions import Intervention


class ModelBackend(Enum):
    """Available model backends."""

    PYVENE = "pyvene"
    MLX = "mlx"
    TRANSFORMERLENS = "transformerlens"
    HUGGINGFACE = "huggingface"
    NNSIGHT = "nnsight"


class Backend(ABC):
    """Abstract base class for model backends.

    All backends must implement these methods to provide a consistent interface
    for model inference and interventions.
    """

    supports_inference_mode: bool = (
        True  # Override to False if backend conflicts with inference_mode
    )

    def __init__(self, runner: Any):
        """Initialize backend with a reference to the ModelRunner.

        Args:
            runner: ModelRunner instance that owns this backend
        """
        self.runner = runner

    @abstractmethod
    def get_tokenizer(self):
        """Get the tokenizer for this backend."""
        ...

    @abstractmethod
    def get_n_layers(self) -> int:
        """Get the number of layers in the model."""
        ...

    @abstractmethod
    def get_d_model(self) -> int:
        """Get the hidden dimension of the model."""
        ...

    @abstractmethod
    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        """Encode text into token IDs tensor."""
        ...

    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        """Get next token probabilities for target tokens."""
        ...

    @abstractmethod
    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        """Get next token probabilities by token ID."""
        ...

    @abstractmethod
    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass and return activation cache."""
        ...

    @abstractmethod
    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass with gradients enabled."""
        ...

    @abstractmethod
    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using prefill logits and frozen KV cache."""
        ...

    @abstractmethod
    def init_kv_cache(self):
        """Initialize a KV cache for the model."""
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return logits.

        Args:
            input_ids: Token IDs tensor of shape [batch, seq_len]

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size]
        """
        ...

    @abstractmethod
    def run_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
    ) -> torch.Tensor:
        """Run forward pass with interventions, returning logits."""
        ...

    @abstractmethod
    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with interventions AND capture activations with gradients."""
        ...

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from the model.

        Args:
            token_ids: Token IDs [batch, seq_len] or [seq_len]

        Returns:
            Embeddings tensor [batch, seq_len, d_model]

        Note: Not all backends support this. Override in subclass if supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support get_embeddings"
        )
