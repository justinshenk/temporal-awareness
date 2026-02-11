"""MLX backend implementation for Apple Silicon optimization."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import torch

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from mlx_lm.utils import generate_step
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from .base import Backend

if TYPE_CHECKING:
    from ..interventions import Intervention


def _check_mlx_available():
    """Check if MLX is available and raise helpful error if not."""
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX is not available. Please install mlx and mlx_lm packages:
"
            "  pip install mlx mlx-lm
"
            "Note: MLX is only supported on Apple Silicon (M1/M2/M3) Macs."
        )


def _torch_to_mlx(tensor: torch.Tensor) -> "mx.array":
    """Convert PyTorch tensor to MLX array."""
    return mx.array(tensor.cpu().numpy())


def _mlx_to_torch(array: "mx.array", device: str = "cpu") -> torch.Tensor:
    """Convert MLX array to PyTorch tensor."""
    import numpy as np
    return torch.from_numpy(np.array(array)).to(device)


class MLXBackend(Backend):
    """Backend using MLX for Apple Silicon optimized inference.

    This backend provides efficient inference on Apple Silicon devices
    using the MLX framework. It supports basic generation and forward
    passes, with activation caching for interpretability research.
    """

    def __init__(self, runner: Any):
        """Initialize MLX backend.

        Args:
            runner: ModelRunner instance that owns this backend
        """
        _check_mlx_available()
        super().__init__(runner)
        self._model = None
        self._tokenizer = None
        self._activation_cache = {}

    def _ensure_model_loaded(self):
        """Ensure the MLX model is loaded."""
        if self._model is None:
            # Load model using mlx_lm
            self._model, self._tokenizer = load(self.runner.model_name)

    def get_tokenizer(self):
        """Get the tokenizer for this backend."""
        self._ensure_model_loaded()
        return self._tokenizer

    def get_n_layers(self) -> int:
        """Get the number of layers in the model."""
        self._ensure_model_loaded()
        # MLX models typically store layers in model.layers
        if hasattr(self._model, "layers"):
            return len(self._model.layers)
        elif hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return len(self._model.model.layers)
        else:
            raise AttributeError("Could not determine number of layers in MLX model")

    def get_d_model(self) -> int:
        """Get the hidden dimension of the model."""
        self._ensure_model_loaded()
        # Try common attribute names for hidden size
        if hasattr(self._model, "args"):
            if hasattr(self._model.args, "hidden_size"):
                return self._model.args.hidden_size
            elif hasattr(self._model.args, "dim"):
                return self._model.args.dim
        if hasattr(self._model, "config"):
            if hasattr(self._model.config, "hidden_size"):
                return self._model.config.hidden_size
        raise AttributeError("Could not determine hidden dimension in MLX model")

    def tokenize(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        """Tokenize text into token IDs tensor."""
        self._ensure_model_loaded()
        # Use the tokenizer to encode
        tokens = self._tokenizer.encode(text)
        if prepend_bos and self._tokenizer.bos_token_id is not None:
            if not tokens or tokens[0] != self._tokenizer.bos_token_id:
                tokens = [self._tokenizer.bos_token_id] + tokens
        return torch.tensor([tokens])

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        self._ensure_model_loaded()
        # Handle both 1D and 2D tensors
        if token_ids.dim() == 2:
            token_ids = token_ids[0]
        return self._tokenizer.decode(token_ids.tolist())

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional["Intervention"],
        past_kv_cache: Any = None,
    ) -> str:
        """Generate text from a prompt."""
        self._ensure_model_loaded()

        if intervention is not None:
            raise NotImplementedError(
                "MLXBackend does not support interventions during generation. "
                "Use TransformerLensBackend or NNsightBackend for intervention support."
            )

        if past_kv_cache is not None:
            raise NotImplementedError(
                "MLXBackend does not support past_kv_cache for generation. "
                "Use TransformerLensBackend for KV cache support."
            )

        # Use mlx_lm generate function
        tokens = self._tokenizer.encode(prompt)
        prompt_tokens = mx.array(tokens)

        # Generate tokens
        generated_tokens = []
        temp = temperature if temperature > 0 else 0.0

        for token, _ in generate_step(
            prompt=prompt_tokens,
            model=self._model,
            temp=temp,
        ):
            generated_tokens.append(token.item())
            if len(generated_tokens) >= max_new_tokens:
                break
            # Check for EOS
            if self._tokenizer.eos_token_id is not None:
                if token.item() == self._tokenizer.eos_token_id:
                    break

        return self._tokenizer.decode(generated_tokens)

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass and return activation cache.

        This captures intermediate activations during the forward pass
        for interpretability analysis.
        """
        self._ensure_model_loaded()

        if past_kv_cache is not None:
            raise NotImplementedError(
                "MLXBackend.run_with_cache does not support past_kv_cache."
            )

        # Convert input to MLX
        mlx_input = _torch_to_mlx(input_ids)

        cache = {}
        n_layers = self.get_n_layers()

        # Get the actual model (might be wrapped)
        model = self._model.model if hasattr(self._model, "model") else self._model

        # Get embeddings
        if hasattr(model, "embed_tokens"):
            hidden_states = model.embed_tokens(mlx_input)
        else:
            raise AttributeError("Could not find embedding layer in MLX model")

        # Run through each layer and capture activations
        for i, layer in enumerate(model.layers):
            hidden_states = layer(hidden_states)

            # Create hook name matching TransformerLens convention
            hook_name = f"blocks.{i}.hook_resid_post"

            if names_filter is None or names_filter(hook_name):
                # Convert to torch and store
                cache[hook_name] = _mlx_to_torch(hidden_states, self.runner.device)

        # Apply final norm if present
        if hasattr(model, "norm"):
            hidden_states = model.norm(hidden_states)

        # Get logits
        if hasattr(self._model, "lm_head"):
            logits = self._model.lm_head(hidden_states)
        elif hasattr(model, "lm_head"):
            logits = model.lm_head(hidden_states)
        else:
            raise AttributeError("Could not find lm_head in MLX model")

        logits_torch = _mlx_to_torch(logits, self.runner.device)

        return logits_torch, cache

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass with gradients enabled.

        Note: MLX has its own autograd system. This implementation
        captures activations but gradient computation differs from PyTorch.
        """
        self._ensure_model_loaded()

        # For now, run_with_cache_and_grad is similar to run_with_cache
        # MLX gradients work differently than PyTorch
        # The activations are captured and can be used for analysis
        return self.run_with_cache(input_ids, names_filter, past_kv_cache=None)

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
        self._ensure_model_loaded()

        # Convert to MLX
        mlx_input = _torch_to_mlx(input_ids)

        # Run forward pass
        logits = self._model(mlx_input)

        # Convert back to torch
        return _mlx_to_torch(logits, self.runner.device)

    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        """Get next token probabilities for target tokens."""
        raise NotImplementedError(
            "MLXBackend.get_next_token_probs is not implemented. "
            "Use TransformerLensBackend for token probability queries."
        )

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: list[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        """Get next token probabilities by token ID."""
        raise NotImplementedError(
            "MLXBackend.get_next_token_probs_by_id is not implemented. "
            "Use TransformerLensBackend for token probability queries."
        )

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using prefill logits and frozen KV cache."""
        raise NotImplementedError(
            "MLXBackend.generate_from_cache is not implemented. "
            "Use TransformerLensBackend for KV cache based generation."
        )

    def init_kv_cache(self):
        """Initialize a KV cache for the model."""
        raise NotImplementedError(
            "MLXBackend.init_kv_cache is not implemented. "
            "Use TransformerLensBackend for KV cache support."
        )

    def forward_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: list["Intervention"],
    ) -> torch.Tensor:
        """Run forward pass with interventions, returning logits."""
        raise NotImplementedError(
            "MLXBackend.forward_with_intervention is not implemented. "
            "Use TransformerLensBackend or NNsightBackend for intervention support."
        )

    def forward_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: list["Intervention"],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with interventions AND capture activations with gradients."""
        raise NotImplementedError(
            "MLXBackend.forward_with_intervention_and_cache is not implemented. "
            "Use TransformerLensBackend or NNsightBackend for intervention support."
        )
