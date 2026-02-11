"""HuggingFace Transformers backend implementation."""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Backend
from ..interventions import Intervention


class HuggingFaceBackend(Backend):
    """Backend using HuggingFace Transformers for model inference.

    This backend provides basic inference capabilities using the HuggingFace
    transformers library. It supports:
    - Text generation
    - Forward passes with activation caching via hooks
    - Tokenization and decoding

    Note: Some advanced features like interventions and KV caching are not
    yet implemented in this backend.
    """

    def get_tokenizer(self):
        """Get the tokenizer for this backend."""
        return self.runner._tokenizer

    def get_n_layers(self) -> int:
        """Get the number of layers in the model."""
        model = self.runner._model
        config = model.config

        # Different model architectures use different attribute names
        if hasattr(config, "num_hidden_layers"):
            return config.num_hidden_layers
        elif hasattr(config, "n_layer"):
            return config.n_layer
        elif hasattr(config, "num_layers"):
            return config.num_layers
        else:
            raise AttributeError(
                f"Cannot determine number of layers for model config: {type(config)}"
            )

    def get_d_model(self) -> int:
        """Get the hidden dimension of the model."""
        model = self.runner._model
        config = model.config

        # Different model architectures use different attribute names
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        elif hasattr(config, "d_model"):
            return config.d_model
        elif hasattr(config, "n_embd"):
            return config.n_embd
        else:
            raise AttributeError(
                f"Cannot determine hidden dimension for model config: {type(config)}"
            )

    def tokenize(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        """Tokenize text into token IDs tensor."""
        tokenizer = self.get_tokenizer()

        # HuggingFace tokenizers typically add BOS automatically if configured
        # We use add_special_tokens to control this behavior
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=prepend_bos,
        )

        return encoded["input_ids"].to(self.runner.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        tokenizer = self.get_tokenizer()

        # Handle both 1D and 2D tensors
        if token_ids.dim() == 2:
            token_ids = token_ids[0]

        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention],
        past_kv_cache: Any = None,
    ) -> str:
        """Generate text from a prompt."""
        if intervention is not None:
            raise NotImplementedError(
                "HuggingFaceBackend does not support interventions during generation. "
                "Use TransformerLens or NNsight backend for intervention support."
            )

        if past_kv_cache is not None:
            raise NotImplementedError(
                "HuggingFaceBackend does not support past_kv_cache in generate. "
                "Use TransformerLens backend for KV cache support."
            )

        input_ids = self.tokenize(prompt)
        prompt_len = input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.get_tokenizer().eos_token_id,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.runner._model.generate(input_ids, **gen_kwargs)

        # Return only the newly generated tokens
        return self.decode(output_ids[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: list[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        """Get next token probabilities for target tokens."""
        raise NotImplementedError(
            "HuggingFaceBackend.get_next_token_probs is not implemented. "
            "Use TransformerLens backend for this functionality."
        )

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: list[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        """Get next token probabilities by token ID."""
        raise NotImplementedError(
            "HuggingFaceBackend.get_next_token_probs_by_id is not implemented. "
            "Use TransformerLens backend for this functionality."
        )

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
        past_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass and return activation cache.

        Uses PyTorch hooks to capture activations from model layers.
        The cache keys follow the TransformerLens naming convention:
        'blocks.{layer}.hook_{component}' for compatibility.
        """
        if past_kv_cache is not None:
            raise NotImplementedError(
                "HuggingFaceBackend does not support past_kv_cache in run_with_cache. "
                "Use TransformerLens backend for KV cache support."
            )

        cache = {}
        handles = []
        model = self.runner._model

        # Get the transformer blocks - different architectures have different names
        blocks = self._get_transformer_blocks()

        def make_hook(hook_name: str):
            def hook_fn(module, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                cache[hook_name] = hidden_states.detach()
            return hook_fn

        # Register hooks for each layer
        for layer_idx, block in enumerate(blocks):
            # We capture the output of each transformer block as resid_post
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            if names_filter is None or names_filter(hook_name):
                handle = block.register_forward_hook(make_hook(hook_name))
                handles.append(handle)

        try:
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=False)
                logits = outputs.logits
        finally:
            # Remove all hooks
            for handle in handles:
                handle.remove()

        return logits, cache

    def run_with_cache_and_grad(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward pass with gradients enabled.

        Similar to run_with_cache but keeps gradients for attribution patching.
        """
        cache = {}
        handles = []
        model = self.runner._model

        # Get the transformer blocks
        blocks = self._get_transformer_blocks()

        def make_hook(hook_name: str):
            def hook_fn(module, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                # Don't detach - keep gradients
                cache[hook_name] = hidden_states
            return hook_fn

        # Register hooks for each layer
        for layer_idx, block in enumerate(blocks):
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            if names_filter is None or names_filter(hook_name):
                handle = block.register_forward_hook(make_hook(hook_name))
                handles.append(handle)

        try:
            # No torch.no_grad() - we want gradients
            outputs = model(input_ids, output_hidden_states=False)
            logits = outputs.logits
        finally:
            # Remove all hooks
            for handle in handles:
                handle.remove()

        return logits, cache

    def generate_from_cache(
        self,
        prefill_logits: torch.Tensor,
        frozen_kv_cache: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using prefill logits and frozen KV cache."""
        raise NotImplementedError(
            "HuggingFaceBackend.generate_from_cache is not implemented. "
            "Use TransformerLens backend for KV cache-based generation."
        )

    def init_kv_cache(self):
        """Initialize a KV cache for the model."""
        raise NotImplementedError(
            "HuggingFaceBackend.init_kv_cache is not implemented. "
            "Use TransformerLens backend for KV cache support."
        )

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
        with torch.no_grad():
            outputs = self.runner._model(input_ids)
            return outputs.logits

    def forward_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: list[Intervention],
    ) -> torch.Tensor:
        """Run forward pass with interventions, returning logits."""
        raise NotImplementedError(
            "HuggingFaceBackend.forward_with_intervention is not implemented. "
            "Use TransformerLens or NNsight backend for intervention support."
        )

    def forward_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: list[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with interventions AND capture activations with gradients."""
        raise NotImplementedError(
            "HuggingFaceBackend.forward_with_intervention_and_cache is not implemented. "
            "Use TransformerLens or NNsight backend for intervention support."
        )

    def _get_transformer_blocks(self):
        """Get the list of transformer blocks from the model.

        Different model architectures organize their layers differently.
        This method handles common patterns.
        """
        model = self.runner._model

        # Try common attribute paths for different model architectures
        # GPT-2, GPT-Neo style
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h

        # LLaMA, Mistral, Qwen style
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers

        # BLOOM style
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h

        # OPT style
        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            if hasattr(model.model.decoder, "layers"):
                return model.model.decoder.layers

        # Falcon style
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h

        # Generic fallback - try to find layers
        if hasattr(model, "base_model"):
            base = model.base_model
            if hasattr(base, "layers"):
                return base.layers
            if hasattr(base, "h"):
                return base.h

        raise AttributeError(
            f"Cannot find transformer blocks for model architecture: {type(model)}. "
            "Please add support for this architecture in _get_transformer_blocks()."
        )
