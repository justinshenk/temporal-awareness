"""Hook management for activation extraction.

Handles registering forward hooks on PyTorch modules and TransformerLens
models, capturing tensors, and managing memory during extraction.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .config import ExtractionConfig, ModuleSpec


class HookManager:
    """Manages PyTorch forward hooks for activation capture.

    Supports two backends:
    1. TransformerLens: Uses HookedTransformer's run_with_cache with name filters
    2. Raw PyTorch: Registers forward hooks on nn.Module instances

    The manager handles:
    - Registering/removing hooks cleanly (context manager)
    - Capturing activations into a buffer
    - Position selection (to avoid storing full sequences)
    - Optional dtype casting
    - Streaming to CPU during capture
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._buffer: dict[str, list[torch.Tensor]] = {}
        self._active = False

    def _resolve_positions(self, seq_len: int) -> Optional[list[int]]:
        """Convert position spec to concrete indices.

        Returns None if all positions should be kept.
        """
        pos = self.config.positions
        if pos == "all":
            return None
        elif pos == "last":
            return [seq_len - 1]
        elif pos == "first":
            return [0]
        elif pos == "first_last":
            return [0, seq_len - 1]
        elif isinstance(pos, list):
            return [p if p >= 0 else seq_len + p for p in pos]
        else:
            raise ValueError(f"Unknown position spec: {pos}")

    def _make_hook_fn(
        self,
        key: str,
        head: Optional[int] = None,
    ) -> Callable:
        """Create a forward hook function that captures activations.

        Args:
            key: Storage key for this hook's activations.
            head: If set, select only this attention head.

        Returns:
            Hook function compatible with PyTorch register_forward_hook.
        """
        target_dtype = self.config.resolve_dtype()
        stream_to = self.config.stream_to

        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            if not self._active:
                return

            # Extract tensor from output (handle tuple outputs)
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            if not isinstance(tensor, torch.Tensor):
                return

            # tensor shape: (batch, seq_len, d_model) or (batch, n_heads, seq, seq)
            # Select positions
            if tensor.dim() >= 3:
                positions = self._resolve_positions(tensor.shape[-2])
                if positions is not None:
                    tensor = tensor[..., positions, :]

            # Select specific attention head
            if head is not None and tensor.dim() == 4:
                tensor = tensor[:, head]

            # Cast dtype
            if target_dtype is not None:
                tensor = tensor.to(target_dtype)

            # Stream to target device
            if stream_to == "cpu":
                tensor = tensor.detach().cpu()
            elif stream_to == "gpu":
                tensor = tensor.detach()
            else:
                tensor = tensor.detach().cpu()

            if key not in self._buffer:
                self._buffer[key] = []
            self._buffer[key].append(tensor)

        return hook_fn

    def register_hooks_pytorch(
        self,
        model: nn.Module,
        modules: list[ModuleSpec],
    ) -> None:
        """Register forward hooks on a raw PyTorch model.

        Maps ModuleSpec objects to actual nn.Module instances in the model
        and registers hooks.

        Args:
            model: The PyTorch model.
            modules: List of ModuleSpec objects specifying what to hook.
        """
        self.clear()

        for spec in modules:
            target = self._find_module(model, spec)
            if target is None:
                print(f"  Warning: Could not find module for {spec.key}, skipping")
                continue

            hook_fn = self._make_hook_fn(spec.key, head=spec.head)
            handle = target.register_forward_hook(hook_fn)
            self._hooks.append(handle)

    def _find_module(self, model: nn.Module, spec: ModuleSpec) -> Optional[nn.Module]:
        """Find the nn.Module corresponding to a ModuleSpec.

        Tries multiple naming conventions to handle different model architectures:
        - HuggingFace Llama/Qwen: model.layers.{N}.self_attn, model.layers.{N}.mlp
        - GPT-2: transformer.h.{N}.attn, transformer.h.{N}.mlp
        - TransformerLens: blocks.{N}.hook_resid_post, etc.
        """
        if spec.module_type == "custom":
            return self._get_module_by_path(model, spec.module_name)

        layer = spec.layer

        # Try common naming patterns
        patterns = self._get_module_patterns(spec.module_type, layer)

        for pattern in patterns:
            module = self._get_module_by_path(model, pattern)
            if module is not None:
                return module

        return None

    def _get_module_patterns(self, module_type: str, layer: int) -> list[str]:
        """Get candidate module paths for a given module type and layer."""
        L = layer
        patterns = {
            "resid_post": [
                # TransformerLens
                f"blocks.{L}.hook_resid_post",
                # HuggingFace Llama/Mistral/Qwen — the layer itself outputs resid_post
                f"model.layers.{L}",
                # GPT-2
                f"transformer.h.{L}",
            ],
            "resid_pre": [
                f"blocks.{L}.hook_resid_pre",
                f"blocks.{L}.ln1",
                f"model.layers.{L}.input_layernorm",
                f"transformer.h.{L}.ln_1",
            ],
            "resid_mid": [
                f"blocks.{L}.hook_resid_mid",
                f"blocks.{L}.hook_mlp_in",
                f"model.layers.{L}.post_attention_layernorm",
                f"transformer.h.{L}.ln_2",
            ],
            "attn_out": [
                f"blocks.{L}.hook_attn_out",
                f"blocks.{L}.attn.hook_result",
                f"model.layers.{L}.self_attn",
                f"transformer.h.{L}.attn",
            ],
            "mlp_out": [
                f"blocks.{L}.hook_mlp_out",
                f"model.layers.{L}.mlp",
                f"transformer.h.{L}.mlp",
            ],
            "attn_pattern": [
                f"blocks.{L}.attn.hook_pattern",
                f"model.layers.{L}.self_attn",  # need post-processing
            ],
            "attn_scores": [
                f"blocks.{L}.attn.hook_attn_scores",
            ],
        }
        return patterns.get(module_type, [])

    @staticmethod
    def _get_module_by_path(model: nn.Module, path: str) -> Optional[nn.Module]:
        """Traverse a model to find a module by dotted path."""
        parts = path.split(".")
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif part.isdigit() and hasattr(current, "__getitem__"):
                try:
                    current = current[int(part)]
                except (IndexError, KeyError):
                    return None
            else:
                return None
        return current if current is not model else None

    def build_transformer_lens_filter(
        self,
        modules: list[ModuleSpec],
    ) -> Callable[[str], bool]:
        """Build a names_filter function for TransformerLens run_with_cache.

        Args:
            modules: List of ModuleSpec objects.

        Returns:
            A function that returns True for hook names we want to capture.
        """
        # Map module_type to TransformerLens hook name patterns
        tl_patterns = set()
        for spec in modules:
            L = spec.layer
            mapping = {
                "resid_post": f"blocks.{L}.hook_resid_post",
                "resid_pre": f"blocks.{L}.hook_resid_pre",
                "resid_mid": f"blocks.{L}.hook_resid_mid",
                "attn_out": f"blocks.{L}.hook_attn_out",
                "mlp_out": f"blocks.{L}.hook_mlp_out",
                "attn_pattern": f"blocks.{L}.attn.hook_pattern",
                "attn_scores": f"blocks.{L}.attn.hook_attn_scores",
            }
            if spec.module_type in mapping:
                tl_patterns.add(mapping[spec.module_type])

        def names_filter(name: str) -> bool:
            return name in tl_patterns

        return names_filter

    def extract_from_cache(
        self,
        cache: dict,
        modules: list[ModuleSpec],
        seq_len: int,
    ) -> dict[str, torch.Tensor]:
        """Extract activations from a TransformerLens cache dict.

        Args:
            cache: The cache dict from run_with_cache.
            modules: ModuleSpec list to extract.
            seq_len: Sequence length for position resolution.

        Returns:
            Dict mapping spec keys to tensors.
        """
        target_dtype = self.config.resolve_dtype()
        positions = self._resolve_positions(seq_len)
        result = {}

        for spec in modules:
            L = spec.layer
            mapping = {
                "resid_post": f"blocks.{L}.hook_resid_post",
                "resid_pre": f"blocks.{L}.hook_resid_pre",
                "resid_mid": f"blocks.{L}.hook_resid_mid",
                "attn_out": f"blocks.{L}.hook_attn_out",
                "mlp_out": f"blocks.{L}.hook_mlp_out",
                "attn_pattern": f"blocks.{L}.attn.hook_pattern",
                "attn_scores": f"blocks.{L}.attn.hook_attn_scores",
            }
            hook_name = mapping.get(spec.module_type)
            if hook_name and hook_name in cache:
                tensor = cache[hook_name]
                if isinstance(tensor, torch.Tensor):
                    # Position selection
                    if positions is not None and tensor.dim() >= 3:
                        tensor = tensor[..., positions, :]

                    # Head selection
                    if spec.head is not None and tensor.dim() == 4:
                        tensor = tensor[:, spec.head]

                    # Dtype cast
                    if target_dtype is not None:
                        tensor = tensor.to(target_dtype)

                    # Stream to CPU
                    if self.config.stream_to in ("cpu", "disk"):
                        tensor = tensor.detach().cpu()
                    else:
                        tensor = tensor.detach()

                    result[spec.key] = tensor

        return result

    @property
    def buffer(self) -> dict[str, list[torch.Tensor]]:
        """Access the captured activation buffer."""
        return self._buffer

    def activate(self):
        """Enable activation capture."""
        self._active = True

    def deactivate(self):
        """Disable activation capture (hooks stay registered but don't store)."""
        self._active = False

    def clear(self):
        """Remove all hooks and clear the buffer."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._buffer.clear()
        self._active = False

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, *args):
        # Only deactivate and remove hooks — preserve the buffer so the
        # caller can read captured activations after the `with` block.
        self.deactivate()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
