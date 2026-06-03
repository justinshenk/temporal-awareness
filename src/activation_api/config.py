"""Configuration objects for the activation extraction API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

import torch


@dataclass
class ModuleSpec:
    """Specification for a single module to hook into.

    Attributes:
        module_type: What kind of activation to capture.
            - "resid_pre": Residual stream before the layer
            - "resid_post": Residual stream after the layer (most common)
            - "resid_mid": Residual stream between attention and MLP
            - "attn_out": Output of the attention sublayer
            - "mlp_out": Output of the MLP sublayer
            - "attn_pattern": Attention pattern matrix (n_heads, seq, seq)
            - "attn_scores": Raw attention scores before softmax
            - "custom": Custom module path (set module_name)
        layer: Layer index (0-indexed). Negative indices count from the end.
        module_name: For "custom" type, the full module path
            (e.g., "model.layers.5.self_attn.o_proj")
        head: Optional attention head index. If set, only captures that head's
            activations (reduces memory). Only applies to attn_out/attn_pattern.
    """
    module_type: str = "resid_post"
    layer: int = 0
    module_name: Optional[str] = None
    head: Optional[int] = None

    def __post_init__(self):
        valid_types = {
            "resid_pre", "resid_post", "resid_mid",
            "attn_out", "mlp_out",
            "attn_pattern", "attn_scores",
            "custom",
        }
        if self.module_type not in valid_types:
            raise ValueError(
                f"Invalid module_type '{self.module_type}'. Must be one of: {valid_types}"
            )
        if self.module_type == "custom" and not self.module_name:
            raise ValueError("module_name required when module_type='custom'")

    @property
    def key(self) -> str:
        """Unique key for this module spec."""
        if self.module_type == "custom":
            return self.module_name
        head_suffix = f".h{self.head}" if self.head is not None else ""
        return f"{self.module_type}.layer{self.layer}{head_suffix}"


@dataclass
class ExtractionConfig:
    """Configuration for activation extraction.

    This is the main config object. You can specify what to extract at three
    levels of granularity:

    1. Simple: just set `layers` and `module_types` — extracts all combinations
    2. Detailed: set `modules` with explicit ModuleSpec objects
    3. Custom: set modules with module_type="custom" and provide module paths

    Attributes:
        layers: Layer indices to extract from. Negative indices count from end.
            e.g., [0, 8, 16, -1] extracts from first, 8th, 16th, and last layers.
        module_types: Which module types to extract at each layer.
            Default is ["resid_post"] (residual stream after each layer).
        modules: Explicit list of ModuleSpec objects. If set, overrides
            layers + module_types.
        positions: Which token positions to capture.
            - "all": Every token position (warning: high memory)
            - "last": Only the last token
            - "first": Only the first token
            - "first_last": First and last tokens
            - list[int]: Specific position indices (negative = from end)
        stream_to: Where to store activations during extraction.
            - "cpu": Move tensors to CPU after each forward pass (default)
            - "disk": Write to disk incrementally (lowest memory)
            - "gpu": Keep on GPU (fastest, but highest memory)
        output_dir: Directory for disk streaming. Required if stream_to="disk".
        output_format: Format for disk storage.
            - "safetensors": safetensors format (default, fast + safe)
            - "pt": PyTorch .pt files
            - "npy": NumPy .npy files
        batch_size: Number of samples to process per forward pass.
        max_seq_len: Maximum sequence length. Longer inputs are truncated.
        dtype: Data type for stored activations.
            - None: Keep original model dtype
            - "float32": Upcast to float32
            - "float16": Downcast to float16 (saves ~50% memory)
            - "bfloat16": Downcast to bfloat16
        collect_metadata: Whether to store tokenization info alongside activations.
        use_transformer_lens: Whether to use TransformerLens hooks (True) or
            raw PyTorch hooks (False). Auto-detected if not set.
        device: Device to run the model on. Auto-detected if not set.
    """

    # What to extract
    layers: list[int] = field(default_factory=lambda: [-1])
    module_types: list[str] = field(default_factory=lambda: ["resid_post"])
    modules: Optional[list[ModuleSpec]] = None
    positions: Union[str, list[int]] = "last"

    # Memory management
    stream_to: Literal["cpu", "disk", "gpu"] = "cpu"
    output_dir: Optional[str] = None
    output_format: Literal["safetensors", "pt", "npy"] = "safetensors"
    batch_size: int = 4
    max_seq_len: Optional[int] = None
    dtype: Optional[str] = None

    # Metadata
    collect_metadata: bool = True

    # Model loading
    model_dtype: Optional[str] = None  # dtype for model weights (e.g., "float16")
                                        # separate from `dtype` which controls activation storage

    # Backend
    use_transformer_lens: Optional[bool] = None
    device: Optional[str] = None

    def __post_init__(self):
        if self.stream_to == "disk" and not self.output_dir:
            raise ValueError("output_dir is required when stream_to='disk'")
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def resolve_modules(self, n_layers: int) -> list[ModuleSpec]:
        """Resolve config into explicit ModuleSpec list.

        Args:
            n_layers: Total number of layers in the model.

        Returns:
            List of ModuleSpec objects to hook.
        """
        if self.modules is not None:
            # Resolve negative layer indices
            resolved = []
            for m in self.modules:
                layer = m.layer
                if layer < 0:
                    layer = n_layers + layer
                resolved.append(ModuleSpec(
                    module_type=m.module_type,
                    layer=layer,
                    module_name=m.module_name,
                    head=m.head,
                ))
            return resolved

        # Build from layers × module_types
        resolved = []
        for layer in self.layers:
            actual_layer = layer if layer >= 0 else n_layers + layer
            if actual_layer < 0 or actual_layer >= n_layers:
                raise ValueError(
                    f"Layer index {layer} out of range for model with {n_layers} layers"
                )
            for mtype in self.module_types:
                resolved.append(ModuleSpec(module_type=mtype, layer=actual_layer))
        return resolved

    def resolve_dtype(self, override: Optional[str] = None) -> Optional["torch.dtype"]:
        """Convert string dtype to torch dtype.

        Args:
            override: If provided, resolve this string instead of self.dtype.
        """
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype_str = override or self.dtype
        if dtype_str is None:
            return None
        if dtype_str not in dtype_map:
            raise ValueError(f"Unknown dtype '{dtype_str}'. Use: {list(dtype_map.keys())}")
        return dtype_map[dtype_str]
