"""Structured result container for extracted activations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch


@dataclass
class ActivationResult:
    """Container for extracted activations with structured access.

    Activations are stored keyed by (module_type, layer, position_label).
    Provides dict-like access and utilities for saving/loading.

    Usage:
        result = extractor.extract(texts)

        # Access by tuple key
        acts = result["resid_post", 16, "last"]  # (n_samples, d_model)

        # Access all activations for a layer
        layer_acts = result.get_layer(16)  # dict of module_type -> tensor

        # Convert to numpy
        np_acts = result.numpy("resid_post", 16, "last")

        # Save to disk
        result.save("./my_activations")

        # Load from disk
        result = ActivationResult.load("./my_activations")
    """

    # Core data: key -> tensor
    # Keys are ModuleSpec.key strings, e.g. "resid_post.layer16"
    activations: dict[str, torch.Tensor] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: Optional[list[list[int]]] = None
    token_strings: Optional[list[list[str]]] = None
    texts: Optional[list[str]] = None

    def __getitem__(self, key: Union[str, tuple]) -> torch.Tensor:
        """Access activations by key.

        Supports multiple key formats:
            result["resid_post.layer16"]          # direct key
            result["resid_post", 16]              # (module_type, layer)
            result["resid_post", 16, "last"]      # with position (ignored in storage)
            result["resid_post", 16, 0]           # head index -> "resid_post.layer16.h0"
        """
        if isinstance(key, str):
            return self.activations[key]

        if isinstance(key, tuple):
            if len(key) == 2:
                module_type, layer = key
                constructed = f"{module_type}.layer{layer}"
                if constructed in self.activations:
                    return self.activations[constructed]
                # Try with head
                for k, v in self.activations.items():
                    if k.startswith(constructed):
                        return v
            elif len(key) == 3:
                module_type, layer, pos_or_head = key
                if isinstance(pos_or_head, int):
                    constructed = f"{module_type}.layer{layer}.h{pos_or_head}"
                    if constructed in self.activations:
                        return self.activations[constructed]
                # Fall back to (module_type, layer)
                constructed = f"{module_type}.layer{layer}"
                if constructed in self.activations:
                    return self.activations[constructed]

        raise KeyError(f"No activations found for key: {key}")

    def __contains__(self, key: Union[str, tuple]) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        return len(self.activations)

    def keys(self):
        return self.activations.keys()

    def values(self):
        return self.activations.values()

    def items(self):
        return self.activations.items()

    @property
    def n_samples(self) -> int:
        """Number of samples in the result."""
        for v in self.activations.values():
            return v.shape[0]
        return 0

    @property
    def layers(self) -> list[int]:
        """Unique layer indices present in the result."""
        layers = set()
        for key in self.activations:
            parts = key.split(".")
            for part in parts:
                if part.startswith("layer"):
                    try:
                        layers.add(int(part[5:]))
                    except ValueError:
                        pass
        return sorted(layers)

    def get_layer(self, layer: int) -> dict[str, torch.Tensor]:
        """Get all activations for a specific layer.

        Returns:
            Dict mapping module_type to tensor.
        """
        result = {}
        target = f"layer{layer}"
        for key, tensor in self.activations.items():
            if target in key:
                result[key] = tensor
        return result

    def numpy(self, *key, squeeze: bool = True) -> np.ndarray:
        """Get activations as numpy array.

        Args:
            *key: Same key formats as __getitem__.
            squeeze: If True, squeeze singleton position dimensions.
                Shape (N, 1, D) becomes (N, D). Default True.

        Returns:
            Numpy array of activations, typically shape (n_samples, d_model).
        """
        tensor = self[key if len(key) > 1 else key[0]]
        if isinstance(tensor, torch.Tensor):
            arr = tensor.float().numpy()
        else:
            arr = np.asarray(tensor)
        if squeeze and arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        return arr

    def save(self, output_dir: str, format: str = "safetensors") -> Path:
        """Save activations to disk.

        Args:
            output_dir: Directory to save to.
            format: "safetensors", "pt", or "npy".

        Returns:
            Path to the output directory.
        """
        import json

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if format == "safetensors":
            self._save_safetensors(out_path)
        elif format == "pt":
            self._save_pt(out_path)
        elif format == "npy":
            self._save_npy(out_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Save metadata
        meta = {
            "n_samples": self.n_samples,
            "keys": list(self.activations.keys()),
            "shapes": {k: list(v.shape) for k, v in self.activations.items()},
            "format": format,
            **self.metadata,
        }
        if self.texts:
            meta["texts"] = self.texts

        with open(out_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return out_path

    def _save_safetensors(self, out_path: Path):
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError("pip install safetensors for safetensors format")

        # safetensors requires string keys with no special chars
        tensor_dict = {}
        for key, tensor in self.activations.items():
            safe_key = key.replace(".", "_")
            if isinstance(tensor, torch.Tensor):
                tensor_dict[safe_key] = tensor.contiguous()
            else:
                tensor_dict[safe_key] = torch.from_numpy(tensor).contiguous()
        save_file(tensor_dict, out_path / "activations.safetensors")

        # Save key mapping
        import json
        key_map = {key.replace(".", "_"): key for key in self.activations}
        with open(out_path / "key_map.json", "w") as f:
            json.dump(key_map, f)

    def _save_pt(self, out_path: Path):
        torch.save(
            {k: v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
             for k, v in self.activations.items()},
            out_path / "activations.pt",
        )

    def _save_npy(self, out_path: Path):
        acts_dir = out_path / "arrays"
        acts_dir.mkdir(exist_ok=True)
        for key, tensor in self.activations.items():
            safe_name = key.replace(".", "_").replace("/", "_")
            arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
            np.save(acts_dir / f"{safe_name}.npy", arr)

    @classmethod
    def load(cls, path: str, format: Optional[str] = None) -> "ActivationResult":
        """Load activations from disk.

        Args:
            path: Directory containing saved activations.
            format: Override auto-detected format.

        Returns:
            ActivationResult with loaded activations.
        """
        import json

        load_path = Path(path)
        meta_path = load_path / "metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        if format is None:
            format = metadata.get("format", "pt")

        if format == "safetensors":
            activations = cls._load_safetensors(load_path)
        elif format == "pt":
            activations = cls._load_pt(load_path)
        elif format == "npy":
            activations = cls._load_npy(load_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        texts = metadata.pop("texts", None)
        # Remove non-metadata fields
        for k in ("n_samples", "keys", "shapes", "format"):
            metadata.pop(k, None)

        return cls(activations=activations, metadata=metadata, texts=texts)

    @classmethod
    def _load_safetensors(cls, path: Path) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file
        import json

        tensors = load_file(path / "activations.safetensors")
        key_map_path = path / "key_map.json"
        if key_map_path.exists():
            with open(key_map_path) as f:
                key_map = json.load(f)
            return {key_map.get(k, k): v for k, v in tensors.items()}
        return tensors

    @classmethod
    def _load_pt(cls, path: Path) -> dict[str, torch.Tensor]:
        return torch.load(path / "activations.pt", weights_only=True)

    @classmethod
    def _load_npy(cls, path: Path) -> dict[str, torch.Tensor]:
        import json
        arrays_dir = path / "arrays"
        result = {}
        # Reconstruct keys from metadata if available
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for key in meta.get("keys", []):
                safe_name = key.replace(".", "_").replace("/", "_")
                npy_path = arrays_dir / f"{safe_name}.npy"
                if npy_path.exists():
                    result[key] = torch.from_numpy(np.load(npy_path))
        else:
            for npy_file in sorted(arrays_dir.glob("*.npy")):
                key = npy_file.stem
                result[key] = torch.from_numpy(np.load(npy_file))
        return result

    def summary(self) -> str:
        """Human-readable summary of the result."""
        lines = [
            f"ActivationResult: {self.n_samples} samples, {len(self.activations)} tensors",
            f"  Layers: {self.layers}",
        ]
        for key, tensor in self.activations.items():
            shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else tensor.shape
            dtype = tensor.dtype
            lines.append(f"  {key}: shape={shape}, dtype={dtype}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
