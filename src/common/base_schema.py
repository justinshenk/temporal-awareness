"""Base schema class and utilities for deterministic ID generation."""

from __future__ import annotations

import hashlib
import json
import math
import types
from dataclasses import dataclass, fields, is_dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from .file_io import load_json


# =============================================================================
# Schema utilities (deterministic ID generation)
# =============================================================================


def _qfloat(x: float, places: int = 8) -> float:
    """Stable decimal rounding: converts via str -> Decimal -> quantize."""
    q = Decimal(1) / (Decimal(10) ** places)  # e.g. 1e-8
    d = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_EVEN)
    # normalize -0.0 to 0.0 for stability
    f = float(d)
    return 0.0 if f == 0.0 else f


def _canon(
    obj: Any,
    places: int = 8,
    max_list_length: int | None = None,
    max_string_length: int | None = None,
):
    """Canonicalize object for deterministic hashing.

    Args:
        obj: Object to canonicalize
        places: Decimal places for float rounding
        max_list_length: If set, replace lists longer than this with "[N items]"
        max_string_length: If set, truncate strings longer than this with "...[N chars]"
    """
    # Handle PyTorch tensors (skip them - they're not serializable)
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return None
    except ImportError:
        pass
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return _qfloat(obj, places)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, str):
        if max_string_length is not None and len(obj) > max_string_length:
            return f"{obj[:max_string_length]}...[{len(obj)} chars]"
        return obj
    if is_dataclass(obj):
        # Manually iterate fields to properly handle nested dataclasses with _to_dict_hook
        # (asdict() would convert them to dicts before we can call their hooks)
        result = {}
        for f in fields(obj):
            if f.name.startswith("_"):
                continue
            val = getattr(obj, f.name)
            result[f.name] = _canon(val, places, max_list_length, max_string_length)
        # Apply _to_dict_hook if available (for prob/odds expansion, etc.)
        if hasattr(obj, "_to_dict_hook"):
            result = obj._to_dict_hook(result)
        return result
    if isinstance(obj, dict):
        # Filter out private fields (starting with _) during serialization
        return {
            k: _canon(v, places, max_list_length, max_string_length)
            for k, v in obj.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    if isinstance(obj, (list, tuple)):
        if max_list_length is not None and len(obj) > max_list_length:
            return f"[{len(obj)} items]"
        return [_canon(v, places, max_list_length, max_string_length) for v in obj]
    return obj


def deterministic_id_from_dataclass(
    data_class_obj: Any, places: int = 8, digest_bytes: int = 16
) -> str:
    """Generate a deterministic ID from a dataclass object."""
    canonical = _canon(data_class_obj, places)
    payload = json.dumps(
        canonical,
        sort_keys=True,  # stable key order
        separators=(",", ":"),  # remove whitespace
        ensure_ascii=False,
        allow_nan=False,
    )
    # fast, strong hash in the stdlib
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_bytes)
    return h.hexdigest()


# =============================================================================
# Base schema dataclass
# =============================================================================


@dataclass
class BaseSchema:
    """Base class for schema dataclasses with deterministic ID generation."""

    # Each schema gets unique id based on values
    def get_id(self) -> str:
        return deterministic_id_from_dataclass(self)

    def to_dict(
        self,
        max_list_length: int | None = None,
        max_string_length: int | None = None,
    ) -> dict:
        """Convert to dictionary.

        Args:
            max_list_length: If set, truncate lists longer than this to "[N items]"
            max_string_length: If set, truncate strings longer than this to "...[N chars]"
        """
        return _canon(
            self, max_list_length=max_list_length, max_string_length=max_string_length
        )

    def to_string(
        self,
        max_list_length: int | None = 5,
        max_string_length: int | None = 25,
    ) -> str:
        result_dict = self.to_dict(max_list_length, max_string_length)
        return json.dumps(result_dict, indent=4, sort_keys=True)

    # For logging ease
    def __str__(self) -> str:
        return self.to_string()

    @classmethod
    def _convert_value(cls, val, field_type):
        """Convert a value to the expected field type."""
        # Unwrap Optional[X] / X | None to get X
        origin = get_origin(field_type)
        if origin is Union or isinstance(field_type, types.UnionType):
            args = [a for a in get_args(field_type) if a is not type(None)]
            if len(args) == 1:
                field_type = args[0]

        # Handle None
        if val is None:
            return None

        # Handle Enum
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            return field_type(val) if not isinstance(val, field_type) else val

        # Handle dataclass (supports dict input; also supports list if from_dict handles it)
        if is_dataclass(field_type) and hasattr(field_type, "from_dict"):
            if isinstance(val, dict):
                return field_type.from_dict(val)
            if isinstance(val, list):
                # Some dataclasses (e.g., TimeValue) support parsing from lists
                return field_type.from_dict(val)

        # Handle list[X]
        if get_origin(field_type) is list:
            item_type = get_args(field_type)[0] if get_args(field_type) else None
            if item_type:
                return [cls._convert_value(item, item_type) for item in val]

        # Handle tuple (convert from list, recursively convert items)
        if get_origin(field_type) is tuple:
            item_types = get_args(field_type)
            if item_types:
                return tuple(
                    cls._convert_value(
                        item, item_types[i] if i < len(item_types) else item_types[-1]
                    )
                    for i, item in enumerate(val)
                )
            return tuple(val)

        # Handle dict[K, V]
        if get_origin(field_type) is dict:
            key_type, val_type = (
                get_args(field_type) if get_args(field_type) else (None, None)
            )
            if val_type and is_dataclass(val_type):
                return {k: cls._convert_value(v, val_type) for k, v in val.items()}

        return val

    @classmethod
    def from_dict(cls, d: dict):
        """Recursively construct a dataclass instance from a nested dict."""
        hints = get_type_hints(cls)
        kwargs = {}
        for f in fields(cls):
            if f.name not in d:
                continue  # Let dataclass use its default
            val = d[f.name]
            field_type = hints.get(f.name)
            kwargs[f.name] = cls._convert_value(val, field_type) if field_type else val
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: Path):
        """Load from JSON file. Override from_dict for custom parsing."""
        data = load_json(path)
        return cls.from_dict(data)
