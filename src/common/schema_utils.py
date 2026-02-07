import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from .io import load_json


def _qfloat(x: float, places: int = 8) -> float:
    # stable decimal rounding: converts via str -> Decimal -> quantize
    q = Decimal(1) / (Decimal(10) ** places)  # e.g. 1e-8
    d = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_EVEN)
    # normalize -0.0 to 0.0 for stability
    f = float(d)
    return 0.0 if f == 0.0 else f


def _canon(obj: Any, places: int = 8):
    if isinstance(obj, float):
        if math.isnan(obj):  # avoid NaN destabilizing hashes
            return "NaN"
        return _qfloat(obj, places)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return _canon(asdict(obj), places)
    if isinstance(obj, dict):
        return {k: _canon(v, places) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v, places) for v in obj]
    return obj


def deterministic_id_from_dataclass(
    data_class_obj: Any, places: int = 8, digest_bytes: int = 16
) -> str:
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


@dataclass
class SchemaClass:
    # Each schema gets unique id based on values
    def get_id(self) -> str:
        return deterministic_id_from_dataclass(self)

    # For logging ease
    def __str__(self) -> str:
        result_dict = asdict(self)
        return json.dumps(result_dict, indent=4)

    # Each trial should have their own set of params
    # We want to make sure schemas are unique and immutable
    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, copy.deepcopy(getattr(self, f.name)))

    def __copy__(self):
        return self.__deepcopy__({})

    def __deepcopy__(self, memo):
        cls = self.__class__
        kwargs = {
            f.name: copy.deepcopy(getattr(self, f.name), memo) for f in fields(self)
        }
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, d: dict):
        """Recursively construct a dataclass instance from a nested dict."""
        hints = get_type_hints(cls)
        kwargs = {}
        for f in fields(cls):
            val = d.get(f.name)
            if val is None:
                kwargs[f.name] = val
                continue
            field_type = hints[f.name]
            # Unwrap Optional[X] (= Union[X, None]) to get X
            if get_origin(field_type) is Union:
                args = [a for a in get_args(field_type) if a is not type(None)]
                if len(args) == 1:
                    field_type = args[0]
            if isinstance(val, dict) and is_dataclass(field_type):
                kwargs[f.name] = field_type.from_dict(val)
            else:
                kwargs[f.name] = val
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: Path):
        """Load from JSON file. Override from_dict for custom parsing."""
        data = load_json(path)
        return cls.from_dict(data)

    def __setattr__(self, name, value):
        super().__setattr__(name, copy.deepcopy(value))
