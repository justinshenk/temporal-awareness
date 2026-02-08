import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum
from typing import Any


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
