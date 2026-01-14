"""
Activation interventions for modifying model behavior during inference.

Intervention types:
- Steering: Add direction vector to activations
- ActivationPatching: Replace activations with cached values
- Ablation: Zero or mean-ablate activations

Example:
    from src.interventions import SteeringConfig, ApplicationMode, create_intervention_hook

    config = SteeringConfig(
        direction=probe.direction,
        layer=26,
        strength=100.0,
    )

    hook, _ = create_intervention_hook(config, model_dtype, model_device)

    with model.hooks(fwd_hooks=[(config.hook_name, hook)]):
        output = model.generate(...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
import torch

from ..common.schema_utils import SchemaClass


# =============================================================================
# Enums
# =============================================================================


class InterventionType(Enum):
    """Type of intervention."""

    STEERING = "steering"
    ACTIVATION_PATCHING = "activation_patching"
    ABLATION = "ablation"


class ApplicationMode(Enum):
    """How to apply intervention during generation."""

    APPLY_TO_ALL = "apply_to_all"
    APPLY_TO_POSITION = "apply_to_position"
    APPLY_TO_PATTERN = "apply_to_pattern"


class AblationType(Enum):
    """Type of ablation."""

    ZERO = "zero"
    MEAN = "mean"


class PatchSource(Enum):
    """Source of patch activations."""

    CACHED = "cached"
    CLEAN_RUN = "clean_run"
    CORRUPTED_RUN = "corrupted_run"


# =============================================================================
# Position Targeting
# =============================================================================


@dataclass
class PositionTarget(SchemaClass):
    """
    Target position(s) for intervention.

    Examples:
        PositionTarget(index=10)
        PositionTarget(indices=[5, 10, 15])
        PositionTarget(pattern="I select:")
    """

    index: Optional[int] = None
    indices: Optional[list[int]] = None
    pattern: Optional[str] = None

    def __post_init__(self):
        set_count = sum(x is not None for x in [self.index, self.indices, self.pattern])
        if set_count == 0:
            raise ValueError("PositionTarget requires index, indices, or pattern")
        if set_count > 1:
            raise ValueError(
                "PositionTarget can only have one of: index, indices, pattern"
            )
        super().__post_init__()

    @classmethod
    def from_value(cls, value: Union[int, str, list[int]]) -> PositionTarget:
        """Create from int, str, or list[int]."""
        if isinstance(value, int):
            return cls(index=value)
        elif isinstance(value, str):
            return cls(pattern=value)
        elif isinstance(value, list):
            return cls(indices=value)
        raise TypeError(f"Expected int, str, or list[int], got {type(value)}")

    def is_pattern_based(self) -> bool:
        return self.pattern is not None

    def get_indices(self) -> list[int]:
        if self.index is not None:
            return [self.index]
        elif self.indices is not None:
            return self.indices
        return []


# =============================================================================
# Intervention Configs
# =============================================================================


@dataclass
class InterventionConfig(SchemaClass, ABC):
    """Base configuration for interventions."""

    layer: int
    mode: ApplicationMode = ApplicationMode.APPLY_TO_ALL
    position: Optional[PositionTarget] = None
    component: str = "resid_post"

    def __post_init__(self):
        if self.mode != ApplicationMode.APPLY_TO_ALL and self.position is None:
            raise ValueError(f"position required when mode is {self.mode.value}")
        super().__post_init__()

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_{self.component}"

    @property
    @abstractmethod
    def intervention_type(self) -> InterventionType:
        pass


@dataclass
class SteeringConfig(InterventionConfig):
    """
    Steering: add direction vector to activations.

    Attributes:
        direction: Unit direction vector [d_model]
        strength: Scaling factor (typical: 50-200)
    """

    direction: np.ndarray = field(default_factory=lambda: np.array([]))
    strength: float = 1.0

    def __post_init__(self):
        if len(self.direction) > 0:
            norm = np.linalg.norm(self.direction)
            if norm > 0:
                self.direction = self.direction.flatten() / norm
        super().__post_init__()

    @property
    def intervention_type(self) -> InterventionType:
        return InterventionType.STEERING


@dataclass
class ActivationPatchingConfig(InterventionConfig):
    """
    Patching: replace activations with cached values.

    Attributes:
        source: Where patch values come from
        source_positions: Source positions (if different from target)
        patch_values: Pre-computed values (required if source is CACHED)
    """

    source: PatchSource = PatchSource.CACHED
    source_positions: Optional[list[int]] = None
    patch_values: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.source == PatchSource.CACHED and self.patch_values is None:
            raise ValueError("patch_values required when source is CACHED")
        super().__post_init__()

    @property
    def intervention_type(self) -> InterventionType:
        return InterventionType.ACTIVATION_PATCHING


@dataclass
class AblationConfig(InterventionConfig):
    """
    Ablation: zero or mean-ablate activations.

    Attributes:
        ablation_type: ZERO or MEAN
        mean_values: Required if ablation_type is MEAN
    """

    ablation_type: AblationType = AblationType.ZERO
    mean_values: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.ablation_type == AblationType.MEAN and self.mean_values is None:
            raise ValueError("mean_values required when ablation_type is MEAN")
        super().__post_init__()

    @property
    def intervention_type(self) -> InterventionType:
        return InterventionType.ABLATION


# =============================================================================
# Hook Creation
# =============================================================================


def create_intervention_hook(
    config: InterventionConfig,
    model_dtype: torch.dtype,
    model_device: str,
    tokenizer: Optional[object] = None,
) -> tuple[Callable, Optional[PatternMatcher]]:
    """Create hook for intervention config."""
    if isinstance(config, SteeringConfig):
        return _create_steering_hook(config, model_dtype, model_device, tokenizer)
    elif isinstance(config, ActivationPatchingConfig):
        return _create_patching_hook(config, model_dtype, model_device, tokenizer)
    elif isinstance(config, AblationConfig):
        return _create_ablation_hook(config, model_dtype, model_device, tokenizer)
    raise TypeError(f"Unknown config type: {type(config)}")


def _create_steering_hook(
    config: SteeringConfig,
    model_dtype: torch.dtype,
    model_device: str,
    tokenizer: Optional[object],
) -> tuple[Callable, Optional[PatternMatcher]]:
    direction = torch.tensor(config.direction, dtype=model_dtype, device=model_device)
    scaled = config.strength * direction

    if config.mode == ApplicationMode.APPLY_TO_ALL:

        def hook(activation: torch.Tensor, hook=None) -> torch.Tensor:
            activation[:, :, :] += scaled
            return activation

        return hook, None

    return _create_position_hook(scaled, config.position, tokenizer, op="add")


def _create_patching_hook(
    config: ActivationPatchingConfig,
    model_dtype: torch.dtype,
    model_device: str,
    tokenizer: Optional[object],
) -> tuple[Callable, Optional[PatternMatcher]]:
    if config.source != PatchSource.CACHED:
        raise NotImplementedError(f"PatchSource {config.source.value} not implemented")

    values = torch.tensor(config.patch_values, dtype=model_dtype, device=model_device)

    if config.mode == ApplicationMode.APPLY_TO_ALL:

        def hook(activation: torch.Tensor, hook=None) -> torch.Tensor:
            if values.dim() == 1:
                activation[:, :, :] = values
            else:
                seq_len = min(activation.shape[1], values.shape[0])
                activation[:, :seq_len, :] = values[:seq_len]
            return activation

        return hook, None

    return _create_position_hook(
        values,
        config.position,
        tokenizer,
        op="replace",
        source_positions=config.source_positions,
    )


def _create_ablation_hook(
    config: AblationConfig,
    model_dtype: torch.dtype,
    model_device: str,
    tokenizer: Optional[object],
) -> tuple[Callable, Optional[PatternMatcher]]:
    if config.ablation_type == AblationType.ZERO:
        value = torch.zeros(1, dtype=model_dtype, device=model_device)
    else:
        value = torch.tensor(config.mean_values, dtype=model_dtype, device=model_device)

    if config.mode == ApplicationMode.APPLY_TO_ALL:

        def hook(activation: torch.Tensor, hook=None) -> torch.Tensor:
            if config.ablation_type == AblationType.ZERO:
                activation[:, :, :] = 0
            else:
                activation[:, :, :] = value
            return activation

        return hook, None

    return _create_position_hook(
        value,
        config.position,
        tokenizer,
        op="ablate",
        ablation_type=config.ablation_type,
    )


def _create_position_hook(
    values: torch.Tensor,
    target: PositionTarget,
    tokenizer: Optional[object],
    op: str,
    source_positions: Optional[list[int]] = None,
    ablation_type: Optional[AblationType] = None,
) -> tuple[Callable, Optional[PatternMatcher]]:
    """Create hook that applies at specific positions."""
    if target.is_pattern_based():
        if tokenizer is None:
            raise ValueError("tokenizer required for pattern-based targeting")
        matcher = PatternMatcher(target.pattern, tokenizer)

        def hook(activation: torch.Tensor, hook=None) -> torch.Tensor:
            if matcher.should_apply_now():
                if op == "add":
                    activation[:, :, :] += values
                elif op == "replace":
                    activation[:, :, :] = values[-1] if values.dim() > 1 else values
                elif op == "ablate":
                    activation[:, :, :] = (
                        0 if ablation_type == AblationType.ZERO else values
                    )
                matcher.applied()
            return activation

        return hook, matcher

    target_indices = target.get_indices()
    src_indices = source_positions or target_indices

    def hook(activation: torch.Tensor, hook=None) -> torch.Tensor:
        seq_len = activation.shape[1]
        if op == "add":
            if seq_len > 1:
                for idx in target_indices:
                    if idx < seq_len:
                        activation[:, idx, :] += values
                activation[:, -1, :] += values
            else:
                activation[:, 0, :] += values
        elif op == "replace":
            if values.dim() == 1:
                for idx in target_indices:
                    if idx < seq_len:
                        activation[:, idx, :] = values
            else:
                for tgt, src in zip(target_indices, src_indices):
                    if tgt < seq_len and src < values.shape[0]:
                        activation[:, tgt, :] = values[src]
        elif op == "ablate":
            for idx in target_indices:
                if idx < seq_len:
                    if ablation_type == AblationType.ZERO:
                        activation[:, idx, :] = 0
                    else:
                        activation[:, idx, :] = (
                            values[idx]
                            if values.dim() > 1 and idx < values.shape[0]
                            else values
                        )
        return activation

    return hook, None


# =============================================================================
# Pattern Matcher
# =============================================================================


class PatternMatcher:
    """Tracks generated tokens and signals when pattern is matched."""

    def __init__(self, pattern: str, tokenizer: object):
        self.pattern = pattern
        self.tokenizer = tokenizer
        self.generated_text = ""
        self._apply_next = False
        self._applied = False

    def update_generated(self, new_token_ids: torch.Tensor) -> None:
        if self._applied:
            return
        self.generated_text += self.tokenizer.decode(
            new_token_ids, skip_special_tokens=True
        )
        if not self._apply_next and self.pattern in self.generated_text:
            self._apply_next = True

    def should_apply_now(self) -> bool:
        return self._apply_next and not self._applied

    def applied(self) -> None:
        self._apply_next = False
        self._applied = True

    def reset(self) -> None:
        self.generated_text = ""
        self._apply_next = False
        self._applied = False
