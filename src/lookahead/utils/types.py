"""Core types for lookahead planning detection.

All data structures used across the lookahead module are defined here
to ensure consistency and prevent implicit schema drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class TaskType(str, Enum):
    """Task types for planning detection."""
    RHYME = "rhyme"
    ACROSTIC = "acrostic"
    CODE_RETURN = "code_return"
    STRUCTURED_LIST = "structured_list"


@dataclass
class PlanningExample:
    """A single planning-detection example with ground-truth commitment target.
    
    Fields:
        task_type: Which planning task this belongs to.
        prompt: The full input prompt given to the model.
        target_value: The ground-truth planned value (e.g., the rhyme word,
            the acrostic letter sequence, the return type).
        target_token_position: The token position(s) where the target appears
            in the *completion* (0-indexed from start of completion).
        commitment_region: Optional known region where commitment should appear
            (e.g., in a couplet, the commitment might start when "red" is seen).
        metadata: Task-specific metadata (rhyme pattern, function signature, etc.)
        example_id: Unique identifier for this example.
    """
    task_type: TaskType
    prompt: str
    target_value: str
    target_token_positions: list[int]  # positions in full sequence where target tokens appear
    commitment_region: Optional[tuple[int, int]] = None  # (earliest_possible, latest_possible)
    metadata: dict = field(default_factory=dict)
    example_id: str = ""


@dataclass
class ActivationCache:
    """Cached activations for a single example across all layers and positions.
    
    Shape conventions:
        activations: dict[layer_idx -> np.ndarray of shape (seq_len, d_model)]
        logits: np.ndarray of shape (seq_len, vocab_size) — only if requested
    """
    example_id: str
    token_ids: list[int]
    token_strings: list[str]
    activations: dict[int, np.ndarray]  # layer -> (seq_len, d_model)
    logits: Optional[np.ndarray] = None  # (seq_len, vocab_size)


@dataclass
class CommitmentCurve:
    """Probe prediction confidence at each token position for a target.
    
    This is the core data structure for measuring when a model "commits"
    to a future structure.
    
    Fields:
        example_id: Which example this curve is for.
        layer: Which layer the probe operates on.
        positions: Token positions (0-indexed in full sequence).
        confidences: Probe confidence (probability of correct target) at each position.
        target_value: What the probe is trying to predict.
        target_position: Where the target actually appears.
    """
    example_id: str
    layer: int
    positions: np.ndarray  # (n_positions,)
    confidences: np.ndarray  # (n_positions,) — P(correct_target) at each position
    target_value: str
    target_position: int  # first token position of the target


@dataclass
class CommitmentPoint:
    """Estimated commitment point — where the model "locks in" a plan.
    
    Computed from a CommitmentCurve by finding where confidence first
    exceeds a threshold and stays above it.
    """
    example_id: str
    layer: int
    position: int  # token position where commitment first exceeds threshold
    confidence_at_commitment: float
    threshold: float
    tokens_before_target: int  # how many tokens before the target appears
    is_valid: bool  # False if confidence never exceeds threshold


@dataclass
class PatchingResult:
    """Result of a causal patching experiment at a specific position/layer.
    
    For necessity: does patching at the commitment point change the output?
    For sufficiency: does injecting a commitment vector steer the output?
    """
    example_id: str
    patch_layer: int
    patch_position: int
    mode: str  # "necessity" or "sufficiency"
    
    # Before intervention
    original_target: str
    original_output: str
    original_target_prob: float
    
    # After intervention  
    intervened_output: str
    intervened_target_prob: float
    
    # Metrics
    prob_delta: float  # change in P(target)
    output_changed: bool  # did the generated output change?
    target_flipped: bool  # did the most-likely target change?


@dataclass 
class BaselineResult:
    """Result from a baseline/control condition.
    
    Controls we run:
    - shuffled_labels: Train probe on shuffled target labels
    - random_positions: Probe at random (non-commitment) positions
    - bag_of_words: Use bag-of-words features instead of activations
    - untrained_model: Use activations from a randomly initialized model
    """
    baseline_type: str
    metric_name: str
    metric_value: float
    confidence_interval: tuple[float, float]  # 95% CI via bootstrap
    n_samples: int
