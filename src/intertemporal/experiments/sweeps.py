"""Layer and position sweep utilities for activation patching.

Provides functions to identify which layers and positions are most important
for temporal preference decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ...common import BaseSchema
from ...common.device_utils import clear_gpu_memory
from ...activation_patching import ActivationPatchingTarget

if TYPE_CHECKING:
    from ..preference import PreferenceDataset
    from .activation_patching import (
        IntertemporalActivationPatchingConfig,
        AggregatedActivationPatchingResult,
    )


@dataclass
class SweepResults(BaseSchema):
    """Results from layer and position sweeps."""

    layer_recovery: dict[int, float] = field(default_factory=dict)
    position_recovery: dict[int, float] = field(default_factory=dict)
    best_layers: list[int] = field(default_factory=list)
    best_positions: list[int] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)

    def get_top_layers(self, n: int = 3) -> list[int]:
        """Get top N layers by recovery."""
        sorted_layers = sorted(
            self.layer_recovery.items(), key=lambda x: x[1], reverse=True
        )
        return [layer for layer, _ in sorted_layers[:n]]

    def get_top_positions(self, n: int = 5) -> list[int]:
        """Get top N positions by recovery."""
        sorted_pos = sorted(
            self.position_recovery.items(), key=lambda x: x[1], reverse=True
        )
        return [pos for pos, _ in sorted_pos[:n]]


def run_layer_sweep(
    pref_dataset: "PreferenceDataset",
    n_pairs: int = 5,
    component: str = "resid_post",
    mode: Literal["noising", "denoising"] = "denoising",
) -> tuple[dict[int, float], "AggregatedActivationPatchingResult"]:
    """Sweep all layers to find most important ones.

    Patches each layer separately with all positions to measure
    which layers have the most causal effect on the choice.

    Args:
        pref_dataset: PreferenceDataset with samples
        n_pairs: Number of contrastive pairs to use
        component: Component to patch (e.g., "resid_post")
        mode: "denoising" or "noising"

    Returns:
        Tuple of (layer -> recovery dict, full aggregated result)
    """
    from .activation_patching import (
        run_activation_patching,
        IntertemporalActivationPatchingConfig,
        DualModeActivationPatchingResult,
    )

    print("\n" + "=" * 60)
    print("LAYER SWEEP PHASE")
    print("=" * 60)

    cfg = IntertemporalActivationPatchingConfig(
        n_pairs=n_pairs,
        target=ActivationPatchingTarget(
            position_mode="all",
            layers="each",
            component=component,
        ),
        mode=mode,
        verify_with_greedy=False,
    )

    result = run_activation_patching(pref_dataset, cfg)
    result = _unwrap_dual_mode_result(result, mode)

    layer_recovery = result.get_recovery_by_layer()
    _print_layer_results(layer_recovery)

    return layer_recovery, result


def run_progressive_position_patching(
    pref_dataset: "PreferenceDataset",
    layer: int,
    sorted_positions: list[int],
    n_pairs: int = 5,
    component: str = "resid_post",
    mode: Literal["noising", "denoising"] = "denoising",
    max_positions: int = 50,
    step: int = 5,
) -> dict[int, float]:
    """Progressively add positions to find minimum needed for flip.

    Starts with `step` positions, then 2*step, 3*step, etc. until
    high recovery is achieved or max_positions is reached.

    Args:
        pref_dataset: PreferenceDataset with samples
        layer: Layer to patch
        sorted_positions: Positions sorted by importance (most important first)
        n_pairs: Number of contrastive pairs
        component: Component to patch
        mode: Patching mode
        max_positions: Maximum positions to try
        step: Increment step size

    Returns:
        Dict mapping n_positions -> mean_recovery
    """
    from .activation_patching import (
        run_activation_patching,
        IntertemporalActivationPatchingConfig,
        DualModeActivationPatchingResult,
    )

    print(f"\n{'=' * 60}")
    print(f"PROGRESSIVE POSITION PATCHING at L{layer}")
    print("=" * 60)

    results = {}
    max_to_try = min(max_positions, len(sorted_positions)) + 1

    for n_pos in range(step, max_to_try, step):
        positions = sorted_positions[:n_pos]

        cfg = IntertemporalActivationPatchingConfig(
            n_pairs=n_pairs,
            target=ActivationPatchingTarget(
                position_mode="explicit",
                token_positions=positions,
                layers=[layer],
                component=component,
            ),
            mode=mode,
            verify_with_greedy=False,
        )

        result = run_activation_patching(pref_dataset, cfg)
        result = _unwrap_dual_mode_result(result, mode)

        results[n_pos] = result.mean_recovery
        print(f"  Top {n_pos} positions: recovery={result.mean_recovery:.4f}, flips={result.flip_rate:.1%}")

        clear_gpu_memory()

        if result.mean_recovery > 0.9 and result.flip_rate > 0.5:
            print(f"  -> Achieved high recovery with {n_pos} positions!")
            break

    return results


def run_full_sweep(
    pref_dataset: "PreferenceDataset",
    n_pairs: int = 5,
    n_top_layers: int = 3,
    component: str = "resid_post",
    mode: Literal["noising", "denoising"] = "denoising",
) -> SweepResults:
    """Run complete layer + progressive position sweep.

    1. Sweeps all layers to find top performers
    2. Uses progressive position patching at best layer to find
       minimum positions needed for flip

    Args:
        pref_dataset: PreferenceDataset with samples
        n_pairs: Number of contrastive pairs
        n_top_layers: Number of top layers to track
        component: Component to patch
        mode: Patching mode

    Returns:
        SweepResults with layer/position recovery data and best candidates
    """
    # Phase 1: Layer sweep
    layer_recovery, _ = run_layer_sweep(pref_dataset, n_pairs, component, mode)
    clear_gpu_memory()

    # Get best layers
    best_layers = _get_top_layers(layer_recovery, n_top_layers)
    if not best_layers:
        print("WARNING: No valid layers found in sweep")
        return SweepResults(layer_recovery=layer_recovery)

    # Phase 2: Progressive position patching at best layer
    best_layer = best_layers[0]
    seq_len = 220  # Typical prompt + response length
    all_positions = list(range(seq_len))

    position_recovery = run_progressive_position_patching(
        pref_dataset,
        layer=best_layer,
        sorted_positions=all_positions,
        n_pairs=n_pairs,
        component=component,
        mode=mode,
        max_positions=100,
        step=10,
    )
    clear_gpu_memory()

    # Find minimum positions for good recovery
    best_n_pos = _find_min_positions_for_recovery(position_recovery, threshold=0.8)

    return SweepResults(
        layer_recovery=layer_recovery,
        position_recovery=position_recovery,
        best_layers=best_layers,
        best_positions=list(range(best_n_pos)),
    )


# ============================================================================
# Helper functions
# ============================================================================


def _unwrap_dual_mode_result(result, mode: str):
    """Extract single-mode result from DualModeActivationPatchingResult."""
    from .activation_patching import DualModeActivationPatchingResult

    if isinstance(result, DualModeActivationPatchingResult):
        if mode == "denoising" and result.denoising:
            return result.denoising
        elif mode == "noising" and result.noising:
            return result.noising
        return result.denoising or result.noising
    return result


def _get_top_layers(layer_recovery: dict[int, float], n: int) -> list[int]:
    """Get top N layers by recovery, filtering None keys."""
    sorted_layers = sorted(
        layer_recovery.items(), key=lambda x: x[1], reverse=True
    )
    return [layer for layer, _ in sorted_layers[:n] if layer is not None]


def _find_min_positions_for_recovery(
    position_recovery: dict[int, float],
    threshold: float = 0.8,
) -> int:
    """Find minimum number of positions that achieve threshold recovery."""
    if not position_recovery:
        return 0

    for n_pos, recovery in sorted(position_recovery.items()):
        if recovery > threshold:
            return n_pos

    return max(position_recovery.keys())


def _print_layer_results(layer_recovery: dict[int, float]) -> None:
    """Print layer sweep results."""
    print("\nLayer sweep results:")
    for layer, recovery in sorted(layer_recovery.items()):
        print(f"  L{layer}: {recovery:.4f}")
