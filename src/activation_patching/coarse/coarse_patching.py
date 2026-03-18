"""Main orchestration for coarse activation patching."""

from __future__ import annotations

from ...common import profile
from ...common.device_utils import clear_gpu_memory
from ...binary_choice import BinaryChoiceRunner
from ...common.contrastive_pair import ContrastivePair
from .coarse_results import CoarseActPatchResults
from .sweep_runners import run_sanity_check, run_layer_sweep, run_position_sweep


@profile
def run_coarse_act_patching(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    min_layer_depth: float = 0.0,
    max_layer_depth: float = 1.0,
    component: str | None = None,
    layer_step_sizes: list[int] | None = None,
    pos_step_sizes: list[int] | None = None,
) -> CoarseActPatchResults:
    """Run sanity check, layer sweep, and position sweep on single pair.

    Args:
        runner: BinaryChoiceRunner for inference
        pair: ContrastivePair to patch
        min_layer_depth: Start layer as fraction of total layers
        max_layer_depth: End layer as fraction of total layers
        component: Component to patch
        layer_step_sizes: List of step sizes for layer sweeps
        pos_step_sizes: List of step sizes for position sweeps

    Returns:
        CoarseActPatchResults with results organized by step size
    """
    # Defaults
    if component is None:
        component = "resid_post"
    if layer_step_sizes is None:
        layer_step_sizes = [1]
    if pos_step_sizes is None:
        pos_step_sizes = [1]

    pair.print_position_mapping_debug("[coarse]")

    # Position where A/B choice tokens diverge - patching must be before this
    # Denoising runs corrupted, noising runs clean - use respective positions
    clean_div_pos, corrupted_div_pos = pair.choice_divergent_positions
    assert clean_div_pos > 0 and corrupted_div_pos > 0, (
        "Divergent positions must be > 0"
    )

    sanity_result = run_sanity_check(
        runner,
        pair,
        component,
        clean_max_pos=clean_div_pos,
        corrupted_max_pos=corrupted_div_pos,
    )

    # Compute layer range
    n_layers = runner.n_layers
    start_layer = int(n_layers * min_layer_depth)
    end_layer = int(n_layers * max_layer_depth)
    layers_of_interest = list(range(start_layer, end_layer))

    # For layer sweep, patch only the position just before divergence for meaningful attribution.
    # Patching all positions at any layer gives uniform results due to residual propagation.
    # Use mode-specific positions: denoising uses corrupted space, noising uses clean space.
    layer_results = run_layer_sweep(
        runner,
        pair,
        component,
        layers_of_interest,
        layer_step_sizes,
        denoising_positions=[corrupted_div_pos - 1],
        noising_positions=[clean_div_pos - 1],
    )

    # Compute position range for position sweep
    start_pos = pair.position_mapping.first_interesting_pos

    position_results = run_position_sweep(
        runner,
        pair,
        component,
        start_pos,
        clean_end_pos=clean_div_pos,
        corrupted_end_pos=corrupted_div_pos,
        step_sizes=pos_step_sizes,
    )

    clear_gpu_memory()
    print("[coarse] Done.")

    return CoarseActPatchResults(
        sanity_result=sanity_result,
        layer_results=layer_results,
        position_results=position_results,
    )
