"""Sweep functions for coarse activation patching."""

from __future__ import annotations

from ...common.device_utils import clear_gpu_memory
from ...inference.interventions.intervention_target import InterventionTarget
from ..patch_choice import patch_target
from ..act_patch_results import ActPatchTargetResult
from ...binary_choice import BinaryChoiceRunner
from ...common.contrastive_pair import ContrastivePair
from .coarse_results import SweepStepResults


def run_sanity_check(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    component: str,
    clean_max_pos: int,
    corrupted_max_pos: int,
) -> ActPatchTargetResult:
    """Run sanity check patching all layers, positions up to (not including) choice.

    Uses mode-specific positions:
    - Denoising (run corrupted): positions in corrupted space, up to corrupted_max_pos
    - Noising (run clean): positions in clean space, up to clean_max_pos
    """
    dn_positions = list(range(corrupted_max_pos))
    ns_positions = list(range(clean_max_pos))

    print(
        f"[coarse] Starting sanity check (all layers, dn_pos=0-{corrupted_max_pos - 1}, ns_pos=0-{clean_max_pos - 1})..."
    )

    dn_target = InterventionTarget.at_positions(dn_positions, component)
    ns_target = InterventionTarget.at_positions(ns_positions, component)

    result = patch_target(
        runner, pair, dn_target, denoising_target=dn_target, noising_target=ns_target
    )

    result.pop_heavy()
    clear_gpu_memory()

    print(f"[coarse] Sanity check done: {result.format_summary()}")
    return result


def run_layer_sweep(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    component: str,
    layers_of_interest: list[int],
    step_sizes: list[int],
    denoising_positions: list[int] | None = None,
    noising_positions: list[int] | None = None,
) -> dict[int, SweepStepResults]:
    """Run layer sweeps for all step sizes.

    Args:
        denoising_positions: Positions to patch for denoising (in corrupted space).
        noising_positions: Positions to patch for noising (in clean space).

    NOTE: For resid_post, patching all positions at any layer gives uniform
    results due to residual stream propagation. Use specific positions
    (e.g., divergent position) for meaningful layer attribution.
    """
    layer_results: dict[int, SweepStepResults] = {}

    dn_desc = f"dn_pos={denoising_positions}" if denoising_positions else "dn_pos=ALL"
    ns_desc = f"ns_pos={noising_positions}" if noising_positions else "ns_pos=ALL"

    for layer_step in step_sizes:
        layer_results[layer_step] = SweepStepResults()
        print(
            f"[coarse] Layer sweep (step={layer_step}, {dn_desc}, {ns_desc}) L{layers_of_interest[0]}-{layers_of_interest[-1]}..."
        )

        n_steps = -(-len(layers_of_interest) // layer_step)  # ceil division
        for i in range(0, len(layers_of_interest), layer_step):
            layer_range = layers_of_interest[i : i + layer_step]

            dn_target = InterventionTarget.at(
                positions=denoising_positions, layers=layer_range, component=component
            )
            ns_target = InterventionTarget.at(
                positions=noising_positions, layers=layer_range, component=component
            )

            result = patch_target(
                runner,
                pair,
                dn_target,
                denoising_target=dn_target,
                noising_target=ns_target,
            )
            layer_results[layer_step][layer_range[0]] = result
            result.pop_heavy()
            clear_gpu_memory()

            print(
                f"[coarse] Layers:{layer_range} {result.format_summary()}, {i // layer_step + 1}/{n_steps}"
            )

        clear_gpu_memory()

    return layer_results


def run_position_sweep(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    component: str,
    start_pos: int,
    clean_end_pos: int,
    corrupted_end_pos: int,
    step_sizes: list[int],
) -> dict[int, SweepStepResults]:
    """Run position sweeps for all step sizes.

    Uses mode-specific end positions:
    - Denoising (run corrupted): positions up to corrupted_end_pos
    - Noising (run clean): positions up to clean_end_pos
    """
    clean_div, corrupted_div = pair.choice_divergent_positions
    assert clean_end_pos <= clean_div, (
        f"clean_end_pos={clean_end_pos} must be <= {clean_div}"
    )
    assert corrupted_end_pos <= corrupted_div, (
        f"corrupted_end_pos={corrupted_end_pos} must be <= {corrupted_div}"
    )

    position_results: dict[int, SweepStepResults] = {}

    for pos_step in step_sizes:
        position_results[pos_step] = SweepStepResults()
        print(
            f"[coarse] Position sweep (step={pos_step}) dn=0-{corrupted_end_pos}, ns=0-{clean_end_pos}..."
        )

        # Iterate over the longer range, create mode-specific targets
        max_end = max(clean_end_pos, corrupted_end_pos)
        for pos in range(start_pos, max_end, pos_step):
            # Denoising positions (corrupted space)
            dn_end = min(pos + pos_step, corrupted_end_pos)
            dn_range = list(range(pos, dn_end)) if pos < corrupted_end_pos else []

            # Noising positions (clean space)
            ns_end = min(pos + pos_step, clean_end_pos)
            ns_range = list(range(pos, ns_end)) if pos < clean_end_pos else []

            if not dn_range and not ns_range:
                continue

            # Create targets for modes that have valid position ranges
            dn_target = (
                InterventionTarget.at_positions(dn_range, component)
                if dn_range
                else None
            )
            ns_target = (
                InterventionTarget.at_positions(ns_range, component)
                if ns_range
                else None
            )

            # Use whichever target exists as the default
            default_target = dn_target or ns_target
            result = patch_target(
                runner,
                pair,
                default_target,
                denoising_target=dn_target,
                noising_target=ns_target,
                skip_denoising=not dn_range,
                skip_noising=not ns_range,
            )
            position_results[pos_step][pos] = result

            result.pop_heavy()
            clear_gpu_memory()

            print(f"[coarse] pos={pos} {result.format_summary()}")

    return position_results
