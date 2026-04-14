"""Trajectory helper functions for attribution patching."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..common.contrastive_pair import ContrastivePair
from ..common.patching_types import PatchingMode, TrajectoryType

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner


def _get_trajectory(
    pair: ContrastivePair,
    which: TrajectoryType,
    mode: PatchingMode,
):
    """Get the appropriate trajectory based on mode and which.

    In denoising mode: we patch from corrupted -> clean (inject clean into corrupted)
    In noising mode: we patch from clean -> corrupted (inject corrupted into clean)
    """
    if mode == "denoising":
        return pair.corrupted_traj if which == "clean" else pair.clean_traj
    else:
        return pair.clean_traj if which == "clean" else pair.corrupted_traj


def get_cache(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    which: TrajectoryType,
    mode: PatchingMode,
    names_filter: callable | None = None,
    with_grad: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Get trajectory cache with optional gradients.

    Args:
        runner: Model runner
        pair: Contrastive pair with trajectories
        which: "clean" or "corrupted" trajectory
        mode: "denoising" or "noising" determines which traj is which
        names_filter: Hook filter for caching
        with_grad: Whether to enable gradients

    Returns:
        (logits, internals_cache) tuple
    """
    traj = _get_trajectory(pair, which, mode)

    if with_grad:
        new_traj = runner.compute_trajectory_with_cache_and_grad(
            traj.token_ids, names_filter
        )
    else:
        with torch.no_grad():
            new_traj = runner.compute_trajectory_with_cache(
                traj.token_ids, names_filter
            )
    return new_traj.full_logits, new_traj.internals


def get_seq_len(cache: dict, hook_name: str) -> int:
    """Get sequence length from cached activation."""
    act = cache[hook_name]
    return act.shape[1] if act.ndim == 3 else act.shape[0]


def get_all_caches(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    mode: PatchingMode,
    names_filter: callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """Get clean and corrupted caches, both with gradients enabled.

    This allows reusing the same caches for both denoising and noising modes.

    Args:
        runner: Model runner
        pair: Contrastive pair with trajectories
        mode: "denoising" or "noising"
        names_filter: Hook filter for caching

    Returns:
        (clean_logits, corr_logits, clean_cache, corr_cache) tuple
        Both caches have requires_grad=True for gradient computation.
    """
    clean_logits, clean_cache = get_cache(
        runner, pair, "clean", mode, names_filter, with_grad=True
    )
    corr_logits, corr_cache = get_cache(
        runner, pair, "corrupted", mode, names_filter, with_grad=True
    )

    return clean_logits, corr_logits, clean_cache, corr_cache
