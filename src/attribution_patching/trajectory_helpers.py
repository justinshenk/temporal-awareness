"""Trajectory helper functions for attribution patching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from ..common.contrastive_pair import ContrastivePair

if TYPE_CHECKING:
    from ..binary_choice import BinaryChoiceRunner


def _get_trajectory(
    pair: ContrastivePair,
    which: Literal["clean", "corrupted"],
    mode: Literal["denoising", "noising"],
):
    """Get the appropriate trajectory based on mode and which.

    In denoising mode: clean=long, corrupted=short
    In noising mode: clean=short, corrupted=long
    """
    if mode == "denoising":
        return pair.long_traj if which == "clean" else pair.short_traj
    else:
        return pair.short_traj if which == "clean" else pair.long_traj


def get_cache(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    which: Literal["clean", "corrupted"],
    mode: Literal["denoising", "noising"],
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
        new_traj = runner.generate_trajectory_with_cache_and_grad(
            traj.token_ids, names_filter
        )
    else:
        with torch.no_grad():
            new_traj = runner.generate_trajectory_with_cache(
                traj.token_ids, names_filter
            )
    return new_traj.full_logits, new_traj.internals


def get_seq_len(cache: dict, hook_name: str) -> int:
    """Get sequence length from cached activation."""
    act = cache[hook_name]
    return act.shape[1] if act.ndim == 3 else act.shape[0]


def get_caches_for_attribution(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    mode: Literal["denoising", "noising"],
    names_filter: callable | None = None,
    grad_at: Literal["clean", "corrupted"] = "corrupted",
) -> tuple[torch.Tensor, dict, dict, dict]:
    """Get clean and corrupted caches with gradients at specified point.

    Args:
        runner: Model runner
        pair: Contrastive pair with trajectories
        mode: "denoising" or "noising"
        names_filter: Hook filter for caching
        grad_at: Where to compute gradients ("clean" or "corrupted")

    Returns:
        (grad_logits, clean_cache, corr_cache, grad_cache) tuple
        grad_cache is a reference to either clean_cache or corr_cache
    """
    if grad_at == "corrupted":
        _, clean_cache = get_cache(runner, pair, "clean", mode, names_filter, with_grad=False)
        grad_logits, corr_cache = get_cache(runner, pair, "corrupted", mode, names_filter, with_grad=True)
        grad_cache = corr_cache
    else:  # grad_at == "clean"
        grad_logits, clean_cache = get_cache(runner, pair, "clean", mode, names_filter, with_grad=True)
        _, corr_cache = get_cache(runner, pair, "corrupted", mode, names_filter, with_grad=False)
        grad_cache = clean_cache

    return grad_logits, clean_cache, corr_cache, grad_cache
