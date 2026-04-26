"""Activation post-processing for selected-node caches."""

from __future__ import annotations

from typing import Any

import torch

try:
    from .cache_activations_nodes import SelectedNodeGroups
except ImportError:
    from cache_activations_nodes import SelectedNodeGroups


def group_node_indices(
    selected_node_groups: SelectedNodeGroups,
) -> dict[tuple[int, str], list[int]]:
    """Return unique selected node indices grouped by layer/component."""
    grouped: dict[tuple[int, str], set[int]] = {}
    for group_nodes in selected_node_groups.values():
        for layer_component, node_index in group_nodes:
            grouped.setdefault(layer_component, set()).add(node_index)

    return {
        layer_component: sorted(node_indices)
        for layer_component, node_indices in grouped.items()
    }


def extract_selected_activations(
    activations: Any,
    selected_node_groups: SelectedNodeGroups,
) -> dict[str, dict[str, list[int] | torch.Tensor]]:
    """Extract selected nodes once per layer/component from an ActivationDict."""
    if any(
        component == "z"
        for group_nodes in selected_node_groups.values()
        for (_, component), _ in group_nodes
    ):
        activations = activations.split_heads()

    selected: dict[str, dict[str, list[int] | torch.Tensor]] = {}
    for (layer, component), node_indices in group_node_indices(
        selected_node_groups
    ).items():
        activation = activations[(layer, component)]
        index = torch.as_tensor(node_indices, device=activation.device)
        if component == "z":
            node_dim = 2 if activation.ndim == 4 else 1
        else:
            node_dim = 2 if activation.ndim == 3 else 1

        selected[f"{component}/{layer}"] = {
            "node_indices": node_indices,
            "values": activation.index_select(node_dim, index).detach().cpu(),
        }
    return selected


def maybe_average_positions(activations: Any, average_positions: bool) -> Any:
    """Optionally average activations across valid token positions."""
    if not average_positions:
        return activations
    return activations.apply(torch.nanmean, dim=1, mask_aware=True)
