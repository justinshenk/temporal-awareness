"""Activation post-processing for selected-node caches."""

from __future__ import annotations

from typing import Any

import torch

try:
    from .cache_activations_nodes import SelectedNodeGroups
except ImportError:
    from cache_activations_nodes import SelectedNodeGroups


def extract_selected_activations(
    activations: Any,
    selected_node_groups: SelectedNodeGroups,
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract only selected neurons/attention heads from an ActivationDict."""
    if any(
        component == "z"
        for group_nodes in selected_node_groups.values()
        for (_, component), _ in group_nodes
    ):
        activations = activations.split_heads()

    selected: dict[str, dict[str, torch.Tensor]] = {}
    for group_name, group_nodes in selected_node_groups.items():
        selected[group_name] = {}
        for (layer, component), node_index in group_nodes:
            activation = activations[(layer, component)]
            if component == "z":
                if activation.ndim == 4:
                    node_activation = activation[:, :, node_index, :]
                else:
                    node_activation = activation[:, node_index, :]
            else:
                if activation.ndim == 3:
                    node_activation = activation[:, :, node_index]
                else:
                    node_activation = activation[:, node_index]

            selected[group_name][f"{component}/{layer}__{node_index}"] = (
                node_activation.detach().cpu()
            )

    return selected


def maybe_average_positions(activations: Any, average_positions: bool) -> Any:
    """Optionally average activations across valid token positions."""
    if not average_positions:
        return activations
    return activations.apply(torch.nanmean, dim=1, mask_aware=True)
