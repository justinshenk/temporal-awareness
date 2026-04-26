"""Selected-node loading for activation caching."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

SelectedNode = tuple[tuple[int, str], int]
SelectedNodeGroups = dict[str, list[SelectedNode]]


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickle only primitive containers used by selected-node files."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")


def load_selected_node_groups(nodes_path: Path) -> SelectedNodeGroups:
    """Load selected nodes as group -> [((layer, component), node_index), ...]."""
    with nodes_path.open("rb") as f:
        raw_nodes = RestrictedUnpickler(f).load()

    if not isinstance(raw_nodes, dict):
        raise ValueError(f"Expected selected nodes to be a dict, got {type(raw_nodes)}")

    selected_node_groups: SelectedNodeGroups = {}
    for group_name, group_nodes in raw_nodes.items():
        if not isinstance(group_name, str):
            raise ValueError(
                f"Selected node group names must be strings: {group_name!r}"
            )

        selected_node_groups[group_name] = []
        for raw_node in sorted(group_nodes):
            if (
                not isinstance(raw_node, tuple)
                or len(raw_node) != 2
                or not isinstance(raw_node[0], str)
                or not isinstance(raw_node[1], int)
            ):
                raise ValueError(f"Invalid selected node entry: {raw_node!r}")

            component_layer, node_index = raw_node
            if "/" not in component_layer:
                raise ValueError(
                    f"Expected component/layer entry, got {component_layer!r}"
                )
            component, layer_text = component_layer.split("/", maxsplit=1)
            selected_node_groups[group_name].append(
                ((int(layer_text), component), node_index)
            )

    return selected_node_groups


def get_unique_layer_components(
    selected_node_groups: SelectedNodeGroups,
) -> list[tuple[int, str]]:
    """Return sorted unique layer/component pairs required by selected nodes."""
    return sorted(
        {
            layer_component
            for group_nodes in selected_node_groups.values()
            for layer_component, _ in group_nodes
        }
    )
