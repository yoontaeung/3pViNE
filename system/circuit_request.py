"""Build the public fixed-size VNR that a facility proves locally."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from zk_encoding import V_NODE_COUNT


def build_local_circuit_vnr(
    original: nx.Graph,
    selected_nodes: Iterable[str],
    *,
    endpoint_demands: Iterable[tuple[str, int]] = (),
) -> nx.Graph:
    """Pad selected nodes and optional endpoint demands into three VNR slots."""
    selected = sorted(selected_nodes)
    if len(selected) != len(set(selected)):
        raise ValueError("local VNR contains a duplicate selected node")
    demands = list(endpoint_demands)
    endpoint_enabled = bool(demands)
    node_capacity = V_NODE_COUNT - 1 if endpoint_enabled else V_NODE_COUNT
    if len(selected) > node_capacity:
        raise ValueError(
            f"local VNR needs {len(selected)} slots, but only {node_capacity} are available"
        )

    local = nx.Graph(req_id=original.graph.get("req_id", "request"))
    original_to_slot: dict[str, str] = {}
    for slot_index in range(V_NODE_COUNT):
        slot = f"v{slot_index}"
        if endpoint_enabled and slot_index == V_NODE_COUNT - 1:
            local.add_node(
                slot,
                cpu=0,
                memory_gib=0,
                arch=0,
                present=1,
                public_endpoint=True,
            )
        elif slot_index < len(selected):
            original_node = selected[slot_index]
            original_to_slot[original_node] = slot
            attributes = dict(original.nodes[original_node])
            attributes["present"] = 1
            local.add_node(slot, **attributes)
        else:
            local.add_node(
                slot,
                cpu=0,
                memory_gib=0,
                arch=0,
                present=0,
                public_endpoint=False,
            )

    selected_set = set(selected)
    for source, target, attributes in original.edges(data=True):
        if source in selected_set and target in selected_set:
            local.add_edge(
                original_to_slot[source],
                original_to_slot[target],
                **dict(attributes),
            )

    if endpoint_enabled:
        endpoint_slot = f"v{V_NODE_COUNT - 1}"
        bandwidth_by_node: dict[str, int] = {}
        for virtual_node, bandwidth in demands:
            if virtual_node not in original_to_slot:
                raise ValueError(
                    f"endpoint demand references unselected node {virtual_node}"
                )
            bandwidth_by_node[virtual_node] = (
                bandwidth_by_node.get(virtual_node, 0) + int(bandwidth)
            )
        for virtual_node, bandwidth in bandwidth_by_node.items():
            local.add_edge(
                original_to_slot[virtual_node],
                endpoint_slot,
                bandwidth=bandwidth,
            )

    return local
