"""Conversion between public API graph payloads and NetworkX graphs."""

from __future__ import annotations

from typing import Any

import networkx as nx

from system.api_models import GraphEdge, GraphNode, GraphPayload


def graph_from_payload(payload: GraphPayload, *, graph_id: str) -> nx.Graph:
    graph = nx.Graph(req_id=graph_id)
    for node in payload.nodes:
        attrs = node.model_dump(exclude={"id"})
        graph.add_node(node.id, **attrs)
    for edge in payload.edges:
        graph.add_edge(edge.u, edge.v, bandwidth=edge.bandwidth)
    return graph


def graph_to_payload(graph: nx.Graph) -> GraphPayload:
    nodes = []
    for node_id, attrs in sorted(graph.nodes(data=True)):
        nodes.append(
            GraphNode(
                id=str(node_id),
                cpu=int(attrs.get("cpu", 0)),
                memory_gib=int(attrs.get("memory_gib", 0)),
                arch=attrs.get("arch", 0),
                public_endpoint=bool(attrs.get("public_endpoint", False)),
            )
        )

    edges = []
    for u, v, attrs in sorted(
        graph.edges(data=True),
        key=lambda item: (min(str(item[0]), str(item[1])), max(str(item[0]), str(item[1]))),
    ):
        edge_u, edge_v = sorted((str(u), str(v)))
        edges.append(
            GraphEdge(
                u=edge_u,
                v=edge_v,
                bandwidth=int(attrs.get("bandwidth", 0)),
            )
        )
    return GraphPayload(nodes=nodes, edges=edges)


def graph_payload_from_dict(value: dict[str, Any]) -> GraphPayload:
    return GraphPayload.model_validate(value)
