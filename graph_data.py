"""Toy graph data for the resource-feasibility ZK demo."""

from __future__ import annotations

import networkx as nx


def build_toy_vnr() -> nx.Graph:
    """Build the public three-node VNR."""
    graph = nx.Graph(req_id="toy-chain-3")

    graph.add_node("v0", cpu=30)
    graph.add_node("v1", cpu=50)
    graph.add_node("v2", cpu=100)

    graph.add_edge("v0", "v1", bandwidth=10)
    graph.add_edge("v1", "v2", bandwidth=5)

    return graph


def build_toy_substrate() -> nx.Graph:
    """Build the private four-node substrate resource state."""
    graph = nx.Graph(facility_id="facility-A")

    graph.add_node("s0", cpu=100)
    graph.add_node("s1", cpu=120)
    graph.add_node("s2", cpu=100)
    graph.add_node("s3", cpu=200)

    graph.add_edge("s0", "s2", bandwidth=20)
    graph.add_edge("s1", "s2", bandwidth=15)
    graph.add_edge("s1", "s3", bandwidth=20)
    graph.add_edge("s2", "s3", bandwidth=20)

    return graph
