"""Reference model for single-cut, multi-facility VNE composition.

This module models the protocol semantics around partial embedding. The first
partial ZK circuit proves local endpoint attachment, while cross-facility
connectivity is public endpoint metadata checked by the portal.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import networkx as nx

from local_solver import LocalEmbedding, SolverPolicy, solve_local_embedding
from zk_encoding import V_NODE_COUNT


@dataclass(frozen=True)
class PublicEndpoint:
    """A public attachment point exposed by a facility."""

    endpoint_id: str
    facility_id: str
    substrate_node: str


@dataclass(frozen=True)
class EndpointAttachmentDemand:
    """A demand from a virtual node to a facility public endpoint."""

    virtual_node: str
    peer_id: str
    bandwidth: int


@dataclass(frozen=True)
class BoundaryDemand:
    """A cut edge that leaves a partial embedding through a public endpoint."""

    source_facility_id: str
    source_endpoint_id: str
    embedded_virtual_node: str
    residual_virtual_node: str
    bandwidth: int


@dataclass(frozen=True)
class ResidualRequest:
    """Residual VNR plus endpoint requirements produced by a partial embedding."""

    request_id: int
    residual_vnr: nx.Graph
    embedded_virtual_nodes: tuple[str, ...]
    boundary_demands: tuple[BoundaryDemand, ...]


@dataclass(frozen=True)
class FullEmbeddingClaim:
    """Full local embedding claim encoded for the fixed-size resource circuit."""

    status: Literal["full"]
    local_vnr: nx.Graph
    embedding: LocalEmbedding
    original_to_slot: dict[str, str]
    slot_to_original: dict[str, str]
    endpoint_virtual_node: str | None = None
    endpoint_substrate_node: str | None = None


@dataclass(frozen=True)
class ConnectedPartialClaim:
    """Connected partial embedding claim using the full circuit on an augmented VNR."""

    status: Literal["partial"]
    embedded_virtual_nodes: tuple[str, ...]
    residual: ResidualRequest
    local_vnr: nx.Graph
    embedding: LocalEmbedding
    original_to_slot: dict[str, str]
    slot_to_original: dict[str, str]
    endpoint_virtual_node: str
    endpoint_substrate_node: str


@dataclass(frozen=True)
class DeclineClaim:
    """Unproven local decline."""

    status: Literal["decline"]
    reason: str


ConnectedSolverClaim = FullEmbeddingClaim | ConnectedPartialClaim | DeclineClaim


@dataclass(frozen=True)
class PublicConnectivity:
    """Publicly checkable connectivity between two public endpoints."""

    endpoint_a: str
    endpoint_b: str
    bandwidth: int
    latency_ms: int | None = None


class PublicConnectivityOracle:
    """Small public endpoint-to-endpoint connectivity table."""

    def __init__(self, links: list[PublicConnectivity]) -> None:
        self._bandwidth: dict[frozenset[str], int] = {}
        for link in links:
            key = frozenset((link.endpoint_a, link.endpoint_b))
            self._bandwidth[key] = max(self._bandwidth.get(key, 0), link.bandwidth)

    def can_satisfy(self, endpoint_a: str, endpoint_b: str, bandwidth: int) -> bool:
        key = frozenset((endpoint_a, endpoint_b))
        return key in self._bandwidth and self._bandwidth[key] >= bandwidth


def solve_connected_vne_claim(
    vnr: nx.Graph,
    substrate: nx.Graph,
    *,
    facility_id: str,
    endpoint: PublicEndpoint | None,
    policy: SolverPolicy | None = None,
    incoming_boundary_demands: tuple[BoundaryDemand, ...] = (),
) -> ConnectedSolverClaim:
    """Try full embedding first, then one connected single-cut partial split."""
    policy = policy or SolverPolicy()
    if len(vnr.nodes) == 0 or not nx.is_connected(vnr):
        return DeclineClaim(status="decline", reason="vnr_not_connected")
    if len(incoming_boundary_demands) > 1:
        return DeclineClaim(status="decline", reason="multiple_crossing_edges_not_supported")
    if incoming_boundary_demands and endpoint is None:
        return DeclineClaim(status="decline", reason="partial_endpoint_unavailable")

    if len(vnr.nodes) <= _local_node_capacity(incoming_boundary_demands):
        endpoint_virtual_node = f"v{V_NODE_COUNT - 1}" if incoming_boundary_demands else None
        endpoint_demands = _incoming_endpoint_demands(incoming_boundary_demands)
        local_vnr, original_to_slot, slot_to_original = build_circuit_vnr_for_nodes(
            vnr,
            sorted(vnr.nodes),
            endpoint_virtual_node=endpoint_virtual_node,
            endpoint_demands=endpoint_demands,
        )
        fixed_mapping = (
            {endpoint_virtual_node: endpoint.substrate_node}
            if endpoint_virtual_node is not None
            else None
        )
        full_embedding = solve_local_embedding(
            local_vnr,
            substrate,
            policy=policy,
            fixed_node_mapping=fixed_mapping,
        )
        if full_embedding is not None:
            return FullEmbeddingClaim(
                status="full",
                local_vnr=local_vnr,
                embedding=full_embedding,
                original_to_slot=original_to_slot,
                slot_to_original=slot_to_original,
                endpoint_virtual_node=endpoint_virtual_node,
                endpoint_substrate_node=endpoint.substrate_node if endpoint_virtual_node else None,
            )

    # A residual request must be completed by the receiving facility. Allowing
    # another partial claim would split one original VNR more than once.
    if incoming_boundary_demands:
        return DeclineClaim(
            status="decline",
            reason="residual_full_embedding_infeasible",
        )
    if endpoint is None:
        return DeclineClaim(status="decline", reason="partial_endpoint_unavailable")

    best_rank: tuple[int, int, int, tuple[str, ...]] | None = None
    best_claim: ConnectedPartialClaim | None = None
    original_nodes = sorted(vnr.nodes)
    for subset_size in range(1, min(len(original_nodes), V_NODE_COUNT)):
        for subset in combinations(original_nodes, subset_size):
            embedded_nodes = set(subset)
            residual_nodes = set(original_nodes) - embedded_nodes
            if not _nodes_connected(vnr, embedded_nodes):
                continue
            if not _nodes_connected(vnr, residual_nodes):
                continue
            if not _incoming_demands_are_covered(incoming_boundary_demands, embedded_nodes):
                continue

            residual = derive_residual_request(
                vnr,
                request_id=_numeric_request_id(vnr),
                embedded_virtual_nodes=embedded_nodes,
                source_facility_id=facility_id,
                source_endpoint_id=endpoint.endpoint_id,
            )
            if len(residual.boundary_demands) != 1:
                continue
            local_vnr, original_to_slot, slot_to_original = build_circuit_vnr_for_nodes(
                vnr,
                sorted(embedded_nodes),
                endpoint_virtual_node=f"v{V_NODE_COUNT - 1}",
                boundary_demands=residual.boundary_demands,
                endpoint_demands=_incoming_endpoint_demands(incoming_boundary_demands),
            )
            fixed_mapping = {f"v{V_NODE_COUNT - 1}": endpoint.substrate_node}
            embedding = solve_local_embedding(
                local_vnr,
                substrate,
                policy=policy,
                fixed_node_mapping=fixed_mapping,
            )
            if embedding is None:
                continue

            path_cost = sum(len(path) - 1 for path in embedding.paths.values())
            rank = (-len(embedded_nodes), len(residual.boundary_demands), path_cost, tuple(subset))
            claim = ConnectedPartialClaim(
                status="partial",
                embedded_virtual_nodes=tuple(sorted(embedded_nodes)),
                residual=residual,
                local_vnr=local_vnr,
                embedding=embedding,
                original_to_slot=original_to_slot,
                slot_to_original=slot_to_original,
                endpoint_virtual_node=f"v{V_NODE_COUNT - 1}",
                endpoint_substrate_node=endpoint.substrate_node,
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_claim = claim

    return best_claim or DeclineClaim(status="decline", reason="no_connected_partial_embedding")


def build_circuit_vnr_for_nodes(
    original_vnr: nx.Graph,
    selected_nodes: list[str],
    *,
    endpoint_virtual_node: str | None = None,
    boundary_demands: tuple[BoundaryDemand, ...] = (),
    endpoint_demands: tuple[EndpointAttachmentDemand, ...] = (),
) -> tuple[nx.Graph, dict[str, str], dict[str, str]]:
    """Pad a selected VNR node set into the fixed three-slot circuit request."""
    if endpoint_virtual_node is None:
        real_slot_count = V_NODE_COUNT
    else:
        real_slot_count = V_NODE_COUNT - 1
        if endpoint_virtual_node != f"v{V_NODE_COUNT - 1}":
            raise ValueError(f"endpoint virtual node must be v{V_NODE_COUNT - 1}")
    if len(selected_nodes) > real_slot_count:
        raise ValueError("selected VNR node set exceeds fixed circuit capacity")

    graph = nx.Graph(req_id=original_vnr.graph.get("req_id", "local"))
    original_to_slot: dict[str, str] = {}
    slot_to_original: dict[str, str] = {}

    for slot_idx in range(V_NODE_COUNT):
        slot = f"v{slot_idx}"
        if endpoint_virtual_node is not None and slot == endpoint_virtual_node:
            graph.add_node(
                slot,
                cpu=0,
                memory_gib=0,
                arch=0,
                present=1,
                public_endpoint=True,
            )
            continue
        if slot_idx < len(selected_nodes):
            original = selected_nodes[slot_idx]
            original_to_slot[original] = slot
            slot_to_original[slot] = original
            attrs = dict(original_vnr.nodes[original])
            attrs["present"] = 1
            graph.add_node(slot, **attrs)
        else:
            graph.add_node(slot, cpu=0, memory_gib=0, arch=0, present=0)

    selected = set(selected_nodes)
    for u, v, attrs in original_vnr.edges(data=True):
        if u in selected and v in selected:
            graph.add_edge(original_to_slot[u], original_to_slot[v], **dict(attrs))

    if endpoint_virtual_node is not None:
        bw_by_embedded_node: dict[str, int] = {}
        for demand in boundary_demands:
            bw_by_embedded_node[demand.embedded_virtual_node] = (
                bw_by_embedded_node.get(demand.embedded_virtual_node, 0) + demand.bandwidth
            )
        for demand in endpoint_demands:
            bw_by_embedded_node[demand.virtual_node] = (
                bw_by_embedded_node.get(demand.virtual_node, 0) + demand.bandwidth
            )
        for original, bandwidth in bw_by_embedded_node.items():
            if original not in original_to_slot:
                raise ValueError(f"boundary demand references unselected node: {original}")
            graph.add_edge(original_to_slot[original], endpoint_virtual_node, bandwidth=bandwidth)

    return graph, original_to_slot, slot_to_original


def _nodes_connected(graph: nx.Graph, nodes: set[str]) -> bool:
    if not nodes:
        return False
    if len(nodes) == 1:
        return True
    return nx.is_connected(graph.subgraph(nodes))


def _numeric_request_id(graph: nx.Graph) -> int:
    value = graph.graph.get("req_id", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _local_node_capacity(incoming_boundary_demands: tuple[BoundaryDemand, ...]) -> int:
    return V_NODE_COUNT - 1 if incoming_boundary_demands else V_NODE_COUNT


def _incoming_endpoint_demands(
    incoming_boundary_demands: tuple[BoundaryDemand, ...],
) -> tuple[EndpointAttachmentDemand, ...]:
    return tuple(
        EndpointAttachmentDemand(
            virtual_node=demand.residual_virtual_node,
            peer_id=demand.source_endpoint_id,
            bandwidth=demand.bandwidth,
        )
        for demand in incoming_boundary_demands
    )


def _incoming_demands_are_covered(
    incoming_boundary_demands: tuple[BoundaryDemand, ...],
    embedded_nodes: set[str],
) -> bool:
    return all(demand.residual_virtual_node in embedded_nodes for demand in incoming_boundary_demands)


def graph_from_config(config: dict[str, Any], *, graph_id: str) -> nx.Graph:
    """Build a graph while preserving protocol-relevant node/edge attributes."""
    graph = nx.Graph(req_id=graph_id)
    for node in config["nodes"]:
        attrs = {key: value for key, value in node.items() if key != "id"}
        if "cpu" in attrs:
            attrs["cpu"] = int(attrs["cpu"])
        if "memory_gib" in attrs:
            attrs["memory_gib"] = int(attrs["memory_gib"])
        graph.add_node(node["id"], **attrs)
    for edge in config["edges"]:
        attrs = {key: value for key, value in edge.items() if key not in {"u", "v"}}
        if "bandwidth" in attrs:
            attrs["bandwidth"] = int(attrs["bandwidth"])
        graph.add_edge(edge["u"], edge["v"], **attrs)
    return graph


def derive_residual_request(
    original_vnr: nx.Graph,
    *,
    request_id: int,
    embedded_virtual_nodes: set[str],
    source_facility_id: str,
    source_endpoint_id: str,
) -> ResidualRequest:
    """Derive residual nodes and endpoint demands from a partial embedding."""
    original_nodes = set(original_vnr.nodes)
    if not embedded_virtual_nodes:
        raise ValueError("partial embedding must cover at least one virtual node")
    if not embedded_virtual_nodes < original_nodes:
        raise ValueError("partial embedding must leave at least one residual virtual node")

    residual_nodes = original_nodes - embedded_virtual_nodes
    residual_vnr = nx.Graph(req_id=f"{request_id}-residual")
    for node in sorted(residual_nodes):
        residual_vnr.add_node(node, **dict(original_vnr.nodes[node]))

    boundary_demands: list[BoundaryDemand] = []
    for u, v, attrs in original_vnr.edges(data=True):
        u_embedded = u in embedded_virtual_nodes
        v_embedded = v in embedded_virtual_nodes
        bandwidth = int(attrs["bandwidth"])
        if not u_embedded and not v_embedded:
            residual_vnr.add_edge(u, v, **dict(attrs))
        elif u_embedded != v_embedded:
            embedded_node = u if u_embedded else v
            residual_node = v if u_embedded else u
            boundary_demands.append(
                BoundaryDemand(
                    source_facility_id=source_facility_id,
                    source_endpoint_id=source_endpoint_id,
                    embedded_virtual_node=embedded_node,
                    residual_virtual_node=residual_node,
                    bandwidth=bandwidth,
                )
            )

    return ResidualRequest(
        request_id=request_id,
        residual_vnr=residual_vnr,
        embedded_virtual_nodes=tuple(sorted(embedded_virtual_nodes)),
        boundary_demands=tuple(
            sorted(
                boundary_demands,
                key=lambda demand: (
                    demand.embedded_virtual_node,
                    demand.residual_virtual_node,
                    demand.source_endpoint_id,
                ),
            )
        ),
    )


def verify_residual_consistency(
    original_vnr: nx.Graph,
    residual: ResidualRequest,
) -> tuple[bool, list[str]]:
    """Check that a residual VNR faithfully represents the unembedded request."""
    errors: list[str] = []
    original_nodes = set(original_vnr.nodes)
    embedded_nodes = set(residual.embedded_virtual_nodes)
    residual_nodes = set(residual.residual_vnr.nodes)

    if embedded_nodes & residual_nodes:
        errors.append("embedded and residual node sets overlap")
    if embedded_nodes | residual_nodes != original_nodes:
        errors.append("embedded and residual node sets do not cover the original VNR")
    if embedded_nodes and not _nodes_connected(original_vnr, embedded_nodes):
        errors.append("embedded VNR partition is not connected")
    if residual_nodes and not _nodes_connected(original_vnr, residual_nodes):
        errors.append("residual VNR partition is not connected")

    for node in residual_nodes:
        if dict(residual.residual_vnr.nodes[node]) != dict(original_vnr.nodes[node]):
            errors.append(f"residual node {node} has modified attributes")

    expected_residual_edges: set[tuple[str, str, int]] = set()
    expected_boundary: set[tuple[str, str, int]] = set()
    for u, v, attrs in original_vnr.edges(data=True):
        edge = tuple(sorted((u, v)))
        bandwidth = int(attrs["bandwidth"])
        if u in residual_nodes and v in residual_nodes:
            expected_residual_edges.add((edge[0], edge[1], bandwidth))
        elif (u in embedded_nodes and v in residual_nodes) or (v in embedded_nodes and u in residual_nodes):
            embedded = u if u in embedded_nodes else v
            remaining = v if u in embedded_nodes else u
            expected_boundary.add((embedded, remaining, bandwidth))

    actual_residual_edges = {
        (min(u, v), max(u, v), int(attrs["bandwidth"]))
        for u, v, attrs in residual.residual_vnr.edges(data=True)
    }
    if actual_residual_edges != expected_residual_edges:
        errors.append("residual internal edges do not match the original VNR")

    actual_boundary = {
        (demand.embedded_virtual_node, demand.residual_virtual_node, demand.bandwidth)
        for demand in residual.boundary_demands
    }
    if actual_boundary != expected_boundary:
        errors.append("boundary demands do not match original cut edges")
    if len(expected_boundary) != 1 or len(residual.boundary_demands) != 1:
        errors.append("partial split must have exactly one crossing virtual edge")

    return not errors, errors
