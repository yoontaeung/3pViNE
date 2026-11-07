"""Public validation for connected, exactly-one-cut residual VNRs."""

from __future__ import annotations

import hashlib
import json

import networkx as nx

from system.api_models import BoundaryDemandPayload, GraphPayload, ResidualPayload
from system.graph_codec import graph_from_payload


class InvalidResidual(Exception):
    """Raised when a partial claim does not faithfully partition its VNR."""


def residual_commitment(
    *,
    request_id: int,
    vnr: GraphPayload,
    boundary_demands: list[BoundaryDemandPayload],
) -> str:
    value = {
        "request_id": request_id,
        "residual_vnr": vnr.model_dump(mode="json"),
        "boundary_demands": [
            demand.model_dump(mode="json") for demand in boundary_demands
        ],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_single_cut_residual(
    *,
    request_id: int,
    original: GraphPayload,
    residual: ResidualPayload,
    source_facility_id: str,
    source_endpoint_id: str,
) -> None:
    original_graph = graph_from_payload(original, graph_id=str(request_id))
    residual_graph = graph_from_payload(
        residual.vnr,
        graph_id=f"{request_id}-residual",
    )
    original_nodes = set(original_graph.nodes)
    embedded_nodes = set(residual.embedded_virtual_nodes)
    residual_nodes = set(residual_graph.nodes)

    errors: list[str] = []
    if not embedded_nodes or not residual_nodes:
        errors.append("partial claim must contain embedded and residual nodes")
    if embedded_nodes & residual_nodes:
        errors.append("embedded and residual node sets overlap")
    if embedded_nodes | residual_nodes != original_nodes:
        errors.append("embedded and residual nodes do not cover the original VNR")
    if not _connected(original_graph, embedded_nodes):
        errors.append("embedded VNR partition is not connected")
    if not _connected(original_graph, residual_nodes):
        errors.append("residual VNR partition is not connected")

    for node in residual_nodes:
        if dict(residual_graph.nodes[node]) != dict(original_graph.nodes[node]):
            errors.append(f"residual node {node} has modified attributes")

    expected_edges: set[tuple[str, str, int]] = set()
    expected_boundary: set[tuple[str, str, int]] = set()
    for source, target, attributes in original_graph.edges(data=True):
        bandwidth = int(attributes.get("bandwidth", 0))
        if source in residual_nodes and target in residual_nodes:
            edge = sorted((source, target))
            expected_edges.add((edge[0], edge[1], bandwidth))
        elif (source in embedded_nodes) != (target in embedded_nodes):
            embedded = source if source in embedded_nodes else target
            remaining = target if source in embedded_nodes else source
            expected_boundary.add((embedded, remaining, bandwidth))

    actual_edges = {
        (min(source, target), max(source, target), int(attributes["bandwidth"]))
        for source, target, attributes in residual_graph.edges(data=True)
    }
    if actual_edges != expected_edges:
        errors.append("residual internal edges do not match the original VNR")

    actual_boundary = {
        (
            demand.embedded_virtual_node,
            demand.residual_virtual_node,
            demand.bandwidth,
        )
        for demand in residual.boundary_demands
    }
    if actual_boundary != expected_boundary:
        errors.append("boundary demands do not match the original cut edges")
    if len(expected_boundary) != 1 or len(residual.boundary_demands) != 1:
        errors.append("partial split must have exactly one crossing virtual edge")
    for demand in residual.boundary_demands:
        if demand.source_facility_id != source_facility_id:
            errors.append("boundary demand has the wrong source facility")
        if demand.source_endpoint_id != source_endpoint_id:
            errors.append("boundary demand has the wrong source endpoint")

    expected_commitment = residual_commitment(
        request_id=request_id,
        vnr=residual.vnr,
        boundary_demands=residual.boundary_demands,
    )
    if residual.residual_commitment != expected_commitment:
        errors.append("residual commitment does not match the residual message")

    if errors:
        raise InvalidResidual("; ".join(errors))


def _connected(graph: nx.Graph, nodes: set[str]) -> bool:
    if not nodes:
        return False
    return len(nodes) == 1 or nx.is_connected(graph.subgraph(nodes))
