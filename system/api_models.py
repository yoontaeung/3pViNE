"""Versioned public API models shared by ZK-VNE services."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


API_VERSION = "v1"
ClaimType = Literal["full", "partial", "decline"]
RequestScope = Literal["original", "residual"]
ProofMode = Literal["groth16", "reference"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphNode(StrictModel):
    id: str
    cpu: int = Field(default=0, ge=0)
    memory_gib: int = Field(default=0, ge=0)
    arch: str | int | None = 0
    public_endpoint: bool = False


class GraphEdge(StrictModel):
    u: str
    v: str
    bandwidth: int = Field(default=0, ge=0)


class GraphPayload(StrictModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    @model_validator(mode="after")
    def validate_graph_references(self) -> "GraphPayload":
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node IDs must be unique")
        known_nodes = set(node_ids)
        seen_edges: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.u not in known_nodes or edge.v not in known_nodes:
                raise ValueError(f"edge ({edge.u}, {edge.v}) references an unknown node")
            if edge.u == edge.v:
                raise ValueError("self-loop edges are not supported")
            edge_key = tuple(sorted((edge.u, edge.v)))
            if edge_key in seen_edges:
                raise ValueError(f"duplicate edge ({edge_key[0]}, {edge_key[1]})")
            seen_edges.add(edge_key)
        return self


class PublicEndpointPayload(StrictModel):
    endpoint_id: str
    facility_id: str
    substrate_index: int | None = Field(default=None, ge=0)


class BoundaryDemandPayload(StrictModel):
    source_facility_id: str
    source_endpoint_id: str
    embedded_virtual_node: str
    residual_virtual_node: str
    bandwidth: int = Field(ge=0)


class ResidualPayload(StrictModel):
    vnr: GraphPayload
    embedded_virtual_nodes: list[str]
    boundary_demands: list[BoundaryDemandPayload]
    residual_commitment: str


class PublicCommitments(StrictModel):
    request: str
    substrate_before: str
    embedding: str
    substrate_after: str


class ProofBundle(StrictModel):
    proof_system: Literal["groth16"] = "groth16"
    proof: dict[str, Any]
    public_inputs: list[str]


class FacilityRegistration(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    facility_id: str
    state_commitment: str
    public_endpoint: PublicEndpointPayload | None = None


class ClaimRequest(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    request_id: int = Field(ge=0)
    request_scope: RequestScope = "original"
    expected_state: str
    vnr: GraphPayload
    incoming_boundary_demands: list[BoundaryDemandPayload] = Field(default_factory=list)


class ClaimResponse(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    request_id: int
    request_scope: RequestScope
    facility_id: str
    claim_type: ClaimType
    state_before: str
    proposed_state_after: str | None = None
    commitments: PublicCommitments | None = None
    covered_virtual_nodes: list[str] = Field(default_factory=list)
    public_endpoint_id: str | None = None
    public_endpoint_index: int | None = Field(default=None, ge=0)
    residual: ResidualPayload | None = None
    proof_mode: ProofMode
    proof_bundle: ProofBundle | None = None
    artifact_id: str | None = None
    claim_record_index: int | None = None
    transition_id: str | None = None
    decline_reason: str | None = None
    timings_seconds: dict[str, float] = Field(default_factory=dict)


class CommitRequest(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    transition_id: str
    expected_state_before: str
    expected_state_after: str
    claim_record_index: int = Field(ge=0)
    verification_record_index: int = Field(ge=0)


class CommitResponse(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    facility_id: str
    request_id: int
    transition_id: str
    state_before: str
    state_after: str
    state_record_index: int | None = None
    status: Literal["committed"] = "committed"


class StateTransitionCreate(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    facility_id: str
    request_id: int
    transition_id: str
    state_before: str
    state_after: str
    claim_record_index: int = Field(ge=0)
    verification_record_index: int = Field(ge=0)


class PublicEventRequest(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    event_type: str
    actor: Literal["facility", "portal", "log"]
    fields: dict[str, Any] = Field(default_factory=dict)


class ArtifactCreate(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    facility_id: str
    request_id: int
    claim_type: Literal["full", "partial"]
    commitments: PublicCommitments
    proof_bundle: ProofBundle


class ArtifactReference(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    artifact_id: str


class HealthResponse(StrictModel):
    service: str
    status: Literal["ok"] = "ok"
    api_version: Literal["v1"] = API_VERSION


class PortalVnrRequest(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    request_id: int = Field(ge=0)
    facility_order: list[str] = Field(min_length=1)
    vnr: GraphPayload

    @model_validator(mode="after")
    def validate_facility_order(self) -> "PortalVnrRequest":
        if len(self.facility_order) != len(set(self.facility_order)):
            raise ValueError("facility order must not contain duplicates")
        return self


class PortalRequestResult(StrictModel):
    api_version: Literal["v1"] = API_VERSION
    request_id: int
    status: Literal["accepted", "declined"]
    claim_type: ClaimType
    involved_facilities: list[str] = Field(default_factory=list)
    final_state_commitments: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None
    timings_seconds: dict[str, float] = Field(default_factory=dict)
