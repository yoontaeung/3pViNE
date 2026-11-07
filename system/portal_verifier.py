"""Portal-side binding checks and independent Groth16 verification."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import networkx as nx

from system.api_models import (
    ArtifactCreate,
    ClaimRequest,
    ClaimResponse,
    PublicEndpointPayload,
)
from system.circuit_request import build_local_circuit_vnr
from system.graph_codec import graph_from_payload
from zk_encoding import (
    V_NODE_IDS,
    vnr_arch_vector,
    vnr_bandwidth_matrix,
    vnr_link_presence_matrix,
    vnr_memory_vector,
    vnr_node_presence_vector,
)


class ProofRejected(Exception):
    """Raised when the proof or its public protocol binding is invalid."""


class PortalProofVerifier:
    def __init__(
        self,
        *,
        verification_key_path: Path,
        work_dir: Path,
    ) -> None:
        self.verification_key_path = verification_key_path
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def verify_claim(
        self,
        *,
        request: ClaimRequest,
        claim: ClaimResponse,
        artifact: ArtifactCreate,
        registered_endpoint: PublicEndpointPayload | None,
    ) -> float:
        self._validate_binding(
            request=request,
            claim=claim,
            artifact=artifact,
            registered_endpoint=registered_endpoint,
        )
        verification_dir = self.work_dir / (
            f"request_{request.request_id:06d}_{claim.facility_id}_{request.request_scope}"
        )
        verification_dir.mkdir(parents=True, exist_ok=True)
        proof_path = verification_dir / "proof.json"
        public_path = verification_dir / "public.json"
        proof_path.write_text(
            json.dumps(artifact.proof_bundle.proof, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        public_path.write_text(
            json.dumps(artifact.proof_bundle.public_inputs) + "\n",
            encoding="utf-8",
        )

        started = time.perf_counter()
        result = subprocess.run(
            [
                "snarkjs",
                "groth16",
                "verify",
                str(self.verification_key_path),
                str(public_path),
                str(proof_path),
            ],
            text=True,
            capture_output=True,
        )
        elapsed = time.perf_counter() - started
        if result.returncode != 0 or "OK!" not in result.stdout:
            detail = (result.stderr or result.stdout).strip()
            raise ProofRejected(f"Groth16 verification failed: {detail}")
        return elapsed

    def _validate_binding(
        self,
        *,
        request: ClaimRequest,
        claim: ClaimResponse,
        artifact: ArtifactCreate,
        registered_endpoint: PublicEndpointPayload | None,
    ) -> None:
        if claim.claim_type not in {"full", "partial"}:
            raise ProofRejected("decline claims do not carry a proof")
        if claim.request_id != request.request_id:
            raise ProofRejected("claim request ID does not match the portal request")
        if claim.request_scope != request.request_scope:
            raise ProofRejected("claim scope does not match the portal request")
        if artifact.request_id != request.request_id:
            raise ProofRejected("artifact request ID does not match the portal request")
        if artifact.facility_id != claim.facility_id:
            raise ProofRejected("artifact facility does not match the claim")
        if artifact.claim_type != claim.claim_type:
            raise ProofRejected("artifact claim type does not match the claim")
        if claim.commitments is None or artifact.commitments != claim.commitments:
            raise ProofRejected("artifact commitments do not match the facility claim")
        if claim.state_before != claim.commitments.substrate_before:
            raise ProofRejected("claim state_before does not match C_S")
        if claim.proposed_state_after != claim.commitments.substrate_after:
            raise ProofRejected("claim proposed state does not match C_S_next")

        local_vnr, endpoint_enabled, endpoint_index = self._local_statement(
            request=request,
            claim=claim,
            registered_endpoint=registered_endpoint,
        )
        expected_public = self._expected_public_inputs(
            request_id=request.request_id,
            local_vnr=local_vnr,
            endpoint_enabled=endpoint_enabled,
            endpoint_index=endpoint_index,
            claim=claim,
        )
        if artifact.proof_bundle.public_inputs != expected_public:
            raise ProofRejected(
                "proof public inputs do not match the requested VNR and commitments"
            )

    @staticmethod
    def _local_statement(
        *,
        request: ClaimRequest,
        claim: ClaimResponse,
        registered_endpoint: PublicEndpointPayload | None,
    ) -> tuple[nx.Graph, int, int]:
        original = graph_from_payload(request.vnr, graph_id=str(request.request_id))
        endpoint_demands: list[tuple[str, int]] = []

        if claim.claim_type == "partial":
            if request.request_scope != "original" or request.incoming_boundary_demands:
                raise ProofRejected("a residual request cannot produce another partial claim")
            if claim.residual is None:
                raise ProofRejected("partial claim is missing its residual VNR")
            selected_nodes = claim.covered_virtual_nodes
            endpoint_demands = [
                (demand.embedded_virtual_node, demand.bandwidth)
                for demand in claim.residual.boundary_demands
            ]
        else:
            if claim.residual is not None:
                raise ProofRejected("full claim must not include a residual VNR")
            selected_nodes = [node.id for node in request.vnr.nodes]
            if set(claim.covered_virtual_nodes) != set(selected_nodes):
                raise ProofRejected("full claim does not cover every requested virtual node")
            endpoint_demands = [
                (demand.residual_virtual_node, demand.bandwidth)
                for demand in request.incoming_boundary_demands
            ]

        endpoint_enabled = int(bool(endpoint_demands))
        endpoint_index = 0
        if endpoint_enabled:
            if registered_endpoint is None or registered_endpoint.substrate_index is None:
                raise ProofRejected("endpoint-enabled proof has no registered endpoint index")
            if claim.public_endpoint_id != registered_endpoint.endpoint_id:
                raise ProofRejected("claim endpoint ID does not match facility registration")
            if claim.public_endpoint_index != registered_endpoint.substrate_index:
                raise ProofRejected("claim endpoint index does not match facility registration")
            endpoint_index = registered_endpoint.substrate_index

        try:
            local_vnr = build_local_circuit_vnr(
                original,
                selected_nodes,
                endpoint_demands=endpoint_demands,
            )
        except ValueError as exc:
            raise ProofRejected(str(exc)) from exc
        return local_vnr, endpoint_enabled, endpoint_index

    @staticmethod
    def _expected_public_inputs(
        *,
        request_id: int,
        local_vnr: nx.Graph,
        endpoint_enabled: int,
        endpoint_index: int,
        claim: ClaimResponse,
    ) -> list[str]:
        assert claim.commitments is not None
        ordered_nodes = sorted(local_vnr.nodes, key=lambda node: V_NODE_IDS[node])
        values = (
            [request_id]
            + vnr_node_presence_vector(local_vnr)
            + [int(local_vnr.nodes[node].get("cpu", 0)) for node in ordered_nodes]
            + vnr_memory_vector(local_vnr)
            + vnr_arch_vector(local_vnr)
            + PortalProofVerifier._flatten(vnr_link_presence_matrix(local_vnr))
            + PortalProofVerifier._flatten(vnr_bandwidth_matrix(local_vnr))
            + [endpoint_enabled, endpoint_index]
            + [
                int(claim.commitments.request),
                int(claim.commitments.substrate_before),
                int(claim.commitments.embedding),
                int(claim.commitments.substrate_after),
            ]
        )
        return [str(value) for value in values]

    @staticmethod
    def _flatten(matrix: list[list[int]]) -> list[int]:
        return [value for row in matrix for value in row]
