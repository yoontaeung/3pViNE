"""Private facility state machine used by the HTTP service."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from partial_protocol import (
    BoundaryDemand,
    ConnectedPartialClaim,
    DeclineClaim,
    FullEmbeddingClaim,
    solve_connected_vne_claim,
)
from protocol_demo import FacilityRuntime, load_facilities
from system.api_models import (
    BoundaryDemandPayload,
    ClaimRequest,
    ClaimResponse,
    CommitRequest,
    CommitResponse,
    FacilityRegistration,
    GraphPayload,
    PublicCommitments,
    PublicEndpointPayload,
    ResidualPayload,
    StateTransitionCreate,
)
from system.graph_codec import graph_from_payload, graph_to_payload
from system.facility_store import FacilityPrivateStore
from system.proof_runtime import ClaimProofRuntime, GeneratedClaimProof
from system.residual_validation import residual_commitment
from zk_encoding import (
    S_NODE_IDS,
    apply_after_state_to_substrate,
    substrate_state_commitment,
)


StateCommitmentFunction = Callable[..., int]


class FacilityStateMismatch(Exception):
    """Raised when a portal request is based on an obsolete state commitment."""


class UnknownTransition(Exception):
    """Raised when a commit references no pending transition."""


@dataclass
class PendingTransition:
    request_id: int
    state_before: str
    state_after: str
    rho_s_next: int
    zk_input: dict[str, Any]


@dataclass
class CommittedTransition:
    request: CommitRequest
    response: CommitResponse


class FacilityEngine:
    """Serialize access to one facility's private substrate and proving state."""

    def __init__(
        self,
        *,
        facility: FacilityRuntime,
        proof_runtime: ClaimProofRuntime,
        state_commitment_fn: StateCommitmentFunction = substrate_state_commitment,
        private_store: FacilityPrivateStore | None = None,
    ) -> None:
        self.facility = facility
        self.proof_runtime = proof_runtime
        self.state_commitment_fn = state_commitment_fn
        self.private_store = private_store
        self._pending: dict[str, PendingTransition] = {}
        self._committed: dict[str, CommittedTransition] = {}
        self._lock = threading.Lock()
        if self.private_store is not None and self.private_store.exists():
            self._restore_private_state()
        else:
            self._persist_private_state()

    @classmethod
    def from_config(
        cls,
        *,
        config_path: Path,
        proof_runtime: ClaimProofRuntime,
        facility_id: str | None = None,
        state_path: Path | None = None,
    ) -> "FacilityEngine":
        facilities, _ = load_facilities(config_path)
        if facility_id is None:
            if len(facilities) != 1:
                raise ValueError("a facility service config must contain exactly one facility")
            facility = next(iter(facilities.values()))
        else:
            try:
                facility = facilities[facility_id]
            except KeyError as exc:
                raise ValueError(f"facility {facility_id} is not present in {config_path}") from exc
        return cls(
            facility=facility,
            proof_runtime=proof_runtime,
            private_store=FacilityPrivateStore(state_path) if state_path is not None else None,
        )

    @property
    def facility_id(self) -> str:
        return self.facility.config.facility_id

    def current_state_commitment(self) -> str:
        return str(
            self.state_commitment_fn(
                self.facility.substrate,
                rho_S=self.facility.current_rho_S,
            )
        )

    def registration(self) -> FacilityRegistration:
        endpoint = self.facility.config.endpoint
        return FacilityRegistration(
            facility_id=self.facility_id,
            state_commitment=self.current_state_commitment(),
            public_endpoint=(
                PublicEndpointPayload(
                    endpoint_id=endpoint.endpoint_id,
                    facility_id=self.facility_id,
                    substrate_index=S_NODE_IDS[endpoint.substrate_node],
                )
                if endpoint is not None
                else None
            ),
        )

    def create_claim(self, request: ClaimRequest) -> ClaimResponse:
        with self._lock:
            state_before = self.current_state_commitment()
            if request.expected_state != state_before:
                raise FacilityStateMismatch(
                    f"request expects {request.expected_state}, "
                    f"but {self.facility_id} is at {state_before}"
                )

            vnr = graph_from_payload(request.vnr, graph_id=str(request.request_id))
            incoming = tuple(
                self._boundary_demand(item) for item in request.incoming_boundary_demands
            )
            solve_started = time.perf_counter()
            claim = solve_connected_vne_claim(
                vnr,
                self.facility.substrate,
                facility_id=self.facility_id,
                endpoint=self.facility.config.endpoint,
                policy=self.facility.config.solver_policy,
                incoming_boundary_demands=incoming,
            )
            solve_seconds = time.perf_counter() - solve_started
            if isinstance(claim, DeclineClaim):
                return ClaimResponse(
                    request_id=request.request_id,
                    request_scope=request.request_scope,
                    facility_id=self.facility_id,
                    claim_type="decline",
                    state_before=state_before,
                    proof_mode=self.proof_runtime.proof_mode,
                    decline_reason=claim.reason,
                    timings_seconds={"solve": solve_seconds},
                )

            proof_index, randomness = self._next_randomness(request.request_id)
            generated = self.proof_runtime.generate(
                request_id=request.request_id,
                facility_id=self.facility_id,
                proof_index=proof_index,
                substrate=self.facility.substrate,
                local_vnr=claim.local_vnr,
                embedding=claim.embedding,
                randomness=randomness,
                endpoint_substrate_node=claim.endpoint_substrate_node,
            )
            response = self._accepted_response(
                request=request,
                claim=claim,
                generated=generated,
                state_before=state_before,
            )
            response = response.model_copy(
                update={
                    "timings_seconds": {
                        "solve": solve_seconds,
                        **response.timings_seconds,
                    }
                }
            )
            assert response.transition_id is not None
            assert response.proposed_state_after is not None
            self._pending[response.transition_id] = PendingTransition(
                request_id=request.request_id,
                state_before=state_before,
                state_after=response.proposed_state_after,
                rho_s_next=randomness["rho_S_next"],
                zk_input=generated.zk_input,
            )
            self._persist_private_state()
            return response

    def commit(self, request: CommitRequest) -> CommitResponse:
        with self._lock:
            committed = self._committed.get(request.transition_id)
            if committed is not None:
                if committed.request != request:
                    raise FacilityStateMismatch(
                        "transition was already committed with different verification metadata"
                    )
                return committed.response

            pending = self._pending.get(request.transition_id)
            if pending is None:
                raise UnknownTransition(f"unknown transition {request.transition_id}")
            current = self.current_state_commitment()
            if current != pending.state_before or current != request.expected_state_before:
                raise FacilityStateMismatch(
                    f"transition starts from {pending.state_before}, but current state is {current}"
                )
            if request.expected_state_after != pending.state_after:
                raise FacilityStateMismatch(
                    "portal and facility disagree on the next state commitment"
                )

            self.facility.substrate = apply_after_state_to_substrate(
                self.facility.substrate,
                pending.zk_input,
            )
            self.facility.current_rho_S = pending.rho_s_next
            del self._pending[request.transition_id]
            response = CommitResponse(
                facility_id=self.facility_id,
                request_id=pending.request_id,
                transition_id=request.transition_id,
                state_before=pending.state_before,
                state_after=pending.state_after,
            )
            self._committed[request.transition_id] = CommittedTransition(
                request=request,
                response=response,
            )
            self._persist_private_state()
            return response

    def unlogged_state_transitions(self) -> list[StateTransitionCreate]:
        with self._lock:
            return [
                self._state_transition(committed)
                for committed in self._committed.values()
                if committed.response.state_record_index is None
            ]

    def state_transition(self, transition_id: str) -> StateTransitionCreate:
        with self._lock:
            try:
                committed = self._committed[transition_id]
            except KeyError as exc:
                raise UnknownTransition(f"transition {transition_id} is not committed") from exc
            return self._state_transition(committed)

    def mark_state_record(self, transition_id: str, record_index: int) -> CommitResponse:
        with self._lock:
            try:
                committed = self._committed[transition_id]
            except KeyError as exc:
                raise UnknownTransition(f"transition {transition_id} is not committed") from exc
            if (
                committed.response.state_record_index is not None
                and committed.response.state_record_index != record_index
            ):
                raise FacilityStateMismatch("transition has a different public state record")
            committed.response = committed.response.model_copy(
                update={"state_record_index": record_index}
            )
            self._persist_private_state()
            return committed.response

    def _next_randomness(self, request_id: int) -> tuple[int, dict[str, int]]:
        proof_index = self.facility.proof_counts.get(request_id, 0)
        configured = self.facility.config.randomness_by_request.get(request_id, [])
        if proof_index < len(configured):
            selected = configured[proof_index]
            rho_m = int(selected["rho_M"])
            rho_s_next = int(selected["rho_S_next"])
        else:
            rho_m = secrets.randbits(128)
            rho_s_next = secrets.randbits(128)
        self.facility.proof_counts[request_id] = proof_index + 1
        return proof_index + 1, {
            "rho_S": self.facility.current_rho_S,
            "rho_M": rho_m,
            "rho_S_next": rho_s_next,
        }

    def _restore_private_state(self) -> None:
        assert self.private_store is not None
        value = self.private_store.load()
        if value["facility_id"] != self.facility_id:
            raise ValueError("persisted facility state belongs to another facility")
        self.facility.substrate = graph_from_payload(
            GraphPayload.model_validate(value["substrate"]),
            graph_id=self.facility_id,
        )
        self.facility.substrate.graph["facility_id"] = self.facility_id
        self.facility.current_rho_S = int(value["current_rho_S"])
        self.facility.proof_counts = {
            int(request_id): int(count)
            for request_id, count in value.get("proof_counts", {}).items()
        }
        self._pending = {
            transition_id: PendingTransition(**pending)
            for transition_id, pending in value.get("pending", {}).items()
        }
        self._committed = {
            transition_id: CommittedTransition(
                request=CommitRequest.model_validate(committed["request"]),
                response=CommitResponse.model_validate(committed["response"]),
            )
            for transition_id, committed in value.get("committed", {}).items()
        }

    def _persist_private_state(self) -> None:
        if self.private_store is None:
            return
        self.private_store.save(
            {
                "facility_id": self.facility_id,
                "current_rho_S": self.facility.current_rho_S,
                "substrate": graph_to_payload(self.facility.substrate).model_dump(mode="json"),
                "proof_counts": {
                    str(request_id): count
                    for request_id, count in self.facility.proof_counts.items()
                },
                "pending": {
                    transition_id: {
                        "request_id": pending.request_id,
                        "state_before": pending.state_before,
                        "state_after": pending.state_after,
                        "rho_s_next": pending.rho_s_next,
                        "zk_input": pending.zk_input,
                    }
                    for transition_id, pending in self._pending.items()
                },
                "committed": {
                    transition_id: {
                        "request": committed.request.model_dump(mode="json"),
                        "response": committed.response.model_dump(mode="json"),
                    }
                    for transition_id, committed in self._committed.items()
                },
            }
        )

    @staticmethod
    def _state_transition(committed: CommittedTransition) -> StateTransitionCreate:
        return StateTransitionCreate(
            facility_id=committed.response.facility_id,
            request_id=committed.response.request_id,
            transition_id=committed.response.transition_id,
            state_before=committed.response.state_before,
            state_after=committed.response.state_after,
            claim_record_index=committed.request.claim_record_index,
            verification_record_index=committed.request.verification_record_index,
        )

    def _accepted_response(
        self,
        *,
        request: ClaimRequest,
        claim: FullEmbeddingClaim | ConnectedPartialClaim,
        generated: GeneratedClaimProof,
        state_before: str,
    ) -> ClaimResponse:
        zk_input = generated.zk_input
        commitments = PublicCommitments(
            request=str(zk_input["C_r"]),
            substrate_before=str(zk_input["C_S"]),
            embedding=str(zk_input["C_M"]),
            substrate_after=str(zk_input["C_S_next"]),
        )
        transition_id = self._transition_id(
            request_id=request.request_id,
            commitments=commitments,
        )

        residual = None
        covered_nodes: list[str]
        if isinstance(claim, ConnectedPartialClaim):
            covered_nodes = list(claim.embedded_virtual_nodes)
            boundary = [self._boundary_payload(item) for item in claim.residual.boundary_demands]
            residual_vnr = graph_to_payload(claim.residual.residual_vnr)
            commitment = residual_commitment(
                request_id=request.request_id,
                vnr=residual_vnr,
                boundary_demands=boundary,
            )
            residual = ResidualPayload(
                vnr=residual_vnr,
                embedded_virtual_nodes=covered_nodes,
                boundary_demands=boundary,
                residual_commitment=commitment,
            )
        else:
            covered_nodes = sorted(claim.slot_to_original.values())

        endpoint = self.facility.config.endpoint
        return ClaimResponse(
            request_id=request.request_id,
            request_scope=request.request_scope,
            facility_id=self.facility_id,
            claim_type=claim.status,
            state_before=state_before,
            proposed_state_after=str(zk_input["C_S_next"]),
            commitments=commitments,
            covered_virtual_nodes=covered_nodes,
            public_endpoint_id=endpoint.endpoint_id if endpoint is not None else None,
            public_endpoint_index=(
                S_NODE_IDS[endpoint.substrate_node] if endpoint is not None else None
            ),
            residual=residual,
            proof_mode=self.proof_runtime.proof_mode,
            proof_bundle=generated.proof_bundle,
            transition_id=transition_id,
            timings_seconds=generated.timings_seconds,
        )

    def _transition_id(self, *, request_id: int, commitments: PublicCommitments) -> str:
        value = {
            "facility_id": self.facility_id,
            "request_id": request_id,
            "commitments": commitments.model_dump(mode="json"),
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _boundary_demand(item: BoundaryDemandPayload) -> BoundaryDemand:
        return BoundaryDemand(**item.model_dump())

    @staticmethod
    def _boundary_payload(item: BoundaryDemand) -> BoundaryDemandPayload:
        return BoundaryDemandPayload(
            source_facility_id=item.source_facility_id,
            source_endpoint_id=item.source_endpoint_id,
            embedded_virtual_node=item.embedded_virtual_node,
            residual_virtual_node=item.residual_virtual_node,
            bandwidth=item.bandwidth,
        )
