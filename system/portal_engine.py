"""Portal orchestration for full and connected single-cut VNE requests."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass

from system.api_models import (
    BoundaryDemandPayload,
    ClaimRequest,
    ClaimResponse,
    ClaimType,
    CommitRequest,
    GraphPayload,
    PortalRequestResult,
    PortalVnrRequest,
    PublicEndpointPayload,
    PublicEventRequest,
)
from system.facility_client import FacilityClient
from system.log_client import TransparencyLogClient
from system.portal_verifier import PortalProofVerifier, ProofRejected
from system.request_trace import RequestTrace, unfinished_request_ids
from system.residual_validation import InvalidResidual, validate_single_cut_residual


class DuplicateRequest(Exception):
    """Raised when a request ID is replayed with different request content."""


class UnknownFacility(Exception):
    """Raised when a request names a facility outside the portal configuration."""


class RequestInProgress(Exception):
    """Raised when another nonterminal request owns the serial protocol slot."""


@dataclass(frozen=True)
class PublicLink:
    endpoint_a: str
    endpoint_b: str
    bandwidth: int

    def __post_init__(self) -> None:
        if self.endpoint_a == self.endpoint_b:
            raise ValueError("a public link must connect two different endpoints")
        if self.bandwidth < 0:
            raise ValueError("public link bandwidth must be non-negative")

    def supports(self, endpoint_a: str, endpoint_b: str, bandwidth: int) -> bool:
        endpoints_match = {self.endpoint_a, self.endpoint_b} == {
            endpoint_a,
            endpoint_b,
        }
        return endpoints_match and self.bandwidth >= bandwidth


class PortalEngine:
    """Serialize requests and coordinate proof-carrying facility transitions."""

    def __init__(
        self,
        *,
        log_client: TransparencyLogClient,
        facilities: dict[str, FacilityClient],
        verifier: PortalProofVerifier,
        public_links: list[PublicLink] | None = None,
    ) -> None:
        self.log_client = log_client
        self.facilities = facilities
        self.verifier = verifier
        self.public_links = public_links or []
        self._lock = threading.Lock()

    def process(self, request: PortalVnrRequest) -> PortalRequestResult:
        with self._lock:
            return self._process_serial(request)

    def _process_serial(self, request: PortalVnrRequest) -> PortalRequestResult:
        started = time.perf_counter()
        request_commitment = self._request_digest(request)
        records = self.log_client.events()
        trace = RequestTrace(records, request.request_id)
        issued = trace.issued()
        if issued is not None:
            self._validate_replayed_request(request, request_commitment, issued)
            completed = trace.result()
            if completed is not None:
                return completed
        else:
            unfinished = unfinished_request_ids(records)
            if unfinished:
                raise RequestInProgress(
                    f"request {unfinished[0]} must be resumed before starting "
                    f"request {request.request_id}"
                )
            for facility_id in request.facility_order:
                if facility_id not in self.facilities:
                    raise UnknownFacility(f"facility {facility_id} is not configured")
            self.log_client.append_event(
                PublicEventRequest(
                    event_type="request_issued",
                    actor="portal",
                    fields={
                        "request_id": request.request_id,
                        "request_commitment": request_commitment,
                        "facility_order": request.facility_order,
                        "vnr": request.vnr.model_dump(mode="json"),
                    },
                )
            )

        for facility_id in request.facility_order:
            if facility_id not in self.facilities:
                raise UnknownFacility(f"facility {facility_id} is not configured")

        current_vnr = request.vnr
        incoming_demands: list[BoundaryDemandPayload] = []
        source_claim: ClaimResponse | None = None
        involved_facilities: list[str] = []
        final_states: dict[str, str] = {}
        timings: dict[str, float] = {}

        for facility_id in request.facility_order:
            facility = self.facilities[facility_id]
            registration = facility.registration()
            if incoming_demands:
                assert source_claim is not None
                if not self._can_forward(
                    source_claim=source_claim,
                    target_endpoint_id=(
                        registration.public_endpoint.endpoint_id
                        if registration.public_endpoint is not None
                        else None
                    ),
                    demands=incoming_demands,
                ):
                    continue
                self._record_residual_forwarding(
                    request_id=request.request_id,
                    source_claim=source_claim,
                    target_facility_id=facility_id,
                )

            request_scope = "residual" if incoming_demands else "original"
            existing_claim = self._trace(request.request_id).claim(
                facility_id,
                request_scope,
            )
            if existing_claim is None:
                expected_state = self.log_client.latest_state(facility_id)
                claim_request = ClaimRequest(
                    request_id=request.request_id,
                    request_scope=request_scope,
                    expected_state=expected_state,
                    vnr=current_vnr,
                    incoming_boundary_demands=incoming_demands,
                )
                claim = facility.create_claim(claim_request)
            else:
                claim = existing_claim
                claim_request = ClaimRequest(
                    request_id=request.request_id,
                    request_scope=request_scope,
                    expected_state=claim.state_before,
                    vnr=current_vnr,
                    incoming_boundary_demands=incoming_demands,
                )
            if claim.claim_type == "decline":
                continue

            if claim.claim_type == "partial":
                try:
                    self._validate_partial_claim(
                        request_id=request.request_id,
                        original=current_vnr,
                        claim=claim,
                    )
                except InvalidResidual as exc:
                    return self._reject_claim(
                        request=request,
                        request_commitment=request_commitment,
                        claim=claim,
                        reason=str(exc),
                        involved_facilities=involved_facilities,
                        final_states=final_states,
                        timings=timings,
                        started=started,
                    )

            try:
                verify_seconds, committed_state = self._verify_and_commit(
                    request=claim_request,
                    claim=claim,
                    facility=facility,
                    registered_endpoint=registration.public_endpoint,
                )
            except ProofRejected as exc:
                return self._reject_claim(
                    request=request,
                    request_commitment=request_commitment,
                    claim=claim,
                    reason=str(exc),
                    involved_facilities=involved_facilities,
                    final_states=final_states,
                    timings=timings,
                    started=started,
                )

            involved_facilities.append(facility_id)
            final_states[facility_id] = committed_state
            prefix = f"{facility_id}_{request_scope}"
            timings.update(
                {f"{prefix}_{name}": value for name, value in claim.timings_seconds.items()}
            )
            timings[f"{prefix}_portal_verify"] = verify_seconds

            if claim.claim_type == "full":
                overall_claim_type = "partial" if source_claim is not None else "full"
                return self._complete_request(
                    request=request,
                    request_commitment=request_commitment,
                    claim_type=overall_claim_type,
                    involved_facilities=involved_facilities,
                    final_states=final_states,
                    timings=timings,
                    started=started,
                )

            assert claim.residual is not None
            source_claim = claim
            current_vnr = claim.residual.vnr
            incoming_demands = claim.residual.boundary_demands

        reason = (
            "residual_not_completed" if source_claim is not None else "all_facilities_declined"
        )
        return self._decline_request(
            request=request,
            request_commitment=request_commitment,
            reason=reason,
            involved_facilities=involved_facilities,
            final_states=final_states,
            timings=timings,
            started=started,
        )

    def _verify_and_commit(
        self,
        *,
        request: ClaimRequest,
        claim: ClaimResponse,
        facility: FacilityClient,
        registered_endpoint: PublicEndpointPayload | None,
    ) -> tuple[float, str]:
        if claim.artifact_id is None or claim.claim_record_index is None:
            raise ProofRejected("facility did not publish its proof artifact")
        if claim.transition_id is None or claim.proposed_state_after is None:
            raise ProofRejected("facility claim is missing its proposed transition")

        trace = self._trace(request.request_id)
        committed_record = trace.committed_state(claim.claim_record_index)
        if committed_record is not None:
            self._require_record_values(
                committed_record,
                {
                    "facility_id": claim.facility_id,
                    "state_before": claim.state_before,
                    "state_after": claim.proposed_state_after,
                    "transition_id": claim.transition_id,
                },
                "committed state",
            )
            proof_record = trace.accepted_proof(claim.claim_record_index)
            verify_seconds = (
                float(proof_record.get("verification_seconds", 0.0))
                if proof_record is not None
                else 0.0
            )
            return verify_seconds, str(committed_record["state_after"])

        latest_state = self.log_client.latest_state(claim.facility_id)
        if claim.state_before != latest_state:
            raise ProofRejected("claim does not start from the latest public state")

        verification = trace.accepted_proof(claim.claim_record_index)
        if verification is None:
            artifact = self.log_client.artifact(claim.artifact_id)
            verify_seconds = self.verifier.verify_claim(
                request=request,
                claim=claim,
                artifact=artifact,
                registered_endpoint=registered_endpoint,
            )
            verification = self.log_client.append_event(
                PublicEventRequest(
                    event_type="proof_accepted",
                    actor="portal",
                    fields={
                        "verifier_id": "portal",
                        "facility_id": claim.facility_id,
                        "request_id": request.request_id,
                        "claim_type": claim.claim_type,
                        "request_scope": claim.request_scope,
                        "parent_claim_record": claim.claim_record_index,
                        "artifact_id": claim.artifact_id,
                        "transition_id": claim.transition_id,
                        "result": "accepted",
                        "verification_seconds": verify_seconds,
                    },
                )
            )
        else:
            self._require_record_values(
                verification,
                {
                    "facility_id": claim.facility_id,
                    "artifact_id": claim.artifact_id,
                    "transition_id": claim.transition_id,
                    "result": "accepted",
                },
                "accepted proof",
            )
            verify_seconds = float(verification.get("verification_seconds", 0.0))

        committed = facility.commit(
            CommitRequest(
                transition_id=claim.transition_id,
                expected_state_before=claim.state_before,
                expected_state_after=claim.proposed_state_after,
                claim_record_index=claim.claim_record_index,
                verification_record_index=verification["record_index"],
            )
        )
        public_latest = self.log_client.latest_state(claim.facility_id)
        if public_latest != committed.state_after:
            raise RuntimeError("facility commit did not advance the public state")
        return verify_seconds, public_latest

    @staticmethod
    def _validate_partial_claim(
        *,
        request_id: int,
        original: GraphPayload,
        claim: ClaimResponse,
    ) -> None:
        if claim.residual is None:
            raise InvalidResidual("partial claim is missing its residual VNR")
        if claim.public_endpoint_id is None:
            raise InvalidResidual("partial claim is missing its public endpoint")
        if set(claim.covered_virtual_nodes) != set(
            claim.residual.embedded_virtual_nodes
        ):
            raise InvalidResidual("covered nodes do not match the residual partition")
        validate_single_cut_residual(
            request_id=request_id,
            original=original,
            residual=claim.residual,
            source_facility_id=claim.facility_id,
            source_endpoint_id=claim.public_endpoint_id,
        )

    def _can_forward(
        self,
        *,
        source_claim: ClaimResponse,
        target_endpoint_id: str | None,
        demands: list[BoundaryDemandPayload],
    ) -> bool:
        if source_claim.public_endpoint_id is None or target_endpoint_id is None:
            return False
        required_bandwidth = sum(demand.bandwidth for demand in demands)
        return any(
            link.supports(
                source_claim.public_endpoint_id,
                target_endpoint_id,
                required_bandwidth,
            )
            for link in self.public_links
        )

    def _record_residual_forwarding(
        self,
        *,
        request_id: int,
        source_claim: ClaimResponse,
        target_facility_id: str,
    ) -> None:
        assert source_claim.residual is not None
        assert source_claim.claim_record_index is not None
        existing = self._trace(request_id).residual_forwarded(
            source_claim.claim_record_index,
            target_facility_id,
        )
        expected = {
            "source_facility_id": source_claim.facility_id,
            "target_facility_id": target_facility_id,
            "residual_commitment": source_claim.residual.residual_commitment,
        }
        if existing is not None:
            self._require_record_values(existing, expected, "residual forwarding")
            return
        self.log_client.append_event(
            PublicEventRequest(
                event_type="residual_forwarded",
                actor="portal",
                fields={
                    "request_id": request_id,
                    "source_facility_id": source_claim.facility_id,
                    "target_facility_id": target_facility_id,
                    "parent_claim_record": source_claim.claim_record_index,
                    "residual_commitment": source_claim.residual.residual_commitment,
                    "residual_vnr": source_claim.residual.vnr.model_dump(mode="json"),
                    "boundary_demands": [
                        demand.model_dump(mode="json")
                        for demand in source_claim.residual.boundary_demands
                    ],
                },
            )
        )

    def _complete_request(
        self,
        *,
        request: PortalVnrRequest,
        request_commitment: str,
        claim_type: ClaimType,
        involved_facilities: list[str],
        final_states: dict[str, str],
        timings: dict[str, float],
        started: float,
    ) -> PortalRequestResult:
        existing = self._trace(request.request_id).result()
        if existing is not None:
            return existing
        completed_timings = dict(timings)
        completed_timings["online_total"] = time.perf_counter() - started
        self.log_client.append_event(
            PublicEventRequest(
                event_type="request_completed",
                actor="portal",
                fields={
                    "request_id": request.request_id,
                    "request_commitment": request_commitment,
                    "claim_type": claim_type,
                    "involved_facilities": involved_facilities,
                    "final_state_commitments": final_states,
                    "timings_seconds": completed_timings,
                },
            )
        )
        return PortalRequestResult(
            request_id=request.request_id,
            status="accepted",
            claim_type=claim_type,
            involved_facilities=involved_facilities,
            final_state_commitments=final_states,
            timings_seconds=completed_timings,
        )

    def _reject_claim(
        self,
        *,
        request: PortalVnrRequest,
        request_commitment: str,
        claim: ClaimResponse,
        reason: str,
        involved_facilities: list[str],
        final_states: dict[str, str],
        timings: dict[str, float],
        started: float,
    ) -> PortalRequestResult:
        if claim.claim_record_index is not None:
            rejected = self._trace(request.request_id).rejected_proof(
                claim.claim_record_index
            )
        else:
            rejected = None
        if rejected is None:
            self.log_client.append_event(
                PublicEventRequest(
                    event_type="proof_rejected",
                    actor="portal",
                    fields={
                        "verifier_id": "portal",
                        "facility_id": claim.facility_id,
                        "request_id": request.request_id,
                        "parent_claim_record": claim.claim_record_index,
                        "artifact_id": claim.artifact_id,
                        "result": "rejected",
                        "reason": reason,
                    },
                )
            )
        return self._decline_request(
            request=request,
            request_commitment=request_commitment,
            reason=reason,
            involved_facilities=involved_facilities,
            final_states=final_states,
            timings=timings,
            started=started,
        )

    def _decline_request(
        self,
        *,
        request: PortalVnrRequest,
        request_commitment: str,
        reason: str,
        involved_facilities: list[str],
        final_states: dict[str, str],
        timings: dict[str, float],
        started: float,
    ) -> PortalRequestResult:
        existing = self._trace(request.request_id).result()
        if existing is not None:
            return existing
        completed_timings = dict(timings)
        completed_timings["online_total"] = time.perf_counter() - started
        self.log_client.append_event(
            PublicEventRequest(
                event_type="request_declined",
                actor="portal",
                fields={
                    "request_id": request.request_id,
                    "request_commitment": request_commitment,
                    "reason": reason,
                    "involved_facilities": involved_facilities,
                    "final_state_commitments": final_states,
                    "timings_seconds": completed_timings,
                },
            )
        )
        return PortalRequestResult(
            request_id=request.request_id,
            status="declined",
            claim_type="decline",
            involved_facilities=involved_facilities,
            final_state_commitments=final_states,
            reason=reason,
            timings_seconds=completed_timings,
        )

    def _trace(self, request_id: int) -> RequestTrace:
        return RequestTrace(self.log_client.events(), request_id)

    @staticmethod
    def _validate_replayed_request(
        request: PortalVnrRequest,
        request_commitment: str,
        issued: dict[str, object],
    ) -> None:
        expected_vnr = request.vnr.model_dump(mode="json")
        if (
            issued.get("request_commitment") != request_commitment
            or issued.get("facility_order") != request.facility_order
            or issued.get("vnr") != expected_vnr
        ):
            raise DuplicateRequest(
                f"request ID {request.request_id} is already bound to different content"
            )

    @staticmethod
    def _require_record_values(
        record: dict[str, object],
        expected: dict[str, object],
        name: str,
    ) -> None:
        mismatched = [
            key for key, value in expected.items() if record.get(key) != value
        ]
        if mismatched:
            raise ProofRejected(
                f"{name} record disagrees on: {', '.join(sorted(mismatched))}"
            )

    @staticmethod
    def _request_digest(request: PortalVnrRequest) -> str:
        value = {
            "request_id": request.request_id,
            "vnr": request.vnr.model_dump(mode="json"),
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
