"""Reconstruct one portal request from public transparency-log records."""

from __future__ import annotations

from typing import Any

from system.api_models import ClaimResponse, PortalRequestResult, RequestScope


class InvalidRequestTrace(Exception):
    """Raised when public records for one request contradict each other."""


def unfinished_request_ids(records: list[dict[str, Any]]) -> list[int]:
    """Return issued request IDs that have no completed or declined record."""
    issued = {
        int(record["request_id"])
        for record in records
        if record.get("event_type") == "request_issued"
    }
    terminal = {
        int(record["request_id"])
        for record in records
        if record.get("event_type") in {"request_completed", "request_declined"}
    }
    return sorted(issued - terminal)


class RequestTrace:
    """Read-only view of the durable public records for one request."""

    def __init__(self, records: list[dict[str, Any]], request_id: int) -> None:
        self.request_id = request_id
        self.records = [record for record in records if record.get("request_id") == request_id]

    def issued(self) -> dict[str, Any] | None:
        return self._single("request_issued")

    def terminal(self) -> dict[str, Any] | None:
        records = [
            record
            for record in self.records
            if record.get("event_type") in {"request_completed", "request_declined"}
        ]
        return self._require_at_most_one(records, "terminal request record")

    def result(self) -> PortalRequestResult | None:
        record = self.terminal()
        if record is None:
            return None
        if record["event_type"] == "request_completed":
            return PortalRequestResult(
                request_id=self.request_id,
                status="accepted",
                claim_type=record["claim_type"],
                involved_facilities=record.get("involved_facilities", []),
                final_state_commitments=record.get("final_state_commitments", {}),
                timings_seconds=record.get("timings_seconds", {}),
            )
        recovered_states = {
            str(item["facility_id"]): str(item["state_after"])
            for item in self.records
            if item.get("event_type") == "state_committed"
        }
        final_states = record.get("final_state_commitments", recovered_states)
        return PortalRequestResult(
            request_id=self.request_id,
            status="declined",
            claim_type="decline",
            involved_facilities=record.get(
                "involved_facilities",
                record.get("committed_facilities", list(final_states)),
            ),
            final_state_commitments=final_states,
            reason=record.get("reason"),
            timings_seconds=record.get("timings_seconds", {}),
        )

    def claim(
        self,
        facility_id: str,
        request_scope: RequestScope,
    ) -> ClaimResponse | None:
        records = [
            record
            for record in self.records
            if record.get("event_type") == "claim_submitted"
            and record.get("facility_id") == facility_id
            and record.get("request_scope") == request_scope
        ]
        record = self._require_at_most_one(
            records,
            f"{facility_id} {request_scope} claim",
        )
        if record is None:
            return None
        return ClaimResponse(
            request_id=self.request_id,
            request_scope=request_scope,
            facility_id=facility_id,
            claim_type=record["claim_type"],
            state_before=record["state_before"],
            proposed_state_after=record.get("proposed_state_after"),
            commitments=record.get("commitments"),
            covered_virtual_nodes=record.get("covered_virtual_nodes", []),
            public_endpoint_id=record.get("public_endpoint_id"),
            public_endpoint_index=record.get("public_endpoint_index"),
            residual=record.get("residual"),
            proof_mode=record.get("proof_mode", "groth16"),
            artifact_id=record.get("artifact_id"),
            claim_record_index=record["record_index"],
            transition_id=record.get("transition_id"),
            decline_reason=record.get("decline_reason"),
            timings_seconds=record.get("timings_seconds", {}),
        )

    def accepted_proof(self, claim_record_index: int) -> dict[str, Any] | None:
        records = [
            record
            for record in self.records
            if record.get("event_type") == "proof_accepted"
            and record.get("parent_claim_record") == claim_record_index
            and record.get("result") == "accepted"
        ]
        return self._require_at_most_one(records, "accepted proof record")

    def rejected_proof(self, claim_record_index: int) -> dict[str, Any] | None:
        records = [
            record
            for record in self.records
            if record.get("event_type") == "proof_rejected"
            and record.get("parent_claim_record") == claim_record_index
        ]
        return self._require_at_most_one(records, "rejected proof record")

    def committed_state(self, claim_record_index: int) -> dict[str, Any] | None:
        records = [
            record
            for record in self.records
            if record.get("event_type") == "state_committed"
            and record.get("parent_claim_record") == claim_record_index
        ]
        return self._require_at_most_one(records, "committed state record")

    def residual_forwarded(
        self,
        claim_record_index: int,
        target_facility_id: str,
    ) -> dict[str, Any] | None:
        records = [
            record
            for record in self.records
            if record.get("event_type") == "residual_forwarded"
            and record.get("parent_claim_record") == claim_record_index
            and record.get("target_facility_id") == target_facility_id
        ]
        return self._require_at_most_one(records, "residual forwarding record")

    def _single(self, event_type: str) -> dict[str, Any] | None:
        records = [
            record for record in self.records if record.get("event_type") == event_type
        ]
        return self._require_at_most_one(records, event_type)

    @staticmethod
    def _require_at_most_one(
        records: list[dict[str, Any]],
        name: str,
    ) -> dict[str, Any] | None:
        if len(records) > 1:
            raise InvalidRequestTrace(f"request has more than one {name}")
        return records[0] if records else None
