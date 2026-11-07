"""Stateful protocol helpers for proof-carrying VNE transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ProtocolError(Exception):
    """Base class for protocol-level consistency errors."""


class StaleStateError(ProtocolError):
    """Raised when a transition does not start from the latest public state."""


PRIVATE_LOG_FIELDS = {
    "endpoint_paths",
    "internal_paths",
    "nodeMap",
    "node_mapping",
    "nodeSel",
    "pathNodes",
    "pathNodeSel",
    "pathLenSel",
    "rho_S",
    "rho_M",
    "rho_S_next",
    "sCpuBefore",
    "sCpuAfter",
    "sMemBefore",
    "sMemAfter",
    "sAdj",
    "sBwBefore",
    "sBwAfter",
    "sNodePresent",
    "substrate",
}


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PublicArtifacts:
    proof_path: str
    public_input_path: str


class TransparencyLog:
    """Append-only public state-transition log for one protocol run."""

    def __init__(self, path: Path, *, pretty_path: Path | None = None) -> None:
        self.path = path
        self.pretty_path = pretty_path or path.with_suffix(".json")
        self.records: list[dict[str, Any]] = []
        self._latest_state: dict[str, str] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load_existing_records()
            self._write_pretty_log()

    def _load_existing_records(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            self.records.append(record)
            if "facility_id" in record and record.get("state_after") is not None:
                self._latest_state[record["facility_id"]] = record["state_after"]

    def latest_state(self, facility_id: str) -> str | None:
        return self._latest_state.get(facility_id)

    # registers the first public substrate commitment
    def append_init(
        self,
        *,
        facility_id: str,
        state_commitment: int | str,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        if facility_id in self._latest_state:
            raise ProtocolError(f"facility {facility_id} is already initialized")
        commitment = str(state_commitment)
        return self._append(
            {
                "facility_id": facility_id,
                "status": "init",
                "state_before": None,
                "state_after": commitment,
                "timestamp_utc": timestamp or utc_timestamp(),
            }
        )

    def append_accepted(
        self,
        *,
        facility_id: str,
        request_id: int,
        state_before: int | str,
        state_after: int | str,
        request_commitment: int | str,
        embedding_commitment: int | str,
        artifacts: PublicArtifacts,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        before = str(state_before)
        self._require_latest(facility_id, before)
        return self._append(
            {
                "facility_id": facility_id,
                "request_id": request_id,
                "status": "accepted",
                "state_before": before,
                "state_after": str(state_after),
                "request_commitment": str(request_commitment),
                "embedding_commitment": str(embedding_commitment),
                "proof_path": artifacts.proof_path,
                "public_input_path": artifacts.public_input_path,
                "timestamp_utc": timestamp or utc_timestamp(),
            }
        )

    def append_declined(
        self,
        *,
        facility_id: str,
        request_id: int,
        state_before: int | str,
        request_commitment: int | str,
        reason: str = "facility_declined_unproven",
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        before = str(state_before)
        self._require_latest(facility_id, before)
        return self._append(
            {
                "facility_id": facility_id,
                "request_id": request_id,
                "status": "declined",
                "state_before": before,
                "state_after": before,
                "request_commitment": str(request_commitment),
                "reason": reason,
                "timestamp_utc": timestamp or utc_timestamp(),
            }
        )

    def append_state_committed(
        self,
        *,
        facility_id: str,
        request_id: int,
        state_before: int | str,
        state_after: int | str,
        parent_claim_record: int,
        proof_accepted_record: int,
        transition_id: str | None = None,
        reason: str = "local_proof_accepted",
        timestamp: str | None = None,
        authenticated_signer: str | None = None,
        authenticated_role: str | None = None,
        auth_message_id: str | None = None,
    ) -> dict[str, Any]:
        before = str(state_before)
        self._require_latest(facility_id, before)
        record = {
            "event_type": "state_committed",
            "actor": "facility",
            "facility_id": facility_id,
            "request_id": request_id,
            "state_before": before,
            "state_after": str(state_after),
            "parent_claim_record": parent_claim_record,
            "proof_accepted_record": proof_accepted_record,
            "reason": reason,
            "timestamp_utc": timestamp or utc_timestamp(),
        }
        if transition_id is not None:
            record["transition_id"] = transition_id
        if authenticated_signer is not None:
            record["authenticated_signer"] = authenticated_signer
        if authenticated_role is not None:
            record["authenticated_role"] = authenticated_role
        if auth_message_id is not None:
            record["auth_message_id"] = auth_message_id
        return self._append(record)

    def append_reference_state_committed(
        self,
        *,
        facility_id: str,
        request_id: int,
        state_before: int | str,
        state_after: int | str,
        parent_claim_record: int,
        validation_accepted_record: int,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Commit a benchmark ablation transition accepted by the Python checker."""
        before = str(state_before)
        self._require_latest(facility_id, before)
        return self._append(
            {
                "event_type": "state_committed",
                "actor": "facility",
                "facility_id": facility_id,
                "request_id": request_id,
                "state_before": before,
                "state_after": str(state_after),
                "parent_claim_record": parent_claim_record,
                "validation_accepted_record": validation_accepted_record,
                "reason": "reference_validation_accepted_no_zkp",
                "timestamp_utc": timestamp or utc_timestamp(),
            }
        )

    def append_event(
        self,
        *,
        event_type: str,
        timestamp: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Append a public protocol event that may or may not update facility state."""
        return self._append(
            {
                "event_type": event_type,
                **fields,
                "timestamp_utc": timestamp or utc_timestamp(),
            }
        )

    # accepted proof must start from the latest public commitment
    def _require_latest(self, facility_id: str, state_before: str) -> None:
        latest = self.latest_state(facility_id)
        if latest is None:
            raise StaleStateError(f"facility {facility_id} has no initial state")
        if latest != state_before:
            raise StaleStateError(
                f"transition starts from {state_before}, but latest state is {latest}"
            )

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        self._reject_private_fields(record)
        record = {"record_index": len(self.records), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.records.append(record)
        if "facility_id" in record and record.get("state_after") is not None:
            self._latest_state[record["facility_id"]] = record["state_after"]
        self._write_pretty_log()
        return record

    def _write_pretty_log(self) -> None:
        self.pretty_path.write_text(
            json.dumps(self.records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _reject_private_fields(record: dict[str, Any]) -> None:
        forbidden = PRIVATE_LOG_FIELDS.intersection(record)
        if forbidden:
            raise ProtocolError(f"public log record contains private fields: {sorted(forbidden)}")
