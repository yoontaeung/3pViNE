"""Persistent storage owned by the transparency-log service."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from protocol import PRIVATE_LOG_FIELDS, ProtocolError, StaleStateError, TransparencyLog
from system.auth import AuthenticatedIdentity
from system.api_models import (
    ArtifactCreate,
    FacilityRegistration,
    PublicEventRequest,
    StateTransitionCreate,
)


class RegistrationConflict(ProtocolError):
    """Raised when a facility tries to replace its registered initial state."""


class ArtifactNotFound(ProtocolError):
    """Raised when a public proof artifact does not exist."""


class InvalidTransition(ProtocolError):
    """Raised when a state update is not backed by an accepted proof record."""


def _private_field_paths(value: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in PRIVATE_LOG_FIELDS:
                paths.append(path)
            paths.extend(_private_field_paths(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(_private_field_paths(nested, prefix=f"{prefix}[{index}]"))
    return paths


class PublicLogStore:
    """Thread-safe facade over the append-only JSONL log and public artifacts."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.artifact_dir = data_dir / "artifacts"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.log = TransparencyLog(data_dir / "transparency_log.jsonl")
        self._lock = threading.Lock()

    def register(
        self,
        registration: FacilityRegistration,
        *,
        identity: AuthenticatedIdentity | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            latest = self.log.latest_state(registration.facility_id)
            if latest is not None:
                if latest != registration.state_commitment:
                    raise RegistrationConflict(
                        f"facility {registration.facility_id} is already registered "
                        f"at state {latest}"
                    )
                for record in reversed(self.log.records):
                    if (
                        record.get("facility_id") == registration.facility_id
                        and record.get("state_after") == latest
                    ):
                        return record
                raise RegistrationConflict(
                    f"facility {registration.facility_id} already has public state "
                    "without a registration"
                )

            return self.log.append_event(
                event_type="facility_registered",
                actor="facility",
                facility_id=registration.facility_id,
                state_before=None,
                state_after=registration.state_commitment,
                public_endpoint=(
                    registration.public_endpoint.model_dump()
                    if registration.public_endpoint is not None
                    else None
                ),
                api_version=registration.api_version,
                **self._auth_fields(identity),
            )

    def append_event(
        self,
        event: PublicEventRequest,
        *,
        identity: AuthenticatedIdentity | None = None,
    ) -> dict[str, Any]:
        reserved_fields = {
            "actor",
            "api_version",
            "auth_message_id",
            "authenticated_role",
            "authenticated_signer",
            "event_type",
            "record_index",
            "timestamp_utc",
        }
        conflicting = reserved_fields.intersection(event.fields)
        if conflicting:
            raise ProtocolError(
                f"public event contains reserved fields: {sorted(conflicting)}"
            )
        if "state_after" in event.fields:
            raise ProtocolError(
                "generic public events cannot update facility state; use a transition endpoint"
            )
        self._reject_private_data(event.fields)
        with self._lock:
            existing = self._authenticated_record(identity)
            if existing is not None:
                return existing
            return self.log.append_event(
                event_type=event.event_type,
                actor=event.actor,
                api_version=event.api_version,
                **self._auth_fields(identity),
                **event.fields,
            )

    def records(self, *, after: int = -1) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self.log.records if record["record_index"] > after]

    def latest_state(self, facility_id: str) -> str | None:
        with self._lock:
            return self.log.latest_state(facility_id)

    def commit_transition(
        self,
        transition: StateTransitionCreate,
        *,
        identity: AuthenticatedIdentity | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            existing = self._transition_record(transition.transition_id)
            if existing is not None:
                expected = {
                    "facility_id": transition.facility_id,
                    "request_id": transition.request_id,
                    "state_before": transition.state_before,
                    "state_after": transition.state_after,
                    "parent_claim_record": transition.claim_record_index,
                    "proof_accepted_record": transition.verification_record_index,
                }
                if any(existing.get(key) != value for key, value in expected.items()):
                    raise InvalidTransition("transition ID was already used with different fields")
                return existing

            claim = self._record_at(transition.claim_record_index, "claim")
            verification = self._record_at(transition.verification_record_index, "verification")
            if claim.get("event_type") != "claim_submitted":
                raise InvalidTransition("claim record is not a claim_submitted event")
            if claim.get("facility_id") != transition.facility_id:
                raise InvalidTransition("claim facility does not match transition facility")
            if claim.get("request_id") != transition.request_id:
                raise InvalidTransition("claim request does not match transition request")
            if claim.get("state_before") != transition.state_before:
                raise InvalidTransition("claim state_before does not match transition")
            if claim.get("proposed_state_after") != transition.state_after:
                raise InvalidTransition("claim proposed state does not match transition")
            if claim.get("transition_id") != transition.transition_id:
                raise InvalidTransition("claim transition ID does not match transition")

            if identity is not None:
                if claim.get("authenticated_role") != "facility":
                    raise InvalidTransition("claim was not authenticated as a facility event")
                if claim.get("authenticated_signer") != transition.facility_id:
                    raise InvalidTransition("claim signer does not match transition facility")

            if verification.get("event_type") != "proof_accepted":
                raise InvalidTransition("verification record is not a proof_accepted event")
            if verification.get("actor") != "portal" or verification.get("result") != "accepted":
                raise InvalidTransition("verification record was not accepted by the portal")
            if verification.get("parent_claim_record") != transition.claim_record_index:
                raise InvalidTransition("verification does not refer to the transition claim")
            if verification.get("facility_id") != transition.facility_id:
                raise InvalidTransition("verification facility does not match transition")
            if verification.get("request_id") != transition.request_id:
                raise InvalidTransition("verification request does not match transition")
            if verification.get("transition_id") != transition.transition_id:
                raise InvalidTransition("verification transition ID does not match transition")
            if identity is not None and verification.get("authenticated_role") != "portal":
                raise InvalidTransition("verification was not authenticated as a portal event")

            try:
                return self.log.append_state_committed(
                    facility_id=transition.facility_id,
                    request_id=transition.request_id,
                    state_before=transition.state_before,
                    state_after=transition.state_after,
                    parent_claim_record=transition.claim_record_index,
                    proof_accepted_record=transition.verification_record_index,
                    transition_id=transition.transition_id,
                    **self._auth_fields(identity),
                )
            except StaleStateError as exc:
                raise InvalidTransition(str(exc)) from exc

    def put_artifact(self, artifact: ArtifactCreate) -> str:
        value = artifact.model_dump(mode="json")
        self._reject_private_data(value)
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        artifact_id = hashlib.sha256(encoded).hexdigest()
        destination = self.artifact_dir / f"{artifact_id}.json"
        with self._lock:
            if not destination.exists():
                temporary = destination.with_suffix(".tmp")
                temporary.write_bytes(encoded + b"\n")
                temporary.replace(destination)
        return artifact_id

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        if len(artifact_id) != 64 or any(char not in "0123456789abcdef" for char in artifact_id):
            raise ArtifactNotFound("invalid artifact identifier")
        path = self.artifact_dir / f"{artifact_id}.json"
        if not path.exists():
            raise ArtifactNotFound(f"artifact {artifact_id} does not exist")
        return json.loads(path.read_text(encoding="utf-8"))

    def _transition_record(self, transition_id: str) -> dict[str, Any] | None:
        for record in reversed(self.log.records):
            if (
                record.get("event_type") == "state_committed"
                and record.get("transition_id") == transition_id
            ):
                return record
        return None

    def _authenticated_record(
        self,
        identity: AuthenticatedIdentity | None,
    ) -> dict[str, Any] | None:
        if identity is None:
            return None
        for record in reversed(self.log.records):
            if record.get("auth_message_id") == identity.message_id:
                if record.get("authenticated_signer") != identity.signer_id:
                    raise ProtocolError("authentication message ID belongs to another signer")
                return record
        return None

    @staticmethod
    def _auth_fields(identity: AuthenticatedIdentity | None) -> dict[str, str]:
        if identity is None:
            return {}
        return {
            "authenticated_signer": identity.signer_id,
            "authenticated_role": identity.role,
            "auth_message_id": identity.message_id,
        }

    def _record_at(self, index: int, name: str) -> dict[str, Any]:
        if index >= len(self.log.records):
            raise InvalidTransition(f"{name} record {index} does not exist")
        return self.log.records[index]

    @staticmethod
    def _reject_private_data(value: Any) -> None:
        forbidden = _private_field_paths(value)
        if forbidden:
            raise ProtocolError(f"public data contains private fields: {sorted(forbidden)}")
