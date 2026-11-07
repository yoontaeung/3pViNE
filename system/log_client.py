"""Small HTTP client used by protocol actors to publish public data."""

from __future__ import annotations

import time
from typing import Any

import httpx

from system.auth import RequestSigner
from system.api_models import (
    ArtifactCreate,
    ArtifactReference,
    FacilityRegistration,
    PublicEventRequest,
    StateTransitionCreate,
)
from system.http_trace import HttpTraceRecorder, canonical_json_bytes


class TransparencyLogClient:
    def __init__(
        self,
        base_url: str,
        *,
        signer: RequestSigner | None = None,
        recorder: HttpTraceRecorder | None = None,
        actor_id: str = "protocol-actor",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self.recorder = recorder
        self.actor_id = actor_id
        self.timeout_seconds = timeout_seconds

    def wait_until_ready(self, *, attempts: int = 30, delay_seconds: float = 1.0) -> None:
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                response = httpx.get(
                    f"{self.base_url}/health",
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                return
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                time.sleep(delay_seconds)
        raise RuntimeError(f"transparency log did not become ready: {last_error}")

    def register(self, registration: FacilityRegistration) -> dict[str, Any]:
        return self._post(
            "/v1/registrations",
            registration.model_dump(mode="json"),
            semantic_type="facility_registration",
        )

    def append_event(self, event: PublicEventRequest) -> dict[str, Any]:
        return self._post(
            "/v1/events",
            event.model_dump(mode="json"),
            semantic_type=event.event_type,
            request_id=event.fields.get("request_id"),
        )

    def put_artifact(self, artifact: ArtifactCreate) -> ArtifactReference:
        value = self._post(
            "/v1/artifacts",
            artifact.model_dump(mode="json"),
            semantic_type="proof_artifact_upload",
            request_id=artifact.request_id,
        )
        return ArtifactReference.model_validate(value)

    def latest_state(self, facility_id: str) -> str:
        path = f"/v1/facilities/{facility_id}/latest"
        started = time.perf_counter()
        response = httpx.get(
            f"{self.base_url}{path}",
            timeout=self.timeout_seconds,
        )
        self._record(
            response=response,
            started=started,
            semantic_type="latest_state_read",
            method="GET",
            path=path,
        )
        response.raise_for_status()
        return str(response.json()["state_commitment"])

    def events(self, *, after: int = -1) -> list[dict[str, Any]]:
        path = "/v1/events"
        started = time.perf_counter()
        response = httpx.get(
            f"{self.base_url}{path}",
            params={"after": after},
            timeout=self.timeout_seconds,
        )
        self._record(
            response=response,
            started=started,
            semantic_type="log_events_read",
            method="GET",
            path=f"{path}?after={after}",
        )
        response.raise_for_status()
        return response.json()

    def artifact(self, artifact_id: str, *, request_id: int | None = None) -> ArtifactCreate:
        path = f"/v1/artifacts/{artifact_id}"
        started = time.perf_counter()
        response = httpx.get(
            f"{self.base_url}{path}",
            timeout=self.timeout_seconds,
        )
        self._record(
            response=response,
            started=started,
            semantic_type="proof_artifact_fetch",
            method="GET",
            path=path,
            request_id=request_id,
        )
        response.raise_for_status()
        return ArtifactCreate.model_validate(response.json())

    def commit_transition(self, transition: StateTransitionCreate) -> dict[str, Any]:
        return self._post(
            "/v1/transitions",
            transition.model_dump(mode="json"),
            semantic_type="state_transition_publish",
            request_id=transition.request_id,
        )

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        semantic_type: str,
        request_id: int | None = None,
    ) -> dict[str, Any]:
        headers = (
            self.signer.headers("POST", path, payload)
            if self.signer is not None
            else {}
        )
        headers["Content-Type"] = "application/json"
        started = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}{path}",
            content=canonical_json_bytes(payload),
            headers=headers,
            timeout=self.timeout_seconds,
        )
        self._record(
            response=response,
            started=started,
            semantic_type=semantic_type,
            method="POST",
            path=path,
            payload=payload,
            headers=headers,
            request_id=request_id,
        )
        response.raise_for_status()
        return response.json()

    def _record(
        self,
        *,
        response: httpx.Response,
        started: float,
        semantic_type: str,
        method: str,
        path: str,
        payload: object | None = None,
        headers: dict[str, str] | None = None,
        request_id: int | None = None,
    ) -> None:
        if self.recorder is None:
            return
        self.recorder.record(
            receiver="transparency-log",
            semantic_type=semantic_type,
            method=method,
            path=path,
            request_payload=payload,
            request_headers=headers,
            response=response,
            elapsed_seconds=time.perf_counter() - started,
            request_id=request_id,
        )
