"""Portal-side HTTP client for one facility service."""

from __future__ import annotations

import time

import httpx

from system.auth import RequestSigner
from system.api_models import (
    ClaimRequest,
    ClaimResponse,
    CommitRequest,
    CommitResponse,
    FacilityRegistration,
)
from system.http_trace import HttpTraceRecorder, canonical_json_bytes


class FacilityClient:
    def __init__(
        self,
        base_url: str,
        *,
        signer: RequestSigner | None = None,
        recorder: HttpTraceRecorder | None = None,
        receiver_id: str = "facility",
        timeout_seconds: float = 1800.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self.recorder = recorder
        self.receiver_id = receiver_id
        self.timeout_seconds = timeout_seconds

    def create_claim(self, request: ClaimRequest) -> ClaimResponse:
        path = "/v1/claims"
        payload = request.model_dump(mode="json")
        headers = self._headers(path, payload) or {}
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
            semantic_type="claim_request",
            method="POST",
            path=path,
            payload=payload,
            headers=headers,
            request_id=request.request_id,
        )
        response.raise_for_status()
        return ClaimResponse.model_validate(response.json())

    def registration(self) -> FacilityRegistration:
        path = "/v1/registration"
        started = time.perf_counter()
        response = httpx.get(
            f"{self.base_url}{path}",
            timeout=self.timeout_seconds,
        )
        self._record(
            response=response,
            started=started,
            semantic_type="registration_read",
            method="GET",
            path=path,
        )
        response.raise_for_status()
        return FacilityRegistration.model_validate(response.json())

    def commit(self, request: CommitRequest) -> CommitResponse:
        path = f"/v1/transitions/{request.transition_id}/commit"
        payload = request.model_dump(mode="json")
        headers = self._headers(path, payload) or {}
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
            semantic_type="state_commit_command",
            method="POST",
            path=path,
            payload=payload,
            headers=headers,
            request_id=request.request_id if hasattr(request, "request_id") else None,
        )
        response.raise_for_status()
        return CommitResponse.model_validate(response.json())

    def _headers(self, path: str, payload: dict[str, object]) -> dict[str, str] | None:
        if self.signer is None:
            return None
        return self.signer.headers("POST", path, payload)

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
            receiver=self.receiver_id,
            semantic_type=semantic_type,
            method=method,
            path=path,
            request_payload=payload,
            request_headers=headers,
            response=response,
            elapsed_seconds=time.perf_counter() - started,
            request_id=request_id,
        )
