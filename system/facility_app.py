"""FastAPI application exposing one facility's private VNE prover."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request as HttpRequest, status

from system.auth import (
    AuthenticationError,
    AuthorizationError,
    RequestAuthenticator,
    authenticator_from_environment,
    require_role,
    signer_from_environment,
)
from system.api_models import (
    ArtifactCreate,
    ClaimRequest,
    ClaimResponse,
    CommitRequest,
    CommitResponse,
    FacilityRegistration,
    HealthResponse,
    PublicEventRequest,
)
from system.facility_engine import FacilityEngine, FacilityStateMismatch, UnknownTransition
from system.log_client import TransparencyLogClient
from system.http_trace import recorder_from_environment
from system.proof_runtime import ClaimProofRuntime


def _engine_from_environment() -> FacilityEngine:
    config_path = Path(os.environ.get("ZK_VNE_FACILITY_CONFIG", "/app/config/facility.json"))
    data_dir = Path(os.environ.get("ZK_VNE_FACILITY_DATA_DIR", "/data"))
    proof_mode = os.environ.get("ZK_VNE_PROOF_MODE", "groth16")
    if proof_mode not in {"groth16", "reference"}:
        raise ValueError("ZK_VNE_PROOF_MODE must be groth16 or reference")
    runtime = ClaimProofRuntime(
        proof_mode=proof_mode,
        work_dir=data_dir / "claims",
    )
    return FacilityEngine.from_config(
        config_path=config_path,
        proof_runtime=runtime,
        state_path=data_dir / "facility_state.json",
    )


def create_app(
    *,
    engine: FacilityEngine | None = None,
    log_client: TransparencyLogClient | None = None,
    authenticator: RequestAuthenticator | None = None,
    auto_register: bool | None = None,
    require_portal_auth: bool | None = None,
) -> FastAPI:
    supplied_engine = engine
    supplied_log_client = log_client
    supplied_authenticator = authenticator
    should_require_portal_auth = (
        require_portal_auth
        if require_portal_auth is not None
        else supplied_authenticator is not None or supplied_engine is None
    )
    should_register = (
        auto_register
        if auto_register is not None
        else os.environ.get("ZK_VNE_AUTO_REGISTER", "1") == "1"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = supplied_engine or _engine_from_environment()
        app.state.authenticator = supplied_authenticator
        if should_require_portal_auth and app.state.authenticator is None:
            app.state.authenticator = authenticator_from_environment()
        app.state.log_client = supplied_log_client
        if app.state.log_client is None and os.environ.get("ZK_VNE_LOG_URL"):
            recorder = recorder_from_environment(actor_id=app.state.engine.facility_id)
            app.state.log_client = TransparencyLogClient(
                os.environ["ZK_VNE_LOG_URL"],
                signer=signer_from_environment(),
                recorder=recorder,
                actor_id=app.state.engine.facility_id,
            )
        if should_register and app.state.log_client is not None:
            app.state.log_client.wait_until_ready()
            for transition in app.state.engine.unlogged_state_transitions():
                record = app.state.log_client.commit_transition(transition)
                app.state.engine.mark_state_record(
                    transition.transition_id,
                    record["record_index"],
                )
            app.state.log_client.register(app.state.engine.registration())
        yield

    app = FastAPI(title="ZK-VNE Facility", version="1.0.0", lifespan=lifespan)

    def active_engine() -> FacilityEngine:
        return app.state.engine

    def authenticate_portal(request: HttpRequest, payload: dict[str, object]) -> None:
        if not should_require_portal_auth:
            return
        try:
            identity = app.state.authenticator.verify(
                method=request.method,
                path=request.url.path,
                payload=payload,
                headers=request.headers,
            )
            require_role(identity, "portal")
        except AuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
            ) from exc
        except AuthorizationError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(exc),
            ) from exc

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(service="facility")

    @app.get("/v1/registration", response_model=FacilityRegistration)
    async def registration() -> FacilityRegistration:
        return active_engine().registration()

    @app.post("/v1/claims", response_model=ClaimResponse)
    async def create_claim(
        request: ClaimRequest,
        http_request: HttpRequest,
    ) -> ClaimResponse:
        authenticate_portal(http_request, request.model_dump(mode="json"))
        try:
            response = active_engine().create_claim(request)
        except FacilityStateMismatch as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        client: TransparencyLogClient | None = app.state.log_client
        if client is None:
            return response

        artifact_id = None
        if response.proof_bundle is not None and response.commitments is not None:
            artifact_id = client.put_artifact(
                ArtifactCreate(
                    facility_id=response.facility_id,
                    request_id=response.request_id,
                    claim_type=response.claim_type,
                    commitments=response.commitments,
                    proof_bundle=response.proof_bundle,
                )
            ).artifact_id

        event = client.append_event(
            PublicEventRequest(
                event_type="claim_submitted",
                actor="facility",
                fields={
                    "facility_id": response.facility_id,
                    "request_id": response.request_id,
                    "request_scope": response.request_scope,
                    "claim_type": response.claim_type,
                    "state_before": response.state_before,
                    "proposed_state_after": response.proposed_state_after,
                    "commitments": (
                        response.commitments.model_dump(mode="json")
                        if response.commitments is not None
                        else None
                    ),
                    "covered_virtual_nodes": response.covered_virtual_nodes,
                    "public_endpoint_id": response.public_endpoint_id,
                    "public_endpoint_index": response.public_endpoint_index,
                    "residual": (
                        response.residual.model_dump(mode="json")
                        if response.residual is not None
                        else None
                    ),
                    "artifact_id": artifact_id,
                    "transition_id": response.transition_id,
                    "decline_reason": response.decline_reason,
                    "proof_mode": response.proof_mode,
                    "timings_seconds": response.timings_seconds,
                },
            )
        )
        return response.model_copy(
            update={
                "artifact_id": artifact_id,
                "claim_record_index": event["record_index"],
                "proof_bundle": None,
            }
        )

    @app.post("/v1/transitions/{transition_id}/commit", response_model=CommitResponse)
    async def commit_transition(
        transition_id: str,
        request: CommitRequest,
        http_request: HttpRequest,
    ) -> CommitResponse:
        authenticate_portal(http_request, request.model_dump(mode="json"))
        if transition_id != request.transition_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="transition path and request body disagree",
            )
        try:
            response = active_engine().commit(request)
        except UnknownTransition as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except FacilityStateMismatch as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        client: TransparencyLogClient | None = app.state.log_client
        if client is None:
            return response
        transition = active_engine().state_transition(transition_id)
        record = client.commit_transition(transition)
        return active_engine().mark_state_record(transition_id, record["record_index"])

    return app


app = create_app()
