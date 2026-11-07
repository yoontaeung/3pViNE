"""FastAPI application for the public transparency log."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request, status

from protocol import ProtocolError
from system.auth import (
    AuthenticatedIdentity,
    AuthenticationError,
    AuthorizationError,
    RequestAuthenticator,
    require_role,
)
from system.api_models import (
    ArtifactCreate,
    ArtifactReference,
    FacilityRegistration,
    HealthResponse,
    PublicEventRequest,
    StateTransitionCreate,
)
from system.log_store import (
    ArtifactNotFound,
    InvalidTransition,
    PublicLogStore,
    RegistrationConflict,
)


EVENT_ROLES = {
    "claim_submitted": "facility",
    "proof_accepted": "portal",
    "proof_rejected": "portal",
    "request_completed": "portal",
    "request_declined": "portal",
    "request_issued": "portal",
    "residual_forwarded": "portal",
}


def create_app(
    *,
    store: PublicLogStore | None = None,
    authenticator: RequestAuthenticator | None = None,
    require_auth: bool | None = None,
) -> FastAPI:
    supplied_store = store
    supplied_authenticator = authenticator
    should_require_auth = (
        require_auth
        if require_auth is not None
        else supplied_authenticator is not None or supplied_store is None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = supplied_store or PublicLogStore(
            Path(os.environ.get("ZK_VNE_LOG_DATA_DIR", "/data"))
        )
        app.state.authenticator = supplied_authenticator
        if should_require_auth and app.state.authenticator is None:
            app.state.authenticator = RequestAuthenticator.from_config(
                Path(os.environ["ZK_VNE_TRUST_CONFIG"])
            )
        yield

    app = FastAPI(
        title="ZK-VNE Transparency Log",
        version="1.0.0",
        lifespan=lifespan,
    )

    def active_store() -> PublicLogStore:
        return app.state.store

    def authenticate(
        request: Request,
        payload: dict[str, Any],
        *,
        role: str,
    ) -> AuthenticatedIdentity | None:
        if not should_require_auth:
            return None
        try:
            identity = app.state.authenticator.verify(
                method=request.method,
                path=request.url.path,
                payload=payload,
                headers=request.headers,
            )
            require_role(identity, role)
            return identity
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

    def require_signer(identity: AuthenticatedIdentity | None, expected: str) -> None:
        if identity is not None and identity.signer_id != expected:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"signer {identity.signer_id} cannot act for {expected}",
            )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(service="transparency-log")

    @app.post("/v1/registrations", status_code=status.HTTP_201_CREATED)
    async def register(
        registration: FacilityRegistration,
        request: Request,
    ) -> dict[str, Any]:
        identity = authenticate(
            request,
            registration.model_dump(mode="json"),
            role="facility",
        )
        require_signer(identity, registration.facility_id)
        try:
            return active_store().register(registration, identity=identity)
        except RegistrationConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.post("/v1/events", status_code=status.HTTP_201_CREATED)
    async def append_event(
        event: PublicEventRequest,
        request: Request,
    ) -> dict[str, Any]:
        expected_role = EVENT_ROLES.get(event.event_type)
        if expected_role is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unsupported public event type {event.event_type}",
            )
        if event.actor != expected_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"event {event.event_type} must be authored by {expected_role}",
            )
        identity = authenticate(
            request,
            event.model_dump(mode="json"),
            role=expected_role,
        )
        if event.actor == "facility":
            facility_id = event.fields.get("facility_id")
            if not isinstance(facility_id, str):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="facility event is missing facility_id",
                )
            require_signer(identity, facility_id)
        try:
            return active_store().append_event(event, identity=identity)
        except ProtocolError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @app.get("/v1/events")
    async def events(after: int = Query(default=-1, ge=-1)) -> list[dict[str, Any]]:
        return active_store().records(after=after)

    @app.get("/v1/facilities/{facility_id}/latest")
    async def latest_state(facility_id: str) -> dict[str, str]:
        commitment = active_store().latest_state(facility_id)
        if commitment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"facility {facility_id} is not registered",
            )
        return {"facility_id": facility_id, "state_commitment": commitment}

    @app.post("/v1/transitions", status_code=status.HTTP_201_CREATED)
    async def commit_transition(
        transition: StateTransitionCreate,
        request: Request,
    ) -> dict[str, Any]:
        identity = authenticate(
            request,
            transition.model_dump(mode="json"),
            role="facility",
        )
        require_signer(identity, transition.facility_id)
        try:
            return active_store().commit_transition(transition, identity=identity)
        except InvalidTransition as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.post(
        "/v1/artifacts",
        response_model=ArtifactReference,
        status_code=status.HTTP_201_CREATED,
    )
    async def put_artifact(
        artifact: ArtifactCreate,
        request: Request,
    ) -> ArtifactReference:
        identity = authenticate(
            request,
            artifact.model_dump(mode="json"),
            role="facility",
        )
        require_signer(identity, artifact.facility_id)
        try:
            return ArtifactReference(artifact_id=active_store().put_artifact(artifact))
        except ProtocolError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @app.get("/v1/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str) -> dict[str, Any]:
        try:
            return active_store().get_artifact(artifact_id)
        except ArtifactNotFound as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return app


app = create_app()
