"""FastAPI application for portal orchestration and proof verification."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, status

from system.auth import signer_from_environment
from system.api_models import HealthResponse, PortalRequestResult, PortalVnrRequest
from system.facility_client import FacilityClient
from system.log_client import TransparencyLogClient
from system.http_trace import recorder_from_environment
from system.portal_engine import (
    DuplicateRequest,
    PortalEngine,
    PublicLink,
    RequestInProgress,
    UnknownFacility,
)
from system.portal_verifier import PortalProofVerifier


def _engine_from_environment() -> PortalEngine:
    config_path = Path(os.environ.get("ZK_VNE_PORTAL_CONFIG", "/app/config/portal.json"))
    data_dir = Path(os.environ.get("ZK_VNE_PORTAL_DATA_DIR", "/data"))
    verification_key_path = Path(
        os.environ.get(
            "ZK_VNE_VERIFICATION_KEY",
            "/app/build/zk/resource_feasibility_verification_key.json",
        )
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    signer = signer_from_environment()
    recorder = recorder_from_environment(actor_id="portal")
    facilities = {
        facility_id: FacilityClient(
            base_url,
            signer=signer,
            recorder=recorder,
            receiver_id=facility_id,
        )
        for facility_id, base_url in config["facilities"].items()
    }
    return PortalEngine(
        log_client=TransparencyLogClient(
            os.environ["ZK_VNE_LOG_URL"],
            signer=signer,
            recorder=recorder,
            actor_id="portal",
        ),
        facilities=facilities,
        verifier=PortalProofVerifier(
            verification_key_path=verification_key_path,
            work_dir=data_dir / "verification",
        ),
        public_links=[PublicLink(**link) for link in config.get("public_links", [])],
    )


def create_app(*, engine: PortalEngine | None = None) -> FastAPI:
    supplied_engine = engine

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = supplied_engine or _engine_from_environment()
        app.state.engine.log_client.wait_until_ready()
        yield

    app = FastAPI(title="ZK-VNE Portal", version="1.0.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(service="portal")

    @app.post(
        "/v1/requests",
        response_model=PortalRequestResult,
        status_code=status.HTTP_200_OK,
    )
    async def submit_request(request: PortalVnrRequest) -> PortalRequestResult:
        try:
            return app.state.engine.process(request)
        except DuplicateRequest as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except RequestInProgress as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except UnknownFacility as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"downstream service failed: {exc}",
            ) from exc

    return app


app = create_app()
