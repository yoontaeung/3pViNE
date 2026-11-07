"""Tests for the independently deployed ZK-VNE service roles."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from protocol_demo import load_facilities
from system.api_models import (
    ArtifactCreate,
    BoundaryDemandPayload,
    ClaimRequest,
    ClaimResponse,
    CommitResponse,
    CommitRequest,
    FacilityRegistration,
    GraphPayload,
    PortalVnrRequest,
    ProofBundle,
    PublicCommitments,
    PublicEndpointPayload,
    PublicEventRequest,
    ResidualPayload,
    StateTransitionCreate,
)
from system.auth import AuthenticationError, RequestAuthenticator, RequestSigner
from system.circuit_request import build_local_circuit_vnr
from system.facility_app import create_app as create_facility_app
from system.facility_engine import FacilityEngine, FacilityStateMismatch
from system.facility_store import FacilityPrivateStore
from system.graph_codec import graph_from_payload
from system.http_trace import HttpTraceRecorder
from system.log_app import create_app as create_log_app
from system.log_store import PublicLogStore
from system.portal_engine import (
    DuplicateRequest,
    PortalEngine,
    PublicLink,
    RequestInProgress,
)
from system.portal_verifier import PortalProofVerifier, ProofRejected
from system.residual_validation import (
    InvalidResidual,
    residual_commitment,
    validate_single_cut_residual,
)
from system.proof_runtime import GeneratedClaimProof
from zk_encoding import (
    substrate_adjacency_matrix,
    substrate_bandwidth_matrix,
    substrate_cpu_vector,
    substrate_memory_vector,
)


ROOT = Path(__file__).resolve().parents[1]
FACILITY_A_CONFIG = ROOT / "configs/system/facility-a.json"


def authentication_fixture() -> tuple[
    RequestSigner,
    RequestSigner,
    RequestSigner,
    RequestAuthenticator,
]:
    portal_key = Ed25519PrivateKey.generate()
    facility_a_key = Ed25519PrivateKey.generate()
    facility_b_key = Ed25519PrivateKey.generate()
    return (
        RequestSigner("portal", portal_key),
        RequestSigner("facility-A", facility_a_key),
        RequestSigner("facility-B", facility_b_key),
        RequestAuthenticator(
            {
                "portal": ("portal", portal_key.public_key()),
                "facility-A": ("facility", facility_a_key.public_key()),
                "facility-B": ("facility", facility_b_key.public_key()),
            }
        ),
    )


@asynccontextmanager
async def app_client(app: object) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


class FakeProofRuntime:
    proof_mode = "reference"

    def generate(self, **values: object) -> GeneratedClaimProof:
        substrate = values["substrate"]
        randomness = values["randomness"]
        assert isinstance(randomness, dict)
        zk_input = {
            "C_r": 10001,
            "C_S": int(randomness["rho_S"]),
            "C_M": 10003,
            "C_S_next": int(randomness["rho_S_next"]),
            "sCpuAfter": substrate_cpu_vector(substrate),
            "sMemAfter": substrate_memory_vector(substrate),
            "sAdj": substrate_adjacency_matrix(substrate),
            "sBwAfter": substrate_bandwidth_matrix(substrate),
        }
        return GeneratedClaimProof(
            zk_input=zk_input,
            proof_bundle=None,
            timings_seconds={"encoding": 0.001, "reference_validation": 0.001},
        )


def facility_engine(*, state_path: Path | None = None) -> FacilityEngine:
    facilities, _ = load_facilities(FACILITY_A_CONFIG)
    return FacilityEngine(
        facility=facilities["facility-A"],
        proof_runtime=FakeProofRuntime(),  # type: ignore[arg-type]
        state_commitment_fn=lambda substrate, *, rho_S: int(rho_S),
        private_store=FacilityPrivateStore(state_path) if state_path is not None else None,
    )


def full_request() -> ClaimRequest:
    config = json.loads((ROOT / "configs/portal_requests.json").read_text(encoding="utf-8"))
    request = config["requests"][0]
    return ClaimRequest(
        request_id=request["request_id"],
        request_scope="original",
        expected_state="7101",
        vnr=GraphPayload.model_validate(request["vnr"]),
    )


class MemoryPortalLog:
    """Small in-memory log client for deterministic portal recovery tests."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = [dict(record) for record in records]

    def events(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self.records]

    def append_event(self, event: Any) -> dict[str, Any]:
        return self.append_raw(
            event_type=event.event_type,
            actor=event.actor,
            **event.fields,
        )

    def append_raw(self, **fields: Any) -> dict[str, Any]:
        record = {"record_index": len(self.records), **fields}
        self.records.append(record)
        return record

    def latest_state(self, facility_id: str) -> str:
        for record in reversed(self.records):
            if record.get("facility_id") == facility_id and record.get("state_after"):
                return str(record["state_after"])
        raise AssertionError(f"no state for {facility_id}")

    @staticmethod
    def artifact(artifact_id: str) -> object:
        return {"artifact_id": artifact_id}


class RecoveryFacility:
    def __init__(self, log: MemoryPortalLog) -> None:
        self.log = log
        self.create_calls = 0
        self.commit_calls = 0

    @staticmethod
    def registration() -> FacilityRegistration:
        return FacilityRegistration(
            facility_id="facility-A",
            state_commitment="S0",
        )

    def create_claim(self, request: ClaimRequest) -> ClaimResponse:
        self.create_calls += 1
        raise AssertionError("a persisted claim must be reused")

    def commit(self, request: CommitRequest) -> CommitResponse:
        self.commit_calls += 1
        record = self.log.append_raw(
            event_type="state_committed",
            actor="facility",
            facility_id="facility-A",
            request_id=990,
            state_before=request.expected_state_before,
            state_after=request.expected_state_after,
            parent_claim_record=request.claim_record_index,
            proof_accepted_record=request.verification_record_index,
            transition_id=request.transition_id,
        )
        return CommitResponse(
            facility_id="facility-A",
            request_id=990,
            transition_id=request.transition_id,
            state_before=request.expected_state_before,
            state_after=request.expected_state_after,
            state_record_index=record["record_index"],
        )


class RecoveryVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify_claim(self, **values: object) -> float:
        self.calls += 1
        return 0.125


def recovery_fixture(stage: str) -> tuple[PortalVnrRequest, list[dict[str, Any]]]:
    fixture = json.loads(
        (ROOT / "configs/portal_requests.json").read_text(encoding="utf-8")
    )["requests"][0]
    request = PortalVnrRequest.model_validate(fixture).model_copy(
        update={"request_id": 990}
    )
    records: list[dict[str, Any]] = [
        {
            "record_index": 0,
            "event_type": "facility_registered",
            "actor": "facility",
            "facility_id": "facility-A",
            "state_before": None,
            "state_after": "S0",
        },
        {
            "record_index": 1,
            "event_type": "request_issued",
            "actor": "portal",
            "request_id": request.request_id,
            "request_commitment": PortalEngine._request_digest(request),
            "facility_order": request.facility_order,
            "vnr": request.vnr.model_dump(mode="json"),
        },
        {
            "record_index": 2,
            "event_type": "claim_submitted",
            "actor": "facility",
            "request_id": request.request_id,
            "request_scope": "original",
            "facility_id": "facility-A",
            "claim_type": "full",
            "state_before": "S0",
            "proposed_state_after": "S1",
            "commitments": {
                "request": "R",
                "substrate_before": "S0",
                "embedding": "M",
                "substrate_after": "S1",
            },
            "covered_virtual_nodes": [node.id for node in request.vnr.nodes],
            "artifact_id": "artifact-990",
            "transition_id": "transition-990",
            "proof_mode": "groth16",
            "timings_seconds": {"prove": 1.5},
        },
    ]
    if stage in {"proof", "state", "terminal"}:
        records.append(
            {
                "record_index": 3,
                "event_type": "proof_accepted",
                "actor": "portal",
                "request_id": request.request_id,
                "facility_id": "facility-A",
                "parent_claim_record": 2,
                "artifact_id": "artifact-990",
                "transition_id": "transition-990",
                "result": "accepted",
                "verification_seconds": 0.25,
            }
        )
    if stage in {"state", "terminal"}:
        records.append(
            {
                "record_index": 4,
                "event_type": "state_committed",
                "actor": "facility",
                "request_id": request.request_id,
                "facility_id": "facility-A",
                "state_before": "S0",
                "state_after": "S1",
                "parent_claim_record": 2,
                "proof_accepted_record": 3,
                "transition_id": "transition-990",
            }
        )
    if stage == "terminal":
        records.append(
            {
                "record_index": 5,
                "event_type": "request_completed",
                "actor": "portal",
                "request_id": request.request_id,
                "request_commitment": PortalEngine._request_digest(request),
                "claim_type": "full",
                "involved_facilities": ["facility-A"],
                "final_state_commitments": {"facility-A": "S1"},
                "timings_seconds": {"online_total": 2.0},
            }
        )
    return request, records


class AuthenticationTests(unittest.TestCase):
    def test_http_trace_records_sizes_and_hashes_without_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            recorder = HttpTraceRecorder(path, actor_id="portal")
            response = httpx.Response(200, content=b'{"ok":true}')
            recorder.record(
                receiver="facility-A",
                semantic_type="claim_request",
                method="POST",
                path="/v1/claims",
                request_payload={"private-looking-value": "must-not-be-copied"},
                request_headers={"X-ZK-VNE-Signer": "portal"},
                response=response,
                elapsed_seconds=0.25,
                request_id=7,
            )
            raw = path.read_text(encoding="utf-8")
            record = json.loads(raw)
            self.assertNotIn("must-not-be-copied", raw)
            self.assertEqual(record["request_id"], 7)
            self.assertGreater(record["request_body_bytes"], 0)
            self.assertEqual(record["response_body_bytes"], len(response.content))

    def test_demo_identity_files_match_the_public_trust_store(self) -> None:
        identities = ROOT / "configs/system/identities"
        signer = RequestSigner.from_pem(
            "portal",
            identities / "portal-demo-private.pem",
        )
        authenticator = RequestAuthenticator.from_config(
            identities / "trusted-signers.json"
        )
        payload = {"request_id": 1}

        identity = authenticator.verify(
            method="POST",
            path="/v1/events",
            payload=payload,
            headers=signer.headers("POST", "/v1/events", payload),
        )

        self.assertEqual(identity.signer_id, "portal")

    def test_signature_binds_method_path_and_payload(self) -> None:
        portal, _, _, authenticator = authentication_fixture()
        payload = {"request_id": 7, "status": "accepted"}
        headers = portal.headers("POST", "/v1/events", payload)

        identity = authenticator.verify(
            method="POST",
            path="/v1/events",
            payload=payload,
            headers=headers,
        )

        self.assertEqual(identity.signer_id, "portal")
        self.assertEqual(identity.role, "portal")
        with self.assertRaises(AuthenticationError):
            authenticator.verify(
                method="POST",
                path="/v1/events",
                payload={"request_id": 8, "status": "accepted"},
                headers=headers,
            )

    def test_log_rejects_unsigned_and_impersonated_registration(self) -> None:
        _, facility_a, facility_b, authenticator = authentication_fixture()
        registration = FacilityRegistration(
            facility_id="facility-A",
            state_commitment="S0",
        ).model_dump(mode="json")
        with tempfile.TemporaryDirectory() as tmp:
            app = create_log_app(
                store=PublicLogStore(Path(tmp)),
                authenticator=authenticator,
                require_auth=True,
            )

            async def exercise() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
                async with app_client(app) as client:
                    unsigned = await client.post("/v1/registrations", json=registration)
                    impersonated = await client.post(
                        "/v1/registrations",
                        json=registration,
                        headers=facility_b.headers(
                            "POST", "/v1/registrations", registration
                        ),
                    )
                    accepted = await client.post(
                        "/v1/registrations",
                        json=registration,
                        headers=facility_a.headers(
                            "POST", "/v1/registrations", registration
                        ),
                    )
                    return unsigned, impersonated, accepted

            unsigned, impersonated, accepted = asyncio.run(exercise())

        self.assertEqual(unsigned.status_code, 401)
        self.assertEqual(impersonated.status_code, 403)
        self.assertEqual(accepted.status_code, 201)
        self.assertEqual(accepted.json()["authenticated_signer"], "facility-A")

    def test_signed_event_retry_is_idempotent_and_roles_are_enforced(self) -> None:
        portal, facility_a, _, authenticator = authentication_fixture()
        event = {
            "api_version": "v1",
            "event_type": "request_issued",
            "actor": "portal",
            "fields": {"request_id": 44},
        }
        with tempfile.TemporaryDirectory() as tmp:
            app = create_log_app(
                store=PublicLogStore(Path(tmp)),
                authenticator=authenticator,
                require_auth=True,
            )

            async def exercise() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
                async with app_client(app) as client:
                    wrong_role = await client.post(
                        "/v1/events",
                        json=event,
                        headers=facility_a.headers("POST", "/v1/events", event),
                    )
                    headers = portal.headers("POST", "/v1/events", event)
                    first = await client.post("/v1/events", json=event, headers=headers)
                    replay = await client.post("/v1/events", json=event, headers=headers)
                    return wrong_role, first, replay

            wrong_role, first, replay = asyncio.run(exercise())

        self.assertEqual(wrong_role.status_code, 403)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(first.json()["record_index"], replay.json()["record_index"])
        self.assertEqual(first.json()["authenticated_signer"], "portal")

    def test_authenticated_claim_proof_and_transition_advance_state(self) -> None:
        portal, facility_a, _, authenticator = authentication_fixture()
        registration = FacilityRegistration(
            facility_id="facility-A",
            state_commitment="S0",
        ).model_dump(mode="json")
        claim = PublicEventRequest(
            event_type="claim_submitted",
            actor="facility",
            fields={
                "facility_id": "facility-A",
                "request_id": 55,
                "state_before": "S0",
                "proposed_state_after": "S1",
                "transition_id": "transition-55",
            },
        ).model_dump(mode="json")

        with tempfile.TemporaryDirectory() as tmp:
            app = create_log_app(
                store=PublicLogStore(Path(tmp)),
                authenticator=authenticator,
                require_auth=True,
            )

            async def exercise() -> tuple[httpx.Response, httpx.Response]:
                async with app_client(app) as client:
                    await client.post(
                        "/v1/registrations",
                        json=registration,
                        headers=facility_a.headers(
                            "POST", "/v1/registrations", registration
                        ),
                    )
                    claim_response = await client.post(
                        "/v1/events",
                        json=claim,
                        headers=facility_a.headers("POST", "/v1/events", claim),
                    )
                    proof = PublicEventRequest(
                        event_type="proof_accepted",
                        actor="portal",
                        fields={
                            "facility_id": "facility-A",
                            "request_id": 55,
                            "parent_claim_record": claim_response.json()["record_index"],
                            "transition_id": "transition-55",
                            "result": "accepted",
                        },
                    ).model_dump(mode="json")
                    proof_response = await client.post(
                        "/v1/events",
                        json=proof,
                        headers=portal.headers("POST", "/v1/events", proof),
                    )
                    transition = StateTransitionCreate(
                        facility_id="facility-A",
                        request_id=55,
                        transition_id="transition-55",
                        state_before="S0",
                        state_after="S1",
                        claim_record_index=claim_response.json()["record_index"],
                        verification_record_index=proof_response.json()["record_index"],
                    ).model_dump(mode="json")
                    committed = await client.post(
                        "/v1/transitions",
                        json=transition,
                        headers=facility_a.headers("POST", "/v1/transitions", transition),
                    )
                    latest = await client.get("/v1/facilities/facility-A/latest")
                    return committed, latest

            committed, latest = asyncio.run(exercise())

        self.assertEqual(committed.status_code, 201, committed.text)
        self.assertEqual(committed.json()["authenticated_signer"], "facility-A")
        self.assertEqual(committed.json()["authenticated_role"], "facility")
        self.assertEqual(latest.json()["state_commitment"], "S1")


class TransparencyLogServiceTests(unittest.TestCase):
    def test_registration_is_idempotent_and_latest_state_is_queryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_log_app(store=PublicLogStore(Path(tmp)))
            registration = {
                "api_version": "v1",
                "facility_id": "facility-A",
                "state_commitment": "S0",
                "public_endpoint": {
                    "endpoint_id": "A-public-0",
                    "facility_id": "facility-A",
                },
            }
            async def exercise() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
                async with app_client(app) as client:
                    first = await client.post("/v1/registrations", json=registration)
                    second = await client.post("/v1/registrations", json=registration)
                    latest = await client.get("/v1/facilities/facility-A/latest")
                    return first, second, latest

            first, second, latest = asyncio.run(exercise())

            self.assertEqual(first.status_code, 201)
            self.assertEqual(second.status_code, 201)
            self.assertEqual(first.json()["record_index"], second.json()["record_index"])
            self.assertEqual(latest.json()["state_commitment"], "S0")

    def test_conflicting_registration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_log_app(store=PublicLogStore(Path(tmp)))
            async def exercise() -> httpx.Response:
                async with app_client(app) as client:
                    await client.post(
                        "/v1/registrations",
                        json={
                            "facility_id": "facility-A",
                            "state_commitment": "S0",
                        },
                    )
                    return await client.post(
                        "/v1/registrations",
                        json={
                            "facility_id": "facility-A",
                            "state_commitment": "S1",
                        },
                    )

            conflict = asyncio.run(exercise())

            self.assertEqual(conflict.status_code, 409)

    def test_nested_private_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_log_app(store=PublicLogStore(Path(tmp)))
            async def exercise() -> httpx.Response:
                async with app_client(app) as client:
                    return await client.post(
                        "/v1/events",
                        json={
                            "event_type": "claim_submitted",
                            "actor": "facility",
                            "fields": {"public": {"node_mapping": {"v0": "s0"}}},
                        },
                    )

            response = asyncio.run(exercise())

            self.assertEqual(response.status_code, 422)

    def test_public_proof_artifacts_are_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_log_app(store=PublicLogStore(Path(tmp)))
            artifact = ArtifactCreate(
                facility_id="facility-A",
                request_id=100,
                claim_type="full",
                commitments=PublicCommitments(
                    request="R",
                    substrate_before="S0",
                    embedding="M",
                    substrate_after="S1",
                ),
                proof_bundle=ProofBundle(
                    proof={"pi_a": ["1", "2", "1"]},
                    public_inputs=["1", "2"],
                ),
            )
            async def exercise() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
                async with app_client(app) as client:
                    first = await client.post(
                        "/v1/artifacts", json=artifact.model_dump(mode="json")
                    )
                    second = await client.post(
                        "/v1/artifacts", json=artifact.model_dump(mode="json")
                    )
                    fetched = await client.get(
                        f"/v1/artifacts/{first.json()['artifact_id']}"
                    )
                    return first, second, fetched

            first, second, fetched = asyncio.run(exercise())

            self.assertEqual(first.status_code, 201)
            self.assertEqual(first.json(), second.json())
            self.assertEqual(fetched.status_code, 200)
            self.assertEqual(fetched.json()["facility_id"], "facility-A")

    def test_state_transition_requires_a_portal_accepted_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PublicLogStore(Path(tmp))
            app = create_log_app(store=store)

            async def exercise() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
                async with app_client(app) as client:
                    await client.post(
                        "/v1/registrations",
                        json={"facility_id": "facility-A", "state_commitment": "S0"},
                    )
                    claim = await client.post(
                        "/v1/events",
                        json={
                            "event_type": "claim_submitted",
                            "actor": "facility",
                            "fields": {
                                "facility_id": "facility-A",
                                "request_id": 77,
                                "state_before": "S0",
                                "proposed_state_after": "S1",
                                "transition_id": "transition-77",
                            },
                        },
                    )
                    premature = await client.post(
                        "/v1/transitions",
                        json={
                            "facility_id": "facility-A",
                            "request_id": 77,
                            "transition_id": "transition-77",
                            "state_before": "S0",
                            "state_after": "S1",
                            "claim_record_index": claim.json()["record_index"],
                            "verification_record_index": claim.json()["record_index"],
                        },
                    )
                    verification = await client.post(
                        "/v1/events",
                        json={
                            "event_type": "proof_accepted",
                            "actor": "portal",
                            "fields": {
                                "facility_id": "facility-A",
                                "request_id": 77,
                                "parent_claim_record": claim.json()["record_index"],
                                "transition_id": "transition-77",
                                "result": "accepted",
                            },
                        },
                    )
                    committed = await client.post(
                        "/v1/transitions",
                        json={
                            "facility_id": "facility-A",
                            "request_id": 77,
                            "transition_id": "transition-77",
                            "state_before": "S0",
                            "state_after": "S1",
                            "claim_record_index": claim.json()["record_index"],
                            "verification_record_index": verification.json()["record_index"],
                        },
                    )
                    latest = await client.get("/v1/facilities/facility-A/latest")
                    return premature, committed, latest

            premature, committed, latest = asyncio.run(exercise())

            self.assertEqual(premature.status_code, 409)
            self.assertEqual(committed.status_code, 201, committed.text)
            self.assertEqual(latest.json()["state_commitment"], "S1")


class FacilityServiceTests(unittest.TestCase):
    def test_facility_accepts_claim_commands_only_from_the_portal(self) -> None:
        portal, facility_a, _, authenticator = authentication_fixture()
        engine = facility_engine()
        app = create_facility_app(
            engine=engine,
            authenticator=authenticator,
            auto_register=False,
            require_portal_auth=True,
        )
        payload = full_request().model_dump(mode="json")

        async def exercise() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
            async with app_client(app) as client:
                unsigned = await client.post("/v1/claims", json=payload)
                wrong_role = await client.post(
                    "/v1/claims",
                    json=payload,
                    headers=facility_a.headers("POST", "/v1/claims", payload),
                )
                accepted = await client.post(
                    "/v1/claims",
                    json=payload,
                    headers=portal.headers("POST", "/v1/claims", payload),
                )
                return unsigned, wrong_role, accepted

        unsigned, wrong_role, accepted = asyncio.run(exercise())

        self.assertEqual(unsigned.status_code, 401)
        self.assertEqual(wrong_role.status_code, 403)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json()["claim_type"], "full")

    def test_full_claim_is_public_and_requires_explicit_commit(self) -> None:
        engine = facility_engine()
        app = create_facility_app(engine=engine, auto_register=False)
        request = full_request()

        async def exercise() -> tuple[httpx.Response, dict[str, object], str]:
            async with app_client(app) as client:
                claim_response = await client.post(
                    "/v1/claims", json=request.model_dump(mode="json")
                )
                claim = claim_response.json()
                before_commit = engine.current_state_commitment()
                commit = CommitRequest(
                    transition_id=claim["transition_id"],
                    expected_state_before=claim["state_before"],
                    expected_state_after=claim["proposed_state_after"],
                    claim_record_index=0,
                    verification_record_index=1,
                )
                commit_response = await client.post(
                    f"/v1/transitions/{claim['transition_id']}/commit",
                    json=commit.model_dump(mode="json"),
                )
                return commit_response, claim, before_commit

        commit_response, claim, before_commit = asyncio.run(exercise())

        self.assertEqual(claim["claim_type"], "full")
        self.assertEqual(claim["state_before"], "7101")
        self.assertEqual(before_commit, "7101")
        serialized = json.dumps(claim)
        for private_name in ("node_mapping", "paths", "rho_S", "sCpuBefore"):
            self.assertNotIn(private_name, serialized)

        self.assertEqual(commit_response.status_code, 200, commit_response.text)
        self.assertEqual(engine.current_state_commitment(), "7003")

    def test_stale_expected_state_is_rejected(self) -> None:
        engine = facility_engine()
        request = full_request().model_copy(update={"expected_state": "old-state"})

        with self.assertRaises(FacilityStateMismatch):
            engine.create_claim(request)

    def test_committed_private_state_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "facility_state.json"
            first = facility_engine(state_path=state_path)
            claim = first.create_claim(full_request())
            assert claim.transition_id is not None
            assert claim.proposed_state_after is not None
            commit_request = CommitRequest(
                transition_id=claim.transition_id,
                expected_state_before=claim.state_before,
                expected_state_after=claim.proposed_state_after,
                claim_record_index=2,
                verification_record_index=3,
            )
            committed = first.commit(commit_request)

            restarted = facility_engine(state_path=state_path)
            retried = restarted.commit(commit_request)

            self.assertEqual(restarted.current_state_commitment(), "7003")
            self.assertEqual(retried, committed)
            self.assertEqual(len(restarted.unlogged_state_transitions()), 1)


class PortalVerifierTests(unittest.TestCase):
    def test_tampered_public_input_is_rejected_before_groth16(self) -> None:
        fixture = json.loads(
            (ROOT / "configs/portal_requests.json").read_text(encoding="utf-8")
        )["requests"][0]
        request = PortalVnrRequest.model_validate(fixture)
        commitments = PublicCommitments(
            request="11",
            substrate_before="12",
            embedding="13",
            substrate_after="14",
        )
        claim = ClaimResponse(
            request_id=request.request_id,
            request_scope="original",
            facility_id="facility-A",
            claim_type="full",
            state_before="12",
            proposed_state_after="14",
            commitments=commitments,
            covered_virtual_nodes=["v0", "v1"],
            proof_mode="groth16",
            transition_id="transition",
        )
        claim_request = ClaimRequest(
            request_id=request.request_id,
            request_scope="original",
            expected_state="12",
            vnr=request.vnr,
        )
        local_vnr, endpoint_enabled, endpoint_index = (
            PortalProofVerifier._local_statement(
                request=claim_request,
                claim=claim,
                registered_endpoint=None,
            )
        )
        public_inputs = PortalProofVerifier._expected_public_inputs(
            request_id=request.request_id,
            local_vnr=local_vnr,
            endpoint_enabled=endpoint_enabled,
            endpoint_index=endpoint_index,
            claim=claim,
        )
        self.assertEqual(len(public_inputs), 37)
        public_inputs[1] = "0"
        artifact = ArtifactCreate(
            facility_id="facility-A",
            request_id=request.request_id,
            claim_type="full",
            commitments=commitments,
            proof_bundle=ProofBundle(proof={}, public_inputs=public_inputs),
        )
        with tempfile.TemporaryDirectory() as tmp:
            verifier = PortalProofVerifier(
                verification_key_path=Path(tmp) / "missing-vkey.json",
                work_dir=Path(tmp) / "work",
            )
            with self.assertRaisesRegex(ProofRejected, "public inputs"):
                verifier.verify_claim(
                    request=claim_request,
                    claim=claim,
                    artifact=artifact,
                    registered_endpoint=None,
                )

    def test_completed_request_replay_returns_the_recorded_result(self) -> None:
        request, records = recovery_fixture("terminal")
        log = MemoryPortalLog(records)
        verifier = RecoveryVerifier()
        engine = PortalEngine(
            log_client=log,  # type: ignore[arg-type]
            facilities={},
            verifier=verifier,  # type: ignore[arg-type]
        )

        result = engine.process(request)

        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.final_state_commitments, {"facility-A": "S1"})
        self.assertEqual(result.timings_seconds, {"online_total": 2.0})
        self.assertEqual(len(log.records), len(records))
        self.assertEqual(verifier.calls, 0)

    def test_replayed_request_id_with_different_content_is_rejected(self) -> None:
        request, records = recovery_fixture("terminal")
        changed_nodes = list(request.vnr.nodes)
        changed_nodes[0] = changed_nodes[0].model_copy(
            update={"cpu": changed_nodes[0].cpu + 1}
        )
        changed_request = request.model_copy(
            update={
                "vnr": GraphPayload(nodes=changed_nodes, edges=request.vnr.edges),
            }
        )
        engine = PortalEngine(
            log_client=MemoryPortalLog(records),  # type: ignore[arg-type]
            facilities={},
            verifier=RecoveryVerifier(),  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(DuplicateRequest, "different content"):
            engine.process(changed_request)

    def test_new_request_waits_for_an_unfinished_request(self) -> None:
        unfinished_request, records = recovery_fixture("claim")
        new_request = unfinished_request.model_copy(update={"request_id": 991})
        engine = PortalEngine(
            log_client=MemoryPortalLog(records),  # type: ignore[arg-type]
            facilities={"facility-A": object()},  # type: ignore[dict-item]
            verifier=RecoveryVerifier(),  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(RequestInProgress, "990 must be resumed"):
            engine.process(new_request)

    def test_portal_resumes_from_each_durable_acceptance_boundary(self) -> None:
        expected_calls = {
            "claim": (1, 1),
            "proof": (0, 1),
            "state": (0, 0),
        }
        for stage, (expected_verify, expected_commit) in expected_calls.items():
            with self.subTest(stage=stage):
                request, records = recovery_fixture(stage)
                log = MemoryPortalLog(records)
                facility = RecoveryFacility(log)
                verifier = RecoveryVerifier()
                engine = PortalEngine(
                    log_client=log,  # type: ignore[arg-type]
                    facilities={"facility-A": facility},  # type: ignore[dict-item]
                    verifier=verifier,  # type: ignore[arg-type]
                )

                result = engine.process(request)

                self.assertEqual(result.status, "accepted")
                self.assertEqual(result.final_state_commitments, {"facility-A": "S1"})
                self.assertEqual(facility.create_calls, 0)
                self.assertEqual(facility.commit_calls, expected_commit)
                self.assertEqual(verifier.calls, expected_verify)
                event_types = [record.get("event_type") for record in log.records]
                self.assertEqual(event_types.count("claim_submitted"), 1)
                self.assertEqual(event_types.count("proof_accepted"), 1)
                self.assertEqual(event_types.count("state_committed"), 1)
                self.assertEqual(event_types.count("request_completed"), 1)

    def test_endpoint_enabled_claim_must_use_registered_endpoint_index(self) -> None:
        fixture = json.loads(
            (ROOT / "configs/portal_requests.json").read_text(encoding="utf-8")
        )["requests"][1]
        vnr = GraphPayload.model_validate(fixture["vnr"])
        demand = BoundaryDemandPayload(
            source_facility_id="facility-A",
            source_endpoint_id="A-public-0",
            embedded_virtual_node="v1",
            residual_virtual_node="v2",
            bandwidth=10,
        )
        claim = ClaimResponse(
            request_id=101,
            request_scope="original",
            facility_id="facility-A",
            claim_type="partial",
            state_before="12",
            proposed_state_after="14",
            commitments=PublicCommitments(
                request="11",
                substrate_before="12",
                embedding="13",
                substrate_after="14",
            ),
            covered_virtual_nodes=["v0", "v1"],
            public_endpoint_id="A-public-0",
            public_endpoint_index=2,
            residual=ResidualPayload(
                vnr=GraphPayload(nodes=[vnr.nodes[2]], edges=[]),
                embedded_virtual_nodes=["v0", "v1"],
                boundary_demands=[demand],
                residual_commitment="unused-in-this-binding-test",
            ),
            proof_mode="groth16",
        )
        request = ClaimRequest(
            request_id=101,
            request_scope="original",
            expected_state="12",
            vnr=vnr,
        )
        endpoint = PublicEndpointPayload(
            endpoint_id="A-public-0",
            facility_id="facility-A",
            substrate_index=3,
        )

        with self.assertRaisesRegex(ProofRejected, "endpoint index"):
            PortalProofVerifier._local_statement(
                request=request,
                claim=claim,
                registered_endpoint=endpoint,
            )


class ResidualServiceTests(unittest.TestCase):
    def test_public_link_is_undirected_and_capacity_checked(self) -> None:
        link = PublicLink("A-public-0", "B-public-0", bandwidth=10)
        self.assertTrue(link.supports("A-public-0", "B-public-0", 10))
        self.assertTrue(link.supports("B-public-0", "A-public-0", 5))
        self.assertFalse(link.supports("A-public-0", "B-public-0", 11))
        self.assertFalse(link.supports("A-public-0", "C-public-0", 1))

    def test_valid_connected_single_cut_residual_is_accepted(self) -> None:
        fixture = json.loads(
            (ROOT / "configs/portal_requests.json").read_text(encoding="utf-8")
        )["requests"][1]
        original = GraphPayload.model_validate(fixture["vnr"])
        residual_vnr = GraphPayload(nodes=[original.nodes[2]], edges=[])
        demands = [
            BoundaryDemandPayload(
                source_facility_id="facility-A",
                source_endpoint_id="A-public-0",
                embedded_virtual_node="v1",
                residual_virtual_node="v2",
                bandwidth=10,
            )
        ]
        residual = ResidualPayload(
            vnr=residual_vnr,
            embedded_virtual_nodes=["v0", "v1"],
            boundary_demands=demands,
            residual_commitment=residual_commitment(
                request_id=fixture["request_id"],
                vnr=residual_vnr,
                boundary_demands=demands,
            ),
        )

        validate_single_cut_residual(
            request_id=fixture["request_id"],
            original=original,
            residual=residual,
            source_facility_id="facility-A",
            source_endpoint_id="A-public-0",
        )

        local = build_local_circuit_vnr(
            graph_from_payload(original, graph_id="101"),
            ["v0", "v1"],
            endpoint_demands=[("v1", 10)],
        )
        self.assertTrue(local.nodes["v2"]["public_endpoint"])
        self.assertEqual(local.edges["v1", "v2"]["bandwidth"], 10)

    def test_two_crossing_edges_are_rejected(self) -> None:
        original = GraphPayload.model_validate(
            {
                "nodes": [
                    {"id": "v0", "cpu": 1},
                    {"id": "v1", "cpu": 1},
                    {"id": "v2", "cpu": 1},
                ],
                "edges": [
                    {"u": "v0", "v": "v1", "bandwidth": 2},
                    {"u": "v1", "v": "v2", "bandwidth": 3},
                ],
            }
        )
        residual_vnr = GraphPayload(nodes=[original.nodes[0], original.nodes[2]], edges=[])
        demands = [
            BoundaryDemandPayload(
                source_facility_id="facility-A",
                source_endpoint_id="A-public-0",
                embedded_virtual_node="v1",
                residual_virtual_node="v0",
                bandwidth=2,
            ),
            BoundaryDemandPayload(
                source_facility_id="facility-A",
                source_endpoint_id="A-public-0",
                embedded_virtual_node="v1",
                residual_virtual_node="v2",
                bandwidth=3,
            ),
        ]
        residual = ResidualPayload(
            vnr=residual_vnr,
            embedded_virtual_nodes=["v1"],
            boundary_demands=demands,
            residual_commitment=residual_commitment(
                request_id=9,
                vnr=residual_vnr,
                boundary_demands=demands,
            ),
        )

        with self.assertRaisesRegex(InvalidResidual, "exactly one crossing"):
            validate_single_cut_residual(
                request_id=9,
                original=original,
                residual=residual,
                source_facility_id="facility-A",
                source_endpoint_id="A-public-0",
            )


if __name__ == "__main__":
    unittest.main()
