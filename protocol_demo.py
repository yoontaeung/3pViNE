"""Unified multi-facility proof-carrying VNE protocol runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import networkx as nx

from local_solver import LocalEmbedding, SolverPolicy
from partial_protocol import (
    BoundaryDemand,
    ConnectedPartialClaim,
    DeclineClaim,
    FullEmbeddingClaim,
    PublicConnectivity,
    PublicConnectivityOracle,
    PublicEndpoint,
    ResidualRequest,
    solve_connected_vne_claim,
    verify_residual_consistency,
)
from protocol import PublicArtifacts, TransparencyLog
from verifier import verify_resource_feasibility
from zk_encoding import (
    S_NODE_IDS,
    apply_after_state_to_substrate,
    build_zk_input,
    substrate_state_commitment,
    write_zk_input,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PORTAL_REQUESTS_PATH = ROOT / "configs" / "portal_requests.json"
DEFAULT_FACILITY_PRIVATE_PATH = ROOT / "configs" / "facility_private.json"
ProofMode = Literal["groth16", "reference"]


@dataclass(frozen=True)
class PortalRequest:
    """Public request and deterministic facility order chosen by the portal."""

    request_id: int
    vnr: nx.Graph
    facility_order: tuple[str, ...]


@dataclass(frozen=True)
class FacilityConfig:
    """Private facility input for local solving and proof generation."""

    facility_id: str
    endpoint: PublicEndpoint | None
    initial_rho_S: int
    substrate: nx.Graph
    solver_policy: SolverPolicy
    randomness_by_request: dict[int, list[dict[str, int]]]


@dataclass
class FacilityRuntime:
    """Mutable facility protocol state."""

    config: FacilityConfig
    substrate: nx.Graph
    current_rho_S: int
    proof_counts: dict[int, int]

    @classmethod
    def from_config(cls, config: FacilityConfig) -> "FacilityRuntime":
        return cls(
            config=config,
            substrate=config.substrate.copy(),
            current_rho_S=config.initial_rho_S,
            proof_counts={},
        )

    def transition_randomness(self, request_id: int) -> dict[str, int]:
        values = self.config.randomness_by_request.get(request_id)
        if not values:
            raise ValueError(
                f"missing private commitment randomness for {self.config.facility_id} "
                f"request {request_id}"
            )
        index = self.proof_counts.get(request_id, 0)
        if index >= len(values):
            raise ValueError(
                f"not enough private commitment randomness for {self.config.facility_id} "
                f"request {request_id}"
            )
        self.proof_counts[request_id] = index + 1
        selected = values[index]
        return {
            "rho_S": self.current_rho_S,
            "rho_M": int(selected["rho_M"]),
            "rho_S_next": int(selected["rho_S_next"]),
        }


def graph_from_config(config: dict[str, Any], *, graph_id: str) -> nx.Graph:
    graph = nx.Graph(req_id=graph_id)
    for node in config["nodes"]:
        attrs = {key: value for key, value in node.items() if key != "id"}
        if "cpu" in attrs:
            attrs["cpu"] = int(attrs["cpu"])
        if "memory_gib" in attrs:
            attrs["memory_gib"] = int(attrs["memory_gib"])
        graph.add_node(node["id"], **attrs)
    for edge in config["edges"]:
        attrs = {key: value for key, value in edge.items() if key not in {"u", "v"}}
        if "bandwidth" in attrs:
            attrs["bandwidth"] = int(attrs["bandwidth"])
        graph.add_edge(edge["u"], edge["v"], **attrs)
    return graph


def graph_public_data(graph: nx.Graph) -> dict[str, Any]:
    return {
        "nodes": [{"id": node, **dict(attrs)} for node, attrs in sorted(graph.nodes(data=True))],
        "edges": [
            {"u": min(u, v), "v": max(u, v), **dict(attrs)}
            for u, v, attrs in sorted(
                graph.edges(data=True),
                key=lambda edge: (min(edge[0], edge[1]), max(edge[0], edge[1])),
            )
        ],
    }


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def boundary_demands_public_data(
    boundary_demands: tuple[BoundaryDemand, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "source_facility_id": demand.source_facility_id,
            "source_endpoint_id": demand.source_endpoint_id,
            "embedded_virtual_node": demand.embedded_virtual_node,
            "residual_virtual_node": demand.residual_virtual_node,
            "bandwidth": demand.bandwidth,
        }
        for demand in boundary_demands
    ]


def compute_residual_commitment(
    *,
    request_id: int,
    residual_vnr: nx.Graph,
    boundary_demands: tuple[BoundaryDemand, ...],
) -> str:
    return stable_digest(
        {
            "request_id": request_id,
            "residual_vnr": graph_public_data(residual_vnr),
            "boundary_demands": boundary_demands_public_data(boundary_demands),
        }
    )


def verify_portal_residual_forwarding(
    log: TransparencyLog,
    forwarding_record: dict[str, Any],
    *,
    source_vnr: nx.Graph,
    residual_vnr: nx.Graph,
    boundary_demands: tuple[BoundaryDemand, ...],
    target_facility_id: str,
) -> tuple[bool, list[str]]:
    """Verify that a portal residual-forwarding event matches a partial claim."""
    errors: list[str] = []
    if forwarding_record.get("event_type") != "residual_forwarded":
        errors.append("record is not a residual_forwarded event")
    if forwarding_record.get("actor") != "portal":
        errors.append("residual forwarding actor is not portal")
    if forwarding_record.get("target_facility_id") != target_facility_id:
        errors.append("target facility does not match residual recipient")

    request_id = forwarding_record.get("request_id")
    try:
        request_id_int = int(request_id)
    except (TypeError, ValueError):
        errors.append("residual forwarding request_id is missing or invalid")
        return False, errors

    expected_commitment = compute_residual_commitment(
        request_id=request_id_int,
        residual_vnr=residual_vnr,
        boundary_demands=boundary_demands,
    )
    if forwarding_record.get("residual_commitment") != expected_commitment:
        errors.append("residual commitment does not match residual message")

    expected_boundary = boundary_demands_public_data(boundary_demands)
    if forwarding_record.get("boundary_demands") != expected_boundary:
        errors.append("boundary demands do not match residual message")
    expected_residual_vnr = graph_public_data(residual_vnr)
    if forwarding_record.get("residual_vnr") != expected_residual_vnr:
        errors.append("residual VNR does not match residual message")

    parent_index = forwarding_record.get("parent_claim_record")
    if not isinstance(parent_index, int) or not 0 <= parent_index < len(log.records):
        errors.append("parent partial claim record is missing")
        return False, errors

    parent = log.records[parent_index]
    if parent.get("event_type") != "claim_submitted":
        errors.append("parent record is not a claim_submitted event")
    if parent.get("claim_type") != "partial":
        errors.append("parent claim is not partial")
    if parent.get("request_id") != request_id_int:
        errors.append("parent claim request_id does not match forwarding event")
    if parent.get("facility_id") != forwarding_record.get("source_facility_id"):
        errors.append("forwarding source facility does not match parent claim")
    if parent.get("residual_commitment") != expected_commitment:
        errors.append("parent claim residual commitment does not match")
    forwarding_index = forwarding_record.get("record_index")
    if not isinstance(forwarding_index, int):
        forwarding_index = len(log.records)
    accepted_validations = [
        record
        for record in log.records
        if record.get("event_type") in {"proof_accepted", "reference_validation_accepted"}
        and record.get("parent_claim_record") == parent_index
        and record.get("result") == "accepted"
        and record.get("record_index", forwarding_index) < forwarding_index
    ]
    if not accepted_validations:
        errors.append("parent partial claim does not have an accepted validation")
    accepted_validation_indices = {record["record_index"] for record in accepted_validations}
    committed_states = [
        record
        for record in log.records
        if (
            record.get("event_type") == "state_committed"
            and record.get("parent_claim_record") == parent_index
            and record.get("facility_id") == forwarding_record.get("source_facility_id")
            and (
                record.get("proof_accepted_record") in accepted_validation_indices
                or record.get("validation_accepted_record") in accepted_validation_indices
            )
            and record.get("record_index", forwarding_index) < forwarding_index
        )
    ]
    if not committed_states:
        errors.append("parent partial claim state has not been committed")

    embedded_nodes = tuple(parent.get("covered_virtual_nodes", ()))
    residual = ResidualRequest(
        request_id=request_id_int,
        residual_vnr=residual_vnr,
        embedded_virtual_nodes=embedded_nodes,
        boundary_demands=boundary_demands,
    )
    consistent, residual_errors = verify_residual_consistency(source_vnr, residual)
    if not consistent:
        errors.extend(f"residual consistency failed: {error}" for error in residual_errors)

    return not errors, errors


def load_portal_requests(config_path: Path = DEFAULT_PORTAL_REQUESTS_PATH) -> list[PortalRequest]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    requests: list[PortalRequest] = []
    for request in config["requests"]:
        request_id = int(request["request_id"])
        facility_order = tuple(request["facility_order"])
        if not facility_order:
            raise ValueError(f"request {request_id} must define at least one facility")
        requests.append(
            PortalRequest(
                request_id=request_id,
                vnr=graph_from_config(request["vnr"], graph_id=str(request_id)),
                facility_order=facility_order,
            )
        )
    return requests


def _randomness_entries(raw: Any) -> list[dict[str, int]]:
    if isinstance(raw, list):
        return [{"rho_M": int(item["rho_M"]), "rho_S_next": int(item["rho_S_next"])} for item in raw]
    if isinstance(raw, dict):
        return [{"rho_M": int(raw["rho_M"]), "rho_S_next": int(raw["rho_S_next"])}]
    raise ValueError("randomness entry must be an object or list of objects")


def load_facilities(
    config_path: Path = DEFAULT_FACILITY_PRIVATE_PATH,
) -> tuple[dict[str, FacilityRuntime], PublicConnectivityOracle]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    facilities: dict[str, FacilityRuntime] = {}

    for facility_config in config["facilities"]:
        facility_id = facility_config["facility_id"]
        endpoint_config = facility_config.get("public_endpoint")
        endpoint = (
            PublicEndpoint(
                endpoint_id=endpoint_config["endpoint_id"],
                facility_id=facility_id,
                substrate_node=endpoint_config["substrate_node"],
            )
            if endpoint_config is not None
            else None
        )
        substrate = graph_from_config(facility_config["substrate"], graph_id=facility_id)
        substrate.graph["facility_id"] = facility_id

        solver_config = facility_config.get("solver", {})
        solver_policy = SolverPolicy(
            objective=solver_config.get("objective", "min_hops"),
            max_seconds=float(solver_config.get("max_seconds", 5.0)),
        )
        randomness_by_request = {
            int(request_id): _randomness_entries(values)
            for request_id, values in facility_config.get("randomness_by_request", {}).items()
        }
        runtime = FacilityRuntime.from_config(
            FacilityConfig(
                facility_id=facility_id,
                endpoint=endpoint,
                initial_rho_S=int(facility_config["initial_rho_S"]),
                substrate=substrate,
                solver_policy=solver_policy,
                randomness_by_request=randomness_by_request,
            )
        )
        facilities[facility_id] = runtime

    connectivity = PublicConnectivityOracle(
        [
            PublicConnectivity(
                link["endpoint_a"],
                link["endpoint_b"],
                bandwidth=int(link["bandwidth"]),
                latency_ms=int(link["latency_ms"]) if "latency_ms" in link else None,
            )
            for link in config.get("public_connectivity", [])
        ]
    )
    return facilities, connectivity


def default_run_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "artifacts" / "protocol" / f"protocol_{timestamp}"


def relative_to_run(path: Path, run_dir: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=ROOT)


def generate_witness(
    *,
    input_path: Path,
    witness_path: Path,
) -> None:
    run_command(
        [
            "node",
            "build/zk/resource_feasibility_js/generate_witness.js",
            "build/zk/resource_feasibility_js/resource_feasibility.wasm",
            str(input_path),
            str(witness_path),
        ]
    )


def generate_groth16_proof(
    *,
    witness_path: Path,
    proof_path: Path,
    public_input_path: Path,
) -> None:
    run_command(
        [
            "snarkjs",
            "groth16",
            "prove",
            "build/zk/resource_feasibility_final.zkey",
            str(witness_path),
            str(proof_path),
            str(public_input_path),
        ]
    )


def verify_proof(
    *,
    public_input_path: Path,
    proof_path: Path,
) -> None:
    run_command(["bash", "scripts/verify.sh", str(public_input_path), str(proof_path)])


def _prove_claim(
    *,
    run_dir: Path,
    private_dir: Path,
    public_dir: Path,
    request_id: int,
    facility: FacilityRuntime,
    local_vnr: nx.Graph,
    embedding: LocalEmbedding,
    randomness: dict[str, int],
    endpoint_substrate_node: str | None,
    proof_mode: ProofMode,
    timing: dict[str, Any],
) -> dict[str, Any]:
    endpoint_idx = S_NODE_IDS[endpoint_substrate_node] if endpoint_substrate_node else None
    started = time.perf_counter()
    zk_input = build_zk_input(
        local_vnr,
        facility.substrate,
        node_mapping=embedding.node_mapping,
        paths=embedding.paths,
        req_id=request_id,
        randomness=randomness,
        endpoint_idx=endpoint_idx,
    )
    timing["encoding_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    accepted, errors = verify_resource_feasibility(zk_input)
    timing["reference_validation_seconds"] = time.perf_counter() - started
    if not accepted:
        raise RuntimeError(
            f"request {request_id} {facility.config.facility_id} witness is invalid: {errors}"
        )

    proof_index = facility.proof_counts[request_id]
    request_name = f"request_{request_id:03d}_{facility.config.facility_id}_{proof_index:02d}"
    request_private_dir = private_dir / request_name
    request_public_dir = public_dir / request_name
    request_private_dir.mkdir(parents=True, exist_ok=True)
    request_public_dir.mkdir(parents=True, exist_ok=True)

    input_path = request_private_dir / "input.json"
    witness_path = request_private_dir / "witness.wtns"
    proof_path = request_public_dir / "proof.json"
    public_input_path = request_public_dir / "public.json"
    started = time.perf_counter()
    write_zk_input(zk_input, input_path)
    timing["input_write_seconds"] = time.perf_counter() - started
    timing["input_bytes"] = input_path.stat().st_size

    if proof_mode == "reference":
        timing["witness_seconds"] = 0.0
        timing["prove_seconds"] = 0.0
        timing["witness_bytes"] = 0
        timing["proof_bytes"] = 0
        timing["public_input_bytes"] = 0
        return zk_input

    started = time.perf_counter()
    generate_witness(
        input_path=input_path,
        witness_path=witness_path,
    )
    timing["witness_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    generate_groth16_proof(
        witness_path=witness_path,
        proof_path=proof_path,
        public_input_path=public_input_path,
    )
    timing["prove_seconds"] = time.perf_counter() - started
    timing["witness_bytes"] = witness_path.stat().st_size
    timing["proof_bytes"] = proof_path.stat().st_size
    timing["public_input_bytes"] = public_input_path.stat().st_size

    zk_input["_public_artifacts"] = PublicArtifacts(
        proof_path=relative_to_run(proof_path, run_dir),
        public_input_path=relative_to_run(public_input_path, run_dir),
    )
    zk_input["_proof_path_abs"] = proof_path
    zk_input["_public_input_path_abs"] = public_input_path
    return zk_input


def _submit_and_validate_claim(
    *,
    log: TransparencyLog,
    request_id: int,
    facility_id: str,
    claim_type: str,
    request_scope: str,
    claim_record: dict[str, Any],
    zk_input: dict[str, Any],
    proof_mode: ProofMode,
    timing: dict[str, Any],
) -> dict[str, Any]:
    if proof_mode == "reference":
        timing["verify_seconds"] = 0.0
        return log.append_event(
            event_type="reference_validation_accepted",
            actor="portal",
            verifier_id="python_reference_checker",
            facility_id=facility_id,
            request_id=request_id,
            claim_type=claim_type,
            request_scope=request_scope,
            parent_claim_record=claim_record["record_index"],
            result="accepted",
            proof_system="none",
        )

    artifacts = zk_input["_public_artifacts"]
    proof_record = log.append_event(
        event_type="proof_submitted",
        actor="facility",
        facility_id=facility_id,
        request_id=request_id,
        claim_type=claim_type,
        request_scope=request_scope,
        parent_claim_record=claim_record["record_index"],
        proof_path=artifacts.proof_path,
        public_input_path=artifacts.public_input_path,
        C_r=str(zk_input["C_r"]),
        C_S=str(zk_input["C_S"]),
        C_M=str(zk_input["C_M"]),
        C_S_next=str(zk_input["C_S_next"]),
    )
    started = time.perf_counter()
    try:
        verify_proof(
            public_input_path=zk_input["_public_input_path_abs"],
            proof_path=zk_input["_proof_path_abs"],
        )
        timing["verify_seconds"] = time.perf_counter() - started
    except subprocess.CalledProcessError:
        timing["verify_seconds"] = time.perf_counter() - started
        log.append_event(
            event_type="proof_rejected",
            actor="portal",
            verifier_id="portal",
            facility_id=facility_id,
            request_id=request_id,
            claim_type=claim_type,
            request_scope=request_scope,
            parent_claim_record=claim_record["record_index"],
            proof_record=proof_record["record_index"],
            result="rejected",
        )
        raise

    return log.append_event(
        event_type="proof_accepted",
        actor="portal",
        verifier_id="portal",
        facility_id=facility_id,
        request_id=request_id,
        claim_type=claim_type,
        request_scope=request_scope,
        parent_claim_record=claim_record["record_index"],
        proof_record=proof_record["record_index"],
        result="accepted",
    )


def _append_registration_events(
    log: TransparencyLog,
    facilities: dict[str, FacilityRuntime],
) -> None:
    for facility_id in sorted(facilities):
        facility = facilities[facility_id]
        commitment = substrate_state_commitment(
            facility.substrate,
            rho_S=facility.current_rho_S,
        )
        endpoint = facility.config.endpoint
        log.append_event(
            event_type="facility_registered",
            actor="facility",
            facility_id=facility_id,
            state_before=None,
            state_after=str(commitment),
            public_endpoint=(
                {
                    "endpoint_id": endpoint.endpoint_id,
                    "facility_id": facility_id,
                    "substrate_node": endpoint.substrate_node,
                }
                if endpoint is not None
                else None
            ),
        )


def _public_connectivity_ok(
    incoming_boundary_demands: tuple[BoundaryDemand, ...],
    *,
    endpoint_id: str,
    connectivity: PublicConnectivityOracle,
) -> tuple[bool, list[dict[str, Any]]]:
    checks = []
    ok = True
    for demand in incoming_boundary_demands:
        satisfied = connectivity.can_satisfy(
            demand.source_endpoint_id,
            endpoint_id,
            demand.bandwidth,
        )
        checks.append(
            {
                "endpoint_a": demand.source_endpoint_id,
                "endpoint_b": endpoint_id,
                "bandwidth": demand.bandwidth,
                "result": "satisfied" if satisfied else "failed",
            }
        )
        ok = ok and satisfied
    return ok, checks


def _commit_verified_state(
    *,
    log: TransparencyLog,
    facility: FacilityRuntime,
    request_id: int,
    claim_record: dict[str, Any],
    validation_accepted_record: dict[str, Any],
    zk_input: dict[str, Any],
    rho_S_next: int,
    proof_mode: ProofMode,
) -> None:
    if proof_mode == "groth16":
        log.append_state_committed(
            facility_id=facility.config.facility_id,
            request_id=request_id,
            state_before=str(zk_input["C_S"]),
            state_after=str(zk_input["C_S_next"]),
            parent_claim_record=claim_record["record_index"],
            proof_accepted_record=validation_accepted_record["record_index"],
        )
    else:
        log.append_reference_state_committed(
            facility_id=facility.config.facility_id,
            request_id=request_id,
            state_before=str(zk_input["C_S"]),
            state_after=str(zk_input["C_S_next"]),
            parent_claim_record=claim_record["record_index"],
            validation_accepted_record=validation_accepted_record["record_index"],
        )
    facility.substrate = apply_after_state_to_substrate(facility.substrate, zk_input)
    facility.current_rho_S = rho_S_next


def _record_decline(
    *,
    log: TransparencyLog,
    request: PortalRequest,
    facility_id: str | None,
    reason: str,
    request_scope: str,
) -> None:
    log.append_event(
        event_type="request_declined",
        actor="portal",
        request_id=request.request_id,
        facility_id=facility_id,
        reason=reason,
        request_scope=request_scope,
        state_before=None,
        state_after=None,
    )


def run_protocol_demo(
    run_dir: Path,
    *,
    portal_requests_path: Path = DEFAULT_PORTAL_REQUESTS_PATH,
    facility_private_path: Path = DEFAULT_FACILITY_PRIVATE_PATH,
    proof_mode: ProofMode = "groth16",
    metrics_output: Path | None = None,
) -> Path:
    run_dir = run_dir.resolve()
    private_dir = run_dir / "private"
    public_dir = run_dir / "public"
    private_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    portal_requests = load_portal_requests(portal_requests_path)
    facilities, connectivity = load_facilities(facility_private_path)
    log = TransparencyLog(run_dir / "transparency_log.jsonl")
    _append_registration_events(log, facilities)
    benchmark_metrics: dict[str, Any] = {
        "schema_version": 1,
        "system": "zk-vne",
        "proof_mode": proof_mode,
        "transitions": [],
        "requests": [],
    }

    for request in portal_requests:
        request_started = time.perf_counter()
        request_result = "declined"
        original_request_commitment = stable_digest(
            {"request_id": request.request_id, "vnr": graph_public_data(request.vnr)}
        )
        log.append_event(
            event_type="request_issued",
            actor="portal",
            request_id=request.request_id,
            request_commitment=original_request_commitment,
            facility_order=list(request.facility_order),
            target_facility_id=request.facility_order[0],
            vnr=graph_public_data(request.vnr),
        )

        current_vnr = request.vnr
        incoming_boundary_demands: tuple[BoundaryDemand, ...] = ()
        committed_facilities: list[str] = []
        completed = False

        for facility_index, facility_id in enumerate(request.facility_order):
            transition_started = time.perf_counter()
            request_scope = "residual" if incoming_boundary_demands else "original"
            transition_timing: dict[str, Any] = {
                "request_id": request.request_id,
                "facility_id": facility_id,
                "request_scope": request_scope,
                "proof_mode": proof_mode,
            }
            next_facility_id = (
                request.facility_order[facility_index + 1]
                if facility_index + 1 < len(request.facility_order)
                else None
            )
            if facility_index > 0 and not incoming_boundary_demands:
                log.append_event(
                    event_type="request_forwarded",
                    actor="portal",
                    request_id=request.request_id,
                    request_scope="original",
                    request_commitment=original_request_commitment,
                    target_facility_id=facility_id,
                    vnr=graph_public_data(current_vnr),
                )
            if facility_id not in facilities:
                raise ValueError(f"request {request.request_id} references unknown facility {facility_id}")
            facility = facilities[facility_id]
            latest_state = log.latest_state(facility_id)
            if latest_state is None:
                raise RuntimeError(f"facility {facility_id} state was not initialized")

            started = time.perf_counter()
            claim = solve_connected_vne_claim(
                current_vnr,
                facility.substrate,
                facility_id=facility_id,
                endpoint=facility.config.endpoint,
                policy=facility.config.solver_policy,
                incoming_boundary_demands=incoming_boundary_demands,
            )
            transition_timing["solver_seconds"] = time.perf_counter() - started

            if isinstance(claim, DeclineClaim):
                log.append_event(
                    event_type="claim_submitted",
                    actor="facility",
                    claim_type="decline",
                    request_scope=request_scope,
                    facility_id=facility_id,
                    request_id=request.request_id,
                    reason=claim.reason,
                    state_before=latest_state,
                    state_after=latest_state,
                    request_commitment=original_request_commitment,
                )
                transition_timing["claim_type"] = "decline"
                transition_timing["online_total_seconds"] = time.perf_counter() - transition_started
                benchmark_metrics["transitions"].append(transition_timing)
                continue

            residual_commitment: str | None = None
            if isinstance(claim, ConnectedPartialClaim):
                consistent, residual_errors = verify_residual_consistency(current_vnr, claim.residual)
                if not consistent:
                    raise RuntimeError(f"invalid residual VNR: {residual_errors}")
                residual_commitment = compute_residual_commitment(
                    request_id=request.request_id,
                    residual_vnr=claim.residual.residual_vnr,
                    boundary_demands=claim.residual.boundary_demands,
                )

            randomness = facility.transition_randomness(request.request_id)
            endpoint_substrate_node = (
                claim.endpoint_substrate_node
                if isinstance(claim, (ConnectedPartialClaim, FullEmbeddingClaim))
                else None
            )
            zk_input = _prove_claim(
                run_dir=run_dir,
                private_dir=private_dir,
                public_dir=public_dir,
                request_id=request.request_id,
                facility=facility,
                local_vnr=claim.local_vnr,
                embedding=claim.embedding,
                randomness=randomness,
                endpoint_substrate_node=endpoint_substrate_node,
                proof_mode=proof_mode,
                timing=transition_timing,
            )
            if latest_state != str(zk_input["C_S"]):
                raise RuntimeError(
                    f"request {request.request_id} starts from C_S={zk_input['C_S']}, "
                    f"but latest public state for {facility_id} is {latest_state}"
                )

            if isinstance(claim, ConnectedPartialClaim):
                assert facility.config.endpoint is not None
                assert residual_commitment is not None
                partial_record = log.append_event(
                    event_type="claim_submitted",
                    actor="facility",
                    claim_type="partial",
                    request_scope=request_scope,
                    facility_id=facility_id,
                    request_id=request.request_id,
                    state_before=str(zk_input["C_S"]),
                    proposed_state_after=str(zk_input["C_S_next"]),
                    request_commitment=original_request_commitment,
                    local_request_commitment=str(zk_input["C_r"]),
                    embedding_commitment=str(zk_input["C_M"]),
                    residual_commitment=residual_commitment,
                    covered_virtual_nodes=list(claim.embedded_virtual_nodes),
                    public_endpoint=facility.config.endpoint.endpoint_id,
                )
                validation_accepted_record = _submit_and_validate_claim(
                    log=log,
                    request_id=request.request_id,
                    facility_id=facility_id,
                    claim_type="partial",
                    request_scope=request_scope,
                    claim_record=partial_record,
                    zk_input=zk_input,
                    proof_mode=proof_mode,
                    timing=transition_timing,
                )
                started = time.perf_counter()
                _commit_verified_state(
                    log=log,
                    facility=facility,
                    request_id=request.request_id,
                    claim_record=partial_record,
                    validation_accepted_record=validation_accepted_record,
                    zk_input=zk_input,
                    rho_S_next=randomness["rho_S_next"],
                    proof_mode=proof_mode,
                )
                transition_timing["state_commit_seconds"] = time.perf_counter() - started
                transition_timing["claim_type"] = "partial"
                transition_timing["online_total_seconds"] = time.perf_counter() - transition_started
                benchmark_metrics["transitions"].append(transition_timing)
                committed_facilities.append(facility_id)

                if next_facility_id is None:
                    _record_decline(
                        log=log,
                        request=request,
                        facility_id=None,
                        reason="no_facility_for_residual",
                        request_scope="residual",
                    )
                    completed = True
                    break

                forwarding_record = log.append_event(
                    event_type="residual_forwarded",
                    actor="portal",
                    request_id=request.request_id,
                    source_facility_id=facility_id,
                    target_facility_id=next_facility_id,
                    parent_claim_record=partial_record["record_index"],
                    residual_commitment=residual_commitment,
                    residual_vnr=graph_public_data(claim.residual.residual_vnr),
                    boundary_demands=boundary_demands_public_data(
                        claim.residual.boundary_demands
                    ),
                )
                forwarding_ok, forwarding_errors = verify_portal_residual_forwarding(
                    log,
                    forwarding_record,
                    source_vnr=current_vnr,
                    residual_vnr=claim.residual.residual_vnr,
                    boundary_demands=claim.residual.boundary_demands,
                    target_facility_id=next_facility_id,
                )
                if not forwarding_ok:
                    raise RuntimeError(f"invalid residual forwarding event: {forwarding_errors}")

                current_vnr = claim.residual.residual_vnr
                incoming_boundary_demands = claim.residual.boundary_demands
                continue

            connectivity_ok, public_connectivity_checks = _public_connectivity_ok(
                incoming_boundary_demands,
                endpoint_id=(
                    facility.config.endpoint.endpoint_id
                    if facility.config.endpoint is not None
                    else ""
                ),
                connectivity=connectivity,
            )
            if not connectivity_ok:
                _record_decline(
                    log=log,
                    request=request,
                    facility_id=facility_id,
                    reason="public_connectivity_failed",
                    request_scope=request_scope,
                )
                completed = True
                break

            full_record = log.append_event(
                event_type="claim_submitted",
                actor="facility",
                claim_type="full",
                request_scope=request_scope,
                facility_id=facility_id,
                request_id=request.request_id,
                state_before=str(zk_input["C_S"]),
                proposed_state_after=str(zk_input["C_S_next"]),
                request_commitment=original_request_commitment,
                local_request_commitment=str(zk_input["C_r"]),
                embedding_commitment=str(zk_input["C_M"]),
                covered_virtual_nodes=sorted(claim.slot_to_original.values()),
                public_endpoint=(
                    facility.config.endpoint.endpoint_id
                    if claim.endpoint_substrate_node and facility.config.endpoint is not None
                    else None
                ),
            )
            validation_accepted_record = _submit_and_validate_claim(
                log=log,
                request_id=request.request_id,
                facility_id=facility_id,
                claim_type="full",
                request_scope=request_scope,
                claim_record=full_record,
                zk_input=zk_input,
                proof_mode=proof_mode,
                timing=transition_timing,
            )
            started = time.perf_counter()
            _commit_verified_state(
                log=log,
                facility=facility,
                request_id=request.request_id,
                claim_record=full_record,
                validation_accepted_record=validation_accepted_record,
                zk_input=zk_input,
                rho_S_next=randomness["rho_S_next"],
                proof_mode=proof_mode,
            )
            transition_timing["state_commit_seconds"] = time.perf_counter() - started
            transition_timing["claim_type"] = "full"
            transition_timing["online_total_seconds"] = time.perf_counter() - transition_started
            benchmark_metrics["transitions"].append(transition_timing)
            committed_facilities.append(facility_id)
            log.append_event(
                event_type="request_completed",
                actor="portal",
                request_id=request.request_id,
                request_commitment=original_request_commitment,
                involved_facilities=committed_facilities,
                public_connectivity_checks=public_connectivity_checks,
            )
            request_result = "accepted"
            completed = True
            break

        if not completed:
            _record_decline(
                log=log,
                request=request,
                facility_id=None,
                reason="facility_order_exhausted",
                request_scope="residual" if incoming_boundary_demands else "original",
            )

        benchmark_metrics["requests"].append(
            {
                "request_id": request.request_id,
                "result": request_result,
                "online_total_seconds": time.perf_counter() - request_started,
                "committed_facilities": committed_facilities,
            }
        )

    if metrics_output is not None:
        metrics_output = metrics_output.resolve()
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_output.write_text(
            json.dumps(benchmark_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return log.path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-facility proof-carrying VNE demo.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_run_dir(),
        help="Directory for private artifacts, public artifacts, and the transparency log.",
    )
    parser.add_argument(
        "--portal-requests",
        type=Path,
        default=DEFAULT_PORTAL_REQUESTS_PATH,
        help="Public portal request list JSON.",
    )
    parser.add_argument(
        "--facility-private",
        type=Path,
        default=DEFAULT_FACILITY_PRIVATE_PATH,
        help="Private multi-facility substrate, solver policy, and randomness JSON.",
    )
    parser.add_argument(
        "--proof-mode",
        choices=("groth16", "reference"),
        default="groth16",
        help="Use Groth16 or the explicit no-ZKP reference-checker ablation.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional machine-readable per-stage timing output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = run_protocol_demo(
        args.run_dir,
        portal_requests_path=args.portal_requests,
        facility_private_path=args.facility_private,
        proof_mode=args.proof_mode,
        metrics_output=args.metrics_output,
    )
    print(f"Transparency log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
