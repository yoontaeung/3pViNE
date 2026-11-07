"""Canonical message and storage accounting for one protocol run."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


MESSAGE_TYPES = {
    "facility_registered": "FacilityRegistration",
    "request_issued": "Request",
    "request_forwarded": "Request",
    "claim_submitted": "Claim",
    "residual_forwarded": "ResidualRequest",
    "proof_submitted": "ProofSubmission",
    "proof_accepted": "VerificationResult",
    "proof_rejected": "VerificationResult",
    "reference_validation_accepted": "VerificationResult",
    "state_committed": "StateCommit",
    "request_completed": "RequestResult",
    "request_declined": "RequestResult",
}

MESSAGE_CATEGORIES = {
    "FacilityRegistration": "registration",
    "Request": "coordination",
    "Claim": "coordination",
    "ResidualRequest": "coordination",
    "RequestResult": "coordination",
    "ProofSubmission": "integrity",
    "VerificationResult": "integrity",
    "StateCommit": "integrity",
}

POLYVINE_TRACE_SCHEMA = "polyvine/ProtocolEvent/v1"
POLYVINE_MESSAGE_TYPES = {"EMBED", "SUCCESS", "FAILURE", "RELAY", "CONNECT", "ACK"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _artifact(run_dir: Path, relative_path: str) -> tuple[Any, int]:
    path = run_dir / relative_path
    return json.loads(path.read_text(encoding="utf-8")), path.stat().st_size


def message_payload(record: dict[str, Any], run_dir: Path) -> tuple[str, dict[str, Any], dict[str, int]]:
    event_type = record["event_type"]
    message_type = MESSAGE_TYPES[event_type]
    fields = {
        key: value
        for key, value in record.items()
        if key not in {"event_type", "record_index", "timestamp_utc", "proof_path", "public_input_path"}
    }
    artifact_sizes: dict[str, int] = {}
    if event_type == "proof_submitted":
        proof, artifact_sizes["proof_bytes"] = _artifact(run_dir, record["proof_path"])
        public_inputs, artifact_sizes["public_input_bytes"] = _artifact(
            run_dir, record["public_input_path"]
        )
        fields["proof"] = proof
        fields["public_inputs"] = public_inputs
    payload = {"schema": f"zk-vne/{message_type}/v1", **fields}
    return message_type, payload, artifact_sizes


def _empty_metrics() -> dict[str, Any]:
    return {
        "logical_messages": 0,
        "messages_by_type": {},
        "messages_by_category": {},
        "canonical_control_plane_bytes": 0,
        "bytes_by_message_type": {},
        "bytes_by_category": {},
        "proof_artifact_bytes": 0,
        "public_input_artifact_bytes": 0,
        "transparency_log_records": 0,
        "transparency_log_bytes": 0,
        "claims": 0,
        "proofs": 0,
        "verifications": 0,
        "state_transitions": 0,
        "request_response_round_trips": 0,
        "forwarding_hops": 0,
        "participating_facilities": [],
        "claim_sequence": [],
        "outcome": None,
    }


def analyze_protocol_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    log_path = run_dir / "transparency_log.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    registration = _empty_metrics()
    requests: dict[int, dict[str, Any]] = {}

    for record in records:
        request_id = record.get("request_id")
        target = registration if request_id is None else requests.setdefault(int(request_id), _empty_metrics())
        encoded_record = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        target["transparency_log_records"] += 1
        target["transparency_log_bytes"] += len(encoded_record)

        event_type = record.get("event_type")
        if event_type not in MESSAGE_TYPES:
            continue
        message_type, payload, artifacts = message_payload(record, run_dir)
        payload_bytes = len(canonical_json_bytes(payload))
        target["logical_messages"] += 1
        target["canonical_control_plane_bytes"] += payload_bytes
        messages = Counter(target["messages_by_type"])
        messages[message_type] += 1
        target["messages_by_type"] = dict(sorted(messages.items()))
        byte_counts = Counter(target["bytes_by_message_type"])
        byte_counts[message_type] += payload_bytes
        target["bytes_by_message_type"] = dict(sorted(byte_counts.items()))
        category = MESSAGE_CATEGORIES[message_type]
        category_messages = Counter(target["messages_by_category"])
        category_messages[category] += 1
        target["messages_by_category"] = dict(sorted(category_messages.items()))
        category_bytes = Counter(target["bytes_by_category"])
        category_bytes[category] += payload_bytes
        target["bytes_by_category"] = dict(sorted(category_bytes.items()))
        target["proof_artifact_bytes"] += artifacts.get("proof_bytes", 0)
        target["public_input_artifact_bytes"] += artifacts.get("public_input_bytes", 0)

        facility_id = record.get("facility_id")
        if request_id is not None and facility_id:
            target["participating_facilities"].append(facility_id)
        if event_type == "claim_submitted":
            target["claims"] += 1
            target["claim_sequence"].append(record["claim_type"])
        elif event_type == "proof_submitted":
            target["proofs"] += 1
        elif event_type in {"proof_accepted", "proof_rejected", "reference_validation_accepted"}:
            target["verifications"] += 1
        elif event_type == "state_committed":
            target["state_transitions"] += 1
        elif event_type in {"request_forwarded", "residual_forwarded"}:
            target["forwarding_hops"] += 1
        elif event_type == "request_completed":
            target["outcome"] = "accepted"
        elif event_type == "request_declined":
            target["outcome"] = "declined"

    for metrics in requests.values():
        metrics["participating_facilities"] = sorted(set(metrics["participating_facilities"]))
        metrics["request_response_round_trips"] = metrics["claims"] + metrics["verifications"]
    registration["participating_facilities"] = sorted(set(registration["participating_facilities"]))

    return {
        "schema_version": 1,
        "encoding": "canonical JSON: UTF-8, sorted keys, no insignificant whitespace",
        "round_definition": "one Request/Claim or ProofSubmission/VerificationResult exchange",
        "registration_prephase": registration,
        "requests": {str(key): value for key, value in sorted(requests.items())},
        "total_log_bytes": log_path.stat().st_size,
        "total_log_records": len(records),
    }


def analyze_polyvine_stats(path: Path) -> dict[str, Any]:
    """Import PolyViNE's native message counters without estimating byte sizes."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("vnr_id,"))
    row = next(line for line in lines[header_index + 1 :] if line and not line.startswith("summary:"))
    parsed = next(csv.DictReader([lines[header_index], row]))
    message_fields = {
        "EMBED": "embed_messages",
        "SUCCESS": "success_messages",
        "FAILURE": "failure_messages",
        "RELAY": "relay_messages",
        "CONNECT": "connect_messages",
        "ACK": "ack_messages",
    }
    messages_by_type = {name: int(parsed[field]) for name, field in message_fields.items()}
    counted_total = sum(messages_by_type.values())
    reported_total = int(parsed["messages"])
    return {
        "schema_version": 1,
        "system": "polyvine",
        "accepted": parsed["accepted"] == "yes",
        "logical_messages": reported_total,
        "messages_by_type": messages_by_type,
        "message_breakdown_sum": counted_total,
        "message_count_consistent": counted_total == reported_total,
        "forwarding_rounds_native": int(parsed["forwarding_rounds"]),
        "max_hops": int(parsed["max_hops"]),
        "participating_facilities": int(parsed["participating_inps"]),
        "canonical_control_plane_bytes": None,
        "byte_measurement_status": (
            "unavailable: PolyViNE counts in-memory message types but does not serialize payloads"
        ),
        "message_count_status": (
            "consistent"
            if counted_total == reported_total
            else "reported total differs from the sum of native message counters"
        ),
    }


def _validate_actor(actor: Any, *, field: str) -> None:
    if not isinstance(actor, dict) or not isinstance(actor.get("role"), str):
        raise ValueError(f"PolyViNE trace {field} must be an actor object")
    if "inp_id" in actor and not isinstance(actor["inp_id"], int):
        raise ValueError(f"PolyViNE trace {field}.inp_id must be an integer")


def _validate_polyvine_event(event: Any, *, sequence: int) -> None:
    if not isinstance(event, dict):
        raise ValueError(f"PolyViNE trace event {sequence} must be an object")
    if event.get("schema") != POLYVINE_TRACE_SCHEMA:
        raise ValueError(f"PolyViNE trace event {sequence} has an unsupported schema")
    if event.get("sequence") != sequence:
        raise ValueError(
            f"PolyViNE trace sequence is not contiguous: expected {sequence}, "
            f"got {event.get('sequence')}"
        )
    message_type = event.get("message_type")
    if message_type not in POLYVINE_MESSAGE_TYPES:
        raise ValueError(f"PolyViNE trace event {sequence} has invalid message type {message_type!r}")
    if not isinstance(event.get("request_id"), int):
        raise ValueError(f"PolyViNE trace event {sequence} has no integer request_id")
    _validate_actor(event.get("sender"), field="sender")
    _validate_actor(event.get("receiver"), field="receiver")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"PolyViNE trace event {sequence} has no payload object")
    if message_type in {"EMBED", "RELAY"}:
        if not isinstance(payload.get("residual_vnr"), dict) or not isinstance(
            payload.get("request_state"), dict
        ):
            raise ValueError(f"PolyViNE {message_type} event {sequence} lacks request data")
    elif message_type in {"SUCCESS", "CONNECT"} and not isinstance(
        payload.get("request_state"), dict
    ):
        raise ValueError(f"PolyViNE {message_type} event {sequence} lacks request_state")
    elif message_type == "FAILURE" and not isinstance(payload.get("reason"), str):
        raise ValueError(f"PolyViNE FAILURE event {sequence} lacks a reason")
    elif message_type == "ACK" and payload.get("selected") is not True:
        raise ValueError(f"PolyViNE ACK event {sequence} is not a selection acknowledgement")


def analyze_polyvine_trace(trace_path: Path, stats_path: Path) -> dict[str, Any]:
    """Measure physical PolyViNE sends from its canonical benchmark trace."""
    lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError("PolyViNE protocol trace is empty")
    events = [json.loads(line) for line in lines]
    for sequence, event in enumerate(events):
        _validate_polyvine_event(event, sequence=sequence)

    request_ids = {event["request_id"] for event in events}
    if len(request_ids) != 1:
        raise ValueError(f"PolyViNE trace contains multiple request IDs: {sorted(request_ids)}")

    message_counts: Counter[str] = Counter()
    byte_counts: Counter[str] = Counter()
    participating_inps: set[int] = set()
    total_bytes = 0
    for event in events:
        message_type = event["message_type"]
        encoded_bytes = len(canonical_json_bytes(event))
        message_counts[message_type] += 1
        byte_counts[message_type] += encoded_bytes
        total_bytes += encoded_bytes
        for actor_field in ("sender", "receiver"):
            inp_id = event[actor_field].get("inp_id")
            if inp_id is not None:
                participating_inps.add(inp_id)

    native = analyze_polyvine_stats(stats_path)
    accepted = native["accepted"]
    if accepted != any(event["message_type"] == "SUCCESS" for event in events):
        raise ValueError("PolyViNE trace SUCCESS event disagrees with native accepted outcome")

    return {
        "schema_version": 1,
        "system": "polyvine",
        "request_id": next(iter(request_ids)),
        "outcome": "accepted" if accepted else "declined",
        "logical_messages": len(events),
        "messages_by_type": dict(sorted(message_counts.items())),
        "messages_by_category": {"coordination": len(events)},
        "canonical_control_plane_bytes": total_bytes,
        "bytes_by_message_type": dict(sorted(byte_counts.items())),
        "bytes_by_category": {"coordination": total_bytes},
        "forwarding_rounds_native": native["forwarding_rounds_native"],
        "participating_facilities": sorted(participating_inps),
        "native_stats": native,
        "legacy_reported_message_count": native["logical_messages"],
        "trace_minus_legacy_reported_count": len(events) - native["logical_messages"],
        "trace_matches_legacy_reported_count": len(events) == native["logical_messages"],
        "counting_policy": (
            "The structured trace counts physical sends. Native EMBED counters also count "
            "processing of a RELAY at its receiver and are retained only as diagnostics."
        ),
        "privacy_notice": (
            "The synthetic benchmark trace contains mapping details for byte accounting; "
            "it is instrumentation output, not a privacy-preserving PolyViNE wire format."
        ),
    }
