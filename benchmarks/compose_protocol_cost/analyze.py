"""Analyze sanitized Compose HTTP traces and public protocol records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from system.http_trace import TRACE_SCHEMA


INTEGRITY_TYPES = {
    "proof_accepted",
    "proof_artifact_fetch",
    "proof_artifact_upload",
    "state_commit_command",
    "state_transition_publish",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sum_float_maps(values: Iterable[dict[str, Any]]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for value in values:
        for key, item in value.items():
            totals[key] += float(item)
    return dict(sorted(totals.items()))


def analyze_compose_run(
    run_dir: Path,
    *,
    request_id: int,
    started_unix_ns: int,
    finished_unix_ns: int,
    online_seconds: float,
) -> dict[str, Any]:
    trace_paths = sorted((run_dir / "http_traces").glob("*.jsonl"))
    exchanges = []
    for path in trace_paths:
        for record in _jsonl(path):
            if record.get("schema") != TRACE_SCHEMA:
                raise ValueError(f"unsupported HTTP trace schema in {path}")
            timestamp = int(record["timestamp_unix_ns"])
            if started_unix_ns <= timestamp <= finished_unix_ns:
                exchanges.append(record)
    exchanges.sort(key=lambda item: (item["timestamp_unix_ns"], item["sender"], item["sequence"]))

    message_counts = Counter(item["semantic_type"] for item in exchanges)
    bytes_by_type: defaultdict[str, int] = defaultdict(int)
    seconds_by_type: defaultdict[str, float] = defaultdict(float)
    category_messages: Counter[str] = Counter()
    category_bytes: defaultdict[str, int] = defaultdict(int)
    total_body_bytes = 0
    total_auth_bytes = 0
    for item in exchanges:
        semantic_type = item["semantic_type"]
        body_bytes = int(item["request_body_bytes"]) + int(item["response_body_bytes"])
        auth_bytes = int(item["auth_header_bytes"])
        exchange_bytes = body_bytes + auth_bytes
        category = "integrity" if semantic_type in INTEGRITY_TYPES else "coordination"
        bytes_by_type[semantic_type] += exchange_bytes
        seconds_by_type[semantic_type] += float(item["elapsed_seconds"])
        category_messages[category] += 1
        category_bytes[category] += exchange_bytes
        total_body_bytes += body_bytes
        total_auth_bytes += auth_bytes

    public_dir = run_dir / "public_log"
    log_path = public_dir / "transparency_log.jsonl"
    records = _jsonl(log_path)
    request_records = [item for item in records if item.get("request_id") == request_id]
    claim_records = [
        item for item in request_records if item.get("event_type") == "claim_submitted"
    ]
    proof_records = [
        item for item in request_records if item.get("event_type") == "proof_accepted"
    ]
    state_records = [
        item for item in request_records if item.get("event_type") == "state_committed"
    ]
    terminal = next(
        (
            item
            for item in reversed(request_records)
            if item.get("event_type") in {"request_completed", "request_declined"}
        ),
        None,
    )
    if terminal is None:
        raise ValueError(f"request {request_id} has no terminal public record")

    artifact_paths = sorted((public_dir / "artifacts").glob("*.json"))
    proof_artifact_bytes = sum(path.stat().st_size for path in artifact_paths)
    log_request_bytes = sum(
        len((json.dumps(item, sort_keys=True) + "\n").encode("utf-8"))
        for item in request_records
    )
    timings = _sum_float_maps(
        item.get("timings_seconds", {}) for item in claim_records
    )
    timings["verify"] = sum(
        float(item.get("verification_seconds", 0.0)) for item in proof_records
    )
    timings["online_total"] = online_seconds

    return {
        "schema_version": 1,
        "system": "zk-vne-compose",
        "request_id": request_id,
        "outcome": "accepted" if terminal["event_type"] == "request_completed" else "declined",
        "claim_sequence": [item["claim_type"] for item in claim_records],
        "participating_facilities": sorted(
            {item["facility_id"] for item in claim_records}
        ),
        "proofs": len(proof_records),
        "state_transitions": len(state_records),
        "physical_http_exchanges": len(exchanges),
        "messages_by_type": dict(sorted(message_counts.items())),
        "messages_by_category": dict(sorted(category_messages.items())),
        "http_body_bytes": total_body_bytes,
        "authentication_header_bytes": total_auth_bytes,
        "measured_http_bytes": total_body_bytes + total_auth_bytes,
        "bytes_by_type": dict(sorted(bytes_by_type.items())),
        "http_seconds_by_type": dict(sorted(seconds_by_type.items())),
        "bytes_by_category": dict(sorted(category_bytes.items())),
        "proof_artifact_storage_bytes": proof_artifact_bytes,
        "transparency_log_request_records": len(request_records),
        "transparency_log_request_bytes": log_request_bytes,
        "timings_seconds": timings,
        "measurement_scope": (
            "JSON request and response bodies plus ZK-VNE authentication headers; "
            "TCP/IP, HTTP framing, and generic HTTP headers are excluded"
        ),
    }
