#!/usr/bin/env python3
"""Submit one VNR through the deployed portal and check its public trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
FULL_EVENTS = [
    "request_issued",
    "claim_submitted",
    "proof_accepted",
    "state_committed",
    "request_completed",
]
PARTIAL_EVENTS = [
    "request_issued",
    "claim_submitted",
    "proof_accepted",
    "state_committed",
    "residual_forwarded",
    "claim_submitted",
    "proof_accepted",
    "state_committed",
    "request_completed",
]
DECLINE_EVENTS = [
    "request_issued",
    "claim_submitted",
    "claim_submitted",
    "request_declined",
]
RESET_HINT = (
    "The deterministic fixtures require a clean, current Compose state. "
    "If this stack contains an older run, reset it with "
    "`docker compose down -v` followed by `docker compose up -d`."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portal-url", default="http://127.0.0.1:18000")
    parser.add_argument("--log-url", default="http://127.0.0.1:18080")
    parser.add_argument(
        "--requests",
        type=Path,
        default=ROOT / "configs/portal_requests.json",
    )
    parser.add_argument(
        "--request-id",
        type=int,
        help="Override the first fixture's request ID for a fresh run.",
    )
    parser.add_argument(
        "--fixture-index",
        type=int,
        default=0,
        help="Index in configs/portal_requests.json (0=full, 1=partial, 2=decline).",
    )
    parser.add_argument(
        "--expected-status",
        choices=("accepted", "declined"),
        default="accepted",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values: dict[str, Any] = json.loads(args.requests.read_text(encoding="utf-8"))
    request = dict(values["requests"][args.fixture_index])
    if args.request_id is not None:
        request["request_id"] = args.request_id
    request.setdefault("facility_order", ["facility-A"])
    request["api_version"] = "v1"

    response = httpx.post(
        f"{args.portal_url}/v1/requests",
        json=request,
        timeout=1800,
    )
    response.raise_for_status()
    result = response.json()
    if result["status"] != args.expected_status:
        raise RuntimeError(
            f"expected {args.expected_status}, but portal returned "
            f"{result['status']}: {result}\n{RESET_HINT}"
        )

    log_response = httpx.get(f"{args.log_url}/v1/events", timeout=30)
    log_response.raise_for_status()
    request_events = [
        event
        for event in log_response.json()
        if event.get("request_id") == request["request_id"]
    ]
    event_types = [event["event_type"] for event in request_events]
    if result["status"] == "declined":
        expected_events = DECLINE_EVENTS
    elif result["claim_type"] == "partial":
        expected_events = PARTIAL_EVENTS
    else:
        expected_events = FULL_EVENTS
    if event_types != expected_events:
        raise RuntimeError(f"unexpected protocol trace: {event_types}\n{RESET_HINT}")

    print(
        json.dumps(
            {
                "request_id": result["request_id"],
                "status": result["status"],
                "claim_type": result["claim_type"],
                "involved_facilities": result["involved_facilities"],
                "final_state_commitments": result["final_state_commitments"],
                "event_types": event_types,
                "timings_seconds": result["timings_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
