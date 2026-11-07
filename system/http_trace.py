"""Opt-in, payload-free HTTP accounting for system experiments."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx

from system.auth import MESSAGE_ID_HEADER, SIGNATURE_HEADER, SIGNER_HEADER


TRACE_SCHEMA = "zk-vne/HttpExchange/v1"
AUTH_HEADERS = (SIGNER_HEADER, MESSAGE_ID_HEADER, SIGNATURE_HEADER)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _selected_header_bytes(headers: Mapping[str, str] | None) -> int:
    if not headers:
        return 0
    lowered = {key.lower(): value for key, value in headers.items()}
    total = 0
    for name in AUTH_HEADERS:
        value = lowered.get(name.lower())
        if value is not None:
            total += len(f"{name}: {value}\r\n".encode("utf-8"))
    return total


class HttpTraceRecorder:
    """Append sanitized exchange metadata without retaining protocol payloads."""

    def __init__(self, path: Path, *, actor_id: str) -> None:
        self.path = path
        self.actor_id = actor_id
        self._lock = threading.Lock()
        self._sequence = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        receiver: str,
        semantic_type: str,
        method: str,
        path: str,
        request_payload: Any | None,
        request_headers: Mapping[str, str] | None,
        response: httpx.Response,
        elapsed_seconds: float,
        request_id: int | None = None,
    ) -> None:
        request_body = (
            canonical_json_bytes(request_payload) if request_payload is not None else b""
        )
        response_body = response.content
        with self._lock:
            record = {
                "schema": TRACE_SCHEMA,
                "sequence": self._sequence,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "timestamp_unix_ns": time.time_ns(),
                "sender": self.actor_id,
                "receiver": receiver,
                "semantic_type": semantic_type,
                "request_id": request_id,
                "method": method.upper(),
                "path": path,
                "status_code": response.status_code,
                "request_body_bytes": len(request_body),
                "response_body_bytes": len(response_body),
                "auth_header_bytes": _selected_header_bytes(request_headers),
                "elapsed_seconds": elapsed_seconds,
                "request_sha256": _sha256(request_body),
                "response_sha256": _sha256(response_body),
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._sequence += 1


def recorder_from_environment(*, actor_id: str) -> HttpTraceRecorder | None:
    path = os.environ.get("ZK_VNE_HTTP_TRACE_PATH")
    if not path:
        return None
    return HttpTraceRecorder(Path(path), actor_id=actor_id)
