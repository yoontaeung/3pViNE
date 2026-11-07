"""Durable, incremental result logging shared by benchmark runners."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def source_manifest(root: Path, paths: list[Path]) -> dict[str, Any]:
    """Fingerprint source inputs so dirty-worktree experiments remain identifiable."""
    files: dict[str, dict[str, Any]] = {}
    for path in sorted({item.resolve() for item in paths}):
        relative = path.relative_to(root.resolve()).as_posix()
        payload = path.read_bytes()
        files[relative] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def git_metadata(root: Path) -> dict[str, Any]:
    """Record the repository revision and dirty-worktree status."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.splitlines()
    return {
        "revision": revision or "unknown",
        "dirty": bool(status),
        "status": status,
    }


class ExperimentLog:
    """Append events to JSONL and maintain a human-readable result checkpoint."""

    def __init__(self, run_dir: Path, *, experiment: str, metadata: dict[str, Any]) -> None:
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.journal_path = self.run_dir / "journal.jsonl"
        self.results_path = self.run_dir / "results.json"
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "experiment": experiment,
            "status": "running",
            "started_utc": utc_timestamp(),
            "completed_utc": None,
            "metadata": metadata,
            "profiles": {},
            "events_recorded": 0,
        }
        self._checkpoint()
        self.append("experiment_started")

    def append(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            "sequence": self.state["events_recorded"],
            "timestamp_utc": utc_timestamp(),
            "event_type": event_type,
            **fields,
        }
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.state["events_recorded"] += 1
        self._checkpoint()
        summary = fields.get("profile_id", "")
        print(f"[{event_type}] {summary}", flush=True)
        return event

    def update_profile(self, profile_id: str, **fields: Any) -> None:
        profile = self.state["profiles"].setdefault(
            profile_id,
            {"status": "pending", "measurements": []},
        )
        profile.update(fields)
        self._checkpoint()

    def add_measurement(self, profile_id: str, measurement: dict[str, Any]) -> None:
        profile = self.state["profiles"].setdefault(
            profile_id,
            {"status": "running", "measurements": []},
        )
        profile["measurements"].append(measurement)
        self.append(
            "measurement_completed",
            profile_id=profile_id,
            repetition=measurement["repetition"],
            online_groth16_seconds=measurement["online_groth16_seconds"],
        )
        self._checkpoint()

    def complete(self) -> None:
        self.state["status"] = "completed"
        self.state["completed_utc"] = utc_timestamp()
        self.append("experiment_completed")
        self._checkpoint()

    def fail(self, error: str) -> None:
        self.state["status"] = "failed"
        self.state["completed_utc"] = utc_timestamp()
        self.state["error"] = error
        self.append("experiment_failed", error=error)
        self._checkpoint()

    def _checkpoint(self) -> None:
        temporary = self.results_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.results_path)
