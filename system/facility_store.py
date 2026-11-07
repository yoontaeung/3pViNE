"""Atomic persistence for one facility's private protocol state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FacilityPrivateStore:
    """Persist private state in a Docker-managed facility volume."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict[str, Any]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported facility state schema")
        return value

    def save(self, value: dict[str, Any]) -> None:
        document = {"schema_version": self.SCHEMA_VERSION, **value}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
