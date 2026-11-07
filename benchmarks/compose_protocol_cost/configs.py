#!/usr/bin/env python3
"""Generate self-contained facility configs from the neutral P1-P5 scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.protocol_cost.scenarios import (
    SCENARIOS,
    WORKLOAD,
    load_scenario_document,
    parse_scenarios,
    scenario_configs,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "configs" / "system" / "benchmarks"


def compose_portal_request(
    scenario: Any,
    workload: dict[str, Any] = WORKLOAD,
) -> dict[str, Any]:
    """Adapt neutral workload fields to the public ZK-VNE API schema."""
    portal, _ = scenario_configs(scenario, workload)
    request = portal["requests"][0]
    return {
        "api_version": "v1",
        "request_id": request["request_id"],
        "facility_order": request["facility_order"],
        "vnr": {
            "nodes": [
                {
                    key: node[key]
                    for key in ("id", "cpu", "memory_gib", "arch")
                    if key in node
                }
                for node in request["vnr"]["nodes"]
            ],
            "edges": [
                {
                    key: edge[key]
                    for key in ("u", "v", "bandwidth")
                    if key in edge
                }
                for edge in request["vnr"]["edges"]
            ],
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_compose_configs(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    scenarios: tuple[Any, ...] = SCENARIOS,
    workload: dict[str, Any] = WORKLOAD,
) -> list[Path]:
    written: list[Path] = []
    for scenario in scenarios:
        _, private = scenario_configs(scenario, workload)
        by_id = {item["facility_id"]: item for item in private["facilities"]}
        for service, facility_id in (
            ("facility-a", "facility-A"),
            ("facility-b", "facility-B"),
        ):
            path = output_dir / service / f"{scenario.scenario_id}.json"
            _write_json(
                path,
                {
                    "facilities": [by_id[facility_id]],
                    "public_connectivity": [],
                },
            )
            written.append(path)

    facility_ids = {item["facility_id"] for item in workload["facilities"]}
    if facility_ids != {"facility-A", "facility-B"}:
        raise ValueError(f"Compose benchmark expects facilities A and B, got {facility_ids}")
    peering = workload["peering"]
    endpoint_by_facility = {
        item["facility_id"]: item["endpoint_id"] for item in workload["facilities"]
    }
    portal_path = output_dir / "portal.json"
    _write_json(
        portal_path,
        {
            "facilities": {
                "facility-A": "http://facility-a:8000",
                "facility-B": "http://facility-b:8000",
            },
            "public_links": [
                {
                    "endpoint_a": endpoint_by_facility[peering["facility_a"]],
                    "endpoint_b": endpoint_by_facility[peering["facility_b"]],
                    "bandwidth": int(peering["bandwidth"]),
                }
            ],
        },
    )
    written.append(portal_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario-config", type=Path)
    args = parser.parse_args()
    if args.scenario_config is None:
        workload, scenarios = WORKLOAD, SCENARIOS
    else:
        workload, scenarios = parse_scenarios(load_scenario_document(args.scenario_config))
    paths = generate_compose_configs(
        args.output_dir,
        scenarios=scenarios,
        workload=workload,
    )
    print(f"Generated {len(paths)} Compose benchmark configs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
