"""Deterministic protocol traces for control-plane cost measurements."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProtocolScenario:
    scenario_id: str
    description: str
    expected_outcome: str
    expected_claims: tuple[str, ...]
    request_id: int
    facility_order: tuple[str, ...]
    facility_cpu: dict[str, tuple[int, ...]]
    polyvine_initial_fanout: int


SCENARIO_PATH = Path(__file__).resolve().parents[1] / "config" / "protocol_scenarios.json"


def load_scenario_document(path: Path = SCENARIO_PATH) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported protocol-cost scenario schema")
    return document


def parse_scenarios(document: dict[str, Any]) -> tuple[dict[str, Any], tuple[ProtocolScenario, ...]]:
    workload = document["workload"]
    scenarios = tuple(
        ProtocolScenario(
            scenario_id=item["scenario_id"],
            description=item["description"],
            expected_outcome=item["expected_outcome"],
            expected_claims=tuple(item["expected_claims"]),
            request_id=int(item["request_id"]),
            facility_order=tuple(item["facility_order"]),
            facility_cpu={
                facility_id: tuple(int(value) for value in capacities)
                for facility_id, capacities in item["facility_cpu"].items()
            },
            polyvine_initial_fanout=int(item.get("polyvine_initial_fanout", 1)),
        )
        for item in document["scenarios"]
    )
    return workload, scenarios


SCENARIO_DOCUMENT = load_scenario_document()
WORKLOAD, SCENARIOS = parse_scenarios(SCENARIO_DOCUMENT)


def _vnr(workload: dict[str, Any] = WORKLOAD) -> dict[str, Any]:
    request = workload["request"]
    return {
        "nodes": [
            {
                "id": node["id"],
                "cpu": int(node["cpu"]),
                "memory_gib": 0,
                "arch": 0,
                "x": int(node["x"]),
                "y": int(node["y"]),
            }
            for node in request["nodes"]
        ],
        "edges": [
            {
                "u": edge["u"],
                "v": edge["v"],
                "bandwidth": int(edge["bandwidth"]),
                "delay": int(edge["delay"]),
            }
            for edge in request["edges"]
        ],
    }


def _facility_template(
    facility_id: str,
    workload: dict[str, Any] = WORKLOAD,
) -> dict[str, Any]:
    return next(
        item for item in workload["facilities"] if item["facility_id"] == facility_id
    )


def _endpoint_node(facility_id: str, workload: dict[str, Any]) -> str:
    peering = workload["peering"]
    if facility_id == peering["facility_a"]:
        return str(peering["node_a"])
    if facility_id == peering["facility_b"]:
        return str(peering["node_b"])
    raise ValueError(f"facility {facility_id} is not part of the configured peering link")


def substrate_edges(workload: dict[str, Any] = WORKLOAD) -> list[dict[str, Any]]:
    configured = workload.get("substrate_edges")
    if configured is not None:
        return list(configured)
    topology = workload["substrate_topology"]
    if topology.get("type") != "full_mesh":
        raise ValueError("only explicit edges and full_mesh substrate topology are supported")
    node_count = int(topology["node_count"])
    return [
        {
            "u": f"s{source}",
            "v": f"s{target}",
            "bandwidth": int(topology["bandwidth"]),
            "delay": int(topology.get("delay", 1)),
        }
        for source in range(node_count)
        for target in range(source + 1, node_count)
    ]


def _substrate(
    facility_id: str,
    cpu: tuple[int, ...],
    workload: dict[str, Any] = WORKLOAD,
) -> dict[str, Any]:
    facility = _facility_template(facility_id, workload)
    endpoint_node = _endpoint_node(facility_id, workload)
    nodes = [
        {
            "id": f"s{index}",
            "cpu": capacity,
            "memory_gib": 0,
            "arch": 0 if f"s{index}" == endpoint_node else "ARM",
            "x": int(facility["coordinate_x"]) + index,
            "y": index,
            **({"public_endpoint": True} if f"s{index}" == endpoint_node else {}),
        }
        for index, capacity in enumerate(cpu)
    ]
    edges = [
        {
            "u": edge["u"],
            "v": edge["v"],
            "bandwidth": int(edge["bandwidth"]),
            "delay": int(edge["delay"]),
        }
        for edge in substrate_edges(workload)
    ]
    return {"nodes": nodes, "edges": edges}


def scenario_configs(
    scenario: ProtocolScenario,
    workload: dict[str, Any] = WORKLOAD,
) -> tuple[dict[str, Any], dict[str, Any]]:
    portal = {
        "requests": [
            {
                "request_id": scenario.request_id,
                "facility_order": list(scenario.facility_order),
                "vnr": _vnr(workload),
            }
        ]
    }
    facilities = []
    for index, facility_id in enumerate(("facility-A", "facility-B"), start=1):
        facility_template = _facility_template(facility_id, workload)
        facilities.append(
            {
                "facility_id": facility_id,
                "initial_rho_S": scenario.request_id * 10 + index,
                "public_endpoint": {
                    "endpoint_id": facility_template["endpoint_id"],
                    "substrate_node": _endpoint_node(facility_id, workload),
                },
                "solver": {"objective": "min_hops", "max_seconds": 5},
                "randomness_by_request": {
                    str(scenario.request_id): {
                        "rho_M": scenario.request_id * 100 + index * 2,
                        "rho_S_next": scenario.request_id * 100 + index * 2 + 1,
                    }
                },
                "substrate": _substrate(
                    facility_id,
                    scenario.facility_cpu[facility_id],
                    workload,
                ),
            }
        )
    private = {
        "facilities": facilities,
        "public_connectivity": [
            {
                "endpoint_a": "endpoint-A",
                "endpoint_b": "endpoint-B",
                "bandwidth": int(workload["peering"]["bandwidth"]),
                "latency_ms": int(workload["peering"]["delay"]),
            }
        ],
    }
    return portal, private


def write_scenario_configs(
    scenario: ProtocolScenario,
    directory: Path,
    workload: dict[str, Any] = WORKLOAD,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    portal, private = scenario_configs(scenario, workload)
    portal_path = directory / "portal_requests.json"
    private_path = directory / "facility_private.json"
    portal_path.write_text(json.dumps(portal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    private_path.write_text(json.dumps(private, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return portal_path, private_path
