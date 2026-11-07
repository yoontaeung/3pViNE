#!/usr/bin/env python3
"""Generate native ZK-VNE and PolyViNE inputs from one neutral scenario."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.protocol_cost.scenarios import (
    ProtocolScenario,
    WORKLOAD,
    scenario_configs,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _facility_template(facility_id: str) -> dict[str, Any]:
    return next(
        item for item in WORKLOAD["facilities"] if item["facility_id"] == facility_id
    )


def _polyvine_nodes(scenario: ProtocolScenario, facility_id: str) -> list[dict[str, Any]]:
    template = _facility_template(facility_id)
    return [
        {
            "id": f"s{index}",
            "global_id": int(template["global_node_base"]) + index,
            "x": int(template["coordinate_x"]) + index,
            "y": index,
            "cpu": scenario.facility_cpu[facility_id][index],
            "is_border": int(index == 3),
        }
        for index in range(4)
    ]


def generate_polyvine_inputs(scenario: ProtocolScenario, output_dir: Path) -> None:
    subs_dir = output_dir / "subs"
    vnrs_dir = output_dir / "vnrs"
    subs_dir.mkdir(parents=True, exist_ok=True)
    vnrs_dir.mkdir(parents=True, exist_ok=True)

    nodes_by_facility: dict[str, list[dict[str, Any]]] = {}
    for facility_id in ("facility-A", "facility-B"):
        template = _facility_template(facility_id)
        nodes = _polyvine_nodes(scenario, facility_id)
        nodes_by_facility[facility_id] = nodes
        node_lookup = {node["id"]: node for node in nodes}
        lines = [
            f"{len(nodes)} {len(WORKLOAD['substrate_edges'])}",
            "cpu is_border",
            "bandwidth delay",
        ]
        for node in nodes:
            lines.append(
                f"{node['global_id']} {node['x']} {node['y']} "
                f"{float(node['cpu']):.6f} {node['is_border']}"
            )
        for edge in WORKLOAD["substrate_edges"]:
            lines.append(
                f"{node_lookup[edge['u']]['global_id']} "
                f"{node_lookup[edge['v']]['global_id']} "
                f"{float(edge['bandwidth']):.6f} {float(edge['delay']):.6f}"
            )
        (subs_dir / f"sn_inp_{template['inp_id']}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    peering = WORKLOAD["peering"]
    node_a = next(
        node for node in nodes_by_facility[peering["facility_a"]] if node["id"] == peering["node_a"]
    )
    node_b = next(
        node for node in nodes_by_facility[peering["facility_b"]] if node["id"] == peering["node_b"]
    )
    template_a = _facility_template(peering["facility_a"])
    template_b = _facility_template(peering["facility_b"])
    (subs_dir / "peering_edges.txt").write_text(
        "1\n"
        "inp_i u inp_j v bandwidth delay\n"
        f"{template_a['inp_id']} {node_a['global_id']} "
        f"{template_b['inp_id']} {node_b['global_id']} "
        f"{float(peering['bandwidth']):.6f} {float(peering['delay']):.6f}\n",
        encoding="utf-8",
    )

    request = WORKLOAD["request"]
    node_index = {node["id"]: index for index, node in enumerate(request["nodes"])}
    request_lines = [
        (
            f"{len(request['nodes'])} {len(request['edges'])} 0 "
            f"{request['arrival_time']} {request['duration']} 0 {request['maximum_distance']}"
        ),
        "cpu",
        "bandwidth delay",
    ]
    request_lines.extend(
        f"{node['x']} {node['y']} {float(node['cpu']):.6f}"
        for node in request["nodes"]
    )
    request_lines.extend(
        f"{node_index[edge['u']]} {node_index[edge['v']]} "
        f"{float(edge['bandwidth']):.6f} {float(edge['delay']):.6f}"
        for edge in request["edges"]
    )
    (vnrs_dir / "req0.txt").write_text("\n".join(request_lines) + "\n", encoding="utf-8")


def write_matched_inputs(scenario: ProtocolScenario, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    portal, private = scenario_configs(scenario)
    zkvne_dir = output_dir / "zkvne"
    portal_path = zkvne_dir / "portal_requests.json"
    private_path = zkvne_dir / "facility_private.json"
    _write_json(portal_path, portal)
    _write_json(private_path, private)
    generate_polyvine_inputs(scenario, output_dir / "polyvine")
    _write_json(
        output_dir / "scenario_manifest.json",
        {
            "scenario_id": scenario.scenario_id,
            "request_id": scenario.request_id,
            "expected_outcome": scenario.expected_outcome,
            "expected_claims": list(scenario.expected_claims),
            "facility_order": list(scenario.facility_order),
            "polyvine_initial_fanout": scenario.polyvine_initial_fanout,
            "facility_cpu": scenario.facility_cpu,
            "workload": WORKLOAD,
        },
    )
    return {
        "portal_requests": portal_path,
        "facility_private": private_path,
        "polyvine_subs": output_dir / "polyvine" / "subs",
        "polyvine_vnrs": output_dir / "polyvine" / "vnrs",
        "manifest": output_dir / "scenario_manifest.json",
    }

