"""Deterministic valid witnesses for arbitrary fixed circuit profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.zk_overhead.profiles import CircuitProfile
from zk_encoding import compute_poseidon_commitments, stringify_field_values


def upper_pairs(vn: int) -> list[tuple[int, int]]:
    return [(u, v) for u in range(vn) for v in range(u + 1, vn)]


def zero_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def one_hot(index: int, size: int) -> list[int]:
    return [int(position == index) for position in range(size)]


def build_replay_fields(profile: CircuitProfile, *, request_id: int = 1) -> dict[str, Any]:
    """Build a chain VNR embedded directly into a complete private substrate."""
    vn, sn, hops = profile.vn, profile.sn, profile.hops
    pairs = upper_pairs(vn)

    v_link_present = zero_matrix(vn)
    v_bw = zero_matrix(vn)
    for vnode in range(vn - 1):
        v_link_present[vnode][vnode + 1] = 1
        v_link_present[vnode + 1][vnode] = 1
        v_bw[vnode][vnode + 1] = 1
        v_bw[vnode + 1][vnode] = 1

    s_adj = zero_matrix(sn)
    s_bw_before = zero_matrix(sn)
    for source in range(sn):
        for target in range(source + 1, sn):
            s_adj[source][target] = s_adj[target][source] = 1
            s_bw_before[source][target] = s_bw_before[target][source] = 100

    # Multiple virtual nodes may share one substrate host. Cycling the mapping
    # keeps the generated replay valid even when the VNR bound exceeds SN.
    node_map = [vnode % sn for vnode in range(vn)]
    node_sel = [one_hot(index, sn) for index in node_map]
    path_nodes = [[0 for _ in range(hops + 1)] for _ in pairs]
    path_node_sel = [
        [[0 for _ in range(sn)] for _ in range(hops + 1)]
        for _ in pairs
    ]
    path_len_sel = [[0 for _ in range(hops + 1)] for _ in pairs]

    for pair_index, (source, target) in enumerate(pairs):
        if target != source + 1:
            continue
        source_host = node_map[source]
        target_host = node_map[target]
        hop_count = int(source_host != target_host)
        path_len_sel[pair_index][hop_count] = 1
        path_nodes[pair_index][0] = source_host
        path_node_sel[pair_index][0] = one_hot(source_host, sn)
        if hop_count:
            path_nodes[pair_index][1] = target_host
            path_node_sel[pair_index][1] = one_hot(target_host, sn)

    s_cpu_before = [100 for _ in range(sn)]
    s_mem_before = [100 for _ in range(sn)]
    s_cpu_after = list(s_cpu_before)
    s_mem_after = list(s_mem_before)
    for substrate_node in node_map:
        s_cpu_after[substrate_node] -= 1
        s_mem_after[substrate_node] -= 1

    s_bw_after = [list(row) for row in s_bw_before]
    for vnode in range(vn - 1):
        source_host = node_map[vnode]
        target_host = node_map[vnode + 1]
        if source_host == target_host:
            continue
        s_bw_after[source_host][target_host] -= 1
        s_bw_after[target_host][source_host] -= 1

    return {
        "reqId": request_id,
        "vNodePresent": [1 for _ in range(vn)],
        "vCpu": [1 for _ in range(vn)],
        "vMem": [1 for _ in range(vn)],
        "vArch": [0 for _ in range(vn)],
        "vLinkPresent": v_link_present,
        "vBw": v_bw,
        "endpointEnabled": 0,
        "endpointIdx": 0,
        "sNodePresent": [1 for _ in range(sn)],
        "sCpuBefore": s_cpu_before,
        "sCpuAfter": s_cpu_after,
        "sMemBefore": s_mem_before,
        "sMemAfter": s_mem_after,
        "sArch": [0 for _ in range(sn)],
        "sAdj": s_adj,
        "sBwBefore": s_bw_before,
        "sBwAfter": s_bw_after,
        "nodeMap": node_map,
        "nodeSel": node_sel,
        "pathNodes": path_nodes,
        "pathNodeSel": path_node_sel,
        "pathLenSel": path_len_sel,
        "rho_S": 11001,
        "rho_M": 11002,
        "rho_S_next": 11003,
    }


def add_commitments(data: dict[str, Any]) -> dict[str, Any]:
    data.update(compute_poseidon_commitments(data))
    return data


def write_input(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stringify_field_values(data), separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_replay_dimensions(profile: CircuitProfile, data: dict[str, Any]) -> None:
    expected = {
        "vNodePresent": profile.vn,
        "vCpu": profile.vn,
        "nodeMap": profile.vn,
        "nodeSel": profile.vn,
        "sNodePresent": profile.sn,
        "sCpuBefore": profile.sn,
        "sAdj": profile.sn,
        "pathNodes": profile.pair_count,
        "pathNodeSel": profile.pair_count,
        "pathLenSel": profile.pair_count,
    }
    for field, length in expected.items():
        if len(data[field]) != length:
            raise ValueError(f"{field} has length {len(data[field])}, expected {length}")
    if any(value < 0 for value in data["sCpuAfter"] + data["sMemAfter"]):
        raise ValueError("deterministic replay produced a negative node resource")
