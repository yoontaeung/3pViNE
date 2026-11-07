"""Reference checker for the resource-feasibility relation."""

from __future__ import annotations

from typing import Any

from zk_encoding import (
    HOP_COUNT,
    PAIR_COUNT,
    S_NODE_COUNT,
    V_NODE_COUNT,
    VIRTUAL_PAIRS,
    path_bandwidth_use_matrix,
    reserve_path_resources,
)


def verify_resource_feasibility(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check the same resource constraints enforced by the small ZK circuit."""
    errors: list[str] = []

    node_sel = data["nodeSel"]
    node_map = data["nodeMap"]
    v_node_present = data.get("vNodePresent", [1 for _ in range(V_NODE_COUNT)])
    v_cpu = data["vCpu"]
    v_mem = data.get("vMem", [0 for _ in range(V_NODE_COUNT)])
    v_arch = data.get("vArch", [0 for _ in range(V_NODE_COUNT)])
    v_link_present = data.get(
        "vLinkPresent",
        [[int(data["vBw"][i][j] > 0) for j in range(V_NODE_COUNT)] for i in range(V_NODE_COUNT)],
    )
    v_bw = data["vBw"]
    s_node_present = data["sNodePresent"]
    s_cpu_before = data["sCpuBefore"]
    s_cpu_after = data["sCpuAfter"]
    s_mem_before = data.get("sMemBefore", [0 for _ in range(S_NODE_COUNT)])
    s_mem_after = data.get("sMemAfter", [0 for _ in range(S_NODE_COUNT)])
    s_arch = data.get("sArch", [0 for _ in range(S_NODE_COUNT)])
    s_adj = data.get(
        "sAdj",
        [[int(data["sBwBefore"][i][j] > 0) for j in range(S_NODE_COUNT)] for i in range(S_NODE_COUNT)],
    )
    s_bw_before = data["sBwBefore"]
    s_bw_after = data["sBwAfter"]
    endpoint_enabled = data.get("endpointEnabled", 0)
    endpoint_idx = data.get("endpointIdx", 0)
    path_nodes = data["pathNodes"]
    path_node_sel = data["pathNodeSel"]
    path_len_sel = data["pathLenSel"]
    used_bw = [[0 for _ in range(S_NODE_COUNT)] for _ in range(S_NODE_COUNT)]

    for vnode in range(V_NODE_COUNT):
        if v_node_present[vnode] not in (0, 1):
            errors.append(f"vNodePresent[{vnode}] must be boolean")
        if sum(node_sel[vnode]) != v_node_present[vnode]:
            errors.append(f"virtual node {vnode} must map exactly when it is active")
        expected_map = sum(snode * node_sel[vnode][snode] for snode in range(S_NODE_COUNT))
        if node_map[vnode] != expected_map:
            errors.append(f"nodeMap[{vnode}] does not match its one-hot selector")
        for snode in range(S_NODE_COUNT):
            if node_sel[vnode][snode] and s_node_present[snode] != 1:
                errors.append(f"virtual node {vnode} maps to dummy substrate node {snode}")
        if v_node_present[vnode] == 0 and any((v_cpu[vnode], v_mem[vnode], v_arch[vnode])):
            errors.append(f"inactive virtual node {vnode} must have zero resources")
        if (
            v_node_present[vnode] == 1
            and 0 <= node_map[vnode] < S_NODE_COUNT
            and v_arch[vnode] != 0
            and s_arch[node_map[vnode]] != v_arch[vnode]
        ):
            errors.append(f"virtual node {vnode} maps to substrate node with wrong architecture")

    for vnode in range(V_NODE_COUNT):
        if v_link_present[vnode][vnode] != 0 or v_bw[vnode][vnode] != 0:
            errors.append(f"virtual link matrices must have zero diagonal at {vnode}")
        for other in range(V_NODE_COUNT):
            link_present = v_link_present[vnode][other]
            if link_present not in (0, 1):
                errors.append(f"vLinkPresent[{vnode}][{other}] must be boolean")
            if link_present and not (v_node_present[vnode] and v_node_present[other]):
                errors.append("virtual links cannot touch inactive virtual nodes")
            if v_bw[vnode][other] != 0 and link_present != 1:
                errors.append("nonzero virtual bandwidth requires a present virtual link")
            if v_link_present[vnode][other] != v_link_present[other][vnode]:
                errors.append(f"vLinkPresent is not symmetric at ({vnode},{other})")
            if v_bw[vnode][other] != v_bw[other][vnode]:
                errors.append(f"vBw is not symmetric at ({vnode},{other})")

    if endpoint_enabled not in (0, 1):
        errors.append("endpointEnabled must be boolean")
    if endpoint_enabled:
        if not 0 <= endpoint_idx < S_NODE_COUNT:
            errors.append("endpointIdx is out of range")
        elif s_node_present[endpoint_idx] != 1:
            errors.append("endpointIdx points to a dummy substrate node")
        if node_map[V_NODE_COUNT - 1] != endpoint_idx:
            errors.append("endpoint virtual slot must map to endpointIdx")

    for pair_idx, (src_vnode, dst_vnode) in enumerate(VIRTUAL_PAIRS):
        demand = v_bw[src_vnode][dst_vnode]
        link_present = v_link_present[src_vnode][dst_vnode]
        selected_lengths = [
            hop_count
            for hop_count, selected in enumerate(path_len_sel[pair_idx])
            if selected == 1
        ]
        if any(selected not in (0, 1) for selected in path_len_sel[pair_idx]):
            errors.append(f"pathLenSel[{pair_idx}] must be boolean")
        if len(selected_lengths) > 1:
            errors.append(f"pathLenSel[{pair_idx}] must choose at most one path length")
        if link_present == 1 and len(selected_lengths) != 1:
            errors.append(f"present virtual pair {pair_idx} must choose one path")
        if link_present == 0 and selected_lengths:
            errors.append(f"absent virtual pair {pair_idx} must not choose a path")

        has_path = len(selected_lengths) == 1
        selected_hop_count = selected_lengths[0] if has_path else 0
        for pos in range(HOP_COUNT + 1):
            is_active_pos = has_path and pos <= selected_hop_count
            selector_sum = sum(path_node_sel[pair_idx][pos])
            expected_node = sum(
                snode * path_node_sel[pair_idx][pos][snode]
                for snode in range(S_NODE_COUNT)
            )
            if selector_sum != int(is_active_pos):
                errors.append(
                    f"pathNodeSel[{pair_idx}][{pos}] must select exactly active path nodes"
                )
            if path_nodes[pair_idx][pos] != expected_node:
                errors.append(f"pathNodes[{pair_idx}][{pos}] does not match its selector")
            for snode in range(S_NODE_COUNT):
                selected = path_node_sel[pair_idx][pos][snode]
                if selected not in (0, 1):
                    errors.append(f"pathNodeSel[{pair_idx}][{pos}][{snode}] must be boolean")
                if selected and s_node_present[snode] != 1:
                    errors.append(f"path for virtual pair {pair_idx} uses dummy node {snode}")

        if has_path:
            if path_nodes[pair_idx][0] != node_map[src_vnode]:
                errors.append(f"path for virtual pair {pair_idx} starts at the wrong node")
            if path_nodes[pair_idx][selected_hop_count] != node_map[dst_vnode]:
                errors.append(f"path for virtual pair {pair_idx} ends at the wrong node")

        for hop in range(selected_hop_count):
            src_snode = path_nodes[pair_idx][hop]
            dst_snode = path_nodes[pair_idx][hop + 1]
            if src_snode == dst_snode:
                errors.append(f"path for virtual pair {pair_idx} contains a self-hop")
            if not (0 <= src_snode < S_NODE_COUNT and 0 <= dst_snode < S_NODE_COUNT):
                errors.append(f"path for virtual pair {pair_idx} uses an out-of-range node")
                continue
            if s_node_present[src_snode] != 1 or s_node_present[dst_snode] != 1:
                errors.append(f"path for virtual pair {pair_idx} uses a dummy hop")
            if s_adj[src_snode][dst_snode] != 1:
                errors.append(
                    f"path for virtual pair {pair_idx} uses nonexistent substrate hop "
                    f"{src_snode}-{dst_snode}"
                )

    try:
        used_bw = path_bandwidth_use_matrix(
            v_bw=v_bw,
            path_nodes=path_nodes,
            path_len_sel=path_len_sel,
        )
        expected_cpu_after, expected_mem_after, expected_bw_after = reserve_path_resources(
            v_cpu=v_cpu,
            v_mem=v_mem,
            v_bw=v_bw,
            s_cpu_before=s_cpu_before,
            s_mem_before=s_mem_before,
            s_bw_before=s_bw_before,
            node_map=node_map,
            path_nodes=path_nodes,
            path_len_sel=path_len_sel,
        )
    except (IndexError, ValueError) as exc:
        errors.append(f"invalid path witness: {exc}")
        expected_cpu_after = list(s_cpu_before)
        expected_mem_after = list(s_mem_before)
        expected_bw_after = [list(row) for row in s_bw_before]

    for snode in range(S_NODE_COUNT):
        if s_node_present[snode] not in (0, 1):
            errors.append(f"sNodePresent[{snode}] must be boolean")
        if s_node_present[snode] == 0 and (
            s_cpu_before[snode] != 0
            or s_cpu_after[snode] != 0
            or s_mem_before[snode] != 0
            or s_mem_after[snode] != 0
        ):
            errors.append(f"dummy substrate node {snode} must have zero node resources")
        if s_node_present[snode] == 0 and s_arch[snode] != 0:
            errors.append(f"dummy substrate node {snode} must have zero architecture")
        used = s_cpu_before[snode] - expected_cpu_after[snode]
        if used > s_cpu_before[snode]:
            errors.append(f"CPU capacity exceeded on substrate node {snode}")
        if s_cpu_after[snode] != expected_cpu_after[snode]:
            errors.append(f"sCpuAfter[{snode}] is not the expected CPU after-state")
        if s_cpu_after[snode] < 0:
            errors.append(f"sCpuAfter[{snode}] is negative")
        if s_mem_after[snode] != expected_mem_after[snode]:
            errors.append(f"sMemAfter[{snode}] is not the expected memory after-state")
        if s_mem_after[snode] < 0:
            errors.append(f"sMemAfter[{snode}] is negative")
        if s_mem_before[snode] - expected_mem_after[snode] > s_mem_before[snode]:
            errors.append(f"memory capacity exceeded on substrate node {snode}")

    for i in range(S_NODE_COUNT):
        if s_adj[i][i] != 0 or s_bw_before[i][i] != 0 or s_bw_after[i][i] != 0:
            errors.append(f"substrate matrices must have zero diagonal at ({i},{i})")
        if used_bw[i][i] != 0:
            errors.append(f"used bandwidth diagonal must be zero at ({i},{i})")
        if s_node_present[i] == 0:
            for j in range(S_NODE_COUNT):
                if s_bw_before[i][j] != 0 or s_bw_before[j][i] != 0:
                    errors.append(f"dummy substrate node {i} must have zero before-bandwidth")
                if s_bw_after[i][j] != 0 or s_bw_after[j][i] != 0:
                    errors.append(f"dummy substrate node {i} must have zero after-bandwidth")
                if s_adj[i][j] != 0 or s_adj[j][i] != 0:
                    errors.append(f"dummy substrate node {i} must have zero adjacency")
        for j in range(S_NODE_COUNT):
            if s_adj[i][j] not in (0, 1):
                errors.append(f"sAdj[{i}][{j}] must be boolean")
            if s_adj[i][j] != s_adj[j][i]:
                errors.append(f"sAdj is not symmetric at ({i},{j})")
            if s_adj[i][j] == 0 and (s_bw_before[i][j] != 0 or s_bw_after[i][j] != 0):
                errors.append(f"non-edge ({i},{j}) must have zero bandwidth")
        for j in range(i + 1, S_NODE_COUNT):
            used = s_bw_before[i][j] - expected_bw_after[i][j]
            if used > s_bw_before[i][j]:
                errors.append(f"bandwidth capacity exceeded on substrate edge {i}-{j}")
            if s_bw_after[i][j] != expected_bw_after[i][j]:
                errors.append(f"sBwAfter[{i}][{j}] is not the expected bandwidth after-state")
            if s_bw_after[i][j] < 0:
                errors.append(f"sBwAfter[{i}][{j}] is negative")
            if s_bw_before[i][j] != s_bw_before[j][i]:
                errors.append(f"sBwBefore is not symmetric at ({i},{j})")
            if s_bw_after[i][j] != s_bw_after[j][i]:
                errors.append(f"sBwAfter is not symmetric at ({i},{j})")

    return not errors, errors
