"""Circuit-friendly encoding for the resource-feasibility ZK proof."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import networkx as nx

from graph_data import build_toy_substrate, build_toy_vnr


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


V_NODE_COUNT = _positive_env_int("ZK_VNE_VN_BOUND", 3)
REAL_S_NODE_COUNT = _positive_env_int("ZK_VNE_REAL_SN_COUNT", 4)
SUBSTRATE_PADDING_NUMERATOR = 3
SUBSTRATE_PADDING_DENOMINATOR = 2
_DEFAULT_S_NODE_COUNT = (
    REAL_S_NODE_COUNT * SUBSTRATE_PADDING_NUMERATOR
    + SUBSTRATE_PADDING_DENOMINATOR
    - 1
) // SUBSTRATE_PADDING_DENOMINATOR
S_NODE_COUNT = _positive_env_int("ZK_VNE_SN_BOUND", _DEFAULT_S_NODE_COUNT)
HOP_COUNT = _positive_env_int("ZK_VNE_HOP_BOUND", 3)
if S_NODE_COUNT < REAL_S_NODE_COUNT:
    raise ValueError("ZK_VNE_SN_BOUND cannot be smaller than ZK_VNE_REAL_SN_COUNT")

VIRTUAL_PAIRS = tuple(
    (source, target)
    for source in range(V_NODE_COUNT)
    for target in range(source + 1, V_NODE_COUNT)
)
PAIR_COUNT = len(VIRTUAL_PAIRS)
PATH_NODE_COUNT = HOP_COUNT + 1
PATH_LENGTH_OPTION_COUNT = HOP_COUNT + 1

REQUEST_DOMAIN = 1001
SUBSTRATE_DOMAIN = 1002
EMBEDDING_DOMAIN = 1003
SUBSTRATE_NEXT_DOMAIN = SUBSTRATE_DOMAIN

ARCH_IDS = {
    "ARM": 1,
    "x86": 2,
}

DEFAULT_REQUEST_ID = 1
DEFAULT_NODE_MAPPING = {
    "v0": "s0",
    "v1": "s3",
    "v2": "s1",
}
DEFAULT_PATHS = {
    ("v0", "v1"): ["s0", "s2", "s3"],
    ("v1", "v2"): ["s3", "s1"],
}
DEFAULT_RANDOMNESS = {
    "rho_S": 1111,
    "rho_M": 2222,
    "rho_S_next": 3333,
}

V_NODE_IDS = {f"v{index}": index for index in range(V_NODE_COUNT)}
S_NODE_IDS = {f"s{index}": index for index in range(REAL_S_NODE_COUNT)}

ROOT = Path(__file__).resolve().parent
POSEIDON_HELPER = ROOT / "scripts" / "poseidon_chain.js"


class PoseidonCommitter:
    """Persistent JSONL client for one initialized circomlibjs Poseidon process."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            ["node", str(POSEIDON_HELPER), "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=ROOT,
        )

    def commit(self, requests: list[tuple[int, list[int]]]) -> list[int]:
        self.start()
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        payload = _poseidon_request_payload(requests)
        self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        response = self._process.stdout.readline()
        if not response:
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read().strip()
            raise RuntimeError(f"persistent Poseidon helper exited unexpectedly: {stderr}")
        values = json.loads(response)
        if not isinstance(values, list) or len(values) != len(requests):
            raise RuntimeError("persistent Poseidon helper returned an invalid response")
        return [int(value) for value in values]

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    def __enter__(self) -> "PoseidonCommitter":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


_GLOBAL_POSEIDON_COMMITTER: PoseidonCommitter | None = None


def _global_poseidon_committer() -> PoseidonCommitter:
    global _GLOBAL_POSEIDON_COMMITTER
    if _GLOBAL_POSEIDON_COMMITTER is None:
        _GLOBAL_POSEIDON_COMMITTER = PoseidonCommitter()
    return _GLOBAL_POSEIDON_COMMITTER


def close_poseidon_committer() -> None:
    global _GLOBAL_POSEIDON_COMMITTER
    if _GLOBAL_POSEIDON_COMMITTER is not None:
        _GLOBAL_POSEIDON_COMMITTER.close()
        _GLOBAL_POSEIDON_COMMITTER = None


atexit.register(close_poseidon_committer)


def _ordered_v_nodes(vnr: nx.Graph) -> list[str]:
    nodes = sorted(vnr.nodes, key=lambda node: V_NODE_IDS[node])
    if len(nodes) != V_NODE_COUNT:
        raise ValueError(f"expected {V_NODE_COUNT} virtual nodes, got {len(nodes)}")
    return nodes


def _ordered_s_nodes(substrate: nx.Graph) -> list[str]:
    nodes = sorted(substrate.nodes, key=lambda node: S_NODE_IDS[node])
    if len(nodes) != REAL_S_NODE_COUNT:
        raise ValueError(f"expected {REAL_S_NODE_COUNT} real substrate nodes, got {len(nodes)}")
    return nodes


def _zero_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def _one_hot(index: int, size: int) -> list[int]:
    if index < 0 or index >= size:
        raise ValueError(f"index {index} is out of range for one-hot size {size}")
    return [1 if i == index else 0 for i in range(size)]


def _flatten(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def _arch_id(value: str | int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    try:
        return ARCH_IDS[value]
    except KeyError as exc:
        raise ValueError(f"unsupported architecture: {value}") from exc


def vnr_bandwidth_matrix(vnr: nx.Graph) -> list[list[int]]:
    matrix = _zero_matrix(V_NODE_COUNT)
    for u, v, attrs in vnr.edges(data=True):
        u_idx = V_NODE_IDS[u]
        v_idx = V_NODE_IDS[v]
        bandwidth = int(attrs.get("bandwidth", 0))
        matrix[u_idx][v_idx] = bandwidth
        matrix[v_idx][u_idx] = bandwidth
    return matrix


def vnr_link_presence_matrix(vnr: nx.Graph) -> list[list[int]]:
    matrix = _zero_matrix(V_NODE_COUNT)
    for u, v in vnr.edges:
        u_idx = V_NODE_IDS[u]
        v_idx = V_NODE_IDS[v]
        matrix[u_idx][v_idx] = 1
        matrix[v_idx][u_idx] = 1
    return matrix


def substrate_bandwidth_matrix(substrate: nx.Graph) -> list[list[int]]:
    matrix = _zero_matrix(S_NODE_COUNT)
    for u, v, attrs in substrate.edges(data=True):
        u_idx = S_NODE_IDS[u]
        v_idx = S_NODE_IDS[v]
        bandwidth = attrs["bandwidth"]
        matrix[u_idx][v_idx] = bandwidth
        matrix[v_idx][u_idx] = bandwidth
    return matrix


def substrate_adjacency_matrix(substrate: nx.Graph) -> list[list[int]]:
    matrix = _zero_matrix(S_NODE_COUNT)
    for u, v in substrate.edges:
        u_idx = S_NODE_IDS[u]
        v_idx = S_NODE_IDS[v]
        matrix[u_idx][v_idx] = 1
        matrix[v_idx][u_idx] = 1
    return matrix


def vnr_node_presence_vector(vnr: nx.Graph) -> list[int]:
    return [int(bool(vnr.nodes[node].get("present", 1))) for node in _ordered_v_nodes(vnr)]


def substrate_cpu_vector(substrate: nx.Graph) -> list[int]:
    s_nodes = _ordered_s_nodes(substrate)
    cpu = [substrate.nodes[node]["cpu"] for node in s_nodes]
    cpu += [0 for _ in range(S_NODE_COUNT - REAL_S_NODE_COUNT)]
    return cpu


def vnr_memory_vector(vnr: nx.Graph) -> list[int]:
    return [int(vnr.nodes[node].get("memory_gib", 0)) for node in _ordered_v_nodes(vnr)]


def substrate_memory_vector(substrate: nx.Graph) -> list[int]:
    s_nodes = _ordered_s_nodes(substrate)
    memory = [int(substrate.nodes[node].get("memory_gib", 0)) for node in s_nodes]
    memory += [0 for _ in range(S_NODE_COUNT - REAL_S_NODE_COUNT)]
    return memory


def vnr_arch_vector(vnr: nx.Graph) -> list[int]:
    v_nodes = _ordered_v_nodes(vnr)
    return [_arch_id(vnr.nodes[node].get("arch")) for node in v_nodes]


def substrate_arch_vector(substrate: nx.Graph) -> list[int]:
    s_nodes = _ordered_s_nodes(substrate)
    arch = [_arch_id(substrate.nodes[node].get("arch")) for node in s_nodes]
    arch += [0 for _ in range(S_NODE_COUNT - REAL_S_NODE_COUNT)]
    return arch


def substrate_node_presence() -> list[int]:
    """Mark real substrate slots and dummy padding slots."""
    return [1 if idx < REAL_S_NODE_COUNT else 0 for idx in range(S_NODE_COUNT)]


def substrate_state_data(substrate: nx.Graph, *, rho_S: int) -> dict[str, Any]:
    """Build the substrate-state object committed by C_S."""
    return {
        "sNodePresent": substrate_node_presence(),
        "sCpuBefore": substrate_cpu_vector(substrate),
        "sMemBefore": substrate_memory_vector(substrate),
        "sArch": substrate_arch_vector(substrate),
        "sAdj": substrate_adjacency_matrix(substrate),
        "sBwBefore": substrate_bandwidth_matrix(substrate),
        "rho_S": rho_S,
    }


def substrate_state_commitment(substrate: nx.Graph, *, rho_S: int) -> int:
    """Commit to a substrate state without constructing a full request witness."""
    return poseidon_chain_commit(
        SUBSTRATE_DOMAIN,
        substrate_fields(substrate_state_data(substrate, rho_S=rho_S)),
    )


def request_commitment(vnr: nx.Graph, *, req_id: int) -> int:
    """Commit to a public VNR request without constructing a full witness."""
    v_nodes = _ordered_v_nodes(vnr)
    data = {
        "reqId": req_id,
        "vNodePresent": vnr_node_presence_vector(vnr),
        "vCpu": [vnr.nodes[node]["cpu"] for node in v_nodes],
        "vMem": vnr_memory_vector(vnr),
        "vArch": vnr_arch_vector(vnr),
        "vLinkPresent": vnr_link_presence_matrix(vnr),
        "vBw": vnr_bandwidth_matrix(vnr),
        "endpointEnabled": 0,
        "endpointIdx": 0,
    }
    return poseidon_chain_commit(REQUEST_DOMAIN, request_fields(data))


def apply_after_state_to_substrate(substrate: nx.Graph, data: dict[str, Any]) -> nx.Graph:
    """Return a new substrate graph whose resources equal the proved after-state."""
    updated = substrate.copy()
    s_by_index = {index: node for node, index in S_NODE_IDS.items()}

    for index in range(REAL_S_NODE_COUNT):
        updated.nodes[s_by_index[index]]["cpu"] = data["sCpuAfter"][index]
        updated.nodes[s_by_index[index]]["memory_gib"] = data["sMemAfter"][index]

    for i in range(REAL_S_NODE_COUNT):
        for j in range(i + 1, REAL_S_NODE_COUNT):
            u = s_by_index[i]
            v = s_by_index[j]
            bandwidth = data["sBwAfter"][i][j]
            if data["sAdj"][i][j] == 1:
                updated.add_edge(u, v, bandwidth=bandwidth)
            elif updated.has_edge(u, v):
                updated.remove_edge(u, v)

    return updated


def _path_length_from_selector(selector: list[int]) -> int | None:
    if len(selector) != PATH_LENGTH_OPTION_COUNT:
        raise ValueError(
            f"expected path length selector of size {PATH_LENGTH_OPTION_COUNT}"
        )
    selected = [idx for idx, value in enumerate(selector) if value == 1]
    if len(selected) > 1:
        raise ValueError("path length selector must choose at most one length")
    return selected[0] if selected else None


def build_path_witness(
    *,
    v_link_present: list[list[int]],
    v_bw: list[list[int]],
    paths: dict[tuple[str, str], list[str]],
) -> tuple[list[list[int]], list[list[list[int]]], list[list[int]]]:
    """Build fixed-size path arrays for all upper-triangular virtual pairs."""
    path_nodes = [[0 for _ in range(PATH_NODE_COUNT)] for _ in range(PAIR_COUNT)]
    path_node_sel = [
        [[0 for _ in range(S_NODE_COUNT)] for _ in range(PATH_NODE_COUNT)]
        for _ in range(PAIR_COUNT)
    ]
    path_len_sel = [
        [0 for _ in range(PATH_LENGTH_OPTION_COUNT)] for _ in range(PAIR_COUNT)
    ]

    for pair_idx, (u, v) in enumerate(VIRTUAL_PAIRS):
        demand = v_bw[u][v]
        link_present = v_link_present[u][v]
        pair_key = (f"v{u}", f"v{v}")
        path = paths.get(pair_key, [])

        if link_present == 0:
            if path:
                raise ValueError(f"absent virtual link {pair_key} must not provide a path")
            if demand != 0:
                raise ValueError(f"absent virtual link {pair_key} cannot request bandwidth")
            continue

        hop_count = len(path) - 1
        if hop_count < 0 or hop_count > HOP_COUNT:
            raise ValueError(f"path for {pair_key} must contain 0..{HOP_COUNT} hops")

        path_len_sel[pair_idx][hop_count] = 1
        for pos, node in enumerate(path):
            snode_idx = S_NODE_IDS[node]
            path_nodes[pair_idx][pos] = snode_idx
            path_node_sel[pair_idx][pos] = _one_hot(snode_idx, S_NODE_COUNT)

    return path_nodes, path_node_sel, path_len_sel


def path_bandwidth_use_matrix(
    *,
    v_bw: list[list[int]],
    path_nodes: list[list[int]],
    path_len_sel: list[list[int]],
) -> list[list[int]]:
    """Compute symmetric substrate bandwidth use from bounded path witnesses."""
    used_bw = _zero_matrix(S_NODE_COUNT)

    for pair_idx, (u, v) in enumerate(VIRTUAL_PAIRS):
        demand = v_bw[u][v]
        hop_count = _path_length_from_selector(path_len_sel[pair_idx])
        if demand == 0 or hop_count is None:
            continue
        for hop in range(hop_count):
            s = path_nodes[pair_idx][hop]
            t = path_nodes[pair_idx][hop + 1]
            used_bw[s][t] += demand
            used_bw[t][s] += demand

    return used_bw


def reserve_path_resources(
    *,
    v_cpu: list[int],
    v_mem: list[int],
    v_bw: list[list[int]],
    s_cpu_before: list[int],
    s_mem_before: list[int],
    s_bw_before: list[list[int]],
    node_map: list[int],
    path_nodes: list[list[int]],
    path_len_sel: list[list[int]],
) -> tuple[list[int], list[int], list[list[int]]]:
    """Reserve node and link resources for a bounded unsplittable path embedding."""
    s_cpu_after = list(s_cpu_before)
    s_mem_after = list(s_mem_before)
    s_bw_after = deepcopy(s_bw_before)

    for vnode, snode in enumerate(node_map):
        s_cpu_after[snode] -= v_cpu[vnode]
        s_mem_after[snode] -= v_mem[vnode]

    used_bw = path_bandwidth_use_matrix(
        v_bw=v_bw,
        path_nodes=path_nodes,
        path_len_sel=path_len_sel,
    )
    for s in range(S_NODE_COUNT):
        for t in range(S_NODE_COUNT):
            s_bw_after[s][t] -= used_bw[s][t]

    return s_cpu_after, s_mem_after, s_bw_after


def poseidon_chain_commit(domain: int, fields: list[int]) -> int:
    """Compute the same chained Poseidon(2) hash used by the Circom circuit."""
    return poseidon_chain_commits([(domain, fields)])[0]


def _poseidon_request_payload(requests: list[tuple[int, list[int]]]) -> dict[str, Any]:
    return {
        "requests": [
            {
                "domain": str(domain),
                "fields": [str(field) for field in fields],
            }
            for domain, fields in requests
        ]
    }


def poseidon_chain_commits(
    requests: list[tuple[int, list[int]]],
    *,
    committer: PoseidonCommitter | None = None,
    persistent: bool | None = None,
) -> list[int]:
    """Compute commitments in a persistent or one-shot circomlibjs process."""
    if committer is not None:
        return committer.commit(requests)

    if persistent is None:
        mode = os.environ.get("ZK_VNE_POSEIDON_MODE", "persistent")
        if mode not in {"persistent", "cold"}:
            raise ValueError("ZK_VNE_POSEIDON_MODE must be 'persistent' or 'cold'")
        persistent = mode == "persistent"
    if persistent:
        return _global_poseidon_committer().commit(requests)

    request = _poseidon_request_payload(requests)
    result = subprocess.run(
        ["node", str(POSEIDON_HELPER)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    return [int(value) for value in json.loads(result.stdout)]


def request_fields(data: dict[str, Any]) -> list[int]:
    return (
        [data["reqId"]]
        + data["vNodePresent"]
        + data["vCpu"]
        + data["vMem"]
        + data["vArch"]
        + _flatten(data["vLinkPresent"])
        + _flatten(data["vBw"])
        + [data["endpointEnabled"], data["endpointIdx"]]
    )


def substrate_fields(data: dict[str, Any], *, next_state: bool = False) -> list[int]:
    cpu_key = "sCpuAfter" if next_state else "sCpuBefore"
    mem_key = "sMemAfter" if next_state else "sMemBefore"
    bw_key = "sBwAfter" if next_state else "sBwBefore"
    rho_key = "rho_S_next" if next_state else "rho_S"
    return (
        data["sNodePresent"]
        + data[cpu_key]
        + data[mem_key]
        + data["sArch"]
        + _flatten(data["sAdj"])
        + _flatten(data[bw_key])
        + [data[rho_key]]
    )


def embedding_fields(data: dict[str, Any]) -> list[int]:
    return (
        data["nodeMap"]
        + _flatten(data["pathLenSel"])
        + _flatten(data["pathNodes"])
        + [data["rho_M"]]
    )


def compute_poseidon_commitments(
    data: dict[str, Any],
    *,
    committer: PoseidonCommitter | None = None,
    persistent: bool | None = None,
) -> dict[str, int]:
    values = poseidon_chain_commits(
        [
            (REQUEST_DOMAIN, request_fields(data)),
            (SUBSTRATE_DOMAIN, substrate_fields(data)),
            (EMBEDDING_DOMAIN, embedding_fields(data)),
            (SUBSTRATE_NEXT_DOMAIN, substrate_fields(data, next_state=True)),
        ],
        committer=committer,
        persistent=persistent,
    )
    return dict(zip(("C_r", "C_S", "C_M", "C_S_next"), values, strict=True))


def build_zk_input(
    vnr: nx.Graph,
    substrate: nx.Graph,
    *,
    node_mapping: dict[str, str] | None = None,
    paths: dict[tuple[str, str], list[str]] | None = None,
    req_id: int = DEFAULT_REQUEST_ID,
    randomness: dict[str, int] | None = None,
    endpoint_idx: int | None = None,
    compute_commitments: bool = True,
    poseidon_committer: PoseidonCommitter | None = None,
) -> dict[str, Any]:
    """Build the Circom input object for the resource-feasibility relation."""
    node_mapping = DEFAULT_NODE_MAPPING if node_mapping is None else node_mapping
    paths = DEFAULT_PATHS if paths is None else paths
    randomness = DEFAULT_RANDOMNESS if randomness is None else randomness

    v_nodes = _ordered_v_nodes(vnr)

    v_node_present = vnr_node_presence_vector(vnr)
    v_cpu = [vnr.nodes[node]["cpu"] for node in v_nodes]
    v_mem = vnr_memory_vector(vnr)
    v_arch = vnr_arch_vector(vnr)
    v_link_present = vnr_link_presence_matrix(vnr)
    v_bw = vnr_bandwidth_matrix(vnr)
    s_node_present = substrate_node_presence()
    s_cpu_before = substrate_cpu_vector(substrate)
    s_mem_before = substrate_memory_vector(substrate)
    s_arch = substrate_arch_vector(substrate)
    s_adj = substrate_adjacency_matrix(substrate)
    s_bw_before = substrate_bandwidth_matrix(substrate)
    endpoint_enabled = 1 if endpoint_idx is not None else 0
    endpoint_idx_value = endpoint_idx if endpoint_idx is not None else 0

    node_map = [
        S_NODE_IDS[node_mapping[vnode]] if v_node_present[index] else 0
        for index, vnode in enumerate(v_nodes)
    ]
    node_sel = [
        _one_hot(snode, S_NODE_COUNT)
        if v_node_present[vnode_index]
        else [0] * S_NODE_COUNT
        for vnode_index, snode in enumerate(node_map)
    ]
    path_nodes, path_node_sel, path_len_sel = build_path_witness(
        v_link_present=v_link_present,
        v_bw=v_bw,
        paths=paths,
    )
    s_cpu_after, s_mem_after, s_bw_after = reserve_path_resources(
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

    data: dict[str, Any] = {
        "reqId": req_id,
        "vNodePresent": v_node_present,
        "vCpu": v_cpu,
        "vMem": v_mem,
        "vArch": v_arch,
        "vLinkPresent": v_link_present,
        "vBw": v_bw,
        "endpointEnabled": endpoint_enabled,
        "endpointIdx": endpoint_idx_value,
        "sNodePresent": s_node_present,
        "sCpuBefore": s_cpu_before,
        "sCpuAfter": s_cpu_after,
        "sMemBefore": s_mem_before,
        "sMemAfter": s_mem_after,
        "sArch": s_arch,
        "sAdj": s_adj,
        "sBwBefore": s_bw_before,
        "sBwAfter": s_bw_after,
        "nodeMap": node_map,
        "nodeSel": node_sel,
        "pathNodes": path_nodes,
        "pathNodeSel": path_node_sel,
        "pathLenSel": path_len_sel,
        "rho_S": randomness["rho_S"],
        "rho_M": randomness["rho_M"],
        "rho_S_next": randomness["rho_S_next"],
    }

    commitments = (
        compute_poseidon_commitments(data, committer=poseidon_committer)
        if compute_commitments
        else {"C_r": 0, "C_S": 0, "C_M": 0, "C_S_next": 0}
    )
    data.update(commitments)
    return data


def stringify_field_values(value: Any) -> Any:
    """Convert ints recursively to decimal strings for snarkjs input JSON."""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return [stringify_field_values(item) for item in value]
    if isinstance(value, dict):
        return {key: stringify_field_values(item) for key, item in value.items()}
    return value


def write_zk_input(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stringify_field_values(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_default_input(output_path: Path) -> None:
    zk_input = build_zk_input(build_toy_vnr(), build_toy_substrate())
    write_zk_input(zk_input, output_path)

    print(f"Resource-feasibility input written to {output_path}")
    print("Public commitments")
    print(f"  C_r: {zk_input['C_r']}")
    print(f"  C_S: {zk_input['C_S']}")
    print(f"  C_M: {zk_input['C_M']}")
    print(f"  C_S_next: {zk_input['C_S_next']}")


if __name__ == "__main__":
    export_default_input(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/zk/input.json"))
