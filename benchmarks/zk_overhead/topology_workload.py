"""Fixed-VNR replay inputs over different private substrate topologies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from benchmarks.zk_overhead.profiles import CircuitProfile
from benchmarks.zk_overhead.replay import build_replay_fields, zero_matrix


TOPOLOGY_NAMES = (
    "full_mesh",
    "ring",
    "waxman",
    "erdos_renyi",
    "barabasi_albert",
)
WAXMAN_ALPHA = 0.4
WAXMAN_BETA = 0.4
WAXMAN_SEED = 0
WAXMAN_MAX_SEED_ATTEMPTS = 1000
ERDOS_RENYI_P = 0.16
ERDOS_RENYI_SEED = 0
ERDOS_RENYI_MAX_SEED_ATTEMPTS = 1000
BARABASI_ALBERT_M = 2
BARABASI_ALBERT_SEED = 0


@dataclass(frozen=True)
class TopologyMetadata:
    name: str
    node_count: int
    edge_count: int
    density: float
    generator: dict[str, int | float | str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "generator": self.generator,
        }


def _relabel_backbone(graph: nx.Graph) -> nx.Graph:
    """Relabel one two-edge path as 0-1-2 without changing graph structure."""
    center = next((node for node, degree in graph.degree if degree >= 2), None)
    if center is None:
        raise ValueError("topology has no two-edge path for the fixed VNR")
    left, right = sorted(graph.neighbors(center))[:2]
    remaining = sorted(set(graph.nodes) - {left, center, right})
    ordered = [left, center, right, *remaining]
    return nx.relabel_nodes(graph, {node: index for index, node in enumerate(ordered)})


def build_substrate_graph(name: str, sn: int) -> tuple[nx.Graph, TopologyMetadata]:
    """Build a deterministic connected topology containing the 0-1-2 backbone."""
    if name not in TOPOLOGY_NAMES:
        raise ValueError(f"unknown topology {name!r}; choose from {TOPOLOGY_NAMES}")
    if sn < 3:
        raise ValueError("topology experiment requires at least three substrate nodes")

    if name == "full_mesh":
        graph = nx.complete_graph(sn)
        generator: dict[str, int | float | str] = {"type": "complete_graph"}
    elif name == "ring":
        graph = nx.cycle_graph(sn)
        generator = {"type": "cycle_graph"}
    elif name == "waxman":
        selected_seed = None
        for seed in range(WAXMAN_SEED, WAXMAN_SEED + WAXMAN_MAX_SEED_ATTEMPTS):
            candidate = nx.waxman_graph(
                sn,
                alpha=WAXMAN_ALPHA,
                beta=WAXMAN_BETA,
                seed=seed,
            )
            if nx.is_connected(candidate):
                graph = candidate
                selected_seed = seed
                break
        if selected_seed is None:
            raise RuntimeError(
                "Waxman parameters did not produce a connected graph within "
                f"{WAXMAN_MAX_SEED_ATTEMPTS} deterministic seeds"
            )
        graph = _relabel_backbone(graph)
        generator = {
            "type": "waxman_graph",
            "alpha": WAXMAN_ALPHA,
            "beta": WAXMAN_BETA,
            "seed": selected_seed,
        }
    elif name == "erdos_renyi":
        selected_seed = None
        for seed in range(
            ERDOS_RENYI_SEED,
            ERDOS_RENYI_SEED + ERDOS_RENYI_MAX_SEED_ATTEMPTS,
        ):
            candidate = nx.gnp_random_graph(sn, ERDOS_RENYI_P, seed=seed)
            if nx.is_connected(candidate):
                graph = candidate
                selected_seed = seed
                break
        if selected_seed is None:
            raise RuntimeError(
                "Erdos-Renyi parameters did not produce a connected graph within "
                f"{ERDOS_RENYI_MAX_SEED_ATTEMPTS} deterministic seeds"
            )
        graph = _relabel_backbone(graph)
        generator = {
            "type": "gnp_random_graph",
            "p": ERDOS_RENYI_P,
            "seed": selected_seed,
        }
    else:
        graph = nx.barabasi_albert_graph(
            sn,
            BARABASI_ALBERT_M,
            seed=BARABASI_ALBERT_SEED,
        )
        graph = _relabel_backbone(graph)
        generator = {
            "type": "barabasi_albert_graph",
            "m": BARABASI_ALBERT_M,
            "seed": BARABASI_ALBERT_SEED,
        }

    if not graph.has_edge(0, 1) or not graph.has_edge(1, 2):
        raise RuntimeError(f"{name} does not contain the required 0-1-2 backbone")
    metadata = TopologyMetadata(
        name=name,
        node_count=sn,
        edge_count=graph.number_of_edges(),
        density=nx.density(graph),
        generator=generator,
    )
    return graph, metadata


def build_topology_replay_fields(
    profile: CircuitProfile,
    topology: str,
    *,
    request_id: int = 1,
) -> tuple[dict[str, Any], TopologyMetadata]:
    """Embed one fixed three-node chain over a selected substrate topology."""
    if profile.vn != 3:
        raise ValueError("topology experiment uses the fixed three-node chain VNR")

    data = build_replay_fields(profile, request_id=request_id)
    graph, metadata = build_substrate_graph(topology, profile.sn)

    s_adj = zero_matrix(profile.sn)
    s_bw_before = zero_matrix(profile.sn)
    for source, target in graph.edges:
        s_adj[source][target] = s_adj[target][source] = 1
        s_bw_before[source][target] = s_bw_before[target][source] = 100

    s_bw_after = [list(row) for row in s_bw_before]
    for source, target in ((0, 1), (1, 2)):
        s_bw_after[source][target] -= 1
        s_bw_after[target][source] -= 1

    data["sAdj"] = s_adj
    data["sBwBefore"] = s_bw_before
    data["sBwAfter"] = s_bw_after
    validate_topology_replay(profile, data, metadata)
    return data, metadata


def validate_topology_replay(
    profile: CircuitProfile,
    data: dict[str, Any],
    metadata: TopologyMetadata,
) -> None:
    """Check the topology-specific invariants before invoking Circom."""
    if metadata.node_count != profile.sn:
        raise ValueError("topology node count does not match the circuit profile")
    if data["nodeMap"] != [0, 1, 2]:
        raise ValueError("topology experiment must use the fixed node mapping")
    if data["sAdj"][0][1] != 1 or data["sAdj"][1][2] != 1:
        raise ValueError("fixed VNR path is absent from the substrate topology")

    edge_count = sum(sum(row) for row in data["sAdj"]) // 2
    if edge_count != metadata.edge_count:
        raise ValueError("adjacency matrix edge count does not match topology metadata")

    for source in range(profile.sn):
        for target in range(profile.sn):
            adjacent = data["sAdj"][source][target]
            before = data["sBwBefore"][source][target]
            after = data["sBwAfter"][source][target]
            if adjacent not in (0, 1):
                raise ValueError("substrate adjacency entries must be boolean")
            if adjacent == 0 and (before != 0 or after != 0):
                raise ValueError("non-edges must have zero bandwidth")
            expected_after = before
            if (source, target) in ((0, 1), (1, 0), (1, 2), (2, 1)):
                expected_after -= 1
            if after != expected_after:
                raise ValueError("substrate bandwidth after-state is inconsistent")
