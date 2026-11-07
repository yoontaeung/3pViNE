"""Local CP-SAT solver for bounded multi-hop VNE witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
from ortools.sat.python import cp_model

from zk_encoding import HOP_COUNT, REAL_S_NODE_COUNT, S_NODE_IDS, VIRTUAL_PAIRS, V_NODE_IDS


SolverObjective = Literal["min_hops", "prefer_multihop_then_min_hops"]


@dataclass(frozen=True)
class SolverPolicy:
    """Private facility policy for choosing one feasible embedding."""

    objective: SolverObjective = "min_hops"
    max_seconds: float = 5.0


@dataclass(frozen=True)
class LocalEmbedding:
    """Embedding witness produced by the local solver."""

    node_mapping: dict[str, str]
    paths: dict[tuple[str, str], list[str]]


@dataclass(frozen=True)
class PathCandidate:
    nodes: tuple[str, ...]
    start: str
    end: str
    hop_count: int
    edges: tuple[tuple[str, str], ...]


def _arch_compatible(vnr: nx.Graph, substrate: nx.Graph, vnode: str, snode: str) -> bool:
    required_arch = vnr.nodes[vnode].get("arch")
    if required_arch in (None, 0):
        return True
    return substrate.nodes[snode].get("arch") == required_arch


def _endpoint_compatible(vnr: nx.Graph, substrate: nx.Graph, vnode: str, snode: str) -> bool:
    vnode_is_endpoint = bool(vnr.nodes[vnode].get("public_endpoint"))
    snode_is_endpoint = bool(substrate.nodes[snode].get("public_endpoint"))
    return vnode_is_endpoint == snode_is_endpoint


def _ordered_v_nodes(vnr: nx.Graph) -> list[str]:
    return sorted(vnr.nodes, key=lambda node: V_NODE_IDS[node])


def _ordered_real_s_nodes(substrate: nx.Graph) -> list[str]:
    nodes = sorted(substrate.nodes, key=lambda node: S_NODE_IDS[node])
    if len(nodes) != REAL_S_NODE_COUNT:
        raise ValueError(f"expected {REAL_S_NODE_COUNT} substrate nodes, got {len(nodes)}")
    return nodes


def _edge_key(u: str, v: str) -> tuple[str, str]:
    return tuple(sorted((u, v), key=lambda node: S_NODE_IDS[node]))


def _active_virtual_pairs(vnr: nx.Graph) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    for src_idx, dst_idx in VIRTUAL_PAIRS:
        src = f"v{src_idx}"
        dst = f"v{dst_idx}"
        if vnr.has_edge(src, dst):
            demand = int(vnr[src][dst].get("bandwidth", 0))
            pairs.append((src_idx, dst_idx, demand))
    return pairs


def _candidate_paths(
    substrate: nx.Graph,
    *,
    demand: int,
    max_hops: int,
) -> list[PathCandidate]:
    candidates: list[PathCandidate] = []
    real_nodes = _ordered_real_s_nodes(substrate)

    for start in real_nodes:
        for end in real_nodes:
            if start == end:
                candidates.append(
                    PathCandidate(
                        nodes=(start,),
                        start=start,
                        end=end,
                        hop_count=0,
                        edges=(),
                    )
                )
                continue
            try:
                paths = nx.all_simple_paths(substrate, start, end, cutoff=max_hops)
            except nx.NetworkXNoPath:
                continue
            for path in paths:
                hop_count = len(path) - 1
                if hop_count < 1 or hop_count > max_hops:
                    continue

                edge_keys: list[tuple[str, str]] = []
                feasible = True
                for u, v in zip(path, path[1:]):
                    bandwidth = substrate[u][v].get("bandwidth", 0)
                    if bandwidth < demand:
                        feasible = False
                        break
                    edge_keys.append(_edge_key(u, v))
                if feasible:
                    candidates.append(
                        PathCandidate(
                            nodes=tuple(path),
                            start=start,
                            end=end,
                            hop_count=hop_count,
                            edges=tuple(edge_keys),
                        )
                    )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.hop_count,
            [S_NODE_IDS[node] for node in candidate.nodes],
        ),
    )


def solve_local_embedding(
    vnr: nx.Graph,
    substrate: nx.Graph,
    *,
    policy: SolverPolicy | None = None,
    fixed_node_mapping: dict[str, str] | None = None,
) -> LocalEmbedding | None:
    """Find one bounded, unsplittable multi-hop embedding or return ``None``."""
    policy = policy or SolverPolicy()
    fixed_node_mapping = fixed_node_mapping or {}
    v_nodes = _ordered_v_nodes(vnr)
    s_nodes = _ordered_real_s_nodes(substrate)
    active_pairs = _active_virtual_pairs(vnr)
    v_node_present = [int(bool(vnr.nodes[node].get("present", 1))) for node in v_nodes]

    model = cp_model.CpModel()
    assign = {
        (v_idx, s_idx): model.NewBoolVar(f"assign_v{v_idx}_s{s_idx}")
        for v_idx in range(len(v_nodes))
        for s_idx in range(len(s_nodes))
    }

    for v_idx in range(len(v_nodes)):
        model.Add(
            sum(assign[v_idx, s_idx] for s_idx in range(len(s_nodes)))
            == v_node_present[v_idx]
        )
        v_node = v_nodes[v_idx]
        for s_idx, s_node in enumerate(s_nodes):
            if not _arch_compatible(vnr, substrate, v_node, s_node):
                model.Add(assign[v_idx, s_idx] == 0)
            if not _endpoint_compatible(vnr, substrate, v_node, s_node):
                model.Add(assign[v_idx, s_idx] == 0)

        if v_node in fixed_node_mapping:
            fixed_snode = fixed_node_mapping[v_node]
            if fixed_snode not in S_NODE_IDS:
                raise ValueError(f"unknown fixed substrate node: {fixed_snode}")
            if fixed_snode not in substrate:
                raise ValueError(f"fixed substrate node is absent from substrate graph: {fixed_snode}")
            if not _arch_compatible(vnr, substrate, v_node, fixed_snode):
                return None
            if not _endpoint_compatible(vnr, substrate, v_node, fixed_snode):
                return None
            model.Add(assign[v_idx, S_NODE_IDS[fixed_snode]] == 1)

    for s_idx, s_node in enumerate(s_nodes):
        model.Add(
            sum(
                int(vnr.nodes[v_node]["cpu"]) * assign[v_idx, s_idx]
                for v_idx, v_node in enumerate(v_nodes)
            )
            <= int(substrate.nodes[s_node]["cpu"])
        )
        model.Add(
            sum(
                int(vnr.nodes[v_node].get("memory_gib", 0)) * assign[v_idx, s_idx]
                for v_idx, v_node in enumerate(v_nodes)
            )
            <= int(substrate.nodes[s_node].get("memory_gib", 0))
        )

    pair_candidates: dict[int, list[PathCandidate]] = {}
    path_choice: dict[tuple[int, int], cp_model.IntVar] = {}
    for pair_order, (src_idx, dst_idx, demand) in enumerate(active_pairs):
        candidates = _candidate_paths(substrate, demand=demand, max_hops=HOP_COUNT)
        if not candidates:
            return None
        pair_candidates[pair_order] = candidates
        for cand_idx, candidate in enumerate(candidates):
            var = model.NewBoolVar(f"path_p{pair_order}_c{cand_idx}")
            path_choice[pair_order, cand_idx] = var
            model.AddImplication(var, assign[src_idx, S_NODE_IDS[candidate.start]])
            model.AddImplication(var, assign[dst_idx, S_NODE_IDS[candidate.end]])
        model.Add(sum(path_choice[pair_order, idx] for idx in range(len(candidates))) == 1)

    for u_idx, u_node in enumerate(s_nodes):
        for v_idx in range(u_idx + 1, len(s_nodes)):
            v_node = s_nodes[v_idx]
            bandwidth = substrate[u_node][v_node]["bandwidth"] if substrate.has_edge(u_node, v_node) else 0
            edge = _edge_key(u_node, v_node)
            terms = []
            for pair_order, (_, _, demand) in enumerate(active_pairs):
                for cand_idx, candidate in enumerate(pair_candidates[pair_order]):
                    if edge in candidate.edges:
                        terms.append(demand * path_choice[pair_order, cand_idx])
            model.Add(sum(terms) <= int(bandwidth))

    objective_terms = []
    for pair_order, candidates in pair_candidates.items():
        for cand_idx, candidate in enumerate(candidates):
            choice = path_choice[pair_order, cand_idx]
            if policy.objective == "prefer_multihop_then_min_hops":
                direct_penalty = 100_000 if candidate.hop_count <= 1 else 0
                objective_terms.append((direct_penalty + candidate.hop_count * 100 + cand_idx) * choice)
            elif policy.objective == "min_hops":
                objective_terms.append((candidate.hop_count * 100 + cand_idx) * choice)
            else:
                raise ValueError(f"unsupported solver objective: {policy.objective}")
    if objective_terms:
        model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = policy.max_seconds
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    node_mapping = {}
    for v_idx, v_node in enumerate(v_nodes):
        if not v_node_present[v_idx]:
            continue
        for s_idx, s_node in enumerate(s_nodes):
            if solver.BooleanValue(assign[v_idx, s_idx]):
                node_mapping[v_node] = s_node
                break

    paths: dict[tuple[str, str], list[str]] = {}
    for pair_order, (src_idx, dst_idx, _) in enumerate(active_pairs):
        for cand_idx, candidate in enumerate(pair_candidates[pair_order]):
            if solver.BooleanValue(path_choice[pair_order, cand_idx]):
                paths[(f"v{src_idx}", f"v{dst_idx}")] = list(candidate.nodes)
                break

    return LocalEmbedding(node_mapping=node_mapping, paths=paths)
