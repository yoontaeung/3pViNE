"""Tests for the local VNE witness solver."""

from __future__ import annotations

import unittest

from graph_data import build_toy_substrate, build_toy_vnr
from local_solver import SolverPolicy, solve_local_embedding
from verifier import verify_resource_feasibility
from zk_encoding import build_zk_input


class LocalSolverTests(unittest.TestCase):
    def test_solver_finds_valid_multi_hop_embedding_for_first_request(self) -> None:
        vnr = build_toy_vnr()
        substrate = build_toy_substrate()
        policy = SolverPolicy(objective="prefer_multihop_then_min_hops")

        embedding = solve_local_embedding(
            vnr,
            substrate,
            policy=policy,
        )

        self.assertIsNotNone(embedding)
        assert embedding is not None
        self.assertTrue(any(len(path) > 2 for path in embedding.paths.values()))

        data = build_zk_input(
            vnr,
            substrate,
            node_mapping=embedding.node_mapping,
            paths=embedding.paths,
            req_id=1,
            randomness={"rho_S": 1111, "rho_M": 2222, "rho_S_next": 3333},
            compute_commitments=False,
        )
        accepted, errors = verify_resource_feasibility(data)
        self.assertTrue(accepted, errors)

    def test_solver_declines_infeasible_request(self) -> None:
        vnr = build_toy_vnr()
        for node in vnr.nodes:
            vnr.nodes[node]["cpu"] = 500

        embedding = solve_local_embedding(
            vnr,
            build_toy_substrate(),
        )

        self.assertIsNone(embedding)

    def test_solver_enforces_architecture_compatibility(self) -> None:
        vnr = build_toy_vnr()
        vnr.nodes["v0"]["arch"] = "x86"
        substrate = build_toy_substrate()
        for node in substrate.nodes:
            substrate.nodes[node]["arch"] = "ARM"

        embedding = solve_local_embedding(vnr, substrate)

        self.assertIsNone(embedding)

    def test_solver_enforces_memory_capacity(self) -> None:
        vnr = build_toy_vnr()
        for node in vnr.nodes:
            vnr.nodes[node]["memory_gib"] = 64
        substrate = build_toy_substrate()
        for node in substrate.nodes:
            substrate.nodes[node]["memory_gib"] = 32

        embedding = solve_local_embedding(vnr, substrate)

        self.assertIsNone(embedding)

    def test_solver_enforces_fixed_node_mapping(self) -> None:
        substrate = build_toy_substrate()
        substrate.nodes["s3"]["public_endpoint"] = True
        vnr = build_toy_vnr()
        vnr.nodes["v2"]["cpu"] = 0
        vnr.nodes["v2"]["public_endpoint"] = True
        vnr.remove_edge("v1", "v2")
        vnr.add_edge("v0", "v2", bandwidth=10)

        embedding = solve_local_embedding(
            vnr,
            substrate,
            fixed_node_mapping={"v2": "s3"},
        )

        self.assertIsNotNone(embedding)
        assert embedding is not None
        self.assertEqual(embedding.node_mapping["v2"], "s3")

    def test_solver_co_locates_virtual_nodes_with_zero_hop_paths(self) -> None:
        vnr = build_toy_vnr()
        for node in vnr.nodes:
            vnr.nodes[node]["cpu"] = 10

        substrate = build_toy_substrate()
        for node in substrate.nodes:
            substrate.nodes[node]["cpu"] = 0
        substrate.nodes["s0"]["cpu"] = 30

        embedding = solve_local_embedding(vnr, substrate)

        self.assertIsNotNone(embedding)
        assert embedding is not None
        self.assertEqual(set(embedding.node_mapping.values()), {"s0"})
        self.assertEqual(embedding.paths[("v0", "v1")], ["s0"])
        self.assertEqual(embedding.paths[("v1", "v2")], ["s0"])

        data = build_zk_input(
            vnr,
            substrate,
            node_mapping=embedding.node_mapping,
            paths=embedding.paths,
            compute_commitments=False,
        )
        accepted, errors = verify_resource_feasibility(data)
        self.assertTrue(accepted, errors)
        self.assertEqual(data["sBwAfter"], data["sBwBefore"])


if __name__ == "__main__":
    unittest.main()
