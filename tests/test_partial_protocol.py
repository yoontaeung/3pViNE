"""Tests for connected partial VNE splitting and proof reuse."""

from __future__ import annotations

import unittest

import networkx as nx

from partial_protocol import (
    BoundaryDemand,
    ConnectedPartialClaim,
    DeclineClaim,
    FullEmbeddingClaim,
    PublicEndpoint,
    build_circuit_vnr_for_nodes,
    derive_residual_request,
    solve_connected_vne_claim,
    verify_residual_consistency,
)
from verifier import verify_resource_feasibility
from zk_encoding import S_NODE_IDS, build_zk_input


def build_arch_vnr() -> nx.Graph:
    graph = nx.Graph(req_id=7)
    graph.add_node("v0", cpu=20, arch="ARM")
    graph.add_node("v1", cpu=30, arch="x86")
    graph.add_edge("v0", "v1", bandwidth=10)
    return graph


def build_connected_split_vnr() -> nx.Graph:
    graph = nx.Graph(req_id=7)
    graph.add_node("v0", cpu=20, arch="ARM")
    graph.add_node("v1", cpu=30, arch="x86")
    graph.add_node("v2", cpu=20, arch="ARM")
    graph.add_edge("v0", "v1", bandwidth=10)
    graph.add_edge("v1", "v2", bandwidth=10)
    return graph


def build_multi_node_partial_vnr() -> nx.Graph:
    graph = nx.Graph(req_id=8)
    graph.add_node("v0", cpu=10, arch="ARM")
    graph.add_node("v1", cpu=10, arch="ARM")
    graph.add_node("v2", cpu=30, arch="x86")
    graph.add_edge("v0", "v1", bandwidth=5)
    graph.add_edge("v1", "v2", bandwidth=10)
    return graph


def build_two_crossing_edge_vnr() -> nx.Graph:
    """Only v1 fits the ARM facility, but selecting it creates two cut edges."""
    graph = nx.Graph(req_id=9)
    graph.add_node("v0", cpu=10, arch="x86")
    graph.add_node("v1", cpu=10, arch="ARM")
    graph.add_node("v2", cpu=10, arch="x86")
    graph.add_edge("v0", "v1", bandwidth=5)
    graph.add_edge("v1", "v2", bandwidth=5)
    graph.add_edge("v0", "v2", bandwidth=5)
    return graph


def build_canonical_arm_facility() -> nx.Graph:
    graph = nx.Graph(facility_id="facility-A")
    graph.add_node("s0", cpu=40, arch="ARM")
    graph.add_node("s1", cpu=20, arch="ARM")
    graph.add_node("s2", cpu=40, arch="ARM")
    graph.add_node("s3", cpu=0, arch=0, public_endpoint=True)
    graph.add_edge("s0", "s2", bandwidth=20)
    graph.add_edge("s1", "s2", bandwidth=20)
    graph.add_edge("s2", "s3", bandwidth=20)
    return graph


def build_canonical_all_arch_facility() -> nx.Graph:
    graph = build_canonical_arm_facility()
    graph.nodes["s1"]["arch"] = "x86"
    graph.nodes["s1"]["cpu"] = 40
    return graph


def build_canonical_x86_facility() -> nx.Graph:
    graph = nx.Graph(facility_id="facility-B")
    graph.add_node("s0", cpu=50, arch="x86")
    graph.add_node("s1", cpu=20, arch="x86")
    graph.add_node("s2", cpu=20, arch="x86")
    graph.add_node("s3", cpu=0, arch=0, public_endpoint=True)
    graph.add_edge("s0", "s3", bandwidth=20)
    graph.add_edge("s1", "s3", bandwidth=20)
    graph.add_edge("s2", "s3", bandwidth=20)
    return graph


class ConnectedPartialProtocolTests(unittest.TestCase):
    def test_residual_request_preserves_cut_edge_as_boundary_demand(self) -> None:
        residual = derive_residual_request(
            build_arch_vnr(),
            request_id=7,
            embedded_virtual_nodes={"v0"},
            source_facility_id="facility-A",
            source_endpoint_id="A-public-0",
        )

        self.assertEqual(set(residual.residual_vnr.nodes), {"v1"})
        self.assertEqual(
            residual.boundary_demands[0],
            BoundaryDemand(
                source_facility_id="facility-A",
                source_endpoint_id="A-public-0",
                embedded_virtual_node="v0",
                residual_virtual_node="v1",
                bandwidth=10,
            ),
        )

        accepted, errors = verify_residual_consistency(build_arch_vnr(), residual)
        self.assertTrue(accepted, errors)

    def test_single_crossing_edge_partial_split_is_accepted(self) -> None:
        residual = derive_residual_request(
            build_connected_split_vnr(),
            request_id=7,
            embedded_virtual_nodes={"v0"},
            source_facility_id="facility-A",
            source_endpoint_id="A-public-0",
        )

        accepted, errors = verify_residual_consistency(build_connected_split_vnr(), residual)

        self.assertTrue(accepted, errors)
        self.assertEqual(len(residual.boundary_demands), 1)

    def test_two_crossing_edge_partial_split_is_rejected(self) -> None:
        vnr = build_two_crossing_edge_vnr()
        residual = derive_residual_request(
            vnr,
            request_id=9,
            embedded_virtual_nodes={"v1"},
            source_facility_id="facility-A",
            source_endpoint_id="A-public-0",
        )

        accepted, errors = verify_residual_consistency(vnr, residual)

        self.assertFalse(accepted)
        self.assertEqual(len(residual.boundary_demands), 2)
        self.assertIn("partial split must have exactly one crossing virtual edge", errors)

    def test_solver_declines_when_only_feasible_partial_has_two_crossing_edges(self) -> None:
        claim = solve_connected_vne_claim(
            build_two_crossing_edge_vnr(),
            build_canonical_arm_facility(),
            facility_id="facility-A",
            endpoint=PublicEndpoint("A-public-0", "facility-A", "s3"),
        )

        self.assertIsInstance(claim, DeclineClaim)
        assert isinstance(claim, DeclineClaim)
        self.assertEqual(claim.reason, "no_connected_partial_embedding")

    def test_residual_consistency_rejects_missing_boundary_demand(self) -> None:
        vnr = build_arch_vnr()
        residual = derive_residual_request(
            vnr,
            request_id=7,
            embedded_virtual_nodes={"v0"},
            source_facility_id="facility-A",
            source_endpoint_id="A-public-0",
        )
        tampered = residual.__class__(
            request_id=residual.request_id,
            residual_vnr=residual.residual_vnr,
            embedded_virtual_nodes=residual.embedded_virtual_nodes,
            boundary_demands=(),
        )

        accepted, errors = verify_residual_consistency(vnr, tampered)

        self.assertFalse(accepted)
        self.assertTrue(any("boundary demands" in error for error in errors))

    def test_connected_solver_returns_full_when_full_embedding_exists(self) -> None:
        claim = solve_connected_vne_claim(
            build_connected_split_vnr(),
            build_canonical_all_arch_facility(),
            facility_id="facility-A",
            endpoint=PublicEndpoint("A-public-0", "facility-A", "s3"),
        )

        self.assertIsInstance(claim, FullEmbeddingClaim)
        assert isinstance(claim, FullEmbeddingClaim)
        self.assertEqual(set(claim.slot_to_original.values()), {"v0", "v1", "v2"})
        self.assertIsNone(claim.endpoint_substrate_node)

        data = build_zk_input(
            claim.local_vnr,
            build_canonical_all_arch_facility(),
            node_mapping=claim.embedding.node_mapping,
            paths=claim.embedding.paths,
            req_id=7,
            randomness={"rho_S": 11, "rho_M": 12, "rho_S_next": 13},
            compute_commitments=False,
        )
        accepted, errors = verify_resource_feasibility(data)

        self.assertTrue(accepted, errors)

    def test_connected_solver_falls_back_to_connected_partial_claim(self) -> None:
        claim = solve_connected_vne_claim(
            build_connected_split_vnr(),
            build_canonical_arm_facility(),
            facility_id="facility-A",
            endpoint=PublicEndpoint("A-public-0", "facility-A", "s3"),
        )

        self.assertIsInstance(claim, ConnectedPartialClaim)
        assert isinstance(claim, ConnectedPartialClaim)
        self.assertEqual(claim.embedded_virtual_nodes, ("v0",))
        self.assertEqual(set(claim.residual.residual_vnr.nodes), {"v1", "v2"})
        self.assertEqual(claim.endpoint_virtual_node, "v2")
        self.assertEqual(claim.embedding.node_mapping["v2"], "s3")
        self.assertIn(("v0", "v2"), claim.embedding.paths)

    def test_connected_solver_does_not_select_disconnected_embedded_subset(self) -> None:
        claim = solve_connected_vne_claim(
            build_connected_split_vnr(),
            build_canonical_arm_facility(),
            facility_id="facility-A",
            endpoint=PublicEndpoint("A-public-0", "facility-A", "s3"),
        )

        self.assertIsInstance(claim, ConnectedPartialClaim)
        assert isinstance(claim, ConnectedPartialClaim)
        self.assertNotEqual(set(claim.embedded_virtual_nodes), {"v0", "v2"})

    def test_augmented_partial_vnr_reuses_full_embedding_reference_checker(self) -> None:
        claim = solve_connected_vne_claim(
            build_connected_split_vnr(),
            build_canonical_arm_facility(),
            facility_id="facility-A",
            endpoint=PublicEndpoint("A-public-0", "facility-A", "s3"),
        )
        self.assertIsInstance(claim, ConnectedPartialClaim)
        assert isinstance(claim, ConnectedPartialClaim)

        data = build_zk_input(
            claim.local_vnr,
            build_canonical_arm_facility(),
            node_mapping=claim.embedding.node_mapping,
            paths=claim.embedding.paths,
            req_id=7,
            randomness={"rho_S": 11, "rho_M": 12, "rho_S_next": 13},
            endpoint_idx=S_NODE_IDS[claim.endpoint_substrate_node],
            compute_commitments=False,
        )
        accepted, errors = verify_resource_feasibility(data)

        self.assertTrue(accepted, errors)

    def test_residual_full_claim_proves_incoming_endpoint_attachment(self) -> None:
        residual = derive_residual_request(
            build_arch_vnr(),
            request_id=7,
            embedded_virtual_nodes={"v0"},
            source_facility_id="facility-A",
            source_endpoint_id="A-public-0",
        )
        claim = solve_connected_vne_claim(
            residual.residual_vnr,
            build_canonical_x86_facility(),
            facility_id="facility-B",
            endpoint=PublicEndpoint("B-public-0", "facility-B", "s3"),
            incoming_boundary_demands=residual.boundary_demands,
        )

        self.assertIsInstance(claim, FullEmbeddingClaim)
        assert isinstance(claim, FullEmbeddingClaim)
        self.assertEqual(claim.endpoint_virtual_node, "v2")
        self.assertEqual(claim.endpoint_substrate_node, "s3")
        self.assertIn(("v0", "v2"), claim.embedding.paths)

    def test_residual_request_cannot_be_split_again(self) -> None:
        residual = derive_residual_request(
            build_connected_split_vnr(),
            request_id=7,
            embedded_virtual_nodes={"v0"},
            source_facility_id="facility-A",
            source_endpoint_id="A-public-0",
        )
        claim = solve_connected_vne_claim(
            residual.residual_vnr,
            build_canonical_arm_facility(),
            facility_id="facility-B",
            endpoint=PublicEndpoint("B-public-0", "facility-B", "s3"),
            incoming_boundary_demands=residual.boundary_demands,
        )

        self.assertIsInstance(claim, DeclineClaim)
        assert isinstance(claim, DeclineClaim)
        self.assertEqual(claim.reason, "residual_full_embedding_infeasible")

    def test_connected_solver_can_embed_multiple_virtual_nodes_before_residual(self) -> None:
        claim = solve_connected_vne_claim(
            build_multi_node_partial_vnr(),
            build_canonical_arm_facility(),
            facility_id="facility-A",
            endpoint=PublicEndpoint("A-public-0", "facility-A", "s3"),
        )

        self.assertIsInstance(claim, ConnectedPartialClaim)
        assert isinstance(claim, ConnectedPartialClaim)
        self.assertEqual(claim.embedded_virtual_nodes, ("v0", "v1"))
        self.assertEqual(set(claim.residual.residual_vnr.nodes), {"v2"})
        self.assertEqual(len(claim.residual.boundary_demands), 1)
        self.assertEqual(claim.residual.boundary_demands[0].embedded_virtual_node, "v1")
        self.assertEqual(claim.residual.boundary_demands[0].residual_virtual_node, "v2")
        self.assertIn(("v0", "v1"), claim.embedding.paths)
        self.assertIn(("v1", "v2"), claim.embedding.paths)

        data = build_zk_input(
            claim.local_vnr,
            build_canonical_arm_facility(),
            node_mapping=claim.embedding.node_mapping,
            paths=claim.embedding.paths,
            req_id=8,
            randomness={"rho_S": 11, "rho_M": 12, "rho_S_next": 13},
            endpoint_idx=S_NODE_IDS[claim.endpoint_substrate_node],
            compute_commitments=False,
        )
        accepted, errors = verify_resource_feasibility(data)

        self.assertTrue(accepted, errors)

    def test_residual_after_multi_node_partial_can_be_completed_by_second_facility(self) -> None:
        claim_a = solve_connected_vne_claim(
            build_multi_node_partial_vnr(),
            build_canonical_arm_facility(),
            facility_id="facility-A",
            endpoint=PublicEndpoint("A-public-0", "facility-A", "s3"),
        )
        self.assertIsInstance(claim_a, ConnectedPartialClaim)
        assert isinstance(claim_a, ConnectedPartialClaim)

        claim_b = solve_connected_vne_claim(
            claim_a.residual.residual_vnr,
            build_canonical_x86_facility(),
            facility_id="facility-B",
            endpoint=PublicEndpoint("B-public-0", "facility-B", "s3"),
            incoming_boundary_demands=claim_a.residual.boundary_demands,
        )

        self.assertIsInstance(claim_b, FullEmbeddingClaim)
        assert isinstance(claim_b, FullEmbeddingClaim)
        self.assertEqual(claim_b.slot_to_original, {"v0": "v2"})
        self.assertEqual(claim_b.endpoint_virtual_node, "v2")
        self.assertEqual(claim_b.endpoint_substrate_node, "s3")

        data = build_zk_input(
            claim_b.local_vnr,
            build_canonical_x86_facility(),
            node_mapping=claim_b.embedding.node_mapping,
            paths=claim_b.embedding.paths,
            req_id=8,
            randomness={"rho_S": 21, "rho_M": 22, "rho_S_next": 23},
            endpoint_idx=S_NODE_IDS[claim_b.endpoint_substrate_node],
            compute_commitments=False,
        )
        accepted, errors = verify_resource_feasibility(data)

        self.assertTrue(accepted, errors)

    def test_augmented_vnr_rejects_too_many_embedded_nodes_for_endpoint_slot(self) -> None:
        with self.assertRaises(ValueError):
            build_circuit_vnr_for_nodes(
                build_connected_split_vnr(),
                ["v0", "v1", "v2"],
                endpoint_virtual_node="v2",
            )


if __name__ == "__main__":
    unittest.main()
