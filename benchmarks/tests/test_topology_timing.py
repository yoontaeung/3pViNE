"""Fast tests for the same-VNR, different-SN-topology experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from benchmarks.zk_overhead.profiles import CircuitProfile
from benchmarks.zk_overhead.topology_analysis import write_topology_analysis
from benchmarks.zk_overhead.topology_run import rotated_order, validate_args
from benchmarks.zk_overhead.topology_workload import (
    TOPOLOGY_NAMES,
    build_substrate_graph,
    build_topology_replay_fields,
)


class TopologyWorkloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = CircuitProfile(vn=3, sn=8, hops=3)

    def test_topology_edge_counts_and_connectivity(self) -> None:
        full, full_metadata = build_substrate_graph("full_mesh", 8)
        ring, ring_metadata = build_substrate_graph("ring", 8)
        waxman, waxman_metadata = build_substrate_graph("waxman", 8)
        erdos_renyi, erdos_renyi_metadata = build_substrate_graph("erdos_renyi", 8)
        barabasi_albert, barabasi_albert_metadata = build_substrate_graph(
            "barabasi_albert", 8
        )

        self.assertEqual(full_metadata.edge_count, 28)
        self.assertEqual(ring_metadata.edge_count, 8)
        self.assertGreaterEqual(waxman_metadata.edge_count, 7)
        self.assertGreaterEqual(erdos_renyi_metadata.edge_count, 7)
        self.assertEqual(barabasi_albert_metadata.edge_count, 12)
        self.assertTrue(nx.is_connected(full))
        self.assertTrue(nx.is_connected(ring))
        self.assertTrue(nx.is_connected(waxman))
        self.assertTrue(nx.is_connected(erdos_renyi))
        self.assertTrue(nx.is_connected(barabasi_albert))
        for graph in (full, ring, waxman, erdos_renyi, barabasi_albert):
            self.assertTrue(graph.has_edge(0, 1))
            self.assertTrue(graph.has_edge(1, 2))

    def test_primary_profile_topology_sizes_are_frozen(self) -> None:
        expected_edges = {
            "full_mesh": 496,
            "ring": 32,
            "waxman": 79,
            "erdos_renyi": 79,
            "barabasi_albert": 60,
        }
        for topology, expected in expected_edges.items():
            _, metadata = build_substrate_graph(topology, 32)
            self.assertEqual(metadata.edge_count, expected)

    def test_only_private_substrate_topology_changes(self) -> None:
        fields = {
            topology: build_topology_replay_fields(self.profile, topology)[0]
            for topology in TOPOLOGY_NAMES
        }
        fixed_fields = (
            "reqId",
            "vNodePresent",
            "vCpu",
            "vMem",
            "vArch",
            "vLinkPresent",
            "vBw",
            "nodeMap",
            "nodeSel",
            "pathNodes",
            "pathNodeSel",
            "pathLenSel",
            "rho_S",
            "rho_M",
            "rho_S_next",
        )
        baseline = fields["full_mesh"]
        for topology in TOPOLOGY_NAMES[1:]:
            for field in fixed_fields:
                self.assertEqual(baseline[field], fields[topology][field], field)
            self.assertNotEqual(baseline["sAdj"], fields[topology]["sAdj"])

    def test_bandwidth_is_reserved_only_on_fixed_backbone(self) -> None:
        for topology in TOPOLOGY_NAMES:
            data, _ = build_topology_replay_fields(self.profile, topology)
            self.assertEqual(data["sBwAfter"][0][1], 99)
            self.assertEqual(data["sBwAfter"][1][2], 99)
            for source in range(self.profile.sn):
                for target in range(self.profile.sn):
                    if (source, target) not in ((0, 1), (1, 0), (1, 2), (2, 1)):
                        self.assertEqual(
                            data["sBwAfter"][source][target],
                            data["sBwBefore"][source][target],
                        )


class TopologyRunnerTests(unittest.TestCase):
    def test_measurement_order_rotates(self) -> None:
        self.assertEqual(rotated_order(0), TOPOLOGY_NAMES)
        self.assertEqual(
            rotated_order(1),
            ("ring", "waxman", "erdos_renyi", "barabasi_albert", "full_mesh"),
        )
        self.assertEqual(
            rotated_order(2),
            ("waxman", "erdos_renyi", "barabasi_albert", "full_mesh", "ring"),
        )

    def test_topology_runner_accepts_five_repetitions(self) -> None:
        validate_args(CircuitProfile(vn=3, sn=8, hops=3), 5, 1)
        validate_args(CircuitProfile(vn=3, sn=8, hops=3), 100, 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_args(CircuitProfile(vn=3, sn=8, hops=3), 0, 1)

    def test_min_mean_max_table_generation(self) -> None:
        def measurement(topology: str, repetition: int, value: float) -> dict[str, object]:
            return {
                "topology": topology,
                "repetition": repetition,
                "edge_count": 8,
                "density": 0.2,
                "witness_seconds": value,
                "prove_seconds": value * 10,
                "verify_seconds": value / 2,
                "verified": True,
            }

        state = {
            "profiles": {
                "vn3_sn8_h3": {
                    "profile": {"vn": 3, "sn": 8, "hops": 3},
                    "r1cs": {"constraints": 100},
                    "measurements": [
                        measurement(topology, repetition, value)
                        for topology in TOPOLOGY_NAMES
                        for repetition, value in enumerate((1.0, 2.0, 3.0), start=1)
                    ],
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = write_topology_analysis(root, state)
            self.assertTrue(all(path.is_file() for path in outputs))
            analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
            prove = analysis["topologies"]["ring"]["timing"]["prove_seconds"]
            self.assertEqual(prove["mean"], 20.0)
            self.assertEqual(prove["variance"], 100.0)
            self.assertEqual(prove["stddev"], 10.0)
            self.assertLess(prove["ci95_low"], 20.0)
            self.assertGreater(prove["ci95_high"], 20.0)
            table = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("min / mean / max", table)
            self.assertIn("Barabasi-Albert", table)


if __name__ == "__main__":
    unittest.main()
