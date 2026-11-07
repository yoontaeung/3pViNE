"""Tests for the resource-feasibility Circom input encoding."""

from __future__ import annotations

import json
import subprocess
import unittest

from graph_data import build_toy_substrate, build_toy_vnr
from verifier import verify_resource_feasibility
from zk_encoding import (
    EMBEDDING_DOMAIN,
    POSEIDON_HELPER,
    PoseidonCommitter,
    REQUEST_DOMAIN,
    S_NODE_COUNT,
    SUBSTRATE_DOMAIN,
    SUBSTRATE_NEXT_DOMAIN,
    build_zk_input,
    compute_poseidon_commitments,
    embedding_fields,
    path_bandwidth_use_matrix,
    request_fields,
    stringify_field_values,
    substrate_fields,
)


def build_encoded_input(compute_commitments: bool = False):
    return build_zk_input(
        build_toy_vnr(),
        build_toy_substrate(),
        compute_commitments=compute_commitments,
    )


def one_hot(index: int) -> list[int]:
    return [1 if snode == index else 0 for snode in range(S_NODE_COUNT)]


def poseidon_helper_available() -> bool:
    try:
        subprocess.run(
            ["node", str(POSEIDON_HELPER)],
            input=json.dumps({"domain": "1", "fields": ["2"]}),
            text=True,
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


class ZkEncodingTests(unittest.TestCase):
    def test_toy_graphs_encode_to_stable_resource_arrays(self) -> None:
        data = build_encoded_input()

        self.assertEqual(data["reqId"], 1)
        self.assertEqual(data["vNodePresent"], [1, 1, 1])
        self.assertEqual(data["vCpu"], [30, 50, 100])
        self.assertEqual(data["vMem"], [0, 0, 0])
        self.assertEqual(data["vArch"], [0, 0, 0])
        self.assertEqual(data["endpointEnabled"], 0)
        self.assertEqual(data["endpointIdx"], 0)
        self.assertEqual(
            data["vLinkPresent"],
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
        )
        self.assertEqual(
            data["vBw"],
            [
                [0, 10, 0],
                [10, 0, 5],
                [0, 5, 0],
            ],
        )

        self.assertEqual(data["sNodePresent"], [1, 1, 1, 1, 0, 0])
        self.assertEqual(data["sCpuBefore"], [100, 120, 100, 200, 0, 0])
        self.assertEqual(data["sMemBefore"], [0, 0, 0, 0, 0, 0])
        self.assertEqual(data["sArch"], [0, 0, 0, 0, 0, 0])
        self.assertEqual(
            data["sAdj"],
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        )
        self.assertEqual(
            data["sBwBefore"],
            [
                [0, 0, 20, 0, 0, 0],
                [0, 0, 15, 20, 0, 0],
                [20, 15, 0, 20, 0, 0],
                [0, 20, 20, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        )

    def test_default_multi_hop_witness_reserves_expected_resources(self) -> None:
        data = build_encoded_input()

        self.assertEqual(data["nodeMap"], [0, 3, 1])
        self.assertEqual(
            data["nodeSel"],
            [[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]],
        )
        self.assertEqual(
            data["pathNodes"],
            [
                [0, 2, 3, 0],
                [0, 0, 0, 0],
                [3, 1, 0, 0],
            ],
        )
        self.assertEqual(
            data["pathLenSel"],
            [[0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
        )
        self.assertEqual(data["sCpuAfter"], [70, 20, 100, 150, 0, 0])
        self.assertEqual(data["sMemAfter"], [0, 0, 0, 0, 0, 0])
        self.assertEqual(
            data["sBwAfter"],
            [
                [0, 0, 10, 0, 0, 0],
                [0, 0, 15, 15, 0, 0],
                [10, 15, 0, 10, 0, 0],
                [0, 15, 10, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        )

    def test_path_bandwidth_use_matches_expected_hops(self) -> None:
        data = build_encoded_input()

        used_bw = path_bandwidth_use_matrix(
            v_bw=data["vBw"],
            path_nodes=data["pathNodes"],
            path_len_sel=data["pathLenSel"],
        )

        self.assertEqual(
            used_bw,
            [
                [0, 0, 10, 0, 0, 0],
                [0, 0, 0, 5, 0, 0],
                [10, 0, 0, 10, 0, 0],
                [0, 5, 10, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        )

    def test_reference_checker_accepts_valid_multi_hop_witness(self) -> None:
        data = build_encoded_input()

        accepted, errors = verify_resource_feasibility(data)

        self.assertTrue(accepted, errors)

    def test_reference_checker_accepts_valid_direct_one_hop_paths(self) -> None:
        data = build_zk_input(
            build_toy_vnr(),
            build_toy_substrate(),
            node_mapping={"v0": "s1", "v1": "s2", "v2": "s0"},
            paths={
                ("v0", "v1"): ["s1", "s2"],
                ("v1", "v2"): ["s2", "s0"],
            },
            compute_commitments=False,
        )

        accepted, errors = verify_resource_feasibility(data)

        self.assertTrue(accepted, errors)

    def test_reference_checker_accepts_many_to_one_zero_hop_embedding(self) -> None:
        substrate = build_toy_substrate()
        data = build_zk_input(
            build_toy_vnr(),
            substrate,
            node_mapping={"v0": "s3", "v1": "s3", "v2": "s3"},
            paths={
                ("v0", "v1"): ["s3"],
                ("v1", "v2"): ["s3"],
            },
            compute_commitments=False,
        )

        accepted, errors = verify_resource_feasibility(data)

        self.assertTrue(accepted, errors)
        self.assertEqual(data["nodeMap"], [3, 3, 3])
        self.assertEqual(data["pathLenSel"][0], [1, 0, 0, 0])
        self.assertEqual(data["pathLenSel"][2], [1, 0, 0, 0])
        self.assertEqual(data["sCpuAfter"], [100, 120, 100, 20, 0, 0])
        self.assertEqual(data["sBwAfter"], data["sBwBefore"])

    def test_zero_bandwidth_virtual_lan_still_requires_a_path(self) -> None:
        vnr = build_toy_vnr()
        vnr["v0"]["v1"]["bandwidth"] = 0
        vnr["v1"]["v2"]["bandwidth"] = 0
        data = build_zk_input(
            vnr,
            build_toy_substrate(),
            node_mapping={"v0": "s3", "v1": "s3", "v2": "s3"},
            paths={
                ("v0", "v1"): ["s3"],
                ("v1", "v2"): ["s3"],
            },
            compute_commitments=False,
        )

        accepted, errors = verify_resource_feasibility(data)

        self.assertTrue(accepted, errors)
        self.assertEqual(data["vLinkPresent"][0][1], 1)
        self.assertEqual(data["vBw"][0][1], 0)
        self.assertEqual(data["pathLenSel"][0], [1, 0, 0, 0])
        self.assertEqual(data["sBwAfter"], data["sBwBefore"])

    def test_present_zero_bandwidth_link_without_path_is_rejected_by_encoder(self) -> None:
        vnr = build_toy_vnr()
        vnr["v0"]["v1"]["bandwidth"] = 0

        with self.assertRaisesRegex(ValueError, "must contain 0"):
            build_zk_input(
                vnr,
                build_toy_substrate(),
                node_mapping={"v0": "s0", "v1": "s3", "v2": "s1"},
                paths={("v1", "v2"): ["s3", "s1"]},
                compute_commitments=False,
            )

    def test_reference_checker_accepts_inactive_padded_virtual_slot(self) -> None:
        vnr = build_toy_vnr()
        vnr.remove_edge("v1", "v2")
        vnr.nodes["v2"].update(cpu=0, memory_gib=0, arch=0, present=0)
        data = build_zk_input(
            vnr,
            build_toy_substrate(),
            node_mapping={"v0": "s0", "v1": "s3"},
            paths={("v0", "v1"): ["s0", "s2", "s3"]},
            compute_commitments=False,
        )

        accepted, errors = verify_resource_feasibility(data)

        self.assertTrue(accepted, errors)
        self.assertEqual(data["vNodePresent"], [1, 1, 0])
        self.assertEqual(data["nodeSel"][2], [0, 0, 0, 0, 0, 0])

    def test_reference_checker_rejects_zero_hop_between_different_nodes(self) -> None:
        data = build_zk_input(
            build_toy_vnr(),
            build_toy_substrate(),
            node_mapping={"v0": "s3", "v1": "s3", "v2": "s1"},
            paths={
                ("v0", "v1"): ["s3"],
                ("v1", "v2"): ["s3"],
            },
            compute_commitments=False,
        )

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("ends at the wrong node" in error for error in errors))

    def test_reference_checker_rejects_wrong_path_start(self) -> None:
        data = build_encoded_input()
        data["pathNodes"][0][0] = 1
        data["pathNodeSel"][0][0] = one_hot(1)

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("starts at the wrong node" in error for error in errors))

    def test_reference_checker_rejects_wrong_path_end(self) -> None:
        data = build_encoded_input()
        data["pathNodes"][0][2] = 1
        data["pathNodeSel"][0][2] = one_hot(1)

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("ends at the wrong node" in error for error in errors))

    def test_reference_checker_rejects_nonexistent_substrate_hop(self) -> None:
        data = build_zk_input(
            build_toy_vnr(),
            build_toy_substrate(),
            paths={
                ("v0", "v1"): ["s0", "s3"],
                ("v1", "v2"): ["s3", "s1"],
            },
            compute_commitments=False,
        )

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("nonexistent substrate hop 0-3" in error for error in errors))

    def test_reference_checker_rejects_path_through_dummy_node(self) -> None:
        data = build_encoded_input()
        data["pathNodes"][0][1] = 4
        data["pathNodeSel"][0][1] = one_hot(4)

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("uses dummy node 4" in error for error in errors))

    def test_reference_checker_rejects_shared_hop_bandwidth_overuse(self) -> None:
        vnr = build_toy_vnr()
        vnr["v1"]["v2"]["bandwidth"] = 15
        data = build_zk_input(
            vnr,
            build_toy_substrate(),
            paths={
                ("v0", "v1"): ["s0", "s2", "s3"],
                ("v1", "v2"): ["s3", "s2", "s1"],
            },
            compute_commitments=False,
        )

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("bandwidth capacity exceeded on substrate edge 2-3" in error for error in errors))

    def test_reference_checker_rejects_tampered_bandwidth_after_state(self) -> None:
        data = build_encoded_input()
        data["sBwAfter"][0][2] = 11

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("sBwAfter[0][2]" in error for error in errors))

    def test_reference_checker_rejects_tampered_memory_after_state(self) -> None:
        substrate = build_toy_substrate()
        for node in substrate.nodes:
            substrate.nodes[node]["memory_gib"] = 256
        vnr = build_toy_vnr()
        for node in vnr.nodes:
            vnr.nodes[node]["memory_gib"] = 4
        data = build_zk_input(vnr, substrate, compute_commitments=False)
        data["sMemAfter"][0] += 1

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("sMemAfter[0]" in error for error in errors))

    def test_reference_checker_rejects_mapping_to_dummy_substrate_node(self) -> None:
        data = build_encoded_input()
        data["nodeMap"][0] = 4
        data["nodeSel"][0] = one_hot(4)

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("dummy substrate node" in error for error in errors))

    def test_reference_checker_rejects_wrong_architecture_mapping(self) -> None:
        substrate = build_toy_substrate()
        substrate.nodes["s0"]["arch"] = "ARM"
        substrate.nodes["s1"]["arch"] = "ARM"
        substrate.nodes["s2"]["arch"] = "ARM"
        substrate.nodes["s3"]["arch"] = "x86"
        vnr = build_toy_vnr()
        vnr.nodes["v0"]["arch"] = "x86"

        data = build_zk_input(
            vnr,
            substrate,
            node_mapping={"v0": "s0", "v1": "s3", "v2": "s1"},
            paths={
                ("v0", "v1"): ["s0", "s2", "s3"],
                ("v1", "v2"): ["s3", "s1"],
            },
            compute_commitments=False,
        )

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("wrong architecture" in error for error in errors))

    def test_reference_checker_enforces_endpoint_mapping(self) -> None:
        data = build_encoded_input()
        data["endpointEnabled"] = 1
        data["endpointIdx"] = 2

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("endpoint virtual slot" in error for error in errors))

    def test_reference_checker_rejects_nonzero_dummy_bandwidth(self) -> None:
        data = build_encoded_input()
        data["sBwBefore"][4][1] = 9

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("dummy substrate node 4" in error for error in errors))

    def test_commitment_field_lengths_match_circuit_templates(self) -> None:
        data = build_encoded_input()

        self.assertEqual(len(request_fields(data)), 33)
        self.assertEqual(len(substrate_fields(data)), 97)
        self.assertEqual(len(embedding_fields(data)), 28)
        self.assertEqual(len(substrate_fields(data, next_state=True)), 97)
        self.assertEqual(SUBSTRATE_NEXT_DOMAIN, SUBSTRATE_DOMAIN)

    def test_stringify_field_values_prepares_snarkjs_json(self) -> None:
        data = stringify_field_values({"a": 1, "b": [2, {"c": 3}]})

        self.assertEqual(data, {"a": "1", "b": ["2", {"c": "3"}]})

    @unittest.skipUnless(poseidon_helper_available(), "circomlibjs is not installed outside the ZK container")
    def test_poseidon_helper_computes_all_public_commitments(self) -> None:
        data = build_encoded_input()
        cold_commitments = compute_poseidon_commitments(data, persistent=False)
        with PoseidonCommitter() as committer:
            commitments = compute_poseidon_commitments(data, committer=committer)
            repeated = compute_poseidon_commitments(data, committer=committer)

        self.assertEqual(
            set(commitments),
            {"C_r", "C_S", "C_M", "C_S_next"},
        )
        for value in commitments.values():
            self.assertGreater(value, 0)
        self.assertEqual(commitments, cold_commitments)
        self.assertEqual(commitments, repeated)

        self.assertEqual(REQUEST_DOMAIN, 1001)
        self.assertEqual(SUBSTRATE_DOMAIN, 1002)
        self.assertEqual(EMBEDDING_DOMAIN, 1003)


if __name__ == "__main__":
    unittest.main()
