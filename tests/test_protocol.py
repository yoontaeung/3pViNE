"""Tests for protocol state-chain helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from protocol import PRIVATE_LOG_FIELDS, PublicArtifacts, StaleStateError, TransparencyLog
from protocol_demo import (
    DEFAULT_FACILITY_PRIVATE_PATH,
    DEFAULT_PORTAL_REQUESTS_PATH,
    boundary_demands_public_data,
    compute_residual_commitment,
    graph_public_data,
    load_facilities,
    load_portal_requests,
    verify_portal_residual_forwarding,
)
from partial_protocol import ConnectedPartialClaim, FullEmbeddingClaim, solve_connected_vne_claim
from verifier import verify_resource_feasibility
from zk_encoding import S_NODE_IDS, build_zk_input


ARTIFACTS = PublicArtifacts(
    proof_path="public/request_001/proof.json",
    public_input_path="public/request_001/public.json",
)


class TransparencyLogTests(unittest.TestCase):
    def test_initial_state_registration_appends_init_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = TransparencyLog(Path(tmp) / "transparency_log.jsonl")

            record = log.append_init(
                facility_id="facility-A",
                state_commitment="S0",
                timestamp="2026-01-01T00:00:00Z",
            )

            self.assertEqual(record["status"], "init")
            self.assertEqual(record["record_index"], 0)
            self.assertEqual(log.latest_state("facility-A"), "S0")
            self.assertEqual(len(log.records), 1)

            pretty_log = json.loads((Path(tmp) / "transparency_log.json").read_text())
            self.assertEqual(pretty_log, log.records)

    def test_accepted_transition_requires_latest_state_and_updates_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = TransparencyLog(Path(tmp) / "transparency_log.jsonl")
            log.append_init(facility_id="facility-A", state_commitment="S0")

            record = log.append_accepted(
                facility_id="facility-A",
                request_id=1,
                state_before="S0",
                state_after="S1",
                request_commitment="R1",
                embedding_commitment="M1",
                artifacts=ARTIFACTS,
            )

            self.assertEqual(record["status"], "accepted")
            self.assertEqual(log.latest_state("facility-A"), "S1")

    def test_declined_transition_leaves_latest_state_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = TransparencyLog(Path(tmp) / "transparency_log.jsonl")
            log.append_init(facility_id="facility-A", state_commitment="S0")

            record = log.append_declined(
                facility_id="facility-A",
                request_id=3,
                state_before="S0",
                request_commitment="R3",
            )

            self.assertEqual(record["status"], "declined")
            self.assertEqual(record["state_after"], "S0")
            self.assertEqual(record["reason"], "facility_declined_unproven")
            self.assertEqual(log.latest_state("facility-A"), "S0")

    def test_stale_accepted_transition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = TransparencyLog(Path(tmp) / "transparency_log.jsonl")
            log.append_init(facility_id="facility-A", state_commitment="S0")
            log.append_accepted(
                facility_id="facility-A",
                request_id=1,
                state_before="S0",
                state_after="S1",
                request_commitment="R1",
                embedding_commitment="M1",
                artifacts=ARTIFACTS,
            )

            with self.assertRaises(StaleStateError):
                log.append_accepted(
                    facility_id="facility-A",
                    request_id=2,
                    state_before="S0",
                    state_after="S2",
                    request_commitment="R2",
                    embedding_commitment="M2",
                    artifacts=ARTIFACTS,
                )

    def test_public_log_records_do_not_contain_private_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = TransparencyLog(Path(tmp) / "transparency_log.jsonl")
            log.append_init(facility_id="facility-A", state_commitment="S0")
            log.append_declined(
                facility_id="facility-A",
                request_id=3,
                state_before="S0",
                request_commitment="R3",
            )

            for record in log.records:
                self.assertFalse(PRIVATE_LOG_FIELDS.intersection(record))

    def test_verified_state_is_committed_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = TransparencyLog(Path(tmp) / "transparency_log.jsonl")
            log.append_init(facility_id="facility-A", state_commitment="S0")
            claim = log.append_event(
                event_type="claim_submitted",
                claim_type="partial",
                request_scope="original",
                facility_id="facility-A",
                request_id=1,
            )

            proof = log.append_event(
                event_type="proof_accepted",
                actor="portal",
                facility_id="facility-A",
                request_id=1,
                parent_claim_record=claim["record_index"],
                result="accepted",
            )
            commit = log.append_state_committed(
                facility_id="facility-A",
                request_id=1,
                state_before="S0",
                state_after="S1",
                parent_claim_record=claim["record_index"],
                proof_accepted_record=proof["record_index"],
            )

            self.assertEqual(commit["event_type"], "state_committed")
            self.assertEqual(log.latest_state("facility-A"), "S1")

    def test_reference_validation_can_commit_state_without_a_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = TransparencyLog(Path(tmp) / "transparency_log.jsonl")
            log.append_init(facility_id="facility-A", state_commitment="S0")
            claim = log.append_event(
                event_type="claim_submitted",
                claim_type="partial",
                facility_id="facility-A",
                request_id=1,
            )
            validation = log.append_event(
                event_type="reference_validation_accepted",
                actor="portal",
                facility_id="facility-A",
                request_id=1,
                parent_claim_record=claim["record_index"],
                result="accepted",
            )

            commit = log.append_reference_state_committed(
                facility_id="facility-A",
                request_id=1,
                state_before="S0",
                state_after="S1",
                parent_claim_record=claim["record_index"],
                validation_accepted_record=validation["record_index"],
            )

            self.assertEqual(commit["reason"], "reference_validation_accepted_no_zkp")
            self.assertEqual(log.latest_state("facility-A"), "S1")

    def test_residual_decline_does_not_rollback_partial_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = TransparencyLog(Path(tmp) / "transparency_log.jsonl")
            log.append_init(facility_id="facility-A", state_commitment="S0")
            claim = log.append_event(
                event_type="claim_submitted",
                claim_type="partial",
                request_scope="original",
                facility_id="facility-A",
                request_id=1,
            )
            proof = log.append_event(
                event_type="proof_accepted",
                actor="portal",
                facility_id="facility-A",
                request_id=1,
                parent_claim_record=claim["record_index"],
                result="accepted",
            )
            log.append_state_committed(
                facility_id="facility-A",
                request_id=1,
                state_before="S0",
                state_after="S1",
                parent_claim_record=claim["record_index"],
                proof_accepted_record=proof["record_index"],
            )
            log.append_event(
                event_type="request_declined",
                request_id=1,
                facility_id=None,
                request_scope="residual",
                reason="facility_order_exhausted",
                state_before=None,
                state_after=None,
            )

            self.assertEqual(log.latest_state("facility-A"), "S1")
            self.assertNotIn("state_released", [record.get("event_type") for record in log.records])


class ProtocolSequenceTests(unittest.TestCase):
    def test_portal_requests_do_not_contain_private_answers(self) -> None:
        config = json.loads(DEFAULT_PORTAL_REQUESTS_PATH.read_text(encoding="utf-8"))
        forbidden = {"status", "node_mapping", "paths", "rho_S", "rho_M", "rho_S_next"}

        for request in config["requests"]:
            self.assertFalse(forbidden.intersection(request))

    def test_facility_private_config_does_not_contain_portal_answers(self) -> None:
        config = json.loads(DEFAULT_FACILITY_PRIVATE_PATH.read_text(encoding="utf-8"))
        self.assertIn("facilities", config)
        self.assertNotIn("facility_id", config)
        self.assertNotIn("requests", config)
        self.assertNotIn("node_mapping", json.dumps(config))
        self.assertNotIn("paths", json.dumps(config))
        self.assertNotIn("status", json.dumps(config))

    def test_configured_multi_facility_partial_sequence_with_reference_checker(self) -> None:
        requests = {request.request_id: request for request in load_portal_requests()}
        facilities, connectivity = load_facilities()
        facility_a = facilities["facility-A"]
        facility_b = facilities["facility-B"]

        full_request = requests[100]
        full_claim = solve_connected_vne_claim(
            full_request.vnr,
            facility_a.substrate,
            facility_id="facility-A",
            endpoint=facility_a.config.endpoint,
            policy=facility_a.config.solver_policy,
        )
        self.assertIsInstance(full_claim, FullEmbeddingClaim)
        assert isinstance(full_claim, FullEmbeddingClaim)
        full_data = build_zk_input(
            full_claim.local_vnr,
            facility_a.substrate,
            node_mapping=full_claim.embedding.node_mapping,
            paths=full_claim.embedding.paths,
            req_id=full_request.request_id,
            randomness={"rho_S": 1, "rho_M": 2, "rho_S_next": 3},
            compute_commitments=False,
        )
        accepted, errors = verify_resource_feasibility(full_data)
        self.assertTrue(accepted, errors)

        request = requests[101]
        claim_a = solve_connected_vne_claim(
            request.vnr,
            facility_a.substrate,
            facility_id="facility-A",
            endpoint=facility_a.config.endpoint,
            policy=facility_a.config.solver_policy,
        )
        self.assertIsInstance(claim_a, ConnectedPartialClaim)
        assert isinstance(claim_a, ConnectedPartialClaim)

        data_a = build_zk_input(
            claim_a.local_vnr,
            facility_a.substrate,
            node_mapping=claim_a.embedding.node_mapping,
            paths=claim_a.embedding.paths,
            req_id=request.request_id,
            randomness={"rho_S": 1, "rho_M": 2, "rho_S_next": 3},
            endpoint_idx=S_NODE_IDS[claim_a.endpoint_substrate_node],
            compute_commitments=False,
        )
        accepted, errors = verify_resource_feasibility(data_a)
        self.assertTrue(accepted, errors)

        claim_b = solve_connected_vne_claim(
            claim_a.residual.residual_vnr,
            facility_b.substrate,
            facility_id="facility-B",
            endpoint=facility_b.config.endpoint,
            policy=facility_b.config.solver_policy,
            incoming_boundary_demands=claim_a.residual.boundary_demands,
        )
        self.assertIsInstance(claim_b, FullEmbeddingClaim)
        assert isinstance(claim_b, FullEmbeddingClaim)
        self.assertIsNotNone(claim_b.endpoint_substrate_node)

        data_b = build_zk_input(
            claim_b.local_vnr,
            facility_b.substrate,
            node_mapping=claim_b.embedding.node_mapping,
            paths=claim_b.embedding.paths,
            req_id=request.request_id,
            randomness={"rho_S": 4, "rho_M": 5, "rho_S_next": 6},
            endpoint_idx=S_NODE_IDS[claim_b.endpoint_substrate_node],
            compute_commitments=False,
        )
        accepted, errors = verify_resource_feasibility(data_b)
        self.assertTrue(accepted, errors)
        self.assertTrue(
            connectivity.can_satisfy(
                claim_a.residual.boundary_demands[0].source_endpoint_id,
                facility_b.config.endpoint.endpoint_id,
                claim_a.residual.boundary_demands[0].bandwidth,
            )
        )

    def test_portal_residual_forwarding_event_is_verifiable_by_target_facility(self) -> None:
        request = {request.request_id: request for request in load_portal_requests()}[101]
        facilities, _ = load_facilities()
        facility_a = facilities["facility-A"]

        claim = solve_connected_vne_claim(
            request.vnr,
            facility_a.substrate,
            facility_id="facility-A",
            endpoint=facility_a.config.endpoint,
            policy=facility_a.config.solver_policy,
        )
        self.assertIsInstance(claim, ConnectedPartialClaim)
        assert isinstance(claim, ConnectedPartialClaim)

        residual_commitment = compute_residual_commitment(
            request_id=request.request_id,
            residual_vnr=claim.residual.residual_vnr,
            boundary_demands=claim.residual.boundary_demands,
        )

        with tempfile.TemporaryDirectory() as tmp:
            log = TransparencyLog(Path(tmp) / "transparency_log.jsonl")
            log.append_init(facility_id="facility-A", state_commitment="S0")
            parent = log.append_event(
                event_type="claim_submitted",
                claim_type="partial",
                request_scope="original",
                facility_id="facility-A",
                request_id=request.request_id,
                residual_commitment=residual_commitment,
                covered_virtual_nodes=list(claim.embedded_virtual_nodes),
            )
            proof = log.append_event(
                event_type="proof_submitted",
                actor="facility",
                facility_id="facility-A",
                request_id=request.request_id,
                claim_type="partial",
                request_scope="original",
                parent_claim_record=parent["record_index"],
                proof_path="public/request_101_facility-A_01/proof.json",
                public_input_path="public/request_101_facility-A_01/public.json",
            )
            proof_accepted = log.append_event(
                event_type="proof_accepted",
                actor="portal",
                verifier_id="portal",
                facility_id="facility-A",
                request_id=request.request_id,
                claim_type="partial",
                request_scope="original",
                parent_claim_record=parent["record_index"],
                proof_record=proof["record_index"],
                result="accepted",
            )
            uncommitted_forward = {
                "record_index": len(log.records),
                "event_type": "residual_forwarded",
                "actor": "portal",
                "request_id": request.request_id,
                "source_facility_id": "facility-A",
                "target_facility_id": "facility-B",
                "parent_claim_record": parent["record_index"],
                "residual_commitment": residual_commitment,
                "residual_vnr": graph_public_data(claim.residual.residual_vnr),
                "boundary_demands": boundary_demands_public_data(
                    claim.residual.boundary_demands
                ),
            }
            accepted, errors = verify_portal_residual_forwarding(
                log,
                uncommitted_forward,
                source_vnr=request.vnr,
                residual_vnr=claim.residual.residual_vnr,
                boundary_demands=claim.residual.boundary_demands,
                target_facility_id="facility-B",
            )
            self.assertFalse(accepted)
            self.assertTrue(any("has not been committed" in error for error in errors))

            log.append_state_committed(
                facility_id="facility-A",
                request_id=request.request_id,
                state_before="S0",
                state_after="S1",
                parent_claim_record=parent["record_index"],
                proof_accepted_record=proof_accepted["record_index"],
            )
            forwarded = log.append_event(
                event_type="residual_forwarded",
                actor="portal",
                request_id=request.request_id,
                source_facility_id="facility-A",
                target_facility_id="facility-B",
                parent_claim_record=parent["record_index"],
                residual_commitment=residual_commitment,
                residual_vnr=graph_public_data(claim.residual.residual_vnr),
                boundary_demands=boundary_demands_public_data(
                    claim.residual.boundary_demands
                ),
            )

            accepted, errors = verify_portal_residual_forwarding(
                log,
                forwarded,
                source_vnr=request.vnr,
                residual_vnr=claim.residual.residual_vnr,
                boundary_demands=claim.residual.boundary_demands,
                target_facility_id="facility-B",
            )

            self.assertTrue(accepted, errors)

    def test_portal_residual_forwarding_rejects_tampered_commitment_or_target(self) -> None:
        request = {request.request_id: request for request in load_portal_requests()}[101]
        facilities, _ = load_facilities()
        facility_a = facilities["facility-A"]

        claim = solve_connected_vne_claim(
            request.vnr,
            facility_a.substrate,
            facility_id="facility-A",
            endpoint=facility_a.config.endpoint,
            policy=facility_a.config.solver_policy,
        )
        self.assertIsInstance(claim, ConnectedPartialClaim)
        assert isinstance(claim, ConnectedPartialClaim)

        residual_commitment = compute_residual_commitment(
            request_id=request.request_id,
            residual_vnr=claim.residual.residual_vnr,
            boundary_demands=claim.residual.boundary_demands,
        )

        with tempfile.TemporaryDirectory() as tmp:
            log = TransparencyLog(Path(tmp) / "transparency_log.jsonl")
            log.append_init(facility_id="facility-A", state_commitment="S0")
            parent = log.append_event(
                event_type="claim_submitted",
                claim_type="partial",
                request_scope="original",
                facility_id="facility-A",
                request_id=request.request_id,
                residual_commitment=residual_commitment,
                covered_virtual_nodes=list(claim.embedded_virtual_nodes),
            )
            proof = log.append_event(
                event_type="proof_submitted",
                actor="facility",
                facility_id="facility-A",
                request_id=request.request_id,
                claim_type="partial",
                request_scope="original",
                parent_claim_record=parent["record_index"],
                proof_path="public/request_101_facility-A_01/proof.json",
                public_input_path="public/request_101_facility-A_01/public.json",
            )
            proof_accepted = log.append_event(
                event_type="proof_accepted",
                actor="portal",
                verifier_id="portal",
                facility_id="facility-A",
                request_id=request.request_id,
                claim_type="partial",
                request_scope="original",
                parent_claim_record=parent["record_index"],
                proof_record=proof["record_index"],
                result="accepted",
            )
            log.append_state_committed(
                facility_id="facility-A",
                request_id=request.request_id,
                state_before="S0",
                state_after="S1",
                parent_claim_record=parent["record_index"],
                proof_accepted_record=proof_accepted["record_index"],
            )
            forwarded = log.append_event(
                event_type="residual_forwarded",
                actor="portal",
                request_id=request.request_id,
                source_facility_id="facility-A",
                target_facility_id="facility-B",
                parent_claim_record=parent["record_index"],
                residual_commitment=residual_commitment,
                residual_vnr=graph_public_data(claim.residual.residual_vnr),
                boundary_demands=boundary_demands_public_data(
                    claim.residual.boundary_demands
                ),
            )

            bad_commitment = {**forwarded, "residual_commitment": "bad"}
            accepted, errors = verify_portal_residual_forwarding(
                log,
                bad_commitment,
                source_vnr=request.vnr,
                residual_vnr=claim.residual.residual_vnr,
                boundary_demands=claim.residual.boundary_demands,
                target_facility_id="facility-B",
            )
            self.assertFalse(accepted)
            self.assertTrue(any("commitment" in error for error in errors))

            wrong_target = {**forwarded, "target_facility_id": "facility-C"}
            accepted, errors = verify_portal_residual_forwarding(
                log,
                wrong_target,
                source_vnr=request.vnr,
                residual_vnr=claim.residual.residual_vnr,
                boundary_demands=claim.residual.boundary_demands,
                target_facility_id="facility-B",
            )
            self.assertFalse(accepted)
            self.assertTrue(any("target facility" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
