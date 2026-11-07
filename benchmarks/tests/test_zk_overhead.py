"""Fast tests for parameterized replay experiment inputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.zk_overhead.analyze import TIMING_FIELDS, analyze_results
from benchmarks.zk_overhead.profiles import (
    CircuitProfile,
    generate_circuit_entrypoint,
    parse_profile,
)
from benchmarks.zk_overhead.replay import build_replay_fields, validate_replay_dimensions
from benchmarks.common.experiment_log import ExperimentLog
from benchmarks.zk_overhead.run import (
    result_path_reference,
    run_logged_command_atomic_output,
    validate_measurement_counts,
)


ROOT = Path(__file__).resolve().parents[2]


class CircuitProfileTests(unittest.TestCase):
    def test_single_node_profile_has_no_virtual_pairs(self) -> None:
        profile = parse_profile("vn1_sn30_h3")
        self.assertEqual(profile.pair_count, 0)
        self.assertEqual(profile.sn_bits, 5)

    def test_profile_derives_pair_count_and_index_bits(self) -> None:
        profile = parse_profile("vn5_sn10_h4")
        self.assertEqual(profile, CircuitProfile(vn=5, sn=10, hops=4))
        self.assertEqual(profile.pair_count, 10)
        self.assertEqual(profile.sn_bits, 4)
        self.assertEqual(profile.as_dict()["dominant_term"], 4000)

    def test_generated_entrypoint_uses_profile_parameters(self) -> None:
        profile = CircuitProfile(vn=4, sn=8, hops=2)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "profile.circom"
            generate_circuit_entrypoint(
                profile,
                base_circuit=ROOT / "circuits" / "resource_feasibility.circom",
                output_path=output,
            )
            source = output.read_text(encoding="utf-8")
            self.assertIn("ResourceFeasibility(4, 8, 3, 6, 2);", source)
            self.assertNotIn("ResourceFeasibility(3, 6, 3, 3, 3);", source)

    def test_result_path_can_reference_shared_ptau_cache(self) -> None:
        run_dir = ROOT / "artifacts" / "benchmarks" / "zk_overhead" / "one-run"
        cache_log = ROOT / "artifacts" / "benchmarks" / "ptau" / "pot19_new.log"
        self.assertEqual(
            result_path_reference(run_dir, cache_log),
            "artifacts/benchmarks/ptau/pot19_new.log",
        )

    def test_exploratory_repetitions_are_capped_at_three(self) -> None:
        validate_measurement_counts(3, 0)
        with self.assertRaisesRegex(ValueError, "capped at 3"):
            validate_measurement_counts(4, 0)

    def test_atomic_output_is_published_only_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            recorder = ExperimentLog(
                run_dir,
                experiment="atomic-output-test",
                metadata={},
            )
            output = run_dir / "pot.ptau"

            def fake_run(*args: object, **kwargs: object) -> tuple[float, str]:
                command = kwargs["command"]
                Path(command[-1]).write_bytes(b"complete")
                return 1.0, ""

            with patch("benchmarks.zk_overhead.run.run_logged_command", fake_run):
                run_logged_command_atomic_output(
                    recorder,
                    profile_id="profile",
                    stage="ptau_prepare",
                    command=["snarkjs", "prepare", str(output)],
                    output_index=2,
                    output_path=output,
                    log_path=run_dir / "prepare.log",
                )

            self.assertEqual(output.read_bytes(), b"complete")
            self.assertFalse((run_dir / ".pot.ptau.tmp").exists())


class ReplayWitnessTests(unittest.TestCase):
    def test_single_node_replay_is_a_node_only_full_embedding(self) -> None:
        profile = CircuitProfile(vn=1, sn=30, hops=3)
        data = build_replay_fields(profile)
        validate_replay_dimensions(profile, data)

        self.assertEqual(data["nodeMap"], [0])
        self.assertEqual(data["pathNodes"], [])
        self.assertEqual(data["sCpuAfter"][0], 99)
        self.assertTrue(all(value == 0 for row in data["vBw"] for value in row))

    def test_replay_dimensions_follow_profile(self) -> None:
        profile = CircuitProfile(vn=4, sn=8, hops=3)
        data = build_replay_fields(profile)
        validate_replay_dimensions(profile, data)

        self.assertEqual(len(data["vBw"]), 4)
        self.assertEqual(len(data["sBwBefore"]), 8)
        self.assertEqual(len(data["pathNodes"]), 6)
        self.assertTrue(all(len(path) == 4 for path in data["pathNodes"]))

    def test_chain_replay_reserves_expected_resources(self) -> None:
        profile = CircuitProfile(vn=3, sn=6, hops=3)
        data = build_replay_fields(profile)

        self.assertEqual(data["sCpuAfter"][:3], [99, 99, 99])
        self.assertEqual(data["sBwAfter"][0][1], 99)
        self.assertEqual(data["sBwAfter"][1][2], 99)
        self.assertEqual(data["sBwAfter"][0][2], 100)


class ResultAnalysisTests(unittest.TestCase):
    def test_analysis_validates_proofs_and_compares_profiles(self) -> None:
        def measurement(value: float) -> dict[str, object]:
            return {
                **{field: value for field in TIMING_FIELDS},
                "verified": True,
                "witness_bytes": int(value * 100),
                "proof_bytes": 800,
            }

        state = {
            "status": "completed",
            "profiles": {
                "small": {
                    "profile": {"sn": 4},
                    "r1cs": {"constraints": 100},
                    "measurements": [measurement(1.0), measurement(1.0)],
                },
                "large": {
                    "profile": {"sn": 8},
                    "r1cs": {"constraints": 150},
                    "measurements": [measurement(2.0), measurement(2.0)],
                },
            },
        }
        analysis = analyze_results(state, [])
        self.assertTrue(analysis["all_proofs_verified"])
        comparison = analysis["comparisons_to_baseline"]["large"]
        self.assertEqual(comparison["constraint_increase_percent"], 50.0)
        self.assertEqual(
            comparison["median_timing_increase_percent"]["prove_seconds"],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
