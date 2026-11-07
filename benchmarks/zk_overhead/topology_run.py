#!/usr/bin/env python3
"""Measure one fixed VNR over five private substrate topologies."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.experiment_log import ExperimentLog, source_manifest
from benchmarks.zk_overhead.profiles import CircuitProfile, parse_profile
from benchmarks.zk_overhead.replay import add_commitments, write_input
from benchmarks.zk_overhead.run import (
    compile_and_setup,
    environment_metadata,
    hash_file,
    run_logged_command,
)
from benchmarks.zk_overhead.topology_analysis import write_topology_analysis
from benchmarks.zk_overhead.topology_workload import (
    TOPOLOGY_NAMES,
    build_topology_replay_fields,
)


DEFAULT_PROFILE = "vn3_sn32_h3"
SOURCE_INPUTS = (
    ROOT / "Dockerfile",
    ROOT / "requirements.txt",
    ROOT / "circuits" / "resource_feasibility.circom",
    ROOT / "zk_encoding.py",
    ROOT / "benchmarks" / "common" / "experiment_log.py",
    ROOT / "benchmarks" / "zk_overhead" / "profiles.py",
    ROOT / "benchmarks" / "zk_overhead" / "replay.py",
    ROOT / "benchmarks" / "zk_overhead" / "run.py",
    ROOT / "benchmarks" / "zk_overhead" / "topology_workload.py",
    ROOT / "benchmarks" / "zk_overhead" / "topology_analysis.py",
    ROOT / "benchmarks" / "zk_overhead" / "topology_run.py",
    ROOT / "benchmarks" / "scripts" / "run_topology_timing.sh",
)


def rotated_order(round_index: int) -> tuple[str, ...]:
    offset = round_index % len(TOPOLOGY_NAMES)
    return TOPOLOGY_NAMES[offset:] + TOPOLOGY_NAMES[:offset]


def validate_args(profile: CircuitProfile, repetitions: int, warmups: int) -> None:
    if profile.vn != 3:
        raise ValueError("the topology experiment requires VN=3")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if warmups < 0:
        raise ValueError("warmups cannot be negative")


def append_measurement_csv(
    path: Path,
    *,
    profile: CircuitProfile,
    constraints: int,
    measurement: dict[str, Any],
) -> None:
    row = {
        "topology": measurement["topology"],
        "repetition": measurement["repetition"],
        "vn": profile.vn,
        "sn": profile.sn,
        "hops": profile.hops,
        "edge_count": measurement["edge_count"],
        "density": measurement["density"],
        "constraints": constraints,
        "commitment_seconds": measurement["commitment_seconds"],
        "witness_seconds": measurement["witness_seconds"],
        "prove_seconds": measurement["prove_seconds"],
        "verify_seconds": measurement["verify_seconds"],
        "proof_bytes": measurement["proof_bytes"],
        "verified": measurement["verified"],
    }
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def execute_topology_replay(
    recorder: ExperimentLog,
    profile: CircuitProfile,
    profile_dir: Path,
    setup: dict[str, Any],
    *,
    topology: str,
    repetition: int,
    measured: bool,
) -> dict[str, Any]:
    run_type = "measurements" if measured else "warmups"
    label = f"{topology}_{'repetition' if measured else 'warmup'}_{repetition:03d}"
    output_dir = profile_dir / run_type / topology / f"round_{repetition:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data, topology_metadata = build_topology_replay_fields(
        profile,
        topology,
        request_id=1,
    )
    started = time.perf_counter()
    add_commitments(data)
    commitment_seconds = time.perf_counter() - started

    input_path = output_dir / "input.json"
    write_input(data, input_path)
    paths = setup["paths"]
    witness_path = output_dir / "witness.wtns"
    proof_path = output_dir / "proof.json"
    public_path = output_dir / "public.json"
    logs_dir = output_dir / "logs"

    witness_seconds, _ = run_logged_command(
        recorder,
        profile_id=profile.profile_id,
        stage=f"{label}_witness",
        command=[
            "node",
            str(paths["witness_generator"]),
            str(paths["wasm"]),
            str(input_path),
            str(witness_path),
        ],
        log_path=logs_dir / "witness.log",
    )
    prove_seconds, _ = run_logged_command(
        recorder,
        profile_id=profile.profile_id,
        stage=f"{label}_prove",
        command=[
            "snarkjs",
            "groth16",
            "prove",
            str(paths["proving_key"]),
            str(witness_path),
            str(proof_path),
            str(public_path),
        ],
        log_path=logs_dir / "prove.log",
    )
    verify_seconds, verify_output = run_logged_command(
        recorder,
        profile_id=profile.profile_id,
        stage=f"{label}_verify",
        command=[
            "snarkjs",
            "groth16",
            "verify",
            str(paths["verification_key"]),
            str(public_path),
            str(proof_path),
        ],
        log_path=logs_dir / "verify.log",
    )
    if "OK" not in verify_output:
        raise RuntimeError(f"proof verification failed for {topology}")

    return {
        "topology": topology,
        "repetition": repetition,
        "edge_count": topology_metadata.edge_count,
        "density": topology_metadata.density,
        "generator": topology_metadata.generator,
        "commitment_seconds": commitment_seconds,
        "witness_seconds": witness_seconds,
        "prove_seconds": prove_seconds,
        "verify_seconds": verify_seconds,
        "online_groth16_seconds": (
            commitment_seconds + witness_seconds + prove_seconds + verify_seconds
        ),
        "input_bytes": input_path.stat().st_size,
        "witness_bytes": witness_path.stat().st_size,
        "proof_bytes": proof_path.stat().st_size,
        "public_input_bytes": public_path.stat().st_size,
        "proof_sha256": hash_file(proof_path),
        "verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = parse_profile(args.profile)
    validate_args(profile, args.repetitions, args.warmups)
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("use a new empty run directory to avoid mixing measurements")
    recorder = ExperimentLog(
        args.run_dir,
        experiment="same_vnr_different_sn_topology",
        metadata={
            **environment_metadata(sys.argv),
            "profile": profile.as_dict(),
            "topologies": list(TOPOLOGY_NAMES),
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "vnr": {
                "nodes": [0, 1, 2],
                "edges": [[0, 1], [1, 2]],
                "node_mapping": [0, 1, 2],
                "substrate_paths": [[0, 1], [1, 2]],
            },
            "timing_scope": "commitment, witness generation, Groth16 prove, verification",
            "source_snapshot": source_manifest(ROOT, list(SOURCE_INPUTS)),
        },
    )
    profile_id = profile.profile_id
    profile_dir = recorder.run_dir / "profile"
    try:
        recorder.update_profile(profile_id, status="building", profile=profile.as_dict())
        setup = compile_and_setup(recorder, profile, profile_dir)
        serializable_setup = {key: value for key, value in setup.items() if key != "paths"}
        recorder.update_profile(profile_id, status="warming_up", **serializable_setup)

        for warmup in range(1, args.warmups + 1):
            for topology in rotated_order(warmup - 1):
                execute_topology_replay(
                    recorder,
                    profile,
                    profile_dir,
                    setup,
                    topology=topology,
                    repetition=warmup,
                    measured=False,
                )

        recorder.update_profile(profile_id, status="measuring")
        for repetition in range(1, args.repetitions + 1):
            for topology in rotated_order(repetition - 1):
                measurement = execute_topology_replay(
                    recorder,
                    profile,
                    profile_dir,
                    setup,
                    topology=topology,
                    repetition=repetition,
                    measured=True,
                )
                recorder.add_measurement(profile_id, measurement)
                append_measurement_csv(
                    recorder.run_dir / "measurements.csv",
                    profile=profile,
                    constraints=setup["r1cs"]["constraints"],
                    measurement=measurement,
                )
                write_topology_analysis(recorder.run_dir, recorder.state)
                print(f"Completed {topology} {repetition}/{args.repetitions}", flush=True)

        recorder.update_profile(profile_id, status="completed")
        recorder.complete()
        write_topology_analysis(recorder.run_dir, recorder.state)
    except Exception as exc:
        recorder.fail(f"{type(exc).__name__}: {exc}")
        raise

    print(f"Results: {recorder.results_path}")
    print(f"Measurements: {recorder.run_dir / 'measurements.csv'}")
    print(f"Summary: {recorder.run_dir / 'summary.csv'}")
    print(f"Table: {recorder.run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
