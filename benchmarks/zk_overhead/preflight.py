#!/usr/bin/env python3
"""Compile circuit profiles and report their exact R1CS size without setup."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.experiment_log import ExperimentLog, source_manifest
from benchmarks.zk_overhead.profiles import generate_circuit_entrypoint, parse_profile
from benchmarks.zk_overhead.run import (
    CIRCOMLIB_PATH,
    environment_metadata,
    parse_r1cs_info,
    run_logged_command,
)


SOURCE_INPUTS = (
    ROOT / "circuits" / "resource_feasibility.circom",
    ROOT / "benchmarks" / "common" / "experiment_log.py",
    ROOT / "benchmarks" / "zk_overhead" / "profiles.py",
    ROOT / "benchmarks" / "zk_overhead" / "run.py",
    ROOT / "benchmarks" / "zk_overhead" / "preflight.py",
)


def required_ptau_power(constraints: int) -> int:
    return max(18, math.ceil(math.log2(constraints + 1)))


def write_summary(path: Path, profiles: dict[str, Any]) -> None:
    rows = []
    for profile_id, result in profiles.items():
        if result.get("status") != "completed":
            continue
        profile = result["profile"]
        rows.append(
            {
                "profile_id": profile_id,
                "vn": profile["vn"],
                "sn": profile["sn"],
                "hops": profile["hops"],
                "pair_count": profile["pair_count"],
                "constraints": result["r1cs"]["constraints"],
                "wires": result["r1cs"]["wires"],
                "required_ptau_power": result["required_ptau_power"],
                "compile_seconds": result["compile_seconds"],
                "r1cs_bytes": result["r1cs_bytes"],
            }
        )
    if not rows:
        return
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = [parse_profile(value) for value in args.profiles]
    recorder = ExperimentLog(
        args.run_dir,
        experiment="zkp_scaling_preflight",
        metadata={
            **environment_metadata(sys.argv),
            "requested_profiles": [profile.as_dict() for profile in profiles],
            "scope": "Circom compile and R1CS inspection only; no Groth16 setup or proof",
            "source_snapshot": source_manifest(ROOT, list(SOURCE_INPUTS)),
        },
    )
    try:
        for profile in profiles:
            profile_id = profile.profile_id
            profile_dir = recorder.run_dir / "profiles" / profile_id
            build_dir = profile_dir / "build"
            logs_dir = profile_dir / "logs"
            source_path = profile_dir / f"{profile_id}.circom"
            build_dir.mkdir(parents=True, exist_ok=True)
            generate_circuit_entrypoint(
                profile,
                base_circuit=ROOT / "circuits" / "resource_feasibility.circom",
                output_path=source_path,
            )
            recorder.update_profile(profile_id, status="compiling", profile=profile.as_dict())
            compile_seconds, _ = run_logged_command(
                recorder,
                profile_id=profile_id,
                stage="circom_compile_preflight",
                command=[
                    "circom",
                    str(source_path),
                    "--r1cs",
                    "-l",
                    str(CIRCOMLIB_PATH),
                    "-o",
                    str(build_dir),
                ],
                log_path=logs_dir / "circom_compile.log",
            )
            r1cs_path = build_dir / f"{profile_id}.r1cs"
            _, info_output = run_logged_command(
                recorder,
                profile_id=profile_id,
                stage="r1cs_info_preflight",
                command=["snarkjs", "r1cs", "info", str(r1cs_path)],
                log_path=logs_dir / "r1cs_info.log",
            )
            r1cs = parse_r1cs_info(info_output)
            recorder.update_profile(
                profile_id,
                status="completed",
                r1cs=r1cs,
                compile_seconds=compile_seconds,
                required_ptau_power=required_ptau_power(r1cs["constraints"]),
                r1cs_bytes=r1cs_path.stat().st_size,
            )
            write_summary(recorder.run_dir / "summary.csv", recorder.state["profiles"])
            recorder.append("profile_completed", profile_id=profile_id)
        recorder.complete()
    except Exception as exc:
        recorder.fail(f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
