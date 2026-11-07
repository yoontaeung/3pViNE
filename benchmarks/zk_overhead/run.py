#!/usr/bin/env python3
"""Run parameterized Groth16 overhead measurements inside the ZK container."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.experiment_log import ExperimentLog, source_manifest
from benchmarks.zk_overhead.analyze import write_analysis
from benchmarks.zk_overhead.profiles import (
    CircuitProfile,
    generate_circuit_entrypoint,
    parse_profile,
)
from benchmarks.zk_overhead.replay import (
    add_commitments,
    build_replay_fields,
    validate_replay_dimensions,
    write_input,
)


CIRCOMLIB_PATH = Path("/opt/zk-vne-js/node_modules/circomlib/circuits")
DEFAULT_PROFILES = ("vn3_sn6_h3",)
MAX_EXPLORATORY_REPETITIONS = 3
SOURCE_INPUTS = (
    ROOT / "Dockerfile",
    ROOT / "package.json",
    ROOT / "requirements.txt",
    ROOT / "circuits" / "resource_feasibility.circom",
    ROOT / "scripts" / "poseidon_chain.js",
    ROOT / "zk_encoding.py",
    ROOT / "benchmarks" / "common" / "experiment_log.py",
    ROOT / "benchmarks" / "zk_overhead" / "profiles.py",
    ROOT / "benchmarks" / "zk_overhead" / "replay.py",
    ROOT / "benchmarks" / "zk_overhead" / "analyze.py",
    ROOT / "benchmarks" / "zk_overhead" / "run.py",
    ROOT / "benchmarks" / "scripts" / "run_zkp_overhead.sh",
)


def result_path_reference(run_dir: Path, path: Path) -> str:
    """Return a stable path for artifacts inside or outside one result directory."""
    resolved = path.resolve()
    for base in (run_dir.resolve(), ROOT.resolve()):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(resolved)


def validate_measurement_counts(repetitions: int, warmups: int) -> None:
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups cannot be negative")
    if repetitions > MAX_EXPLORATORY_REPETITIONS:
        raise ValueError(
            f"exploratory runs are capped at {MAX_EXPLORATORY_REPETITIONS} repetitions"
        )


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    output = (result.stdout or result.stderr).strip()
    if output:
        return output.splitlines()[0]
    return f"unavailable: command exited with status {result.returncode}"


def git_metadata() -> dict[str, Any]:
    revision = command_version(["git", "rev-parse", "HEAD"])
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def environment_metadata(argv: list[str]) -> dict[str, Any]:
    return {
        "command": argv,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "circom": command_version(["circom", "--version"]),
        "node": command_version(["node", "--version"]),
        "node_options": os.environ.get("NODE_OPTIONS", ""),
        "snarkjs": command_version(["snarkjs", "--help"]),
        "git": git_metadata(),
    }


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged_command(
    recorder: ExperimentLog,
    *,
    profile_id: str,
    stage: str,
    command: list[str],
    log_path: Path,
) -> tuple[float, str]:
    recorder.append("stage_started", profile_id=profile_id, stage=stage, command=command)
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = result.stdout + result.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    recorded_log_path = result_path_reference(recorder.run_dir, log_path)
    if result.returncode != 0:
        recorder.append(
            "stage_failed",
            profile_id=profile_id,
            stage=stage,
            elapsed_seconds=elapsed,
            returncode=result.returncode,
            log_path=recorded_log_path,
        )
        raise subprocess.CalledProcessError(result.returncode, command, output=output)
    recorder.append(
        "stage_completed",
        profile_id=profile_id,
        stage=stage,
        elapsed_seconds=elapsed,
        log_path=recorded_log_path,
    )
    return elapsed, output


def run_logged_command_atomic_output(
    recorder: ExperimentLog,
    *,
    profile_id: str,
    stage: str,
    command: list[str],
    output_index: int,
    output_path: Path,
    log_path: Path,
) -> tuple[float, str]:
    """Publish a generated artifact only after its command completes."""
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    atomic_command = list(command)
    atomic_command[output_index] = str(temporary_path)
    try:
        elapsed, output = run_logged_command(
            recorder,
            profile_id=profile_id,
            stage=stage,
            command=atomic_command,
            log_path=log_path,
        )
        if not temporary_path.is_file():
            raise RuntimeError(f"{stage} did not create {temporary_path}")
        temporary_path.replace(output_path)
        return elapsed, output
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_r1cs_info(output: str) -> dict[str, int]:
    patterns = {
        "wires": r"# of Wires:\s*(\d+)",
        "constraints": r"# of Constraints:\s*(\d+)",
        "private_inputs": r"# of Private Inputs:\s*(\d+)",
        "public_inputs": r"# of Public Inputs:\s*(\d+)",
        "labels": r"# of Labels:\s*(\d+)",
        "outputs": r"# of Outputs:\s*(\d+)",
    }
    parsed: dict[str, int] = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, output)
        if not match:
            raise RuntimeError(f"snarkjs r1cs info did not report {field}")
        parsed[field] = int(match.group(1))
    return parsed


def ensure_ptau(
    recorder: ExperimentLog,
    *,
    profile_id: str,
    constraints: int,
    cache_dir: Path,
) -> tuple[Path, int, float]:
    power = max(18, math.ceil(math.log2(constraints + 1)))
    existing_demo = ROOT / "build" / "zk" / "pot18_final.ptau"
    if power == 18 and existing_demo.exists():
        recorder.append(
            "ptau_reused",
            profile_id=profile_id,
            power=power,
            path=str(existing_demo.relative_to(ROOT)),
        )
        return existing_demo, power, 0.0

    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / f"pot{power}_final.ptau"
    if final_path.exists():
        recorder.append(
            "ptau_reused",
            profile_id=profile_id,
            power=power,
            path=str(final_path.relative_to(ROOT)),
        )
        return final_path, power, 0.0

    initial = cache_dir / f"pot{power}_0000.ptau"
    contributed = cache_dir / f"pot{power}_0001.ptau"
    total = 0.0
    for stage, command, output_index, output_path, log_path in (
        (
            "ptau_new",
            ["snarkjs", "powersoftau", "new", "bn128", str(power), str(initial)],
            5,
            initial,
            cache_dir / f"pot{power}_new.log",
        ),
        (
            "ptau_contribute",
            [
                "snarkjs",
                "powersoftau",
                "contribute",
                str(initial),
                str(contributed),
                "--name=zk-vne evaluation",
                "-e=zk-vne evaluation deterministic demo entropy",
            ],
            4,
            contributed,
            cache_dir / f"pot{power}_contribute.log",
        ),
        (
            "ptau_prepare",
            [
                "snarkjs",
                "powersoftau",
                "prepare",
                "phase2",
                str(contributed),
                str(final_path),
            ],
            5,
            final_path,
            cache_dir / f"pot{power}_prepare.log",
        ),
    ):
        elapsed, _ = run_logged_command_atomic_output(
            recorder,
            profile_id=profile_id,
            stage=stage,
            command=command,
            output_index=output_index,
            output_path=output_path,
            log_path=log_path,
        )
        total += elapsed
    return final_path, power, total


def artifact_metadata(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hash_file(path),
        }
        for name, path in paths.items()
        if path.exists()
    }


def compile_and_setup(
    recorder: ExperimentLog,
    profile: CircuitProfile,
    profile_dir: Path,
) -> dict[str, Any]:
    profile_id = profile.profile_id
    source_path = profile_dir / f"{profile_id}.circom"
    build_dir = profile_dir / "build"
    logs_dir = profile_dir / "logs"
    build_dir.mkdir(parents=True, exist_ok=True)
    generate_circuit_entrypoint(
        profile,
        base_circuit=ROOT / "circuits" / "resource_feasibility.circom",
        output_path=source_path,
    )

    compile_seconds, _ = run_logged_command(
        recorder,
        profile_id=profile_id,
        stage="circom_compile",
        command=[
            "circom",
            str(source_path),
            "--r1cs",
            "--wasm",
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
        stage="r1cs_info",
        command=["snarkjs", "r1cs", "info", str(r1cs_path)],
        log_path=logs_dir / "r1cs_info.log",
    )
    r1cs = parse_r1cs_info(info_output)
    ptau_path, ptau_power, ptau_seconds = ensure_ptau(
        recorder,
        profile_id=profile_id,
        constraints=r1cs["constraints"],
        cache_dir=ROOT / "artifacts" / "benchmarks" / "ptau",
    )

    initial_zkey = build_dir / f"{profile_id}_0000.zkey"
    final_zkey = build_dir / f"{profile_id}_final.zkey"
    verification_key = build_dir / f"{profile_id}_verification_key.json"
    setup_seconds, _ = run_logged_command(
        recorder,
        profile_id=profile_id,
        stage="groth16_setup",
        command=[
            "snarkjs",
            "groth16",
            "setup",
            str(r1cs_path),
            str(ptau_path),
            str(initial_zkey),
        ],
        log_path=logs_dir / "groth16_setup.log",
    )
    contribution_seconds, _ = run_logged_command(
        recorder,
        profile_id=profile_id,
        stage="zkey_contribute",
        command=[
            "snarkjs",
            "zkey",
            "contribute",
            str(initial_zkey),
            str(final_zkey),
            "--name=zk-vne evaluation",
            "-e=zk-vne evaluation deterministic demo phase2 entropy",
        ],
        log_path=logs_dir / "zkey_contribute.log",
    )
    export_seconds, _ = run_logged_command(
        recorder,
        profile_id=profile_id,
        stage="verification_key_export",
        command=[
            "snarkjs",
            "zkey",
            "export",
            "verificationkey",
            str(final_zkey),
            str(verification_key),
        ],
        log_path=logs_dir / "verification_key_export.log",
    )

    wasm_path = build_dir / f"{profile_id}_js" / f"{profile_id}.wasm"
    artifacts = artifact_metadata(
        {
            "source": source_path,
            "r1cs": r1cs_path,
            "wasm": wasm_path,
            "proving_key": final_zkey,
            "verification_key": verification_key,
        }
    )
    return {
        "profile": profile.as_dict(),
        "r1cs": r1cs,
        "ptau_power": ptau_power,
        "offline_seconds": {
            "circom_compile": compile_seconds,
            "ptau_generation": ptau_seconds,
            "groth16_setup": setup_seconds,
            "zkey_contribute": contribution_seconds,
            "verification_key_export": export_seconds,
        },
        "artifacts": artifacts,
        "paths": {
            "wasm": wasm_path,
            "witness_generator": build_dir / f"{profile_id}_js" / "generate_witness.js",
            "proving_key": final_zkey,
            "verification_key": verification_key,
        },
    }


def execute_replay(
    recorder: ExperimentLog,
    profile: CircuitProfile,
    profile_dir: Path,
    setup: dict[str, Any],
    *,
    repetition: int,
    measured: bool,
) -> dict[str, Any]:
    profile_id = profile.profile_id
    label = f"repetition_{repetition:03d}" if measured else f"warmup_{repetition:03d}"
    output_dir = profile_dir / ("measurements" if measured else "warmups") / label
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data = build_replay_fields(profile, request_id=1000 + repetition)
    validate_replay_dimensions(profile, data)
    transition_seconds = time.perf_counter() - started

    started = time.perf_counter()
    add_commitments(data)
    commitment_seconds = time.perf_counter() - started

    input_path = output_dir / "input.json"
    started = time.perf_counter()
    write_input(data, input_path)
    input_write_seconds = time.perf_counter() - started

    paths = setup["paths"]
    witness_path = output_dir / "witness.wtns"
    proof_path = output_dir / "proof.json"
    public_path = output_dir / "public.json"
    logs_dir = output_dir / "logs"

    witness_seconds, _ = run_logged_command(
        recorder,
        profile_id=profile_id,
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
        profile_id=profile_id,
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
        profile_id=profile_id,
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
        raise RuntimeError(f"verification output did not report success for {profile_id}")

    commitment_mode_seconds = transition_seconds + commitment_seconds + input_write_seconds
    zkp_incremental_seconds = witness_seconds + prove_seconds + verify_seconds
    return {
        "repetition": repetition,
        "transition_seconds": transition_seconds,
        "commitment_seconds": commitment_seconds,
        "input_write_seconds": input_write_seconds,
        "commitment_mode_seconds": commitment_mode_seconds,
        "witness_seconds": witness_seconds,
        "prove_seconds": prove_seconds,
        "verify_seconds": verify_seconds,
        "zkp_incremental_seconds": zkp_incremental_seconds,
        "online_groth16_seconds": commitment_mode_seconds + zkp_incremental_seconds,
        "input_bytes": input_path.stat().st_size,
        "witness_bytes": witness_path.stat().st_size,
        "proof_bytes": proof_path.stat().st_size,
        "public_input_bytes": public_path.stat().st_size,
        "proof_sha256": hash_file(proof_path),
        "verified": True,
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def summarize_measurements(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    timing_fields = (
        "transition_seconds",
        "commitment_seconds",
        "input_write_seconds",
        "commitment_mode_seconds",
        "witness_seconds",
        "prove_seconds",
        "verify_seconds",
        "zkp_incremental_seconds",
        "online_groth16_seconds",
    )
    summary: dict[str, Any] = {}
    for field in timing_fields:
        values = [float(item[field]) for item in measurements]
        summary[field] = {
            "min": min(values),
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
    for field in ("input_bytes", "witness_bytes", "proof_bytes", "public_input_bytes"):
        summary[field] = {
            "min": min(int(item[field]) for item in measurements),
            "max": max(int(item[field]) for item in measurements),
        }
    return summary


def append_measurement_csv(
    path: Path,
    *,
    profile: CircuitProfile,
    constraints: int,
    measurement: dict[str, Any],
) -> None:
    row = {
        "profile_id": profile.profile_id,
        "vn": profile.vn,
        "sn": profile.sn,
        "hops": profile.hops,
        "pair_count": profile.pair_count,
        "dominant_term": profile.dominant_term,
        "constraints": constraints,
        **measurement,
    }
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_profile_summary_csv(path: Path, profiles: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for profile_id, result in profiles.items():
        if result.get("status") != "completed":
            continue
        profile = result["profile"]
        timing = result["summary"]
        offline = result["offline_seconds"]
        rows.append(
            {
                "profile_id": profile_id,
                "vn": profile["vn"],
                "sn": profile["sn"],
                "hops": profile["hops"],
                "pair_count": profile["pair_count"],
                "dominant_term": profile["dominant_term"],
                "constraints": result["r1cs"]["constraints"],
                "compile_seconds": offline["circom_compile"],
                "setup_seconds": offline["groth16_setup"],
                "commitment_median_seconds": timing["commitment_seconds"]["median"],
                "witness_median_seconds": timing["witness_seconds"]["median"],
                "witness_p95_seconds": timing["witness_seconds"]["p95"],
                "prove_median_seconds": timing["prove_seconds"]["median"],
                "prove_p95_seconds": timing["prove_seconds"]["p95"],
                "verify_median_seconds": timing["verify_seconds"]["median"],
                "verify_p95_seconds": timing["verify_seconds"]["p95"],
                "zkp_incremental_median_seconds": timing["zkp_incremental_seconds"]["median"],
                "online_median_seconds": timing["online_groth16_seconds"]["median"],
                "witness_bytes": timing["witness_bytes"]["max"],
                "proof_bytes_max": timing["proof_bytes"]["max"],
                "public_input_bytes_max": timing["public_input_bytes"]["max"],
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
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_measurement_counts(args.repetitions, args.warmups)
    profiles = [parse_profile(value) for value in args.profiles]
    recorder = ExperimentLog(
        args.run_dir,
        experiment="zkp_overhead",
        metadata={
            **environment_metadata(sys.argv),
            "requested_profiles": [profile.as_dict() for profile in profiles],
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "source_snapshot": source_manifest(ROOT, list(SOURCE_INPUTS)),
            "timing_boundary": (
                "precomputed transition, batched commitments, input write, witness, "
                "prove, and verify; solver, compile, and setup excluded from online total"
            ),
        },
    )
    try:
        for profile in profiles:
            profile_id = profile.profile_id
            profile_dir = recorder.run_dir / "profiles" / profile_id
            recorder.update_profile(profile_id, status="building", profile=profile.as_dict())
            setup = compile_and_setup(recorder, profile, profile_dir)
            serializable_setup = {key: value for key, value in setup.items() if key != "paths"}
            recorder.update_profile(profile_id, status="warming_up", **serializable_setup)

            for warmup in range(1, args.warmups + 1):
                recorder.append("warmup_started", profile_id=profile_id, repetition=warmup)
                execute_replay(
                    recorder,
                    profile,
                    profile_dir,
                    setup,
                    repetition=warmup,
                    measured=False,
                )
                recorder.append("warmup_completed", profile_id=profile_id, repetition=warmup)

            recorder.update_profile(profile_id, status="measuring")
            for repetition in range(1, args.repetitions + 1):
                measurement = execute_replay(
                    recorder,
                    profile,
                    profile_dir,
                    setup,
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

            measurements = recorder.state["profiles"][profile_id]["measurements"]
            recorder.update_profile(
                profile_id,
                status="completed",
                summary=summarize_measurements(measurements),
            )
            write_profile_summary_csv(
                recorder.run_dir / "summary.csv",
                recorder.state["profiles"],
            )
            recorder.append("profile_completed", profile_id=profile_id)
        recorder.complete()
        write_analysis(recorder.run_dir, recorder.state)
    except Exception as exc:
        recorder.fail(f"{type(exc).__name__}: {exc}")
        raise

    print(f"Results: {recorder.results_path}")
    print(f"Journal: {recorder.journal_path}")
    print(f"Analysis: {recorder.run_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
