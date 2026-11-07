#!/usr/bin/env python3
"""Run the substrate-bound and VNR-bound scaling experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.experiment_log import ExperimentLog, source_manifest
from benchmarks.zk_overhead.scaling_results import write_scaling_results


SUBSTRATE_PROFILES = (
    "vn3_sn10_h3",
    "vn3_sn20_h3",
    "vn3_sn30_h3",
    "vn3_sn40_h3",
    "vn3_sn50_h3",
)
VNR_PROFILES = (
    "vn1_sn30_h3",
    "vn5_sn30_h3",
    "vn10_sn30_h3",
    "vn15_sn30_h3",
    "vn20_sn30_h3",
    "vn25_sn30_h3",
    "vn30_sn30_h3",
)
PREFLIGHT_PROFILES = ("vn3_sn50_h3",) + VNR_PROFILES
ALL_PROFILES = SUBSTRATE_PROFILES + VNR_PROFILES
DEFAULT_MAX_MEASUREMENT_PTAU_POWER = 23
SOURCE_INPUTS = (
    ROOT / "circuits" / "resource_feasibility.circom",
    ROOT / "benchmarks" / "common" / "experiment_log.py",
    ROOT / "benchmarks" / "zk_overhead" / "preflight.py",
    ROOT / "benchmarks" / "zk_overhead" / "run.py",
    ROOT / "benchmarks" / "zk_overhead" / "scaling_results.py",
    ROOT / "benchmarks" / "zk_overhead" / "scaling_experiment.py",
    ROOT / "benchmarks" / "scripts" / "start_scaling_experiment.sh",
)


def append_status(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_preflight_ptau_power(case_dir: Path, profile: str) -> int:
    state = json.loads((case_dir / "results.json").read_text(encoding="utf-8"))
    result = state.get("profiles", {}).get(profile, {})
    if result.get("status") != "completed":
        raise RuntimeError(f"preflight result is incomplete for {profile}")
    return int(result["required_ptau_power"])


def measurement_allowed(required_power: int, maximum_power: int) -> bool:
    return required_power <= maximum_power


def run_case(
    *,
    phase: str,
    profile: str,
    command: list[str],
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = [
        "timeout",
        "--signal=TERM",
        "--kill-after=60s",
        f"{timeout_seconds}s",
        *command,
    ]
    print(f"[{phase}] starting {profile} (timeout={timeout_seconds}s)", flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            wrapped,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if result.returncode == 0:
        status = "completed"
    elif result.returncode in (124, 137):
        status = "timeout"
    else:
        status = "failed"
    print(f"[{phase}] {profile}: {status} ({elapsed:.1f}s)", flush=True)
    return {
        "phase": phase,
        "profile": profile,
        "status": status,
        "exit_code": result.returncode,
        "elapsed_seconds": elapsed,
        "log_path": str(log_path.relative_to(ROOT)),
        "result_path": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--case-timeout-seconds", type=int, default=10800)
    parser.add_argument(
        "--max-measurement-ptau-power",
        type=int,
        default=DEFAULT_MAX_MEASUREMENT_PTAU_POWER,
        help=(
            "run full setup/proof measurement only when preflight requires at most "
            "this Powers of Tau exponent"
        ),
    )
    parser.add_argument(
        "--preflight-profiles",
        nargs="+",
        default=list(PREFLIGHT_PROFILES),
        help="override only for controlled smoke or recovery runs",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(ALL_PROFILES),
        help="override only for controlled smoke or recovery runs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1 or args.repetitions > 3:
        raise ValueError("repetitions must be between 1 and 3")
    if args.case_timeout_seconds < 1 or args.case_timeout_seconds > 18000:
        raise ValueError("case timeout must be between 1 and 18000 seconds")
    if args.max_measurement_ptau_power < 18:
        raise ValueError("maximum measurement Powers of Tau exponent must be at least 18")

    recorder = ExperimentLog(
        args.run_dir,
        experiment="zkp_scaling_v3",
        metadata={
            "substrate_profiles": list(SUBSTRATE_PROFILES),
            "vnr_profiles": list(VNR_PROFILES),
            "requested_preflight_profiles": list(args.preflight_profiles),
            "requested_profiles": list(args.profiles),
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "case_timeout_seconds": args.case_timeout_seconds,
            "max_measurement_ptau_power": args.max_measurement_ptau_power,
            "local_solver_timed": False,
            "memory_measured": False,
            "source_snapshot": source_manifest(ROOT, list(SOURCE_INPUTS)),
        },
    )
    status_path = recorder.run_dir / "status.csv"
    preflight_outcomes: dict[str, str] = {}
    preflight_failures = 0
    measurement_failures = 0
    resource_limit_skips = 0
    try:
        def perform_preflight(profile: str) -> tuple[bool, int | None]:
            case_dir = recorder.run_dir / "preflight" / profile
            outcome = run_case(
                phase="preflight",
                profile=profile,
                command=[
                    sys.executable,
                    "benchmarks/zk_overhead/preflight.py",
                    "--run-dir",
                    str(case_dir),
                    "--profiles",
                    profile,
                ],
                log_path=recorder.run_dir / "logs" / f"preflight_{profile}.log",
                timeout_seconds=args.case_timeout_seconds,
            )
            append_status(status_path, outcome)
            recorder.update_profile(f"preflight:{profile}", **outcome)
            preflight_outcomes[profile] = outcome["status"]
            if outcome["status"] != "completed":
                return False, None
            return True, read_preflight_ptau_power(case_dir, profile)

        def skip_measurement_for_ptau_limit(profile: str, required_power: int) -> None:
            outcome = {
                "phase": "measurement",
                "profile": profile,
                "status": "skipped_ptau_limit",
                "exit_code": "",
                "elapsed_seconds": 0.0,
                "log_path": "",
                "result_path": "",
            }
            append_status(status_path, outcome)
            recorder.update_profile(
                profile,
                **outcome,
                required_ptau_power=required_power,
                max_measurement_ptau_power=args.max_measurement_ptau_power,
            )

        def perform_measurement(profile: str, preflight_ready: bool) -> bool:
            if not preflight_ready:
                outcome = {
                    "phase": "measurement",
                    "profile": profile,
                    "status": "skipped_preflight_failed",
                    "exit_code": "",
                    "elapsed_seconds": 0.0,
                    "log_path": "",
                    "result_path": "",
                }
                append_status(status_path, outcome)
                recorder.update_profile(profile, **outcome)
                return False

            case_dir = recorder.run_dir / "cases" / profile
            outcome = run_case(
                phase="measurement",
                profile=profile,
                command=[
                    sys.executable,
                    "benchmarks/zk_overhead/run.py",
                    "--run-dir",
                    str(case_dir),
                    "--repetitions",
                    str(args.repetitions),
                    "--warmups",
                    str(args.warmups),
                    "--profiles",
                    profile,
                ],
                log_path=recorder.run_dir / "logs" / f"measurement_{profile}.log",
                timeout_seconds=args.case_timeout_seconds,
            )
            outcome["result_path"] = str(
                (case_dir / "results.json").relative_to(ROOT)
            )
            append_status(status_path, outcome)
            recorder.update_profile(profile, **outcome)
            write_scaling_results(recorder.run_dir)
            return outcome["status"] == "completed"

        base_preflights = [
            profile
            for profile in args.preflight_profiles
            if profile not in VNR_PROFILES
        ]
        substrate_ready = True
        substrate_ptau_power: int | None = None
        for profile in base_preflights:
            succeeded, required_power = perform_preflight(profile)
            substrate_ready = substrate_ready and succeeded
            if required_power is not None:
                substrate_ptau_power = required_power
            if not succeeded:
                preflight_failures += 1

        for profile in args.profiles:
            if profile not in SUBSTRATE_PROFILES:
                continue
            within_limit = (
                substrate_ptau_power is not None
                and measurement_allowed(
                    substrate_ptau_power, args.max_measurement_ptau_power
                )
            )
            if substrate_ready and not within_limit:
                skip_measurement_for_ptau_limit(profile, substrate_ptau_power)
                resource_limit_skips += 1
                continue
            if not perform_measurement(profile, substrate_ready and within_limit):
                if substrate_ready:
                    measurement_failures += 1

        for profile in args.profiles:
            if profile not in VNR_PROFILES:
                continue
            if profile in args.preflight_profiles:
                succeeded, required_power = perform_preflight(profile)
                if not succeeded:
                    preflight_failures += 1
            else:
                succeeded = True
                required_power = args.max_measurement_ptau_power
            if (
                succeeded
                and required_power is not None
                and not measurement_allowed(
                    required_power, args.max_measurement_ptau_power
                )
            ):
                skip_measurement_for_ptau_limit(profile, required_power)
                resource_limit_skips += 1
                continue
            if not perform_measurement(profile, succeeded):
                if succeeded:
                    measurement_failures += 1

        write_scaling_results(recorder.run_dir)
        completed_measurements = sum(
            profile.get("phase") == "measurement" and profile.get("status") == "completed"
            for profile in recorder.state["profiles"].values()
        )
        if completed_measurements == 0:
            recorder.fail("no measurement profile completed")
            return 1
        recorder.state["partial_results"] = bool(
            preflight_failures or measurement_failures or resource_limit_skips
        )
        recorder.state["preflight_failure_count"] = preflight_failures
        recorder.state["measurement_failure_count"] = measurement_failures
        recorder.state["resource_limit_skip_count"] = resource_limit_skips
        recorder.append(
            "sweep_completed",
            completed_measurements=completed_measurements,
            preflight_failures=preflight_failures,
            measurement_failures=measurement_failures,
            resource_limit_skips=resource_limit_skips,
            partial_results=recorder.state["partial_results"],
        )
        recorder.complete()
        return 0
    except Exception as exc:
        recorder.fail(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
