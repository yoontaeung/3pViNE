#!/usr/bin/env python3
"""Measure P1-P5 in the Compose system for a configurable circuit profile."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.common.experiment_log import git_metadata, source_manifest
from benchmarks.common.timing_statistics import timing_statistics
from benchmarks.compose_protocol_cost.configs import (
    compose_portal_request,
    generate_compose_configs,
)
from benchmarks.compose_protocol_cost.run import ROOT, _compose, _run, _run_compose_case
from benchmarks.protocol_cost.scenarios import load_scenario_document, parse_scenarios


SOURCE_INPUTS = (
    ROOT / "benchmarks/compose_protocol_cost/run_end_to_end.py",
    ROOT / "benchmarks/compose_protocol_cost/run.py",
    ROOT / "circuits/resource_feasibility.circom",
    ROOT / "compose.yml",
    ROOT / "docker/Dockerfile.system",
    ROOT / "zk_encoding.py",
)

TIMING_FIELDS = (
    "solve",
    "encoding",
    "witness",
    "prove",
    "verify",
    "artifact_upload",
    "artifact_fetch",
    "state_commit",
    "online_total",
    "trial_wall",
)


def _measurement(scenario: Any, repetition: int, metrics: dict[str, Any]) -> dict[str, Any]:
    timings = metrics["timings_seconds"]
    http = metrics["http_seconds_by_type"]
    return {
        "scenario_id": scenario.scenario_id,
        "repetition": repetition,
        "claim_sequence": ">".join(metrics["claim_sequence"]),
        "proofs": metrics["proofs"],
        "solve": timings.get("solve", 0.0),
        "encoding": timings.get("encoding", 0.0),
        "witness": timings.get("witness", 0.0),
        "prove": timings.get("prove", 0.0),
        "verify": timings.get("verify", 0.0),
        "artifact_upload": http.get("proof_artifact_upload", 0.0),
        "artifact_fetch": http.get("proof_artifact_fetch", 0.0),
        "state_commit": http.get("state_commit_command", 0.0),
        "online_total": timings.get("online_total", 0.0),
        "trial_wall": metrics["trial_wall_seconds"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scenario_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for scenario_id, samples in grouped.items():
        result: dict[str, Any] = {
            "scenario_id": scenario_id,
            "repetitions": len(samples),
            "claim_sequence": samples[0]["claim_sequence"],
            "proofs": samples[0]["proofs"],
        }
        for field in TIMING_FIELDS:
            values = [float(sample[field]) for sample in samples]
            for name, value in timing_statistics(values).items():
                result[f"{field}_{name}"] = value
        output.append(result)
    _write_csv(path, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=ROOT / "benchmarks/config/protocol_scenarios_medium.json",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        raise ValueError("repetitions must be positive and warmups nonnegative")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("use a new empty run directory to avoid mixing measurements")
    document = load_scenario_document(args.scenario_config)
    workload, all_scenarios = parse_scenarios(document)
    selected_ids = args.scenarios or [item.scenario_id for item in all_scenarios]
    by_id = {item.scenario_id: item for item in all_scenarios}
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("scenario IDs must not be repeated")

    circuit = document["circuit"]
    config_set = "benchmarks-medium"
    config_dir = ROOT / "configs/system" / config_set
    generate_compose_configs(
        config_dir,
        scenarios=all_scenarios,
        workload=workload,
    )
    image_tag = f"vn{circuit['vn']}-sn{circuit['sn']}-h{circuit['hops']}"
    compose_env = {
        "ZK_VNE_SYSTEM_IMAGE_TAG": image_tag,
        "ZK_VNE_BENCHMARK_CONFIG_SET": config_set,
        "ZK_VNE_VN_BOUND": str(circuit["vn"]),
        "ZK_VNE_REAL_SN_COUNT": str(circuit["real_sn"]),
        "ZK_VNE_SN_BOUND": str(circuit["sn"]),
        "ZK_VNE_HOP_BOUND": str(circuit["hops"]),
        "ZK_VNE_PTAU_POWER": str(circuit["ptau_power"]),
    }

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "experiment": "compose_end_to_end_medium",
        "command": sys.argv,
        "host": platform.node(),
        "platform": platform.platform(),
        "proof_mode": "groth16",
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "measurement_order": "rotating scenarios within each round",
        "ci_method": "normal approximation, mean +/- 1.959964 * sample SEM",
        "scenario_ids": selected_ids,
        "circuit": circuit,
        "network_delay_policy": "no artificial network delay",
        "trial_wall_scope": "Compose startup, online request, artifact collection, teardown",
        "git": git_metadata(ROOT),
        "source_manifest": source_manifest(
            ROOT,
            [args.scenario_config.resolve(), *SOURCE_INPUTS],
        ),
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not args.skip_build:
        print("Building Table 2 images; offline work is excluded from online latency.", flush=True)
        env = dict(os.environ)
        env.update(compose_env)
        _run(
            [*_compose("zkvne-e2e-build"), "build"],
            env=env,
            log=run_dir / "compose_build.log",
        )

    status_path = run_dir / "status.jsonl"
    rows: list[dict[str, Any]] = []
    for index in range(-args.warmups, args.repetitions):
        offset = max(index, 0) % len(selected_ids)
        for scenario_id in selected_ids[offset:] + selected_ids[:offset]:
            scenario = by_id[scenario_id]
            measured = index >= 0
            repetition = index + 1 if measured else index + args.warmups + 1
            kind = "measured" if measured else "warmup"
            label = f"{scenario_id}-{'rep' if measured else 'warmup'}{repetition}"
            case_dir = run_dir / kind / scenario_id / f"rep_{repetition}"
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "started", "case": label, "time": time.time()}) + "\n")
            print(f"Starting {label}", flush=True)
            trial_started = time.perf_counter()
            try:
                metrics = _run_compose_case(
                    scenario,
                    case_dir,
                    proof_mode="groth16",
                    tag=f"medium-{label}",
                    payload=compose_portal_request(scenario, workload),
                    compose_env=compose_env,
                )
                metrics["trial_wall_seconds"] = time.perf_counter() - trial_started
                (case_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if measured:
                    rows.append(_measurement(scenario, repetition, metrics))
                    _write_csv(run_dir / "measurements.csv", rows)
                    _write_summary(run_dir / "summary.csv", rows)
                event = {
                    "event": "completed", "case": label, "time": time.time(),
                    "trial_wall_seconds": metrics["trial_wall_seconds"],
                    "online_seconds": metrics["timings_seconds"]["online_total"],
                }
            except Exception as exc:
                event = {
                    "event": "failed",
                    "case": label,
                    "time": time.time(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                with status_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                raise
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event), flush=True)

    print(f"Results: {run_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
