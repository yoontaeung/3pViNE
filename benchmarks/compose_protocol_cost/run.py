#!/usr/bin/env python3
"""Run matched ZK-VNE Compose and PolyViNE protocol-cost scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.compose_protocol_cost.analyze import analyze_compose_run
from benchmarks.compose_protocol_cost.configs import (
    compose_portal_request,
    generate_compose_configs,
)
from benchmarks.protocol_cost.analyze import analyze_polyvine_trace
from benchmarks.protocol_cost.matched_inputs import write_matched_inputs
from benchmarks.protocol_cost.scenarios import SCENARIOS
from benchmarks.common.experiment_log import git_metadata, source_manifest
from system.http_trace import HttpTraceRecorder, canonical_json_bytes


BASE_COMPOSE = ROOT / "compose.yml"
EXPERIMENT_COMPOSE = ROOT / "benchmarks" / "compose_protocol_cost" / "compose.yml"
SOURCE_INPUTS = (
    ROOT / "benchmarks/config/protocol_scenarios.json",
    ROOT / "benchmarks/compose_protocol_cost/analyze.py",
    ROOT / "benchmarks/compose_protocol_cost/configs.py",
    ROOT / "benchmarks/compose_protocol_cost/run.py",
    ROOT / "benchmarks/compose_protocol_cost/compose.yml",
    ROOT / "compose.yml",
    ROOT / "circuits/resource_feasibility.circom",
)


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    log: Path | None = None,
) -> None:
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(command)}", flush=True)
    context = log.open("w", encoding="utf-8") if log is not None else nullcontext()
    with context as handle:
        with subprocess.Popen(
            command, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                if handle is not None:
                    handle.write(line)
                    handle.flush()
            returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _compose(project: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(BASE_COMPOSE),
        "-f",
        str(EXPERIMENT_COMPOSE),
        "-p",
        project,
    ]


def _copy_from_service(
    command: list[str],
    service: str,
    source: str,
    target: Path,
    *,
    env: dict[str, str] | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [*command, "cp", f"{service}:{source}", str(target)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 and "Could not find" not in result.stderr:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            result.stdout,
            result.stderr,
        )


def _run_compose_case(
    scenario: Any,
    run_dir: Path,
    *,
    proof_mode: str,
    tag: str,
    payload: dict[str, Any] | None = None,
    compose_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    project = f"zkvne-pc-{tag}".lower().replace("_", "-")[:55]
    portal_port = _free_port()
    log_port = _free_port()
    env = dict(os.environ)
    if compose_env is not None:
        env.update(compose_env)
    env.update(
        {
            "ZK_VNE_SCENARIO_ID": scenario.scenario_id,
            "ZK_VNE_PROOF_MODE": proof_mode,
            "ZK_VNE_PORTAL_PORT": str(portal_port),
            "ZK_VNE_LOG_PORT": str(log_port),
        }
    )
    command = _compose(project)
    traces = run_dir / "http_traces"
    traces.mkdir(parents=True, exist_ok=True)
    try:
        _run(
            [*command, "up", "-d", "--wait", "--no-build"],
            env=env,
            log=run_dir / "compose_up.log",
        )
        payload = payload or compose_portal_request(scenario)
        recorder = HttpTraceRecorder(traces / "user.jsonl", actor_id="user")
        started_ns = time.time_ns()
        started = time.perf_counter()
        response = httpx.post(
            f"http://127.0.0.1:{portal_port}/v1/requests",
            content=canonical_json_bytes(payload),
            headers={"Content-Type": "application/json"},
            timeout=3600.0,
        )
        online_seconds = time.perf_counter() - started
        recorder.record(
            receiver="portal",
            semantic_type="user_request",
            method="POST",
            path="/v1/requests",
            request_payload=payload,
            request_headers={"Content-Type": "application/json"},
            response=response,
            elapsed_seconds=online_seconds,
            request_id=scenario.request_id,
        )
        finished_ns = time.time_ns()
        if response.is_error:
            (run_dir / "http_error.json").write_text(
                json.dumps(response.json(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        response.raise_for_status()
        result = response.json()
        (run_dir / "portal_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        for service, name in (
            ("portal", "portal.jsonl"),
            ("facility-a", "facility-a.jsonl"),
            ("facility-b", "facility-b.jsonl"),
        ):
            _copy_from_service(
                command,
                service,
                "/data/http_trace.jsonl",
                traces / name,
                env=env,
            )
        public_dir = run_dir / "public_log"
        public_dir.mkdir(parents=True, exist_ok=True)
        _copy_from_service(
            command,
            "transparency-log",
            "/data/.",
            public_dir,
            env=env,
        )
        metrics = analyze_compose_run(
            run_dir,
            request_id=scenario.request_id,
            started_unix_ns=started_ns,
            finished_unix_ns=finished_ns,
            online_seconds=online_seconds,
        )
        (run_dir / "measurement_window.json").write_text(
            json.dumps(
                {
                    "request_id": scenario.request_id,
                    "started_unix_ns": started_ns,
                    "finished_unix_ns": finished_ns,
                    "online_seconds": online_seconds,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if metrics["outcome"] != scenario.expected_outcome:
            raise RuntimeError(f"unexpected Compose outcome: {metrics['outcome']}")
        if tuple(metrics["claim_sequence"]) != scenario.expected_claims:
            raise RuntimeError(f"unexpected Compose claims: {metrics['claim_sequence']}")
        return metrics
    finally:
        logs = subprocess.run(
            [*command, "logs", "--no-color"], cwd=ROOT, env=env, text=True,
            capture_output=True, check=False,
        )
        (run_dir / "compose.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
        subprocess.run(
            [*command, "down", "-v", "--remove-orphans"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )


def _container_path(path: Path) -> str:
    return "/workspace/" + path.resolve().relative_to(ROOT).as_posix()


def _run_polyvine_case(
    scenario: Any,
    run_dir: Path,
    *,
    image: str,
) -> dict[str, Any]:
    inputs = write_matched_inputs(scenario, run_dir / "inputs")
    stats = run_dir / "stats.out"
    trace = run_dir / "protocol_trace.jsonl"
    command = [
        "docker", "run", "--rm", "--network", "none",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{ROOT}:/workspace", "-w", "/workspace", image,
        "--multi_sub_dir", _container_path(inputs["polyvine_subs"]),
        "--vnr", _container_path(inputs["polyvine_vnrs"]),
        "--num_inps", "2", "--num_vnr", "1",
        "--polyvine_k_sp", str(scenario.polyvine_initial_fanout),
        "--polyvine_k_inp", "1", "--polyvine_max_hops", "2",
        "--polyvine_forwarding", "nearest", "--polyvine_backend", "simulated",
        "--random_seed", "0", "--stats_output", _container_path(stats),
        "--trace_output", _container_path(trace),
    ]
    _run(command, log=run_dir / "polyvine.log")
    metrics = analyze_polyvine_trace(trace, stats)
    if metrics["outcome"] != scenario.expected_outcome:
        raise RuntimeError(f"unexpected PolyViNE outcome: {metrics['outcome']}")
    return metrics


def _summary_row(
    scenario: Any,
    repetition: int,
    zk: dict[str, Any],
    poly: dict[str, Any],
) -> dict[str, Any]:
    timings = zk["timings_seconds"]
    http_timings = zk["http_seconds_by_type"]
    return {
        "scenario_id": scenario.scenario_id,
        "repetition": repetition,
        "outcome": scenario.expected_outcome,
        "claim_sequence": ">".join(zk["claim_sequence"]),
        "zk_http_exchanges": zk["physical_http_exchanges"],
        "zk_measured_http_bytes": zk["measured_http_bytes"],
        "zk_coordination_bytes": zk["bytes_by_category"].get("coordination", 0),
        "zk_integrity_bytes": zk["bytes_by_category"].get("integrity", 0),
        "zk_proof_artifact_bytes": zk["proof_artifact_storage_bytes"],
        "zk_log_bytes": zk["transparency_log_request_bytes"],
        "zk_solve_seconds": timings.get("solve", 0.0),
        "zk_encode_seconds": timings.get("encoding", 0.0),
        "zk_witness_seconds": timings.get("witness", 0.0),
        "zk_prove_seconds": timings.get("prove", 0.0),
        "zk_verify_seconds": timings.get("verify", 0.0),
        "zk_artifact_upload_seconds": http_timings.get("proof_artifact_upload", 0.0),
        "zk_artifact_fetch_seconds": http_timings.get("proof_artifact_fetch", 0.0),
        "zk_commit_seconds": http_timings.get("state_commit_command", 0.0),
        "zk_online_seconds": timings.get("online_total", 0.0),
        "polyvine_messages": poly["logical_messages"],
        "polyvine_bytes": poly["canonical_control_plane_bytes"],
        "polyvine_rounds": poly["forwarding_rounds_native"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate(path: Path, rows: list[dict[str, Any]]) -> None:
    numeric_fields = [
        "zk_http_exchanges",
        "zk_measured_http_bytes",
        "zk_coordination_bytes",
        "zk_integrity_bytes",
        "zk_proof_artifact_bytes",
        "zk_log_bytes",
        "zk_solve_seconds",
        "zk_encode_seconds",
        "zk_witness_seconds",
        "zk_prove_seconds",
        "zk_verify_seconds",
        "zk_artifact_upload_seconds",
        "zk_artifact_fetch_seconds",
        "zk_commit_seconds",
        "zk_online_seconds",
        "polyvine_messages",
        "polyvine_bytes",
        "polyvine_rounds",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scenario_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for scenario_id, scenario_rows in grouped.items():
        result: dict[str, Any] = {
            "scenario_id": scenario_id,
            "repetitions": len(scenario_rows),
        }
        for field in numeric_fields:
            values = [float(row[field]) for row in scenario_rows]
            result[f"{field}_min"] = min(values)
            result[f"{field}_mean"] = statistics.fmean(values)
            result[f"{field}_max"] = max(values)
        output.append(result)
    _write_csv(path, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--proof-mode",
        choices=("groth16", "reference"),
        default="groth16",
    )
    parser.add_argument("--polyvine-image", default="zk-vne-polyvine:protocol-cost")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=[item.scenario_id for item in SCENARIOS])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    generate_compose_configs()
    by_id = {item.scenario_id: item for item in SCENARIOS}
    unknown = sorted(set(args.scenarios) - set(by_id))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")

    (run_dir / "experiment.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "compose_protocol_cost_vs_polyvine",
                "command": sys.argv,
                "host": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "proof_mode": args.proof_mode,
                "repetitions": args.repetitions,
                "warmups": args.warmups,
                "scenario_ids": args.scenarios,
                "polyvine_image": args.polyvine_image,
                "git": git_metadata(ROOT),
                "source_manifest": source_manifest(ROOT, list(SOURCE_INPUTS)),
                "network_delay_policy": "no artificial network delay",
                "timing_comparison_policy": (
                    "PolyViNE wall time is diagnostic only; cross-system runtime "
                    "ratios are not reported"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    status_path = run_dir / "status.jsonl"
    rows: list[dict[str, Any]] = []
    for scenario_id in args.scenarios:
        scenario = by_id[scenario_id]
        for index in range(-args.warmups, args.repetitions):
            measured = index >= 0
            repetition = index + 1 if measured else 0
            label = f"{scenario_id}-{'rep' if measured else 'warmup'}{repetition}"
            case_dir = (
                run_dir
                / ("measured" if measured else "warmup")
                / scenario_id
                / f"rep_{repetition}"
            )
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"event": "started", "case": label, "time": time.time()}
                    )
                    + "\n"
                )
            try:
                zk = _run_compose_case(
                    scenario,
                    case_dir / "zk_vne",
                    proof_mode=args.proof_mode,
                    tag=label,
                )
                poly = _run_polyvine_case(
                    scenario,
                    case_dir / "polyvine",
                    image=args.polyvine_image,
                )
                (case_dir / "zk_vne_metrics.json").write_text(
                    json.dumps(zk, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (case_dir / "polyvine_metrics.json").write_text(
                    json.dumps(poly, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if measured:
                    rows.append(_summary_row(scenario, repetition, zk, poly))
                    _write_csv(run_dir / "summary.csv", rows)
                    _write_aggregate(run_dir / "summary_stats.csv", rows)
                event = {"event": "completed", "case": label, "time": time.time()}
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
    print(f"Results: {run_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
