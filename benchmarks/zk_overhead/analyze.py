#!/usr/bin/env python3
"""Validate and summarize a completed ZKP overhead experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


TIMING_FIELDS = (
    "commitment_seconds",
    "witness_seconds",
    "prove_seconds",
    "verify_seconds",
    "zkp_incremental_seconds",
    "online_groth16_seconds",
)


def sample_stats(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    return {
        "min": min(values),
        "mean": mean,
        "median": statistics.median(values),
        "max": max(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "coefficient_of_variation": (
            statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0
        ),
    }


def analyze_results(state: dict[str, Any], journal_events: list[dict[str, Any]]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for profile_id, result in state["profiles"].items():
        measurements = result["measurements"]
        profiles[profile_id] = {
            "profile": result["profile"],
            "constraints": result["r1cs"]["constraints"],
            "measurement_count": len(measurements),
            "verified_count": sum(bool(item["verified"]) for item in measurements),
            "witness_bytes": sorted({int(item["witness_bytes"]) for item in measurements}),
            "proof_bytes": {
                "min": min(int(item["proof_bytes"]) for item in measurements),
                "max": max(int(item["proof_bytes"]) for item in measurements),
            },
            "timing": {
                field: sample_stats([float(item[field]) for item in measurements])
                for field in TIMING_FIELDS
            },
        }

    ordered_ids = list(state["profiles"])
    baseline_id = ordered_ids[0]
    baseline = profiles[baseline_id]
    comparisons: dict[str, Any] = {}
    for profile_id in ordered_ids[1:]:
        candidate = profiles[profile_id]
        comparison: dict[str, Any] = {
            "baseline_profile_id": baseline_id,
            "constraint_increase_percent": (
                candidate["constraints"] / baseline["constraints"] - 1
            )
            * 100,
            "witness_size_increase_percent": (
                candidate["witness_bytes"][0] / baseline["witness_bytes"][0] - 1
            )
            * 100,
            "median_timing_increase_percent": {},
        }
        for field in TIMING_FIELDS:
            base_value = baseline["timing"][field]["median"]
            candidate_value = candidate["timing"][field]["median"]
            comparison["median_timing_increase_percent"][field] = (
                candidate_value / base_value - 1
            ) * 100
        comparisons[profile_id] = comparison

    failed_events = [
        event for event in journal_events if event["event_type"].endswith("failed")
    ]
    return {
        "schema_version": 1,
        "experiment_status": state["status"],
        "baseline_profile_id": baseline_id,
        "all_proofs_verified": all(
            profile["measurement_count"] == profile["verified_count"]
            for profile in profiles.values()
        ),
        "failed_stage_count": len(failed_events),
        "failed_stages": failed_events,
        "profiles": profiles,
        "comparisons_to_baseline": comparisons,
        "interpretation": (
            "Timing comparisons describe the current dense circuit and CLI toolchain. "
            "They do not include local solving, graph generation, or network delay."
        ),
    }


def write_analysis(run_dir: Path, state: dict[str, Any] | None = None) -> Path:
    run_dir = run_dir.resolve()
    if state is None:
        state = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    journal_events = [
        json.loads(line)
        for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    analysis = analyze_results(state, journal_events)
    output = run_dir / "analysis.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = write_analysis(args.run_dir)
    print(f"Analysis: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

