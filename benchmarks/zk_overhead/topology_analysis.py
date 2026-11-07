"""Summarize repeated topology-timing measurements as compact tables."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any
from benchmarks.common.timing_statistics import timing_statistics


TIMING_FIELDS = ("witness_seconds", "prove_seconds", "verify_seconds")
TOPOLOGY_LABELS = {
    "full_mesh": "Full mesh",
    "ring": "Ring",
    "waxman": "Waxman",
    "erdos_renyi": "Erdos-Renyi",
    "barabasi_albert": "Barabasi-Albert",
}


def sample_stats(values: list[float]) -> dict[str, float]:
    return timing_statistics(values)


def summarize_topologies(state: dict[str, Any]) -> dict[str, Any]:
    profile = next(iter(state["profiles"].values()))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for measurement in profile["measurements"]:
        grouped.setdefault(measurement["topology"], []).append(measurement)

    summaries: dict[str, Any] = {}
    for topology, measurements in grouped.items():
        summaries[topology] = {
            "measurement_count": len(measurements),
            "verified_count": sum(bool(item["verified"]) for item in measurements),
            "edge_count": measurements[0]["edge_count"],
            "density": measurements[0]["density"],
            "timing": {
                field: sample_stats([float(item[field]) for item in measurements])
                for field in TIMING_FIELDS
            },
        }

    return {
        "schema_version": 2,
        "ci_method": "normal approximation, mean +/- 1.959964 * sample SEM",
        "experiment": "same_vnr_different_sn_topology",
        "profile": profile["profile"],
        "constraints": profile["r1cs"]["constraints"],
        "all_proofs_verified": all(
            item["measurement_count"] == item["verified_count"]
            for item in summaries.values()
        ),
        "topologies": summaries,
    }


def _summary_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topology, summary in analysis["topologies"].items():
        row: dict[str, Any] = {
            "topology": topology,
            "sn": analysis["profile"]["sn"],
            "edge_count": summary["edge_count"],
            "density": summary["density"],
            "constraints": analysis["constraints"],
            "repetitions": summary["measurement_count"],
        }
        for field in TIMING_FIELDS:
            prefix = field.removesuffix("_seconds")
            timing = summary["timing"][field]
            row.update(
                {
                    f"{prefix}_min_seconds": timing["min"],
                    f"{prefix}_mean_seconds": timing["mean"],
                    f"{prefix}_max_seconds": timing["max"],
                }
            )
            for statistic in ("variance", "stddev", "sem", "ci95_low", "ci95_high"):
                row[f"{prefix}_{statistic}"] = timing[statistic]
        rows.append(row)
    return rows


def _write_markdown_table(path: Path, analysis: dict[str, Any]) -> None:
    profile = analysis["profile"]
    lines = [
        "# Same VNR, Different Private SN Topologies",
        "",
        f"Circuit profile: `VN={profile['vn']}, SN={profile['sn']}, H={profile['hops']}`  ",
        f"R1CS constraints: `{analysis['constraints']:,}`  ",
        "Each timing cell reports **min / mean / max** in seconds.",
        "",
        "| Topology | Edges | Density | Witness (s) | Prove (s) | Verify (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for topology, summary in analysis["topologies"].items():
        cells = []
        for field in TIMING_FIELDS:
            timing = summary["timing"][field]
            cells.append(
                f"{timing['min']:.3f} / {timing['mean']:.3f} / {timing['max']:.3f}"
            )
        lines.append(
            f"| {TOPOLOGY_LABELS[topology]} | {summary['edge_count']} | "
            f"{summary['density']:.3f} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_topology_analysis(
    run_dir: Path,
    state: dict[str, Any],
) -> tuple[Path, Path, Path]:
    analysis = summarize_topologies(state)
    json_path = run_dir / "analysis.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(json_path)

    rows = _summary_rows(analysis)
    csv_path = run_dir / "summary.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    markdown_path = run_dir / "summary.md"
    temporary_markdown = markdown_path.with_suffix(".md.tmp")
    _write_markdown_table(temporary_markdown, analysis)
    temporary_markdown.replace(markdown_path)
    return json_path, csv_path, markdown_path
