"""Aggregate scaling cases and generate min/mean/max tables and figures."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TIMING_FIELDS = ("witness_seconds", "prove_seconds", "verify_seconds")
SERIES_ORDER = {"substrate": 0, "vnr": 1}
COLORS = {"substrate": "#2563A6", "vnr": "#D97706"}
LABELS = {"substrate": "Substrate bound", "vnr": "VNR bound"}


def sample_stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def collect_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases_dir = run_dir / "cases"
    if not cases_dir.exists():
        return rows
    for result_path in cases_dir.glob("*/results.json"):
        state = json.loads(result_path.read_text(encoding="utf-8"))
        if state.get("status") != "completed":
            continue
        for profile_id, result in state["profiles"].items():
            if result.get("status") != "completed" or not result.get("measurements"):
                continue
            profile = result["profile"]
            series = "substrate" if profile["vn"] == 3 else "vnr"
            row: dict[str, Any] = {
                "series": series,
                "profile_id": profile_id,
                "vn": profile["vn"],
                "sn": profile["sn"],
                "hops": profile["hops"],
                "pair_count": profile["pair_count"],
                "constraints": result["r1cs"]["constraints"],
                "repetitions": len(result["measurements"]),
                "all_proofs_verified": all(
                    bool(item["verified"]) for item in result["measurements"]
                ),
                "compile_seconds": result["offline_seconds"]["circom_compile"],
                "setup_seconds": result["offline_seconds"]["groth16_setup"],
            }
            for field in TIMING_FIELDS:
                prefix = field.removesuffix("_seconds")
                stats = sample_stats(
                    [float(item[field]) for item in result["measurements"]]
                )
                for statistic, value in stats.items():
                    row[f"{prefix}_{statistic}_seconds"] = value
            rows.append(row)
    rows.sort(
        key=lambda row: (
            SERIES_ORDER[row["series"]],
            row["sn"] if row["series"] == "substrate" else row["vn"],
        )
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _timing_cell(row: dict[str, Any], stage: str) -> str:
    return " / ".join(
        f"{row[f'{stage}_{statistic}_seconds']:.3f}"
        for statistic in ("min", "mean", "max")
    )


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ZKP Scaling Results",
        "",
        "Timing cells report **min / mean / max** in seconds.",
        "Compilation and Groth16 setup are offline costs.",
    ]
    for series in ("substrate", "vnr"):
        selected = [row for row in rows if row["series"] == series]
        if not selected:
            continue
        lines.extend(
            [
                "",
                f"## {LABELS[series]}",
                "",
                "| Profile | Constraints | Witness (s) | Prove (s) | Verify (s) | Setup (s) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            lines.append(
                f"| `{row['profile_id']}` | {row['constraints']:,} | "
                f"{_timing_cell(row, 'witness')} | {_timing_cell(row, 'prove')} | "
                f"{_timing_cell(row, 'verify')} | {row['setup_seconds']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _series_xy(rows: list[dict[str, Any]], series: str) -> tuple[list[float], list[dict[str, Any]]]:
    selected = [row for row in rows if row["series"] == series]
    x = [row["sn"] if series == "substrate" else row["vn"] for row in selected]
    return x, selected


def _save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_scaling(rows: list[dict[str, Any]], output_dir: Path) -> None:
    _configure_matplotlib()
    available_series = tuple(
        series
        for series in ("substrate", "vnr")
        if any(row["series"] == series for row in rows)
    )
    fig, axes = plt.subplots(
        2,
        len(available_series),
        figsize=(3.8 * len(available_series), 5.0),
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )
    for column, series in enumerate(available_series):
        x, selected = _series_xy(rows, series)
        constraints = [row["constraints"] / 1_000_000 for row in selected]
        means = [row["prove_mean_seconds"] for row in selected]
        lower = [
            row["prove_mean_seconds"] - row["prove_min_seconds"]
            for row in selected
        ]
        upper = [
            row["prove_max_seconds"] - row["prove_mean_seconds"]
            for row in selected
        ]
        color = COLORS[series]
        axes[0, column].plot(x, constraints, marker="o", color=color, linewidth=1.8)
        axes[1, column].errorbar(
            x,
            means,
            yerr=[lower, upper],
            marker="o",
            color=color,
            linewidth=1.8,
            capsize=3,
        )
        bound_label = "SN bound" if series == "substrate" else "VN bound"
        axes[0, column].set_title(LABELS[series], fontweight="bold")
        axes[1, column].set_xlabel(bound_label)
        for axis in axes[:, column]:
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axes[0, 0].set_ylabel("R1CS constraints (millions)")
    axes[1, 0].set_ylabel("Groth16 proving time (s)")
    fig.suptitle("Fixed-bound ZK-VNE scaling", fontweight="bold")
    _save_figure(fig, output_dir, "scaling_by_bound")

    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    for series in ("substrate", "vnr"):
        _, selected = _series_xy(rows, series)
        if not selected:
            continue
        x = [row["constraints"] / 1_000_000 for row in selected]
        y = [row["prove_mean_seconds"] for row in selected]
        lower = [row["prove_mean_seconds"] - row["prove_min_seconds"] for row in selected]
        upper = [row["prove_max_seconds"] - row["prove_mean_seconds"] for row in selected]
        ax.errorbar(
            x,
            y,
            yerr=[lower, upper],
            marker="o",
            color=COLORS[series],
            linewidth=1.8,
            capsize=3,
            label=LABELS[series],
        )
    ax.set_xlabel("R1CS constraints (millions)")
    ax.set_ylabel("Groth16 proving time (s)")
    ax.set_title("Constraint count and proving time", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, "constraints_vs_proving_time")


def write_scaling_results(run_dir: Path) -> list[dict[str, Any]]:
    rows = collect_rows(run_dir)
    if not rows:
        return rows
    _write_csv(run_dir / "summary.csv", rows)
    _write_markdown(run_dir / "summary.md", rows)
    analysis = {
        "schema_version": 1,
        "completed_profile_count": len(rows),
        "all_proofs_verified": all(row["all_proofs_verified"] for row in rows),
        "profiles": rows,
    }
    temporary = (run_dir / "analysis.json").with_suffix(".json.tmp")
    temporary.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(run_dir / "analysis.json")
    plot_scaling(rows, run_dir / "figures")
    return rows
