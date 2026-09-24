#!/usr/bin/env python3
"""Aggregate the score-free PIS/DIS GMM grid with the ASBS controller."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
from typing import Any


PATTERN = re.compile(
    r"gmm_(mode|dim)_(k(?:5|10|20|30|40)|d(?:10|25|50|100))_"
    r"(pis|dis)_asbs_control_no_score_seed([012])"
)
EXPECTED_VALUES = {"mode": {5, 10, 20, 30, 40}, "dim": {10, 25, 50, 100}}
EXPECTED_SIGMA = {("mode", "dis"): 2, ("dim", "dis"): 4,
                  ("mode", "pis"): 128, ("dim", "pis"): 32}
SELECTIONS = {
    "final": "final",
    "best_loss_SD": "min_online_sd",
    "best_eval_raw_path_ess": "max_online_raw_ess",
}
METRICS = [
    "loss/SD",
    "loss/tvd_soft",
    "loss/W_2",
    "loss/Sliced_W_2",
    "eval/raw_path_ess",
    "logZ_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.5g}"


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, int]] = set()

    for run_directory in sorted(run_root.iterdir()):
        match = PATTERN.fullmatch(run_directory.name)
        if match is None:
            continue
        suite, label, method, seed_text = match.groups()
        value = int(label[1:])
        seed = int(seed_text)
        key = (suite, value, method, seed)
        if key in seen:
            raise RuntimeError(f"Duplicate run: {run_directory}")
        seen.add(key)

        path = run_directory / "checkpoints/checkpoint_evaluations.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text())
        if report.get("algorithm") != method or int(report.get("seed", -1)) != seed:
            raise RuntimeError(f"Run metadata mismatch: {path}")
        if report.get("score_hint_mode") != "none" or report.get("uses_score_hint") is not False:
            raise RuntimeError(f"Score hint was not disabled: {path}")
        if not str(report.get("control_class", "")).endswith(".ASBSFourierControl"):
            raise RuntimeError(f"ASBS controller metadata mismatch: {path}")

        for source_selection, selection in SELECTIONS.items():
            record = report.get("selections", {}).get(source_selection)
            if record is None:
                raise RuntimeError(f"Missing {source_selection}: {path}")
            if int(record.get("num_samples", -1)) != 4000:
                raise RuntimeError(f"Final sample count mismatch: {path}")
            if int(record.get("report_seed", -1)) != 12345:
                raise RuntimeError(f"Report seed mismatch: {path}")
            metrics = record.get("metrics", {})
            row: dict[str, Any] = {
                "suite": suite,
                "grid_value": value,
                "method": method,
                "sigma_max": EXPECTED_SIGMA[(suite, method)],
                "seed": seed,
                "selection": selection,
                "selected_iteration": int(record["selected_iteration"]),
                "final_iteration": int(report["final_iteration"]),
                "early_stopped": bool(report.get("early_stopped", False)),
                "control_parameters": int(report["control_parameters"]),
                "report_path": str(path),
            }
            for metric in METRICS:
                if metric not in metrics:
                    raise RuntimeError(f"Missing {metric}: {path}")
                row[metric] = float(metrics[metric])
            rows.append(row)

    expected = {
        (suite, value, method, seed)
        for suite, values in EXPECTED_VALUES.items()
        for value in values
        for method in ("dis", "pis")
        for seed in (0, 1, 2)
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise RuntimeError(f"Incomplete grid: missing={missing}, extra={extra}")
    if len(rows) != 162:
        raise RuntimeError(f"Expected 162 checkpoint rows, got {len(rows)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["suite"], row["grid_value"], row["method"], row["seed"], row["selection"]))
    checkpoint_csv = output_dir / "checkpoint_metrics.csv"
    with checkpoint_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, Any]] = []
    for suite, values in EXPECTED_VALUES.items():
        for value in sorted(values):
            for method in ("dis", "pis"):
                for selection in SELECTIONS.values():
                    group = [
                        row for row in rows
                        if row["suite"] == suite
                        and row["grid_value"] == value
                        and row["method"] == method
                        and row["selection"] == selection
                    ]
                    aggregate: dict[str, Any] = {
                        "suite": suite,
                        "grid_value": value,
                        "method": method,
                        "sigma_max": EXPECTED_SIGMA[(suite, method)],
                        "selection": selection,
                        "n_seeds": len(group),
                        "early_stop_count": sum(bool(row["early_stopped"]) for row in group),
                        "control_parameters": group[0]["control_parameters"],
                    }
                    for metric in METRICS:
                        values_for_metric = [float(row[metric]) for row in group]
                        aggregate[f"{metric}_mean"] = statistics.mean(values_for_metric)
                        aggregate[f"{metric}_sd"] = statistics.stdev(values_for_metric)
                    aggregate_rows.append(aggregate)

    aggregate_csv = output_dir / "aggregate_metrics.csv"
    with aggregate_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    lines = [
        "# Score-free PIS/DIS with the ASBS FourierMLP controller",
        "",
        "All entries use fixed target seed 0, model seeds 0/1/2, 128 transitions, and independent 4,000-sample checkpoint evaluation with report seed 12345.",
        "",
        "| suite | value | method | sigma | checkpoint | SD | soft TVD | W2 | raw ESS | stopped |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['suite']} | {row['grid_value']} | {row['method'].upper()} | "
            f"{row['sigma_max']} | {row['selection']} | "
            f"{fmt(row['loss/SD_mean'])}±{fmt(row['loss/SD_sd'])} | "
            f"{fmt(row['loss/tvd_soft_mean'])}±{fmt(row['loss/tvd_soft_sd'])} | "
            f"{fmt(row['loss/W_2_mean'])}±{fmt(row['loss/W_2_sd'])} | "
            f"{fmt(row['eval/raw_path_ess_mean'])}±{fmt(row['eval/raw_path_ess_sd'])} | "
            f"{row['early_stop_count']}/3 |"
        )
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(
        f"checkpoint_rows={len(rows)} aggregate_rows={len(aggregate_rows)} "
        f"report={report_path}"
    )


if __name__ == "__main__":
    main()
