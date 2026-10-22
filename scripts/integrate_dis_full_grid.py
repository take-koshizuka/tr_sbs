#!/usr/bin/env python3
"""Add completed DIS GMM sweeps to the existing ASBS aggregate tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


SELECTIONS = {
    "final": "final",
    "best_loss_SD": "min_online_sd",
    "best_eval_raw_path_ess": "max_online_raw_ess",
}
SELECTION_ORDER = ["final", "min_online_sd", "max_online_raw_ess"]
METRICS = {
    "sd": "loss/SD",
    "tvd_soft": "loss/tvd_soft",
    "w2": "loss/W_2",
    "sliced_w2": "loss/Sliced_W_2",
    "rbf_mmd": "loss/RBF_MMD",
    "energy_w2": "loss/energy_W_2",
    "neg_avg_energy": "loss/neg_avg_energy",
    "raw_path_ess": "eval/raw_path_ess",
}
RUN_PATTERNS = {
    "mode": re.compile(r"gmm_mode_k(5|10|20|30|40)_dis_seed([012])"),
    "dim": re.compile(r"gmm_dim_d(10|25|50|100)_dis_seed([012])"),
}
EXPECTED_VALUES = {
    "mode": [5, 10, 20, 30, 40],
    "dim": [10, 25, 50, 100],
}
VALUE_COLUMNS = {"mode": "num_components", "dim": "dimension"}
METHOD_ORDER = {
    "asbs": 0,
    "gamma_asbs": 1,
    "gamma_asbs_iw_fixed": 2,
    "gamma_asbs_iw_trust_region_0p1": 3,
    "gamma_asbs_iw_trust_region_0p3": 3,
    "dis": 4,
}
METHOD_LABELS = {
    "asbs": "ASBS",
    "gamma_asbs": "Gamma-ASBS",
    "gamma_asbs_iw_fixed": "IW fixed",
    "gamma_asbs_iw_trust_region_0p1": "IW TR",
    "gamma_asbs_iw_trust_region_0p3": "IW TR",
    "dis": "DIS",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode-base-csv", type=Path, required=True)
    parser.add_argument("--dim-base-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _finite_float(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value for {context}: {value!r}")
    return result


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty value list.")
    return statistics.fmean(values), statistics.stdev(values) if len(
        values
    ) > 1 else 0.0


def _read_dis_runs(run_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {"mode": [], "dim": []}
    seen: set[tuple[str, int, int]] = set()
    for run_directory in sorted(run_root.resolve().iterdir()):
        if not run_directory.is_dir():
            continue
        matched_suite: str | None = None
        match: re.Match[str] | None = None
        for suite, pattern in RUN_PATTERNS.items():
            candidate = pattern.fullmatch(run_directory.name)
            if candidate is not None:
                matched_suite = suite
                match = candidate
                break
        if matched_suite is None or match is None:
            continue

        value, seed = (int(item) for item in match.groups())
        key = (matched_suite, value, seed)
        if key in seen:
            raise RuntimeError(f"Duplicate DIS run: {key}")
        seen.add(key)

        report_path = run_directory / "checkpoints" / "checkpoint_evaluations.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text())
        if report.get("algorithm") != "dis" or int(report.get("seed", -1)) != seed:
            raise RuntimeError(f"Run metadata mismatch in {report_path}")
        selections = report.get("selections", {})
        if not set(SELECTIONS).issubset(selections):
            missing = sorted(set(SELECTIONS) - set(selections))
            raise RuntimeError(f"Missing selections {missing} in {report_path}")

        for source_selection, table_selection in SELECTIONS.items():
            record = selections[source_selection]
            if int(record.get("num_samples", 0)) != 4000:
                raise RuntimeError(f"Expected 4000 report samples in {report_path}")
            if int(record.get("report_seed", -1)) != 12345:
                raise RuntimeError(f"Expected report_seed=12345 in {report_path}")
            metric_values = record.get("metrics", {})
            output = {
                "suite": matched_suite,
                "value": value,
                "seed": seed,
                "selection": table_selection,
                "selected_epoch": _finite_float(
                    record["selected_iteration"],
                    f"{run_directory.name} selected_iteration",
                ),
                "online_selection_value": (
                    None
                    if record.get("online_selection_value") is None
                    else _finite_float(
                        record["online_selection_value"],
                        f"{run_directory.name} online selection",
                    )
                ),
                "early_stopped": bool(report.get("early_stopped", False)),
                "final_iteration": int(report["final_iteration"]),
                "report_path": str(report_path),
            }
            for output_name, metric_name in METRICS.items():
                if metric_name not in metric_values:
                    raise RuntimeError(f"Missing metric {metric_name} in {report_path}")
                output[output_name] = _finite_float(
                    metric_values[metric_name],
                    f"{run_directory.name} {metric_name}",
                )
            rows[matched_suite].append(output)

    for suite, expected_values in EXPECTED_VALUES.items():
        expected = {
            (value, seed, selection)
            for value in expected_values
            for seed in range(3)
            for selection in SELECTION_ORDER
        }
        actual = {(row["value"], row["seed"], row["selection"]) for row in rows[suite]}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                f"Incomplete {suite} DIS grid: missing={missing}, extra={extra}"
            )
    return rows


def _aggregate_dis_rows(
    suite: str,
    raw_rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> list[dict[str, Any]]:
    value_column = VALUE_COLUMNS[suite]
    aggregated: list[dict[str, Any]] = []
    for value in EXPECTED_VALUES[suite]:
        run_rows = [row for row in raw_rows if row["value"] == value]
        early_stop_n = len(
            {row["seed"] for row in run_rows if bool(row["early_stopped"])}
        )
        for selection in SELECTION_ORDER:
            selected = [row for row in run_rows if row["selection"] == selection]
            if {row["seed"] for row in selected} != {0, 1, 2}:
                raise RuntimeError(
                    f"Expected seeds 0,1,2 for {suite}={value} {selection}"
                )
            row: dict[str, Any] = {field: None for field in fieldnames}
            row.update(
                {
                    value_column: value,
                    "method": "dis",
                    "selection": selection,
                    "n_seeds": 3,
                    "early_stop_n": early_stop_n,
                }
            )
            selected_mean, selected_sd = _mean_sd(
                [item["selected_epoch"] for item in selected]
            )
            row["selected_epoch_mean"] = selected_mean
            row["selected_epoch_sample_sd"] = selected_sd

            online_values = [
                item["online_selection_value"]
                for item in selected
                if item["online_selection_value"] is not None
            ]
            if online_values:
                if len(online_values) != 3:
                    raise RuntimeError(
                        f"Partial online selection values for {suite}={value} {selection}"
                    )
                online_mean, online_sd = _mean_sd(online_values)
                row["online_selection_value_mean"] = online_mean
                row["online_selection_value_sample_sd"] = online_sd

            for metric_name in METRICS:
                mean, sample_sd = _mean_sd([item[metric_name] for item in selected])
                row[f"{metric_name}_mean"] = mean
                row[f"{metric_name}_sample_sd"] = sample_sd
            aggregated.append(row)
    return aggregated


def _read_base_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.resolve().open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _format_number(value: Any, *, scientific: bool = False) -> str:
    if value in (None, ""):
        return ""
    number = float(value)
    if scientific:
        return f"{number:.3e}"
    if abs(number) >= 1000:
        return f"{number:.3e}"
    return f"{number:.3f}"


def _format_pair(row: dict[str, Any], prefix: str, *, scientific: bool = False) -> str:
    mean = _format_number(row.get(f"{prefix}_mean"), scientific=scientific)
    sample_sd = _format_number(
        row.get(f"{prefix}_sample_sd"),
        scientific=scientific,
    )
    return f"{mean} ± {sample_sd}"


def _write_report(
    path: Path,
    *,
    suite: str,
    source_csv: Path,
    rows: list[dict[str, Any]],
) -> None:
    value_column = VALUE_COLUMNS[suite]
    value_label = "modes" if suite == "mode" else "dimension"
    dis_early_stops = sum(
        int(row["early_stop_n"])
        for row in rows
        if row["method"] == "dis" and row["selection"] == "final"
    )
    lines = [
        f"# GMM {suite} full-grid aggregate: ASBS family + DIS",
        "",
        f"- Existing ASBS-family rows were copied from `{source_csv}`.",
        "- DIS uses model seeds 0/1/2 with `potential.seed=0` fixed.",
        f"- DIS sigma_max: `{2 if suite == 'mode' else 4}`.",
        f"- DIS early stops: {dis_early_stops}/{len(EXPECTED_VALUES[suite]) * 3} runs.",
        "- All displayed checkpoint metrics are independent 4,000-sample reevaluations with `report_seed=12345`.",
        "",
        "Use minimum-online-SD as the primary checkpoint, max-online-raw-ESS as a supplementary view, and final as a stability diagnostic. Mean ± sample SD is across three model seeds.",
        "",
        "The retained compatibility column `selected_epoch` has method-specific units: ASBS-family training epoch for existing rows and optimizer iteration for DIS.",
        "",
    ]
    headings = {
        "min_online_sd": "Primary: minimum-online-SD checkpoint",
        "max_online_raw_ess": "Supplementary: maximum-online-raw-ESS checkpoint",
        "final": "Diagnostic: final checkpoint",
    }
    display_order = ["min_online_sd", "max_online_raw_ess", "final"]
    for selection in display_order:
        lines.extend(
            [
                f"## {headings[selection]}",
                "",
                f"| {value_label} | method | stopped | selected index | SD | TVD-soft | W2 | Sliced-W2 | RBF-MMD | energy-W2 | -avg energy | raw ESS |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected_rows = [row for row in rows if row["selection"] == selection]
        selected_rows.sort(
            key=lambda row: (
                int(row[value_column]),
                METHOD_ORDER.get(str(row["method"]), 99),
            )
        )
        for row in selected_rows:
            lines.append(
                f"| {row[value_column]} | {METHOD_LABELS.get(str(row['method']), row['method'])} | "
                f"{row['early_stop_n']}/{row['n_seeds']} | "
                f"{_format_pair(row, 'selected_epoch')} | "
                f"{_format_pair(row, 'sd')} | "
                f"{_format_pair(row, 'tvd_soft')} | "
                f"{_format_pair(row, 'w2')} | "
                f"{_format_pair(row, 'sliced_w2')} | "
                f"{_format_pair(row, 'rbf_mmd')} | "
                f"{_format_pair(row, 'energy_w2')} | "
                f"{_format_pair(row, 'neg_avg_energy')} | "
                f"{_format_pair(row, 'raw_path_ess', scientific=True)} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    run_rows = _read_dis_runs(args.run_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for suite, base_path in (
        ("mode", args.mode_base_csv),
        ("dim", args.dim_base_csv),
    ):
        fieldnames, base_rows = _read_base_csv(base_path)
        expected_first = VALUE_COLUMNS[suite]
        if not fieldnames or fieldnames[0] != expected_first:
            raise RuntimeError(
                f"Unexpected {suite} base CSV schema: first field is {fieldnames[:1]}"
            )
        base_rows = [row for row in base_rows if row.get("method") != "dis"]
        dis_rows = _aggregate_dis_rows(suite, run_rows[suite], fieldnames)
        combined = [*base_rows, *dis_rows]
        combined.sort(
            key=lambda row: (
                int(row[VALUE_COLUMNS[suite]]),
                METHOD_ORDER.get(str(row["method"]), 99),
                SELECTION_ORDER.index(str(row["selection"])),
            )
        )
        csv_path = output_dir / f"gmm_{suite}_full_grid_best_v1_with_dis_aggregate.csv"
        report_path = output_dir / f"gmm_{suite}_full_grid_best_v1_with_dis_report.md"
        _write_csv(csv_path, fieldnames, combined)
        _write_report(
            report_path,
            suite=suite,
            source_csv=base_path,
            rows=combined,
        )
        expected_count = len(EXPECTED_VALUES[suite]) * 5 * 3
        if len(combined) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} combined {suite} rows, got {len(combined)}"
            )
        print(f"{suite}: rows={len(combined)} csv={csv_path} report={report_path}")


if __name__ == "__main__":
    main()
