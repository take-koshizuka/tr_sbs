#!/usr/bin/env python3
"""Aggregate common-seed checkpoint reports from the PIS stage-2 tuning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any


PATTERN = re.compile(r"tune2_gmm_(mode|dim)_(k40|d100)_pis_smax(\d+)_seed0")
EXPECTED = {
    "mode": {32, 64, 96, 128},
    "dim": {8, 16, 24, 32},
}
SELECTIONS = {
    "final": "final",
    "best_loss_SD": "min_online_sd",
    "best_eval_raw_path_ess": "max_online_raw_ess",
}
METRICS = [
    "loss/SD",
    "loss/tvd_soft",
    "loss/tvd",
    "loss/W_2",
    "loss/Sliced_W_2",
    "loss/RBF_MMD",
    "loss/energy_W_2",
    "loss/neg_avg_energy",
    "eval/raw_path_ess",
    "eval/log_z",
    "logZ_error",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _format(value: Any, digits: int = 5) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}g}"


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, Any]] = []
    seen: dict[str, set[int]] = {"mode": set(), "dim": set()}
    for run_directory in sorted(args.run_root.resolve().iterdir()):
        match = PATTERN.fullmatch(run_directory.name)
        if match is None:
            continue
        suite, difficulty, sigma_text = match.groups()
        sigma = int(sigma_text)
        if sigma not in EXPECTED[suite]:
            raise RuntimeError(f"Unexpected sigma candidate: {run_directory.name}")
        if sigma in seen[suite]:
            raise RuntimeError(f"Duplicate sigma candidate: {run_directory.name}")
        seen[suite].add(sigma)

        path = run_directory / "checkpoints" / "checkpoint_evaluations.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text())
        if report.get("algorithm") != "pis" or int(report.get("seed", -1)) != 0:
            raise RuntimeError(f"Run metadata mismatch: {path}")
        selections = report.get("selections", {})
        for source_selection, selection in SELECTIONS.items():
            if source_selection not in selections:
                raise RuntimeError(f"Missing {source_selection}: {path}")
            record = selections[source_selection]
            metrics = record.get("metrics", {})
            row = {
                "suite": suite,
                "difficulty": difficulty,
                "sigma_max": sigma,
                "selection": selection,
                "selected_iteration": int(record["selected_iteration"]),
                "online_selection_metric": record.get("online_selection_metric"),
                "online_selection_value": record.get("online_selection_value"),
                "final_iteration": int(report["final_iteration"]),
                "early_stopped": bool(report.get("early_stopped", False)),
                "early_stop_reason": report.get("early_stop_reason"),
                "num_samples": int(record["num_samples"]),
                "report_seed": int(record["report_seed"]),
                "report_path": str(path),
            }
            for metric in METRICS:
                if metric not in metrics:
                    raise RuntimeError(f"Missing {metric}: {path}")
                row[metric] = float(metrics[metric])
            rows.append(row)

    for suite, expected in EXPECTED.items():
        if seen[suite] != expected:
            raise RuntimeError(
                f"Incomplete {suite} grid: expected={sorted(expected)}, got={sorted(seen[suite])}"
            )
    if len(rows) != 24:
        raise RuntimeError(f"Expected 24 checkpoint rows, got {len(rows)}")

    selection_order = list(SELECTIONS.values())
    rows.sort(
        key=lambda row: (
            row["suite"],
            row["sigma_max"],
            selection_order.index(row["selection"]),
        )
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "checkpoint_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PIS GMM sigma_max stage-2 tuning",
        "",
        "Seed-0 hardest-setting tuning with fixed target seed 0. Every retained checkpoint is reevaluated using 4,000 samples and common report seed 12345.",
        "",
        "Minimum-online-SD is the primary checkpoint; mode coverage (soft TVD) must be checked before treating a high path ESS as valid.",
        "",
        "| setting | sigma | stopped | selection | iteration | SD | soft TVD | W2 | sliced W2 | raw ESS | logZ error |",
        "|---|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['difficulty']} | {row['sigma_max']} | "
            f"{'yes' if row['early_stopped'] else 'no'} | {row['selection']} | "
            f"{row['selected_iteration']} | {_format(row['loss/SD'])} | "
            f"{_format(row['loss/tvd_soft'])} | {_format(row['loss/W_2'])} | "
            f"{_format(row['loss/Sliced_W_2'])} | "
            f"{_format(row['eval/raw_path_ess'])} | {_format(row['logZ_error'])} |"
        )
    lines.extend(
        [
            "",
            "No candidate is promoted automatically by this script. Select a sigma only if its minimum-online-SD checkpoint improves coverage relative to stage 1 without an unacceptable SD/stability tradeoff.",
            "",
        ]
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"rows={len(rows)} csv={csv_path} report={report_path}")


if __name__ == "__main__":
    main()
