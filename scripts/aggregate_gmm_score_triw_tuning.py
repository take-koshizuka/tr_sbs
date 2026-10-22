#!/usr/bin/env python3
"""Aggregate and select the staged score-conditioned TR-IW GMM tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


RUN_PATTERN = re.compile(
    r"score_triw_(s1|s2)_gmm_(mode|dim)_score_triw_(k40|d100)_"
    r"smax(\d+)_tr([0-9p]+)_seed([0-2])"
)
GRIDS = {
    "mode": {(sigma, bound) for sigma in (2, 4, 8) for bound in (0.1, 0.3, 1.0)},
    "dim": {(sigma, bound) for sigma in (1, 2, 4) for bound in (0.03, 0.1, 0.3)},
}
SELECTION_ORDER = ("final", "best_loss_SD", "best_eval_raw_path_ess")
METRICS = (
    "loss/SD",
    "loss/tvd_soft",
    "loss/tvd",
    "loss/W_2",
    "loss/Sliced_W_2",
    "loss/energy_W_2",
    "loss/neg_avg_energy",
    "eval/raw_path_ess",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage1", "final"), required=True)
    parser.add_argument("--candidates", type=Path)
    return parser.parse_args()


def report_path(run_directory: Path) -> Path:
    paths = sorted(run_directory.glob("checkpoints/*/checkpoint_evaluations.json"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one checkpoint report below {run_directory}, found {len(paths)}"
        )
    return paths[0]


def load_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, float, int]] = set()
    for run_directory in sorted(run_root.resolve().glob("score_triw_*")):
        match = RUN_PATTERN.fullmatch(run_directory.name)
        if match is None:
            continue
        stage, suite, difficulty, sigma_text, bound_label, seed_text = match.groups()
        sigma = int(sigma_text)
        bound = float(bound_label.replace("p", "."))
        seed = int(seed_text)
        key = (suite, sigma, bound, seed)
        if key in seen:
            raise RuntimeError(f"Duplicate run for {key}: {run_directory}")
        seen.add(key)
        path = report_path(run_directory)
        report = json.loads(path.read_text())
        if report.get("algorithm") != "score_conditioned_tr_iw":
            raise RuntimeError(f"Algorithm mismatch: {path}")
        if int(report.get("seed", -1)) != seed:
            raise RuntimeError(f"Seed mismatch: {path}")
        capacity = report.get("capacity", {})
        if int(capacity.get("integration_steps", -1)) != 128:
            raise RuntimeError(f"Integration-step mismatch: {path}")
        if int(capacity.get("forward_controller_parameters", 0)) <= 0:
            raise RuntimeError(f"Missing capacity metadata: {path}")
        selections = report.get("selections", {})
        for selection in SELECTION_ORDER:
            if selection not in selections:
                raise RuntimeError(f"Missing {selection}: {path}")
            record = selections[selection]
            if int(record.get("num_samples", -1)) != 4000:
                raise RuntimeError(f"Final sample budget mismatch: {path}")
            if int(record.get("report_seed", -1)) != 12345:
                raise RuntimeError(f"Report seed mismatch: {path}")
            metrics = record.get("metrics", {})
            for metric in METRICS:
                if metric not in metrics:
                    raise RuntimeError(f"Missing {metric} in {selection}: {path}")
            row = {
                "stage": stage,
                "suite": suite,
                "difficulty": difficulty,
                "sigma_max": sigma,
                "trust_region_bound": bound,
                "seed": seed,
                "selection": selection,
                "selected_epoch": int(record["selected_epoch"]),
                "online_selection_metric": record.get("online_selection_metric"),
                "online_selection_value": record.get("online_selection_value"),
                "final_epoch": int(report["final_epoch"]),
                "early_stopped": bool(report.get("early_stopped", False)),
                "early_stop_reason": report.get("early_stop_reason"),
                "forward_controller_parameters": int(
                    capacity["forward_controller_parameters"]
                ),
                "corrector_parameters": int(capacity["corrector_parameters"]),
                "total_trainable_parameters": int(
                    capacity["total_trainable_parameters"]
                ),
                "integration_steps": int(capacity["integration_steps"]),
                "num_samples": int(record["num_samples"]),
                "report_seed": int(record["report_seed"]),
                "report_path": str(path),
            }
            row.update({metric: float(metrics[metric]) for metric in METRICS})
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
    values = (
        float(row["loss/SD"]),
        float(row["loss/tvd_soft"]),
        float(row["loss/W_2"]),
        -float(row["eval/raw_path_ess"]),
    )
    return tuple(value if math.isfinite(value) else math.inf for value in values)


def aggregate(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise RuntimeError(f"Expected three seeds, got {len(values)}")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    return mean, sd, sd / math.sqrt(len(values))


def read_candidates(path: Path) -> dict[str, list[tuple[int, float]]]:
    selected: dict[str, list[tuple[int, float]]] = {"mode": [], "dim": []}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            selected[row["suite"]].append(
                (int(row["sigma_max"]), float(row["trust_region_bound"]))
            )
    if any(len(values) != 2 for values in selected.values()):
        raise RuntimeError(f"Expected two candidates per suite in {path}")
    return selected


def stage1(rows: list[dict[str, Any]], output_dir: Path) -> None:
    stage_rows = [row for row in rows if row["stage"] == "s1" and row["seed"] == 0]
    found = {
        suite: {
            (row["sigma_max"], row["trust_region_bound"])
            for row in stage_rows
            if row["suite"] == suite and row["selection"] == "best_loss_SD"
        }
        for suite in GRIDS
    }
    for suite, expected in GRIDS.items():
        if found[suite] != expected:
            raise RuntimeError(
                f"Incomplete {suite} stage-1 grid: expected={sorted(expected)}, "
                f"found={sorted(found[suite])}"
            )
    if len(stage_rows) != 54:
        raise RuntimeError(f"Expected 54 stage-1 checkpoint rows, got {len(stage_rows)}")
    stage_rows.sort(
        key=lambda row: (
            row["suite"],
            row["sigma_max"],
            row["trust_region_bound"],
            SELECTION_ORDER.index(row["selection"]),
        )
    )
    write_csv(output_dir / "stage1_checkpoint_metrics.csv", stage_rows)

    top_rows: list[dict[str, Any]] = []
    for suite in ("mode", "dim"):
        candidates = [
            row
            for row in stage_rows
            if row["suite"] == suite and row["selection"] == "best_loss_SD"
        ]
        candidates.sort(key=finite_rank)
        for rank, row in enumerate(candidates[:2], start=1):
            top_rows.append(
                {
                    "suite": suite,
                    "rank": rank,
                    "sigma_max": row["sigma_max"],
                    "trust_region_bound": row["trust_region_bound"],
                }
            )
    with (output_dir / "top2_candidates.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(top_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(top_rows)

    lines = [
        "# Score-conditioned TR-IW stage-1 tuning",
        "",
        "Seed 0, target seed 0, 128 integration transitions, and common 4,000-sample report seed 12345.",
        "",
        "| setting | sigma | bound | stopped | checkpoint | epoch | SD | soft TVD | W2 | raw ESS |",
        "|---|---:|---:|:---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in stage_rows:
        lines.append(
            f"| {row['difficulty']} | {row['sigma_max']} | {row['trust_region_bound']} | "
            f"{'yes' if row['early_stopped'] else 'no'} | {row['selection']} | "
            f"{row['selected_epoch']} | {row['loss/SD']:.6g} | "
            f"{row['loss/tvd_soft']:.6g} | {row['loss/W_2']:.6g} | "
            f"{row['eval/raw_path_ess']:.6g} |"
        )
    lines.extend(["", "## Candidates promoted to seeds 1 and 2", ""])
    for row in top_rows:
        lines.append(
            f"- {row['suite']} rank {row['rank']}: sigma_max={row['sigma_max']}, "
            f"trust_region_bound={row['trust_region_bound']}"
        )
    (output_dir / "REPORT_STAGE1.md").write_text("\n".join(lines) + "\n")


def final(rows: list[dict[str, Any]], output_dir: Path, candidates_path: Path) -> None:
    candidates = read_candidates(candidates_path)
    selected_rows = [
        row
        for row in rows
        if (row["sigma_max"], row["trust_region_bound"])
        in candidates[row["suite"]]
    ]
    for suite, suite_candidates in candidates.items():
        for sigma, bound in suite_candidates:
            for seed in (0, 1, 2):
                matches = [
                    row
                    for row in selected_rows
                    if row["suite"] == suite
                    and row["sigma_max"] == sigma
                    and row["trust_region_bound"] == bound
                    and row["seed"] == seed
                ]
                if len(matches) != 3:
                    raise RuntimeError(
                        f"Expected three checkpoint selections for "
                        f"{suite}, sigma={sigma}, bound={bound}, seed={seed}; "
                        f"got {len(matches)}"
                    )
    selected_rows.sort(
        key=lambda row: (
            row["suite"],
            row["sigma_max"],
            row["trust_region_bound"],
            row["seed"],
            SELECTION_ORDER.index(row["selection"]),
        )
    )
    write_csv(output_dir / "final_checkpoint_metrics.csv", selected_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for suite, suite_candidates in candidates.items():
        for sigma, bound in suite_candidates:
            for selection in SELECTION_ORDER:
                group = [
                    row
                    for row in selected_rows
                    if row["suite"] == suite
                    and row["sigma_max"] == sigma
                    and row["trust_region_bound"] == bound
                    and row["selection"] == selection
                ]
                aggregate_row: dict[str, Any] = {
                    "suite": suite,
                    "difficulty": group[0]["difficulty"],
                    "sigma_max": sigma,
                    "trust_region_bound": bound,
                    "selection": selection,
                    "n_seeds": 3,
                }
                for metric in METRICS:
                    mean, sd, se = aggregate([float(row[metric]) for row in group])
                    aggregate_row[f"{metric}_mean"] = mean
                    aggregate_row[f"{metric}_sd"] = sd
                    aggregate_row[f"{metric}_se"] = se
                aggregate_rows.append(aggregate_row)
    aggregate_rows.sort(
        key=lambda row: (
            row["suite"],
            row["sigma_max"],
            row["trust_region_bound"],
            SELECTION_ORDER.index(row["selection"]),
        )
    )
    write_csv(output_dir / "three_seed_summary.csv", aggregate_rows)

    winners: dict[str, dict[str, Any]] = {}
    for suite in ("mode", "dim"):
        ranked = [
            row
            for row in aggregate_rows
            if row["suite"] == suite and row["selection"] == "best_loss_SD"
        ]
        ranked.sort(key=lambda row: row["loss/SD_mean"])
        best, runner_up = ranked
        combined_se = math.sqrt(
            best["loss/SD_se"] ** 2 + runner_up["loss/SD_se"] ** 2
        )
        if abs(best["loss/SD_mean"] - runner_up["loss/SD_mean"]) <= combined_se:
            ranked.sort(
                key=lambda row: (
                    row["loss/tvd_soft_mean"],
                    row["loss/W_2_mean"],
                    -row["eval/raw_path_ess_mean"],
                )
            )
            best = ranked[0]
            best["selection_reason"] = "SD difference within combined SE; TVD/W2/ESS tie-break"
        else:
            best["selection_reason"] = "lowest three-seed mean SD"
        winners[suite] = best
    (output_dir / "winners.json").write_text(json.dumps(winners, indent=2, sort_keys=True))

    lines = [
        "# Score-conditioned TR-IW final tuning report",
        "",
        "Three model seeds, fixed target seed 0, 128 integration transitions, and common 4,000-sample report seed 12345.",
        "",
        "| setting | sigma | bound | checkpoint | SD mean±SD | soft TVD mean±SD | W2 mean±SD | raw ESS mean±SD |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['difficulty']} | {row['sigma_max']} | {row['trust_region_bound']} | "
            f"{row['selection']} | {row['loss/SD_mean']:.6g}±{row['loss/SD_sd']:.3g} | "
            f"{row['loss/tvd_soft_mean']:.6g}±{row['loss/tvd_soft_sd']:.3g} | "
            f"{row['loss/W_2_mean']:.6g}±{row['loss/W_2_sd']:.3g} | "
            f"{row['eval/raw_path_ess_mean']:.6g}±{row['eval/raw_path_ess_sd']:.3g} |"
        )
    lines.extend(["", "## Winners", ""])
    for suite, winner in winners.items():
        lines.append(
            f"- {suite}: sigma_max={winner['sigma_max']}, "
            f"trust_region_bound={winner['trust_region_bound']} "
            f"({winner['selection_reason']})."
        )
    (output_dir / "REPORT_FINAL.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.run_root)
    if args.stage == "stage1":
        stage1(rows, output_dir)
    else:
        if args.candidates is None:
            raise SystemExit("--candidates is required for --stage final")
        final(rows, output_dir, args.candidates.resolve())
    print(output_dir)


if __name__ == "__main__":
    main()
