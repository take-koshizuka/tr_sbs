#!/usr/bin/env python3
"""Build paired three-seed score-hint on/off comparisons for PIS and DIS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
from typing import Any


RUN_PATTERN = re.compile(
    r"hint_(zero|provided)_gmm_(mode|dim)_(k40|d100)_(pis|dis)_smax(\d+)_seed([0-2])"
)
SELECTIONS = ("final", "best_loss_SD", "best_eval_raw_path_ess")
METRICS = (
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
)
SETTINGS = {
    ("mode", "dis"): ("k40", 2),
    ("dim", "dis"): ("d100", 4),
    ("mode", "pis"): ("k40", 128),
    ("dim", "pis"): ("d100", 32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def find_report(run_directory: Path) -> Path:
    paths = sorted(run_directory.glob("checkpoints/**/checkpoint_evaluations.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one report below {run_directory}, found {len(paths)}")
    return paths[0]


def load_report(
    report_path: Path,
    *,
    suite: str,
    difficulty: str,
    method: str,
    sigma: int,
    hint_mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text())
    if report.get("algorithm") != method or int(report.get("seed", -1)) != seed:
        raise RuntimeError(f"Metadata mismatch: {report_path}")
    recorded_mode = report.get("score_hint_mode")
    if recorded_mode is not None and recorded_mode != hint_mode:
        raise RuntimeError(f"Score-hint mode mismatch: {report_path}")
    rows = []
    for selection in SELECTIONS:
        record = report.get("selections", {}).get(selection)
        if record is None:
            raise RuntimeError(f"Missing {selection}: {report_path}")
        if int(record.get("num_samples", -1)) != 4000:
            raise RuntimeError(f"Sample budget mismatch: {report_path}")
        if int(record.get("report_seed", -1)) != 12345:
            raise RuntimeError(f"Report seed mismatch: {report_path}")
        metrics = record.get("metrics", {})
        for metric in METRICS:
            if metric not in metrics:
                raise RuntimeError(f"Missing {metric}: {report_path}")
        row = {
            "suite": suite,
            "difficulty": difficulty,
            "method": method,
            "sigma_max": sigma,
            "score_hint_mode": hint_mode,
            "seed": seed,
            "selection": selection,
            "selected_iteration": int(record["selected_iteration"]),
            "final_iteration": int(report["final_iteration"]),
            "early_stopped": bool(report.get("early_stopped", False)),
            "report_path": str(report_path),
        }
        row.update({metric: float(metrics[metric]) for metric in METRICS})
        rows.append(row)
    return rows


def baseline_directory(repo_root: Path, suite: str, method: str, seed: int) -> Path:
    difficulty, sigma = SETTINGS[(suite, method)]
    if method == "dis":
        return (
            repo_root
            / "runs/gmm_dis_full_grid_best_v1"
            / f"gmm_{suite}_{difficulty}_dis_seed{seed}"
        )
    if seed == 0:
        return (
            repo_root
            / "runs/gmm_pis_sigma_tuning_stage2_seed0_v1"
            / f"tune2_gmm_{suite}_{difficulty}_pis_smax{sigma}_seed0"
        )
    return (
        repo_root
        / "runs/gmm_score_hint_ablation_v1"
        / f"hint_provided_gmm_{suite}_{difficulty}_pis_smax{sigma}_seed{seed}"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for (suite, method), (difficulty, sigma) in SETTINGS.items():
        for seed in (0, 1, 2):
            on_directory = baseline_directory(repo_root, suite, method, seed)
            rows.extend(
                load_report(
                    find_report(on_directory),
                    suite=suite,
                    difficulty=difficulty,
                    method=method,
                    sigma=sigma,
                    hint_mode="provided",
                    seed=seed,
                )
            )
            off_directory = (
                run_root
                / f"hint_zero_gmm_{suite}_{difficulty}_{method}_smax{sigma}_seed{seed}"
            )
            rows.extend(
                load_report(
                    find_report(off_directory),
                    suite=suite,
                    difficulty=difficulty,
                    method=method,
                    sigma=sigma,
                    hint_mode="zero",
                    seed=seed,
                )
            )
    if len(rows) != 72:
        raise RuntimeError(f"Expected 72 checkpoint rows, got {len(rows)}")
    rows.sort(
        key=lambda row: (
            row["suite"],
            row["method"],
            row["seed"],
            row["score_hint_mode"],
            SELECTIONS.index(row["selection"]),
        )
    )
    write_csv(output_dir / "checkpoint_metrics.csv", rows)

    summaries: list[dict[str, Any]] = []
    for (suite, method), (difficulty, sigma) in SETTINGS.items():
        for selection in SELECTIONS:
            on_rows = {
                row["seed"]: row
                for row in rows
                if row["suite"] == suite
                and row["method"] == method
                and row["selection"] == selection
                and row["score_hint_mode"] == "provided"
            }
            off_rows = {
                row["seed"]: row
                for row in rows
                if row["suite"] == suite
                and row["method"] == method
                and row["selection"] == selection
                and row["score_hint_mode"] == "zero"
            }
            if set(on_rows) != {0, 1, 2} or set(off_rows) != {0, 1, 2}:
                raise RuntimeError(f"Incomplete paired seeds for {suite}/{method}/{selection}")
            summary: dict[str, Any] = {
                "suite": suite,
                "difficulty": difficulty,
                "method": method,
                "sigma_max": sigma,
                "selection": selection,
                "n_paired_seeds": 3,
            }
            for metric in METRICS:
                on_values = [on_rows[seed][metric] for seed in (0, 1, 2)]
                off_values = [off_rows[seed][metric] for seed in (0, 1, 2)]
                deltas = [off - on for on, off in zip(on_values, off_values)]
                summary[f"{metric}_on_mean"] = statistics.fmean(on_values)
                summary[f"{metric}_on_sd"] = statistics.stdev(on_values)
                summary[f"{metric}_off_mean"] = statistics.fmean(off_values)
                summary[f"{metric}_off_sd"] = statistics.stdev(off_values)
                summary[f"{metric}_paired_delta_mean"] = statistics.fmean(deltas)
                summary[f"{metric}_paired_delta_sd"] = statistics.stdev(deltas)
            sd_deltas = [
                off_rows[seed]["loss/SD"] - on_rows[seed]["loss/SD"]
                for seed in (0, 1, 2)
            ]
            summary["sd_worse_seed_count"] = sum(delta > 0 for delta in sd_deltas)
            summary["sd_degraded"] = statistics.fmean(sd_deltas) > 0
            summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            row["suite"], row["method"], SELECTIONS.index(row["selection"])
        )
    )
    write_csv(output_dir / "paired_summary.csv", summaries)

    lines = [
        "# GMM PIS/DIS score-hint ablation",
        "",
        "Score-off replaces the score hint by an all-zero tensor without changing the network shape, schedule, training budget, or evaluation RNG. Deltas are paired as off minus on over model seeds 0/1/2.",
        "",
        "| setting | method | sigma | checkpoint | SD on | SD off | paired ΔSD | worse seeds | TVD Δ | W2 Δ | raw ESS Δ |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['difficulty']} | {row['method'].upper()} | {row['sigma_max']} | "
            f"{row['selection']} | {row['loss/SD_on_mean']:.6g}±{row['loss/SD_on_sd']:.3g} | "
            f"{row['loss/SD_off_mean']:.6g}±{row['loss/SD_off_sd']:.3g} | "
            f"{row['loss/SD_paired_delta_mean']:+.6g}±{row['loss/SD_paired_delta_sd']:.3g} | "
            f"{row['sd_worse_seed_count']}/3 | "
            f"{row['loss/tvd_soft_paired_delta_mean']:+.4g} | "
            f"{row['loss/W_2_paired_delta_mean']:+.4g} | "
            f"{row['eval/raw_path_ess_paired_delta_mean']:+.4g} |"
        )
    lines.extend(
        [
            "",
            "Positive SD/TVD/W2 deltas mean score-off is worse; negative raw-ESS delta means score-off is worse. With only three paired seeds these are descriptive comparisons, not significance claims.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    print(output_dir)


if __name__ == "__main__":
    main()

