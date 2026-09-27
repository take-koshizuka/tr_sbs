#!/usr/bin/env python3
"""Aggregate the two K=40, d=100 cosine TR-IW target geometries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SELECTIONS = ("final", "best_loss_SD", "best_eval_raw_path_ess")
METRICS = ("loss/SD", "loss/tvd_soft", "loss/W_2", "eval/raw_path_ess")
GRIDS = {
    "dim": {(sigma, bound) for sigma in (8, 16, 32) for bound in (0.03, 0.1, 0.3)},
    "mode": {(sigma, bound) for sigma in (32, 64, 128) for bound in (0.1, 0.3, 1.0)},
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_table(path: Path, rows: list[dict], delimiter: str = ",") -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def report_rows(manifest: Path) -> list[dict]:
    rows = []
    for item in read_tsv(manifest):
        run_dir = Path(item["run_directory"])
        paths = list(run_dir.glob("checkpoints/*/checkpoint_evaluations.json"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one report in {run_dir}, found {len(paths)}")
        report = json.loads(paths[0].read_text())
        if report.get("algorithm") != "score_conditioned_tr_iw":
            raise RuntimeError(f"Wrong algorithm in {paths[0]}")
        if int(report.get("seed", -1)) != int(item["seed"]):
            raise RuntimeError(f"Wrong model seed in {paths[0]}")
        if int(report.get("capacity", {}).get("integration_steps", -1)) != 128:
            raise RuntimeError(f"Wrong integration count in {paths[0]}")
        for selection in SELECTIONS:
            record = report.get("selections", {}).get(selection)
            if record is None:
                raise RuntimeError(f"Missing {selection} in {paths[0]}")
            if int(record.get("num_samples", -1)) != 4000 or int(record.get("report_seed", -1)) != 12345:
                raise RuntimeError(f"Wrong evaluation protocol in {paths[0]}")
            metrics = record["metrics"]
            values = {metric: float(metrics[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"Nonfinite {selection} metrics in {paths[0]}")
            rows.append({
                "stage": item["stage"],
                "suite": item["suite"],
                "sigma_max": int(item["sigma_max"]),
                "trust_region_bound": float(item["trust_region_bound"]),
                "seed": int(item["seed"]),
                "selection": selection,
                "selected_epoch": int(record["selected_epoch"]),
                "final_epoch": int(report["final_epoch"]),
                "early_stopped": bool(report.get("early_stopped", False)),
                "report_path": str(paths[0]),
                **values,
            })
    return rows


def rank_values(row: dict) -> tuple[float, float, float, float]:
    return (
        row["loss/SD"],
        row["loss/tvd_soft"],
        row["loss/W_2"],
        -row["eval/raw_path_ess"],
    )


def stage1(rows: list[dict], output_dir: Path) -> None:
    selected = [row for row in rows if row["selection"] == "best_loss_SD"]
    for suite, grid in GRIDS.items():
        found = {(row["sigma_max"], row["trust_region_bound"]) for row in selected if row["suite"] == suite}
        if found != grid:
            raise RuntimeError(f"Incomplete {suite} grid: expected={grid}, found={found}")
    if len(selected) != 18 or len(rows) != 54:
        raise RuntimeError(f"Expected 18 runs / 54 checkpoint rows, got {len(selected)} / {len(rows)}")
    candidates = []
    for suite in ("dim", "mode"):
        ranked = sorted((row for row in selected if row["suite"] == suite), key=rank_values)
        for rank, row in enumerate(ranked[:2], 1):
            candidates.append({
                "suite": suite,
                "rank": rank,
                "sigma_max": row["sigma_max"],
                "trust_region_bound": row["trust_region_bound"],
                "seed0_SD": row["loss/SD"],
                "seed0_TVD_soft": row["loss/tvd_soft"],
                "seed0_W2": row["loss/W_2"],
                "seed0_raw_path_ESS": row["eval/raw_path_ess"],
            })
    write_table(output_dir / "top2_candidates.tsv", candidates, delimiter="\t")
    lines = [
        "# K=40, d=100 cosine TR-IW tuning: stage 1",
        "",
        "Score-conditioned control; 128 exponential-integrator transitions; PIS zero-drift cosine noise; target seed 0; model seed 0.",
        "The reported sigma_max is the PIS schedule parameter (the attained maximum is approximately half).",
        "",
        "| Target geometry | Rank | sigma_max | TR bound | SD | TVD-soft | W2 | raw path ESS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['suite']} | {row['rank']} | {row['sigma_max']} | {row['trust_region_bound']} | "
            f"{row['seed0_SD']:.5g} | {row['seed0_TVD_soft']:.5g} | {row['seed0_W2']:.5g} | {row['seed0_raw_path_ESS']:.5g} |"
        )
    lines.extend(["", "All final, minimum-SD, and maximum-ESS checkpoint metrics are in `checkpoint_metrics.csv`.", ""])
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def final(rows: list[dict], candidates_path: Path, output_dir: Path) -> None:
    candidate_rows = read_tsv(candidates_path)
    if len(candidate_rows) != 4:
        raise RuntimeError("Expected two candidates per target geometry")
    summary = []
    for candidate in candidate_rows:
        suite = candidate["suite"]
        sigma = int(candidate["sigma_max"])
        bound = float(candidate["trust_region_bound"])
        for selection in SELECTIONS:
            matching = [
                row for row in rows
                if row["suite"] == suite and row["sigma_max"] == sigma
                and row["trust_region_bound"] == bound and row["selection"] == selection
            ]
            if sorted(row["seed"] for row in matching) != [0, 1, 2]:
                raise RuntimeError(f"Missing seed for {(suite, sigma, bound, selection)}")
            values = {}
            for metric in METRICS:
                metric_values = [row[metric] for row in matching]
                values[f"{metric}_mean"] = statistics.fmean(metric_values)
                values[f"{metric}_se"] = statistics.stdev(metric_values) / math.sqrt(3)
            summary.append({
                "suite": suite,
                "sigma_max": sigma,
                "trust_region_bound": bound,
                "selection": selection,
                **values,
            })
    write_table(output_dir / "three_seed_summary.csv", summary)
    lines = [
        "# K=40, d=100 cosine TR-IW tuning: final three-seed report",
        "",
        "Two target geometries are reported separately: dim (loc_scaling=7, source.scale=5) and mode (loc_scaling=40, source.scale=30). Target seed is fixed at 0; model seeds are 0/1/2.",
        "All metrics use the same 4,000-sample evaluation seed 12345. Means are shown with standard errors. The three checkpoint selections are kept separate.",
        "",
        "| Target | sigma_max | TR bound | Checkpoint | SD | TVD-soft | W2 | raw path ESS |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        def cell(metric: str) -> str:
            return f"{row[f'{metric}_mean']:.4g} ± {row[f'{metric}_se']:.2g}"
        lines.append(
            f"| {row['suite']} | {row['sigma_max']} | {row['trust_region_bound']} | {row['selection']} | "
            f"{cell('loss/SD')} | {cell('loss/tvd_soft')} | {cell('loss/W_2')} | {cell('eval/raw_path_ess')} |"
        )
    lines.extend(["", "## Selected parameters", ""])
    for suite in ("dim", "mode"):
        choices = [row for row in summary if row["suite"] == suite and row["selection"] == "best_loss_SD"]
        first, second = sorted(choices, key=lambda row: row["loss/SD_mean"])
        combined_se = math.hypot(first["loss/SD_se"], second["loss/SD_se"])
        if second["loss/SD_mean"] - first["loss/SD_mean"] <= combined_se:
            first, second = sorted(
                choices,
                key=lambda row: (
                    row["loss/tvd_soft_mean"],
                    row["loss/W_2_mean"],
                    -row["eval/raw_path_ess_mean"],
                ),
            )
            rule = "SD difference within combined standard error; TVD-soft/W2/raw ESS tie-break"
        else:
            rule = "lower three-seed mean SD"
        lines.append(
            f"- {suite}: sigma_max={first['sigma_max']}, TR bound={first['trust_region_bound']} ({rule})."
        )
    lines.extend(["", "All per-run checkpoint metrics are in `checkpoint_metrics.csv`.", ""])
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-manifest", type=Path, required=True)
    parser.add_argument("--stage2-manifest", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = report_rows(args.stage1_manifest)
    if args.stage2_manifest is not None:
        if args.candidates is None:
            parser.error("--candidates is required with --stage2-manifest")
        rows.extend(report_rows(args.stage2_manifest))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.output_dir / "checkpoint_metrics.csv", rows)
    if args.stage2_manifest is None:
        stage1(rows, args.output_dir)
    else:
        final(rows, args.candidates, args.output_dir)
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
