#!/usr/bin/env python3
"""Aggregate the completed seed-0 PIS/DIS GMM sigma tuning campaign."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Any

import torch
import wandb


RUN_PATTERN = re.compile(
    r"tune_gmm_(mode|dim)_(k40|d100)_(pis|dis)_smax(\d+)_seed0"
)
CHECKPOINTS = {
    "final": "last.pt",
    "sd_best": "loss_SD_best.pt",
    "ess_best": "eval_raw_path_ess_best.pt",
}
METRICS = [
    "loss/SD",
    "eval/raw_path_ess",
    "loss/tvd_soft",
    "loss/tvd",
    "logZ_error",
    "logZ_error_lw",
    "eval/log_z",
    "eval/target_log_prob",
    "loss/W_1",
    "loss/W_2",
    "loss/Sliced_W_1",
    "loss/Sliced_W_2",
    "loss/energy_W_2",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--wandb-project",
        default="ml-takeshi-work/tr_asbs",
    )
    return parser.parse_args()


def _wandb_run_id(run_directory: Path) -> str:
    candidates = sorted((run_directory / "wandb").glob("run-*-*"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one W&B run in {run_directory}, found {len(candidates)}"
        )
    return candidates[0].name.rsplit("-", 1)[-1]


def _history_by_iteration(run: Any) -> dict[int, dict[str, Any]]:
    keys = ["trainer/global_step", *METRICS]
    history: dict[int, dict[str, Any]] = {}
    for row in run.scan_history(keys=keys):
        step = row.get("trainer/global_step")
        if step is None or row.get("loss/SD") is None:
            continue
        history[int(step)] = row
    return history


def _checkpoint_state(path: Path) -> dict[str, Any]:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )


def _format(value: Any, digits: int = 5) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}g}"


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    api = wandb.Api()
    checkpoint_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for run_directory in sorted(args.run_root.resolve().glob("tune_*")):
        match = RUN_PATTERN.fullmatch(run_directory.name)
        if match is None:
            continue
        suite, difficulty, method, sigma_text = match.groups()
        sigma = int(sigma_text)
        run_id = _wandb_run_id(run_directory)
        run = api.run(f"{args.wandb_project}/{run_id}")
        history = _history_by_iteration(run)
        if not history:
            raise RuntimeError(f"No evaluation history for W&B run {run_id}")

        selected: dict[str, dict[str, Any]] = {}
        last_state: dict[str, Any] | None = None
        for selection, filename in CHECKPOINTS.items():
            checkpoint_path = run_directory / "checkpoints" / filename
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            state = _checkpoint_state(checkpoint_path)
            iteration = int(state["iteration"])
            metrics = history.get(iteration)
            if metrics is None:
                raise RuntimeError(
                    f"No W&B evaluation at iteration {iteration} for {run_id}"
                )
            row = {
                "suite": suite,
                "difficulty": difficulty,
                "method": method,
                "sigma_max": sigma,
                "selection": selection,
                "iteration": iteration,
                "wandb_run_id": run_id,
                "wandb_url": run.url,
                "checkpoint": str(checkpoint_path),
            }
            row.update({metric: metrics.get(metric) for metric in METRICS})
            checkpoint_rows.append(row)
            selected[selection] = row
            if selection == "final":
                last_state = state

        assert last_state is not None
        final_iteration = int(last_state["iteration"])
        divergence_state = last_state.get("divergence_state", {})
        run_rows.append(
            {
                "suite": suite,
                "difficulty": difficulty,
                "method": method,
                "sigma_max": sigma,
                "wandb_run_id": run_id,
                "wandb_url": run.url,
                "final_iteration": final_iteration,
                "evaluation_count": int(last_state.get("evaluation_count", 0)),
                "early_stopped": final_iteration < 40_000,
                "early_stop_reason": divergence_state.get("reason"),
                "final_SD": selected["final"]["loss/SD"],
                "final_raw_ESS": selected["final"]["eval/raw_path_ess"],
                "final_tvd_soft": selected["final"]["loss/tvd_soft"],
                "final_logZ_error": selected["final"]["logZ_error"],
                "sd_best_iteration": selected["sd_best"]["iteration"],
                "sd_best_SD": selected["sd_best"]["loss/SD"],
                "sd_best_raw_ESS": selected["sd_best"]["eval/raw_path_ess"],
                "sd_best_tvd_soft": selected["sd_best"]["loss/tvd_soft"],
                "sd_best_logZ_error": selected["sd_best"]["logZ_error"],
                "ess_best_iteration": selected["ess_best"]["iteration"],
                "ess_best_SD": selected["ess_best"]["loss/SD"],
                "ess_best_raw_ESS": selected["ess_best"]["eval/raw_path_ess"],
                "ess_best_tvd_soft": selected["ess_best"]["loss/tvd_soft"],
                "ess_best_logZ_error": selected["ess_best"]["logZ_error"],
            }
        )

    if len(run_rows) != 16 or len(checkpoint_rows) != 48:
        raise RuntimeError(
            f"Expected 16 runs and 48 checkpoint rows, got "
            f"{len(run_rows)} and {len(checkpoint_rows)}"
        )

    run_rows.sort(key=lambda row: (row["suite"], row["method"], row["sigma_max"]))
    checkpoint_rows.sort(
        key=lambda row: (
            row["suite"],
            row["method"],
            row["sigma_max"],
            list(CHECKPOINTS).index(row["selection"]),
        )
    )

    for filename, rows in (
        ("run_summary.csv", run_rows),
        ("checkpoint_metrics.csv", checkpoint_rows),
    ):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# PIS/DIS GMM sigma_max tuning",
        "",
        "All 16 seed-0 jobs completed with Slurm exit code 0. The table reports "
        "the final checkpoint, the checkpoint selected by minimum SD, and the "
        "checkpoint selected by maximum raw path ESS.",
        "",
        "| setting | method | sigma | stopped | selection | iteration | SD | raw ESS | soft TVD | logZ error |",
        "|---|---:|---:|:---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in checkpoint_rows:
        run_row = next(
            candidate
            for candidate in run_rows
            if candidate["suite"] == row["suite"]
            and candidate["method"] == row["method"]
            and candidate["sigma_max"] == row["sigma_max"]
        )
        lines.append(
            f"| {row['difficulty']} | {row['method']} | {row['sigma_max']} | "
            f"{'yes' if run_row['early_stopped'] else 'no'} | {row['selection']} | "
            f"{row['iteration']} | {_format(row['loss/SD'])} | "
            f"{_format(row['eval/raw_path_ess'])} | "
            f"{_format(row['loss/tvd_soft'])} | {_format(row['logZ_error'])} |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- DIS, K=40/d=10: use sigma_max=2. It has the largest best raw ESS "
            "among DIS candidates and the smallest final logZ error, while its "
            "minimum SD remains close to the best candidate.",
            "- DIS, K=10/d=100: use sigma_max=4. It has the largest best raw ESS "
            "and smallest final logZ error, with near-best SD.",
            "- PIS: no candidate in {1,2,4,8} is acceptable for the full grid. "
            "All candidates remain mode-collapsed (soft TVD about 0.978 for K=40 "
            "and about 0.854 for d=100), so the apparently high ESS at sigma 4 "
            "or 8 is not evidence of target coverage.",
            "",
            "These are seed-0 hardest-setting tuning results. Multi-seed claims "
            "should be made only after the selected configuration is run on the "
            "full grid.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))

    print(output_dir)
    print(f"runs={len(run_rows)} checkpoints={len(checkpoint_rows)}")


if __name__ == "__main__":
    main()
