#!/usr/bin/env python3
"""Idempotently monitor and advance the staged remote TR-IW tuning campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "RESIZING",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--advance", action="store_true")
    return parser.parse_args()


def latest(root: Path, pattern: str) -> Path | None:
    paths = sorted(root.glob(pattern))
    return paths[-1] if paths else None


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run(command: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def scheduler_state(job_id: str, repo_root: Path) -> tuple[str, str]:
    queued_result = run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        repo_root,
        check=False,
    )
    # Slurm may print "Invalid job id specified" to stderr after a completed
    # job has left squeue.  ``run`` merges stderr into stdout, so only consume
    # the output as a queue state when squeue itself succeeded; otherwise fall
    # through to sacct, which retains the terminal state.
    queued = queued_result.stdout.strip()
    if queued_result.returncode == 0 and queued:
        return queued.splitlines()[0].strip().upper(), ""
    accounting = run(
        ["sacct", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State,ExitCode"],
        repo_root,
        check=False,
    ).stdout
    for line in accounting.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] == job_id:
            state = fields[1].split()[0].rstrip("+").upper()
            return state, fields[2]
    return "UNKNOWN", ""


def manifest_status(path: Path, repo_root: Path) -> dict[str, Any]:
    rows = read_manifest(path)
    jobs = []
    counts: dict[str, int] = {}
    for row in rows:
        state, exit_code = scheduler_state(row["job_id"], repo_root)
        counts[state] = counts.get(state, 0) + 1
        jobs.append(
            {
                "job_id": row["job_id"],
                "state": state,
                "exit_code": exit_code,
                "log_path": row["log_path"],
                "run_directory": row["run_directory"],
            }
        )
    failed = [
        job
        for job in jobs
        if job["state"] not in ACTIVE_STATES | {"COMPLETED", "UNKNOWN"}
        or (job["state"] == "COMPLETED" and job["exit_code"] not in ("", "0:0"))
    ]
    return {
        "manifest": str(path),
        "counts": counts,
        "jobs": jobs,
        "all_completed": bool(jobs)
        and all(job["state"] == "COMPLETED" and job["exit_code"] == "0:0" for job in jobs),
        "failed": failed,
    }


def tail_failures(status: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = []
    for job in status["failed"]:
        log_path = Path(job["log_path"])
        tail = "log missing"
        if log_path.is_file():
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        diagnostics.append(
            {
                "job_id": job["job_id"],
                "state": job["state"],
                "exit_code": job["exit_code"],
                "log_path": str(log_path),
                "log_tail": tail,
            }
        )
    return diagnostics


def validate_smoke(run_directory: Path) -> dict[str, Any]:
    reports = sorted(run_directory.glob("checkpoints/*/checkpoint_evaluations.json"))
    if len(reports) != 1:
        raise RuntimeError(f"Expected one smoke checkpoint report, found {len(reports)}")
    report = json.loads(reports[0].read_text())
    capacity = report.get("capacity", {})
    if report.get("algorithm") != "score_conditioned_tr_iw":
        raise RuntimeError("Smoke algorithm metadata mismatch")
    if int(capacity.get("integration_steps", -1)) != 128:
        raise RuntimeError("Smoke did not use 128 integration transitions")
    for selection in ("final", "best_loss_SD", "best_eval_raw_path_ess"):
        record = report.get("selections", {}).get(selection)
        if record is None:
            raise RuntimeError(f"Smoke report is missing {selection}")
        if int(record.get("num_samples", -1)) != 4000:
            raise RuntimeError(f"Smoke {selection} did not use 4,000 samples")
        if int(record.get("report_seed", -1)) != 12345:
            raise RuntimeError(f"Smoke {selection} report seed mismatch")
        for metric in ("loss/SD", "loss/tvd_soft", "loss/W_2", "eval/raw_path_ess"):
            value = float(record.get("metrics", {}).get(metric, float("nan")))
            if not math.isfinite(value):
                raise RuntimeError(f"Smoke {selection} has non-finite {metric}")
    return {
        "report": str(reports[0]),
        "capacity": capacity,
        "selections": list(report["selections"]),
    }


def submit(command: list[str], repo_root: Path) -> str:
    result = run(command, repo_root)
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    log_root = repo_root / "slurm_logs" / "gmm_score_triw_tuning_v1"
    smoke_log_root = repo_root / "slurm_logs" / "gmm_score_triw_smoke_v1"
    run_root = repo_root / "runs" / "gmm_score_triw_tuning_v1"
    state: dict[str, Any] = {"repo_root": str(repo_root), "advanced": []}

    smoke_manifest = latest(smoke_log_root, "smoke_submission_*.tsv")
    if smoke_manifest is None:
        raise RuntimeError("Smoke manifest is missing")
    smoke_status = manifest_status(smoke_manifest, repo_root)
    state["smoke"] = smoke_status
    if smoke_status["failed"]:
        state["failure_diagnostics"] = tail_failures(smoke_status)
        print(json.dumps(state, indent=2))
        raise SystemExit(2)
    if not smoke_status["all_completed"]:
        print(json.dumps(state, indent=2))
        return
    smoke_run = Path(smoke_status["jobs"][0]["run_directory"])
    state["smoke_validation"] = validate_smoke(smoke_run)

    stage1_manifest = latest(log_root, "stage1_submissions_*.tsv")
    if stage1_manifest is None:
        if args.advance:
            state["advanced"].append(
                submit(["bash", "scripts/slurm/submit_gmm_score_triw_tuning_stage1.sh"], repo_root)
            )
        print(json.dumps(state, indent=2))
        return

    stage1_status = manifest_status(stage1_manifest, repo_root)
    state["stage1"] = stage1_status
    if stage1_status["failed"]:
        state["failure_diagnostics"] = tail_failures(stage1_status)
        print(json.dumps(state, indent=2))
        raise SystemExit(2)
    if not stage1_status["all_completed"]:
        print(json.dumps(state, indent=2))
        return

    stage1_output = run_root / "aggregation_stage1"
    candidates_path = stage1_output / "top2_candidates.tsv"
    if not candidates_path.is_file():
        state["advanced"].append(
            submit(
                [
                    str(repo_root / ".venv" / "bin" / "python"),
                    "scripts/aggregate_gmm_score_triw_tuning.py",
                    "--run-root",
                    str(run_root),
                    "--output-dir",
                    str(stage1_output),
                    "--stage",
                    "stage1",
                ],
                repo_root,
            )
        )
    state["stage1_candidates"] = str(candidates_path)

    stage2_manifest = latest(log_root, "stage2_submissions_*.tsv")
    if stage2_manifest is None:
        if args.advance:
            state["advanced"].append(
                submit(
                    [
                        "bash",
                        "scripts/slurm/submit_gmm_score_triw_tuning_stage2.sh",
                        "--candidates",
                        str(candidates_path),
                    ],
                    repo_root,
                )
            )
        print(json.dumps(state, indent=2))
        return

    stage2_status = manifest_status(stage2_manifest, repo_root)
    state["stage2"] = stage2_status
    if stage2_status["failed"]:
        state["failure_diagnostics"] = tail_failures(stage2_status)
        print(json.dumps(state, indent=2))
        raise SystemExit(2)
    if not stage2_status["all_completed"]:
        print(json.dumps(state, indent=2))
        return

    final_output = run_root / "aggregation_final"
    final_report = final_output / "REPORT_FINAL.md"
    if not final_report.is_file():
        state["advanced"].append(
            submit(
                [
                    str(repo_root / ".venv" / "bin" / "python"),
                    "scripts/aggregate_gmm_score_triw_tuning.py",
                    "--run-root",
                    str(run_root),
                    "--output-dir",
                    str(final_output),
                    "--stage",
                    "final",
                    "--candidates",
                    str(candidates_path),
                ],
                repo_root,
            )
        )
    state["final_report"] = str(final_report)
    state["complete"] = True
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
