#!/usr/bin/env python3
"""Idempotent z3 submission and scheduled monitoring for joint cosine TR-IW."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess

from aggregate_gmm_joint_cosine_triw import CAMPAIGN_GRIDS, METRICS, SELECTIONS


ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING"}
FIELDS = (
    "job_id", "stage", "suite", "sigma_max", "trust_region_bound", "seed", "target_seed",
    "experiment", "loc_scaling", "source_scale", "run_directory", "log_path", "git_commit",
)
SMOKES = {
    "joint-k40-d100": {"dim": (16, 0.1), "mode": (64, 0.3)},
    "mode-k40-d10": {"mode": (32, 0.3)},
}


def run(command: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check
    )


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def planned(stage: str, campaign: str, candidates: Path | None = None) -> list[tuple[str, int, float, int]]:
    if stage == "smoke":
        return [(suite, sigma, bound, 0) for suite, (sigma, bound) in SMOKES[campaign].items()]
    if stage == "stage1":
        return [
            (suite, sigma, bound, 0)
            for suite, grid in CAMPAIGN_GRIDS[campaign].items()
            for sigma, bound in sorted(grid)
        ]
    if stage == "stage2" and candidates is not None:
        with candidates.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        expected = 2 * len(CAMPAIGN_GRIDS[campaign])
        if len(rows) != expected:
            raise RuntimeError(f"Expected {expected} stage-2 candidates in {candidates}")
        return [
            (row["suite"], int(row["sigma_max"]), float(row["trust_region_bound"]), seed)
            for row in rows for seed in (1, 2)
        ]
    raise ValueError(f"Invalid stage or missing candidates: {stage}")


def key(suite: str, sigma: int, bound: float, seed: int) -> tuple[str, int, float, int]:
    return suite, sigma, bound, seed


def run_name(stage: str, suite: str, sigma: int, bound: float, seed: int, campaign: str) -> str:
    difficulty = "k40_d10" if campaign == "mode-k40-d10" else "k40_d100"
    return f"cosine_triw_{stage}_{suite}_{difficulty}_smax{sigma}_tr{str(bound).replace('.', 'p')}_seed{seed}"


def submit_missing(
    stage: str,
    items: list[tuple[str, int, float, int]],
    repo: Path,
    root: Path,
    log_root: Path,
    python: Path,
    commit: str,
    campaign: str,
) -> list[dict[str, str]]:
    manifest = log_root / f"{stage}.tsv"
    existing = load_manifest(manifest)
    for row in existing:
        if row["git_commit"] != commit:
            raise RuntimeError(f"Manifest {manifest} belongs to another code commit")
    submitted = {
        key(row["suite"], int(row["sigma_max"]), float(row["trust_region_bound"]), int(row["seed"]))
        for row in existing
    }
    if not manifest.exists():
        with manifest.open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t").writeheader()
    new_rows = []
    for suite, sigma, bound, seed in items:
        if key(suite, sigma, bound, seed) in submitted:
            continue
        experiment = ("gmm_mode_score_triw_cosine" if campaign == "mode-k40-d10"
                      else f"gmm_joint_{suite}_score_triw_cosine")
        difficulty = "k40_d10" if campaign == "mode-k40-d10" else "k40_d100"
        name = run_name(stage, suite, sigma, bound, seed, campaign)
        job_name = f"jcos-{stage[0]}-{suite[0]}-s{sigma}-b{str(bound).replace('.', 'p')}-r{seed}"
        log_template = log_root / "%x-%j.out"
        exports = ",".join((
            "ALL", f"TR_ASBS_REPO_ROOT={repo}", f"TR_ASBS_ROOT={root}",
            f"TR_ASBS_PYTHON={python}", f"GMM_SUITE={suite}", f"GMM_DIFFICULTY={difficulty}",
            f"GMM_SIGMA_MAX={sigma}", f"GMM_TRUST_REGION_BOUND={bound}",
            f"GMM_SEED={seed}", f"GMM_STAGE={stage}",
        ))
        result = run([
            "sbatch", "--parsable", f"--job-name={job_name}",
            f"--output={log_template}", f"--export={exports}",
            str(repo / "exec_gmm_joint_cosine_triw.sh"),
        ], repo)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Could not parse sbatch response: {result.stdout}")
        row = {
            "job_id": job_id,
            "stage": stage,
            "suite": suite,
            "sigma_max": str(sigma),
            "trust_region_bound": str(bound),
            "seed": str(seed),
            "target_seed": "0",
            "experiment": experiment,
            "loc_scaling": "7" if suite == "dim" else "40",
            "source_scale": "5" if suite == "dim" else "30",
            "run_directory": str(root / name),
            "log_path": str(log_root / f"{job_name}-{job_id}.out"),
            "git_commit": commit,
        }
        with manifest.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t").writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        new_rows.append(row)
    return new_rows


def scheduler_state(job_id: str, repo: Path) -> tuple[str, str]:
    queue = run(["squeue", "-h", "-j", job_id, "-o", "%T"], repo, check=False)
    if queue.returncode == 0 and queue.stdout.strip():
        return queue.stdout.splitlines()[0].strip().upper(), ""
    accounting = run(
        ["sacct", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State,ExitCode"],
        repo, check=False,
    )
    for line in accounting.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3 and parts[0] == job_id:
            return parts[1].split()[0].rstrip("+").upper(), parts[2]
    return "UNKNOWN", ""


def validate_report(row: dict[str, str]) -> str:
    paths = list(Path(row["run_directory"]).glob("checkpoints/*/checkpoint_evaluations.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one report for job {row['job_id']}, found {len(paths)}")
    report = json.loads(paths[0].read_text())
    if report.get("algorithm") != "score_conditioned_tr_iw":
        raise RuntimeError(f"Algorithm mismatch: {paths[0]}")
    if int(report.get("seed", -1)) != int(row["seed"]):
        raise RuntimeError(f"Seed mismatch: {paths[0]}")
    if int(report.get("capacity", {}).get("integration_steps", -1)) != 128:
        raise RuntimeError(f"Integration count mismatch: {paths[0]}")
    for selection in SELECTIONS:
        selected = report.get("selections", {}).get(selection)
        if selected is None:
            raise RuntimeError(f"Missing {selection}: {paths[0]}")
        if int(selected.get("num_samples", -1)) != 4000 or int(selected.get("report_seed", -1)) != 12345:
            raise RuntimeError(f"Wrong evaluation protocol for {selection}: {paths[0]}")
        if not all(math.isfinite(float(selected["metrics"][metric])) for metric in METRICS):
            raise RuntimeError(f"Nonfinite metrics for {selection}: {paths[0]}")
        if not (paths[0].parent / f"{selection}_data_1.npy").is_file():
            raise RuntimeError(f"Missing generated samples for {selection}: {paths[0]}")
    return str(paths[0])


def stage_status(stage: str, expected: int, log_root: Path, repo: Path) -> dict:
    manifest = log_root / f"{stage}.tsv"
    rows = load_manifest(manifest)
    counts: dict[str, int] = {}
    failures = []
    report_paths = []
    for row in rows:
        state, exit_code = scheduler_state(row["job_id"], repo)
        counts[state] = counts.get(state, 0) + 1
        if state == "COMPLETED" and exit_code == "0:0":
            try:
                report_paths.append(validate_report(row))
            except Exception as exc:
                failures.append({"job_id": row["job_id"], "reason": str(exc), "log_path": row["log_path"]})
        elif state not in ACTIVE and state != "UNKNOWN":
            failures.append({
                "job_id": row["job_id"], "reason": f"{state} {exit_code}", "log_path": row["log_path"]
            })
    for failure in failures:
        path = Path(failure["log_path"])
        failure["log_tail"] = "\n".join(path.read_text(errors="replace").splitlines()[-60:]) if path.is_file() else "log missing"
    return {
        "manifest": str(manifest), "submitted": len(rows), "expected": expected,
        "counts": counts, "validated_reports": len(report_paths), "failures": failures,
        "complete": len(rows) == expected and len(report_paths) == expected and not failures,
    }


def aggregate(repo: Path, python: Path, stage1_manifest: Path, output: Path, campaign: str,
              stage2_manifest: Path | None = None, candidates: Path | None = None) -> str:
    command = [
        str(python),
        str(repo / "scripts" / "aggregate_gmm_joint_cosine_triw.py"),
        "--stage1-manifest", str(stage1_manifest), "--output-dir", str(output),
        "--campaign", campaign,
    ]
    if stage2_manifest is not None:
        command.extend(("--stage2-manifest", str(stage2_manifest), "--candidates", str(candidates)))
    return run(command, repo).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--advance", action="store_true")
    parser.add_argument("--campaign", choices=CAMPAIGN_GRIDS, default="joint-k40-d100")
    args = parser.parse_args()
    repo, root, log_root = (p.resolve() for p in (args.repo_root, args.run_root, args.log_root))
    # Keep the virtualenv's symlink path. Path.resolve() would follow
    # .venv/bin/python to its base interpreter and silently lose site-packages.
    python = args.python.absolute()
    log_root.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    with (log_root / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        commit = run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
        state: dict = {"commit": commit, "campaign": args.campaign, "submitted_now": []}
        for stage, items in (("smoke", planned("smoke", args.campaign)), ("stage1", planned("stage1", args.campaign))):
            if stage == "stage1" and not state["smoke"]["complete"]:
                break
            prior_status = stage_status(stage, len(items), log_root, repo)
            if prior_status["failures"]:
                state[stage] = prior_status
                break
            if args.advance:
                state["submitted_now"].extend(submit_missing(stage, items, repo, root, log_root, python, commit, args.campaign))
            status = stage_status(stage, len(items), log_root, repo)
            state[stage] = status
            if status["failures"] or not status["complete"]:
                break
        if state.get("stage1", {}).get("complete"):
            stage1_dir = root / "aggregation_stage1"
            candidates = stage1_dir / "top2_candidates.tsv"
            if not candidates.is_file():
                state["stage1_report"] = aggregate(repo, python, log_root / "stage1.tsv", stage1_dir, args.campaign)
            else:
                state["stage1_report"] = str(stage1_dir / "REPORT.md")
            items = planned("stage2", args.campaign, candidates)
            prior_status = stage_status("stage2", len(items), log_root, repo)
            if args.advance and not prior_status["failures"]:
                state["submitted_now"].extend(submit_missing("stage2", items, repo, root, log_root, python, commit, args.campaign))
            status = stage_status("stage2", len(items), log_root, repo)
            state["stage2"] = status
            if status["complete"]:
                final_dir = root / "aggregation_final"
                final_report = final_dir / "REPORT.md"
                if not final_report.is_file():
                    state["final_report"] = aggregate(
                        repo, python, log_root / "stage1.tsv", final_dir, args.campaign,
                        log_root / "stage2.tsv", candidates,
                    )
                else:
                    state["final_report"] = str(final_report)
        print(json.dumps(state, indent=2))
        if any(status.get("failures") for stage, status in state.items() if stage in ("smoke", "stage1", "stage2")):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
