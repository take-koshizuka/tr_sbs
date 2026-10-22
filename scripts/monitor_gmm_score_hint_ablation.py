#!/usr/bin/env python3
"""Monitor the score-hint ablation and aggregate it after all jobs pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from monitor_gmm_score_triw_tuning import latest, manifest_status, tail_failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    log_root = repo_root / "slurm_logs/gmm_score_hint_ablation_v1"
    run_root = repo_root / "runs/gmm_score_hint_ablation_v1"
    manifest = latest(log_root, "submissions_*.tsv")
    if manifest is None:
        raise RuntimeError("Score-hint ablation manifest is missing")
    status = manifest_status(manifest, repo_root)
    state = {"repo_root": str(repo_root), "jobs": status}
    if status["failed"]:
        state["failure_diagnostics"] = tail_failures(status)
        print(json.dumps(state, indent=2))
        raise SystemExit(2)
    if not status["all_completed"]:
        print(json.dumps(state, indent=2))
        return
    output_dir = run_root / "aggregation"
    report = output_dir / "REPORT.md"
    if not report.is_file():
        result = subprocess.run(
            [
                str(repo_root / ".venv/bin/python"),
                "scripts/aggregate_gmm_score_hint_ablation.py",
                "--repo-root",
                str(repo_root),
                "--run-root",
                str(run_root),
                "--output-dir",
                str(output_dir),
            ],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        state["aggregation_output"] = result.stdout.strip()
    state["complete"] = True
    state["report"] = str(report)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()

