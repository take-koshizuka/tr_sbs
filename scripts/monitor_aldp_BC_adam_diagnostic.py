#!/usr/bin/env python3
"""One bounded Slurm snapshot for JAX backend verification; no job waits."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

from monitor_aldp_chart_campaign import ACTIVE, execute, read_manifest
from run_aldp_chart_campaign import ROOT, sha256, verify_source, write_json

FIELDS = ("job_id", "stage", "candidate", "seed", "run_directory", "log_path",
          "repo", "python", "source_sha256", "plan_sha256")


def submit(campaign, plan, python, source_sha):
    rows = read_manifest(campaign / "submitted_jobs.tsv")
    if rows:
        return rows[0]
    output = campaign / "backend_preflight" / "jax_seed0"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Unledgered output exists; reconcile before any submission")
    intent = campaign / "submission_intents" / "backend_preflight_jax_seed0.json"
    if intent.exists():
        raise RuntimeError("Existing submission intent must be reconciled; never dispatch twice")
    log = campaign / "logs" / "aldp-BC-growth-%j.out"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = ["sbatch", "--parsable", "--job-name=aldp-BC-growth", "--partition=gpu",
        "--gres=gpu:1", "--cpus-per-task=4", "--mem=32G", "--time=00:30:00",
        "--exclude=htc-gpu024", f"--output={log}",
        str(ROOT / "exec_aldp_BC_adam_diagnostic.sh"), str(ROOT), str(python), str(campaign)]
    write_json(intent, dict(status="dispatching", command=command, output=str(output)))
    result = execute(command)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError("Unparseable Slurm job ID; reconcile intent")
    row = dict(job_id=job, stage="backend_preflight", candidate="jax", seed=0,
        run_directory=str(output), log_path=str(log).replace("%j", job), repo=str(ROOT),
        python=str(python), source_sha256=source_sha, plan_sha256=sha256(campaign / "plan.json"))
    write_json(intent, dict(status="submitted", command=command, row=row,
                           submitted_at=datetime.now(timezone.utc).isoformat()))
    with (campaign / "submitted_jobs.tsv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader(); writer.writerow(row); stream.flush(); os.fsync(stream.fileno())
    return row


def accounting(job):
    result = execute(["sacct", "-n", "-P", "-j", job,
        "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList"], False)
    rows = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 5:
            rows[parts[0]] = dict(state=parts[1].split()[0].rstrip("+"),
                exit_code=parts[2], elapsed=parts[3], nodes=parts[4])
    queue = execute(["squeue", "-h", "-j", job, "-o", "%T"], False)
    state = queue.stdout.strip().splitlines()[0] if queue.returncode == 0 and queue.stdout.strip() else rows.get(job, {}).get("state", "UNKNOWN")
    return state, rows


def verify_completed(row, campaign, source_sha):
    if row["source_sha256"] != source_sha or row["plan_sha256"] != sha256(campaign / "plan.json"):
        raise RuntimeError("Ledger source/plan SHA mismatch")
    output = Path(row["run_directory"])
    rec = json.loads((output / "receipt.json").read_text())
    if (rec.get("schema") != "aldp_chart_receipt_v1" or not rec.get("postflight_verified")
        or rec["identity"] != dict(stage="backend_preflight", candidate="jax", seed=0)
        or rec["inputs"]["source_sha256"] != source_sha
        or rec["inputs"]["plan_sha256"] != row["plan_sha256"]):
        raise RuntimeError("Invalid JAX preflight receipt")
    for name, expected in rec["files"].items():
        if sha256(output / name) != expected:
            raise RuntimeError("Receipt SHA mismatch: " + name)
    report=json.loads((output/'update_growth.json').read_text())
    if report['status']!='diagnostic_complete' or report['source_sha256']!=source_sha or report['plan_sha256']!=row['plan_sha256'] or report['OpenMM']!=0 or report['tolerance_relaxation'] or report['full_training_submitted']:raise RuntimeError('Incomplete diagnostic')
    if [r['common_gradient'] for r in report['common_gradient_Adam']]!=['torch_gradient','jax_gradient']:raise RuntimeError('Both common-gradient Adam controls required')
    if 'diagnostic_complete=true' not in Path(row['log_path']).read_text():raise RuntimeError('Missing diagnostic marker')
    return report, len(rec["files"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign / "plan.json").read_text())
    with (campaign / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source_sha = verify_source()
        if source_sha != plan["source_manifest_sha256"]:
            raise RuntimeError("Campaign source snapshot changed")
        for name, expected in plan["frozen_inputs"].items():
            if sha256(name) != expected:
                raise RuntimeError("Frozen input SHA mismatch: " + name)
        if args.submit:
            row = submit(campaign, plan, args.python, source_sha)
            result = dict(status="submitted", row=row, observed_scheduler_state=False,
                quality="not_evaluated", full_reference_training="not_submitted",
                next_check_interval_minutes=10, active_checks=0)
        else:
            rows = read_manifest(campaign / "submitted_jobs.tsv")
            if len(rows) != 1:
                raise RuntimeError("Exactly one backend preflight ledger row is required")
            row = rows[0]
            state, accounts = accounting(row["job_id"])
            previous_path = campaign / "monitor_state.json"
            previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
            active_checks = previous.get("active_checks", 0)+1 if state in ACTIVE else 0
            result = dict(job_id=row["job_id"], scheduler_state=state, accounting=accounts,
                source_sha256=source_sha, plan_sha256=row["plan_sha256"],
                active_checks=active_checks, next_check_interval_minutes=30 if active_checks >= 2 else 10,
                quality="not_evaluated", full_reference_training="not_submitted")
            log_path = Path(row["log_path"])
            if log_path.exists():
                result["log_sha256"] = sha256(log_path)
                result["log_tail"] = "\n".join(log_path.read_text(errors="replace").splitlines()[-16:])
            if state == "COMPLETED":
                required = [row["job_id"], row["job_id"]+".batch", row["job_id"]+".extern"]
                if any(accounts.get(key, {}).get("state") != "COMPLETED"
                    or accounts[key]["exit_code"] != "0:0" for key in required):
                    raise RuntimeError("All root/batch/extern accounting must complete with zero exit")
                report, files = verify_completed(row, campaign, source_sha)
                result.update(status="diagnostic_verified", receipt_files_verified=files,
                    report_sha256=sha256(Path(row["run_directory"])/"update_growth.json"),
                    updates=report["rows"], cost=dict(OpenMM=report["OpenMM"]),
                    numerical_repair=report.get("numerical_repair"),
                    next_action="diagnose error growth; no full training or tolerance relaxation")
            elif state in ACTIVE:
                result["status"] = "active"
            else:
                result["status"] = "failed_or_accounting_unknown"
                result["next_action"] = "inspect failure artifacts/log; retry only identified implementation bugs in a new snapshot/output"
        result["checked_at"] = datetime.now(timezone.utc).isoformat()
        write_json(campaign / "monitor_state.json", result)
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
