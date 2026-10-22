"""One scheduler/provenance pass for the saved-batch gate; never submits jobs."""
import argparse
import json
import subprocess
from pathlib import Path


def check(campaign):
    from run_aldp_chart_campaign import sha256, verify_source
    from run_aldp_scaled_backward_multistep import counters, validate_plan
    import torch
    plan_path = campaign/'plan.json'
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    source = verify_source()
    assert source == plan['source_manifest_sha256']
    for name, expected in plan['frozen_inputs'].items():
        assert sha256(name) == expected, name
    assert plan['tolerances'] == json.loads(Path(plan['tolerance_origin']).read_text())['tolerances']
    submission = json.loads((campaign/'submission.json').read_text())
    job = submission['job_id']
    raw = subprocess.run(['sacct', '-j', job, '-P', '-n', '-o',
                          'JobIDRaw,State,ExitCode,Elapsed,NodeList'],
                         capture_output=True, text=True, check=True).stdout
    accounting = [line.split('|') for line in raw.splitlines() if line.strip()]
    active = any(row[1].split()[0] in ['RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'] for row in accounting)
    output = campaign/'diagnostic'
    gate_path = output/'gate.json'
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    receipt_path = output/'receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if receipt:
        for name, expected in receipt['files'].items():
            assert sha256(output/name) == expected, name
        assert receipt['inputs']['source_sha256'] == source
        assert receipt['inputs']['plan_sha256'] == sha256(plan_path)
    complete = bool(accounting) and not active and all(row[1] == 'COMPLETED' and row[2] == '0:0' for row in accounting)
    status = 'active' if active else 'accounting_unavailable'
    if complete:
        assert {row[0] for row in accounting} == {job, job+'.batch', job+'.extern'}
        assert gate and receipt and gate['status'] == 'passed'
        assert gate['OpenMM'] == 0 and gate['optimizer_updates'] == gate['reference_optimizer_updates'] == 40
        assert gate['optimizer_updates_discarded'] and gate['updates_per_case'] == 20
        assert not gate['production_changed'] and not gate['full_training_submitted']
        assert gate['source_sha256'] == source and gate['plan_sha256'] == sha256(plan_path)
        assert gate['matmul_precision'] == 'highest' and gate['tolerances'] == plan['tolerances']
        assert len(gate['rows']) == 40
        for case in plan['cases']:
            rows = [row for row in gate['rows'] if row['case'] == case]
            assert [row['update'] for row in rows] == list(range(1, 21))
            saved_path = plan['normal_snapshot'] if case == 'normal' else plan['witness']
            saved = torch.load(saved_path, map_location='cpu', weights_only=False)
            initial = counters(saved['optimizers']['controller_optimizer']) or [0]
            assert initial == plan['initial_Adam_counters'][case]
            for row in rows:
                assert row['status'] == 'passed' and all(row['gates'].values())
                assert row['sampling_metadata_verified'] and row['frozen_RNG_and_buffers_restored']
                assert row['matmul_precision'] == 'highest' and row['OpenMM'] == 0
                expected = [v+row['update'] for v in initial]
                assert row['Adam_counters'] == row['reference_Adam_counters'] == row['expected_Adam_counters'] == expected
            final = torch.load(output/(case+'_final_diagnostic_state.pt'), map_location='cpu', weights_only=False)
            assert counters(final['actual_optimizer']) == counters(final['reference_optimizer']) == [v+20 for v in initial]
            assert final['initial_Adam_counters'] == initial
            assert all(torch.isfinite(v).all() for v in [*final['actual_parameters'].values(), *final['reference_parameters'].values()])
        status = 'verified_saved_batch_multistep_pass'
    elif not active and accounting:
        status = 'terminal_failure_requires_classification'
    log = Path(submission['log'])
    return dict(status=status, accounting=accounting, active=active, source_sha256=source,
                plan_sha256=sha256(plan_path), inputs_verified=len(plan['frozen_inputs']),
                files={str(p.relative_to(campaign)): sha256(p) for p in campaign.rglob('*') if p.is_file()},
                gate=gate, receipt_verified=bool(receipt),
                log_tail=log.read_text()[-12000:] if log.exists() else None,
                production_changed=False, full_training_released=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--python')  # Uniform campaign monitor interface; no workers launched.
    args = parser.parse_args()
    print(json.dumps(check(args.campaign), allow_nan=False))


if __name__ == '__main__':
    main()
