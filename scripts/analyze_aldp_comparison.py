#!/usr/bin/env python3
"""Summarize audited ALDP cells; quality selection never implies admission."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

QUALITY_KEYS=('phi_psi_tvd','torsion_js_mean','physical_energy_w2',
              'coarse_basin_tvd','missing_reference_mass')
RANK_KEYS=(*QUALITY_KEYS,'close_distance_fraction','negative_raw_path_ess')


def midranks(values):
    """Zero-based average ranks, preserving ties without candidate-order bias."""
    ordered=sorted(set(values))
    return {value:sum(i for i,v in enumerate(sorted(values)) if v==value)/values.count(value)
            for value in ordered}


def phase_diagnosis(report, receipt):
    diagnostic=report.get('diagnostics',{})
    phases=diagnostic.get('phase_end',[])
    hashes=receipt.get('files',{})
    paired=[]
    for i in range(0,len(phases)-1,2):
        u,h=phases[i:i+2]
        if (u.get('phase'),h.get('phase'),u.get('outer_loop'))!=('u','h',h.get('outer_loop')):
            raise ValueError('Malformed phase ordering')
        u_file=u['label']+'_cartesian.npy'
        h_file=h['label']+'_cartesian.npy'
        same=hashes.get(u_file)==hashes.get(h_file) if u_file in hashes and h_file in hashes else None
        paired.append(dict(outer_loop=u['outer_loop'],u_step=u['global_step'],h_step=h['global_step'],
            identical_forward_endpoint_arrays=same,
            u_raw_path_ess=u['raw_path_ess_fraction'],h_raw_path_ess=h['raw_path_ess_fraction'],
            u_path_ratio_variance=u['raw_log_path_ratio_variance'],
            h_path_ratio_variance=h['raw_log_path_ratio_variance'],
            u_metrics_a=u['metrics']['a'],h_metrics_a=h['metrics']['a'],
            u_gamma=u['gamma'],h_gamma=h['gamma']))
    dual=[dict(outer_loop=row['outer_loop'],global_step=row['global_step'],
        gamma=row['gamma'],bootstrap_q05_q50_q95=row['bootstrap_gamma_q05_q50_q95'],
        independent_solve_gamma=row['independent_solve_gamma'],
        groups=row['groups'],paths_per_group=row['paths_per_group'],
        lambda_near_upper_bound=row['lambda_near_upper_bound']) for row in diagnostic.get('dual',[])]
    return dict(phase_sample_count=diagnostic.get('phase_sample_count'),phase_pairs=paired,dual=dual,
        interpretation='small paired phase diagnostics, not final-quality admission; BC changes reverse path weights while forward u is fixed',
        initial_and_final_sample_size_note='initial/final1024 and phase128 have different draws; no paired trajectory claim across those sample sizes')


def summarize(cells, thresholds):
    """Call only after the caller audits accounting, logs, every SHA and inputs."""
    if set(thresholds)!=set(QUALITY_KEYS) or not all(math.isfinite(v) and v>0 for v in thresholds.values()):
        raise ValueError('Use the frozen complete quality thresholds')
    rows=[]
    for cell in cells:
        proof=cell.get('verification',{})
        required=('log_verified','all_receipt_sha_verified','source_plan_inputs_verified')
        verified=proof.get('accounting')=='COMPLETED/0:0' and all(proof.get(k) is True for k in required)
        row=dict(candidate=cell['candidate'],job_id=cell['job_id'],execution_verified=verified,
                 scientific_status='not_evaluated',selection_eligible=False)
        report=cell.get('report')
        if not verified or not report or report.get('status')!='artifact_complete':
            row['retained_failure_or_pending_status']=cell.get('status','verification_incomplete')
            rows.append(row)
            continue
        a=report['metrics']['a']
        close=report['diagnostics']['trained_policy']['geometry']['fraction_below_reference_q01']
        ess=report['raw_path_ess']
        features=dict(a,close_distance_fraction=close,negative_raw_path_ess=-ess)
        if not all(math.isfinite(features[k]) for k in RANK_KEYS):
            raise ValueError('Nonfinite audited quality feature')
        passes={key:a[key]<=thresholds[key] for key in QUALITY_KEYS}
        roundtrip=report.get('native_cartesian_roundtrip_diagnostic')
        legacy_error=report.get('native_cartesian_roundtrip_error')
        roundtrip_pass=(roundtrip.get('status')=='passed' if roundtrip is not None
                        else isinstance(legacy_error,(int,float)) and math.isfinite(legacy_error)
                             and legacy_error<=2e-5)
        pilot_pass=all(passes.values()) and roundtrip_pass
        row.update(selection_eligible=True,metrics_a=a,metrics_b=report['metrics']['b'],
            quality_gate_passes=passes,pilot_quality_gate_pass=pilot_pass,
            coordinate_roundtrip_pass=roundtrip_pass,
            coordinate_roundtrip_diagnostic=roundtrip,
            scientific_status='pilot_quality_gate_pass' if pilot_pass else 'quality_gate_failed',
            close_distance_fraction=close,raw_path_ess=ess,rank_features=features,
            cost=report['cost'],capacity=report['capacity'],near_cut=report['near_cut_fraction_phi_psi'],
            sampling_metadata=report['sampling_metadata'],global_step=report['global_step'],
            phase_diagnosis=phase_diagnosis(report,cell.get('receipt',{})))
        rows.append(row)
    eligible=[row for row in rows if row['selection_eligible']]
    for row in eligible[1:]:
        if row['capacity']!=eligible[0]['capacity']:
            raise ValueError('Comparison changes model capacity')
        for key in ('source','target','controller','corrector','terminal_cost'):
            if row['sampling_metadata'][key]!=eligible[0]['sampling_metadata'][key]:
                raise ValueError('Comparison changes frozen sampling contract: '+key)
    for row in eligible:
        row['quality_rank_mean']=sum(midranks([r['rank_features'][k] for r in eligible])[
            row['rank_features'][k]] for k in RANK_KEYS)/len(RANK_KEYS)
    passing=[row for row in eligible if row['pilot_quality_gate_pass']]
    if passing:
        selected=min(passing,key=lambda r:(r['cost']['training_and_online_openmm_configurations'],
                                           r['quality_rank_mean'],r['candidate']))
        rule='all A pilot gates pass, minimum training/online OpenMM, then balanced A rank'
    elif eligible:
        selected=min(eligible,key=lambda r:(-sum(r['quality_gate_passes'].values()),
            r['quality_rank_mean'],r['cost']['training_and_online_openmm_configurations'],r['candidate']))
        rule='diagnostic only: most passed A gates, equal mean ranks over five A metrics/close-distance/ESS, then cost'
    else:
        selected=None
        rule='no verified eligible cell'
    ready=bool(cells) and all(row['execution_verified'] or cell.get('terminal_failure_verified',False)
              for row,cell in zip(rows,cells))
    return dict(schema='aldp_audited_comparison_v1',rows=rows,thresholds=thresholds,
        selected_diagnostic_candidate=selected['candidate'] if selected and ready else None,
        selection_ready=ready,selection_rule=rule,rank_keys=RANK_KEYS,
        selection_bank='a',disclosure_bank='b',final_test_closed=True,
        geometry_status='not_admitted_requires_original_confirmation_gates',
        reference_status='finite correlated MD banks, not certified converged reference',
        comparison_limit='one-seed screen with disclosed unequal costs/hardware; no convergence, causal proof or speed ranking')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True,help='JSON cells plus upstream verification evidence')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.input.read_text())
    document=summarize(data['cells'],data['thresholds'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
