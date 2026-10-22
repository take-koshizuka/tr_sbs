#!/usr/bin/env python3
"""Registered A-only reference-rate comparison; BC/RN/ESS are disclosure."""
import argparse,json,math,hashlib
from pathlib import Path

def compare(candidate, control, registration):
    if registration['decision_bank']!='A' or registration['test']!='closed':
        raise ValueError('A-only selection and closed test required')
    for report in (candidate,control):
        if report['global_step']!=300000 or report['rejected_training_steps']:
            raise ValueError('Audited normal-duration completion required')
    current=candidate['diagnostics']['trained_policy'];normal=control['diagnostics']['trained_policy']
    a,b=current['metrics']['a'],normal['metrics']['a'];limits=registration['physical_non_deterioration_limits'];checks={}
    def upper(name,value,bound):
        checks[name]={'value':value,'bound':bound,'passed':math.isfinite(value) and math.isfinite(bound) and value<=bound}
    for field,key in [('torsion_js_mean','JS_add'),('coarse_basin_tvd','basin_TVD_add'),('missing_reference_mass','missing_mass_add')]:
        upper(field,a[field],b[field]+limits[key])
    upper('physical_energy_w2',a['physical_energy_w2'],b['physical_energy_w2']*limits['energy_W2_ratio'])
    upper('close_contact_fraction',current['geometry']['fraction_below_reference_q01'],normal['geometry']['fraction_below_reference_q01']+limits['contact_fraction_add'])
    for i,key in enumerate(['near_cut_phi_add','near_cut_psi_add']):
        upper('near_cut_'+str(i),candidate['near_cut_fraction_phi_psi'][i],control['near_cut_fraction_phi_psi'][i]+limits[key])
    improved=(math.isfinite(a['torsion_js_mean']) and b['torsion_js_mean']-a['torsion_js_mean']>=.01) or (math.isfinite(a['coarse_basin_tvd']) and b['coarse_basin_tvd']-a['coarse_basin_tvd']>=.02)
    return {'schema':'aldp_ou_rate_comparison_v1','checks':checks,'physical_non_deterioration_passed':all(x['passed'] for x in checks.values()),'meaningful_A_improvement':improved,'candidate_for_next_outer':all(x['passed'] for x in checks.values()) and improved,'geometry_admission':'not_established','next_outer_confirmation':'not_evaluated','decision_bank':'A','test':'closed','disclosure_B':{'candidate':current['metrics']['b'],'control':normal['metrics']['b']},'reference_dependent_disclosure':{'candidate':candidate['diagnostics']['inner_boundaries'][-1]['validation'],'control':control['diagnostics']['inner_boundaries'][-1]['validation'],'candidate_RN_variance':candidate['diagnostics']['inner_boundaries'][-1]['policy']['raw_log_path_ratio_variance'],'control_RN_variance':control['diagnostics']['inner_boundaries'][-1]['policy']['raw_log_path_ratio_variance'],'candidate_ESS_fraction':candidate['diagnostics']['inner_boundaries'][-1]['policy']['raw_path_ess_fraction'],'control_ESS_fraction':control['diagnostics']['inner_boundaries'][-1]['policy']['raw_path_ess_fraction'],'cross_reference_loss_gate':False},'cost':{'candidate':candidate['cost'],'cached_control':control['cost'],'control_retrained':False},'tail_quantiles':{'candidate':current['physical_energy']['quantiles'],'control':normal['physical_energy']['quantiles']}}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    for name in ('candidate','control','registration','audit','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();reg=json.loads(args.registration.read_text());audit=json.loads(args.audit.read_text())
    if audit.get('status')!='normal_curve_verified' or audit.get('result_sha256')!=sha(args.candidate):raise ValueError('Verified candidate required')
    if sha(args.control)!=reg['control_result_sha256']:raise ValueError('Cached control changed')
    out=compare(json.loads(args.candidate.read_text()),json.loads(args.control.read_text()),reg)
    out['input_sha256']={k:sha(getattr(args,k)) for k in ('candidate','control','registration','audit')}
    args.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
