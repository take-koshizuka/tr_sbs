#!/usr/bin/env python3
"""Recover failed postflight metadata without training, after exact audit."""
import json,hashlib,math
from pathlib import Path
EXPECTED=[(2,'h',1000,600000),(3,'u',0,600000),(3,'u',2000,800000),(3,'h',0,800000),(3,'h',1000,900000)]
def validate(report):
 d=report['diagnostics']
 if report['global_step']!=900000 or report['rejected_training_steps'] or d['backend_adam_update_counts']!={'u':600000,'h':300000}:raise ValueError('Incomplete/nonfinite training')
 if [(x['outer_loop'],x['phase'],x['phase_epoch'],x['global_step']) for x in d['inner_boundaries']]!=EXPECTED:raise ValueError('Wrong ordered boundary history')
 if report['cost']['total_openmm_configurations']!=260224 or not d['resume_marker']['restored'] or not d['resume_marker']['initial_online_suppressed']:raise ValueError('Cost/resume mismatch')
 if d['inner_boundaries'][0]['openmm_configurations']!=0:raise ValueError('Cached boundary consumed energy work')
 return True

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 B=Path('/scratch/htc/tkoshizuka/aldp_chart_20261007_5a093a95');old=B/'physical_jax_outer_extension_full_v1';run=old/'results/outer_extension/normal';root=B/'physical_jax_outer_postflight_recovery_v1'
 if (root/'results').exists():raise RuntimeError('Existing recovery; retain it')
 plan=json.loads((old/'results/plan.json').read_text());manifest=json.loads((old/'code/source_manifest.json').read_text())
 for f,h in manifest['files'].items():assert sha(old/'code'/f)==h
 for f,h in plan['frozen_inputs'].items():assert sha(f)==h
 rec=json.loads((run/'receipt.json').read_text())
 for f,h in rec['files'].items():assert sha(run/f)==h
 log=old/'results/logs/aldp-outer-full-5231840.out';text=log.read_text();assert text.rstrip().endswith('RuntimeError: Missing third-outer boundaries')
 report=json.loads((run/'result.json').read_text());validate(report)
 assert report['provenance']['plan_sha256']==sha(old/'results/plan.json') and report['source_sha256']==sha(old/'code/source_manifest.json') and report['config_sha256']==sha(run/'config.yaml')
 out=root/'results';out.mkdir()
 report.update(new_optimizer_steps=300000,cached_global_step=600000,geometry_admission='not_established',minimum_inner_duration='not_established',postflight_recovery={'original_job_id':'5231840','scheduler_status':'FAILED1:0','failure_scope':'postflight boundary-count assertion only','training_repeated':False,'original_result_sha256':sha(run/'result.json'),'analysis_source_sha256':sha(Path(__file__))})
 parent=json.loads(Path(plan['registration']['parent_result_path']).read_text());d=report['diagnostics'];limits=plan['curve_contract']['non_deterioration_limits'];c,n=d['trained_policy'],parent['diagnostics']['trained_policy'];a,b=c['metrics']['a'],n['metrics']['a'];checks={}
 def upper(k,v,bound):checks[k]={'value':v,'bound':bound,'passed':math.isfinite(v) and math.isfinite(bound) and v<=bound}
 upper('heldout_loss',d['next_outer_validation']['heldout_loss'],parent['diagnostics']['next_outer_validation']['heldout_loss']*limits['heldout_loss_ratio'])
 for k,lim in [('torsion_js_mean','torsion_js_absolute_increase'),('coarse_basin_tvd','coarse_basin_tvd_absolute_increase'),('missing_reference_mass','missing_mass_absolute_increase')]:upper(k,a[k],b[k]+limits[lim])
 upper('physical_energy_w2',a['physical_energy_w2'],b['physical_energy_w2']*limits['physical_energy_w2_ratio']);upper('close_contact_fraction',c['geometry']['fraction_below_reference_q01'],n['geometry']['fraction_below_reference_q01']+limits['close_contact_fraction_absolute_increase'])
 for i in (0,1):upper('near_cut_'+str(i),c['near_cut_fraction_phi_psi'][i],n['near_cut_fraction_phi_psi'][i]+limits['near_cut_fraction_absolute_increase'])
 upper('raw_log_path_ratio_variance',c['raw_log_path_ratio_variance'],n['raw_log_path_ratio_variance']*limits['raw_log_path_ratio_variance_ratio']);v=c['raw_path_ess_fraction'];bound=n['raw_path_ess_fraction']*limits['raw_path_ess_fraction_ratio_min'];checks['raw_path_ess_fraction']={'value':v,'bound':bound,'passed':math.isfinite(v) and v>=bound}
 improved=b['torsion_js_mean']-a['torsion_js_mean']>=.01 or b['coarse_basin_tvd']-a['coarse_basin_tvd']>=.02
 comparison={'checks':checks,'conditional_candidate':all(x['passed'] for x in checks.values()) and improved,'meaningful_A_improvement':improved,'decision_bank':'a','test':'closed','disclosure_b':c['metrics']['b'],'fixed_BC':d['next_outer_validation'],'own_BC':d['inner_boundaries'][-1]['validation'],'cost':report['cost'],'tail':c['physical_energy']['quantiles'],'geometry_admission':'not_established'}
 for name,obj in [('result.json',report),('comparison.json',comparison)]: (out/name).write_text(json.dumps(obj,indent=2)+'\n')
 audit={'schema':'aldp_analysis_recovery_receipt_v1','status':'postflight_metadata_recovered','original_scheduler_status':'FAILED1:0','training_complete_independently_verified':True,'old_source_sha256':sha(old/'code/source_manifest.json'),'old_plan_sha256':sha(old/'results/plan.json'),'old_receipt_sha256':sha(run/'receipt.json'),'old_result_sha256':sha(run/'result.json'),'old_log_sha256':sha(log),'old_receipt_files':rec['files'],'recovered_files':{f:sha(out/f) for f in ['result.json','comparison.json']},'analysis_source_sha256':sha(Path(__file__)),'new_training_updates':0,'new_OpenMM':0}
 (out/'receipt.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'status':audit['status'],'conditional_candidate':comparison['conditional_candidate'],'checks':checks,'cost':comparison['cost']}))
if __name__=='__main__':main()
