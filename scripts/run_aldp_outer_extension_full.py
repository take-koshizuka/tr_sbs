#!/usr/bin/env python3
"""Exactly one third outer via shared fit and exact parent epoch boundary."""
import argparse,json
from pathlib import Path
from run_aldp_outer_extension import validate_registration,marginal_cost

def configuration(plan,campaign):
 from omegaconf import OmegaConf,open_dict
 cfg=OmegaConf.load(plan['parent_config_path']);out=campaign/'outer_extension/normal'
 with open_dict(cfg):
  cfg.name,cfg.root=out.name,str(out);cfg.fabric.loggers.save_dir=str(out);cfg.checkpoint_directory=str(out/'checkpoints');cfg.sample_directory=str(out/'samples');cfg.resume=False;cfg.max_outer_loops=3
  cfg.exact_inner_resume_input=dict(path=plan['registration']['parent_snapshot_path'],provenance=plan['registration']['expected_parent_provenance'],legacy_resume_used=False)
 if marginal_cost(cfg)!=260224:raise ValueError('Marginal cost differs from registry')
 return cfg,out

def run(plan,campaign,source):
 from copy import deepcopy
 from omegaconf import OmegaConf
 from run_aldp_chart_campaign import sha256,write_json,receipt,train_case
 from run_aldp_inner_continuation import ContinuationObserver,setup_backend,policy_checks
 from tr_asbs.process.chart_source import load_calibration_bundle
 from tr_asbs.experiment.inner_observation import continuation_training_contract
 validate_registration(plan['registration'])
 proof=json.loads(Path(plan['transition_verification_path']).read_text())
 if proof['status']!='transition_verified' or proof['report']['new_updates']!=20:raise RuntimeError('Exact transition proof required')
 for f,h in plan['frozen_inputs'].items():
  if sha256(f)!=h:raise RuntimeError('Input SHA mismatch '+f)
 cfg,out=configuration(plan,campaign)
 if out.exists():raise RuntimeError('Preserve old outputs')
 out.mkdir(parents=True);OmegaConf.save(cfg,out/'config.yaml',resolve=True)
 parent=json.loads(Path(plan['registration']['parent_result_path']).read_text());r=plan['registration']
 case=dict(checkpoint_path=r['parent_snapshot_path'],expected_provenance=r['expected_parent_provenance'],saved_phase='h',saved_phase_epoch=1000,cached_global_step=600000,first_h_validation_path=plan['parent_h_validation_path'],reuse_first_outer_policy=True,parent_boundary={'policy':parent['diagnostics']['trained_policy']},save_first_h_prefix=None)
 class Observer(ContinuationObserver):
  def __call__(self,module,event,payload=None):
   value=super().__call__(module,event,payload)
   if event=='initial':self.report['training_contract']['outer_loops']=3
   if event=='restored':
    if self.report['restored_counts']!=dict(epoch=6000,global_step=600000,phase='h',phase_epoch=1000,outer_step=1):raise RuntimeError('Wrong parent boundary')
    for old,new in [('first_outer_complete','cached_outer2_policy'),('first_outer_own_h_validation','cached_outer2_own_BC'),('first_outer_validation','cached_outer2_fixed_BC')]:self.report[new]=self.report.pop(old)
   return value
 observer=Observer(load_calibration_bundle(cfg.potential.calibration_bundle_path),out,cfg,plan,case,source)
 if continuation_training_contract(cfg)!=plan['training_contract']:raise RuntimeError('Fixed nuisance settings changed')
 train_case(dict(plan,max_openmm_per_geometry_run=320000),campaign,'outer_extension',cfg.potential.chart_key,0,source,config_and_output=(cfg,out),observer=observer,provenance=observer.provenance,module_setup=lambda module:setup_backend(module,observer,plan))
 report=json.loads((out/'result.json').read_text());diag=report['diagnostics']
 if report['global_step']!=900000 or report['rejected_training_steps'] or diag['backend_adam_update_counts']!={'u':600000,'h':300000}:raise RuntimeError('Wrong final update counts')
 if report['cost']['total_openmm_configurations']!=260224 or not diag['resume_marker']['initial_online_suppressed']:raise RuntimeError('Marginal cost/resume mismatch')
 expected=[(3,'u',0,600000),(3,'u',2000,800000),(3,'h',0,800000),(3,'h',1000,900000)]
 if [(x['outer_loop'],x['phase'],x['phase_epoch'],x['global_step']) for x in diag['inner_boundaries']]!=expected:raise RuntimeError('Missing third-outer boundaries')
 candidate={'matched_h_validation':diag['next_outer_validation'],'policy':diag['trained_policy']};control={'matched_h_validation':parent['diagnostics']['next_outer_validation'],'policy':parent['diagnostics']['trained_policy']}
 checks=policy_checks(candidate,control,plan['curve_contract']['non_deterioration_limits']);a,b=candidate['policy']['metrics']['a'],control['policy']['metrics']['a'];improved=b['torsion_js_mean']-a['torsion_js_mean']>=.01 or b['coarse_basin_tvd']-a['coarse_basin_tvd']>=.02
 report.update(new_optimizer_steps=300000,cached_global_step=600000,geometry_admission='not_established',minimum_inner_duration='not_established')
 write_json(out/'result.json',report)
 comparison=dict(schema='aldp_outer3_comparison_v1',checks=checks,non_deterioration_passed=all(x['passed'] for x in checks.values()),meaningful_A_improvement=improved,conditional_candidate=all(x['passed'] for x in checks.values()) and improved,decision_bank='a',test='closed',disclosure_b=candidate['policy']['metrics']['b'],fixed_BC=diag['next_outer_validation'],own_BC=diag['inner_boundaries'][-1]['validation'],cost=report['cost'],cached_control_cost=parent['cost'],cached_control_retrained=False,geometry_admission='not_established')
 write_json(out/'comparison.json',comparison);receipt(out,dict(stage='outer_extension',candidate='normal',seed=0),dict(source_sha256=source,plan_sha256=sha256(campaign/'plan.json')))
 print('postflight_verified=true',flush=True)

def main():
 from run_aldp_chart_campaign import verify_source
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();plan=json.loads((a.campaign/'plan.json').read_text());source=verify_source()
 if source!=plan['source_manifest_sha256']:raise RuntimeError('Source mismatch')
 run(plan,a.campaign,source)
if __name__=='__main__':main()
