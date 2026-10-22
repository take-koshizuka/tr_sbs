#!/usr/bin/env python3
"""BC-weight only: cached u2000 then first h and matched next outer."""
import argparse,json
from pathlib import Path

def validate(plan):
 r=plan['registration']
 if (r['only_scientific_change']!='bc_loss_weight legacy -> reference_variance' or r['new_total_updates']!=400000 or r['final_total_updates']!=600000 or r['expected_new_OpenMM']!=263040 or r['control_retrained']):raise ValueError('Unregistered BC continuation')
 if plan['case']['saved_phase']!='u' or plan['case']['saved_phase_epoch']!=2000 or plan['case']['cached_global_step']!=200000:raise ValueError('Wrong cached prefix')

def analyze(candidate,control,limits):
 from analyze_aldp_BC_weight import analyze as comparison
 return comparison(candidate,control,limits)

def run(plan,campaign,source):
 import torch,jax
 jax.config.update('jax_default_matmul_precision','highest');torch.set_float32_matmul_precision('highest')
 from omegaconf import OmegaConf,open_dict
 from run_aldp_chart_campaign import sha256,write_json,receipt,train_case
 from run_aldp_inner_continuation import ContinuationObserver,setup_backend,expected_cost
 from tr_asbs.experiment.inner_observation import continuation_training_contract
 from tr_asbs.process.chart_source import load_calibration_bundle
 validate(plan)
 if not torch.cuda.is_available() or not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU mandatory')
 for p,h in plan['frozen_inputs'].items():
  if sha256(p)!=h:raise RuntimeError('Input SHA '+p)
 proof=json.loads(Path(plan['backend_verification_path']).read_text())
 if proof['status']!='backend_verified' or proof['job_id']!='5241603':raise RuntimeError('Original precision GPU gate required')
 cfg=OmegaConf.load(plan['parent_config_path']);out=campaign/'continuation/reference_variance';case=plan['case']
 with open_dict(cfg):
  cfg.name,cfg.root=out.name,str(out);cfg.fabric.loggers.save_dir=str(out);cfg.checkpoint_directory=str(out/'checkpoints');cfg.sample_directory=str(out/'samples');cfg.resume=False;cfg.max_outer_loops=2;cfg.bc_loss_weight='reference_variance';cfg.exact_inner_resume_input=dict(path=case['checkpoint_path'],provenance=case['expected_provenance'],legacy_resume_used=False)
 if continuation_training_contract(cfg)!=plan['training_contract']:raise RuntimeError('Nuisance contract changed')
 cost=expected_cost(cfg,case)
 if cost['total_openmm_configurations']!=263040 or cost['new_optimizer_steps']!=400000:raise RuntimeError('Cost registry mismatch')
 if out.exists():raise RuntimeError('Preserve output')
 out.mkdir(parents=True);OmegaConf.save(cfg,out/'config.yaml',resolve=True)
 if sha256(out/'config.yaml')!=plan['expected_config_sha256']:raise RuntimeError('Prepared config changed')
 class Observer(ContinuationObserver):
  def _matched_h_loss(self,module):
   old=module.bc_loss_weight
   try:
    module.bc_loss_weight='legacy';r=super()._matched_h_loss(module);r['weight']='legacy';return r
   finally:module.bc_loss_weight=old
 observer=Observer(load_calibration_bundle(cfg.potential.calibration_bundle_path),out,cfg,plan,case,source)
 train_case(dict(plan,max_openmm_per_geometry_run=320000),campaign,'continuation',cfg.potential.chart_key,0,source,config_and_output=(cfg,out),observer=observer,provenance=observer.provenance,module_setup=lambda m:setup_backend(m,observer,plan))
 report=json.loads((out/'result.json').read_text());d=report['diagnostics']
 if report['global_step']!=600000 or report['rejected_training_steps'] or d['backend_adam_update_counts']!={'u':400000,'h':200000}:raise RuntimeError('Update counts')
 for key,value in cost.items():
  if key.endswith('configurations') and report['cost'][key]!=value:raise RuntimeError('Cost mismatch '+key)
 if not d['resume_marker']['initial_online_suppressed']:raise RuntimeError('Extra online draw')
 expected=[(1,'u',2000,200000),(1,'h',0,200000),(1,'h',1000,300000),(2,'u',0,300000),(2,'u',2000,500000),(2,'h',0,500000),(2,'h',1000,600000)]
 if [(x['outer_loop'],x['phase'],x['phase_epoch'],x['global_step']) for x in d['inner_boundaries']]!=expected:raise RuntimeError('Boundary contract')
 report.update(new_optimizer_steps=400000,cached_global_step=200000,geometry_admission='not_established',production_matmul_precision='highest');write_json(out/'result.json',report)
 control=json.loads(Path(plan['control_result_path']).read_text());write_json(out/'comparison.json',analyze(report,control,plan['curve_contract']['non_deterioration_limits']));receipt(out,dict(stage='continuation',candidate='reference_variance',seed=0),dict(source_sha256=source,plan_sha256=sha256(campaign/'plan.json')));print('postflight_verified=true',flush=True)

def main():
 from run_aldp_chart_campaign import verify_source
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();plan=json.loads((a.campaign/'plan.json').read_text());source=verify_source()
 if source!=plan['source_manifest_sha256']:raise RuntimeError('Source mismatch')
 run(plan,a.campaign,source)
if __name__=='__main__':main()
