#!/usr/bin/env python3
"""Registered BC weight-only exact-prefix and paired backend GPU gate."""
import argparse,json,copy
from pathlib import Path

def derive_metadata(payload):
 if payload['sampling_metadata'].get('bc_loss_weight')!='legacy':raise ValueError('Registered legacy parent only')
 c=payload['counters']
 if (c['global_step'],c['current_epoch'],c['outer_step'],c['phase'],c['phase_epoch'])!=(200000,2000,0,'u',2000):raise ValueError('Wrong firstouter u-end prefix')
 result=copy.deepcopy(payload);result['sampling_metadata']['bc_loss_weight']='reference_variance'
 return result

def configuration(plan,campaign):
 from omegaconf import OmegaConf,open_dict
 cfg=OmegaConf.load(plan['parent_config_path']);out=campaign/'backend_preflight/jax_seed0'
 with open_dict(cfg):
  cfg.name,cfg.root=out.name,str(out);cfg.fabric.loggers.save_dir=str(out);cfg.checkpoint_directory=str(out/'checkpoints');cfg.sample_directory=str(out/'samples');cfg.resume=False;cfg.max_outer_loops=2;cfg.bc_loss_weight='reference_variance'
 return cfg,out

def run(plan,campaign,source):
 import torch,numpy as np,random,jax
 jax.config.update("jax_default_matmul_precision", "highest")
 torch.set_float32_matmul_precision("highest")
 from omegaconf import OmegaConf
 from run_aldp_chart_campaign import sha256,write_json,receipt
 from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
 from tr_asbs.experiment.jax_inner import JaxInnerBackend,prepare_bc_batch
 from tr_asbs.experiment.inner_observation import restore_inner_snapshot,save_inner_snapshot
 from run_aldp_jax_inner_preflight import gradient_comparison,benchmark_updates,torch_prepared_bc
 from run_aldp_mechanism_campaign import diagnostic_rng
 for p,h in plan['frozen_inputs'].items():
  if sha256(p)!=h:raise RuntimeError('Frozen input mismatch '+p)
 if not torch.cuda.is_available() or not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU mandatory')
 cfg,out=configuration(plan,campaign)
 if out.exists():raise RuntimeError('Preserve prior output')
 out.mkdir(parents=True);OmegaConf.save(cfg,out/'config.yaml',resolve=True)
 original=torch.load(plan['parent_snapshot_path'],map_location='cpu',weights_only=False);derived=derive_metadata(original)
 def equal(a,b):
  if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
  if isinstance(a,np.ndarray):return np.array_equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 for key in original:
  if key!='sampling_metadata':assert equal(original[key],derived[key]),key
 changed={k for k in original['sampling_metadata'] if not equal(original['sampling_metadata'][k],derived['sampling_metadata'][k])};assert changed=={'bc_loss_weight'}
 path=out/'derived_u2000.pt';torch.save(derived,path)
 module=GammaASBSModule(cfg);backend=JaxInnerBackend(module).attach();restore_inner_snapshot(module,path,expected_provenance=original['provenance'])
 restored=out/'restored_u2000.pt';save_inner_snapshot(module,restored,original['provenance']);actual=torch.load(restored,map_location='cpu',weights_only=False)
 for k in derived:assert equal(derived[k],actual[k]),k
 # Finish the zero-update resumed u phase, then prepare the existing h buffer.
 module.inner_loop_u();assert module.global_step==200000 and module.phase_epoch==2000
 module.phase,module.phase_epoch='h',0;module.populate_buffer_corrector_matching()
 ubatch=next(iter(module.create_dataloader('adjoint_matching')));hbatch=next(iter(module.create_dataloader('corrector_matching')))
 # Benchmarks must call actual Torch steps; attached JAX is used only for restore above.
 module.training_step=GammaASBSModule.training_step.__get__(module)
 with diagnostic_rng(module.fabric.device):
  prepared=prepare_bc_batch(module,hbatch);prepared_loss=torch_prepared_bc(module,prepared)
 with diagnostic_rng(module.fabric.device):torch_loss=module.corrector_matching_step(hbatch)['loss']
 assert torch.allclose(prepared_loss,torch_loss,rtol=plan['tolerances']['loss_relative'],atol=1e-5)
 with diagnostic_rng(module.fabric.device):
  times=torch.tensor([module.bridge_time_epsilon,.5,1-module.bridge_time_epsilon],device=module.fabric.device)
  _,_,variance=module.base_sde.transition_coefficients(0,times);coeff=1+module.base_sde.diffusion(times).square()*(1-times)/variance.to(times);assert torch.isfinite(coeff).all() and (coeff>0).all()
 gradients=[];updates=[]
 for phase,batch in [('u',ubatch),('h',hbatch)]:
  gradients.append(gradient_comparison(module,phase,batch,plan['tolerances']))
  updates.append(benchmark_updates(module,phase,batch,20,3,plan['tolerances']))
 write_json(out/'paired_diagnostic.json',dict(gradients=gradients,updates=updates,tolerances=plan['tolerances'],OpenMM=module.potential.aldp_potential.openmm_configurations_evaluated,scientific_status='not_evaluated'))
 torch.save(dict(u=ubatch,h=hbatch),out/'paired_batches.pt')
 assert all(v['status']=='passed' for v in gradients+updates)
 for v,start in zip(updates,[200000,0]):assert v['adam_restart_counters_match'] and v['adam_restart_counters']==dict(initial=start,torch_start=start,jax_start=start,torch_end=start+20,jax_end=start+20)
 cost=module.potential.aldp_potential.openmm_configurations_evaluated
 if cost>plan['preflight_OpenMM_cap']:raise RuntimeError('Preflight cost cap')
 report=dict(matmul_precision=dict(torch=torch.get_float32_matmul_precision(),jax=jax.config.jax_default_matmul_precision),status='passed',scientific_status='not_evaluated',no_quality_gate=True,source_sha256=source,plan_sha256=sha256(campaign/'plan.json'),config_sha256=sha256(out/'config.yaml'),gradients=gradients,updates=updates,metadata_only_change='bc_loss_weight',all_other_snapshot_fields_bitwise_equal=True,restore_all_fields_bitwise_equal=True,parent_snapshot_sha256=sha256(plan['parent_snapshot_path']),derived_snapshot_sha256=sha256(path),original_provenance=original['provenance'],Torch_original_BC_matches_prepared=True,endpoint_coefficients=coeff.detach().cpu().tolist(),OpenMM=cost,full_training_submitted=False)
 write_json(out/'result.json',report);receipt(out,dict(stage='backend_preflight',candidate='jax',seed=0),dict(source_sha256=source,plan_sha256=report['plan_sha256']));print('postflight_verified=true',flush=True)

def main():
 from run_aldp_chart_campaign import verify_source
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();plan=json.loads((a.campaign/'plan.json').read_text());source=verify_source()
 if source!=plan['source_manifest_sha256']:raise RuntimeError('Source mismatch')
 run(plan,a.campaign,source)
if __name__=='__main__':main()
