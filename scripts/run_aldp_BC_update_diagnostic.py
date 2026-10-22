#!/usr/bin/env python3
"""Frozen-batch short-update error growth; diagnosis only, no admission."""
import argparse,json
from pathlib import Path

def main():
 import torch,jax
 from omegaconf import OmegaConf
 from run_aldp_chart_campaign import sha256,verify_source,write_json,receipt
 from run_aldp_BC_weight_preflight import configuration
 from run_aldp_jax_inner_preflight import benchmark_updates
 from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
 from tr_asbs.experiment.jax_inner import JaxInnerBackend
 from tr_asbs.experiment.inner_observation import restore_inner_snapshot
 a=argparse.ArgumentParser();a.add_argument('--campaign',type=Path,required=True);args=a.parse_args();c=args.campaign;p=json.loads((c/'plan.json').read_text());source=verify_source()
 assert source==p['source_manifest_sha256']
 for n,h in p['frozen_inputs'].items():assert sha256(n)==h,n
 assert torch.cuda.is_available() and any(d.platform=='gpu' for d in jax.devices())
 cfg,out=configuration(p,c);assert not out.exists();out.mkdir(parents=True);OmegaConf.save(cfg,out/'config.yaml',resolve=True)
 module=GammaASBSModule(cfg);JaxInnerBackend(module).attach();payload=torch.load(p['derived_snapshot_path'],map_location='cpu',weights_only=False);batches=torch.load(p['paired_batches_path'],map_location='cpu',weights_only=False)
 # Each measured prefix resets all trained state; use saved original batches.
 rows=[]
 for phase in ('u','h'):
  for steps in (1,2,5):
   restore_inner_snapshot(module,p['derived_snapshot_path'],expected_provenance=payload['provenance']);module.training_step=GammaASBSModule.training_step.__get__(module)
   batch=batches[phase]
   def device(v):
    if isinstance(v,torch.Tensor):return v.to(module.fabric.device)
    if isinstance(v,dict):return {k:device(x) for k,x in v.items()}
    if isinstance(v,list):return [device(x) for x in v]
    if isinstance(v,tuple):return tuple(device(x) for x in v)
    return v
   r=benchmark_updates(module,phase,device(batch),steps,0,p['tolerances']);rows.append(r)
   write_json(out/'update_growth.json',dict(status='diagnostic_complete' if len(rows)==6 else 'partial',rows=rows,scientific_status='not_evaluated',source_sha256=source,plan_sha256=sha256(c/'plan.json'),OpenMM=module.potential.aldp_potential.openmm_configurations_evaluated,tolerance_relaxation=False,full_training_submitted=False))
 assert module.potential.aldp_potential.openmm_configurations_evaluated==0
 receipt(out,dict(stage='backend_preflight',candidate='jax',seed=0),dict(source_sha256=source,plan_sha256=sha256(c/'plan.json')));print('diagnostic_complete=true',flush=True)
if __name__=='__main__':main()
