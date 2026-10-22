#!/usr/bin/env python3
"""Single h gradient: common-gradient Adam vs native gradients, diagnosis only."""
import argparse,json,copy
from pathlib import Path

def main():
 import torch,jax,numpy as np
 from run_aldp_chart_campaign import verify_source,sha256,write_json,receipt
 from run_aldp_BC_weight_preflight import configuration
 from run_aldp_jax_inner_preflight import torch_prepared_bc
 from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
 from tr_asbs.experiment.inner_observation import restore_inner_snapshot
 from tr_asbs.experiment.jax_inner import JaxInnerBackend,prepare_bc_batch,export_fourier,bc_loss,from_torch,adam_update,_adam_from_torch
 from omegaconf import OmegaConf
 a=argparse.ArgumentParser();a.add_argument('--campaign',type=Path,required=True);args=a.parse_args();c=args.campaign;p=json.loads((c/'plan.json').read_text());source=verify_source();assert source==p['source_manifest_sha256']
 for n,h in p['frozen_inputs'].items():assert sha256(n)==h,n
 assert torch.cuda.is_available() and any(d.platform=='gpu' for d in jax.devices())
 cfg,out=configuration(p,c);assert not out.exists();out.mkdir(parents=True);OmegaConf.save(cfg,out/'config.yaml',resolve=True)
 module=GammaASBSModule(cfg);JaxInnerBackend(module).attach();payload=torch.load(p['derived_snapshot_path'],map_location='cpu',weights_only=False);restore_inner_snapshot(module,p['derived_snapshot_path'],expected_provenance=payload['provenance'])
 batches=torch.load(p['paired_batches_path'],map_location=module.fabric.device,weights_only=False);torch.manual_seed(23547);torch.cuda.manual_seed_all(23547);prepared=prepare_bc_batch(module,batches['h'])
 model,params,coeff,spec=export_fourier(module.corrector);optimizer,adam_spec,state=_adam_from_torch(module.corrector_optimizer,model,params,cfg.gradient_clip_val)
 initial_weights=copy.deepcopy(model.state_dict());initial_optimizer=copy.deepcopy(optimizer.state_dict());loss=torch_prepared_bc(module,prepared);loss.backward();tg={n:from_torch(v.grad.detach().clone()) for n,v in model.named_parameters()};jg=jax.jit(jax.grad(lambda x:bc_loss(x,coeff,spec,prepared)[0]))(params)
 norm=sum(float((v.grad.detach().double()**2).sum()) for v in model.parameters())**.5;assert norm<cfg.gradient_clip_val
 reports=[]
 for name,grads in [('torch_gradient',tg),('jax_gradient',jg)]:
  model.load_state_dict(initial_weights);optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
  for n,v in model.named_parameters():v.grad=torch.utils.dlpack.from_dlpack(grads[n]).to(v).clone()
  optimizer.step();torch_params={n:v.detach().cpu().numpy().copy() for n,v in model.named_parameters()}
  (updated,_),_,valid=jax.jit(lambda x,g:adam_update(x,state,g,loss.detach().item(),adam_spec))(params,grads);jax.block_until_ready(updated);assert bool(valid)
  errors={n:float(np.abs(np.asarray(updated[n])-v).max()) for n,v in torch_params.items()};reports.append(dict(common_gradient=name,parameter_max_absolute_error=max(errors.values()),parameter_errors=errors))
 # Per-element evidence at worst native-gradient first-update discrepancy.
 (tu,_),_,_=adam_update(params,state,tg,loss.detach().item(),adam_spec);(ju,_),_,_=adam_update(params,state,jg,loss.detach().item(),adam_spec)
 name=max(tu,key=lambda n:float(np.abs(np.asarray(tu[n])-np.asarray(ju[n])).max()));diff=np.abs(np.asarray(tu[name])-np.asarray(ju[name]));idx=np.unravel_index(np.argmax(diff),diff.shape)
 worst=dict(parameter=name,index=[int(x) for x in idx],initial=float(np.asarray(params[name])[idx]),torch_gradient=float(np.asarray(tg[name])[idx]),jax_gradient=float(np.asarray(jg[name])[idx]),torch_gradient_update=float(np.asarray(tu[name])[idx]-np.asarray(params[name])[idx]),jax_gradient_update=float(np.asarray(ju[name])[idx]-np.asarray(params[name])[idx]),gradient_sign_differs=bool(np.sign(np.asarray(tg[name])[idx])!=np.sign(np.asarray(jg[name])[idx])),adam_epsilon=adam_spec.eps)
 assert module.potential.aldp_potential.openmm_configurations_evaluated==0
 report=dict(status='diagnostic_complete',rows=[],common_gradient_Adam=reports,worst_native_gradient_element=worst,scientific_status='not_evaluated',source_sha256=source,plan_sha256=sha256(c/'plan.json'),OpenMM=0,tolerance_relaxation=False,full_training_submitted=False)
 write_json(out/'update_growth.json',report);torch.save(dict(prepared={k:torch.utils.dlpack.from_dlpack(v).cpu() for k,v in prepared.items()},torch_gradients={k:np.asarray(v) for k,v in tg.items()},jax_gradients={k:np.asarray(v) for k,v in jg.items()}),out/'gradient_evidence.pt');receipt(out,dict(stage='backend_preflight',candidate='jax',seed=0),dict(source_sha256=source,plan_sha256=sha256(c/'plan.json')));print('diagnostic_complete=true',flush=True)
if __name__=='__main__':main()
