#!/usr/bin/env python3
"""Frozen h prepared values in FP64: diagnosis, no precision policy change."""
import argparse,json,copy
from pathlib import Path

def validate_plan(plan):
 if plan.get('precision_diagnostic_only') is not True or plan.get('expected_OpenMM')!=0:raise ValueError('Only zero-OpenMM precision diagnosis permitted')
 if plan['tolerances']['parameter_max_absolute']!=1e-5:raise ValueError('Original tolerance required')
 if not plan.get('gradient_evidence_path') or not plan.get('derived_snapshot_path'):raise ValueError('Frozen evidence/state required')

def main():
 import torch,jax,jax.numpy as jnp,numpy as np
 from run_aldp_chart_campaign import verify_source,sha256,write_json,receipt
 from run_aldp_BC_weight_preflight import configuration
 from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
 from tr_asbs.experiment.inner_observation import restore_inner_snapshot
 from tr_asbs.experiment.jax_inner import JaxInnerBackend,export_fourier,bc_loss
 from omegaconf import OmegaConf
 a=argparse.ArgumentParser();a.add_argument('--campaign',type=Path,required=True);args=a.parse_args();c=args.campaign;p=json.loads((c/'plan.json').read_text());validate_plan(p);source=verify_source();assert source==p['source_manifest_sha256']
 for n,h in p['frozen_inputs'].items():assert sha256(n)==h,n
 assert torch.cuda.is_available() and any(d.platform=='gpu' for d in jax.devices()) and jax.config.x64_enabled
 cfg,out=configuration(p,c);assert not out.exists();out.mkdir(parents=True);OmegaConf.save(cfg,out/'config.yaml',resolve=True)
 module=GammaASBSModule(cfg);JaxInnerBackend(module).attach();payload=torch.load(p['derived_snapshot_path'],map_location='cpu',weights_only=False);restore_inner_snapshot(module,p['derived_snapshot_path'],expected_provenance=payload['provenance'])
 evidence=torch.load(p['gradient_evidence_path'],map_location='cpu',weights_only=False)
 base,params,coeff,spec=export_fourier(module.corrector);model=copy.deepcopy(base).double();batch={k:v.to(module.fabric.device,dtype=torch.float64) for k,v in evidence['prepared'].items()};size=len(batch['time'])
 values=model(torch.cat((batch['time'],torch.ones_like(batch['time']))),torch.cat((batch['data_t'],batch['data_1'])))
 loss=(((values[:size]-batch['score_t'])**2).mean(-1)/batch['coef_t']).mean()+((values[size:]-batch['score_1'])**2).mean();loss.backward()
 params64={k:v.astype(jnp.float64) for k,v in params.items()};batch64={k:jnp.asarray(v.cpu().numpy(),dtype=jnp.float64) for k,v in batch.items()}
 jl,jg=jax.jit(jax.value_and_grad(lambda x:bc_loss(x,coeff.astype(jnp.float64),spec,batch64)[0]))(params64);jax.block_until_ready((jl,jg));tg={n:v.grad.detach().cpu().numpy() for n,v in model.named_parameters()};jg={n:np.asarray(v) for n,v in jg.items()}
 def metric(a,b):
  err=sum(float(np.square(a[n]-b[n]).sum()) for n in a);norm=sum(float(np.square(a[n]).sum()) for n in a);return dict(relative_l2=err**.5/max(norm**.5,1e-30),max_absolute=max(float(np.abs(a[n]-b[n]).max()) for n in a))
 idx=(1,76);element=dict(parameter='out_layer.weight',index=list(idx),torch_fp64_gradient=float(tg['out_layer.weight'][idx]),jax_fp64_gradient=float(jg['out_layer.weight'][idx]),torch_cached_fp32=float(evidence['torch_gradients']['out_layer.weight'][idx]),jax_cached_fp32=float(evidence['jax_gradients']['out_layer.weight'][idx]))
 # Original Adam0 coordinate impact: use the loaded optimizer's actual LR/eps.
 opt=module.corrector_optimizer
 while hasattr(opt,'optimizer'):opt=opt.optimizer
 group=opt.param_groups[0];element['adam_epsilon']=group['eps'];element['learning_rate']=group['lr'];element['first_update_by_gradient']={k:float(-group['lr']*element[k]/(abs(element[k])+group['eps'])) for k in ('torch_fp64_gradient','jax_fp64_gradient','torch_cached_fp32','jax_cached_fp32')}
 assert module.potential.aldp_potential.openmm_configurations_evaluated==0
 report=dict(status='diagnostic_complete',rows=[],scientific_status='not_evaluated',source_sha256=source,plan_sha256=sha256(c/'plan.json'),OpenMM=0,tolerance_relaxation=False,full_training_submitted=False,production_precision_changed=False,fp64_loss=dict(torch=float(loss.detach()),jax=float(jl)),fp64_gradient=metric(tg,jg),cached_fp32_vs_torch_fp64={k:metric(tg,{n:np.asarray(v,dtype=np.float64) for n,v in evidence[k].items()}) for k in ('torch_gradients','jax_gradients')},worst_registered_element=element)
 write_json(out/'update_growth.json',report);torch.save(dict(torch_fp64_gradients=tg,jax_fp64_gradients=jg),out/'fp64_gradient_evidence.pt');receipt(out,dict(stage='backend_preflight',candidate='jax',seed=0),dict(source_sha256=source,plan_sha256=sha256(c/'plan.json')));print('diagnostic_complete=true',flush=True)
if __name__=='__main__':main()
